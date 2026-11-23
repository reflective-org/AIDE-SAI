"""Aerosol -> radiation -> heating driver coupling TOMAS bins to jax-rrtmgp.

Pipeline per coupling step:
    1. Per-bin aerosol optical properties from precomputed Mie tables
       (bhmie from tomas-jax + Palmer & Williams 1975 75 wt% H2SO4 refractive
       indices, band-averaged onto the 14 SW + 16 LW RRTMGP bands).
    2. jax-rrtmgp two-stream solve (clear-sky gases + our aerosol) on the
       full native CESM column (all levels with p >= 1.005 Pa, the RRTMGP
       lookup-table range), aerosol only on the coupled band levels.
    3. Heating rate [K/s] returned on the band levels in CESM level order.

Anomaly mode (the default use): heating is computed twice per step -- once
with the evolved TOMAS bins at T_eff = T_CESM + dT, once with reference
MAM4-binned aerosol at T_CESM -- and the difference drives d(dT)/dt. This
isolates the radiative effect of the aerosol *evolution* without the model
drifting toward radiative equilibrium (there is no other physics to balance
the full heating rate while the circulation is prescribed).

MVP approximations (each is a flagged refinement, not a hidden assumption):
    * clear sky (no clouds; hourly cloud water is not in the CESM archive)
    * refractive index clamped outside 0.36-25 um (Palmer & Williams range)
      and outside 25-95.6 wt% (their six tabulated solution strengths)
    * solar zenith from a standard declination formula (no equation of time)
    * solar geometry: RAD_SOLAR=window_gq (default; instant under the emulator)
      solves the SW at Gauss nodes of the sunlit window, LW once. RAD_SOLAR=instant holds the sun of the
      call hour for the whole interval; RAD_SOLAR=window uses the sunlit-mean
      cos(zenith) of the interval and scales the TOA irradiance by its sunlit
      fraction, so the interval's insolation is exact (see solar_window)

(Until 2026-08-03 the optics also used DRY bin diameters at a fixed 75 wt%
composition. They now use the RH/T-dependent wet droplet -- see WET_OPTICS.)
"""
import os
import sys

# --- the two dependency repos, as submodules under models/ -------------------
# Duplicated from coupling.py on purpose: radiation.py is imported directly by
# scripts/validation/test_radiation.py and validate_radiation.py, which never load
# coupling.py, so it cannot rely on coupling having fixed up sys.path first.
# Keep the two copies in step. See coupling._dep_path for the full rationale.
# Bootstrap src/ (the parent of this module's own directory) onto sys.path so
# _paths imports. radiation.py is loaded directly by the validation harnesses,
# which never load coupling.py, so it cannot rely on coupling having done this.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import _paths                      # noqa: E402 -- needs the insert above
_HERE = _paths.SRC
_MODELS = _paths.MODELS
for _env, _name in (('TOMAS_JAX_PATH', 'tomas-jax'), ('RRTMGP_PATH', 'jax-rrtmgp')):
    for _cand in (os.environ.get(_env),
                  os.path.join(_MODELS, _name)):
        if _cand and os.path.isdir(_cand):
            if _cand not in sys.path:
                sys.path.insert(0, _cand)
            break

import copy
import functools
import dataclasses
import hashlib
import numpy as np
import jax
import jax.numpy as jnp
import netCDF4

try:
    from rrtmgp.config import radiative_transfer
    from rrtmgp import rrtmgp as rrtmgp_lib
    from rrtmgp import rrtmgp_common
    from rrtmgp import constants as rrtmgp_constants
    from tomas_jax.physics.bhmie import bhmie_qsca_jax
except ImportError as _e:
    raise SystemExit(
        f"radiation.py: cannot import a dependency repo ({_e}).\n"
        "  jax-rrtmgp (radiation) and tomas-jax (Mie) are separate repos, NOT\n"
        "  vendored here. Run `git submodule update --init` to populate\n"
        "  models/jax-rrtmgp and models/tomas-jax, install them, or set\n"
        "  RRTMGP_PATH / TOMAS_JAX_PATH.\n"
        "  Run without radiation entirely with RAD=0.")

GRAV   = 9.80665
#stefan-boltzmann constant [W/m2/K4] for computing surface temperature from net upward flux
SIGMA_SB = 5.670374419e-8
RRTMGP_PKG = os.path.dirname(rrtmgp_lib.__file__)          # .../jax-rrtmgp/rrtmgp
RRTMGP_DATA = os.path.join(RRTMGP_PKG, 'optics/rrtmgp_data')
RRTMGP_TESTDATA = os.path.join(RRTMGP_PKG, 'optics/test_data')
# Repo-root-relative, not __file__-relative: the static input data lives in
# inputs/, which is a sibling of wherever this module sits, not a child of it.
RI_FILE = os.path.join(_paths.INPUTS, 'rad_data',
                       'palmer_williams_h2so4.dat')

CO2_PPM  = float(os.environ.get('CO2_PPM', '380.0'))   # ~2005 value for 1996-2014
N2O_PPB  = float(os.environ.get('N2O_PPB', '319.0'))
SFC_EMIS = 0.98
ALB_FALLBACK = 0.15            # night / FSDS~0 columns (SW is zeroed anyway)
TSI_DEFAULT = 1361.0           # W/m2; per-step irradiance override possible
# rows per rrtmgp call. Sweep (bench_rad2.py, 192 lat, 55-lvl column, 1 GPU,
# 1-pass heating): CHUNK 16->45.9s, 32->16.0s, 48->10.6s, 96->9.3s(best), 192->16.2s.
# 16 (the old rad_smoke value) is pathological: 12 tiny launches too small to
# saturate the H100, per-launch dispatch dominates. 96 (2 chunks) is ~5x faster,
# BIT-IDENTICAL (columns independent). Default 96; drop to 48 if radiation OOMs on
# a memory-tight card (per-band fields are ~3GB/globe, so fewer/bigger chunks cost
# more memory -- 96=2 chunks ~1.5GB each, 48=4 chunks ~0.75GB each).
# The RAD_FAST path (see the RAD_FAST block below) takes its own default of
# 192 = the whole globe in one chunk: the solve is launch-bound, so every extra
# chunk costs ~0.45 s; peak device memory 8.4 GB vs 6.6 GB at 96. An explicit
# RAD_LAT_CHUNK sets both. RAD_LAT_CHUNK is the chunk loop's, RAD_LAT_CHUNK_FAST
# the fast path's; which one a driver uses is decided per driver (RAD_FAST=auto).
#
# RAD_FAST (perf switch; 2026-11-23 default auto, was 0):
#   unset/auto  ON where validated -- decided per RadiationDriver at construction
#               (one visible GPU, wet optics, RAD_SOLAR=window_gq, sub-knobs at 1),
#               otherwise OFF; on an ON driver every call the fast path does not
#               cover (split=True, a one-solve solar mode, _gq_full) runs the
#               chunk loop, i.e. exactly the RAD_FAST=0 path. The header says
#               "RAD_FAST=auto -> 0|1 (reason)".
#   1           forced: SystemExit outside the validated path (as before).
#   0           off: the chunk loop, bit-identical to every run before the switch.
_RAD_FAST_ENV = os.environ.get('RAD_FAST', 'auto').strip().lower()
RAD_FAST_MODE = 'auto' if _RAD_FAST_ENV in ('', 'auto') else ('off' if _RAD_FAST_ENV == '0' else 'on')
RAD_FAST = RAD_FAST_MODE != 'off'     # requested (auto or 1); the driver resolves it (self.fast)
_RAD_LAT_CHUNK_ENV = os.environ.get('RAD_LAT_CHUNK')
RAD_LAT_CHUNK = int(_RAD_LAT_CHUNK_ENV or '96')
RAD_LAT_CHUNK_FAST = int(_RAD_LAT_CHUNK_ENV or '192')
# Size the optics on the WET H2SO4/H2O droplet rather than the dry SO4 core.
# ON by default as of 2026-08-03; 0 reproduces the legacy dry-diameter optics.
WET_OPTICS = os.environ.get('WET_OPTICS', '1') != '0'
N_QUAD_WL = 3                  # wavelength quadrature points per band
# Solar geometry of a radiation call (2026-11-11). The call's heating is held
# for the whole interval it stands for (STEP_HOURS * RAD_EVERY), and the steps
# land on the same UTC hours every day, so a single instant of sun:
#   instant  the sun AT the call hour, held for the interval. Bit-identical to
#            every run before this flag. Each longitude sees the same 4 local
#            times all run long: daily-mean TOA insolation carries a stationary
#            wave-4 error of -24..+23 % (equator to 60 deg, all seasons, 6-h
#            steps; scripts/validation/test_solar_window.py), and columns
#            near the terminator get all-or-nothing SW for 6 h -- the dusk/dawn
#            artifact in dT_rad.
#   window   the sunlit-mean cos(zenith) over the interval CENTRED on the call
#            hour, with the TOA irradiance scaled by the sunlit fraction, so
#            TSI * f * mu_eff = the interval-mean insolation exactly (Hogan &
#            Hirahara 2016, GRL 43, 482). Same cost: one solve. What it does not
#            capture is the non-linearity of absorption/scattering in mu across
#            the interval -- NOT small: measured +20 % daily-mean global ARF vs
#            hourly radiation (5 days; instant +0.1 %), though local error halves. Not
#            the default until fixed; docs/RADIATION.md "Solar geometry".
# Experimental variants (2026-11-12), the 2x2 of {mu choice} x {solves per call};
# results in docs/RADIATION_ARTIFACT_EXPLORATION.md:
#   window_ins   one solve at the ENERGY-weighted cos(zenith) of the interval,
#                mu_ins = int(mu^2)/int(mu) (Cronin 2014, JAS 71, 2994), TOA
#                irradiance TSI * mu_bar / mu_ins so the insolation stays exact
#   window2      TWO solves per call: the interval's two halves, each with its
#                own sunlit-mean mu, sunlit fraction and window albedo, averaged
#   window2_ins  two halves, each at its energy-weighted mu
#   window_blend one solve at mu = (1 - a) * mu_sunlit + a * mu_ins, a =
#                RAD_SOLAR_ALPHA, TOA irradiance TSI * mu_bar / mu so the
#                insolation stays exact. window and window_ins bias the forcing
#                in opposite directions; a is fitted on days independent of the
#                five test days (runs/solar_blend/RESULTS.md).
# Experimental (2026-11-12), the two follow-ups to the 2x2:
#   dither       one solve at the INSTANT sun of a time shifted within the
#                interval, t + d_n, d_n = W * (frac(n * g) - 1/2), g the golden
#                ratio conjugate, n the absolute call index (from the date, so
#                restarts and the reference cache see the same offsets). Plain
#                TSI as under instant. Unbiased like instant (expectation over d
#                = the interval mean of every nonlinear response, and every call
#                still samples all local times across longitudes), but the
#                sampling error no longer sits at fixed longitudes, so it should
#                average out in dT_rad instead of building the wave-4 pattern.
#   window_gq    RAD_SOLAR_NODES solves per call at the Gauss-Legendre nodes of
#                the SUNLIT part of the interval (per column), each at its
#                instantaneous sun, weighted by the quadrature; the insolation
#                is renormalised to the exact interval mean. Converges to the
#                interval mean (of a frozen state) as the node count grows.
# The order is the PHYS_CFG code (coupling.py _SOLAR_MODES): append-only.
SOLAR_MODES = ('instant', 'window', 'window_ins', 'window2', 'window2_ins', 'window_blend',
               'dither', 'window_gq')
SOLAR_TWO = ('window2', 'window2_ins')          # two solves per radiation call
SOLAR_MULTI = SOLAR_TWO + ('window_gq',)        # more than one solve per radiation call
# Default window_gq at 3 nodes (2026-11-17): the one-year A/B stops the stripes in
# dT_rad growing (101 vs 484 mK at day 365) at ~+10 % run time with the reference
# cached (docs/RADIATION_ARTIFACT_EXPLORATION.md 3.9.5). Under RAD_ENGINE=emulator,
# which cannot solve window_gq, the default stays instant. coupling.py duplicates
# this rule for PHYS_CFG; keep the two in step.
RAD_SOLAR_DEFAULT = ('instant' if os.environ.get('RAD_ENGINE', 'rrtmgp').lower() == 'emulator'
                     else 'window_gq')
RAD_SOLAR = os.environ.get('RAD_SOLAR', RAD_SOLAR_DEFAULT).lower()
if RAD_SOLAR not in SOLAR_MODES:
    raise SystemExit(f"RAD_SOLAR={RAD_SOLAR!r} is not valid: {' | '.join(SOLAR_MODES)}")
# the blend weight of window_blend: 0 = window (sunlit-mean mu), 1 = window_ins.
# Default 0.53: where the daily-mean global ARF error vs hourly radiation crosses
# zero, mean of 6 fit days (per-day crossings 0.39..0.70; runs/solar_blend/RESULTS.md)
RAD_SOLAR_ALPHA = float(os.environ.get('RAD_SOLAR_ALPHA', '0.53'))
if not 0.0 <= RAD_SOLAR_ALPHA <= 1.0:
    raise SystemExit(f"RAD_SOLAR_ALPHA={RAD_SOLAR_ALPHA} is not in [0, 1]")
# window_gq's node count (solves per radiation call). Default 3 (2026-11-17; was 2):
# global ARF +1.1 % vs +4.2 % against hourly radiation, for +16 % vs +7 % per step.
RAD_SOLAR_NODES = int(os.environ.get('RAD_SOLAR_NODES', '3'))
if not 1 <= RAD_SOLAR_NODES <= 8:
    raise SystemExit(f"RAD_SOLAR_NODES={RAD_SOLAR_NODES} is not in 1..8")
# where window_gq's window sits relative to the call hour t, the LAST hour of the
# coupling step whose end state radiation sees: 'trailing' = t - W .. t, the step
# just finished (default, 2026-11-17), so the heating added to dT_rad at t is the
# heating of hours that have happened; 'centred' = t - W/2 .. t + W/2 (the first
# window_gq, 2026-11-12..17), half of it in the next step. 3-day check from both
# day-365 states against hourly radiation (scripts/validation/check_solar_align.py):
# column error equal (9.3 / 12.0 % both), per-level error +0.3..0.6 pt, stripes
# -0.5 pt, global ARF +1.14 vs +1.19 %. Only window_gq reads it; the other window
# modes stay centred.
RAD_SOLAR_ALIGN = os.environ.get('RAD_SOLAR_ALIGN', 'trailing').lower().replace('centered', 'centred')
if RAD_SOLAR_ALIGN not in ('trailing', 'centred'):
    raise SystemExit(f"RAD_SOLAR_ALIGN={RAD_SOLAR_ALIGN!r} is not trailing | centred")


# =========================================================================
# Palmer & Williams (1975) 75 wt% H2SO4 refractive index
# =========================================================================
# =========================================================================
# WET aerosol size + composition (H2SO4/H2O solution droplets)
# =========================================================================
# TOMAS carries DRY SO4 mass. Optics must use the WET droplet, because that is
# the particle light actually sees. Before 2026-08-03 the Mie tables were built
# on the dry diameter DP_BIN (= cbrt(MMID/RHO_AER*6/pi), RHO_AER=1770) while the
# refractive index was that of a 75 wt% H2SO4 SOLUTION -- i.e. a droplet already
# 25% water by mass, sized as if it were dry. That mismatch alone underestimated
# 550 nm extinction by ~47% for this model's size distribution; with the real
# RH/T-dependent composition (which reaches 40-55 wt% near the moist 143 hPa base
# of the band) the error runs to a factor of 2 or more.
#
# The composition and density parameterizations are the SAME ones the fast
# microphysics engine already applies internally every step
# (tomas_jax.fast.water.h2so4_equilibrium_wt, Tabazadeh et al. 1997 GRL 24,
# 1931-1934; tomas_jax.fast.density.calc_density, Tang 1997 JGR 102, 1883-1893),
# so the optics now see the same droplet the coagulation kernel does. Previously
# the engine computed equilibrium water, used it for the microphysics, and then
# driver_fast.py read back only the dry SO4 column -- so the wet size existed
# inside the timestep and was discarded before radiation ever ran.
# tang_density/wet_size live in settling.py: gravitational settling needs the same
# wet diameter and solution density, and settling.py is dependency-light (numpy +
# jax only), so putting the canonical copy there lets both consumers share one
# implementation without radiation.py's netCDF4/rrtmgp imports leaking into it.
from settling import tang_density, wet_size

WT_GRID = np.arange(10.0, 80.0 + 1e-9, 5.0)      # wt% H2SO4, Tabazadeh Table 1 grid
_WT_DGRID = float(WT_GRID[1] - WT_GRID[0])       # uniform 5 wt% spacing

# Palmer & Williams (1975) tabulates n,k at these six solution strengths; the
# .dat columns are [cm-1, um, 25%, 38%, 50%, 75%, 84.5%, 95.6%].
_RI_WT = np.array([25.0, 38.0, 50.0, 75.0, 84.5, 95.6])
_RI_COL = np.array([2, 3, 4, 5, 6, 7])

##the water fraction is whatever the local thermodynamics says!!!!!!!!!!! same with settling, both a function of RH

def refindex_at_wt(wt_pct, path=RI_FILE):
    """(wl_um, n, k) for an H2SO4 solution of wt_pct, linearly interpolated
    across Palmer & Williams' six tabulated strengths.

    Composition changes the refractive index as well as the size -- a droplet
    diluting from 80 to 45 wt% moves n,k toward water's. Holding the RI at 75 wt%
    while growing the diameter would trade one inconsistency for another.
    Outside 25-95.6 wt% the nearest tabulated column is used (clamped), which
    only bites below 25 wt%; the Tabazadeh composition floor is 10 wt%.
    """
    wl, ncols, kcols = _load_refindex_all(path)
    w = float(np.clip(wt_pct, _RI_WT[0], _RI_WT[-1]))
    j = int(np.clip(np.searchsorted(_RI_WT, w), 1, len(_RI_WT) - 1))
    w0, w1 = _RI_WT[j - 1], _RI_WT[j]
    f = (w - w0) / (w1 - w0)
    n = (1.0 - f) * ncols[:, j - 1] + f * ncols[:, j]
    k = (1.0 - f) * kcols[:, j - 1] + f * kcols[:, j]
    return wl, n, k


def _load_refindex_all(path=RI_FILE):
    """(wl_um ascending, n[:, 6], k[:, 6]) -- all six solution strengths."""
    rows = []
    with open(path) as f:
        for line in f:
            parts = line.split()
            if len(parts) == 8:
                try:
                    rows.append([float(p) for p in parts])
                except ValueError:
                    continue
    rows = np.asarray(rows)
    assert rows.shape[0] % 2 == 0 and rows.shape[0] > 0, \
        f'unexpected palmer_williams format ({rows.shape[0]} numeric rows)'
    half = rows.shape[0] // 2
    real, imag = rows[:half], rows[half:]
    assert np.allclose(real[:, 0], imag[:, 0]), 'real/imag wavenumber mismatch'
    wl = real[:, 1]
    order = np.argsort(wl)
    return wl[order], real[order][:, _RI_COL], imag[order][:, _RI_COL]


def load_h2so4_refindex(path=RI_FILE, wt_col=5):
    """
    The refractive index of sulfuric acid tells you how those droplets interact with light
    75% weight is the equilibrium composition of H2SO4 in the stratosphere, so we use that for the Mie calculations (75% of droplet weight is sulfuric acid, 25% is water).
    
    Parse the HITRAN palmer_williams_h2so4.dat file.
    https://hitran.org/data/Aerosols/Aerosols%20data%20from%20previous%20HITRAN/Aerosols-2016/ascii/single_files/palmer_williams_h2so4.dat
    wt_col: column index in [cm-1, um, 25%, 38%, 50%, 75%, 84.5%, 95.6%] (could pick any of these...);
    default 5 -> 75 wt%. Returns (wl_um ascending, n, k).
"""
    rows = []
    with open(path) as f:
        for line in f:
            parts = line.split()
            if len(parts) == 8:
                try:
                    rows.append([float(p) for p in parts])
                except ValueError:
                    continue
    rows = np.asarray(rows)
    assert rows.shape[0] % 2 == 0 and rows.shape[0] > 0, \
        f'unexpected palmer_williams format ({rows.shape[0]} numeric rows)'
    half = rows.shape[0] // 2
    real, imag = rows[:half], rows[half:]
    # both blocks are sorted by wavenumber; sanity-check they line up
    assert np.allclose(real[:, 0], imag[:, 0]), 'real/imag wavenumber mismatch'
    wl = real[:, 1]
    order = np.argsort(wl)
    return wl[order], real[order, wt_col], imag[order, wt_col]


def band_limits_um(nc_path):
    """RRTMGP band limits [wavenumber cm-1] -> (n_bnd, 2) wavelengths in um."""
    with netCDF4.Dataset(nc_path) as d:
        wn = np.asarray(d['bnd_limits_wavenumber'][:])   # (n_bnd, 2) cm-1
    wl = 1.0e4 / wn[:, ::-1]                             # cm to um, ascending per band by flipping axis
    return wl


# =========================================================================
# Mie tables: per-bin x per-band extinction cross-section, ssa, g
# =========================================================================
def build_mie_tables(dp_bin_m, nc_path, nq=N_QUAD_WL, ri_path=RI_FILE,
                     wt_pct=75.0):
    """Band-averaged Mie properties for each TOMAS bin.

    dp_bin_m : (NBINS,) bin diameters [m] -- WET diameters in normal use; the
               caller is responsible for having grown them (see wet_size).
    wt_pct   : H2SO4 weight percent selecting the refractive index, so size and
               composition stay consistent. 75.0 reproduces the pre-2026-08-03
               fixed-composition tables.
    Returns dict of np arrays: sigma_ext (NBINS, n_bnd) [m2/particle],
    ssa (NBINS, n_bnd), g (NBINS, n_bnd).
    Band average = mean over nq log-spaced wavelengths inside the band
    (flat weighting; bands are narrow enough that this beats band-center
    evaluation without needing solar/Planck weights).
    """
    #refractive index of the solution at THIS weight percent
    wl_tab, n_tab, k_tab = refindex_at_wt(wt_pct, ri_path)
    #get wavelength edges of each RRTMGP band in microns
    lims = band_limits_um(nc_path)                       # (nb, 2) um
    nb = lims.shape[0]
    nbin = dp_bin_m.size
    # quadrature wavelengths per band (um), log-spaced interior points
    tq = (np.arange(nq) + 0.5) / nq
    #log-linear interpolation between band edges to get quadrature points
    wlq = np.exp(np.log(lims[:, :1]) +
                 tq[None, :] * (np.log(lims[:, 1:]) - np.log(lims[:, :1])))  # (nb, nq)
    # refractive index at each quadrature wavelength (clamped to data range)
    wlq_c = np.clip(wlq, wl_tab[0], wl_tab[-1])
    #linearly interpolate real and imaginary refractive index to quadrature wavelengths
    n_q = np.interp(wlq_c, wl_tab, n_tab)
    k_q = np.interp(wlq_c, wl_tab, k_tab)
    # size parameters: (nb, nq, nbin) pi*diameter/wavelength
    # dimensionless ratio of a particle's circumference to the wavelength of light; determines scattering regime
    x = np.pi * dp_bin_m[None, None, :] / (wlq[:, :, None] * 1e-6)
    # bhmie's fixed series length caps x ~< 220; beyond that Mie efficiencies
    # have converged to the geometric-optics limit, so clamping is safe.
    x = np.minimum(x, 200.0)
    #complex refractive index: n + i*k
    m = (n_q + 1j * np.maximum(k_q, 1e-12))[:, :, None] * np.ones_like(x)
    #given a size parameter and complex refractive index, bhmie computes the scattering and extinction efficiencies, and what direction light is scattered in (g)
    qe, qs, gs = jax.vmap(bhmie_qsca_jax)(jnp.asarray(x.ravel()),
                                          jnp.asarray(m.ravel()))
    qe = np.asarray(qe).reshape(nb, nq, nbin)
    qs = np.asarray(qs).reshape(nb, nq, nbin)
    gs = np.asarray(gs).reshape(nb, nq, nbin)
    qs = np.minimum(qs, qe)                              # numerical guard...scattering efficiency cannot exceed extinction efficiency
    area = np.pi / 4.0 * dp_bin_m**2                     # geometric cross-sectional area of a sphere using diameter (pi*r^2 = pi*(d/2)^2 = pi*d^2/4)
    #average scattering and extinction efficiencies over quadrature points, then multiply by geometric area to get cross-sections
    sigma_ext = qe.mean(axis=1).T * area[:, None]        # (nbin, nb)
    sigma_sca = qs.mean(axis=1).T * area[:, None]
    #average asymmetry parameter over quadrature points (-1 (backwards) to 1 (forwards)), weighted by scattering efficiency (qs)
    #g values where scattering is high are weighted more, as they contribute more to the overall scattering behavior of the particle
    g_bin = ((gs * qs).mean(axis=1) / np.maximum(qs.mean(axis=1), 1e-30)).T
    # ratio of scattering to extinction cross-section, dimensionless (single scattering albedo)
    ssa = sigma_sca / np.maximum(sigma_ext, 1e-30)
    return {'sigma_ext': sigma_ext, 'ssa': ssa, 'g': g_bin}


def build_wet_mie_tables(mmid_kg, rho_dry, nc_path, nq=N_QUAD_WL,
                         ri_path=RI_FILE, wt_grid=WT_GRID):
    """Mie tables on the (wt%, bin, band) mesh for wet H2SO4/H2O droplets.

    For each solution strength in wt_grid the bins are grown to their wet
    diameter (wet_size) AND the refractive index is taken at that same strength,
    then the usual band-averaged Mie integration runs. heating() interpolates
    these in wt% per grid cell.

    Cost: len(wt_grid) Mie builds instead of one, paid once at init. The runtime
    cost is len(wt_grid) small einsums per radiation call, which is negligible
    next to the RRTMGP two-stream solve.

    Returns dict of (n_wt, NBINS, n_bnd) arrays plus 'dp_wet' (n_wt, NBINS) [m].
    """
    out = {k: [] for k in ('sigma_ext', 'ssa', 'g')}
    dp_wet_all = []
    for w in wt_grid:
        dp_wet, _ = wet_size(mmid_kg, w, rho_dry)
        dp_wet = np.asarray(dp_wet)        # wet_size is jnp-based; Mie build is numpy
        t = build_mie_tables(dp_wet, nc_path, nq=nq, ri_path=ri_path, wt_pct=w)
        for k in out:
            out[k].append(t[k])
        dp_wet_all.append(dp_wet)
    res = {k: np.stack(v, axis=0) for k, v in out.items()}
    res['dp_wet'] = np.stack(dp_wet_all, axis=0)
    return res


def wt_weights(wt_pct, wt_grid=WT_GRID, dwt=_WT_DGRID):
    """Linear-interpolation weights of wt_pct onto the uniform wt_grid.

    Returns (n_wt, ...) hat-function weights summing to 1 along axis 0. Used
    instead of an index gather because gathering per-cell Mie tables would
    materialize (cells, NBINS, n_bnd) -- tens of GB. With weights we instead
    accumulate n_wt cheap einsums, each the size of the original single-table
    one, and the result is identical to per-cell linear interpolation.
    """
    w = jnp.clip(jnp.asarray(wt_pct), wt_grid[0], wt_grid[-1])
    d = jnp.abs(w[None, ...] - jnp.asarray(wt_grid).reshape(
        (-1,) + (1,) * jnp.ndim(w)))
    return jnp.clip(1.0 - d / dwt, 0.0, 1.0)


def mie_at_wavelength(dp_bin_m, wl_um, ri_path=RI_FILE, wt_pct=75.0):
    """(sigma_ext, sigma_sca) per bin at a single wavelength (diagnostics).
    Similar to function above but only evaluates at one wavelength, so no quadrature or band averaging.
    Production path never uses this default wt_pct"""
    wl_tab, n_tab, k_tab = refindex_at_wt(wt_pct, ri_path)
    #interpolate n and k at one wavelength
    n = np.interp(wl_um, wl_tab, n_tab); k = np.interp(wl_um, wl_tab, k_tab)
    x = jnp.asarray(np.pi * dp_bin_m / (wl_um * 1e-6))
    m = jnp.full(x.shape, n + 1j * max(k, 1e-12))
    qe, qs, _ = jax.vmap(bhmie_qsca_jax)(x, m)
    area = np.pi / 4.0 * dp_bin_m**2
    # convert dimensionless efficiencies to cross-sections by multiplying by geometric area
    #used later to compte column AOD
    return np.asarray(qe) * area, np.asarray(qs) * area


# =========================================================================
# Aerosol optical fields from bin number mixing ratios
# =========================================================================
@jax.jit
def _aod550(num_mr, dp_pa, sig_ext_550):
    """Column AOD at 550 nm: (NBINS, nzb, nx, nlon) -> (nx, nlon).
    converting number mixing ratio to column number density by using layer pressure thickness and gravity
    from #/kg to #/m^2"""
    ncol = num_mr * (dp_pa[None, :, None, None] / GRAV)
    return jnp.einsum('k,kzyx->yx', sig_ext_550, ncol)


@jax.jit
def _aod550_wet(num_mr, dp_pa, sig550_wt, wwt):
    """As _aod550 but with per-cell composition. sig550_wt (n_wt, NBINS),
    wwt (n_wt, nzb, nx, nlon) the wt%-interpolation weights."""
    ncol = num_mr * (dp_pa[None, :, None, None] / GRAV)
    # sum_j sum_k sig[j,k] * (ncol[k,cell] * w[j,cell])
    return jnp.einsum('jk,jkzyx->yx', sig550_wt, ncol[None] * wwt[:, None])


@jax.jit
def _aerosol_props_wet(num_mr, dp_pa, sigma_ext, ssa, g, wwt):
    """Per-cell wet-composition aerosol optics.

    sigma_ext/ssa/g : (n_wt, NBINS, n_bnd) tables from build_wet_mie_tables
    wwt             : (n_wt, nzb, nlat, nlon) linear weights over WT_GRID
    *g_f             : (n_bnd, nzb, nlat, nlon) extinction-weighted asymmetry factor

    Mathematically identical to evaluating each cell's own interpolated Mie
    table, but accumulated as n_wt einsums so nothing of size
    (cells, NBINS, n_bnd) is ever materialized. tau/sca/ag are summed over the
    composition axis BEFORE the ssa/g ratios are formed, which is what keeps the
    result a proper extinction-weighted mixture rather than an average of ratios.
    """
    ncol = num_mr * (dp_pa[None, :, None, None] / GRAV)      # (k,z,y,x)
    nw = ncol[None] * wwt[:, None]                           # (j,k,z,y,x)
    tau = jnp.einsum('jkb,jkzyx->bzyx', sigma_ext, nw)
    sca = jnp.einsum('jkb,jkzyx->bzyx', sigma_ext * ssa, nw)
    ag = jnp.einsum('jkb,jkzyx->bzyx', sigma_ext * ssa * g, nw)
    ssa_f = sca / jnp.maximum(tau, 1e-30)
    g_f = ag / jnp.maximum(sca, 1e-30)
    return tau, ssa_f, g_f


@jax.jit
def _aerosol_props(num_mr, dp_pa, sigma_ext, ssa, g):
    """num_mr (NBINS, nzb, nlat, nlon) [#/kg]; dp_pa (nzb,) layer thickness.
    Returns tau, ssa, g each (n_bnd, nzb, nlat, nlon) in CESM level order."""
    ncol = num_mr * (dp_pa[None, :, None, None] / GRAV)          # #/m2 per layer
    #cross-section times number density gives optical depth per layer; sum over bins to get total optical depth
    tau = jnp.einsum('kb,kzyx->bzyx', sigma_ext, ncol)
    #scattering optical depth
    sca = jnp.einsum('kb,kzyx->bzyx', sigma_ext * ssa, ncol)
    #scattering cross-section weighted by asymmetry factor
    ag  = jnp.einsum('kb,kzyx->bzyx', sigma_ext * ssa * g, ncol)
    #extinction weighted single scattering albedo and asymmetry factor
    ssa_f = sca / jnp.maximum(tau, 1e-30)
    g_f   = ag / jnp.maximum(sca, 1e-30)
    return tau, ssa_f, g_f


# =========================================================================
# Solar geometry (noleap calendar)
# =========================================================================
"""https://www.ncei.noaa.gov/sites/default/files/2021-07/GOES-R_ABI_solar_zenith_angle_description_12.docx"""
def solar_zenith(time_val, lat_deg, lon_deg, offset_h=0.0):
    """Solar zenith angle [rad], shape (nlat, nlon), offset_h hours from
    time_val (cftime noleap). offset_h=0 is bit-identical to the original."""
    hour = time_val.hour + offset_h       # == time_val.hour exactly when offset_h == 0
    #fractional day of year
    doy = time_val.dayofyr + hour / 24.0
    #solar declination (ranging from -23.44 to +23.44 degrees over the year)
    #+10 shifts to solstice (doy=172) for max declination, and to equinox (doy=80) for zero declination
    dec = -23.44 * np.pi / 180.0 * np.cos(2 * np.pi * (doy + 10.0) / 365.0)
    #how far the sun has rotated from local noon (ranging from -pi to +pi over the day)
    hang = 2 * np.pi * (hour / 24.0 - 0.5) + np.deg2rad(lon_deg)
    phi = np.deg2rad(lat_deg)
    #
    cosz = (np.sin(phi)[:, None] * np.sin(dec) +
            np.cos(phi)[:, None] * np.cos(dec) * np.cos(hang)[None, :])
    #convert to zenith angle, clamp to [-1, 1] to avoid NaN from arccos
    return np.arccos(np.clip(cosz, -1.0, 1.0))


GOLDEN = 0.5 * (np.sqrt(5.0) - 1.0)


def dither_offset(time_val, width_h):
    """RAD_SOLAR=dither: hours to shift the sun of the call at time_val, in
    [-width_h/2, width_h/2). A golden-ratio (Weyl) sequence in the absolute
    call index n = round(hours since 0000-01-01 / width_h), so it depends only on
    the date: consecutive calls, and the same UTC hour on consecutive days
    (n steps by 24/width_h), both fill the interval quasi-uniformly."""
    hrs = ((time_val.year * 365 + time_val.dayofyr - 1) * 24 + time_val.hour)
    n = int(round(hrs / float(width_h)))
    return float(width_h) * (((n * GOLDEN) % 1.0) - 0.5)


def solar_gauss(time_val, lat_deg, lon_deg, width_h, nodes, offset_h=0.0):
    """RAD_SOLAR=window_gq: [(zenith, irradiance scale), ...] of `nodes` solves,
    each (nlat, nlon), at the Gauss-Legendre nodes of the sunlit part of the
    width_h-hour window centred on time_val. The scales carry the quadrature
    weight TIMES `nodes`, so the plain MEAN of the solves (heating()) is the
    quadrature sum; they are renormalised so that mean(scale * cos zenith) is
    the window-mean insolation mu_bar exactly. A column whose window holds a
    short night between two sunlit pieces (polar summer only) puts its nodes on
    the longer piece (the renormalisation keeps its insolation exact); a fully
    sunlit window (polar day) on the whole window. Night columns: zenith = pi,
    scale 0. Declination held at the window centre, as in solar_window().
    offset_h moves the window centre offset_h hours from time_val (0 = centred
    on the call; -width_h/2 = the window ending at the call)."""
    hour = time_val.hour + offset_h
    doy = time_val.dayofyr + hour / 24.0
    dec = -23.44 * np.pi / 180.0 * np.cos(2 * np.pi * (doy + 10.0) / 365.0)
    phi = np.deg2rad(np.asarray(lat_deg, np.float64))[:, None]
    a = np.sin(phi) * np.sin(dec)
    b = np.cos(phi) * np.cos(dec)
    H = np.arccos(np.clip(-a / np.maximum(b, 1e-12), -1.0, 1.0))     # (nlat, 1)
    hc = 2 * np.pi * (hour / 24.0 - 0.5) + np.deg2rad(np.asarray(lon_deg, np.float64))[None, :]
    hw = np.mod(hc + np.pi, 2 * np.pi) - np.pi                        # window centre in [-pi, pi)
    half = np.pi * float(width_h) / 24.0
    w0, w1 = hw - half, hw + half
    los, his = [], []
    for m in (-1, 0, 1):
        los.append(np.maximum(w0, -H + 2 * np.pi * m)); his.append(np.minimum(w1, H + 2 * np.pi * m))
    k = np.argmax(np.stack([h - l for l, h in zip(los, his)]), axis=0)  # the longest sunlit piece
    lo = np.choose(k, los); hi = np.choose(k, his)
    whole = np.broadcast_to(H >= np.pi - 1e-12, lo.shape)             # polar day: all of it
    lo = np.where(whole, w0, lo); hi = np.where(whole, w1, np.maximum(hi, lo))
    x, wq = np.polynomial.legendre.leggauss(int(nodes))
    zs, mus = solar_window(time_val, lat_deg, lon_deg, width_h, offset_h=offset_h)
    mu_bar = np.where(mus > 0, mus * np.cos(zs), 0.0)
    parts, ins = [], 0.0
    for xi, wi in zip(x, wq):
        h = lo + 0.5 * (xi + 1.0) * (hi - lo)
        mu = np.maximum(a + b * np.cos(h), 0.0)
        sc = nodes * 0.5 * wi * (hi - lo) / (2 * half)
        parts.append([mu, sc]); ins = ins + sc * mu / nodes
    lit = (mu_bar > 1e-12) & (ins > 1e-15)
    ren = np.where(lit, mu_bar / np.where(lit, ins, 1.0), 0.0)
    out = []
    for mu, sc in parts:
        on = lit & (mu > 0)
        out.append((np.where(on, np.arccos(np.clip(mu, 0.0, 1.0)), np.pi), np.where(on, sc * ren, 0.0)))
    return out


def solar_window(time_val, lat_deg, lon_deg, width_h, offset_h=0.0, mu='sunlit', alpha=None):
    """Effective solar zenith [rad] and TOA-irradiance scale over the
    width_h-hour window centred offset_h hours from time_val, each (nlat, nlon).
    Night (no sun in the window) columns get zenith = pi and scale 0. Same
    declination and hour-angle conventions as solar_zenith(), declination held
    at the window centre (< 0.1 deg change over 6 h).

    mu='sunlit'  zenith = arccos(mu_eff), mu_eff = mu_bar / f, the sunlit-mean
                 cos zenith (Hogan & Hirahara 2016); scale = f, the sunlit
                 fraction. width_h -> 0 recovers solar_zenith() on the sunlit side.
    mu='energy'  zenith = arccos(mu_ins), mu_ins = int(mu^2) / int(mu), the
                 insolation-weighted cos zenith (Cronin 2014); scale =
                 mu_bar / mu_ins.
    mu='blend'   zenith = arccos((1 - alpha) * mu_eff_sunlit + alpha * mu_ins),
                 scale = mu_bar / that; alpha in [0, 1] (0 = sunlit, 1 = energy).
    Either way scale * cos(zenith) = mu_bar, the window mean of
    max(cos zenith, 0), so the caller's TSI * scale makes the window's
    insolation TSI * mu_bar exact.

    Closed form, exact for any width (also > 24 h): with cos(zenith) =
    a + b*cos(h) in the hour angle h, the sunlit arc is |h| < H with
    H = arccos(-a/b), and the integrals of max(cos zenith, 0), of its square
    and of the sunlit indicator have the periodic antiderivatives G, Q, L below.
    """
    if mu not in ('sunlit', 'energy', 'blend'):
        raise ValueError(f'solar_window: mu={mu!r} is not sunlit | energy | blend')
    if mu == 'blend' and (alpha is None or not 0.0 <= float(alpha) <= 1.0):
        raise ValueError(f'solar_window: mu=blend needs alpha in [0, 1], got {alpha!r}')
    hour = time_val.hour + offset_h       # == time_val.hour exactly when offset_h == 0
    doy = time_val.dayofyr + hour / 24.0
    dec = -23.44 * np.pi / 180.0 * np.cos(2 * np.pi * (doy + 10.0) / 365.0)
    phi = np.deg2rad(np.asarray(lat_deg, np.float64))[:, None]
    a = np.sin(phi) * np.sin(dec)                       # (nlat, 1)
    b = np.cos(phi) * np.cos(dec)                       # (nlat, 1), >= 0
    # half-length of the sunlit hour-angle arc: 0 = polar night, pi = polar day
    # (b ~ 0 at the poles: the ratio saturates to the right side of the clip)
    H = np.arccos(np.clip(-a / np.maximum(b, 1e-12), -1.0, 1.0))
    sinH = np.sin(H)
    P = 2.0 * (a * H + b * sinH)                         # integral over one day
    PQ = 2.0 * a * a * H + 4.0 * a * b * sinH + b * b * (H + 0.5 * np.sin(2.0 * H))

    def _cum(x):
        """(G, L, Q): integrals of max(a + b cos h, 0), of [a + b cos h > 0]
        and of max(a + b cos h, 0)^2 from -pi to x, continued periodically."""
        n = np.floor((x + np.pi) / (2 * np.pi))
        yc = np.clip(x - 2 * np.pi * n, -H, H)
        G = n * P + a * (yc + H) + b * (np.sin(yc) + sinH)
        L = n * 2.0 * H + (yc + H)
        Q = (n * PQ + a * a * (yc + H) + 2.0 * a * b * (np.sin(yc) + sinH)
             + b * b * (0.5 * (yc + H) + 0.25 * (np.sin(2.0 * yc) + np.sin(2.0 * H))))
        return G, L, Q

    hc = 2 * np.pi * (hour / 24.0 - 0.5) + np.deg2rad(np.asarray(lon_deg, np.float64))[None, :]
    half = np.pi * float(width_h) / 24.0
    G2, L2, Q2 = _cum(hc + half)
    G1, L1, Q1 = _cum(hc - half)
    mu_bar = np.maximum(G2 - G1, 0.0) / (2 * half)
    f = np.clip((L2 - L1) / (2 * half), 0.0, 1.0)
    lit = (f > 1e-9) & (mu_bar > 1e-12)
    if mu == 'sunlit':
        mu_eff = np.where(lit, np.clip(mu_bar / np.where(lit, f, 1.0), 0.0, 1.0), 0.0)
        scale = f
    else:
        mu_eff = np.where(lit, np.clip(np.maximum(Q2 - Q1, 0.0) / np.where(lit, G2 - G1, 1.0),
                                       0.0, 1.0), 0.0)
        if mu == 'blend':
            # mu_ins >= mu_bar / f (Cauchy-Schwarz), so the blend lies between them
            mu_s = np.where(lit, np.clip(mu_bar / np.where(lit, f, 1.0), 0.0, 1.0), 0.0)
            a = float(alpha)
            mu_eff = (1.0 - a) * mu_s + a * mu_eff
        scale = np.where(lit, mu_bar / np.where(lit & (mu_eff > 0), mu_eff, 1.0), 0.0)
    zen = np.where(lit, np.arccos(mu_eff), np.pi)
    return zen, np.where(lit, scale, 0.0)


# =========================================================================
# RAD_FAST: the radiation fast path (perf switch; default auto since 2026-11-23,
# see RAD_FAST_MODE above; 0 = the chunk loop, unchanged). Ported by hand from
# AIDE-SAI-link 9d693d2 (rad agent, 2026-11-13) and ab74f46 (window_gq fast path,
# 2026-11-17). On a driver where it is on (ONE device, wet optics, window_gq),
# heating()'s window_gq anomaly solve runs:
#   * one LW solve per lat chunk (RAD_LAT_CHUNK_FAST defaults to 192 = one chunk),
#   * the SW solve on the sunlit columns only (compacted, padded to RAD_SW_PAD),
#     once per Gauss node under RAD_SOLAR=window_gq (RAD_GQ_FAST), combined under
#     jit exactly as _hr_multi_impl does,
#   * g-points vmap'ed RAD_GBATCH at a time (rrtmgp_batched.py; same per-element
#     ops, summed in g order),
#   * the minor-gas scan over each band's own intervals only (RAD_DYN_MINOR) and
#     the lower/upper minor tables on their own levels only (RAD_SPLIT_TROPO;
#     needs level-only, monotone column pressure -- asserted at trace time),
#   * no cloud optics (RAD_SKIP_CLOUD; clear sky, zero-cloud combine kept for SW).
# The sub-knobs default to 1 unless RAD_FAST=0 and are inert on a driver where
# RAD_FAST resolved off. Multi-GPU and dry optics keep the chunk loop.
#
# What is decided when (RAD_FAST=auto):
#   import       the env vars only (RAD_FAST_MODE, the sub-knobs, RAD_LAT_CHUNK*).
#                Nothing is patched or imported here any more.
#   construction RadiationDriver.__init__ resolves self.fast from the visible
#                device count, the optics mode (WET_OPTICS and mmid_kg) and the
#                driver's solar mode; only if it is on does it import
#                rrtmgp_batched, install its minor-gas patches (once per process,
#                _install_fast_patches) and build the cloud-skipping optics
#                (_skip_clouds on a shallow copy of the optics library, so
#                self.rt stays the unpatched RAD_FAST=0 instance).
#   each call    _heating takes the fast path only for the window_gq anomaly
#                solve; any other call on a fast driver runs the chunk loop (auto)
#                or exits (RAD_FAST=1).
# A driver that resolves off therefore runs with nothing installed: the
# RAD_FAST=0 code path. (The patches are also inert outside rrtmgp_batched's
# dyn_ctx -- the wrapped lax.scan / minor_scan_length / compute_minor_optical_depth
# pass straight through when _CTX['on'] is unset -- so an off call in a process
# where another driver turned them on still runs the library's own graph.)
# =========================================================================
def _fast_knob(name, default='1'):
    return RAD_FAST and os.environ.get(name, default) != '0'


RAD_GBATCH = int(os.environ.get('RAD_GBATCH', '4')) if RAD_FAST else 1   # g-points per loop iteration
RAD_DYN_MINOR = _fast_knob('RAD_DYN_MINOR')
RAD_SPLIT_TROPO = _fast_knob('RAD_SPLIT_TROPO')
RAD_SW_PAD = int(os.environ.get('RAD_SW_PAD', '4096'))
RAD_SW_MAXCOLS = int(os.environ.get('RAD_SW_MAXCOLS', '1000000'))
RAD_SW_SKIP_NIGHT = _fast_knob('RAD_SW_SKIP_NIGHT')
RAD_SKIP_CLOUD = _fast_knob('RAD_SKIP_CLOUD')
RAD_GQ_FAST = _fast_knob('RAD_GQ_FAST')     # window_gq through the fast path; 0 = the chunk loop
rrtmgp_batched = None                       # imported by _install_fast_patches
_FAST_PATCHED = False


def _install_fast_patches():
    """Import rrtmgp_batched and install its minor-gas patches, once per process,
    the first time a driver resolves RAD_FAST on (was: at import under RAD_FAST=1)."""
    global rrtmgp_batched, _FAST_PATCHED
    if _FAST_PATCHED:
        return
    import rrtmgp_batched as _rb             # src/radiation/, via _paths
    rrtmgp_batched = _rb
    if RAD_DYN_MINOR:
        rrtmgp_batched.install_dynamic_minor_scan(split_tropo=RAD_SPLIT_TROPO)
    _FAST_PATCHED = True


def rad_fast_summary(fast=None, why=()):
    """One line for the run header. fast = the driver's resolved RAD_FAST
    (None: what the environment asks for); why = the reasons it is off."""
    fast = RAD_FAST if fast is None else fast
    tag = 'RAD_FAST=auto -> ' if RAD_FAST_MODE == 'auto' else 'RAD_FAST='
    if not fast:
        reason = '; '.join(why)               # empty under RAD_FAST=0
        return (f"{tag}0 ({reason + ': ' if reason else ''}"
                f"radiation chunk loop, RAD_LAT_CHUNK={RAD_LAT_CHUNK}, library g-point loop)")
    return (f"{tag}1 (RAD_LAT_CHUNK={RAD_LAT_CHUNK_FAST} RAD_GBATCH={RAD_GBATCH} "
            f"RAD_DYN_MINOR={int(RAD_DYN_MINOR)} RAD_SPLIT_TROPO={int(RAD_SPLIT_TROPO)} "
            f"RAD_SKIP_CLOUD={int(RAD_SKIP_CLOUD)} RAD_SW_SKIP_NIGHT={int(RAD_SW_SKIP_NIGHT)} "
            f"RAD_GQ_FAST={int(RAD_GQ_FAST)} RAD_SW_PAD={RAD_SW_PAD}"
            + ("; calls other than the window_gq solve (split=True, one-solve solar modes) "
               "run the chunk loop, RAD_LAT_CHUNK=" + str(RAD_LAT_CHUNK)
               if RAD_FAST_MODE == 'auto' else "") + ")")


def _no_mark(name, *arrs):
    """Stage-timing hook of the fast path (the rad agent's RAD_PROF); a no-op here."""
    return None


def c_shape(a, ncol):
    return (a.shape[0], a.shape[1], ncol)


def _skip_clouds(lib):
    """RAD_SKIP_CLOUD: this driver is clear-sky -- cloud paths are exactly zero -- but
    jax-rrtmgp is handed zero-valued cloud arrays (not None), so for every one of the 128 LW + 112 SW
    g-points it runs the cloud-optics table lookups and mass-weighted combine of an all-zero cloud.
    Instance-level override (the submodule is untouched) of the two optical-property methods that
    drops the cloud branch but keeps the arithmetic bit-identical:
      LW: gas props have ssa=g=0, so combining with a zero-tau cloud returns exactly (tau, 0, 0);
      SW: the combine is still applied with an all-zero cloud, because (tau*ssa)/tau is not
          bitwise ssa -- skipping it would change the last bit of ssa."""
    if not RAD_SKIP_CLOUD:
        return
    lw0, sw0 = lib.compute_lw_optical_properties, lib.compute_sw_optical_properties

    def lw(pressure, temperature, molecules, igpt, vmr_fields=None, cloud_r_eff_liq=None,
           cloud_path_liq=None, cloud_r_eff_ice=None, cloud_path_ice=None):
        return lw0(pressure, temperature, molecules, igpt, vmr_fields)

    def sw(pressure, temperature, molecules, igpt, vmr_fields=None, cloud_r_eff_liq=None,
           cloud_path_liq=None, cloud_r_eff_ice=None, cloud_path_ice=None):
        gas = sw0(pressure, temperature, molecules, igpt, vmr_fields)
        z = jnp.zeros_like(gas['ssa'])
        return lib.combine_optical_properties(
            gas, {'optical_depth': z, 'ssa': z, 'asymmetry_factor': z})

    lib.compute_lw_optical_properties, lib.compute_sw_optical_properties = lw, sw


@jax.jit
def _gq_combine(hr_lw, olr, lwd, parts):
    """window_gq fast path: _hr_multi_impl's accumulation, (0.0 + s_0 + ...) / N, under jit
    (done eagerly it differs by 1 ulp and the sign of zero: XLA folds the 0.0 + and fuses the / N)."""
    n = len(parts)
    h = dn = up = ds = 0.0
    for p in parts:
        h = h + p[0]; dn = dn + p[1]; up = up + p[2]; ds = ds + p[3]
    h, dn, up, ds = h / n, dn / n, up / n, ds / n
    return hr_lw + h, jnp.stack([dn, up, olr, ds, lwd]), h


# =========================================================================
# Radiation driver
# =========================================================================
class RadiationDriver:
    """Owns the RRTMGP object, CESM inputs, Mie tables, and the column layout.

    open_var : callable(str) -> xr.Dataset (coupling.py's opener, reused so
               file conventions live in one place)
    klevs    : native level indices of the coupled aerosol band
    """

    def __init__(self, open_var, klevs, lat, lon, dp_bin_m,
                 g_lw='rrtmgp-gas-lw-g128.nc', g_sw='rrtmgp-gas-sw-g112.nc',
                 mmid_kg=None, rho_dry=None, solar=None, solar_window_h=6.0,
                 solar_alpha=None, solar_nodes=None, solar_align=None):
        self.lat, self.lon = np.asarray(lat), np.asarray(lon)
        self.nlat, self.nlon = self.lat.size, self.lon.size
        # solar geometry (RAD_SOLAR above): solar_window_h is the interval one
        # call stands for -- coupling.py passes STEP_HOURS * RAD_EVERY
        self.solar_mode = (solar or RAD_SOLAR).lower()
        if self.solar_mode not in SOLAR_MODES:
            raise SystemExit(f"radiation: solar={self.solar_mode!r} is not valid: "
                             f"{' | '.join(SOLAR_MODES)}")
        self.solar_window_h = float(solar_window_h)
        # window_blend's weight (RAD_SOLAR_ALPHA); switchable between calls, like solar_mode
        self.alpha = float(RAD_SOLAR_ALPHA if solar_alpha is None else solar_alpha)
        if not 0.0 <= self.alpha <= 1.0:
            raise SystemExit(f"radiation: solar_alpha={self.alpha} is not in [0, 1]")
        # window_gq's node count (RAD_SOLAR_NODES); switchable between calls
        self.nodes = int(RAD_SOLAR_NODES if solar_nodes is None else solar_nodes)
        # window_gq's window (RAD_SOLAR_ALIGN): its centre relative to the call hour
        # [h], 0 = centred, -solar_window_h/2 = trailing (the window ends at the call)
        self.align = (solar_align or RAD_SOLAR_ALIGN).lower().replace('centered', 'centred')
        if self.align not in ('trailing', 'centred'):
            raise SystemExit(f"radiation: solar_align={self.align!r} is not trailing | centred")
        self.solar_offset_h = -self.solar_window_h / 2 if self.align == 'trailing' else 0.0
        W = f'{self.solar_window_h:g} h'
        print("  solar geometry: " + self.solar_mode + {
            'instant': " (the sun at the call hour, held for the interval)",
            'window': f" (sunlit-mean cos(zenith) over the {W} centred on each call, "
                      "TOA irradiance x sunlit fraction)",
            'window_ins': f" (energy-weighted cos(zenith) over the {W} centred on each call, "
                          "TOA irradiance x mu_bar/mu_ins) [experimental]",
            'window2': f" (two solves per call: the halves of the {W} window, each at its "
                       "sunlit-mean cos(zenith)) [experimental]",
            'window2_ins': f" (two solves per call: the halves of the {W} window, each at its "
                           "energy-weighted cos(zenith)) [experimental]",
            'window_blend': f" (cos(zenith) = {1 - self.alpha:g} x sunlit-mean + {self.alpha:g} x "
                            f"energy-weighted over the {W} centred on each call, TOA irradiance "
                            "x mu_bar/mu) [experimental]",
            'dither': f" (the sun at a golden-ratio offset within the {W} centred on each call, "
                      "plain TSI) [experimental]",
            'window_gq': f" ({self.nodes} solves per call at the Gauss nodes of the sunlit part of "
                         f"the {W} window "
                         + ("ending at" if self.align == 'trailing' else "centred on")
                         + " each call, insolation exact)",
        }[self.solar_mode], flush=True)

        # ---- CESM inputs ----
        self.dT = open_var('T'); self.dQ = open_var('Q')
        self.dO3 = open_var('O3'); self.dCH4 = open_var('CH4')
        self.dFLDS = open_var('FLDS'); self.dFLNS = open_var('FLNS')
        self.dFSDS = open_var('FSDS'); self.dFSNS = open_var('FSNS')
        self.time = self.dT['time'].values

        # ---- vertical layout: full native column within the LUT p-range ----
        ds = self.dT
        #pressure levels and layer interfaces
        plev = ds['hyam'].values * ds['P0'].values + ds['hybm'].values * 1.0e5
        pilev = ds['hyai'].values * ds['P0'].values + ds['hybi'].values * 1.0e5
        #find which native CESM levels are within the RRTMGP lookup-table range (>= 1.005 Pa)
        self.krad = np.where(plev >= 1.005)[0]           # native indices, top->bot
        self.p_lay = plev[self.krad]                     # (nzc,) Pa, top->bot
        self.dp_lay = np.diff(pilev)[self.krad]          # interface thickness, Pa
        self.nzc = self.krad.size
        #dictionary mapping native level index to its position in the band-level array (for slicing)
        kmap = {k: i for i, k in enumerate(self.krad)}
        #translates the coupled aerosol level indices into their positions within the radiation column subset**** (very important)
        self.band_pos = np.array([kmap[k] for k in klevs])   # band rows in column
        self.dp_band = self.dp_lay[self.band_pos]
        self.klevs = list(klevs)

        # ---- RRTMGP ----
        cfg = radiative_transfer.RadiativeTransfer(
            optics=radiative_transfer.OpticsParameters(
                optics=radiative_transfer.RRTMOptics(
                    longwave_nc_filepath=os.path.join(RRTMGP_DATA, g_lw),
                    shortwave_nc_filepath=os.path.join(RRTMGP_DATA, g_sw),
                    cloud_longwave_nc_filepath=os.path.join(RRTMGP_DATA, 'cloudysky_lw.nc'),
                    cloud_shortwave_nc_filepath=os.path.join(RRTMGP_DATA, 'cloudysky_sw.nc'),
                )),
            atmospheric_state_cfg=radiative_transfer.AtmosphericStateCfg(
                sfc_emis=SFC_EMIS, sfc_alb=ALB_FALLBACK,
                zenith=0.0, irrad=TSI_DEFAULT, toa_flux_lw=0.0,
                vmr_global_mean_filepath=os.path.join(
                    RRTMGP_TESTDATA, 'rcemip_global_mean_vmr.json'),
                vmr_sounding_filepath=os.path.join(
                    RRTMGP_TESTDATA, 'rcemip_vmr_sounding.csv'),
            ))
        # dz only feeds cloud paths; we are clear-sky, so any scalar works (for LWP/IWP calcs).
        self.rt = rrtmgp_lib.RRTMGP(cfg, dz=500.0)   # never patched: the chunk loop's
        self.cfg = cfg
        self.g_lw, self.g_sw = g_lw, g_sw
        # A second RRTMGP instance that ALSO returns the SW/LW split of the
        # heating rate, built lazily: RAD_MODE=flux, the reference-cache builder
        # and the solar-geometry check harnesses ask for it (heating(..., split=True)). Same
        # solve, same numbers -- the flag only adds output keys -- but it is a
        # separate jit, so the default path's compiled graph is untouched.
        self._rt_split = None
        self._jit_hr_split = None
        # rad_cache.RefCache of the archive-only fields (set by coupling.py when
        # RAD_REF_CACHE is on); None = always compute. See rad_cache.py.
        self.ref_cache = None
        self.nb_sw = self.rt.optics_lib.gas_optics_sw.n_bnd
        self.nb_lw = self.rt.optics_lib.gas_optics_lw.n_bnd

        # ---- Mie tables ----
        # WET optics (default): tables on the (wt%, bin, band) mesh, with both the
        # diameter and the refractive index taken at each solution strength. The
        # per-cell equilibrium wt% comes from Tabazadeh via _wt_field() below.
        # WET_OPTICS=0 restores the legacy dry-diameter/fixed-75wt% tables, which
        # underestimate 550 nm extinction by ~47% for this size distribution --
        # kept only for reproducing pre-2026-08-03 runs.
        self.wet_optics = WET_OPTICS and mmid_kg is not None
        if self.wet_optics:
            self.dRHq = None                       # RH derived from Q, see _wt_field
            wet_sw = build_wet_mie_tables(mmid_kg, rho_dry,
                                          os.path.join(RRTMGP_DATA, g_sw))
            wet_lw = build_wet_mie_tables(mmid_kg, rho_dry,
                                          os.path.join(RRTMGP_DATA, g_lw))
            self.dp_wet = wet_sw.pop('dp_wet')     # (n_wt, NBINS) [m]
            wet_lw.pop('dp_wet')
            self.mie_sw = {k: jnp.asarray(v) for k, v in wet_sw.items()}
            self.mie_lw = {k: jnp.asarray(v) for k, v in wet_lw.items()}
            # 550 nm cross-section per (wt%, bin), same wet diameters + RI
            self.sig_ext_550 = np.stack(
                [mie_at_wavelength(self.dp_wet[j], 0.550, wt_pct=w)[0]
                 for j, w in enumerate(WT_GRID)], axis=0)
            gf = self.dp_wet / np.asarray(dp_bin_m)[None, :]
            print(f"  optics: WET (Tabazadeh 1997 wt%, Tang 1997 density, "
                  f"P&W composition-dependent RI); {len(WT_GRID)} wt% nodes "
                  f"{WT_GRID[0]:.0f}-{WT_GRID[-1]:.0f}%, "
                  f"D_wet/D_dry {gf.min():.3f}-{gf.max():.3f}", flush=True)
        else:
            mie_sw = build_mie_tables(dp_bin_m, os.path.join(RRTMGP_DATA, g_sw))
            mie_lw = build_mie_tables(dp_bin_m, os.path.join(RRTMGP_DATA, g_lw))
            self.mie_sw = {k: jnp.asarray(v) for k, v in mie_sw.items()}
            self.mie_lw = {k: jnp.asarray(v) for k, v in mie_lw.items()}
            #compute 550 nm extinction cross-section per bin, scatter not needed for AOD
            self.sig_ext_550, _ = mie_at_wavelength(dp_bin_m, 0.550)
            print("  optics: DRY diameters, fixed 75 wt% RI (LEGACY -- "
                  "underestimates extinction ~47%; set WET_OPTICS=1)", flush=True)

        self._jit_hr = jax.jit(self._hr_impl)
        # ---- RAD_FAST: resolved here, per driver (see the RAD_FAST block) ----
        # RAD_FAST is validated (bit-identical to RAD_FAST=0, 2026-11-23) only
        # where its fast path engages: single GPU, wet optics, window_gq through
        # _solve_fast_gq. RAD_FAST=1 refuses the rest instead of running an
        # untested mix (the chunk loop under the cloud-skip patches, or
        # _solve_fast); auto turns itself off there.
        self.fast, self.fast_why = False, []
        if RAD_FAST:
            nd = len(jax.devices())
            self.fast_why = [w for c, w in [
                (nd != 1, f'{nd} GPUs (needs 1)'),
                (not self.wet_optics, 'dry optics (needs WET_OPTICS=1)'),
                (self.solar_mode != 'window_gq', f'RAD_SOLAR={self.solar_mode} (needs window_gq)'),
                (not (RAD_SW_SKIP_NIGHT and RAD_SKIP_CLOUD and RAD_GQ_FAST),
                 'RAD_SW_SKIP_NIGHT/RAD_SKIP_CLOUD/RAD_GQ_FAST must stay 1')] if c]
            if self.fast_why and RAD_FAST_MODE == 'on':
                raise SystemExit('radiation: RAD_FAST=1 is validated only for one GPU, wet optics and '
                                 'RAD_SOLAR=window_gq; here: ' + '; '.join(self.fast_why)
                                 + '. Use RAD_FAST=0 (or leave RAD_FAST unset: auto).')
            self.fast = not self.fast_why
        self._fallback_said = set()
        if self.fast:
            _install_fast_patches()
            # the fast path's own optics library: a shallow copy of self.rt's
            # (same tables, no reload) with the cloud branch dropped; self.rt
            # itself stays unpatched for the chunk loop
            self._fast_lib = copy.copy(self.rt.optics_lib)
            _skip_clouds(self._fast_lib)
            # (nz+2,) column pressures, for RAD_SPLIT_TROPO's static lower/upper split
            self._pcol = np.asarray(self._p_xyz(1, 1))[0, 0] if RAD_DYN_MINOR else None
            self._jit_lw = jax.jit(self._lw_impl)
            self._jit_sw = jax.jit(self._sw_impl)
        print("  " + rad_fast_summary(self.fast, self.fast_why), flush=True)
        # window_gq: one LW + N SW solves per chunk (see _hr_multi_impl); split=True
        # for the label writer / reference-cache builder
        self._jit_hr_multi = jax.jit(self._hr_multi_impl, static_argnames=('split',))
        self._gq_full = False      # tests only: True = window_gq by N full solves (the reference)
        self.nbins = int(np.asarray(self.mie_sw['sigma_ext']).shape[-2])
        self.nzb = len(self.klevs)

        # ---- multi-GPU: the lat-chunk loop in heating() shards across all
        # visible devices (rrtmgp columns are independent, so this is exact).
        # Pre-place the small per-bin constants on every device so each chunk's
        # optics run entirely on its own GPU; rrtmgp's own tables are placed by
        # JAX automatically on first use per device.
        self.devices = jax.devices()
        self.ndev = len(self.devices)
        self._mie_sw_dev = [jax.device_put(self.mie_sw, d) for d in self.devices]
        self._mie_lw_dev = [jax.device_put(self.mie_lw, d) for d in self.devices]
        self._dp_band_dev = [jax.device_put(jnp.asarray(self.dp_band), d)
                             for d in self.devices]
        self._sig550_dev = [jax.device_put(jnp.asarray(self.sig_ext_550), d)
                            for d in self.devices]

    # ------------------------------------------------------------------
    # per-hour CESM fields
    # ------------------------------------------------------------------
    def _read3d(self, ds, var, t):
        """(nzc, nlat, nlon) native full-column field at absolute hour t."""
        return ds[var].isel(time=t, lev=self.krad).values

    def solar(self, t):
        """(zenith [rad], TOA-irradiance scale) of the call at absolute hour t,
        each (nlat, nlon): the one geometry of a ONE-solve mode. The scale is
        None under RAD_SOLAR=instant (the TOA irradiance is then the plain TSI,
        exactly as before the flag); it is the sunlit fraction under window and
        mu_bar/mu_ins under window_ins and mu_bar/mu under window_blend. The
        two-solve modes have no single geometry: see solar_parts()."""
        if self.solar_mode == 'instant':
            return solar_zenith(self.time[t], self.lat, self.lon), None
        if self.solar_mode == 'dither':
            return solar_zenith(self.time[t], self.lat, self.lon,
                                offset_h=dither_offset(self.time[t], self.solar_window_h)), None
        if self.solar_mode in SOLAR_MULTI:
            raise SystemExit(f'radiation: RAD_SOLAR={self.solar_mode} makes several solves per call '
                             f'and has no single geometry (solar_parts); the caller does not support it')
        if self.solar_mode == 'window_blend':
            return solar_window(self.time[t], self.lat, self.lon, self.solar_window_h,
                                mu='blend', alpha=self.alpha)
        return solar_window(self.time[t], self.lat, self.lon, self.solar_window_h,
                            mu='energy' if self.solar_mode == 'window_ins' else 'sunlit')

    def solar_parts(self, t):
        """[(zenith, irradiance scale, albedo), ...] of the solves a call at
        absolute hour t makes under a two-solve mode: the window's two halves,
        centred -W/4 and +W/4 from the call hour, each with its own geometry and
        its own insolation-weighted surface albedo. heating() averages them."""
        w = self.solar_window_h
        if self.solar_mode == 'window_gq':
            # every node uses the window's insolation-weighted albedo (the
            # nodes' times differ per column, so no per-node hourly record)
            o = self.solar_offset_h
            alb = self._window_albedo(t, o - w / 2, o + w / 2)
            return [(zen, sc, alb) for zen, sc in
                    solar_gauss(self.time[t], self.lat, self.lon, w, self.nodes, offset_h=o)]
        mu = 'energy' if self.solar_mode == 'window2_ins' else 'sunlit'
        out = []
        for start, end in ((-w / 2, 0.0), (0.0, w / 2)):
            zen, sc = solar_window(self.time[t], self.lat, self.lon, w / 2,
                                   offset_h=0.5 * (start + end), mu=mu)
            out.append((zen, sc, self._window_albedo(t, start, end)))
        return out

    def _albedo(self, fsds, fsns):
        # SURFACE albedo = reflected/incoming = (flux down - flux net)/ flux down = (FSDS - FSNS)/FSDS = 1 - FSNS/FSDS
        return np.where(fsds > 5.0,
                        np.clip(1.0 - fsns / np.maximum(fsds, 1e-3), 0.02, 0.95),
                        ALB_FALLBACK)

    def _window_albedo(self, t, start_h, end_h):
        """Insolation-weighted surface albedo, 1 - sum(FSNS)/sum(FSDS), over the
        hourly records inside [t + start_h, t + end_h] (record k is the mean over
        [k-1 h, k], per time_bnds; for the 6-h window around t: t-2 .. t+3).
        One hour per read, not a time slice: the BC_SRC=clim archive
        (src/met/bc_clim.py ClimArchive.read) serves only scalar hours. Same
        values and reduction as a slice read of the CESM archive."""
        lo = max(0, t + int(np.floor(start_h)) + 1)
        hi = min(self.time.size - 1, t + int(np.ceil(end_h)))
        fsds = np.stack([self.dFSDS['FSDS'].isel(time=k).values
                         for k in range(lo, hi + 1)]).mean(axis=0)
        fsns = np.stack([self.dFSNS['FSNS'].isel(time=k).values
                         for k in range(lo, hi + 1)]).mean(axis=0)
        return self._albedo(fsds, fsns)

    """Surface fluxes"""
    def surface(self, t):
        flds = self.dFLDS['FLDS'].isel(time=t).values
        flns = self.dFLNS['FLNS'].isel(time=t).values
        # FLNS = up - down at surface  =>  sigma*Ts^4 = FLNS + FLDS (emis ~ 1)
        # getting the upward flux ... F = sigma*T^4 => T = (F/sigma)^(1/4)
        ts = ((np.maximum(flns + flds, 1.0)) / SIGMA_SB) ** 0.25
        if self.solar_mode == 'dither':
            # the albedo of the hour around the SAMPLED sun time, not of hour t
            d = dither_offset(self.time[t], self.solar_window_h)
            return ts, self._window_albedo(t, d - 0.5, d + 0.5)
        if self.solar_mode != 'instant':
            # The albedo of the WINDOW, not of hour t: a column dark at t but
            # sunlit later in the window would otherwise get ALB_FALLBACK, and a
            # low sun at t would set a grazing-angle albedo for the whole window.
            # (The two-solve modes override it per half, see solar_parts.)
            w = self.solar_window_h
            return ts, self._window_albedo(t, -w / 2, w / 2)
        fsds = self.dFSDS['FSDS'].isel(time=t).values
        fsns = self.dFSNS['FSNS'].isel(time=t).values
        return ts, self._albedo(fsds, fsns)

    # ------------------------------------------------------------------
    # column assembly: CESM (lev top->bot, lat, lon) -> rrtmgp (lat, lon, z-up)
    # with one halo cell at each z end
    # ------------------------------------------------------------------
    @staticmethod
    def _to_xyz(a, floor=None):
        """(nz, nlat, nlon) top->bottom  ->  (nlat, nlon, nz+2) bottom->top.

        Halos are linearly extrapolated; `floor` clamps them (linear
        extrapolation of small positive tracers can go negative, which the
        gas-optics lookups turn into NaN)."""
        #flip z-axis to bottom-up
        a = jnp.moveaxis(jnp.asarray(a), 0, -1)[..., ::-1]      # z now bottom-up
        #radiation needs a layer above and below the column; we linearly extrapolate the first and last layers to get these halo layers
        lo = 2.0 * a[..., :1] - a[..., 1:2]                     # linear halo
        hi = 2.0 * a[..., -1:] - a[..., -2:-1]
        #clamps the halo values to a minimum value (floor) to avoid negative values which can cause issues in the gas-optics lookups
        if floor is not None:
            lo = jnp.maximum(lo, floor); hi = jnp.maximum(hi, floor)
        return jnp.concatenate([lo, a, hi], axis=-1)

    def _band_to_xyz(self, prop):
        """(nb, nzb, nx, nlon) band-level aerosol -> (nb, nx, nlon, nz+2)."""
        nb, _, nx, nlon_ = prop.shape             # nlon_ = 1 for RAD_FAST's compacted SW columns
        full = jnp.zeros((nb, self.nzc, nx, nlon_), prop.dtype)
        full = full.at[:, self.band_pos].set(prop)
        full = jnp.moveaxis(full, 1, -1)[..., ::-1]
        #instead of linear extrapolation, we just pad with zeros for the halo layers
        #may need to think about whether zeroing the halo layers is physically reasonable (rather than fixing to boundary conditions), 
        #but for now it avoids NaN issues in the gas-optics lookups
        pad = jnp.zeros(full.shape[:-1] + (1,), prop.dtype)
        return jnp.concatenate([pad, full, pad], axis=-1)

    def _hr_impl(self, T, q, o3, ch4, ts, alb, zen, aer_sw, aer_lw, irr=None, split=False):
        """One rrtmgp call on already-chunked (x=lat rows) xyz fields.

        irr: per-column TOA irradiance (x, y) [W/m2] (RAD_SOLAR=window), or
        None for the scalar TSI_DEFAULT (RAD_SOLAR=instant, the original graph).
        split=True (RAD_MODE=flux, cache builder, check harnesses) runs the SAME solve on
        the split-output RRTMGP instance and returns (hr, diags, hr_sw)."""
        rt = self._rt_split if split else self.rt
        p = self._p_xyz_chunk(T.shape[0])
        rho = p / (rrtmgp_constants.R_D * T)
        zero = jnp.zeros_like(T)
        #constants
        co2 = jnp.full_like(T, CO2_PPM * 1e-6)
        n2o = jnp.full_like(T, N2O_PPB * 1e-9)
        #this is the core RRTMGP call
        #5 zero arrays for cloud fields and empty dict for cloud optics
        out = rt.compute_heating_rate(
            rho, q, zero, zero, zero, zero, zero, T, ts, p, {},
            zenith=zen, irrad=TSI_DEFAULT if irr is None else irr,
            sfc_alb=alb, sfc_emis=SFC_EMIS,
            vmr_fields={'o3': o3, 'ch4': ch4, 'co2': co2, 'n2o': n2o},
            aerosol_optics_sw={'optical_depth': aer_sw[0], 'ssa': aer_sw[1],
                               'asymmetry_factor': aer_sw[2]},
            aerosol_optics_lw={'optical_depth': aer_lw[0], 'ssa': aer_lw[1],
                               'asymmetry_factor': aer_lw[2]})
        #extracts the heating from from the output dictionary ()
        hr = out[rrtmgp_common.KEY_STORED_RADIATION]            # K/s, (x,y,nz)
        diags = jnp.stack([out['sw_flux_down_full'][..., -1],   # TOA SW down
                           out['sw_flux_up_full'][..., -1],     # TOA SW up
                           out['lw_flux_up_full'][..., -1],     # OLR
                           out['sw_flux_down_full'][..., 0],    # sfc SW down
                           out['lw_flux_down_full'][..., 0]])   # sfc LW down
        #return the heating rate and diagnosti fluxes 
        if split:
            return hr, diags, out['rad_heat_sw_3d']
        return hr, diags

    # ---- RAD_FAST: split LW / SW path (AIDE-SAI-link 9d693d2) ----------------------------
    # Same arithmetic as RRTMGP.compute_heating_rate (clear sky, no clouds), written out so the
    # SW solve can run on the sunlit columns only: night columns are zeroed by solve_sw anyway
    # (see its horizon handling) but still cost the full 112-g-point solve. Columns are independent,
    # so solving a compacted (n,1,nz) subset gives the same per-column bits.
    def _front(self, T, q, o3, ch4, p):
        from rrtmgp.optics import lookup_volume_mixing_ratio
        rt = self.rt
        zero = jnp.zeros_like(T)
        co2 = jnp.full_like(T, CO2_PPM * 1e-6)
        n2o = jnp.full_like(T, N2O_PPB * 1e-9)
        Tf = T.at[:, :, 0].set(2 * T[:, :, 1] - T[:, :, 2])
        Tf = Tf.at[:, :, -1].set(2 * T[:, :, -2] - T[:, :, -3])
        q_t = jnp.clip(q, 0.0, None)
        vmr = lookup_volume_mixing_ratio.reconstruct_vmr_fields_from_pressure(rt.atmospheric_state.vmr, p)
        vmr.update({'o3': o3, 'ch4': ch4, 'co2': co2, 'n2o': n2o})
        vmr['h2o'] = rrtmgp_lib._humidity_to_volume_mixing_ratio(q_t, zero)
        mol = rrtmgp_lib._air_molecules_per_area(p, vmr['h2o'])
        return Tf, vmr, mol

    def _lw_impl(self, T, q, o3, ch4, ts, aer_lw, p):
        from rrtmgp.rte import two_stream
        rt, lib = self.rt, self._fast_lib
        Tf, vmr, mol = self._front(T, q, o3, ch4, p)
        state = dataclasses.replace(rt.atmospheric_state, sfc_emis=SFC_EMIS)
        aer = {'optical_depth': aer_lw[0], 'ssa': aer_lw[1], 'asymmetry_factor': aer_lw[2]}
        if RAD_GBATCH > 1:
            lw = rrtmgp_batched.solve_lw(p, Tf, mol, lib, state, vmr, ts, aer, RAD_GBATCH, RAD_DYN_MINOR, self._pcol)
        else:
            lw = two_stream.solve_lw(p, Tf, mol, lib, state, vmr, ts, aerosol_optics=aer)
        hr = two_stream.compute_heating_rate(lw['flux_net'], p)
        return hr, lw['flux_up'][..., -1], lw['flux_down'][..., 1]

    def _sw_impl(self, T, q, o3, ch4, alb, zen, aer_sw, p, irr):
        from rrtmgp.rte import two_stream
        rt, lib = self.rt, self._fast_lib
        Tf, vmr, mol = self._front(T, q, o3, ch4, p)
        state = dataclasses.replace(rt.atmospheric_state, zenith=zen, irrad=irr, sfc_alb=alb, sfc_emis=SFC_EMIS)
        aer = {'optical_depth': aer_sw[0], 'ssa': aer_sw[1], 'asymmetry_factor': aer_sw[2]}
        if RAD_GBATCH > 1:
            sw = rrtmgp_batched.solve_sw(p, Tf, mol, lib, state, vmr, aer, RAD_GBATCH, RAD_DYN_MINOR, self._pcol)
        else:
            sw = two_stream.solve_sw(p, Tf, mol, lib, state, vmr, aerosol_optics=aer)
        hr = two_stream.compute_heating_rate(sw['flux_net'], p)
        return hr, sw['flux_down'][..., -1], sw['flux_up'][..., -1], sw['flux_down'][..., 1]

    def _hr_multi_impl(self, T, q, o3, ch4, ts, albs, zens, aer_sw, aer_lw, irrs, split=False):
        """RAD_SOLAR=window_gq on one chunk: ONE longwave solve and len(zens)
        shortwave solves (zens (N, x, y, 1), irrs and albs (N, x, y)), returning
        (hr, diags[, hr_sw]) with the shortwave parts AVERAGED -- their irradiances
        carry the quadrature weight x node count (solar_gauss). Equal, to
        roundoff, to the mean of N _hr_impl calls, since the longwave does not
        see the sun: this is RRTMGP.compute_heating_rate's clear-sky path written
        out so the longwave is solved once (scripts/validation/check_solar_gq_sw.py
        compares the two on the GPU). Kept in step with jax-rrtmgp by that check."""
        import dataclasses
        from rrtmgp.rte import two_stream
        from rrtmgp.optics import lookup_volume_mixing_ratio
        rt = self.rt
        p = self._p_xyz_chunk(T.shape[0])
        rho = p / (rrtmgp_constants.R_D * T)
        zero = jnp.zeros_like(T)
        co2 = jnp.full_like(T, CO2_PPM * 1e-6)
        n2o = jnp.full_like(T, N2O_PPB * 1e-9)
        # -- as compute_heating_rate: halo-filled temperature, clipped humidity,
        # merged vmr with h2o from q, molecules per area, (zero) cloud paths
        Tf = T.at[:, :, 0].set(2 * T[:, :, 1] - T[:, :, 2])
        Tf = Tf.at[:, :, -1].set(2 * T[:, :, -2] - T[:, :, -3])
        q_t = jnp.clip(q, 0.0, None)
        vmr = lookup_volume_mixing_ratio.reconstruct_vmr_fields_from_pressure(
            rt.atmospheric_state.vmr, p)
        vmr.update({'o3': o3, 'ch4': ch4, 'co2': co2, 'n2o': n2o})
        vmr['h2o'] = rrtmgp_lib._humidity_to_volume_mixing_ratio(q_t, zero)
        mol = rrtmgp_lib._air_molecules_per_area(p, vmr['h2o'])
        cpath = rho * zero * 500.0
        clouds = dict(cloud_r_eff_liq=zero, cloud_path_liq=cpath, cloud_r_eff_ice=zero,
                      cloud_path_ice=cpath, use_scan=False,
                      cloud_path_liq_per_gpt=None, cloud_path_ice_per_gpt=None)
        aer = lambda a: {'optical_depth': a[0], 'ssa': a[1], 'asymmetry_factor': a[2]}
        state = lambda k: dataclasses.replace(rt.atmospheric_state, zenith=zens[k], irrad=irrs[k],
                                              sfc_alb=albs[k], sfc_emis=SFC_EMIS)
        lw = two_stream.solve_lw(p, Tf, mol, rt.optics_lib, state(0), vmr, ts,
                                 aerosol_optics=aer(aer_lw), **clouds)
        hr_lw = two_stream.compute_heating_rate(lw['flux_net'], p)
        n = zens.shape[0]
        hr_sw = sw_dn = sw_up = 0.0
        for k in range(n):
            sw = two_stream.solve_sw(p, Tf, mol, rt.optics_lib, state(k), vmr,
                                     aerosol_optics=aer(aer_sw), **clouds)
            hr_sw = hr_sw + two_stream.compute_heating_rate(sw['flux_net'], p)
            sw_dn = sw_dn + sw['flux_down']; sw_up = sw_up + sw['flux_up']
        hr_sw, sw_dn, sw_up = hr_sw / n, sw_dn / n, sw_up / n
        hr = hr_lw + hr_sw
        # compute_heating_rate's *_full profiles drop the surface halo (index 0)
        diags = jnp.stack([sw_dn[..., -1], sw_up[..., -1], lw['flux_up'][..., -1],
                           sw_dn[..., 1], lw['flux_down'][..., 1]])
        if split:
            return hr, diags, hr_sw
        return hr, diags

    def _wt_field(self, T_band, q_band):
        """Equilibrium H2SO4 weight percent on the band levels, (nzb,nlat,nlon).

        RH is derived from the CONSERVED water vapour mixing ratio q and the
        temperature actually being evaluated, NOT from CESM's RELHUM field. That
        matters in anomaly mode: the perturbed call runs at T_eff = T_CESM + dT,
        and CESM's RELHUM was computed at T_CESM, so reusing it would hold RH
        fixed while the temperature moved -- physically the vapour is what is
        conserved and RH must fall as the layer warms. Doing it this way makes
        aerosol warming shrink the droplet (warmer -> lower RH -> more
        concentrated -> smaller -> less extinction), a real negative feedback
        that the fixed-RH version would have suppressed.

        Uses the same saturation-vapour-pressure form as the Tabazadeh fit, so
        the RH we form here and the P_sat it divides back out cancel exactly.
        """
        import settling                                      # src/settling/, via _paths
        from settling import h2so4_equilibrium_wt            # tomas-jax dev's Tabazadeh, vmapped
        p = self.p_lay[self.band_pos][:, None, None]         # Pa
        # RH over liquid from q (settling.rh_liquid_pct, the model's one RH
        # since 2026-11-17; the arithmetic that used to be inlined here)
        Tb = np.maximum(np.asarray(T_band), 150.0)
        rh_pct = settling.rh_liquid_pct(Tb, q_band, p)
        return np.asarray(h2so4_equilibrium_wt(jnp.asarray(Tb),
                                               jnp.asarray(rh_pct)))

    def _p_xyz_chunk(self, nx):
        p = np.broadcast_to(self.p_lay[:, None, None],
                            (self.nzc, nx, self.nlon))
        px = self._to_xyz(p)
        # keep halo pressures inside the gas-optics lookup range
        return jnp.clip(px, 1.006, 1.09e5)

    def _p_xyz(self, nx, ny):
        p = np.broadcast_to(self.p_lay[:, None, None], (self.nzc, nx, ny))
        return jnp.clip(self._to_xyz(p), 1.006, 1.09e5)

    _SW_SIZES = (16384, 8192, 4096, 2048, 1024)

    def _solve_fast(self, Tx, qx, o3x, ch4x, tsj, albj, zen, irrj, num_j, wwt_all, _mk):
        """One-solve (instant/window) path: LW per lat chunk exactly as before, SW on the sunlit
        columns only (see _sw_impl). Returns (hr (nlat,nlon,nz+2), diags (5,nlat,nlon), aod550, hr_sw)."""
        hr_lw, olr, lwd, aod550, ctx = self._fast_lw(Tx, qx, o3x, ch4x, tsj, num_j, wwt_all, _mk)
        hsw, dn_t, up_t, dn_s = self._fast_sw(ctx, albj, zen, irrj, _mk)
        diags = jnp.stack([dn_t, up_t, olr, dn_s, lwd])
        hr = hr_lw + hsw
        return hr, diags, aod550, hsw

    def _solve_fast_gq(self, Tx, qx, o3x, ch4x, tsj, albs, zens, irrs, num_j, wwt_all, _mk):
        """RAD_SOLAR=window_gq on the fast path (2026-11-17, rgq): ONE LW solve and, per Gauss node,
        one SW solve on that node's sunlit columns only. Accumulates exactly as _hr_multi_impl:
        (0.0 + s_0 + ... + s_{N-1}) / N for the SW heating and fluxes, then hr_lw + hr_sw.
        albs/zens/irrs carry a leading node axis."""
        hr_lw, olr, lwd, aod550, ctx = self._fast_lw(Tx, qx, o3x, ch4x, tsj, num_j, wwt_all, _mk)
        parts = [self._fast_sw(ctx, albs[k], zens[k], irrs[k], _mk) for k in range(zens.shape[0])]
        # combine under jit, as _hr_multi_impl does (XLA folds the leading 0.0 + and fuses the / n;
        # done eagerly the result differs by 1 ulp and the sign of zero)
        hr, diags, hsw = _gq_combine(hr_lw, olr, lwd, parts)
        return hr, diags, aod550, hsw

    def _fast_lw(self, Tx, qx, o3x, ch4x, tsj, num_j, wwt_all, _mk):
        """Aerosol optics + LW per lat chunk; returns (hr_lw, olr, lwd, aod550, ctx for _fast_sw)."""
        nlat, nlon = self.nlat, self.nlon
        ncol = nlat * nlon
        C = RAD_LAT_CHUNK_FAST
        nchunk = int(np.ceil(nlat / C)); C = int(np.ceil(nlat / nchunk))
        dp_d, mie_sw, mie_lw, sig = (self._dp_band_dev[0], self._mie_sw_dev[0], self._mie_lw_dev[0],
                                     self._sig550_dev[0])
        lw_hr, olr, lwd, aod_rows, sw_raw = [], [], [], [], []
        for a in range(0, nlat, C):
            b = min(a + C, nlat)
            nc = num_j[:, :, a:b]; ww = wwt_all[:, :, a:b]
            tau_s, ssa_s, g_s = _aerosol_props_wet(nc, dp_d, mie_sw['sigma_ext'], mie_sw['ssa'], mie_sw['g'], ww)
            tau_l, ssa_l, g_l = _aerosol_props_wet(nc, dp_d, mie_lw['sigma_ext'], mie_lw['ssa'], mie_lw['g'], ww)
            aer_lw = tuple(self._band_to_xyz(v) for v in (tau_l, ssa_l, g_l))
            _mk('optics', *aer_lw, tau_s, ssa_s, g_s)
            h, o, d = self._jit_lw(Tx[a:b], qx[a:b], o3x[a:b], ch4x[a:b], tsj[a:b], aer_lw,
                                   self._p_xyz_chunk(b - a))
            _mk('lw_solve', h, o, d)
            lw_hr.append(h); olr.append(o); lwd.append(d)
            sw_raw.append((tau_s, ssa_s, g_s))
            aod_rows.append(_aod550_wet(nc, dp_d, sig, ww))
            _mk('aod', aod_rows[-1])
        cat = lambda xs, ax=0: xs[0] if len(xs) == 1 else jnp.concatenate(xs, axis=ax)
        hr_lw = cat(lw_hr); olr = cat(olr); lwd = cat(lwd); aod550 = cat(aod_rows)
        sw_props = tuple(cat([c[i] for c in sw_raw], 2).reshape(c_shape(sw_raw[0][i], ncol))
                         for i in range(3))
        nz2 = self.nzc + 2
        flat = tuple(x.reshape(ncol, nz2) for x in (Tx, qx, o3x, ch4x))
        return hr_lw, olr, lwd, aod550, (sw_props, flat, Tx.dtype)

    def _fast_sw(self, ctx, albj, zen, irrj, _mk):
        """SW on the sunlit columns of one geometry. Returns (hsw (nlat,nlon,nz+2), sw_dn_toa,
        sw_up_toa, sw_dn_sfc), each (nlat, nlon); night columns get the solver's own zeros."""
        sw_props, (Tf_, qf_, o3f_, ch4f_), dt = ctx
        nlat, nlon = self.nlat, self.nlon
        ncol = nlat * nlon
        # sunlit columns: the solver's own test is zenith >= pi/2 -> zeroed
        day = ~(np.asarray(zen).reshape(-1) >= 0.5 * np.pi)
        idx = np.flatnonzero(day); nd = idx.size
        nz2 = self.nzc + 2
        # night: solve_sw's zeroed flux -> heating rate is -0.0 (G*0/dp/cp, dp<0); keep the sign bit
        hsw = jnp.full((ncol, nz2), -0.0); dn_t = jnp.zeros(ncol); up_t = jnp.zeros(ncol); dn_s = jnp.zeros(ncol)
        hsw = hsw.astype(dt); dn_t = dn_t.astype(dt); up_t = up_t.astype(dt); dn_s = dn_s.astype(dt)
        pos = 0
        alb_f = albj.reshape(ncol); zen_f = jnp.asarray(zen).reshape(ncol)
        irr_f = None if irrj is None else irrj.reshape(ncol)
        while pos < nd:
            # ONE solve over all sunlit columns, padded to a multiple of RAD_SW_PAD (a handful of
            # distinct shapes over a year -> a handful of one-off compiles); each extra call costs
            # ~0.1 s of pure launch overhead, so no chunking
            nreal = min(nd - pos, RAD_SW_MAXCOLS)
            n = -(-nreal // RAD_SW_PAD) * RAD_SW_PAD
            ic = idx[pos:pos + nreal]
            if nreal < n:
                ic = np.concatenate([ic, np.full(n - nreal, idx[0], ic.dtype)])
            ji = jnp.asarray(ic)
            g = lambda f: f[ji][:, None, :]
            aer = tuple(self._band_to_xyz(p_[:, :, ji][..., None]) for p_ in sw_props)
            irr = TSI_DEFAULT if irr_f is None else irr_f[ji][:, None]
            h, dt_, ut_, ds_ = self._jit_sw(g(Tf_), g(qf_), g(o3f_), g(ch4f_), alb_f[ji][:, None],
                                            zen_f[ji][:, None, None], aer, self._p_xyz(n, 1), irr)
            _mk('sw_solve', h, dt_, ut_, ds_)
            ri = jnp.asarray(idx[pos:pos + nreal])
            hsw = hsw.at[ri].set(h[:nreal, 0]); dn_t = dn_t.at[ri].set(dt_[:nreal, 0])
            up_t = up_t.at[ri].set(ut_[:nreal, 0]); dn_s = dn_s.at[ri].set(ds_[:nreal, 0])
            pos += nreal
        return (hsw.reshape(nlat, nlon, nz2), dn_t.reshape(nlat, nlon), up_t.reshape(nlat, nlon),
                dn_s.reshape(nlat, nlon))

    # ------------------------------------------------------------------
    def _ensure_split(self):
        if self._rt_split is None:
            import dataclasses
            cfg = dataclasses.replace(self.cfg, save_lw_sw_heating_rates=True)
            self._rt_split = rrtmgp_lib.RRTMGP(cfg, dz=500.0)
            self._jit_hr_split = jax.jit(functools.partial(self._hr_impl, split=True))

    def heating(self, t, num_band, T_band_override=None, split=False, T_optics_band=None):
        """Heating rate for absolute hour index t with aerosol num_band.

        num_band : (NBINS, nzb, nlat, nlon) [#/kg] on the coupled band levels
        T_band_override : optional (nzb, nlat, nlon) replacing CESM T on the
                          band levels (T_eff = T_CESM + dT for the perturbed call)
        Returns hr_band (nzb, nlat, nlon) [K/s] in CESM level order, and a
        diag dict (TOA/sfc fluxes, band AOD at 550 nm).
        T_optics_band : optional (nzb, nlat, nlon), the band temperature of the
                          aerosol OPTICS only (the wet wt%, size and refractive
                          index); None = the solve's own band T (the override if
                          given). 2026-11-17: RAD_T_FEEDBACK=0 passes T_met here
                          and T_met + dT_rad as the override, so dT_rad sets the
                          emission temperature but not the droplets.
        num_band=None is the zero-aerosol (clear-sky) call. split=True adds
        diag['hr_sw_band'], the shortwave part of hr_band (same solve).
        Under a two-solve RAD_SOLAR mode both returns are the mean of the two
        half-window solves (each stands for half the interval).
        """
        if self.solar_mode not in SOLAR_MULTI:
            return self._heating(t, num_band, T_band_override, split,
                                 T_optics_band=T_optics_band)
        if self.solar_mode == 'window_gq' and not self._gq_full:
            # one LW + N SW solves (the longwave does not see the sun)
            return self._heating(t, num_band, T_band_override, split, geoms=self.solar_parts(t),
                                 T_optics_band=T_optics_band)
        outs = [self._heating(t, num_band, T_band_override, split, geom=g,
                              T_optics_band=T_optics_band)
                for g in self.solar_parts(t)]
        # plain mean: window2's halves each stand for half the interval, and
        # window_gq's scales already carry quadrature weight x node count
        # (SW is linear in the irradiance; the LW parts are identical)
        n = len(outs)
        hr = sum(o[0] for o in outs) / n
        diag = {k: sum(o[1][k] for o in outs) / n for k in outs[0][1]}
        return hr, diag

    def _heating(self, t, num_band, T_band_override=None, split=False, geom=None, geoms=None,
                 T_optics_band=None):
        """One solve: heating() for the one-solve modes, or one part of a
        two-solve call with geom = (zenith, irradiance scale, albedo); or, with
        geoms = [geom, ...] (window_gq), one LW and len(geoms) SW solves."""
        if num_band is None:
            num_band = jnp.zeros((self.nbins, self.nzb, self.nlat, self.nlon))
        use_fast = self.fast and not (split or geoms is None)
        if self.fast and not use_fast:
            if RAD_FAST_MODE == 'on':
                raise SystemExit('radiation: RAD_FAST=1 supports only the window_gq anomaly solve '
                                 '(no split=True: RAD_MODE=flux, the cache builder, the check '
                                 'harnesses; no RAD_SOLAR full-solve test mode). Use RAD_FAST=0 '
                                 '(or leave RAD_FAST unset: auto).')
            # auto: this call runs the chunk loop below, i.e. the RAD_FAST=0 path
            # (self.rt unpatched, RAD_LAT_CHUNK); say so once per kind of call
            why = 'split=True' if split else f'one-solve call (RAD_SOLAR={self.solar_mode})'
            if why not in self._fallback_said:
                self._fallback_said.add(why)
                print(f"  RAD_FAST=auto: {why} -> chunk loop (the RAD_FAST=0 path)", flush=True)
        if split:
            self._ensure_split()
        T = self._read3d(self.dT, 'T', t).astype(np.float64)
        #replaces the temperature on the coupled band levels to emulate the aerosol direct effect on heating rate (T_eff = T_CESM + dT)
        if T_band_override is not None:
            T[self.band_pos] = np.asarray(T_band_override)
        q = self._read3d(self.dQ, 'Q', t)
        o3 = self._read3d(self.dO3, 'O3', t)
        ch4 = self._read3d(self.dCH4, 'CH4', t)
        ts, alb = self.surface(t)
        if geoms is not None:
            zen = np.stack([g[0] for g in geoms]); fsun = np.stack([g[1] for g in geoms])
            alb = np.stack([g[2] for g in geoms])
        elif geom is None:
            zen, fsun = self.solar(t)
        else:
            zen, fsun, alb = geom
        # RAD_SOLAR=window: per-column TOA irradiance TSI * sunlit fraction
        # (SW is linear in it); instant: None -> the scalar TSI, original graph
        irrj = None if fsun is None else jnp.asarray(TSI_DEFAULT * fsun)
        #converts 3D fields to bottom-up xyz coordinates with halos
        Tx = self._to_xyz(T)
        qx = self._to_xyz(np.maximum(q, 1e-12), floor=1e-12)
        o3x = self._to_xyz(np.maximum(o3, 0.0), floor=0.0)
        ch4x = self._to_xyz(np.maximum(ch4, 0.0), floor=0.0)
        #converts fields to jax arrays
        tsj = jnp.asarray(ts); albj = jnp.asarray(alb)
        zenj = jnp.asarray(zen)[..., None]                      # ([N,] nlat,nlon,1)

        # aerosol optics are computed PER lat-chunk: the full-globe per-band
        # fields are ~3 GB and were the OOM driver on memory-tight GPUs.
        hr_rows, dg_rows, aod_rows, sw_rows = [], [], [], []
        C = RAD_LAT_CHUNK
        num_j = jnp.asarray(num_band)
        # per-cell composition -> interpolation weights over WT_GRID. Built from
        # the SAME T array the radiation sees (incl. any T_band_override), so the
        # droplet composition tracks the perturbed temperature -- unless
        # T_optics_band is given (RAD_T_FEEDBACK=0: the droplets are the
        # microphysics' and settling's, at T_met; see heating()).
        if self.wet_optics:
            T_opt = T[self.band_pos] if T_optics_band is None else np.asarray(T_optics_band, np.float64)
            wwt_all = wt_weights(self._wt_field(T_opt, q[self.band_pos]))
        fast = (use_fast and RAD_SW_SKIP_NIGHT and RAD_SKIP_CLOUD and self.ndev == 1
                and self.wet_optics and (geoms is None or RAD_GQ_FAST))
        _mk = _no_mark
        if fast and geoms is not None:
            hr, diags, aod550, hsw_full = self._solve_fast_gq(
                Tx, qx, o3x, ch4x, tsj, albj, zen, irrj, num_j, wwt_all, _mk)
            sw_rows = [hsw_full]; d0 = self.devices[0]
        elif fast:
            hr, diags, aod550, hsw_full = self._solve_fast(
                Tx, qx, o3x, ch4x, tsj, albj, zen, irrj, num_j, wwt_all, _mk)
            sw_rows = [hsw_full]; d0 = self.devices[0]
        else:
            # MULTI-GPU: keep each chunk <= RAD_LAT_CHUNK rows (memory bound) but make
            # the chunk count a multiple of ndev and send chunk i to device i%ndev, so
            # the chunks run concurrently across GPUs. Columns are independent in
            # rrtmgp, so the result is bit-identical to the serial single-GPU loop.
            nchunk = self.ndev * int(np.ceil(self.nlat / (self.ndev * C)))
            C = int(np.ceil(self.nlat / nchunk))
            for i, a in enumerate(range(0, self.nlat, C)):
                b = min(a + C, self.nlat)
                k = i % self.ndev                       # target device for this chunk
                put = lambda x: jax.device_put(x, self.devices[k])
                dp_d = self._dp_band_dev[k]
                nc = put(num_j[:, :, a:b])
                #compuets aerosol optical properties (tau, ssa, g) for both SW and LW bands
                #using the precomputed Mie tables and the number mixing ratios for each bin
                if self.wet_optics:
                    ww = put(wwt_all[:, :, a:b])
                    tau_s, ssa_s, g_s = _aerosol_props_wet(
                        nc, dp_d, self._mie_sw_dev[k]['sigma_ext'],
                        self._mie_sw_dev[k]['ssa'], self._mie_sw_dev[k]['g'], ww)
                    tau_l, ssa_l, g_l = _aerosol_props_wet(
                        nc, dp_d, self._mie_lw_dev[k]['sigma_ext'],
                        self._mie_lw_dev[k]['ssa'], self._mie_lw_dev[k]['g'], ww)
                else:
                    tau_s, ssa_s, g_s = _aerosol_props(nc, dp_d,
                                                   self._mie_sw_dev[k]['sigma_ext'],
                                                   self._mie_sw_dev[k]['ssa'], self._mie_sw_dev[k]['g'])
                    tau_l, ssa_l, g_l = _aerosol_props(nc, dp_d,
                                                   self._mie_lw_dev[k]['sigma_ext'],
                                                   self._mie_lw_dev[k]['ssa'], self._mie_lw_dev[k]['g'])
                #converts the aerosol optical properties to bottom-up xyz coordinates with zero halos
                aer_sw = tuple(self._band_to_xyz(v) for v in (tau_s, ssa_s, g_s))
                aer_lw = tuple(self._band_to_xyz(v) for v in (tau_l, ssa_l, g_l))
                if geoms is not None:                   # window_gq: (N, rows, ...) geometry
                    _m = self._jit_hr_multi(put(Tx[a:b]), put(qx[a:b]), put(o3x[a:b]), put(ch4x[a:b]),
                                            put(tsj[a:b]), put(albj[:, a:b]), put(zenj[:, a:b]),
                                            aer_sw, aer_lw, put(irrj[:, a:b]), split=split)
                    hr, dg = _m[0], _m[1]
                    if split:
                        sw_rows.append(_m[2])
                _args = (put(Tx[a:b]), put(qx[a:b]), put(o3x[a:b]), put(ch4x[a:b]),
                         put(tsj[a:b]), put(albj[a:b]), put(zenj[a:b]), aer_sw, aer_lw)
                # irr only when RAD_SOLAR=window, so the instant path calls the
                # jitted function exactly as before the flag existed
                _kw = {} if irrj is None else {'irr': put(irrj[a:b])}
                if geoms is not None:
                    pass
                elif split:
                    hr, dg, hsw = self._jit_hr_split(*_args, **_kw)
                    sw_rows.append(hsw)
                else:
                    hr, dg = self._jit_hr(*_args, **_kw)
                #550 nm AOD
                if self.wet_optics:
                    aod_rows.append(_aod550_wet(nc, dp_d, self._sig550_dev[k], ww))
                else:
                    aod_rows.append(_aod550(nc, dp_d, self._sig550_dev[k]))
                hr_rows.append(hr); dg_rows.append(dg)
            # gather every chunk back to the primary device before concatenating
            # (no-op when ndev==1, so single-GPU behaviour is unchanged)
            d0 = self.devices[0]
            hr = jnp.concatenate([jax.device_put(x, d0) for x in hr_rows], axis=0)     # (nlat,nlon,nz)
            diags = jnp.concatenate([jax.device_put(x, d0) for x in dg_rows], axis=1)  # (5,nlat,nlon)
            aod550 = jnp.concatenate([jax.device_put(x, d0) for x in aod_rows], axis=0)# (nlat,nlon)

        # back to CESM order, band levels only (strip halos, flip z)
        hr_cesm = jnp.moveaxis(hr[..., 1:-1][..., ::-1], -1, 0)  # (nzc,nlat,nlon)
        #extracts the heating rate for the coupled band levels only, in CESM level order
        hr_band = hr_cesm[self.band_pos]

        dnames = ('sw_dn_toa', 'sw_up_toa', 'olr', 'sw_dn_sfc', 'lw_dn_sfc')
        #creates a dictionary mapping diagnostic names to their corresponding arrays
        diag = {n: diags[i] for i, n in enumerate(dnames)}
        diag['aod550'] = aod550
        if split:
            hsw = jnp.concatenate([jax.device_put(x, d0) for x in sw_rows], axis=0)
            diag['hr_sw_band'] = jnp.moveaxis(hsw[..., 1:-1][..., ::-1], -1, 0)[self.band_pos]
        #returns the heating rate for the coupled band levels and a dictionary of diagnostic fluxes and AOD
        return hr_band, diag

    def heating_anomaly(self, t, num_evolved, num_ref, dT_band, T_band=None,
                        optics_at_met=False):
        """(HR(evolved bins, T_CESM+dT) - HR(reference bins, T_CESM)) [K/s].

        The perturbed call sees T_eff so LW cooling responds to accumulated
        warming (negative Planck feedback bounds dT); the reference call is
        pure CESM. Returns (hr_anom, diag_perturbed, diag_reference).
        T_band (nzb, nlat, nlon), optional: the band temperature of BOTH calls
        in place of the archive's (coupling.py MET_SRC=file); the reference
        call then depends on it and bypasses the per-hour cache.
        optics_at_met (2026-11-17, coupling.py RAD_T_FEEDBACK=0): the perturbed
        call's aerosol optics at the unwarmed band T (T_CESM or T_band), its
        emission at T_eff. False = the optics at T_eff too (bit-identical to
        before the flag).
        """
        #CESM temp on just aerosol levels
        T_cesm_band = (self._read3d(self.dT, 'T', t)[self.band_pos] if T_band is None
                       else np.asarray(T_band, np.float64))
        #evolved tomas bins and temperature 
        #dt_rad contains the accumulated warming from all previous steps
        hr1, d1 = self.heating(t, num_evolved,
                               T_band_override=T_cesm_band + np.asarray(dT_band),
                               T_optics_band=T_cesm_band if optics_at_met else None)
        #background reference bins and pure CESM temperature (cached when possible)
        hr0, d0 = self.reference(t, num_ref, T_band=T_band)
        #heating rate anomoly
        return hr1 - hr0, d1, d0

    # ------------------------------------------------------------------
    # the archive-only reference call and its cache (rad_cache.py)
    # ------------------------------------------------------------------
    def ref_ready(self, t):
        """True when hour t's reference solve can be served without num_ref
        (so the caller may skip re-binning the reference aerosol)."""
        return self.ref_cache is not None and self.ref_cache.has(t)

    def reference(self, t, num_ref, T_band=None):
        """(hr0, d0) = heating(t, num_ref) -- served from the cache when the
        hour is stored (float64 on disk, so bit-identical to computing it),
        else computed and, if a cache is attached, stored. num_ref may be
        None only on a hit. With T_band (a band temperature that is not the
        archive's) the solve is computed with it and the cache is not touched."""
        import rad_cache
        if T_band is not None:
            if num_ref is None:
                raise RuntimeError('radiation: a reference solve at a non-archive '
                                   'temperature needs the reference aerosol')
            return self.heating(t, num_ref, T_band_override=np.asarray(T_band, np.float64))
        c = self.ref_cache
        if c is not None:
            e = c.get(t)
            if e is not None:
                d0 = {n: jnp.asarray(v) for n, v in
                      rad_cache.arrays_to_diag('d0', e).items()}
                return jnp.asarray(e['hr0']), d0
        if num_ref is None:
            raise RuntimeError(f'radiation: hour {t} is not in the reference '
                               f'cache and no reference aerosol was given')
        hr0, d0 = self.heating(t, num_ref)
        if c is not None:
            c.put(t, hr0=np.asarray(hr0, np.float64),
                  **rad_cache.diag_to_arrays('d0', d0, np.float64))
        return hr0, d0

    def cache_key(self, **ref_src):
        """Everything that determines heating(t, num_ref(t)): the archive, the
        column, the optics tables and flags, the gases, the reference-aerosol
        source (passed in by the caller) and the solver commit."""
        import rad_cache
        with open(RI_FILE, 'rb') as f:
            ri_hash = hashlib.sha1(f.read()).hexdigest()[:12]
        key = dict(
            archive=os.path.basename(str(self.dT.encoding.get('source', ''))),
            klevs=[int(k) for k in self.klevs], krad=[int(self.krad[0]), int(self.krad[-1])],
            nlat=int(self.nlat), nlon=int(self.nlon),
            wet_optics=bool(self.wet_optics), wt_grid=[float(w) for w in WT_GRID],
            n_quad_wl=N_QUAD_WL, ri_file=ri_hash, nbins=int(self.nbins),
            dp_wet_hash=hashlib.sha1(np.ascontiguousarray(
                np.asarray(self.dp_wet if self.wet_optics else self.mie_sw['sigma_ext']),
                dtype=np.float64).tobytes()).hexdigest()[:12],
            g_lw=self.g_lw, g_sw=self.g_sw, co2_ppm=CO2_PPM, n2o_ppb=N2O_PPB,
            sfc_emis=SFC_EMIS, alb_fallback=ALB_FALLBACK, tsi=TSI_DEFAULT,
            rrtmgp_commit=rad_cache._submodule_commit(
                _paths.REPO_ROOT, 'jax-rrtmgp', os.environ.get('RRTMGP_PATH') or None),
            ref_src=ref_src)
        # The solar geometry determines every cached field (reference solve and
        # clear-sky fields alike). Added ONLY when it is not the original
        # 'instant', so the hash -- and every existing cache directory -- of an
        # instant run is unchanged; a window run gets its own directory per width.
        if self.solar_mode != 'instant':
            key['solar'] = dict(mode=self.solar_mode, window_h=self.solar_window_h)
            if self.solar_mode == 'window_blend':      # only here, so the other modes' keys are unchanged
                key['solar']['alpha'] = self.alpha
            if self.solar_mode == 'window_gq':
                key['solar']['nodes'] = self.nodes
                if self.align != 'centred':     # only here, so the centred caches' keys are unchanged
                    key['solar']['align'] = self.align
        if self.fast:     # only then, so every RAD_FAST=0 cache key is unchanged
            key['rad_fast'] = 1
        return key
