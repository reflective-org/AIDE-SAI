"""One-way coupled CESM -> TOMAS aerosol microphysics + transport.

Combines:
  * ../advection/fct.py   -> flux-corrected PPM 3-D advection (via fct_core.py)
  * models/tomas-jax      -> TOMAS sectional coagulation microphysics

One-way coupling, meaning specifically to CESM: CESM supplies the meteorology
(winds U/V/OMEGA, temperature T) that drives BOTH the transport and the
microphysics, and nothing here propagates back to it. The winds in particular are
prescribed throughout, so the circulation never responds to the aerosol.

RADIATIVE HEATING (RAD_T_FEEDBACK, 2026-11-17). The aerosol sets the optics, the
optics heat the layer, and that heating accumulates in dT_rad. What dT_rad is
allowed to touch is a switch:
  RAD_T_FEEDBACK=0 (default): dT_rad is a DIAGNOSTIC.
      The meteorology's own temperature T_met drives the microphysics, the
      settling, the RH and the radiation's aerosol optics; dT_rad enters only the
      emission temperature of the radiation's perturbed call (T_met + dT_rad), so
      the layer's extra longwave emission damps it (the Planck response) and it
      relaxes toward radiative balance instead of growing without bound. The
      heating rate of every call is written out as hr_kday [K/day]: FRAME3D
      'hr_kday', frames files 'frames_hr_kday' / 'frames_zm_hr_kday', timeseries
      'hr_min' / 'hr_max' / 'hr_rms', state ckpt and final file 'hr_kday'.
  RAD_T_FEEDBACK=1: the previous behaviour. T_met + dT_rad also drives the
      microphysics, settling, RH (MET_RH=liq|q) and optics: aerosol -> radiation
      -> temperature -> microphysics is a closed loop (only the dynamical one is
      absent).

State tracked per grid cell = TWO moments per size bin only:
    num[bin]  number mixing ratio   [# / kg air]
    mas[bin]  total dry-mass m.r.    [kg / kg air]
(40 bins -> 80 aerosol fields; SO2 and H2SO4 are advected too, so 82 in all).
Coagulation redistributes number and mass across bins; mass is conserved, number
decreases as particles merge.

The tomas coagulation kernel needs a 44-wide Mk only to compute particle
density; we place total mass in the SO4 slot (others zero) so density is a
constant ~1770 kg/m3 (pure sulfate). Nothing beyond (num, mas) is stored.

Initialization: CESM MAM4 modal aerosol (num_a{1,2,3}, so4_a{1,2,3}) is binned
onto the 40-bin TOMAS grid via per-mode dry log-normal distributions.

Open vertical boundaries (same rationale as ../advection/fct_openbc.py): the
top N_BC_TOP and bottom N_BC_BOT band levels are reset every coupling step to
the hourly CESM MAM4 aerosol binned onto the TOMAS grid. These reservoirs
carry the net effect of all the physics outside/off in this MVP (emissions, nucleation,
condensation, wet removal), giving a flux-through system instead of a sealed
one. The polar caps (|lat| > LAT_FREEZE) are likewise refreshed from MAM4 every
step rather than frozen at the IC. Number and mass are always pinned as a
consistent (Nk, Mk) pair from the same binning, never rescaled separately.
NO global mass fixer: with open boundaries the burden legitimately changes.

SAI extension (2026-07-16): the model now carries the full SO2 -> H2SO4 -> SO4
source chain and a gravitational-settling sink.
  * TWO extra advected gas tracers: so2 and h2so4 mass mixing ratios [kg/kg]
    (80 bin tracers + 2 gases = 82). Gas ICs/open-BCs/polar refresh come from
    CESM's own SO2 and H2SO4 h1 fields (mol/mol -> kg/kg), mirroring the MAM4
    treatment of the aerosol.
  * SOURCE: continuous SO2 injection (INJ_SO2_TG_YR at INJ_LAT/INJ_LON/INJ_HPA,
    optionally spread around the latitude ring with INJ_ZONAL) applied at each
    step start so the pulse is advected + oxidized within the same step.
  * CHEMISTRY+MICRO (MICRO=full, the default): per cell, the tomas-jax chain
    so2_chemistry -> nucleation -> coagulation -> condensation -- SO2+OH ->
    H2SO4 (Sun et al. 2022 Troe rate, OH from CESM's hourly OH field), H2SO4
    nucleation (Riccobono+Dunne) and condensation (PPM) forced by CESM
    RELHUM. Coagulation uses the FORTRAN-equivalent ADAPTIVE Euler solver
    (see the import note: make_step's fixed-substep coag is unstable after
    nucleation bursts at coupled-model dt). MICRO=coag falls back to the
    legacy coagulation-only path, bit-identical to the pre-SAI model.
  * SINK: per-bin slip-corrected Stokes settling with an implicit upwind
    column sweep and an OPEN bottom face (see settling.py) -- mass crossing
    the lowest level exits the model (sedimentation into the troposphere).
    This is the aerosol's one true sink; the staged budget gains a 'settle'
    stage that should match the recorded bottom outflow exactly.
"""
import os, sys, time
import collections
import numpy as np

# --- the two dependency repos, as submodules under models/ -------------------
# tomas-jax (microphysics) and jax-rrtmgp (radiation) are SEPARATE repos, not
# vendored here, and this used to be a bare absolute `sys.path.insert` -- which
# meant a fresh clone could not import its own first module on any other machine.
# They were sibling clones (`../<name>`) until 2026-08-27 and are now submodules
# under models/, which is what pins the exact commit behind a result.
# Bootstrap src/ onto sys.path so _paths imports, then let _paths put the
# process subdirectories (advection/, radiation/, settling/, microphysics/) on
# it too. That single import is what keeps every `import settling` / `import
# fct_lr` below working unchanged now that the files live in subdirectories.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _paths                      # noqa: E402 -- needs the insert above
_HERE = _paths.SRC
_MODELS = _paths.MODELS


def _dep_path(env, name):
    """Put a dependency submodule on sys.path; return what was used, or None.

    Search order: $<env> wins, so one variable overrides everything; then
    `models/<name>`, which is what `git submodule update --init` gives you.

    Finding nothing is NOT an error. Anything already importable -- pip install
    -e, PYTHONPATH, a venv -- needs none of these; the import below is the real
    test, and it reports the knob to set when it fails. But an env var that IS
    set and does NOT exist is a typo, and falling back silently would leave you
    debugging the wrong dependency, so that case raises.
    """
    explicit = os.environ.get(env)
    if explicit and not os.path.isdir(explicit):
        raise SystemExit(f"coupling.py: ${env}={explicit!r} is not a directory.\n"
                         f"  Point it at your {name} checkout, or unset it to "
                         f"search models/{name} and the import path.")
    for cand in (explicit, os.path.join(_MODELS, name)):
        if cand and os.path.isdir(cand):
            if cand not in sys.path:
                sys.path.insert(0, cand)
            return cand
    return None


TOMAS_JAX_PATH = _dep_path('TOMAS_JAX_PATH', 'tomas-jax')
RRTMGP_PATH = _dep_path('RRTMGP_PATH', 'jax-rrtmgp')   # used by radiation.py

# config import enables float64 before any jnp use
try:
    import tomas_jax.core.config as tconfig
except ImportError as _e:
    raise SystemExit(
        f"coupling.py: cannot import tomas_jax ({_e}).\n"
        "  tomas-jax is a separate repo and is NOT vendored in this one. Either\n"
        "  run `git submodule update --init` to populate models/tomas-jax,\n"
        "  install it, or point TOMAS_JAX_PATH at it:\n"
        "      TOMAS_JAX_PATH=/path/to/tomas-jax python3 driver_fast.py\n"
        f"  tried: {TOMAS_JAX_PATH or '(no candidate directory exists)'}")
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import xarray as xr

from tomas_jax.core.config import xk_boundaries, NBINS, ICOMP
from tomas_jax.solvers.diffrax import coag_euler_step
# full-chain microphysics (SAI): gas indices + the individual process steps.
# NOTE the Gc index quirk: SRTSO4 slot holds H2SO4 VAPOR in the gas array,
# and SRTSO2 (=43, the slot water uses in Mk) holds SO2.
from tomas_jax.core.config import (N_GAS_SPECIES, SRTSO4, SRTSO2, SRTH2O,
                                   MW_H2SO4, MW_SO2, KB, ICOMP_NODIAG)
# MICRO=full is tomas-jax dev's own make_step (dev 835ee24; needs a dev checkout
# in the tomas-jax slot -- the shipped presets pin gpu-fast 787c991 and run
# MICRO_ENGINE=fast), called per cell under vmap -- see run_microphysics_full.
# 2026-11-18: this replaced the hand-assembled per-cell chain (adaptive
# euler_step coagulation, capped nucleation) that used to live here.
# calc_oh_parabola is NOT a tomas-jax function -- it exists in no branch of the
# shared repo (checked 2026-07-29). It used to be imported from here, which meant
# coupling.py could only be imported at all via driver_fast.py's compat
# shim, and that shim aliased it to calc_oh_concentration(oh_const, cos_sza,
# use_diurnal=0) -- DIFFERENT argument order, constant mode by default -- so
# calc_oh_parabola(cs, OH_PEAK) silently returned cos(SZA) itself, i.e. OH ~ 1
# molec/cm3 instead of 2.2e6. Any OH_SZA=1 fast run made before this date had SO2
# oxidation effectively switched off. The fit now lives below, next to OH_PEAK.
from tomas_jax.physics.so2_chemistry import (so2_oxidation_step,
                                             calc_solar_zenith_angle)
# tomas-jax dev's make_step (MICRO=full) is built lazily by _dev_step(): its
# arguments (mnfix_trim, coag_solver, default_nucleation_params,
# WATER_SCHEME_H2SO4) exist only from dev 835ee24, and an import-time build
# would lock out the gpu-fast pin, whose driver swaps in tomas_jax.fast instead.

import settling

# --- advection: the flux-form scheme at the production config ------------------
# (fct_lr's sweeps; the production transport is ADV_SCHEME=sl, sl_lr.py, which
# uses fct_lr for its vertical -- see the ADV_SCHEME note below)
# This used to be `from fct_core import advect_step_batch`, i.e. the legacy
# sealed-face sweep. Nothing ever ran it: driver_fast.py rebinds this module's
# global to fct_lr before main() starts, so the only way to reach fct_core was to
# run coupling.py bare -- which silently gave you DIFFERENT transport (sealed
# vertical faces, no air-mass tracking) from every production run, under the same
# diagnostics. Importing fct_lr here makes standalone and production agree.
#
# The env-driven partial mirrors driver_fast.py exactly (ADV_CFL=0.5, ADV_F32=1)
# rather than taking fct_lr's own module defaults (cfl=0.2, float64), because the
# module defaults are NOT the validated production precision -- MANIFEST records
# two positivity-limiter bugs that were invisible in f64 and fatal in f32. The
# driver still rebinds this on top; identical value, so the rebind is a no-op.
#
# Consequence worth knowing: fct_lr exposes `return_vflux`, so the ADV_VFLUX probe
# below now answers True where fct_core made it False, which in turn moves the
# BC_EDGE default from 'clamp' to 'open'. That is the production boundary
# treatment -- see the note at that probe.
import functools
import fct_lr                      # src/advection/, put on sys.path by _paths
# ADV_SCHEME=sl (2026-10-13): semi-Lagrangian transport, advection/sl_lr.py --
# dinosaur's operator (vendored in sl_core.py) at the same call signature. No
# CFL sub-stepping: ADV_SL_K fixed sub-steps on the hourly CESM winds, a closed
# extended column with a per-tracer mass fixer, and Dirichlet reservoir levels
# (ADV_WCONT defaults to 0 under sl, see below). ~49 ms/step vs 6.6 s for lr
# at this grid; docs/TRANSPORT_SL_HANDOFF.md. Opt-in; 'fast' is honoured by
# driver_fast.py only (as before), bare coupling.py runs the default below.
# ADV_SCHEME=ffsl (2026-10-14): fct_lr itself at large Courant number
# (advection/ffsl_lr.py, the flux-form semi-Lagrangian extension of Lin & Rood
# 1996): the same flux-form operators, open faces, budgets and limiter, at
# ADV_FFSL_K sub-steps instead of 70-250 CFL sub-steps. THE PRODUCTION DEFAULT
# since 2026-10-15 (K=24 on the two step snapshots, ADV_SL_HOURLY=0): 30 days
# vs lr within 0.2 % on the budgets and 2 % on every quantity, advect 1.5 s/step
# instead of 7.0 (docs/PROCESSES.md 1.9). ADV_SCHEME=lr is the previous default,
# byte for byte (fct_lr is untouched; ffsl_lr imports from it).
# 2026-11-17: the default is ADV_SCHEME=sl (K=12, sl_lr.py), the recommended transport since 2026-11-05 (CH4 truth
# test, docs/PROCESSES.md 1.10) and what every emulator-linked run used; ffsl / lr stay selectable.
# 2026-11-17 (full numerical CESM linking): the default was briefly ADV_SCHEME=lr -- the
# then-production transport (fct_lr, CFL-sub-stepped flux-form Lin-Rood on the two
# step-end CESM snapshots, linear in time; numerically the earlier operator: fct_lr
# only had advect_hour_batch renamed advect_step_batch). driver_fast.py (and other drivers)
# now take the scheme from here instead of keeping their own default, which had
# drifted (they said ffsl while this said sl).
# 2026-11-18 (user decision): default switched to sl, K=12 (ADV_SL_K) on the
# bracketing 6-hourly snapshots -- 0.76x lr's CH4-truth RMSE and ~0.9 s vs
# 6-8 s per step at 82 tracers (advection-comparison REPORT.md). lr stays
# selectable; the PHYS_CFG 'ADV_SCHEME=lr' stamp is now 0 for a default run.
ADV_SCHEME = os.environ.get('ADV_SCHEME', 'sl').lower()
if ADV_SCHEME not in ('lr', 'sl', 'ffsl', 'fast'):   # fast = fct_fast, driver_fast.py only
    raise SystemExit(f"ADV_SCHEME={ADV_SCHEME!r} is not valid: lr | sl | ffsl | fast")
if ADV_SCHEME == 'sl':
    import sl_lr
    _adv_fn = sl_lr.advect_step_batch
elif ADV_SCHEME == 'ffsl':
    import ffsl_lr
    _adv_fn = ffsl_lr.advect_step_batch
else:
    _adv_fn = fct_lr.advect_step_batch
advect_step_batch = functools.partial(
    _adv_fn,
    cfl=float(os.environ.get('ADV_CFL', '0.5')),
    dtype=jnp.float32 if os.environ.get('ADV_F32', '1') != '0' else jnp.float64)

# ---- aerosol-representation seam (2026-11-05, MAM4 modal variant, run 3) ----
# The main loop used to assume the aerosol rows of the tracer stack are the
# NBINS TOMAS bins. a modal driver (a MAM4 driver, not part of this repo yet) carries 4 modes x (num, so4) instead and
# rebinds these; the defaults are the TOMAS behaviour, bit for bit. Everything
# bin-based that must stay untouched (Mie optics, per-bin Stokes settling, the
# size diagnostics, frames3d) sees the state through aer_to_bins. The settling
# module is bound late (it is imported below), hence the function wrapper.
AER_NROWS = NBINS            # rows per moment in the tracer stack (num, mas)
AER_FILL_FN = None           # (ds_mam, t, levs, lat_idx) -> (num, mas) in the
                             # prognostic representation; None = bin_mam4


def aer_to_bins(num, mas):   # prognostic (AER_NROWS, ...) -> TOMAS bins (NBINS, ...)
    return num, mas


def aer_settle(num, mas, T3d, pres3d, dp, dt, dpw, rhow):
    return settling.settle_step(num, mas, T3d, pres3d, dp, dt, dpw, rhow)

# Earth radius [m] and degrees->radians. Defined here rather than imported from an
# advection module: they are universal constants, not transport parameters, and
# importing them made an advection module look like a live dependency of the
# diagnostics. Identical to the values in fct_core and fast_advection/fct_fast.
RAD = 6.371e6
DEG = np.pi / 180.0

# =========================================================================
# Configuration
# =========================================================================
# ---- CESM forcing: the h1 hourly time series ------------------------------
# The single input dataset the model cannot run without. All four pieces are
# env-overridable (added 2026-08-04): the defaults are the FWHIST run these
# results were made from, on the machine they were made on, but a clone
# elsewhere points CESM_DIR at its own archive. The layout assumed is
# CESM's own tseries convention, one file per variable:
#     $CESM_DIR/hour_1/$CESM_PREFIX.h1.<VAR>$CESM_SUF
HDIR   = os.environ.get('CESM_DIR',
         '/data/cesm2.1.5_output/histSST/'
         'f.e21.FWHIST.f09_f09_mg17.atmos-scale_fixedSST_1996-2014.001/'
         'archive/atm/proc/tseries')
PREFIX = os.environ.get('CESM_PREFIX',
         'f.e21.FWHIST.f09_f09_mg17.atmos-scale_fixedSST_1996-2014.001.cam')
H1     = f'{HDIR}/hour_1/{PREFIX}.h1'
SUF_H  = os.environ.get('CESM_SUF', '.1996010100-2014123100.nc')

PS_REF   = 1.0e5          # reference surface pressure for level pressures (fct convention)
P_LO_HPA = float(os.environ.get('P_LO_HPA', '1.0'))    # top of band
# bottom of band. Default 100 hPa = stratosphere only (tropopause-ish); was 1000
# (=surface, full column). SAI aerosol is stratospheric and there is no
# tropospheric wet removal here, so the troposphere is both costly and unphysical
# to carry; env-overridable to retune per experiment.
P_HI_HPA = float(os.environ.get('P_HI_HPA', '100.0'))
N_LEV    = int(os.environ.get('N_LEV', '0'))   # 0 => full contiguous band; else sub-sample to ~N_LEV levels
RD       = 287.05         # J/kg/K, dry-air gas constant
GRAV     = 9.80665
BOXVOL   = 1.0e6          # cm3 == 1 m3; Nk/Mk are then per-m3 concentrations

# run controls (env-overridable so scaling up is a one-liner)
N_DAYS   = int(os.environ.get('N_DAYS', '2'))
H0       = int(os.environ.get('H0', '0'))       # start hour index into h1 series
# Coupling timestep. EVERYTHING runs on this cadence: winds are sampled at t and
# t+STEP_HOURS, the tracers are advected across that interval (with internal CFL
# sub-stepping), and coagulation is called once with dt = STEP_HOURS. PARADIS
# output is 6-hourly, so STEP_HOURS=6 makes the whole model consistent with the
# PARADIS snapshots (one advect+coag per snapshot interval).
STEP_HOURS = int(os.environ.get('STEP_HOURS', '6'))   # hours per coupling step
DT_MICRO = STEP_HOURS * 3600.0                  # coag dt [s] (spans the step)
STEP_SEC = STEP_HOURS * 3600.0                  # advection interval [s]
N_COAG_SUBSTEPS = int(os.environ.get('N_COAG_SUBSTEPS', '3'))
CELL_CHUNK = int(os.environ.get('CELL_CHUNK', '50000'))  # cells per micro vmap call (memory/overhead only; 50k benchmarked, and splits evenly over MICRO_GPUS)
# Advect this many of the 82 tracers per batch (0 = all at once). Chunking the
# tracer axis caps advection peak memory so the band fits on a memory-tight GPU.
# The ~9 GB single allocation that motivated this knob was measured on the older
# 43-level band with 80 tracers; the production 24-level band needs less, but the
# knob is the lever either way if advection OOMs.
TRACER_CHUNK = int(os.environ.get('TRACER_CHUNK', '0'))
LAT_FREEZE = 80.0
# Transport-scheme fixes live in fast_advection/fct_fast.py and are read from the
# same env knobs there, so the burden weight here always matches the sweep.
# ADV_METRIC=1 (default) => the y-sweep carries the cos(phi) area metric and the
# burden must be weighted by the exact cell-mean cos(phi) (see A below).
ADV_METRIC = os.environ.get('ADV_METRIC', '1') != '0'
# ADV_WCONT=1 (default) => omega is rederived from discrete continuity and the
# slab's vertical faces are OPEN, so the aerosol gains a real advective
# source/sink there. That exchange is ~10%/day of the slab GROSS, so it is
# diagnosed explicitly into the mass budget as the 'vflux' term.
# Under ADV_SCHEME=sl with ADV_SL_VERT=sl (the handoff's 3-D design, A/B-failed)
# the default is 0: that operator's column is CLOSED and mass-fixed, so the
# exchange with the reservoir is the Dirichlet reset of the edge levels after
# the step (BC_EDGE=clamp, booked in B_bc); ADV_WCONT=1 there leaves the edge
# levels free with no exchange at all, and the header warns. The default
# ADV_SL_VERT=flux (SL horizontal + fct_lr's flux-form vertical) has lr's open
# faces and takes lr's default, 1.
_SL_CLOSED = ADV_SCHEME == 'sl' and sl_lr.ADV_SL_VERT == 'sl'
ADV_WCONT = os.environ.get('ADV_WCONT', '0' if _SL_CLOSED else '1') != '0'
# sl and ffsl: ADV_SL_HOURLY=1 reads the hourly CESM winds between the two step
# snapshots so each of the K sub-steps advects on its own hour's wind (~1.3 s/step
# of extra I/O). Default 0 since 2026-10-15: the two step-end snapshots, linear
# in time at every sub-step -- what lr always did, and the cadence production
# receives (six-hourly winds from CESM today, from an atmospheric emulator later).
# The hourly mode is a diagnostic; its value is ~3 % in number and forcing at
# day 30 (docs/PROCESSES.md 1.9, wind cadence).
ADV_SL_HOURLY = ADV_SCHEME in ('sl', 'ffsl') and os.environ.get('ADV_SL_HOURLY', '0') != '0'
# ---- meteorology source (2026-11-05) -----------------------------------------
# MET_SRC=cesm (default): U/V/OMEGA/T hourly from the CESM h1 archive, as always.
# MET_SRC=file: the four fields at the two step ends from MET_DIR, written by
# a converter (emu_to_met.py, not part of this repo yet) from a JCM-emulator free run regridded onto this
# band and grid (format documented with the converter; hour0 in its meta.json puts frame 0
# on the archive's hour axis, so H0 + t indexes both the same way). Everything
# the emulator cannot supply -- the aerosol IC/BC reservoirs, the gases, water
# vapour, the radiation column below the band, the surface -- still comes from
# the archive at the matching hour, or from CLIM_FILE under BC_SRC=clim
# (docs/BOUNDARY_CONDITIONS.md). The file holds only the step ends, so
# ADV_SL_HOURLY is refused rather than silently fed interpolated winds.
MET_SRC = os.environ.get('MET_SRC', 'cesm').lower()
MET_DIR = os.environ.get('MET_DIR', '')
# RH under MET_SRC=file. 'q' (default): the archive's water vapour at the hour
# with the FILE's temperature, saturation over ICE (Goff-Gratch) -- because that
# is what CESM's RELHUM is on these levels (checked 2026-11-05, hour 8766:
# Q/qsat_ice / RELHUM = 1.001 median, 1.014 p95; over water it is 0.59), so the
# water uptake and nucleation see the convention every CESM-driven run gave
# them, now paired with the temperature that actually drives the run.
# 'cesm': the archive's RELHUM as is (CESM's temperature paired with the file's).
# 2026-11-17 (full numerical CESM linking): new value and new DEFAULT for every
# MET_SRC, 'liq': RH over (supercooled) LIQUID water, recomputed every step from
# the BC source's water vapour Q, the meteorology's temperature T_met and the
# level pressure, with the Tabazadeh et al. (1997) eq. 1 saturation pressure --
# settling.rh_liquid_pct(), the same function the radiation's wet optics use. Every
# consumer of coupling's RH (the Tabazadeh H2SO4/H2O wt% in the microphysics
# water, the settling's wet size) takes RH over liquid -- the radiation's wet
# optics compute the same liquid RH from Q themselves whatever MET_RH is, and
# SO2+OH's Buck term is within 2.3% of it; nucleation takes no RH; feeding them
# the ice RH ('q', 'cesm') overstated it 1.6-1.9x on the band (190-220 K) and
# made the settling droplets wetter than the optics' ones. 'q' and 'cesm' stay
# selectable to reproduce older runs.
MET_RH = os.environ.get('MET_RH', 'liq').lower()
# 2026-11-18 (JCM grid support): MET_SRC=jcm. The meteorology (U/V/OMEGA/T,
# band and full radiation column) from a jcm-strat run directory JCM_DIR
# (src/met/met_jcm.py), 6-hourly, AND the model GRID with it: lat/lon/levels
# are whatever the met source declares (JCM: T63 Gaussian 96 x 192, 95 hybrid
# levels), so nothing downstream assumes f09. Everything JCM does not provide
# comes from CLIM_FILE, which must be the CESM climatology regridded onto the
# same grid (scripts/met/regrid_clim_to_grid.py); install_bc_source refuses a
# mismatch. Needs BC_SRC=clim. Only the step ends are read (6-hourly output),
# so ADV_SL_HOURLY is refused, as under MET_SRC=file.
JCM_DIR = os.environ.get('JCM_DIR', '/data/JCM_stripped/jcm-strat/runs/p31/p31_AL7h70')
# MET_Q (MET_SRC=jcm only): where water vapour Q comes from. 'bc' (default): the
# BC source (the regridded CESM climatology) -- because jcm-strat is a dry model
# and its specific_humidity is a passive constant 2.32e-3 kg/kg (met_jcm.py).
# 'met': JCM's own field (only meaningful for a moist JCM run).
MET_Q = os.environ.get('MET_Q', 'bc').lower()
# 2026-11-18 (live JCM): MET_SRC=jcm_online. The same fields on the same
# grid as MET_SRC=jcm, but from the jcm-strat model RUNNING alongside this one
# (src/met/met_jcm_online.py + jcm_server.py) with the AL7h configuration and all
# of its forcing active -- ERA5 nudging below 400 hPa, QBO nudging, the AL7h T_e
# correction, gravity-wave drag -- instead of from that run's stored output.
# JCM runs in its own interpreter (JCM_PY, a venv built for jcm-strat) as a
# subprocess on the same GPU, started at 1 January 00Z of the first hour read
# from a jcm checkpoint: this run's own (JCM_WORK/jcm_YYYY0101.ckpt, written at
# every year end) or else the JCM_DIR chain's. JCM_TEMPLATE: the resolved Hydra
# config the model is built from (the AL7h70 1996 segment's). Everything said
# above for MET_SRC=jcm (grid, BC_SRC=clim, MET_Q, 6-hourly) holds for both.
# The jcm-strat CODE (jcm_strat and, from its external/jax-gcm submodule, jcm)
# is the jcm-strat model slot: JCM_STRAT_PATH, set by the preset's #@submodule
# line (default models/jcm-strat); the server imports both from there, never
# from the venv's own installs. Its machine data is not in a checkout and has
# its own knobs: JCM_CACHE (ERA5 windows in era5/, jcm boundary files in
# huggingface/), JCM_SCRATCH (JAX compile cache). JCM_TE_CORRECTION: the AL7h
# correction file, default the slot's tracked copy (the template's path names a
# jcm-strat worktree that no longer exists). None of these is read unless
# MET_SRC=jcm_online.
JCM_TEMPLATE = os.environ.get('JCM_TEMPLATE', JCM_DIR.rstrip('/') + '/'
                              + os.path.basename(JCM_DIR.rstrip('/'))
                              + '_19960101/.hydra/config.yaml')
JCM_STRAT_PATH = os.path.abspath(os.environ.get('JCM_STRAT_PATH')
                                 or os.path.join(_MODELS, 'jcm-strat'))
JCM_PY = os.environ.get('JCM_PY') or os.path.join(JCM_STRAT_PATH, '.venv/bin/python')
JCM_CACHE = os.environ.get('JCM_CACHE') or os.path.join(JCM_STRAT_PATH, 'cache')
JCM_SCRATCH = os.environ.get('JCM_SCRATCH') or os.path.join(JCM_STRAT_PATH, 'scratch')
JCM_WORK = os.environ.get('JCM_WORK', 'jcm_online')
JCM_TE_CORRECTION = os.environ.get('JCM_TE_CORRECTION') or None   # None: the slot's AL7h.nc
if MET_SRC not in ('cesm', 'file', 'jcm', 'jcm_online'):
    raise SystemExit(f"MET_SRC={MET_SRC!r} is not valid: cesm | file | jcm | jcm_online")
MET_JCM = MET_SRC in ('jcm', 'jcm_online')      # the JCM grid, whichever way it is fed
if MET_Q not in ('bc', 'met'):
    raise SystemExit(f"MET_Q={MET_Q!r} is not valid: bc | met")
if MET_JCM and ADV_SL_HOURLY:
    raise SystemExit(f"ADV_SL_HOURLY=1 needs hourly winds; MET_SRC={MET_SRC} is 6-hourly")
if MET_JCM and os.environ.get('BC_SRC', 'cesm').lower() != 'clim':
    raise SystemExit(f"MET_SRC={MET_SRC} needs BC_SRC=clim with CLIM_FILE on the JCM grid "
                     "(scripts/met/regrid_clim_to_grid.py)")
if MET_SRC == 'file' and not MET_DIR:
    raise SystemExit("MET_SRC=file needs MET_DIR (an emu_to_met.py output; that converter is not part of this repo)")
if MET_SRC == 'file' and ADV_SL_HOURLY:
    raise SystemExit("ADV_SL_HOURLY=1 needs hourly winds; MET_SRC=file holds only the step ends")
if MET_RH not in ('liq', 'q', 'cesm'):
    raise SystemExit(f"MET_RH={MET_RH!r} is not valid: liq | q | cesm")
# ---- boundary source (2026-11-05) ----------------------------------------------
# BC_SRC=cesm (default): every non-meteorology field (aerosol IC/slabs/caps and
# the anomaly-mode reference aerosol, SO2/H2SO4, Q/O3/CH4 and the surface for
# the radiation column, RELHUM, OH) from the hourly archive at the hour.
# BC_SRC=clim: the same fields from CLIM_FILE, a monthly day-of-year climatology
# of the archive (scripts/met/make_bc_climatology.py), served through the same
# open_var() interface by src/met/bc_clim.py; U/V/OMEGA/T come from the met
# file, including the full radiation column T. The run then opens NO archive
# file -- docs/BOUNDARY_CONDITIONS.md.
# 2026-11-17 (full numerical CESM linking): BC_SRC=clim with MET_SRC=cesm is allowed.
# The opener is then a per-variable ROUTER (main(), open_var): U/V/OMEGA/T --
# the meteorology, band and full radiation column alike -- from the hourly
# archive; everything else (Q/O3/CH4, the surface fluxes, the MAM4 reservoirs
# and the radiation reference aerosol, SO2/H2SO4, OH, RELHUM) from the
# climatology at the archive hour's day of year. The climatology's own T is
# never served under the router (it is a monthly mean).
BC_SRC = os.environ.get('BC_SRC', 'cesm').lower()
CLIM_FILE = os.environ.get('CLIM_FILE', '')
if BC_SRC not in ('cesm', 'clim'):
    raise SystemExit(f"BC_SRC={BC_SRC!r} is not valid: cesm | clim")
if BC_SRC == 'clim' and not CLIM_FILE:
    raise SystemExit("BC_SRC=clim needs CLIM_FILE (a scripts/met/make_bc_climatology.py output)")
# ADV_PASSIVE=1 (2026-11-05): the passive-tracer transport TEST. Instead of the
# 80 bin moments, ONE tracer -- the total sulfate mass per cell -- is advected
# (plus SO2 and H2SO4 as before), and every bin's number and mass in a cell is
# then rescaled by the cell's mass ratio after/before, i.e. the size
# distribution SHAPE of each cell is frozen through the transport step. A cell
# that was empty and receives mass takes the pre-step zonal-mean shape of its
# level (the global-mean shape of the level if the row is empty too). Transport
# is linear and the SL scheme already shares the departure points across the
# whole stack, so this is NOT a speed knob (the 82-tracer step costs under a
# second); it measures what the per-bin transport does beyond moving the total:
# plume and background distributions differ, and settling and coagulation make
# the bins' spatial fields differ, so a run with this on is scored against the
# same run with it off (the campaign's run 6 vs run 2). Not a production option.
ADV_PASSIVE = os.environ.get('ADV_PASSIVE', '0') != '0'


def qsat_ice(T, p):
    """Saturation specific humidity over ice [kg/kg] from T [K], p [Pa]:
    Goff-Gratch ice (CAM's GoffGratch_svp_ice), the formula behind CESM's
    RELHUM on the band levels (see MET_RH above)."""
    t0 = 273.16
    es = 10.0 ** (-9.09718 * (t0 / T - 1.0) - 3.56654 * np.log10(t0 / T)
                  + 0.876793 * (1.0 - T / t0) + np.log10(6.1071)) * 100.0
    return 0.622 * es / np.maximum(p - 0.378 * es, 1.0)
# AEROSOL concentration carried by air flowing INTO the slab through the bottom
# face, as a scale on the reservoir (CARMA/MAM4) value there. This is a first-order
# control on the aerosol budget: with the faces open, ~10%/day of the slab's air is
# exchanged through ~88 hPa, so what that air carries sets the steady-state burden.
#   1.0 = reservoir value (default; the CARMA 88 hPa level, which is aerosol-RICH
#         and also carries a ~17600 #/cm3 ultrafine mode -- see the
#         number-diagnostic notes)
#   0.0 = aerosol-free inflow: tropical upwelling brings up tropospheric air that
#         has essentially no stratospheric sulfate. Physically the better default
#         for SAI work, and it removes the artificial ultrafine injection.
# GASES are unaffected -- SO2/H2SO4 inflow always uses the reservoir, since
# tropospheric air genuinely is the SO2 source for the band.
BC_BOT_AER = float(os.environ.get('BC_BOT_AER', '1.0'))
N_BC_TOP = int(os.environ.get('N_BC_TOP', '1'))  # top band levels pinned to hourly MAM4
N_BC_BOT = int(os.environ.get('N_BC_BOT', '1'))  # bottom band levels pinned to hourly MAM4
PROBE_HPA  = float(os.environ.get('PROBE_HPA', '50')) # level [hPa] for diagnostics/frames
# ---- SAI sources & sinks (SO2 -> H2SO4 -> SO4 chain + settling) ----------
# 'full' = so2_chemistry + nucleation + coagulation + condensation per cell;
# 'coag' = legacy coagulation-only path (bit-identical to the pre-SAI model);
# 'off'  = NO microphysics at all -- transport only. Added 2026-08-27 for
#          timing/benchmark runs: it goes through the same step loop (advection,
#          boundaries, budget, checkpointing) so what is measured is the
#          production advection, not a separate harness that can drift from it.
#          Not a physics configuration -- the aerosol only moves, never evolves.
MICRO_MODE = os.environ.get('MICRO', 'full')
# Validate here rather than letting a typo fall through to the legacy 'coag'
# branch: MICRO=fulll would otherwise silently run -- and be reported as -- the
# pre-SAI coagulation-only model.
if MICRO_MODE not in ('full', 'coag', 'off'):
    raise SystemExit(f"coupling: MICRO must be full|coag|off, got {MICRO_MODE!r}")
# (MICRO_SUBSTEPS, the old chain's 1 h pieces, is gone with it: MICRO=full now
# sub-steps at MICRO_DT, dev's step length -- see run_microphysics_full.)
if 'MICRO_SUBSTEPS' in os.environ:
    raise SystemExit("coupling: MICRO_SUBSTEPS no longer exists -- MICRO=full is "
                     "tomas-jax dev make_step at MICRO_DT (default 60 s)")
# continuous SO2 release, the SAI forcing, in Tg(SO2)/yr. DEFAULT 0 = OFF, i.e. a
# no-injection control run. It defaulted to 10 until 2026-08-03; the default was
# dropped to 0 so that forgetting the flag gives you an obviously-unforced baseline
# instead of a silent standard-magnitude SAI scenario that looks like a real result.
# To reproduce the prod90d/prod1yr runs, pass INJ_SO2_TG_YR=10 explicitly.
INJ_SO2_TG_YR = float(os.environ.get('INJ_SO2_TG_YR', '0.0'))
# Ad-hoc H2SO4 injection (off by default): added for the microphysics emulator
# (not part of this repo yet), which consumed H2SO4 but not SO2; it sources the sulfur
# directly as gas-phase H2SO4 at the same geometry. S-equivalent to X Tg SO2/yr is X*(98.08/64.06)=X*1.531 Tg H2SO4/yr.
INJ_H2SO4_TG_YR = float(os.environ.get('INJ_H2SO4_TG_YR', '0.0'))
INJ_LAT  = float(os.environ.get('INJ_LAT', '0.0'))
_INJ_LON_DEFAULT = 180.0
INJ_LON  = float(os.environ.get('INJ_LON', str(_INJ_LON_DEFAULT)))
INJ_HPA  = float(os.environ.get('INJ_HPA', '55.0'))
# Default 1 (zonal ring), matching run_prod.py's production config -- until
# 2026-08-03 this defaulted to 0 (single cell), so a bare `python3 coupling.py`
# and a bare `./run_prod.py` injected at DIFFERENT geometries. run_prod.py always
# exports this explicitly, so production itself was never affected; only the
# standalone/dev path was.
INJ_ZONAL = os.environ.get('INJ_ZONAL', '1') != '0'  # 1 = spread over the full lat ring
# INJ_MIRROR=1 releases at BOTH +INJ_LAT and -INJ_LAT, splitting the SAME total
# rate 50/50 between them -- INJ_SO2_TG_YR=10 INJ_LAT=45 INJ_MIRROR=1 puts 5 Tg/yr
# at 45N and 5 Tg/yr at 45S. The total is what you asked for, NOT doubled. This is
# the standard symmetric-pair SAI configuration; a single off-equator band forces
# one hemisphere and drives a cross-equatorial gradient that is usually an artifact
# of the setup rather than the intent. No-op at INJ_LAT=0 (a row is its own mirror).
INJ_MIRROR = os.environ.get('INJ_MIRROR', '0') != '0'
# The scenario as one array, for the state ckpt (so a RESUME cannot cross scenarios)
# and for every output file (so a sweep's npz says which scenario produced it -- the
# filename TAG is a label the user chose, not evidence). APPEND-ONLY: the RESUME
# check compares the common prefix, so a checkpoint written before a field was added
# stays resumable instead of being locked out by a length change.
INJ_CFG = np.array([INJ_SO2_TG_YR, INJ_H2SO4_TG_YR, INJ_LAT, INJ_LON, INJ_HPA,
                    float(INJ_ZONAL), float(INJ_MIRROR)])
INJ_CFG_KEYS = np.array(['INJ_SO2_TG_YR', 'INJ_H2SO4_TG_YR', 'INJ_LAT',
                         'INJ_LON', 'INJ_HPA', 'INJ_ZONAL', 'INJ_MIRROR'])
SETTLE_ENABLE = os.environ.get('SETTLE', '1') != '0'
# Size the settling particle as the WET H2SO4/H2O droplet (Tabazadeh composition,
# Tang density) instead of the dry SO4 core. ON by default as of 2026-08-03;
# 0 restores the dry DP_BIN/RHO_AER sizing of earlier runs. See the settling block.
WET_SETTLING = os.environ.get('WET_SETTLING', '1') != '0'
# Mirrors radiation.py's WET_OPTICS (same env var, same default), read here so the
# SIZE DIAGNOSTICS report the same particle the Mie tables integrate. Keep the two
# in step: reporting a dry r_eff for wet optics is what this flag exists to prevent.
WET_OPTICS = os.environ.get('WET_OPTICS', '1') != '0'
# The PHYSICS-MODE flags, stamped into the state ckpt alongside INJ_CFG so a RESUME
# can say whether it is continuing the same model. Kept SEPARATE from INJ_CFG, and
# checked with a WARNING rather than a refusal, because the two failure modes are
# genuinely different:
#   INJ_CFG mismatch  -> the checkpoint's aerosol was made by a different SOURCE, so
#                        the resumed output would be misattributed. Nothing downstream
#                        can detect it. Refuse.
#   PHYS_CFG mismatch -> the state is still valid; the model integrating it forward
#                        changes at the seam. That is sometimes exactly what you want
#                        (the 2026-08-03 wet-physics flip has to be applied to an
#                        in-flight run somehow) and refusing would strand every
#                        pre-flip checkpoint, including the 91.5%-complete prod1yr.
#                        So: say it loudly, in the log the run is self-describing by,
#                        and continue.
# APPEND-ONLY, same as INJ_CFG -- the check compares the common prefix, so adding a
# field here does not lock out checkpoints written before it existed.
# radiation.py SOLAR_MODES, duplicated here because PHYS_CFG is built at import and
# coupling.py must not import radiation (RAD=0 drops the jax-rrtmgp dependency).
# Keep the two in step; the order is the PHYS_CFG code and is append-only.
_SOLAR_MODES = ('instant', 'window', 'window_ins', 'window2', 'window2_ins', 'window_blend',
                'dither', 'window_gq')
# radiation.py RAD_SOLAR_DEFAULT, duplicated: window_gq, or instant under RAD_ENGINE=emulator.
_RAD_SOLAR = os.environ.get('RAD_SOLAR', 'instant' if os.environ.get('RAD_ENGINE', 'rrtmgp').lower() == 'emulator'
                            else 'window_gq').lower()
PHYS_CFG = np.array([float(WET_SETTLING), float(WET_OPTICS), float(SETTLE_ENABLE),
                     float(os.environ.get('ADV_VPOS', '1') != '0'),
                     # 2026-10-09: which engine fills the microphysics seam
                     # (driver_fast.py MICRO_ENGINE; 0 = tomas_jax.fast, 1 = the
                     # learned emulator). Appended, so older ckpts still resume.
                     float(os.environ.get('MICRO_ENGINE', 'fast').lower() == 'emulator'),
                     # 2026-10-13: semi-Lagrangian transport (ADV_SCHEME=sl). Appended.
                     float(ADV_SCHEME == 'sl'),
                     # 2026-10-13: which engine solves the evolved radiation call
                     # (RAD_ENGINE; 0 = rrtmgp, 1 = the column networks). Appended.
                     float(os.environ.get('RAD_ENGINE', 'rrtmgp').lower() == 'emulator'),
                     # 2026-10-14: fct_lr at large Courant number (ADV_SCHEME=ffsl). Appended.
                     float(ADV_SCHEME == 'ffsl'),
                     # 2026-11-05: meteorology from a file, not the archive (MET_SRC=file). Appended.
                     float(MET_SRC == 'file'),
                     # 2026-11-05: boundaries from the climatology file (BC_SRC=clim). Appended.
                     float(BC_SRC == 'clim'),
                     # 2026-11-05: the passive-tracer transport test (ADV_PASSIVE). Appended.
                     float(ADV_PASSIVE),
                     # 2026-11-11: solar geometry per radiation call (radiation.py
                     # RAD_SOLAR), as its index in _SOLAR_MODES: 0 = instant, 1 = window,
                     # 2 = window_ins, 3 = window2, 4 = window2_ins, 5 = window_blend, 6 = dither,
                     # 7 = window_gq. Appended.
                     float(_SOLAR_MODES.index(_RAD_SOLAR)
                           if _RAD_SOLAR in _SOLAR_MODES
                           else -1),
                     # 2026-11-12: window_blend's weight (radiation.py RAD_SOLAR_ALPHA; its
                     # default 0.53 is duplicated here), -1 under every other mode so a
                     # stray RAD_SOLAR_ALPHA does not warn on their resume. Appended.
                     (float(os.environ.get('RAD_SOLAR_ALPHA', '0.53'))
                      if _RAD_SOLAR == 'window_blend'
                      else -1.0),
                     # 2026-11-12: window_gq's node count (radiation.py RAD_SOLAR_NODES,
                     # default 3 duplicated here), -1 under every other mode. Appended.
                     (float(os.environ.get('RAD_SOLAR_NODES', '3'))
                      if _RAD_SOLAR == 'window_gq'
                      else -1.0),
                     # 2026-11-17: window_gq's window (radiation.py RAD_SOLAR_ALIGN, default
                     # trailing duplicated here): 1 = trailing (t-W..t), 0 = centred,
                     # -1 under every other mode. Appended.
                     ((0.0 if os.environ.get('RAD_SOLAR_ALIGN', 'trailing').lower() in ('centred', 'centered')
                       else 1.0)
                      if _RAD_SOLAR == 'window_gq'
                      else -1.0),
                     # 2026-11-17 (full numerical CESM linking), all appended:
                     # coupling.py's own full TOMAS chain fills the micro seam
                     # (driver_fast.py MICRO_ENGINE=full).
                     # Stamped here as what coupling.py alone runs (MICRO=full
                     # is its chain); driver_fast.py (or another driver) overwrites
                     # it with the engine they actually install.
                     float(MICRO_MODE == 'full'),
                     # the transport is fct_lr (ADV_SCHEME=lr; default from 2026-11-17 to
                     # 2026-11-18, sl since).
                     float(ADV_SCHEME == 'lr'),
                     # RH convention (MET_RH): 0 = archive RELHUM, 1 = Q over ice,
                     # 2 = Q over liquid (Tabazadeh eq. 1).
                     float({'cesm': 0, 'q': 1, 'liq': 2}[MET_RH]),
                     # dT_rad drives the microphysics/settling/RH/optics (1) or is
                     # diagnostic, emission temperature only (0).
                     float(os.environ.get('RAD_T_FEEDBACK', '0') != '0'),
                     # 2026-11-18, all appended: MICRO=full is tomas-jax dev
                     # make_step; its time step [s], coag sub-steps per step,
                     # coag solver (0 = euler_fixed, 1 = euler_adaptive) and the
                     # ion-pair production rate [cm-3 s-1] it is called with.
                     float(os.environ.get('MICRO_DT', '60')),
                     float(os.environ.get('MICRO_NCOAG', '1')),
                     float(os.environ.get('MICRO_COAG', 'fixed').lower() == 'adaptive'),
                     float(os.environ.get('MICRO_FION', '0')),
                     # 2026-11-18 (JCM grid support), appended: meteorology
                     # and grid from JCM (MET_SRC=jcm), and Q from the met source
                     # rather than the BC source (MET_Q=met). The grid itself is
                     # in the separate, REFUSED GRID_CFG stamp.
                     float(MET_SRC == 'jcm'),
                     float(MET_Q == 'met'),
                     # 2026-11-18 (live JCM), appended: JCM run live
                     # (MET_SRC=jcm_online) rather than read from its output.
                     float(MET_SRC == 'jcm_online'),
                     # 2026-11-22, appended: sl_lr renormalises each level's air
                     # mass after the horizontal step (ADV_SL_RHONORM, default 1;
                     # 0 reproduces the runs before 2026-11-22). -1 when the
                     # transport is not sl.
                     (float(os.environ.get('ADV_SL_RHONORM', '1') != '0')
                      if ADV_SCHEME == 'sl' else -1.0)])
PHYS_CFG_KEYS = np.array(['WET_SETTLING', 'WET_OPTICS', 'SETTLE', 'ADV_VPOS',
                          'MICRO_ENGINE=emulator', 'ADV_SCHEME=sl', 'RAD_ENGINE=emulator',
                          'ADV_SCHEME=ffsl', 'MET_SRC=file', 'BC_SRC=clim', 'ADV_PASSIVE',
                          'RAD_SOLAR', 'RAD_SOLAR_ALPHA', 'RAD_SOLAR_NODES',
                          'RAD_SOLAR_ALIGN', 'MICRO_ENGINE=full', 'ADV_SCHEME=lr',
                          'MET_RH', 'RAD_T_FEEDBACK', 'MICRO_DT', 'MICRO_NCOAG',
                          'MICRO_COAG=adaptive', 'MICRO_FION', 'MET_SRC=jcm', 'MET_Q=met',
                          'MET_SRC=jcm_online', 'ADV_SL_RHONORM'])
ALPHA_COND = float(os.environ.get('ALPHA_COND', '1.0'))  # H2SO4 accommodation coeff (SAI box-model value)
# ---- diurnal OH from a solar-zenith-angle curve (Hanisco et al. 2001, Fig. 1) ----
# DEFAULT ON. The SO2+OH chemistry uses OH(theta) read off the paper's figure
# (theta = SZA in deg, knots in OH_SZA_KNOTS below) instead of CESM's OH field,
# evaluated PER micro substep at
# each grid point's local zenith angle so the oxidation resolves the diurnal
# cycle. CESM OH is otherwise sampled only once per STEP_HOURS (at the step start)
# and held constant across the 6 substeps, which aliases the strong OH day/night
# swing. OH_SZA=0 restores that legacy behaviour. OH_PEAK is the noon peak
# [molec/cm3].
#
# PARABOLA IN mu = cos(SZA), NOT IN theta (changed 2026-07-29, user's chosen form):
#   OH(mu) = a*mu^2 + b*mu,   a,b from lstsq over OH_SZA_KNOTS, no constant term
# The missing constant term is what forces OH(mu=0) = 0 at the terminator. It is a
# LEAST-SQUARES fit, so it does NOT pass through the knots -- it reads +4.0% at
# SZA=0, -4.8% at 30, -7.2% at 45, +19.2% at 60, exact 0 at 90. That is accepted:
# a smooth 2-parameter function of mu is more defensible than an interpolant
# through five hand-digitized points (OH production tracks actinic flux ~ mu).
# The superseded form was OH_PEAK*max(0,1-(theta/90)^2), which decayed far too
# slowly -- 1.22e6 at theta=60 against the paper's 0.7e6, 74% high.
#
# WHY mu IS CLAMPED AND NOT THE OUTPUT: a*mu^2 + b*mu is a parabola with roots at
# mu=0 and mu=-b/a, so it turns POSITIVE AGAIN for mu < -b/a = -0.652, i.e. beyond
# SZA 130.7 deg -- it would emit up to +5.0e5 molec/cm3 of OH at local midnight.
# A jnp.maximum(0, ...) on the result does NOT catch this (the value is positive).
# Clamping mu into [0,1] is what makes the whole night exactly zero. Do not
# "simplify" that clip away. A plot over SZA in [-90, 90] cannot show this bug,
# because that range is exactly mu >= 0.
#
# TWO THINGS TO KNOW BEFORE READING A RUN THAT USES THIS:
#  1. It is a REPLACEMENT, not a resampling -- the curve discards CESM's OH
#     field entirely, so the step-mean magnitude changes too (and with it the SO2
#     lifetime), not just the time resolution.
#  2. BOTH drivers now resolve the cycle, but at their own inner resolution.
#     MICRO=full (tomas-jax dev make_step) reads OH_SUBSTEPS samples per step,
#     dev step i taking sample i*OH_SUBSTEPS//NSTEP_MICRO (60 samples = one per
#     6 min at the defaults; the curve is smooth on that scale).
OH_SZA  = os.environ.get('OH_SZA', '1') != '0'
OH_PEAK = float(os.environ.get('OH_PEAK', '2.3e6'))   # peak OH at SZA=0 [molec/cm3]
# (SZA [deg], OH [molec/cm3]) digitized off the paper's OH-vs-SZA figure. THIS is
# the data the curve is fitted to -- to refine it, edit/add points here and the
# refit happens at import; nothing else changes. Must stay sorted in SZA and
# non-increasing in OH (asserted at import).
OH_SZA_KNOTS = ((0.0, 2.3e6), (30.0, 2.0e6), (45.0, 1.5e6),
                (60.0, 0.7e6), (90.0, 0.0))
_OH_TH = np.array([k[0] for k in OH_SZA_KNOTS])
_OH_OH = np.array([k[1] for k in OH_SZA_KNOTS])
assert np.all(np.diff(_OH_TH) > 0), 'OH_SZA_KNOTS must be sorted by SZA'
assert np.all(np.diff(_OH_OH) <= 0), 'OH_SZA_KNOTS must be non-increasing in OH'
# least-squares parabola in mu = cos(SZA), no constant term so OH(SZA=90) == 0:
#   [mu^2, mu] @ [a, b] ~= OH        ->  _OH_A, _OH_B
_OH_MU = np.cos(np.radians(_OH_TH))
_OH_A, _OH_B = np.linalg.lstsq(np.vstack([_OH_MU ** 2, _OH_MU]).T,
                               _OH_OH, rcond=None)[0]
# OH_PEAK rescales the whole curve; it is normalized against the SZA=0 KNOT, so
# the default OH_PEAK == that knot reproduces the raw fit exactly (scale 1.0).
# NOTE the fit's own value at SZA=0 is 1.0402x the knot (2.393e6 vs 2.3e6) because
# it is least-squares, not interpolating -- so peak OH != OH_PEAK by 4%.
_OH_SCALE = 1.0 / OH_SZA_KNOTS[0][1]


def calc_oh_parabola(cos_sza, oh_peak):
    """Diurnal OH [molec/cm3] from the SZA parabola (Hanisco et al. 2001, Fig. 1).

    OH = oh_peak/OH_SZA_KNOTS[0][1] * (a*mu^2 + b*mu),  mu = max(cos(SZA), 0)

    a, b are the least-squares fit over OH_SZA_KNOTS with no constant term, so the
    curve passes through zero at the terminator by construction. Being LSQ it does
    NOT interpolate the knots (+4.0/-4.8/-7.2/+19.2% at SZA 0/30/45/60, exact 0 at
    90); see the note at OH_SZA_KNOTS for why that trade is deliberate.

    mu is CLAMPED to [0, 1] rather than clamping the output: a*mu^2 + b*mu has its
    second root at mu = -b/a = -0.652, so it goes positive again past SZA 130.7 deg
    and would emit OH at local midnight. Clamping mu makes the entire night exactly
    zero. cos_sza comes from calc_solar_zenith_angle, so this is a pure function of
    latitude, longitude and time of day; it carries no vertical structure.
    """
    mu = jnp.clip(cos_sza, 0.0, 1.0)          # night -> mu = 0 -> OH = 0 exactly
    return (oh_peak * _OH_SCALE) * (_OH_A * mu * mu + _OH_B * mu)


# how many OH samples oh_sza() returns per coupling step (substep centres).
# MICRO=full holds each sample over NSTEP_MICRO/OH_SUBSTEPS dev steps;
# MICRO_ENGINE=fast (tomas_fast.py) resamples from any count (_oh_to_substeps).
OH_SUBSTEPS = int(os.environ.get('OH_SUBSTEPS', '60'))


# ---- MICRO=full: tomas-jax dev make_step (2026-11-18) ------------------------
# Process list and options are dev's documented stratospheric setup
# (models/tomas-jax docs/stratosphere_setup.md, sensitivities/dunne_nuc/common.py):
# so2_chemistry -> nucleation (Riccobono+Dunne, rho_nuc 1770, org_frac 0) ->
# coagulation (euler_fixed) -> condensation (PPM, Tabazadeh H2SO4/H2O water),
# sparse_density 1770, max_nuc_substeps 20, max_nucleation_frac 0.5. Composition
# is sulfate + water with SO2/H2SO4 the only gases: org_conc and nh3_conc are not
# passed (dev defaults them to 0). The knobs below are how it is CALLED -- none
# changes dev's physics except MICRO_FION (a physical input) and MICRO_DT /
# MICRO_NCOAG (time discretisation; errors vs dt 60 / 10 substeps are in
# MANIFEST.md).
MICRO_DT = float(os.environ.get('MICRO_DT', '60'))          # dev step [s]
# forward-Euler coag sub-steps per dev step. dev's default is 10; 1 is within
# 5e-4 of it in total N (max over 20k real cells, 6 h) at 1/3 of the cost.
MICRO_NCOAG = int(os.environ.get('MICRO_NCOAG', '1'))
MICRO_COAG = os.environ.get('MICRO_COAG', 'fixed').lower()   # fixed | adaptive
if MICRO_COAG not in ('fixed', 'adaptive'):
    raise SystemExit(f"coupling: MICRO_COAG must be fixed|adaptive, got {MICRO_COAG!r}")
# ion-pair production [cm-3 s-1] for ion-induced nucleation. dev's strat setup
# uses 10; the user set ions to zero on 2026-11-18 (default 0 = neutral only).
MICRO_FION = float(os.environ.get('MICRO_FION', '0'))
# MNFIX over only the out-of-range bins (dev's own default). Bit-identical to
# trim=False and 5x faster here, although dev's docstring advises False under
# vmap (measured 2026-11-18: 153 vs 30 ms per 20k-cell coag step).
MICRO_MNFIX_TRIM = os.environ.get('MICRO_MNFIX_TRIM', '1') != '0'
# wall-clock budget for ONE step's microphysics [s]. A step that exceeds it
# aborts the run (the user's limit, 2026-11-18): a slow micro step means a
# misconfiguration, not something to wait out.
MICRO_MAX_WALL = float(os.environ.get('MICRO_MAX_WALL', '300'))
# GPUs the micro cells are spread over: 1 = this process's first device (one
# GPU per run, the default); N > 1 = the first N visible devices, one host
# thread each. dev's step carries jax.debug.callback warnings, which serialise
# a single dispatch thread across devices, hence the threads.
MICRO_GPUS = int(os.environ.get('MICRO_GPUS', '1'))
MW_AIR = 28.9644     # g/mol, for CESM vmr (mol/mol) -> mass mixing ratio [kg/kg]
# ---- radiation coupling (aerosol -> heating -> dT; see radiation.py) ----
RAD_ENABLE = os.environ.get('RAD', '1') != '0'
RAD_EVERY  = int(os.environ.get('RAD_EVERY', '1'))     # coupling steps between radiation calls
# 'anomaly': dT/dt = HR(evolved bins, T+dT) - HR(reference MAM4 bins, T_CESM)
# 'full'   : dT/dt = HR(evolved bins, T+dT)   (drifts toward radiative eq.;
#            only meaningful once a dynamical core balances it)
# 'flux'   : (2026-11-05, the no-feedback forcing mode) ONE solve per step with
#            the evolved bins at the meteorology's own temperature; dT_rad stays
#            0 and nothing feeds back. Fluxes and the absolute heating profile
#            go to frames3d; the forcing is a difference taken AFTER the run
#            against a paired run on the same meteorology (an injection run
#            minus its no-injection control = the SAI forcing).
RAD_MODE   = os.environ.get('RAD_MODE', 'anomaly')
if RAD_MODE not in ('anomaly', 'full', 'flux'):
    raise SystemExit(f"RAD_MODE={RAD_MODE!r} is not valid: anomaly | full | flux")
# RAD_T_FEEDBACK (2026-11-17; module docstring).
#   0 (default): dT_rad is diagnostic. The microphysics, the settling, the RH,
#     the OH number density and the radiation's aerosol optics (wt%, wet size,
#     refractive index) all see the meteorology's T_met; dT_rad enters ONLY the
#     emission temperature of the perturbed radiation call (T_met + dT_rad), so
#     its longwave response (the Planck damping) still bounds it. The heating
#     rate of every call is written out.
#   1: the earlier coupled behaviour -- T_met + dT_rad drives all of the above.
# RAD_MODE=flux has no dT_rad (it stays 0) and is unaffected.
RAD_T_FEEDBACK = os.environ.get('RAD_T_FEEDBACK', '0') != '0'
# ---- radiation engine / reference cache / A-B / sampler (2026-10-13) ----
# RAD_ENGINE  rrtmgp (the only engine in this repo). The radiation emulator
#             (RAD_ENGINE=emulator, RAD_REF A/B) is not part of this repo (yet);
#             it would fill the radiation slot behind this selector. The name and
#             the PHYS_CFG stamp are kept so the slot stays swappable and
#             checkpoints compatible.
# RAD_REF_CACHE  directory of the per-archive-hour cache of the reference-
#             aerosol solve (RAD_MODE=anomaly) and of the clear-sky fields the
#             emulator used (src/radiation/rad_cache.py). A hit is bit-identical
#             to computing the field (float64 on disk, same code), a miss computes
#             and stores it, so the default (on) changes no number -- only the
#             cost: the reference solve AND its MAM4 re-binning are skipped on a
#             hit (~5.6 of the 8.5 s stage). '0' disables.
# RAD_SAMPLE_EVERY  write rad_sample_<TAG>_s<step>.npz (evolved bins, dT_rad,
#             hour) every N radiation steps: training data for a radiation
#             emulator (not part of this repo). 0 = off.
RAD_ENGINE = os.environ.get('RAD_ENGINE', 'rrtmgp').lower()
RAD_REF    = os.environ.get('RAD_REF', '').lower()
RAD_REF_EVERY = int(os.environ.get('RAD_REF_EVERY', '1'))
RAD_REF_CACHE = os.environ.get('RAD_REF_CACHE',
                               os.path.join(_paths.INPUTS, 'cache', 'rad_ref'))
RAD_SAMPLE_EVERY = int(os.environ.get('RAD_SAMPLE_EVERY', '0'))
if RAD_ENGINE == 'emulator' or RAD_REF:
    raise SystemExit("RAD_ENGINE=emulator / RAD_REF: the radiation emulator is not part "
                     "of this repo")
if RAD_ENGINE != 'rrtmgp':
    raise SystemExit(f"RAD_ENGINE={RAD_ENGINE!r} is not valid: rrtmgp")
# ARF_toa is an INSTANTANEOUS TOA flux difference at the radiation call's own
# solar time. Its SW part therefore swings with the diurnal cycle, and with the
# default 6h step it is sampled 4x/day -- so a single sample's global mean
# depends on which longitudes happen to be sunlit and the reported forcing is
# phase-dependent (visibly a 4-per-day sawtooth of amplitude comparable to the
# signal itself). ARF_AVG_H sets the trailing window, in hours, over which the
# samples are averaged for the *reported* number: with 6h steps a 24h window is
# exactly 4 uniformly-spaced local times, i.e. a proper diurnal mean. The raw
# instantaneous value is still logged and stored as arf_toa. 0 disables.
ARF_AVG_H  = float(os.environ.get('ARF_AVG_H', '24'))
# ---- crash resume -----------------------------------------------------------
# Multi-hour runs on a shared GPU die (contention with other jobs; FAST_SORT's
# unchunked allocation is the usual trigger -- set FAST_SORT=0 if the card is
# loaded), and a death at step 300/360 threw away the compute even though the
# frames/timeseries ckpt files kept the data. STATE_CKPT=1 (default) additionally
# writes the FULL 3-D prognostic state to coupled_state_<tag>_ckpt.npz at the
# frame cadence -- ~400 MB, overwritten in place, written atomically via a .tmp
# + os.replace so a crash mid-write cannot corrupt the previous good checkpoint.
# RESUME=1 picks that file back up and continues from the step after it.
# The frames/timeseries ckpts are written in the SAME block, so all three are
# always from the same step and restore consistently.
STATE_CKPT = os.environ.get('STATE_CKPT', '1') != '0'
# CKPT_EVERY_FRAMES=N (2026-11-09): write the three _ckpt files only every N-th frame (and at the last step) instead
# of every frame. Output-only: the physics and the final coupled_* files are unchanged. Why: the frames ckpt holds the
# whole frame history (14.5 GB after one year, growing), so rewriting it every frame put ~15 GB per run per model day
# on /data NFS; six concurrent runs saturated it (write queue ~2.5 s/op) and the runs waited on it. A crash now loses
# at most N frames. Default 1 = the previous behaviour.
CKPT_EVERY_FRAMES = max(1, int(os.environ.get('CKPT_EVERY_FRAMES', '1')))
# CKPT_ASYNC (perf switch, default 1 since 2026-11-23; 0 = write the three _ckpt files synchronously, the old default): snapshot
# the checkpoint arrays to fresh host copies on the main thread, then do the np.savez work (zip + disk, ~0.9 GB
# state + the growing frames history per write) in ONE background writer thread while the next steps run.
# Output-only: the state the model evolves is never touched by the writer, and each file holds exactly what the
# synchronous path would have written for that step. The writer is joined before the next checkpoint is queued
# (so at most one is in flight and the frames/ts/state trio is still written in the same order, state LAST, each
# atomically via savez_atomic) and before the final coupled_* files; a writer error is re-raised on the main
# thread at that join. The thread is non-daemon, so an exception elsewhere still lets an in-flight write finish
# before the interpreter exits. Cost: one extra host copy of the 3-D state per checkpoint (memcpy, not disk).
CKPT_ASYNC = os.environ.get('CKPT_ASYNC', '1') != '0'
RESUME     = os.environ.get('RESUME', '0') != '0'
LOG_EVERY  = int(os.environ.get('LOG_EVERY', '1'))    # progress line every N steps
FRAME_EVERY = int(os.environ.get('FRAME_EVERY', '24'))  # save size-bin snapshot every N hours
# What the spatial frames COVER vertically. Independent of FRAME_EVERY, which
# sets how OFTEN they are written.
#   'probe' (default)  full lat-lon at the single PROBE_HPA level, per bin --
#                      what the filmstrip, the so4/dTrad gifs and the size-dist
#                      read. ~13 GB for a 366-frame year.
#   'all'              the above PLUS a ZONAL-MEAN frame at EVERY band level,
#                      per bin: (nf, NBINS, nlev, nlat). Unlocks a Hovmoller at
#                      any altitude and a lat-height animation; +1.1 GB/year.
# 'all' is a zonal MEAN and not a full 3-D history on purpose: per-bin 3-D
# frames are (nf, NBINS, nlev, nlat, nlon) = ~155 GB for a year, 12x the probe
# history, for a field nothing in this repo plots at full longitude resolution.
# It ADDS to the probe slabs rather than replacing them so every existing figure
# keeps reading exactly what it always read.
FRAME_LEVELS = os.environ.get('FRAME_LEVELS', 'probe').lower()
# FRAME3D=1 (2026-11-05): the FULL 3-D state and its per-cell size statistics
# after EVERY coupling step, one uncompressed frames3d/step_<s>.npz per step
# (~480 MB at 24 levels: 40 Nk + 40 Mk + SO2 + H2SO4 + dT_rad + 6 derived 3-D
# fields + the 2-D radiation fields; ~700 GB per model year). Independent of
# FRAME_EVERY/FRAME_LEVELS, which keep writing the probe/zonal frames. The
# campaign that asked for it: every cell, every time stamp, the bins, the
# gases, mean/sigma/effective diameter, total number and mass, AOD, dT and
# the TOA fluxes (absolute net, and the aerosol forcing vs the reference).
FRAME3D = os.environ.get('FRAME3D', '0') != '0'
# FRAME3D_EVERY=N (2026-11-18): write the frames3d file only every N-th step (and
# at the last step) instead of every step. Output-only. N coprime to 4 (e.g. 21)
# rotates the snapshot through all four 6 h UTC times, so a mean over the
# snapshots samples the diurnal cycle evenly; 21 steps = 5.25 days, ~33 GB/yr.
FRAME3D_EVERY = max(1, int(os.environ.get('FRAME3D_EVERY', '1')))
if FRAME_LEVELS not in ('probe', 'all'):
    raise SystemExit(f"FRAME_LEVELS={FRAME_LEVELS!r} is not valid: "
                     "use 'probe' (single PROBE_HPA level) or 'all' "
                     "(adds a zonal-mean frame at every band level)")
OUT_TAG    = os.environ.get('OUT_TAG', f'{N_DAYS}day')

# MAM4 mode geometric standard deviations (CESM MAM4 prescribed)
#https://gmd.copernicus.org/articles/5/709/2012/
# Default = physical MAM4 widths (accumulation, Aitken, coarse). INIT_SIGMA
# overrides as "s1,s2,s3" for modes 1/2/3 -- set INIT_SIGMA=1.6,1.8,1.6 to match
# the emulator's TRAINING binner (modes 1&2 swapped vs physical),
# which the axial operator was trained on and needs to see in-distribution states.
_sig = os.environ.get('INIT_SIGMA')
if _sig:
    _sv = [float(x) for x in _sig.split(',')]
    assert len(_sv) == 3, f"INIT_SIGMA must be 's1,s2,s3'; got {_sig!r}"
    MAM_SIGMA = {1: _sv[0], 2: _sv[1], 3: _sv[2]}
else:
    MAM_SIGMA = {1: 1.8, 2: 1.6, 3: 1.8}   # accumulation, Aitken, coarse
# Init binning method. 'dgnum' distributes each mode's number by its CESM wet
# number-median diameter dgnumwet (lognormal PDF at bin diameters, renormalized
# over the 40 bins) -- matches the emulator's training-data binning and seeds the
# small bins consistently. 'so4' is the legacy path that derives the median mass
# from so4/number, which under-fills the small bins (37% empty vs ~12% here; see
# the init-binning-mismatch diagnosis). Default 'so4' since 2026-07-29.
#
# WHY THE DEFAULT FLIPPED FROM 'dgnum' TO 'so4' (2026-07-29): 'dgnum' conserves
# MAM4's NUMBER but INVENTS its MASS -- bin_mam4 sets mas = num*MMID and never
# reads so4_a*, so every particle is booked as pure sulfate at RHO_AER. MAM4's
# coarse mode is mostly dust/sea salt with a sulfate COATING, so its whole volume
# became sulfate. Measured at t=0 over 1-150 hPa: total sulfate mass 4.29x MAM4
# truth (mode 3 alone 6.68x), which inflated Dp(massw) to ~1920 nm and -- because
# settling goes as D^2 -- created a spurious ~1%/day settling loss. 'so4' derives
# the median mass from so4/number: mass 1.020x, number 0.987x, Dp(massw) ~890 nm.
# The "37% empty" objection above does NOT hold in this band: measured per-cell
# empty-bin fraction is 3.2% for 'so4' vs 0.0% for 'dgnum'. Validated by the
# 18 h A/B pair smoke_newdefaults (dgnum) vs smoke_so4bin (so4).
# NB: this flip also reaches any EMULATOR driver (not part of this repo) that
# reads C._INIT_BIN, so it bins with 'so4' too. That moves the emulator off its
# dgnum training-data binning; if you go back to the emulator, set
# INIT_BIN=dgnum explicitly there (it is already out of distribution, so this
# is not the binding problem).
_INIT_BIN = os.environ.get('INIT_BIN', 'so4').lower()
# MAM4_BIN_DEVICE (perf switch, default 1 since 2026-11-23; 0 = the numpy path, the old default): bin_mode's
# (41, cells) erf part runs on the device as separate eager jax ops -- the same
# operations in the same order, so it should give the same bits -- instead of a
# numpy array sent to the GPU for erf alone and back (2 x ~430 MB round trip at
# full grid). Used every step by the open-BC slabs and the polar refresh.
MAM4_BIN_DEVICE = os.environ.get('MAM4_BIN_DEVICE', '1') != '0'
N_HOURS   = int(os.environ.get('N_HOURS', 24 * N_DAYS))   # override for smoke tests

# Aerosol IC / boundary-fill source: 'mam4' (default) bins CESM MAM4 modes onto
# the TOMAS grid at every coupling step (evolving reservoir). 'carma' projects a CARMA
# sulfate size distribution (PRSUL pure + MXAER mixed-group sulfate) onto the
# grid ONCE as a static reservoir (the CARMA run is a different epoch -- 1991
# pre-Pinatubo background -- so it is not time-matched to the CESM meteorology;
# it seeds a physically self-consistent, well-resolved distribution incl. a real
# nucleation mode where CARMA produced one). The flag governs the aerosol IC,
# the per-step top/bottom open-BC fill, the polar-cap refresh and the radiation
# anomaly reference. Gas phase (SO2/H2SO4) is ALWAYS forced by CESM regardless,
# so 'carma' means CARMA aerosol + CESM gas forcing.
AER_SRC = os.environ.get('AER_SRC', 'mam4').lower()
CARMA_FILE = os.environ.get('CARMA_FILE',
    '/data/CESM_sims/cesm2.2_CARMA16node_freerun_1wk_19910601_1deg/run/'
    'cesm2.2_CARMA16node_freerun_1wk_19910601_1deg.cam.h1.1991-06-01-01800.nc')
CARMA_FRAME = int(os.environ.get('CARMA_FRAME', '0'))   # time index into CARMA file (48 frames)
# Verified fixed density of the CARMA sulfate mass grid: single-particle sulfate
# mass = (4/3) pi r^3 * CARMA_RHO matches the file's mass/number to 4 sig figs,
# constant across all bins/groups/levels (RH-independent -> dry sulfate radius).
CARMA_RHO = float(os.environ.get('CARMA_RHO', '1923.0'))
# CARMA_SUBBIN=1 (default): remap each CARMA bin as the mass INTERVAL it really
# represents, spread over every TOMAS bin it overlaps (see _carma_remap_weights).
# CARMA_SUBBIN=0: legacy point-deposit -- the whole CARMA bin lands in the ONE
# TOMAS bin containing its centre mass. Because PRSUL steps 3.67x in mass while
# TOMAS steps 2x, consecutive CARMA bins land ~1.88 TOMAS bins apart and every
# other TOMAS bin is left EXACTLY empty (the 'comb' in the day-0 size
# distribution). Empty bins carry no condensation sink, which biases the
# nucleation/condensation split of injected sulfur for the first days of a run.
# Kept only to reproduce runs made before 2026-07-26.
CARMA_SUBBIN = os.environ.get('CARMA_SUBBIN', '1') != '0'

# N_BINS overrides the tomas-jax default resolution (40) for this run only --
# rebinds the local NBINS/XK names in this module; tomas_jax.core.config itself
# (and anything else importing it fresh, e.g. the microphysics emulator, which
# is shape-locked to 40 bins) is never touched.
N_BINS = int(os.environ.get('N_BINS', '0'))   # 0 = use tomas-jax default (NBINS=40)
if N_BINS > 0 and N_BINS != NBINS:
    # Keep the SAME physical size range as the default grid (XK0 .. XK0*2^40),
    # just coarser: solve doubling_factor so it spans that range in N_BINS steps
    # (doubling_factor**N_BINS == 2.0**NBINS) instead of truncating the top of
    # the distribution. E.g. N_BINS=20 -> doubling_factor=4.0 (mass-quadrupling).
    _dbl = 2.0 ** (NBINS / N_BINS)
    NBINS = N_BINS
    XK = tconfig.make_grid(nbins=NBINS, xk0=tconfig.XK0, doubling_factor=_dbl)
else:
    XK = xk_boundaries()                  # (41,) bin-boundary masses [kg]
XK_NP = np.asarray(XK)
MMID = np.sqrt(XK_NP[:-1] * XK_NP[1:])  # (40,) geometric-mean bin mass [kg]
# dev's stratospheric nucleation parameters (sensitivities/dunne_nuc/common.py
# NUC_PARAMS): Riccobono+Dunne defaults with the cluster density at the grid
# density and no organic fraction.
# (built with the step in _dev_step())
RHO_AER = 1770.0                        # fixed sulfate density [kg/m3] (mass<->size)
#volume of a sphere = (4/3) pi r^3 = (pi/6) D^3  ->  D = (6 V / pi)^(1/3)
DP_BIN  = np.cbrt(MMID / RHO_AER * 6.0/np.pi) * 1e9   # (40,) bin diameter [nm]


# =========================================================================
# MAM4 -> 40-bin TOMAS initialization
# =========================================================================
#error function (https://en.wikipedia.org/wiki/Normal_distribution)
def _erf(x):
    return np.asarray(jax.scipy.special.erf(jnp.asarray(x)))
# CDF equation (https://en.wikipedia.org/wiki/Normal_distribution)
def _phi(z):
    return 0.5 * (1.0 + _erf(z / np.sqrt(2.0)))

def bin_mode(num_m, so4_m, sigma_g):
    """Bin one MAM4 mode onto the TOMAS grid.

    num_m : number mixing ratio [#/kg], shape (G,)
    so4_m : sulfate mass m.r.    [kg/kg], shape (G,)
    Returns number m.r. per bin, shape (NBINS, G).
    """
    #number of grid cells (flattened)
    G = num_m.shape[0]
    out = np.zeros((NBINS, G))
    #standard deviation of mass is proportional to std of diameter cubes...using log rules we get the following
    s = 3.0 * np.log(sigma_g)                      # ln-mass shape (mass ~ D^3)
    valid = (num_m > 0) & (so4_m > 0)
    if not np.any(valid):
        return out
    #artimetic mean total mass/total number
    amm = np.where(valid, so4_m / np.maximum(num_m, 1e-300), 0.0)  # arith. mean particle mass [kg]
    #mass coming from the log-normal definition (https://www.rdocumentation.org/packages/stats/versions/3.6.2/topics/Lognormal)
    mg  = np.where(valid, amm / np.exp(0.5 * s * s), 1.0)          # number-median particle mass
    lnmg = np.log(np.maximum(mg, 1e-300))                          # (G,)
    lnxk = np.log(XK_NP)                                           # (41,)
    # arg[k, g] = (ln xk[k] - lnmg[g]) / s
    #z = (x-mu)/sigma in log space
    if MAM4_BIN_DEVICE:
        # same ops/order as the numpy branch below, each one eager jax op (no jit
        # fusion: fusing changes the last bit), without the host round trip
        arg = (jnp.asarray(lnxk)[:, None] - jnp.asarray(lnmg)[None, :]) / s   # (41, G)
        phi = 0.5 * (1.0 + jax.scipy.special.erf(arg / np.sqrt(2.0)))          # (41, G), = _phi(arg)
        frac = phi[1:, :] - phi[:-1, :]                                       # (40, G)
        out = np.array(frac * jnp.asarray(num_m)[None, :])                     # number m.r. per bin
    else:
        arg = (lnxk[:, None] - lnmg[None, :]) / s                      # (41, G)
        phi = _phi(arg)                                               # (41, G)
        #fraction of log-normal distribution that falls in each bin
        frac = phi[1:, :] - phi[:-1, :]                              # (40, G)
        #fraction of the mode's number mixing ratio that falls in each bin
        out = frac * num_m[None, :]                                   # number m.r. per bin
    #zero out all bins for any grid cell that had zero number or mass (so the binning is consistent)
    out[:, ~valid] = 0.0
    return out


def bin_mode_dgnum(num_m, dgn_m, sigma_g):
    """Bin one MAM4 mode onto TOMAS by its CESM wet number-median diameter.

    num_m : number mixing ratio [#/kg], shape (G,)
    dgn_m : wet number-median diameter dgnumwet [m], shape (G,)
    Distributes the mode's number by a lognormal PDF evaluated at each bin's
    diameter and renormalized over the 40 bins -- the same scheme as the
    emulator's training generator (distribute_lognormal_batched). A mode centered
    above the small bins still seeds their tail rather than zeroing them, unlike
    bin_mode which needs so4>0 and derives the size from so4/number. sigma_g is the
    canonical MAM4 mode width (MAM_SIGMA). Returns number m.r. per bin (NBINS,G).
    """
    G = num_m.shape[0]
    out = np.zeros((NBINS, G))
    valid = (num_m > 0) & np.isfinite(dgn_m) & (dgn_m > 0)
    if not np.any(valid):
        return out
    ln_dp  = np.log(DP_BIN * 1e-9)[:, None]                       # (40,1) bin diameters [m]
    ln_dgn = np.log(np.where(valid, dgn_m, 1e-12))[None, :]       # (1,G)
    #lognormal PDF at each bin diameter, in ln-diameter space
    pdf = np.exp(-0.5 * ((ln_dp - ln_dgn) / np.log(sigma_g)) ** 2)   # (40,G)
    #renormalize over the 40 bins so the mode's number is conserved into the grid
    pdf /= np.maximum(pdf.sum(axis=0, keepdims=True), 1e-30)
    out = pdf * num_m[None, :]
    out[:, ~valid] = 0.0
    return out


def bin_mam4(ds_by_var, t, levs, lat_idx=None):
    """Bin MAM4 modal aerosol at h1 hour-index t (relative to H0) onto TOMAS bins.

    levs    : native level indices to read
    lat_idx : optional latitude indices (default: all latitudes)
    Returns num (NBINS,nlev_s,nlat_s,nlon)  [#/kg]
            mas (NBINS,nlev_s,nlat_s,nlon)  [kg/kg]
    Used for the IC (all levels), the per-step open-BC slabs (edge levels) and
    the per-step polar-cap refresh (all levels, polar lats). num/mas always come
    from the same binning so each bin's (Nk, Mk) pair is physically consistent.
    """
    def read(var):
        da = ds_by_var[var][var].isel(time=H0 + t, lev=levs)
        if lat_idx is not None:
            da = da.isel(lat=lat_idx)
        return da.values
    num_bin, shape = None, None
    #iter through the MAM4 modes; distribute each mode's number onto the 40 bins
    #by its CESM wet diameter (dgnum, default) or the legacy so4/number size (so4)
    for m in (1, 2, 3):
        num_m = read(f'num_a{m}')
        shape = num_m.shape
        #flattens the spatial dimensions into one axis
        if _INIT_BIN == 'dgnum':
            dgn_m = read(f'dgnumwet{m}')                            # wet median diameter [m]
            b = bin_mode_dgnum(num_m.reshape(-1), dgn_m.reshape(-1), MAM_SIGMA[m])
        else:
            so4_m = read(f'so4_a{m}')
            b = bin_mode(num_m.reshape(-1), so4_m.reshape(-1), MAM_SIGMA[m])
        #accumulate bin contributions from all three aerosol modes in MAM
        num_bin = b if num_bin is None else num_bin + b
    num = num_bin.reshape(NBINS, *shape)
    # mass m.r. per bin = number m.r. * geometric-mean bin mass (constant density)....assumes all particles in the bin have the same mass
    # mass = num (#/kg_air)* mass_per_particle (kg/particle) = kg_particle/kg_air
    mas = num * MMID[:, None, None, None]
    #return number and mass mixing ratios per bin
    return num, mas


def read_gases(ds_gas, t, levs, lat_idx=None):
    """Read CESM SO2 and H2SO4 (mol/mol) at h1 hour-index t -> kg/kg.

    Same three roles as bin_mam4: the IC, the per-step open-BC slabs and the
    per-step polar refresh -- the gas phase is forced by CESM at exactly the
    same places the binned aerosol is, so the (gas, aerosol) boundary state
    stays mutually consistent.
    """
    def read(var):
        da = ds_gas[var][var].isel(time=H0 + t, lev=levs)
        if lat_idx is not None:
            da = da.isel(lat=lat_idx)
        return da.values
    # volume mixing ratio -> mass mixing ratio: multiply by MW_gas / MW_air
    so2   = read('SO2')   * (MW_SO2   / MW_AIR)
    h2so4 = read('H2SO4') * (MW_H2SO4 / MW_AIR)
    return so2, h2so4


# =========================================================================
# CARMA -> TOMAS initialization (alternative IC/BC source; AER_SRC=carma)
# =========================================================================
# CARMA bin-center RADII [um], read from the file's mass-tracer long_names and
# confirmed by the mass/number density check (rho = CARMA_RHO, constant). Two
# sulfate groups on different grids:
#   PRSUL (pure sulfate)      : Dp 0.69nm -> 2.59um  (the nucleation/fine mode)
#   MXAER (sulfate in mixed)  : Dp 100nm  -> 17.4um  (accumulation/coarse; the
#                               mixed particle also carries SOA/BC/dust/salt, but
#                               its bin radius is defined by SULFATE mass alone,
#                               so taking MXAER sulfate mass is size-consistent).
_PRSUL_R_UM = np.array([
    0.3430E-03, 0.5291E-03, 0.8161E-03, 0.1259E-02, 0.1942E-02, 0.2995E-02,
    0.4620E-02, 0.7126E-02, 0.1099E-01, 0.1695E-01, 0.2615E-01, 0.4034E-01,
    0.6222E-01, 0.9598E-01, 0.1480E+00, 0.2284E+00, 0.3522E+00, 0.5433E+00,
    0.8381E+00, 0.1293E+01])
_MXAER_R_UM = np.array([
    0.5000E-01, 0.6560E-01, 0.8608E-01, 0.1129E+00, 0.1482E+00, 0.1944E+00,
    0.2551E+00, 0.3347E+00, 0.4392E+00, 0.5762E+00, 0.7561E+00, 0.9920E+00,
    0.1302E+01, 0.1708E+01, 0.2241E+01, 0.2940E+01, 0.3858E+01, 0.5061E+01,
    0.6641E+01, 0.8714E+01])


def _subbin_alpha(s, mbar_over_g):
    """Exponent of the sub-bin shape dN/dlnm = A*m^alpha reproducing mean mass.

    Works in mass normalized by the bin's geometric-mean mass g=sqrt(a*b), so the
    interval is [1/s, s] with s=sqrt(b/a); this keeps m^alpha off the float range
    limits (raw bin masses are ~1e-24 kg, so m**alpha would under/overflow).

    On the interval the shape gives
        N = A*(s^a - s^-a)/a          M = A*g*(s^(a+1) - s^-(a+1))/(a+1)
    so mbar/g = [a/(a+1)]*(s^(a+1) - s^-(a+1))/(s^a - s^-a) depends on alpha
    alone and increases monotonically from 1/s (alpha->-inf) to s (alpha->+inf).
    Bisection is therefore safe for any mean strictly inside the bin. For a
    geometric grid the mean IS the geometric mean (mbar/g == 1) and the exact
    answer is -1/2; the solve is kept general so a non-geometric CARMA grid, or
    the clamped end bins, still get a consistent two-moment shape.
    """
    ln_s = np.log(s)

    def mbar(a):
        if abs(a) < 1e-9:                      # dN/dlnm flat: N=A*2*ln(s)
            return (s - 1.0 / s) / (2.0 * ln_s)
        if abs(a + 1.0) < 1e-9:                # dM/dlnm flat
            return 2.0 * ln_s / (s - 1.0 / s)
        return ((a / (a + 1.0)) * (s ** (a + 1.0) - s ** -(a + 1.0))
                / (s ** a - s ** -a))

    lo, hi = -50.0, 50.0                       # s^50 ~ 1e14 for s~1.9: safe
    if mbar_over_g <= mbar(lo):
        return lo
    if mbar_over_g >= mbar(hi):
        return hi
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if mbar(mid) < mbar_over_g:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def _carma_remap_weights(radii_um):
    """Two-moment conservative remap weights from one CARMA group onto TOMAS.

    Returns (w_num, w_mas), both (20, NBINS): the fraction of CARMA bin i's
    NUMBER and of its MASS that belongs in TOMAS bin k. Each CARMA bin is treated
    as the mass INTERVAL [a_i,b_i] it actually represents (geometric edges taken
    from the file's geometric radius grid) populated by the sub-bin shape above,
    rather than as a single point mass at the bin centre. Both weight rows sum to
    1, so number and mass are each conserved exactly; and because every receiving
    piece [u,v] lies inside the TOMAS bin's own bounds, the resulting mean
    particle mass is in-bounds by construction (no mnfix repair needed).

    Mass falling below XK[0] or above XK[-1] is snapped to the edge bin: its
    number is kept and its mass is set to that number at the edge-bin geometric
    mean, matching the legacy behaviour (the out-of-range tails are the sub-2nm
    PRSUL bins and the >6um MXAER bins -- negligible mass, non-negligible number).
    """
    r = np.asarray(radii_um, dtype=np.float64) * 1e-6                # [m]
    n_c = r.size
    ratio = (r[-1] / r[0]) ** (1.0 / (n_c - 1))                      # geometric grid
    half = np.sqrt(ratio)
    c = (4.0 / 3.0) * np.pi * CARMA_RHO
    a_all = c * (r / half) ** 3                                      # lower mass edge
    b_all = c * (r * half) ** 3                                      # upper mass edge
    mp_all = c * r ** 3                                              # bin-centre mass

    w_num = np.zeros((n_c, NBINS))
    w_mas = np.zeros((n_c, NBINS))
    for i in range(n_c):
        a, b, mp = a_all[i], b_all[i], mp_all[i]
        g = np.sqrt(a * b)
        s = np.sqrt(b / a)
        al = _subbin_alpha(s, mp / g)
        # normalized-mass cumulative moments, so the weights are ratios of
        # differences taken over [1/s, s]
        def _cum(x, p):                        # int of A*m^p dlnm, normalized
            return np.log(x) if abs(p) < 1e-9 else (x ** p) / p
        tot_n = _cum(s, al) - _cum(1.0 / s, al)
        tot_m = _cum(s, al + 1.0) - _cum(1.0 / s, al + 1.0)
        # the part of [a,b] inside the TOMAS range, split across its bins
        for k in range(NBINS):
            u = max(a, XK_NP[k])
            v = min(b, XK_NP[k + 1])
            if v <= u:
                continue
            w_num[i, k] += (_cum(v / g, al) - _cum(u / g, al)) / tot_n
            w_mas[i, k] += (_cum(v / g, al + 1.0)
                            - _cum(u / g, al + 1.0)) / tot_m
        # out-of-range tails -> edge bins, number-conserving
        if a < XK_NP[0]:
            v = min(b, XK_NP[0])
            f = (_cum(v / g, al) - _cum(a / g, al)) / tot_n
            w_num[i, 0] += f
            w_mas[i, 0] += f * MMID[0] / mp    # mass implied by the kept number
        if b > XK_NP[-1]:
            u = max(a, XK_NP[-1])
            f = (_cum(b / g, al) - _cum(u / g, al)) / tot_n
            w_num[i, -1] += f
            w_mas[i, -1] += f * MMID[-1] / mp
    return w_num, w_mas


def build_carma_reservoir(levs):
    """Project the CARMA sulfate distribution onto the TOMAS grid (static IC/BC).

    Reads the PRSUL (pure) + MXAER (mixed-group sulfate) mass mixing ratios
    [kg/kg] at CARMA_FRAME and native level indices `levs` (the CARMA file shares
    the FWHIST f09L70 grid exactly -- verified: identical lat/lon/lev/P0 -- so the
    coupling's klevs are valid CARMA indices with no interpolation). Two-moment
    remap: each CARMA bin i holds particles of single-particle sulfate mass
    mp_i = (4/3) pi r_i^3 * CARMA_RHO; its mass M_i is added to the TOMAS bin
    whose [XK[k], XK[k+1]] contains mp_i, and its number M_i/mp_i (#/kg) to the
    same bin -- conserving both number and mass, with mean particle mass landing
    inside the target bin's bounds by construction. CARMA bins below XK[0] (sub-
    ~2nm, the finest PRSUL) or above XK[-1] are snapped to bin 0 / bin NBINS-1 at
    that edge bin's geometric-mean mass (conserves number, negligible mass shift).

    Returns num (NBINS,nlev,nlat,nlon) [#/kg], mas (...) [kg/kg] -- same shapes,
    units and (num,mas) consistency as bin_mam4, so it is a drop-in reservoir.
    Gas phase is unaffected (still CESM-forced via read_gases).
    """
    ds = xr.open_dataset(CARMA_FILE, decode_times=False)
    # guard: the projection assumes the CARMA horizontal/vertical grid matches
    # the FWHIST run (indices == pressures). Cheap sanity check on level count.
    if ds.sizes['lev'] < max(levs) + 1:
        raise SystemExit(f"  ERROR: CARMA file has {ds.sizes['lev']} levels but "
                         f"coupling needs native index {max(levs)}.")
    num_c = mas_c = None
    for grp, radii in (('PRSUL', _PRSUL_R_UM), ('MXAER', _MXAER_R_UM)):
        mp = (4.0 / 3.0) * np.pi * (radii * 1e-6) ** 3 * CARMA_RHO   # (20,) kg/particle
        if CARMA_SUBBIN:
            w_num, w_mas = _carma_remap_weights(radii)
        for i in range(20):
            da = ds[f'{grp}{i + 1:02d}'].isel(time=CARMA_FRAME, lev=levs)
            m = np.nan_to_num(np.asarray(da.values, dtype=np.float64), nan=0.0)
            m = np.maximum(m, 0.0)                              # (nlev,nlat,nlon) kg/kg
            if num_c is None:
                num_c = np.zeros((NBINS,) + m.shape)
                mas_c = np.zeros((NBINS,) + m.shape)
            mp_i = float(mp[i])
            n_add = m / mp_i                                   # #/kg (true number)
            if CARMA_SUBBIN:
                # spread over every TOMAS bin this CARMA bin's mass range covers
                for k in np.nonzero(w_num[i] + w_mas[i])[0]:
                    num_c[k] += w_num[i, k] * n_add
                    mas_c[k] += w_mas[i, k] * m
            elif mp_i < XK_NP[0]:
                num_c[0] += n_add;  mas_c[0] += n_add * MMID[0]
            elif mp_i >= XK_NP[-1]:
                num_c[-1] += n_add; mas_c[-1] += n_add * MMID[-1]
            else:
                k = int(np.searchsorted(XK_NP, mp_i, side='right') - 1)
                num_c[k] += n_add;  mas_c[k] += m              # true mass in-bin
    ds.close()
    return num_c, mas_c


# =========================================================================
# Microphysics: per-cell coagulation, vmapped + chunked
# =========================================================================
NEPS_N = 1.0e-10   # #/box below this -> bin treated as empty

def _coag_cell(Nk, Mtot, temp, pres):
    # Number/mass are advected as independent tracers, so a bin can arrive with
    # Nk>0 but Mtot~0 (or vice versa). Enforce the physical constraint that a
    # bin's mean particle mass lies within its mass bounds [xk_k, xk_{k+1}];
    # this keeps Dp finite and >0 so the coag kernel is well posed.

    #zero out bins below this threshold
    Nk = jnp.where(Nk > NEPS_N, Nk, 0.0)
    #compute the lower and upper mass bounds for each bin, then clip the total mass to be within those bounds
    lo = Nk * XK[:-1]
    hi = Nk * XK[1:]
    Mtot = jnp.clip(Mtot, lo, hi)        # empty bins: lo=hi=0 -> Mtot=0
    #index 0 is the sulfate component, which is where we put all the mass (Mk[0] = Mtot). The other components are zeroed out.
    # for SAI this is a reasonable approximation since the particles are mostly sulfate. For other applications, this may need to be revisited.
    Mk = jnp.zeros((NBINS, ICOMP), dtype=jnp.float64).at[:, 0].set(Mtot)
    #tomas coagulation solver!! local temperature and pressure are used to compute the coagulation kernel
    Nk2, Mk2 = coag_euler_step(Nk, Mk, XK, temp, pres, BOXVOL,
                               dt=DT_MICRO, n_substeps=N_COAG_SUBSTEPS)
    #again, we want the first column of Mk2, which is the sulfate component
    return Nk2, Mk2[:, 0]

_coag_vmap = jax.jit(jax.vmap(_coag_cell, in_axes=(0, 0, 0, 0)))

# Microphysics is per-cell independent -> shard cells across all visible GPUs.
_DEVICES = jax.devices()
NDEV = len(_DEVICES)
_coag_pmap = jax.pmap(jax.vmap(_coag_cell, in_axes=(0, 0, 0, 0)))


# ---- full microphysics (MICRO=full): tomas-jax dev make_step ----------------
# One dev step = so2_chemistry -> nucleation -> coagulation -> condensation on
# MICRO_DT; a coupling step is NSTEP_MICRO of them in a fori_loop per cell, the
# cells vmapped, chunked by CELL_CHUNK and spread over MICRO_GPUS devices.
NSTEP_MICRO = int(round(DT_MICRO / MICRO_DT))
if abs(NSTEP_MICRO * MICRO_DT - DT_MICRO) > 1e-6 or NSTEP_MICRO < 1:
    raise SystemExit(f"coupling: MICRO_DT={MICRO_DT:g} s must divide the "
                     f"{DT_MICRO:g} s coupling step")
_DEV = {}


def _dev_step():
    """(step_fn, nucl_params) for MICRO=full, built on first use (see imports)."""
    if not _DEV:
        from tomas_jax.physics.nucleation import default_nucleation_params
        from tomas_jax.physics.water_equilibrium import WATER_SCHEME_H2SO4
        from tomas_jax.solvers.condensation import make_step
        _DEV['step'] = make_step(
            ['so2_chemistry', 'nucleation', 'coagulation', 'condensation'],
            n_coag_substeps=MICRO_NCOAG,
            coag_solver='euler_fixed' if MICRO_COAG == 'fixed' else 'euler_adaptive',
            max_nucleation_frac=0.5, max_nuc_substeps=20,
            water_scheme=WATER_SCHEME_H2SO4, sparse_density=1770.0,
            mnfix_trim=MICRO_MNFIX_TRIM, jit=False)
        _DEV['nuc'] = default_nucleation_params()._replace(rho_nuc=1770.0, org_frac=0.0)
    return _DEV['step'], _DEV['nuc']


def describe_full_chain():
    """One-line description of MICRO=full for the run header."""
    return (f"tomas-jax dev make_step [so2_chemistry, nucleation, coagulation, "
            f"condensation], {NSTEP_MICRO} x {MICRO_DT:g} s per step: "
            f"coag euler_{MICRO_COAG} ({MICRO_NCOAG} sub-steps), Riccobono+Dunne "
            f"(rho_nuc 1770, org_frac 0, fion {MICRO_FION:g}, no org/NH3), "
            f"PPM condensation (alpha {ALPHA_COND:g}), Tabazadeh H2SO4/H2O water, "
            f"sparse_density 1770, mnfix_trim {int(MICRO_MNFIX_TRIM)}; "
            f"{MICRO_GPUS} GPU(s), abort if one step's micro > {MICRO_MAX_WALL:g} s")


def _micro_cell(Nk, Mtot, so2, h2so4, temp, pres, rh, oh_samp):
    """MICRO=full for one cell (per-box units, boxvol 1 m3): NSTEP_MICRO dev steps.

    so2, h2so4 : gas [kg/box]   rh : [0-1]
    oh_samp    : (NOH,) OH [molec/cm3] samples evenly spaced over the step; dev
                 step i uses sample i*NOH//NSTEP_MICRO (NOH = 1: constant OH).
    Returns (Nk, dry Mtot, so2, h2so4).
    """
    # two-moment consistency clip (advected Nk/Mtot arrive independent): pin
    # each bin's mean particle mass into its bounds; run_microphysics_full
    # reports the mass this adds/removes
    Nk = jnp.where(Nk > NEPS_N, Nk, 0.0)
    Mtot = jnp.clip(Mtot, Nk * XK[:-1], Nk * XK[1:])
    Mk = jnp.zeros((NBINS, ICOMP), dtype=jnp.float64).at[:, SRTSO4].set(Mtot)
    # gas vector: the SRTSO4 slot holds H2SO4 vapour, SRTSO2 holds SO2
    Gc = jnp.zeros(N_GAS_SPECIES, dtype=jnp.float64)
    Gc = Gc.at[SRTSO4].set(h2so4).at[SRTSO2].set(so2)
    noh = oh_samp.shape[0]
    step, nuc = _dev_step()

    def body(i, c):
        oh = oh_samp[(i * noh) // NSTEP_MICRO]
        return step(*c, XK, temp, pres, BOXVOL, rh, ALPHA_COND, MICRO_DT,
                    oh_conc=oh, fion=MICRO_FION, nucl_params=nuc)
    Nk, Mk, Gc = jax.lax.fori_loop(0, NSTEP_MICRO, body, (Nk, Mk, Gc))
    # coupled state is DRY aerosol mass (water is diagnostic): every slot below
    # SRTH2O, which at this composition is the sulfate alone
    return Nk, Mk[:, :SRTH2O].sum(axis=1), Gc[SRTSO2], Gc[SRTSO4]


_micro_vmap = jax.jit(jax.vmap(_micro_cell, in_axes=(0,) * 8))


class MicroTooSlow(SystemExit):
    """Raised when one step's microphysics exceeds MICRO_MAX_WALL."""


def run_microphysics(num, mas, temp3d, pres3d, wgt3d):
    """Advance coagulation one coupling step (DT_MICRO) over the whole band.

    num, mas : (NBINS, nlev, nlat, nlon) mixing ratios
    temp3d   : (nlev,nlat,nlon) [K]      pres3d: (nlev,nlat,nlon) [Pa]
    wgt3d    : (nlev,nlat,nlon) burden weight (same as burdens()), used to
               report how much mass the two-moment consistency clip in
               _coag_cell adds/removes (it is NOT mass-conserving).
    Returns num2, mas2, clip_add, clip_rem  (clip_* in burden units, rem <= 0).
    """
    nbin, nlev, nlat, nlon = num.shape
    ncell = nlev * nlat * nlon
    rho = pres3d / (RD * temp3d)                       # kg/m3, (nlev,nlat,nlon)
    rho_f = np.asarray(rho).reshape(ncell)
    inv_rho = 1.0 / np.maximum(rho_f, 1e-30)
    T_f   = np.asarray(temp3d).reshape(ncell)
    P_f   = np.asarray(pres3d).reshape(ncell)

    # mixing ratio -> per-box (1 m3) concentration using density
    Nk_all = np.asarray(num).reshape(nbin, ncell).T * rho_f[:, None]   # (ncell,nbin)
    Mt_all = np.asarray(mas).reshape(nbin, ncell).T * rho_f[:, None]

    # ---- clip diagnostic: mirror _coag_cell's Nk floor + Mtot clip and
    # measure the mass it will add/remove, burden-weighted like burdens() ----
    Nk_eff = np.where(Nk_all > NEPS_N, Nk_all, 0.0)
    dM_mr = (np.clip(Mt_all, Nk_eff * XK_NP[None, :-1],
                     Nk_eff * XK_NP[None, 1:]) - Mt_all) * inv_rho[:, None]
    #burden weight
    w = np.asarray(wgt3d).reshape(ncell)[:, None]
    #global burden impact of the clip (additions and removals)
    clip_add = float((np.maximum(dM_mr, 0.0) * w).sum())
    clip_rem = float((np.minimum(dM_mr, 0.0) * w).sum())
    del Nk_eff, dM_mr
    #determined at load time...pmap or vmap used here for multiple or single GPU, respectively
    if NDEV > 1:
        # pad to a multiple of NDEV; pad cells are empty (Nk=Mt=0, safe T/p)
        pad = (-ncell) % NDEV
        if pad:
            Nk_all = np.concatenate([Nk_all, np.zeros((pad, nbin))], 0)
            Mt_all = np.concatenate([Mt_all, np.zeros((pad, nbin))], 0)
            T_f = np.concatenate([T_f, np.full(pad, 250.0)])
            P_f = np.concatenate([P_f, np.full(pad, 1.0e4)])
        per = (ncell + pad) // NDEV
        rs = lambda a: a.reshape(NDEV, per, *a.shape[1:])
        Nk_r, Mt_r, T_r, P_r = rs(Nk_all), rs(Mt_all), rs(T_f), rs(P_f)
        out_N_r = np.empty((NDEV, per, nbin)); out_M_r = np.empty((NDEV, per, nbin))
        #coagulate in parallel across all devices, each device gets a chunk of the cells
        # sub-chunk each device's cells by CELL_CHUNK so per-GPU peak memory stays
        # bounded (coag working set is ~tens of KB/cell); without this, a run with
        # only a few free GPUs would hand each one ncell/NDEV cells at once and OOM.
        for a in range(0, per, CELL_CHUNK):
            b = min(a + CELL_CHUNK, per)
            Nk2, Mt2 = _coag_pmap(Nk_r[:, a:b], Mt_r[:, a:b], T_r[:, a:b], P_r[:, a:b])
            out_N_r[:, a:b] = np.asarray(Nk2); out_M_r[:, a:b] = np.asarray(Mt2)
        out_N = out_N_r.reshape(ncell + pad, nbin)[:ncell]
        out_M = out_M_r.reshape(ncell + pad, nbin)[:ncell]
    else:
        out_N = np.empty_like(Nk_all)
        out_M = np.empty_like(Mt_all)
        for a in range(0, ncell, CELL_CHUNK):
            b = min(a + CELL_CHUNK, ncell)
            Nk2, Mt2 = _coag_vmap(jnp.asarray(Nk_all[a:b]), jnp.asarray(Mt_all[a:b]),
                                  jnp.asarray(T_f[a:b]),    jnp.asarray(P_f[a:b]))
            out_N[a:b] = np.asarray(Nk2)
            out_M[a:b] = np.asarray(Mt2)
    #convert back to mixing ratios using density, then reshape to (nbin,nlev,nlat,nlon)
    num2 = (out_N * inv_rho[:, None]).T.reshape(nbin, nlev, nlat, nlon)
    mas2 = (out_M * inv_rho[:, None]).T.reshape(nbin, nlev, nlat, nlon)
    return num2, mas2, clip_add, clip_rem


def _oh_to_substeps(oh_arr, ncell, nsub_out=None):
    """OH field -> (ncell, nsub_out) [molec/cm3], one column per substep.

    Accepts either (nlev,nlat,nlon) [one OH for the whole step, the CESM-field
    convention -> repeated across substeps] or (nsub_in,nlev,nlat,nlon) [the SZA
    diurnal path, nsub_in = OH_SUBSTEPS]. nsub_in need NOT equal nsub_out
    (default OH_SUBSTEPS): the OH sample count is set by whichever driver
    built the field, so resample onto the consumer's substep grid.
      nsub_in % nsub_out == 0  -> block-MEAN each group of fine samples. For a
        pseudo-first-order sink, exp(-k*<OH>*dt) over the coarse substep is the
        consistent reduction of the fine samples it spans.
      otherwise                -> nearest sample CENTER (the grids do not nest;
        picking is honest about that where averaging would fake a resolution).
    """
    nsub_out = OH_SUBSTEPS if nsub_out is None else nsub_out
    if oh_arr.ndim != 4:
        return np.repeat(oh_arr.reshape(ncell, 1), nsub_out, axis=1)
    nin = oh_arr.shape[0]
    oh_f = oh_arr.reshape(nin, ncell)          # copies if oh_arr is a bcast view
    if nin == nsub_out:
        return oh_f.T
    if nin % nsub_out == 0:
        return oh_f.reshape(nsub_out, nin // nsub_out, ncell).mean(axis=1).T
    c_in  = (np.arange(nin) + 0.5) / nin       # sample centers on [0,1]
    c_out = (np.arange(nsub_out) + 0.5) / nsub_out
    return oh_f[np.abs(c_in[None, :] - c_out[:, None]).argmin(axis=1)].T


def run_microphysics_full(num, mas, so2, h2so4, temp3d, pres3d, rh3d, oh3d,
                          wgt3d):
    """Advance MICRO=full (tomas-jax dev make_step) one coupling step.

    Same mixing-ratio <-> per-box conversion and two-moment clip diagnostic as
    run_microphysics, with the two gases carried through.

    num, mas   : (NBINS, nlev, nlat, nlon) mixing ratios
    so2, h2so4 : (nlev,nlat,nlon) gas mass mixing ratios [kg/kg]
    temp3d     : (nlev,nlat,nlon) [K]     pres3d: (nlev,nlat,nlon) [Pa]
    rh3d       : (nlev,nlat,nlon) [0-1]
    oh3d       : (nlev,nlat,nlon) or (NOH,nlev,nlat,nlon) [molec/cm3]
    wgt3d      : burden weight for the clip diagnostic (as run_microphysics)
    Returns num2, mas2, so2_2, h2so4_2, clip_add, clip_rem.
    Raises MicroTooSlow when the step's wall time passes MICRO_MAX_WALL -- checked
    after every chunk, so a hopeless step stops early rather than finishing.
    """
    t_start = time.time()
    nbin, nlev, nlat, nlon = num.shape
    ncell = nlev * nlat * nlon
    rho = pres3d / (RD * temp3d)                       # kg/m3, (nlev,nlat,nlon)
    rho_f = np.asarray(rho).reshape(ncell)
    inv_rho = 1.0 / np.maximum(rho_f, 1e-30)
    T_f  = np.asarray(temp3d, np.float64).reshape(ncell)
    P_f  = np.asarray(pres3d, np.float64).reshape(ncell)
    RH_f = np.asarray(rh3d, np.float64).reshape(ncell)
    oh = np.asarray(oh3d, np.float64)
    OH_f = (oh.reshape(ncell, 1) if oh.ndim == 3
            else np.ascontiguousarray(oh.reshape(oh.shape[0], ncell).T))   # (ncell, NOH)

    Nk_all = np.asarray(num).reshape(nbin, ncell).T * rho_f[:, None]   # (ncell,nbin)
    Mt_all = np.asarray(mas).reshape(nbin, ncell).T * rho_f[:, None]
    S_all  = np.asarray(so2).reshape(ncell)   * rho_f    # kg SO2 per box
    H_all  = np.asarray(h2so4).reshape(ncell) * rho_f    # kg H2SO4 per box

    # clip diagnostic: mirror _micro_cell's Nk floor + Mtot clip, burden-weighted
    Nk_eff = np.where(Nk_all > NEPS_N, Nk_all, 0.0)
    dM_mr = (np.clip(Mt_all, Nk_eff * XK_NP[None, :-1],
                     Nk_eff * XK_NP[None, 1:]) - Mt_all) * inv_rho[:, None]
    w = np.asarray(wgt3d).reshape(ncell)[:, None]
    clip_add = float((np.maximum(dM_mr, 0.0) * w).sum())
    clip_rem = float((np.minimum(dM_mr, 0.0) * w).sum())
    del Nk_eff, dM_mr

    out_N = np.empty_like(Nk_all); out_M = np.empty_like(Mt_all)
    out_S = np.empty_like(S_all);  out_H = np.empty_like(H_all)
    devs = _DEVICES[:max(1, min(MICRO_GPUS, NDEV))]
    # equal-size chunks (pad the last by repeating its final cell) so each
    # device compiles once and every call reuses it
    chunk = min(CELL_CHUNK, ncell)
    pieces = [(a, min(a + chunk, ncell)) for a in range(0, ncell, chunk)]
    ins = (Nk_all, Mt_all, S_all, H_all, T_f, P_f, RH_f, OH_f)
    slow = []

    def put(a, b, d):
        k = chunk - (b - a)
        return [jax.device_put(np.concatenate([x[a:b], np.repeat(x[b - 1:b], k, 0)])
                               if k else x[a:b], d) for x in ins]

    def worker(j):
        d = devs[j]
        for a, b in pieces[j::len(devs)]:
            if slow:
                return
            o = _micro_vmap(*put(a, b, d))
            n = b - a
            out_N[a:b] = np.asarray(o[0])[:n]; out_M[a:b] = np.asarray(o[1])[:n]
            out_S[a:b] = np.asarray(o[2])[:n]; out_H[a:b] = np.asarray(o[3])[:n]
            if os.environ.get('DEBUG'):
                print(f"  [dbg] micro chunk {b}/{ncell} on {d} "
                      f"({time.time() - t_start:.0f}s)", flush=True)
            if time.time() - t_start > MICRO_MAX_WALL:
                slow.append(b)
                return

    if len(devs) == 1:
        worker(0)
    else:
        from concurrent.futures import ThreadPoolExecutor
        with ThreadPoolExecutor(len(devs)) as ex:
            list(ex.map(worker, range(len(devs))))
    t_wall = time.time() - t_start
    if slow or t_wall > MICRO_MAX_WALL:
        raise MicroTooSlow(
            f"coupling: MICRO=full took {t_wall:.0f} s for this step, over the "
            f"MICRO_MAX_WALL={MICRO_MAX_WALL:g} s budget -- aborting the run "
            f"({describe_full_chain()}). A slow micro step means a configuration "
            f"to fix (MICRO_DT, MICRO_NCOAG, MICRO_COAG, MICRO_GPUS), not one to wait out.")
    run_microphysics_full.last_wall = t_wall

    num2 = (out_N * inv_rho[:, None]).T.reshape(nbin, nlev, nlat, nlon)
    mas2 = (out_M * inv_rho[:, None]).T.reshape(nbin, nlev, nlat, nlon)
    so2_2   = (out_S * inv_rho).reshape(nlev, nlat, nlon)
    h2so4_2 = (out_H * inv_rho).reshape(nlev, nlat, nlon)
    return num2, mas2, so2_2, h2so4_2, clip_add, clip_rem


# =========================================================================
# checkpoint I/O
# =========================================================================
def savez_atomic(path, **arrays):
    """np.savez to `path` via a temp file + os.replace, so an interrupted write
    leaves the PREVIOUS good checkpoint intact instead of a truncated one.

    Why this exists: a host stall inside a 5.9 GB frames-ckpt write, followed by
    a hard reset, truncated the good file in place at 90% -- frames_num cut off,
    frames_mas/dT/so2/h2so4 never on disk at all. np.savez writes straight to the
    destination, ext4 has no snapshots, there was no second copy, and RESUME=1 did
    an unconditional np.load on that file, so a 91.5%-complete 1-year run could
    not restart until the file was replaced by hand. The state ckpt came through
    untouched because it alone already used this pattern (see STATE_CKPT below).

    The exposure window is not small: the frames write is ~142 s of every
    ~19-minute step cycle AND the run's ~25 GB peak RSS, so it is the likeliest
    moment in the cycle to be caught by a stall or a kill. os.replace is atomic
    within a filesystem, so the worst case becomes losing the NEWEST frame.

    NB: the temp name must itself end in .npz -- np.savez appends '.npz' to any
    path that lacks it, so a '.npz.tmp' name would land at '.npz.tmp.npz' and the
    replace below would fail on a missing source.
    """
    assert path.endswith('.npz'), f"savez_atomic needs a .npz path, got {path}"
    tmp = path[:-4] + '.tmp.npz'
    np.savez(tmp, **arrays)
    os.replace(tmp, path)


class _CkptWriter:
    """Writes checkpoint batches [(path, arrays), ...] with savez_atomic, in
    order. CKPT_ASYNC=0: on the caller's thread, each batch entry built (its
    thunk called) only right before its own write, exactly as the inline code
    did. CKPT_ASYNC=1: every thunk is called on the caller's thread first (the
    snapshot), then the writes run in one background thread; submit() and
    join() wait for the previous batch and re-raise its error, if any."""
    def __init__(self, async_):
        self.async_ = async_
        self._th = None
        self._err = None
        self.t_wait = 0.0          # main-thread seconds spent waiting in join()

    def _write(self, batch):
        try:
            for path, arrays in batch:
                savez_atomic(path, **arrays)
        except BaseException as e:          # surfaced at the next join()
            self._err = e

    def join(self):
        if self._th is not None:
            tw = time.time()
            self._th.join()
            self.t_wait += time.time() - tw
            self._th = None
        if self._err is not None:
            e, self._err = self._err, None
            raise RuntimeError("CKPT_ASYNC: background checkpoint write failed") from e

    def submit(self, batch):
        """batch: list of (path, thunk) with thunk() -> dict of arrays."""
        if not self.async_:
            for path, thunk in batch:
                savez_atomic(path, **thunk())
            return
        self.join()
        snap = [(path, thunk()) for path, thunk in batch]
        import threading
        self._th = threading.Thread(target=self._write, args=(snap,),
                                    name='ckpt_writer', daemon=False)
        self._th.start()


# =========================================================================
# CESM I/O
# =========================================================================
def rad_ref_src():
    """The part of the radiation reference-cache key that describes the
    REFERENCE AEROSOL (aer_fill): source and binning. One function so the
    run (main) and scripts/radiation/build_ref_cache.py agree on the key."""
    d = dict(AER_SRC=AER_SRC)
    if AER_SRC == 'mam4':
        d.update(INIT_BIN=_INIT_BIN,
                 MAM_SIGMA=[float(MAM_SIGMA[m]) for m in (1, 2, 3)])
    else:
        d.update(CARMA_FILE=os.path.basename(CARMA_FILE), CARMA_FRAME=CARMA_FRAME,
                 CARMA_RHO=CARMA_RHO, CARMA_SUBBIN=CARMA_SUBBIN)
    # 2026-11-17: under BC_SRC=clim the reference solve's aerosol, gases and
    # surface are the climatology's, not the archive's, while the key's
    # 'archive' entry (the T file) is the same. Added ONLY for clim, so every
    # archive-BC key -- and the caches built under it -- is unchanged.
    if BC_SRC == 'clim':
        if not _BC_INSTALLED:
            raise SystemExit("rad_ref_src: BC_SRC=clim but open_var still reads the "
                             "archive -- call coupling.install_bc_source() first, or "
                             "archive-BC solves would be cached under the clim key")
        d.update(BC_SRC='clim', CLIM_FILE=os.path.basename(CLIM_FILE),
                 CLIM_BYTES=int(os.path.getsize(CLIM_FILE)) if os.path.exists(CLIM_FILE) else -1,
                 # size alone cannot tell a rebuild (fixed-shape arrays): the
                 # mtime does. A copy without -p only costs a cache miss.
                 CLIM_MTIME=int(os.stat(CLIM_FILE).st_mtime) if os.path.exists(CLIM_FILE) else -1)
    return d


def open_var(var):
    fn = f'{H1}.{var}{SUF_H}'
    # Say which knob is wrong, once, instead of letting xarray raise a bare
    # FileNotFoundError on a 150-character path -- on a fresh clone this is the
    # first thing that fails, and the cause is almost always CESM_DIR.
    if not os.path.exists(fn):
        raise SystemExit(
            f"coupling.py: CESM forcing file not found:\n    {fn}\n"
            f"  Built from CESM_DIR={HDIR}\n"
            f"              CESM_PREFIX={PREFIX}\n"
            f"              CESM_SUF={SUF_H}\n"
            "  This repo does not ship the meteorology (it is ~TB of CESM h1\n"
            "  output). Point CESM_DIR at an archive holding\n"
            f"  hour_1/<prefix>.h1.{var}<suffix>, or override the prefix/suffix\n"
            "  if your files are named differently. Variables needed: see\n"
            "  docs/COUPLING_VARIABLES.md.")
    print(f"  opening {fn.split('/')[-1]}", flush=True)
    return xr.open_dataset(fn)


_BC_INSTALLED = False


def install_bc_source():
    """Rebind the module's open_var to the configured boundary source (BC_SRC /
    MET_SRC). Idempotent; main() calls it, and so must any script that reads
    fields through C.open_var under BC_SRC=clim."""
    global open_var, _BC_INSTALLED
    if _BC_INSTALLED:
        return
    _BC_INSTALLED = True
    if BC_SRC == 'clim' and MET_SRC == 'file':
        import bc_clim
        _clim = bc_clim.ClimArchive(CLIM_FILE, MET_DIR)
        open_var = _clim.open_var
        print(f"=== BC_SRC=clim: no archive files; {_clim.describe()} ===", flush=True)
    elif BC_SRC == 'clim' and MET_JCM:
        # MET_SRC=jcm (2026-11-18, JCM grid support): the same per-variable
        # router, with the meteorology from a jcm-strat run directory instead of
        # the CESM archive. The met source also DEFINES the grid (lat, lon, the
        # hybrid levels) -- main() takes it from ds['lat'] etc. as always -- so the
        # climatology must be on the same grid and levels; refused otherwise.
        import bc_clim
        import met_jcm
        import datetime as _dt
        # the segments that hold H0 .. H0+N_HOURS (noleap -> calendar year); each
        # segment starts at 06Z on 1 January, so the year before H0 is indexed too
        _y0 = (bc_clim.ORIGIN + _dt.timedelta(hours=H0)).year
        _y1 = (bc_clim.ORIGIN + _dt.timedelta(hours=H0 + N_HOURS)).year
        if MET_SRC == 'jcm_online':
            import met_jcm_online
            _jcm = met_jcm_online.JcmOnlineMet(JCM_DIR, JCM_TEMPLATE, H0 + N_HOURS, JCM_WORK,
                                               jcm_strat=JCM_STRAT_PATH, python=JCM_PY,
                                               cache=JCM_CACHE, scratch=JCM_SCRATCH,
                                               te_correction=JCM_TE_CORRECTION)
        else:
            _jcm = met_jcm.JcmMet(JCM_DIR, list(range(_y0 - 1, _y1 + 1)))
        _clim = bc_clim.ClimArchive(CLIM_FILE, None, ntime=_jcm.ntime)
        for _k in ('lat', 'lon', 'hyam', 'hybm'):
            _a, _b = getattr(_jcm, _k), getattr(_clim, _k)
            if _a.shape != _b.shape or not np.allclose(_a, _b, rtol=1e-6, atol=1e-9):
                raise SystemExit(f"MET_SRC={MET_SRC}: CLIM_FILE {CLIM_FILE} is not on the JCM grid "
                                 f"({_k}: {_b.shape} vs {_a.shape}); build it with "
                                 f"scripts/met/regrid_clim_to_grid.py")
        _MET_VARS = ('U', 'V', 'OMEGA', 'T') + (('Q',) if MET_Q == 'met' else ())

        def open_var(var):
            return _jcm.open_var(var) if var in _MET_VARS else _clim.open_var(var)
        print(f"=== MET_SRC={MET_SRC} + BC_SRC=clim: {'/'.join(_MET_VARS)} from {_jcm.describe()}; "
              f"everything else (incl. {'' if MET_Q == 'met' else 'Q, '}O3, CH4, aerosol, gases, "
              f"surface) from the CESM {_clim.describe()} ===", flush=True)
    elif BC_SRC == 'clim':
        # MET_SRC=cesm + BC_SRC=clim (2026-11-17): the per-variable ROUTER. The
        # meteorology -- U/V/OMEGA/T, on the band and on the full radiation
        # column -- from the hourly archive; every other field from the
        # climatology (bc_clim.ClimArchive with no met file refuses U/V/OMEGA/T,
        # so its monthly-mean T can never leak in). A script that calls
        # C.open_var directly must call install_bc_source() first: rad_ref_src()
        # refuses a BC_SRC=clim key until it has, so archive-BC reference solves
        # can never be stored under the climatology's cache key.
        import bc_clim
        _clim = bc_clim.ClimArchive(CLIM_FILE, None)
        _archive_open = open_var
        _MET_VARS = ('U', 'V', 'OMEGA', 'T')

        def open_var(var):
            return _archive_open(var) if var in _MET_VARS else _clim.open_var(var)
        print(f"=== MET_SRC=cesm + BC_SRC=clim: U/V/OMEGA/T hourly from the CESM h1 archive; "
              f"everything else from the {_clim.describe()} ===", flush=True)
    else:
        print("=== opening CESM h1 datasets ===", flush=True)


def main():
    t0 = time.time()
    # Clear stale savez_atomic temp files from a previous crash. The truncated
    # bytes land in the .tmp (that is the point), but a killed run leaves one
    # behind -- up to ~6 GB for the frames ckpt, and it matches the
    # coupled_frames_*.npz glob that gif_run.py uses to list available TAGs, so
    # it would show up there as a phantom '<tag>_ckpt.tmp'. The next write would
    # overwrite it anyway (the temp name is deterministic); this is so the dead
    # space and the phantom tag do not sit around in between.
    for _stale in (f'coupled_frames_{OUT_TAG}_ckpt.tmp.npz',
                   f'coupled_timeseries_{OUT_TAG}_ckpt.tmp.npz',
                   f'coupled_state_{OUT_TAG}_ckpt.tmp.npz'):
        if os.path.exists(_stale):
            print(f"  removing stale temp checkpoint {_stale} "
                  f"({os.path.getsize(_stale)/1e9:.2f} GB) from an earlier crash",
                  flush=True)
            os.remove(_stale)
    install_bc_source()
    dU = open_var('U'); dV = open_var('V'); dW = open_var('OMEGA')
    dT = open_var('T')
    ds_mam = {f'{p}_a{m}': open_var(f'{p}_a{m}')
              for p in ('num', 'so4') for m in (1, 2, 3)}
    if _INIT_BIN == 'dgnum':
        ds_mam.update({f'dgnumwet{m}': open_var(f'dgnumwet{m}') for m in (1, 2, 3)})
    # gas-phase datasets (MICRO=full): SO2/H2SO4 are the advected tracers'
    # CESM counterparts (ICs + BCs), OH drives the SO2 oxidation; RELHUM is
    # read only under MET_RH=cesm (the RH block below replaces it otherwise)
    ds_gas = {v: open_var(v) for v in ('SO2', 'H2SO4')}
    dOH = open_var('OH'); dRH = open_var('RELHUM')

    lat = dU['lat'].values; lon = dU['lon'].values
    nlat, nlon = lat.size, lon.size

    # ---- stratospheric band (reference-pressure levels, fct convention) ----
    plev = dU['hyam'].values * dU['P0'].values + dU['hybm'].values * PS_REF   # Pa
    band = np.where((plev >= P_LO_HPA * 100) & (plev <= P_HI_HPA * 100))[0]
    full = list(range(band.min(), band.max() + 1))
    if N_LEV and N_LEV < len(full):
        # sub-sample the band to ~N_LEV native levels, evenly spaced by index
        sel = np.unique(np.round(np.linspace(0, len(full) - 1, N_LEV)).astype(int))
        klevs = [full[i] for i in sel]
    else:
        klevs = full
    PLEV_PA = plev[klevs]
    DP = np.gradient(PLEV_PA)                        # Pa, for vertical advection & weighting
    nlev = len(klevs)
    print(f"  grid {nlat}x{nlon}, {nlev} levels "
          f"({PLEV_PA[0]/100:.1f}..{PLEV_PA[-1]/100:.1f} hPa), "
          f"micro on {NDEV} GPU(s)", flush=True)
    # GRID_CFG (2026-11-18, JCM grid support): the met source and the grid
    # it defines, stamped into the state checkpoint and REFUSED on a RESUME
    # mismatch (unlike PHYS_CFG, which only warns): a state on one grid is not
    # a state on another, and two grids can share (nlat, nlon, nlev) -- the
    # geometry tuple alone would not see it. [met code (0 cesm, 1 file, 2 jcm, 3 jcm_online),
    # nlat, nlon, nlev, first lat, last lat, top hPa, bottom hPa]. Append-only.
    GRID_CFG = np.array([float({'cesm': 0, 'file': 1, 'jcm': 2, 'jcm_online': 3}[MET_SRC]),
                         float(nlat), float(nlon), float(nlev),
                         float(lat[0]), float(lat[-1]),
                         float(PLEV_PA[0] / 100), float(PLEV_PA[-1] / 100)])
    GRID_CFG_KEYS = np.array(['MET_SRC', 'nlat', 'nlon', 'nlev', 'lat0', 'lat_last',
                              'top_hPa', 'bottom_hPa'])
    print(f"  GRID_CFG: " + ', '.join(f'{k}={v:.6g}' for k, v in zip(GRID_CFG_KEYS, GRID_CFG)),
          flush=True)
    ntime = dU.sizes['time']
    print(f"  hourly time length: {ntime}, running {N_HOURS} h from H0={H0}",
          flush=True)
    # hour h reads winds/MAM4 at index H0+h+1 (h up to N_HOURS-1 => max index
    # H0+N_HOURS). Fail early with a clear message instead of an opaque
    # out-of-bounds partway through a long run.
    if H0 + N_HOURS > ntime - 1:
        raise SystemExit(
            f"  ERROR: run needs time index up to {H0 + N_HOURS} but the h1 "
            f"series has only {ntime} steps (max index {ntime - 1}). "
            f"Reduce N_HOURS/N_DAYS or H0.")

    pres3d = np.broadcast_to(PLEV_PA[:, None, None], (nlev, nlat, nlon)).copy()

    # ---- radiation driver + prognostic radiative temperature increment ----
    rad = None
    if RAD_ENABLE:
        import radiation
        print("=== building radiation driver (rrtmgp + Mie tables) ===", flush=True)
        # MMID/RHO_AER are passed so the driver can size the WET H2SO4/H2O
        # droplet from the dry SO4 bin mass, rather than treating DP_BIN (a dry
        # diameter) as the optical size. DP_BIN is still passed as the dry
        # reference the growth factor is reported against.
        # solar_window_h: the interval one radiation call's heating is held
        # for (dT_rad += hr * STEP_SEC * RAD_EVERY), which RAD_SOLAR=window
        # averages the sun over, centred on the call hour it1
        rad = radiation.RadiationDriver(open_var, klevs, lat, lon,
                                        DP_BIN * 1e-9,
                                        mmid_kg=MMID, rho_dry=RHO_AER,
                                        solar_window_h=STEP_HOURS * RAD_EVERY,
                                        split_only=(RAD_MODE == 'flux'))
        if RAD_REF_CACHE not in ('0', ''):
            import rad_cache
            rad.ref_cache = rad_cache.RefCache(RAD_REF_CACHE, rad.cache_key(**rad_ref_src()))
            print(f"  reference cache: {rad.ref_cache.dir}", flush=True)
        print(f"  RAD_ENGINE={RAD_ENGINE} RAD_REF={RAD_REF or '-'} "
              f"RAD_REF_EVERY={RAD_REF_EVERY} RAD_REF_CACHE={RAD_REF_CACHE} "
              f"RAD_SAMPLE_EVERY={RAD_SAMPLE_EVERY} "
              f"RAD_SOLAR={rad.solar_mode} (window {rad.solar_window_h:g} h"
              + (f", RAD_SOLAR_ALPHA={rad.alpha:g}" if rad.solar_mode == 'window_blend' else "")
              + (f", RAD_SOLAR_NODES={rad.nodes}, RAD_SOLAR_ALIGN={rad.align}"
                 if rad.solar_mode == 'window_gq' else "")
              + ")", flush=True)
        print(f"  RAD_MODE={RAD_MODE} RAD_T_FEEDBACK={int(RAD_T_FEEDBACK)}: "
              + ("T_met + dT_rad drives microphysics, settling, RH and the optics "
                 "(closed loop)" if RAD_T_FEEDBACK else
                 "T_met drives microphysics, settling, RH and the optics; dT_rad and "
                 "hr_kday are diagnostics (dT_rad enters only the perturbed call's "
                 "emission temperature)"), flush=True)
    #change in temperature from radiation intialized here    
    dT_rad = jnp.zeros((nlev, nlat, nlon), dtype=jnp.float64)
    # Heating rate of the LAST radiation call [K/day], held between calls (it
    # is what dT_rad was advanced by). Pure diagnostic: nothing reads it back.
    # Anomaly mode: the aerosol's heating anomaly hr1-hr0; flux/full: the
    # absolute rate of the evolved call.
    hr_kday = np.zeros((nlev, nlat, nlon), dtype=np.float64)

    def winds(t):
        u = dU['U'].isel(time=H0 + t, lev=klevs).values
        v = dV['V'].isel(time=H0 + t, lev=klevs).values
        w = dW['OMEGA'].isel(time=H0 + t, lev=klevs).values
        return u, v, w

    def winds_block(t0, t1):
        """winds(t) for t0..t1 inclusive as one slab read per variable: the h1
        files hold one time step per chunk, so this only saves the per-call
        overhead (~25 %: 1.5 s vs 2.0 s for 7 hours), not the decompression.
        Returns a list of (u, v, w) tuples, hour by hour. SL path only."""
        blk = [d[v].isel(time=slice(H0 + t0, H0 + t1 + 1), lev=klevs).values
               for d, v in ((dU, 'U'), (dV, 'V'), (dW, 'OMEGA'))]
        return [tuple(b[i] for b in blk) for i in range(t1 - t0 + 1)]

    def temp(t):
        return dT['T'].isel(time=H0 + t, lev=klevs).values

    def oh_molec(t, T3d):
        """CESM OH (mol/mol) -> molec/cm3: n = vmr * p/(kB T) * 1e-6.

        Takes the already-read temperature so a step does one T read, not two."""
        vmr = dOH['OH'].isel(time=H0 + t, lev=klevs).values
        return vmr * pres3d / (KB * T3d) * 1.0e-6

    def relhum(t):
        """CESM RELHUM (percent) -> fraction, clipped to [0,1]."""
        return np.clip(dRH['RELHUM'].isel(time=H0 + t, lev=klevs).values / 100.0,
                       0.0, 1.0)

    if MET_SRC == 'file':
        # The four meteorology readers above, re-pointed at MET_DIR (see the
        # MET_SRC note at the top). The file is checked against THIS run's grid
        # and band, and every hour the loop asks for must be one of its frames.
        import json
        with open(os.path.join(MET_DIR, 'meta.json')) as _f:
            met_meta = json.load(_f)
        met = {k: np.load(os.path.join(MET_DIR, f'{k}.npy'), mmap_mode='r')
               for k in ('u', 'v', 'omega', 'T')}
        _mk = [int(k) for k in met_meta['klevs']]
        if _mk != [int(k) for k in klevs]:
            raise SystemExit(f"MET_DIR band levels {_mk[0]}..{_mk[-1]} != this run's "
                             f"{klevs[0]}..{klevs[-1]}: rebuild it with the same P_LO_HPA/P_HI_HPA")
        if not (np.allclose(met_meta['lat'], lat) and np.allclose(met_meta['lon'], lon)):
            raise SystemExit("MET_DIR lat/lon differ from the archive grid")
        MET_H0 = int(met_meta['hour0']); MET_STEP = int(met_meta['step_hours'])
        MET_N = int(met_meta['nframe'])
        if MET_STEP != STEP_HOURS:
            raise SystemExit(f"MET_DIR step {MET_STEP} h != STEP_HOURS {STEP_HOURS}")
        if (H0 - MET_H0) % MET_STEP:
            raise SystemExit(f"H0={H0} is not on the MET file's {MET_STEP}-h grid (hour0 {MET_H0})")
        _need = (H0 + N_HOURS - MET_H0) // MET_STEP
        if H0 < MET_H0 or _need >= MET_N:
            raise SystemExit(f"run needs MET frames {(H0-MET_H0)//MET_STEP}..{_need} but the file "
                             f"holds 0..{MET_N-1} (hour0 {MET_H0}); shorten N_HOURS or extend the file")

        def met_frame(t):
            h = H0 + t
            if (h - MET_H0) % MET_STEP:
                raise SystemExit(f"hour {h} is not on the MET file's {MET_STEP}-h grid")
            i = (h - MET_H0) // MET_STEP
            if not 0 <= i < MET_N:
                raise SystemExit(f"hour {h} = MET frame {i}, outside 0..{MET_N-1}")
            return i

        def winds(t):
            i = met_frame(t)
            return (np.array(met['u'][i]), np.array(met['v'][i]), np.array(met['omega'][i]))

        def winds_block(t0, t1):
            raise RuntimeError('hourly winds are not available under MET_SRC=file')

        def temp(t):
            return np.array(met['T'][met_frame(t)])

        print(f"  MET_SRC=file: {MET_DIR}\n"
              f"    {MET_N} frames x {MET_STEP} h from archive hour {MET_H0} "
              f"({met_meta.get('t0')}), source {met_meta.get('source')}; U/V/OMEGA/T from the file; "
              f"gases, reservoirs, surface and the radiation column below the band from "
              f"{'the climatology' if BC_SRC == 'clim' else 'the archive'}",
              flush=True)

    # ---- RH (MET_RH; 2026-11-17: 'liq' default, any MET_SRC) -----------------
    # q from the BC source (archive hourly, or the climatology), T from the
    # meteorology (temp(t) as finally bound above: archive or met file) and the
    # level pressure, every step. T_met under RAD_T_FEEDBACK=0 (nothing but the
    # radiation's emission temperature sees dT_rad); T_met + dT_rad under 1, the
    # same T the microphysics, settling and the optics' own RH then use.
    if MET_RH in ('liq', 'q'):
        dQ = open_var('Q')

        def relhum(t):
            q = dQ['Q'].isel(time=H0 + t, lev=klevs).values.astype(np.float64)
            T = temp(t).astype(np.float64)
            if RAD_T_FEEDBACK:        # closed loop: RH at the T the run uses
                T = T + np.asarray(dT_rad, np.float64)
            if MET_RH == 'liq':
                return np.clip(settling.rh_liquid_pct(T, q, pres3d) / 100.0, 0.0, 1.0)
            return np.clip(q / qsat_ice(T, pres3d), 0.0, 1.0)
    print(f"  RH (MET_RH={MET_RH}): "
          + {'liq': 'over LIQUID water from Q, T_met and p every step (Tabazadeh eq. 1; '
                    'settling.rh_liquid_pct, the radiation optics\' formula)',
             'q': 'over ICE from Q, T_met and p (Goff-Gratch; legacy)',
             'cesm': 'the RELHUM field as is (over ice, at its own T; legacy)'}[MET_RH]
          + f"; Q from {('JCM (MET_Q=met)' if MET_Q == 'met' else 'the CESM climatology (MET_Q=bc)') if MET_JCM else 'the climatology' if BC_SRC == 'clim' else 'the archive'}",
          flush=True)

    # SZA-parabola OH grids (built once; SZA is lat/lon/time only, not level)
    _lat2d = jnp.asarray(lat)[:, None]                 # (nlat,1)
    _lon2d = jnp.asarray(lon)[None, :]                 # (1,nlon)
    _oh_peak_j = jnp.asarray(OH_PEAK)

    def oh_sza(t):
        """Per-substep diurnal OH [molec/cm3] from the SZA curve (Hanisco
        et al. 2001, Fig. 1). Evaluated at each grid point's local solar zenith
        angle at the CENTER of each substep, so SO2+OH oxidation resolves the
        diurnal cycle within the STEP_HOURS step. SZA is level-independent,
        so the (nlat,nlon) field is broadcast across the band.
        Returns (OH_SUBSTEPS, nlev, nlat, nlon).

        OH_SUBSTEPS is read from the module (not captured) so a driver can set
        C.OH_SUBSTEPS to ITS inner-step count before the loop starts: the
        default 60 serves MICRO=full and the adapters alike. The level axis is a broadcast VIEW, not a copy --
        at 60 samples a materialized (60,11,192,288) float64 is 292 MB of the
        same field repeated 11 times, and both consumers reshape (hence copy)
        only the layout they actually need."""
        nsub = OH_SUBSTEPS
        dt_h = (DT_MICRO / nsub) / 3600.0              # substep length [h]
        oh = np.empty((nsub, nlat, nlon))
        for m in range(nsub):
            hr = H0 + t + (m + 0.5) * dt_h             # abs hour, substep center
            doy = float((int(hr // 24) % 365) + 1)     # day-of-year (noleap)
            hour_utc = float(hr % 24.0)
            cs = calc_solar_zenith_angle(_lat2d, jnp.asarray(doy),
                                         jnp.asarray(hour_utc), _lon2d)  # (nlat,nlon)
            oh[m] = np.asarray(calc_oh_parabola(cs, _oh_peak_j))         # (nlat,nlon)
        return np.broadcast_to(oh[:, None, :, :], (nsub, nlev, nlat, nlon))

    # ---- aerosol IC / boundary-fill source (MAM4 evolving, or CARMA static) ----
    # aer_fill(t, levs, lat_idx) returns (num[#/kg], mas[kg/kg]) with the SAME
    # signature and (num,mas) consistency as bin_mam4, so the IC / open-BC /
    # polar / radiation-reference call sites are source-agnostic. MAM4 re-bins
    # per step (t used); CARMA is a static reservoir built once (t ignored).
    if AER_SRC == 'carma':
        print(f"=== initializing size bins from CARMA (frame {CARMA_FRAME}, "
              f"rho={CARMA_RHO:.0f}) ===", flush=True)
        print(f"  projecting PRSUL+MXAER from {CARMA_FILE.split('/')[-1]}", flush=True)
        _num_c, _mas_c = build_carma_reservoir(klevs)          # (NBINS,nlev,nlat,nlon)
        _klev_pos = {lv: i for i, lv in enumerate(klevs)}
        def aer_fill(t, levs, lat_idx=None):
            pos = [_klev_pos[lv] for lv in levs]
            n = _num_c[:, pos]; m = _mas_c[:, pos]
            if lat_idx is not None:
                n = n[:, :, lat_idx]; m = m[:, :, lat_idx]
            return n, m
    elif AER_SRC == 'mam4':
        print("=== initializing size bins from MAM4 ===", flush=True)
        if _INIT_BIN != 'dgnum':
            print(f"  MAM4_BIN_DEVICE={int(MAM4_BIN_DEVICE)}: bin_mode erf part "
                  f"{'on the device (eager jax ops)' if MAM4_BIN_DEVICE else 'in numpy (erf round trip to the device)'}",
                  flush=True)
        def aer_fill(t, levs, lat_idx=None):
            return bin_mam4(ds_mam, t, levs, lat_idx)
    else:
        raise SystemExit(f"  ERROR: AER_SRC must be 'mam4' or 'carma', got {AER_SRC!r}")
    # the radiation anomaly reference is ALWAYS the binned archive aerosol
    # (radiation.py integrates per-bin Mie tables); the prognostic state may be
    # something else (e.g. a modal scheme, via AER_FILL_FN)
    aer_fill_rad = aer_fill
    if AER_FILL_FN is not None:
        def aer_fill(t, levs, lat_idx=None):
            return AER_FILL_FN(ds_mam, t, levs, lat_idx)
    #bin aerosol onto TOMAS grid (source per AER_SRC)
    num, mas = aer_fill(0, klevs)
    num = jnp.asarray(num, dtype=jnp.float64)
    mas = jnp.asarray(mas, dtype=jnp.float64)
    # gas ICs from CESM's own SO2/H2SO4 fields (kg/kg), advected alongside
    so2_g, h2so4_g = read_gases(ds_gas, 0, klevs)
    so2   = jnp.asarray(so2_g,   dtype=jnp.float64)
    h2so4 = jnp.asarray(h2so4_g, dtype=jnp.float64)
    NA = AER_NROWS        # aerosol rows per moment (40 bins, or 4 modes)
    ntr = 2 * NA + 2      # 2*NA aerosol rows + SO2 + H2SO4
    print(f"  bins={NBINS}, aerosol rows/moment={NA}, advected tracers={ntr}, "
          f"init N burden(sum num)={float(num.sum()):.3e}, "
          f"M burden(sum mas)={float(mas.sum()):.3e}", flush=True)

    # polar freeze target; starts at the IC and is refreshed from MAM4 at the
    # end of each step (so a step advects against its own end-hour polar values)
    # (gases ride along in the same stack: rows 2*NBINS and 2*NBINS+1)
    qfroz = jnp.concatenate([num, mas, so2[None], h2so4[None]], axis=0)  # (ntr,nlev,nlat,nlon)
    # apply the bottom-face aerosol inflow scaling to step 0 as well (the per-step
    # refresh below only takes effect from the end of step 0 onward)
    if ADV_WCONT and BC_BOT_AER != 1.0:
        qfroz = qfroz.at[:2*NA, -N_BC_BOT:].multiply(BC_BOT_AER)
    pol_idx = np.where(np.abs(lat) > LAT_FREEZE)[0]

    # open-BC slab levels (native indices into the h1 lev axis)
    lev_top = klevs[:N_BC_TOP]
    lev_bot = klevs[-N_BC_BOT:]

    # Can the active advection driver report the vertical face exchange? The fast
    # drivers monkeypatch advect_step_batch, so ask the object we actually hold
    # rather than assuming. fct_lr (the default since 2026-08-03, and what
    # driver_fast.py rebinds to) can; the legacy sealed-face fct_core cannot, so
    # this answers False only if something restores that import by hand.
    # NB inspect.signature sees through functools.partial: a keyword bound by the
    # partial stays in .parameters (with the bound value as its default), so the
    # probe still finds 'return_vflux' through the wrapper applied at import.
    import inspect
    try:
        ADV_VFLUX = ('return_vflux' in
                     inspect.signature(advect_step_batch).parameters) and ADV_WCONT
    except (TypeError, ValueError):
        ADV_VFLUX = False
    # With ADV_WCONT the vertical faces are a real FLUX boundary (inflow at the
    # reservoir concentration, free outflow), so a Dirichlet overwrite of the edge
    # levels on top of that would double-count the boundary AND go back to
    # discarding whatever transport delivers there. Default to 'open' in that case;
    # 'clamp' stays the default for the legacy sealed-face scheme.
    _bc_edge_default0 = 'open' if ADV_WCONT else 'clamp'
    # Resolve the aerosol edge BC ONCE here. It used to be re-read from the
    # environment at each of its three use sites, which is how the gas default
    # below came to be derived from the DEFAULT rather than from the setting
    # actually in force.
    _bc_edge0 = os.environ.get('BC_EDGE', _bc_edge_default0).lower()
    # GASES GET THE SAME BOUNDARY AS THE AEROSOL. Changed 2026-07-30 at the user's
    # explicit instruction, in both trees. This is a bug fix, not a preference. Only
    # three of the four aerosol/gas boundary combinations are coherent:
    #     clamp/clamp  -- Dirichlet sub-domain, consistent
    #     flux /flux   -- open sub-domain, consistent
    #     flux /clamp  -- INCOHERENT: an unbounded gas SOURCE at a level whose
    #                     particles are free to leave, i.e. nucleation with no sink
    #     clamp/flux   -- (never used)
    # The old default was the third one, and only because the BC_EDGE=open migration
    # (added to fix the frozen-reservoir mass leak) never revisited the gas branch.
    # The pathology was measured and written into the comment at the BC_GAS site
    # WITHOUT the default being changed: at the 13.3 hPa top level over 24 h that one
    # level went from 0.3% to ~50% of the model's TOTAL number, as a 6-8 nm mode --
    # continuous nucleation fed by clamped H2SO4, not transport.
    # Deriving it from the RESOLVED _bc_edge0 (not from _bc_edge_default0, which is
    # what this line read until 2026-07-30) means the two boundaries cannot desync
    # even under an explicit override: `BC_EDGE=clamp` alone used to leave the gases
    # on the flux default and land in the never-used clamp/flux corner. BC_GAS=clamp
    # still works explicitly, and is what a run made under the old default needs
    # in order to reproduce.
    _bc_gas_default0 = 'flux' if _bc_edge0 == 'open' else 'clamp'
    _bc_gas0 = os.environ.get('BC_GAS', _bc_gas_default0).lower()

    # air-mass-ish weight for burden diagnostics as a function of latitude and pressure
    # The latitude weight MUST match the area metric the y-sweep conserves, or the
    # mass budget is open by construction. fast_advection/fct_fast.grid_metric
    # conserves sum(ac*q) with ac = the exact cell-mean of cos(phi); plain
    # cos(lat_j) differs from it by O(dphi^2) in the interior and, worse, assigns
    # the +-90 rows EXACTLY ZERO area, so anything transported into them leaves the
    # diagnosed burden for free. (Kept identical to grid_metric by construction --
    # same edge convention, same formula.)
    if ADV_METRIC:
        # edges from fct_fast.lat_edges (2026-11-18): half cells at CESM's +-90
        # rows as before (bit-identical), the poles themselves as the outer
        # edges of a Gaussian grid (JCM), whose outermost rows are not at +-90.
        # dphi = lat[1]-lat[0] is only the normalisation, as in grid_metric.
        from fct_fast import lat_edges as _lat_edges
        _e = _lat_edges(lat) * DEG
        W_LAT = (np.sin(_e[1:]) - np.sin(_e[:-1])) / ((lat[1] - lat[0]) * DEG)
    else:
        W_LAT = np.cos(lat * DEG)          # legacy weight, matches legacy sweep
    A = (DP[:, None, None] * W_LAT[None, :, None])
    A = np.broadcast_to(A, (nlev, nlat, nlon))
    A_j = jnp.asarray(A)
    # face weight for the vertical-face exchange diagnostic (no dp: the returned
    # flux is already in q*Pa units)
    AF_j = jnp.asarray(np.broadcast_to(W_LAT[:, None], (nlat, nlon)))

    # INTERIOR burden weight: zeroes the top/bottom BC levels, i.e. exactly the
    # levels where the reservoir is WRITTEN IN, so that 'int' measures what the
    # model itself owns rather than partly re-measuring our own boundary condition.
    # N/N0 over the FULL slab is a misleading diagnostic with a CARMA IC -- 70.9%
    # of the initial number sits in the bottom level alone (87.8 hPa, ~17600 #/cm3
    # of 2-8 nm particles) and 97.8% in the bottom two, so "total N" mostly measures
    # what happens to a tropopause ultrafine pile whose self-coagulation lifetime is
    # ~1.6 h. The interior is where the stratospheric aerosol signal lives.
    # The window is a pressure range so it can be stated unambiguously.
    #
    # SYMMETRY FIX 2026-07-29: the bottom index was `nlev - N_BC_BOT - 2`, i.e.
    # -1 for the BC level plus ONE EXTRA level. That extra level was the CARMA
    # "tropopause transition" (the old comment here said the window excludes the
    # BC levels AND the transition below them), needed when the CARMA IC put 73%
    # of all number at 87.8 hPa alone. It is void under MAM4 -- see the measured
    # note below: the bottom two levels hold 5.8% of initial number, not 99.2%.
    # Left as-is it made the window ASYMMETRIC with the top: over 1-150 hPa the top
    # dropped idx 0 (1.245 hPa) = exactly N_BC_TOP, but the bottom dropped BOTH
    # idx 23 (143.0 hPa, pinned) AND idx 22 (121.5 hPa, pinned by nothing), so
    # neither printed burden was "the interior the model owns": full-slab included
    # the reservoir writes, and 'int' excluded one real level beyond them.
    # Now -1 on both ends -> 22/24 levels, 1.6-121.5 hPa.
    # If you ever want the old CARMA behaviour back, do NOT re-add the -2; set
    # DIAG_CORE_HPA explicitly so the choice is recorded in the run log.
    _cw = os.environ.get('DIAG_CORE_HPA', '')
    if _cw:
        _clo, _chi = (float(x) for x in _cw.split(','))
    else:
        _clo = PLEV_PA[N_BC_TOP] / 100.0
        _chi = PLEV_PA[max(0, nlev - N_BC_BOT - 1)] / 100.0
    _core = (PLEV_PA / 100.0 >= _clo) & (PLEV_PA / 100.0 <= _chi)
    A_int = np.array(A, copy=True)
    A_int[~_core] = 0.0
    A_int_j = jnp.asarray(A_int)
    # NB the comment block above describes the CARMA IC only. Measured for the
    # MAM4 IC over 1-150 hPa (INIT_BIN=so4): there is NO tropopause pile -- the
    # bottom two levels hold 5.8% of initial number (CARMA: 99.2%), number peaks
    # at 8-13 hPa and mass at 29-52 hPa. So under AER_SRC=mam4 the full-slab
    # N/N0 is a usable signal and the default window (BC levels excluded) is
    # right; do NOT narrow it to 20,55 as was correct for CARMA, that drops the
    # number peak and keeps only ~9% of N.
    _pile = 'the CARMA tropopause ultrafine pile' if AER_SRC == 'carma' else None
    print(f"  diagnostic core window: {_clo:.1f}-{_chi:.1f} hPa "
          f"({int(_core.sum())}/{nlev} levels) -- 'int' burdens below use this"
          + (f"; the full-slab N/N0 is dominated by {_pile} and is NOT a "
             f"stratospheric signal" if _pile else
             f"; AER_SRC={AER_SRC} has no tropopause number pile, so the "
             f"full-slab N/N0 is meaningful too"), flush=True)

    def burdens_int(num_, mas_):
        return (float((num_.sum(0) * A_int_j).sum()),
                float((mas_.sum(0) * A_int_j).sum()))

    #global total burden of number and mass, weighted by A_j (Pa * cos(lat))
    def burdens(num_, mas_):
        Ntot = float((num_.sum(0) * A_j).sum())
        Mtot = float((mas_.sum(0) * A_j).sum())
        return Ntot, Mtot

    # mass burden with an optional lat restriction, for the staged budget:
    #   'np'  -> non-polar rows only  |  'pol' -> polar caps only  |  'all'
    pol_mask_j = jnp.asarray(np.abs(lat) > LAT_FREEZE)   # (nlat,) polar rows
    def Mbur(mas_, which='all'):
        if which == 'np':
            w = A_j * (~pol_mask_j)[None, :, None]
        elif which == 'pol':
            w = A_j * pol_mask_j[None, :, None]
        else:
            w = A_j
        return float((mas_.sum(0) * w).sum())
    #get initial total number and mass burdens
    N0, M0 = burdens(num, mas)
    N0i, M0i = burdens_int(num, mas)   # interior-only reference (see A_int)

    #gas burdens use the same Pa*cos(lat) weight as the aerosol
    def gas_burden(g_):
        return float((g_ * A_j).sum())
    S0 = gas_burden(so2)         # initial SO2 burden (for normalized logging)

    # ---- SAI SO2 injection geometry & per-step mixing-ratio increment ----
    # continuous release into one grid cell (or the whole latitude ring with
    # INJ_ZONAL), converted to a mixing-ratio increment per coupling step:
    #   dq = (kg SO2 released into the cell over the step) / (kg air in cell)
    inj_dq = None
    inj_dq_h2so4 = None
    if INJ_SO2_TG_YR > 0 or INJ_H2SO4_TG_YR > 0:
        k_inj = int(np.argmin(np.abs(PLEV_PA / 100.0 - INJ_HPA)))
        j_inj = int(np.argmin(np.abs(lat - INJ_LAT)))
        i_inj = int(np.argmin(np.abs(lon - INJ_LON)))
        # cell air mass = (dp/g) * cell area; f09 is uniform in lon and nearly
        # uniform in lat (np.gradient handles the half-width pole rows). On a
        # Gaussian grid (MET_SRC=jcm, T63) cos(lat)*gradient(lat) is within 1e-4
        # of the exact sin-edge area at +-30 (checked 2026-11-18), as on f09; kept
        # rather than switched to lat_edges so the CESM path's release is unchanged.
        dlat = np.gradient(lat) * DEG
        dlon = (lon[1] - lon[0]) * DEG
        area = (RAD ** 2) * np.cos(lat * DEG) * dlat * dlon    # (nlat,) m2

        # target row(s) and each one's share of the TOTAL release rate. INJ_MIRROR
        # adds the -INJ_LAT row at 50/50; the total stays INJ_SO2_TG_YR either way.
        j_mir = int(np.argmin(np.abs(lat + INJ_LAT)))
        if INJ_MIRROR and j_mir != j_inj:
            _rows = [(j_inj, 0.5), (j_mir, 0.5)]
        else:
            _rows = [(j_inj, 1.0)]

        # Air mass per cell is row-dependent (area goes as cos(lat)), so each row
        # must divide by ITS OWN air mass. Using one row's airm for both would make
        # the mirrored pair release unequal MASS while looking symmetric in the
        # mixing-ratio increment -- harmless on the symmetric f09 grid where the two
        # rows have equal area, but wrong the moment INJ_LAT snaps asymmetrically or
        # the grid changes. The budget's inj_cum weights by A_j, so a mismatch here
        # would show up as a plume that does not match the requested Tg/yr.
        def _airm(j):
            return DP[k_inj] / GRAV * area[j]                  # kg air per cell

        def _inj_increment(tg_yr):
            # kg species/yr -> per-step mixing-ratio increment at the inj cell(s)
            rate = tg_yr * 1e9 / (365.0 * 86400.0)             # kg species / s
            a = np.zeros((nlev, nlat, nlon))
            for _j, _frac in _rows:
                if INJ_ZONAL:
                    # split this row's share equally over the ring's nlon cells
                    a[k_inj, _j, :] += (rate * _frac / nlon) * STEP_SEC / _airm(_j)
                else:
                    a[k_inj, _j, i_inj] += (rate * _frac) * STEP_SEC / _airm(_j)
            return jnp.asarray(a)

        _latlbl = (f"lat {lat[j_inj]:+.1f}" if len(_rows) == 1 else
                   f"lat {lat[j_inj]:+.1f} & {lat[j_mir]:+.1f} (mirrored, 50/50)")
        loc = (f"{PLEV_PA[k_inj]/100:.1f} hPa, {_latlbl}, "
               f"{'zonal ring' if INJ_ZONAL else f'lon {lon[i_inj]:.1f}'} "
               f"(cell air mass {_airm(j_inj):.2e} kg)")
        if INJ_SO2_TG_YR > 0:
            inj_dq = _inj_increment(INJ_SO2_TG_YR)
            print(f"  SAI injection: {INJ_SO2_TG_YR:g} Tg SO2/yr at {loc}", flush=True)
        if INJ_H2SO4_TG_YR > 0:
            inj_dq_h2so4 = _inj_increment(INJ_H2SO4_TG_YR)
            print(f"  SAI injection: {INJ_H2SO4_TG_YR:g} Tg H2SO4/yr (gas, ad-hoc) "
                  f"at {loc}", flush=True)
        # INJ_LON is meaningless under INJ_ZONAL=1 -- the zonal branch of
        # _inj_increment fills the whole lon axis, so i_inj is computed and then
        # never used. Silence here is the dangerous case: a longitude SWEEP left at
        # the default INJ_ZONAL=1 produces N identical runs under N different tags,
        # and nothing in the log distinguishes them (the location string prints
        # 'zonal ring' and omits the longitude entirely). Compared against the
        # default rather than mere presence in os.environ, because run_prod.py
        # always exports INJ_LON to document its default -- presence would warn on
        # every ordinary run and get ignored.
        if INJ_MIRROR and j_mir == j_inj:
            print(f"  NOTE: INJ_MIRROR=1 has no effect at INJ_LAT={INJ_LAT:g} -- "
                  f"row {j_inj} (lat {lat[j_inj]:+.1f}) is its own mirror, so the "
                  "full rate goes into that single row.", flush=True)
        if INJ_ZONAL and INJ_LON != _INJ_LON_DEFAULT:
            print(f"  WARNING: INJ_LON={INJ_LON:g} was set but is IGNORED -- "
                  "INJ_ZONAL=1 releases around the entire latitude ring.\n"
                  "    Every longitude gets rate/nlon regardless of INJ_LON. Set "
                  "INJ_ZONAL=0 for a single-cell source at that longitude.\n"
                  "    If you are sweeping longitude, this run is IDENTICAL to the "
                  "other INJ_ZONAL=1 runs in the sweep.", flush=True)
    print(f"  micro mode: {MICRO_MODE}"
          + (f" ({describe_full_chain()})" if MICRO_MODE == 'full' else '')
          + f", settling {'ON' if SETTLE_ENABLE else 'OFF'}", flush=True)
    if MICRO_MODE == 'full':
        print(f"  OH source: "
              + (f"SZA curve (Hanisco 2001, parabola in cos(SZA) fitted to "
                 f"{len(OH_SZA_KNOTS)} knots: a={_OH_A:.4e} b={_OH_B:.4e}), "
                 f"peak {OH_PEAK * _OH_SCALE * (_OH_A + _OH_B):.3e} molec/cm3 "
                 f"at SZA=0, "
                 f"diurnal at {OH_SUBSTEPS} samples/step "
                 f"({DT_MICRO / OH_SUBSTEPS / 60:.0f} min)" if OH_SZA
                 else "CESM OH field (constant over step)"), flush=True)

    # diagnostics containers
    ts = {k: [] for k in ('hours', 'Nburden', 'Mburden', 'nsub',
                          'Nmin', 'Nmax', 'meanDp_nm', 'meanDp_num_nm',
                          'meanDp_mass_nm', 'reff_nm',
                          'clipMadd_cum', 'clipMrem_cum',
                          'B_adv_np', 'B_adv_pol', 'B_floor', 'B_micro', 'B_bc',
                          'B_settle', 'B_vf_in', 'B_vf_out', 'Nfloor_cum',
                          'SO2burden', 'H2SO4burden', 'injSO2_cum', 'settleM_cum',
                          'dT_min', 'dT_max', 'dT_rms', 'arf_toa', 'arf_toa_avg',
                          'hr_min', 'hr_max', 'hr_rms',
                          'aod550')}
    # trailing window of instantaneous ARF samples -> diurnal mean (see ARF_AVG_H)
    _arf_win = max(1, int(round(ARF_AVG_H / (STEP_HOURS * RAD_EVERY)))) \
        if ARF_AVG_H > 0 else 1
    arf_hist = collections.deque(maxlen=_arf_win)
    if rad is not None:
        print(f"  ARF_toa reporting: instantaneous sample every "
              f"{STEP_HOURS * RAD_EVERY}h"
              + (f", reported as a trailing {_arf_win}-sample "
                 f"({_arf_win * STEP_HOURS * RAD_EVERY}h) mean"
                 if _arf_win > 1 else " (NO diurnal averaging: ARF_AVG_H=0)"),
              flush=True)
        if getattr(rad, 'arf_exact', False) and RAD_REF_EVERY > 1 and _arf_win > 1:
            print(f"  WARNING: RAD_EMU_ARF=exact samples the forcing only every "
                  f"RAD_REF_EVERY={RAD_REF_EVERY} steps, so the {_arf_win}-finite-sample "
                  f"arf_toa_avg spans {_arf_win * STEP_HOURS * RAD_EVERY * RAD_REF_EVERY}h, "
                  f"not {_arf_win * STEP_HOURS * RAD_EVERY}h as in a physics run: do not "
                  f"compare arf_toa_avg across such runs (compare arf_toa at matching hours "
                  f"or rad_bench's arf_emu/arf_phys). RAD_EMU_ARF=emu avoids this.", flush=True)
    clip_add_cum = 0.0; clip_rem_cum = 0.0   # cumulative clip mass (burden units)
    nfloor_cum = 0.0   # cumulative NUMBER manufactured by the post-advection floor
    inj_cum = 0.0      # cumulative injected SO2 (burden units, gas budget)
    inj_h2so4_cum = 0.0  # cumulative ad-hoc injected H2SO4 gas (burden units)
    settle_cum = 0.0   # cumulative aerosol mass settled out the bottom (burden units)
    aod_gm = float('nan')                    # last radiation-step global AOD550
    # staged mass-budget attribution, cumulative and normalized by M0. Each hour
    # the change in total M burden is split by measuring M at checkpoints:
    #   adv_np  : advective transport in the non-polar interior
    #   adv_pol : polar-cap reset (|lat|>LAT_FREEZE cells overwritten in advect)
    #   floor   : the jnp.maximum(.,0) clamp after advection (adds mass only)
    #   micro   : coagulation + two-moment clip (should match clipM add+rem)
    #             (MICRO=full: also nucleation + condensation, i.e. real
    #              gas->particle growth, so this stage is now a genuine source)
    #   settle  : gravitational settling; interior redistribution conserves
    #             the burden so this stage == -(bottom outflow) exactly
    #   bc      : open-BC refill of the top/bottom band levels
    # By construction sum(cumB) == M/M0 - 1 exactly (closure check in the log).
    cumB = {k: 0.0 for k in ('adv_np', 'adv_pol', 'floor', 'micro', 'settle', 'bc')}
    # vertical FACE exchange, diagnosed from the transport scheme (see ADV_WCONT).
    # These are an ATTRIBUTION of part of adv_np/adv_pol, not extra stages.
    cumV = {'in': 0.0, 'out': 0.0}
    frames_num = []; frames_mas = []; frames_dT = []; frame_hours = []
    frames_so2 = []; frames_h2so4 = []; frames_hr = []
    # FRAME_LEVELS='all' keeps a zonal-mean frame at every level BESIDE the probe
    # slabs. It carries its OWN hour list rather than being index-aligned to
    # frame_hours: a RESUME from a checkpoint written under FRAME_LEVELS=probe
    # has no zonal history at all, so this list is legitimately shorter, and a
    # file that says so is one a plot can read without guessing an offset.
    frames_zm_num = []; frames_zm_mas = []; frames_zm_dT = []
    frames_zm_so2 = []; frames_zm_h2so4 = []; frames_zm_hours = []
    frames_zm_hr = []

    def _zm_capture(hr):
        """Append one zonal-mean frame (mean over longitude) at every level."""
        if FRAME_LEVELS != 'all':
            return
        # PLAIN mean over longitude. Every cell in a latitude row has the same
        # area on this grid, so no cos(lat) weight belongs here -- that metric is
        # a MERIDIONAL mean's, and folding it in would double-count against the
        # wlat_full() weighting the plots already apply.
        frames_zm_num.append(np.asarray(num.mean(axis=3)).copy())
        frames_zm_mas.append(np.asarray(mas.mean(axis=3)).copy())
        frames_zm_dT.append(np.asarray(dT_rad.mean(axis=2)).copy())
        frames_zm_hr.append(np.asarray(hr_kday.mean(axis=2), np.float32).copy())
        frames_zm_so2.append(np.asarray(so2.mean(axis=2)).copy())
        frames_zm_h2so4.append(np.asarray(h2so4.mean(axis=2)).copy())
        frames_zm_hours.append(int(hr))

    def _zm_arrays():
        """Zonal-mean arrays for a savez call; empty when none were captured."""
        if not frames_zm_hours:
            return {}
        return dict(frames_zm_hours=np.array(frames_zm_hours),
                    frames_zm_num=np.stack(frames_zm_num),
                    frames_zm_mas=np.stack(frames_zm_mas),
                    frames_zm_dT=np.stack(frames_zm_dT),
                    frames_zm_hr_kday=np.stack(frames_zm_hr),
                    frames_zm_so2=np.stack(frames_zm_so2),
                    frames_zm_h2so4=np.stack(frames_zm_h2so4))
    KPROBE = int(np.argmin(np.abs(PLEV_PA / 100 - PROBE_HPA)))   # diagnostic probe level
    _res = 'per-step MAM4' if AER_SRC == 'mam4' else 'static CARMA'
    _edge = _bc_edge0
    # The gases are ALWAYS read from the CESM SO2/H2SO4 h1 fields (read_gases),
    # never from the AER_SRC reservoir -- _res describes the AEROSOL source only.
    # Printing _res for the gases mislabelled CESM-forced gases as "CARMA-forced"
    # whenever AER_SRC=carma, which is the wrong provenance to copy into a writeup.
    # BC_GAS=flux also means they are not forced at all, so "always" was wrong too.
    _gas = ('CESM-forced (Dirichlet)'
            if _bc_gas0 != 'flux'
            else 'FLUX (open faces, NOT pinned to CESM)')
    print(f"  STEP_HOURS={STEP_HOURS}h (advect+coag per step); "
          f"vertical BC at {np.round(PLEV_PA[:N_BC_TOP]/100, 1)} / "
          f"{np.round(PLEV_PA[-N_BC_BOT:]/100, 1)} hPa: "
          + ("FLUX (continuity omega, open faces; aerosol inflow at the "
             f"{_res} reservoir, free outflow)" if ADV_WCONT
             else ("closed column (SL), exchange = the edge-level reset"
                   if _SL_CLOSED else "sealed faces (legacy omega)"))
          + (f"; aerosol edge levels {'left free' if _edge == 'open' else 'CLAMPED to '+_res}")
          + f"; gases {_gas} at the edges; "
          + (f"polar caps (|lat|>{LAT_FREEZE}, {pol_idx.size} rows) "
             + ("stirred to one well-mixed cell/level (mass-conserving)"
                if os.environ.get('ADV_POLAR', 'zonal').lower() == 'zonal'
                else "overwritten from the reservoir (LEGACY, discards mass)"))
          + (f"; bottom-face aerosol inflow = {BC_BOT_AER:g}x reservoir"
             + ("  (AEROSOL-FREE upwelling)" if BC_BOT_AER == 0.0 else "")
             if ADV_WCONT else "")
          + f"; y-metric {'on' if ADV_METRIC else 'OFF'}, "
          + f"dx-fix {'on' if os.environ.get('ADV_DXFIX','1') != '0' else 'OFF'}; "
          + ("per-tracer mass fixer on the closed extended column" if _SL_CLOSED
             else "per-level SL fixer + column fixer to the face-flux target (SL split)"
             if ADV_SCHEME == 'sl' else "NO mass fixer (open system)"), flush=True)
    if ADV_SCHEME == 'ffsl':
        print(f"  TRANSPORT: {ffsl_lr.describe()}; winds = "
              + ("hourly CESM snapshots at the sub-step midpoints" if ADV_SL_HOURLY
                 else "the two step snapshots interpolated in time"), flush=True)
    if ADV_SCHEME == 'sl':
        print(f"  TRANSPORT: {sl_lr.describe()}; winds = "
              + ("hourly CESM snapshots resampled to the sub-steps" if ADV_SL_HOURLY
                 else "the two bracketing snapshots, linear in time")
              + ("\n  WARNING: ADV_WCONT=1 with ADV_SL_VERT=sl: that column is closed, "
                 "so with the edge levels left free (BC_EDGE=open) the band cannot "
                 "exchange with the reservoir at all. Use ADV_WCONT=0."
                 if ADV_WCONT and _SL_CLOSED else ""), flush=True)
    print(f"  probe/frames at {PLEV_PA[KPROBE]/100:.1f} hPa"
          + ("; FRAME_LEVELS=all -> plus a zonal-mean frame at all "
             f"{nlev} levels ({PLEV_PA[0]/100:.1f}-{PLEV_PA[-1]/100:.1f} hPa)"
             if FRAME_LEVELS == 'all' else
             "; FRAME_LEVELS=probe (this level only)"), flush=True)

    # capture the true initial state as frame 0
    frames_num.append(np.asarray(num[:, KPROBE]).copy())
    frames_mas.append(np.asarray(mas[:, KPROBE]).copy())
    frames_dT.append(np.asarray(dT_rad[KPROBE]).copy())
    frames_hr.append(np.asarray(hr_kday[KPROBE], np.float32).copy())
    frames_so2.append(np.asarray(so2[KPROBE]).copy())
    frames_h2so4.append(np.asarray(h2so4[KPROBE]).copy())
    frame_hours.append(0)
    _zm_capture(0)

    N_STEPS = N_HOURS // STEP_HOURS
    FRAME_EVERY_STEPS = max(1, round(FRAME_EVERY / STEP_HOURS))

    # ---- resume from a state checkpoint (see STATE_CKPT/RESUME) -------------
    # Everything above this point has built a valid step-0 state; if we are
    # resuming, that state and every cumulative counter is now REPLACED by the
    # checkpoint. N0/M0/N0i/M0i/S0 come from the checkpoint too -- they are the
    # day-0 normalizations and recomputing them from the resumed state would
    # silently redefine every ratio in the log.
    _state_f = f'coupled_state_{OUT_TAG}_ckpt.npz'
    s_start = 0
    if RESUME:
        if not os.path.exists(_state_f):
            raise SystemExit(f"  RESUME=1 but {_state_f} does not exist")
        _ck = np.load(_state_f, allow_pickle=False)
        _fp = (int(_ck['nbins']), int(_ck['nlev']), int(_ck['nlat']),
               int(_ck['nlon']), int(_ck['step_hours']))
        _now = (NA, nlev, nlat, nlon, STEP_HOURS)
        if _fp != _now:
            raise SystemExit(f"  RESUME refused: checkpoint geometry {_fp} != "
                             f"this run's {_now} (bins,lev,lat,lon,step_hours)")
        # The met source and its grid must match (GRID_CFG, refused). A checkpoint
        # from before the stamp existed was written on the CESM f09 grid by
        # MET_SRC=cesm or file (both on the archive grid), so it is compared as
        # that: refused under MET_SRC=jcm, accepted otherwise.
        if 'grid_cfg' in _ck.files:
            _g_ck = tuple(float(x) for x in _ck['grid_cfg'])
            _g_now = tuple(float(x) for x in GRID_CFG)
            _n = min(len(_g_ck), len(_g_now))
            # stored and live JCM share the grid (PHYS_CFG warns on the switch)
            _same = (_g_ck[0] == _g_now[0] or {_g_ck[0], _g_now[0]} <= {0.0, 1.0}
                     or {_g_ck[0], _g_now[0]} <= {2.0, 3.0}) and \
                np.allclose(_g_ck[1:_n], _g_now[1:_n], rtol=1e-6, atol=1e-6)
            if not _same:
                raise SystemExit(
                    "  RESUME refused: checkpoint grid/met source "
                    + ', '.join(f'{k}={a:.6g}' for k, a in zip(GRID_CFG_KEYS, _g_ck))
                    + "\n    != this run's "
                    + ', '.join(f'{k}={b:.6g}' for k, b in zip(GRID_CFG_KEYS, _g_now)))
        elif MET_JCM:
            raise SystemExit(f"  RESUME refused: {_state_f} has no grid stamp, i.e. it "
                             "predates MET_SRC=jcm and was written on the CESM grid")
        # Injection scenario must match too. The geometry check above cannot see it:
        # two scenarios differing only in Tg/yr or in where the plume is released
        # have identical (bins,lev,lat,lon,step_hours), so without this a
        # `RESUME=1 OUT_TAG=<wrong tag>` silently continues scenario A's aerosol
        # under scenario B's source and the output is attributed to B. Nothing
        # downstream could detect that -- the state carries no record of how it was
        # made. Checkpoints written before this key existed are grandfathered in
        # (absent key = skip) rather than being made unresumable.
        if 'inj_cfg' in _ck.files:
            _inj_ck = tuple(float(x) for x in _ck['inj_cfg'])
            _inj_now = tuple(float(x) for x in INJ_CFG)
            # Compare the COMMON PREFIX only: INJ_CFG is append-only, so a shorter
            # stamp means the checkpoint predates a field, not that it disagrees
            # about one. Comparing full tuples would refuse every such checkpoint on
            # a length mismatch alone -- turning each new scenario field into a
            # silent tripwire that strands in-flight runs.
            _n = min(len(_inj_ck), len(_inj_now))
            if _inj_ck[:_n] != _inj_now[:_n]:
                _lbl = [str(k) for k in INJ_CFG_KEYS]
                _diff = '; '.join(f"{l}: ckpt {a:g} != now {b:g}"
                                  for l, a, b in zip(_lbl, _inj_ck[:_n],
                                                     _inj_now[:_n])
                                  if a != b)
                raise SystemExit(
                    f"  RESUME refused: {_state_f} was written by a DIFFERENT "
                    f"injection scenario.\n    {_diff}\n"
                    "    Resuming would continue that run's aerosol under this "
                    "run's source. Use that scenario's own OUT_TAG, or set the "
                    "INJ_* values back to match the checkpoint.")
        # Physics-mode flags: WARN, never refuse -- see the PHYS_CFG comment at the
        # top of this module for why the two checks differ. A checkpoint written
        # before this key existed (every pre-2026-08-04 one, including the
        # 91.5%-complete prod1yr) simply skips the check rather than being blocked.
        if 'phys_cfg' in _ck.files:
            _ph_ck = tuple(float(x) for x in _ck['phys_cfg'])
            _ph_now = tuple(float(x) for x in PHYS_CFG)
            _n = min(len(_ph_ck), len(_ph_now))
            _pdiff = [(str(k), a, b) for k, a, b
                      in zip(PHYS_CFG_KEYS[:_n], _ph_ck[:_n], _ph_now[:_n])
                      if a != b]
            # Stamps appended after the checkpoint was written are compared
            # against the value the code IMPLIED then, where that is known:
            # before 2026-11-17 dT_rad always fed back, and RH was RELHUM under
            # MET_SRC=cesm, Q over ice under file. A bare resume of such a ckpt
            # on the current defaults is therefore reported as the seam it is.
            # MICRO_FION: dev's stratospheric setup's 10 (the user set it to 0
            # on 2026-11-18). The other MICRO_* stamps have no single implied
            # value (older ckpts ran other engines), so they are not compared.
            _legacy = {'RAD_T_FEEDBACK': 1.0,
                       'MET_RH': 0.0 if MET_SRC == 'cesm' else 1.0,
                       'MICRO_FION': 10.0}
            for _i in range(_n, len(_ph_now)):
                _k = str(PHYS_CFG_KEYS[_i])
                if _k in _legacy and _legacy[_k] != _ph_now[_i]:
                    _pdiff.append((_k + ' (implied by an older ckpt)',
                                   _legacy[_k], _ph_now[_i]))
            if _pdiff:
                print("  WARNING: this RESUME changes the PHYSICS mid-run. "
                      f"{_state_f} was written with:", flush=True)
                for _k, _a, _b in _pdiff:
                    print(f"      {_k}: ckpt {_a:g} -> now {_b:g}", flush=True)
                print("    The state itself is valid and the run will continue "
                      "normally -- but the trajectory\n"
                      "    has a SEAM at this step, and any timeseries spanning it "
                      "is two models, not one.\n"
                      "    Intended when applying a physics fix to an in-flight "
                      "run; otherwise pass the\n"
                      "    flags above at their checkpoint values to keep the run "
                      "homogeneous.", flush=True)
        s_start = int(_ck['s_done']) + 1
        if s_start >= N_STEPS:
            raise SystemExit(f"  RESUME: checkpoint is already at step "
                             f"{s_start}/{N_STEPS} -- nothing to do")
        num = jnp.asarray(_ck['num']); mas = jnp.asarray(_ck['mas'])
        so2 = jnp.asarray(_ck['so2']); h2so4 = jnp.asarray(_ck['h2so4'])
        dT_rad = jnp.asarray(_ck['dT_rad'])
        # optional: checkpoints from before the heating-rate output have none,
        # and zeros are only wrong until the next radiation call refreshes it
        if 'hr_kday' in _ck.files:
            hr_kday = np.asarray(_ck['hr_kday'], np.float64)
        N0 = float(_ck['N0']); M0 = float(_ck['M0'])
        N0i = float(_ck['N0i']); M0i = float(_ck['M0i']); S0 = float(_ck['S0'])
        clip_add_cum = float(_ck['clip_add_cum'])
        clip_rem_cum = float(_ck['clip_rem_cum'])
        nfloor_cum = float(_ck['nfloor_cum']); inj_cum = float(_ck['inj_cum'])
        inj_h2so4_cum = float(_ck['inj_h2so4_cum'])
        settle_cum = float(_ck['settle_cum']); aod_gm = float(_ck['aod_gm'])
        cumB = {k: float(v) for k, v in zip(_ck['cumB_keys'], _ck['cumB_vals'])}
        cumV = {k: float(v) for k, v in zip(_ck['cumV_keys'], _ck['cumV_vals'])}
        arf_hist = collections.deque(
            [float(x) for x in _ck['arf_hist']], maxlen=_arf_win)
        # frames + timeseries are written in the same block as the state, but NOT
        # in the same instant, so they are only consistent with it if that block
        # ran to completion. Replace (not append to) the lists, which already hold
        # the freshly-built frame 0.
        #
        # The frames ckpt must NEVER hard-block a resume. It is a visualization
        # artifact -- probe-level slabs for the gifs and filmstrip -- while the
        # state ckpt holds the physics. An unconditional np.load here is what
        # stopped a 91.5%-complete prod1yr run from restarting after the
        # 2026-08-03 OOM truncated the frames file: the run was fine, the movie
        # frames were not, and the movie frames won. Degrade instead: warn loudly,
        # start the frame history empty, and let the run finish.
        _ff = f'coupled_frames_{OUT_TAG}_ckpt.npz'
        try:
            _fk = np.load(_ff)
            frame_hours = [int(h) for h in _fk['frame_hours']]
            frames_num = [a.copy() for a in _fk['frames_num']]
            frames_mas = [a.copy() for a in _fk['frames_mas']]
            frames_dT = [a.copy() for a in _fk['frames_dT']]
            # heating-rate history is optional on the way in (older ckpts have
            # none): NaN-pad it to frame_hours so the two stay index-aligned
            frames_hr = ([a.copy() for a in _fk['frames_hr_kday']]
                         if 'frames_hr_kday' in _fk.files else
                         [np.full(a.shape, np.nan, np.float32) for a in frames_dT])
            frames_so2 = [a.copy() for a in _fk['frames_so2']]
            frames_h2so4 = [a.copy() for a in _fk['frames_h2so4']]
            # Zonal history is OPTIONAL on the way in: a checkpoint written
            # before FRAME_LEVELS existed, or under FRAME_LEVELS=probe, simply
            # has none. Starting it empty is correct rather than an error --
            # frames_zm_hours records where it really begins, so the gap is
            # visible in the file instead of being silently mis-aligned.
            if 'frames_zm_hours' in _fk.files:
                frames_zm_hours = [int(h) for h in _fk['frames_zm_hours']]
                frames_zm_num = [a.copy() for a in _fk['frames_zm_num']]
                frames_zm_mas = [a.copy() for a in _fk['frames_zm_mas']]
                frames_zm_dT = [a.copy() for a in _fk['frames_zm_dT']]
                frames_zm_hr = ([a.copy() for a in _fk['frames_zm_hr_kday']]
                                if 'frames_zm_hr_kday' in _fk.files else
                                [np.full(a.shape, np.nan, np.float32)
                                 for a in frames_zm_dT])
                frames_zm_so2 = [a.copy() for a in _fk['frames_zm_so2']]
                frames_zm_h2so4 = [a.copy() for a in _fk['frames_zm_h2so4']]
            elif FRAME_LEVELS == 'all':
                print("  NOTE: FRAME_LEVELS=all but the frames ckpt has no zonal "
                      "history (written under FRAME_LEVELS=probe).\n"
                      "    The zonal frames will start at this resume point; the "
                      "probe-level history is unaffected.", flush=True)
        except Exception as _e:
            print(f"  WARNING: frames ckpt {_ff} is unreadable "
                  f"({type(_e).__name__}: {_e}).\n"
                  "    The PHYSICS state is unaffected -- resuming with an EMPTY "
                  "frame history.\n"
                  "    Frames from before this resume are lost; the run will "
                  "produce frames from here on,\n"
                  "    so the gifs/filmstrip will cover only the resumed segment. "
                  "Seed the file from a\n"
                  "    known-good frames ckpt before resuming if you need the "
                  "earlier frames.", flush=True)
            frame_hours = []
            frames_num = []; frames_mas = []; frames_dT = []
            frames_so2 = []; frames_h2so4 = []; frames_hr = []
            frames_zm_num = []; frames_zm_mas = []; frames_zm_dT = []
            frames_zm_so2 = []; frames_zm_h2so4 = []; frames_zm_hours = []
            frames_zm_hr = []
        # The timeseries ckpt gets the same treatment as the frames one above, and
        # for the same reason: it is a DIAGNOSTIC record, not the physics. Every
        # cumulative counter the run needs to continue correctly (cumB/cumV, the
        # *_cum totals, arf_hist) was already restored from the state ckpt above --
        # `ts` is only the plotted history. An unconditional np.load here would
        # reintroduce exactly the failure that stranded the 91.5%-complete prod1yr
        # run, just via a 328 KB file instead of a 5.9 GB one. Smaller means less
        # likely to be caught mid-write, NOT impossible: it is written in the same
        # block, so a kill lands in it with the same probability per byte.
        _tf = f'coupled_timeseries_{OUT_TAG}_ckpt.npz'
        try:
            _tk = np.load(_tf)
            # Restore every ts key, not just the ones the checkpoint happens to
            # have: a diagnostic added since the checkpoint was written must still
            # exist as a list, NaN-padded to the same length, or the first append
            # would desynchronize it from 'hours' and corrupt every later plot.
            _nrec = len(_tk['hours'])
            ts = {k: (list(_tk[k]) if k in _tk.files else [float('nan')] * _nrec)
                  for k in ts}
        except Exception as _e:
            print(f"  WARNING: timeseries ckpt {_tf} is unreadable "
                  f"({type(_e).__name__}: {_e}).\n"
                  "    The PHYSICS state is unaffected -- the cumulative budget "
                  "counters come from the state\n"
                  "    ckpt, not from here -- so resuming with an EMPTY timeseries "
                  "history.\n"
                  "    The step log and every timeseries panel will cover only the "
                  "resumed segment.", flush=True)
            ts = {k: [] for k in ts}
        # ---- frames/ts ahead of the state -----------------------------------
        # The state is written LAST, so a kill inside the ckpt block can leave the
        # frames and/or timeseries one cycle NEWER than the state. Resuming from
        # such a set replays steps that those two files already contain: the ts
        # would get a second record at the same hour, and frame_hours a duplicate
        # entry, silently double-counting a step in every plot that indexes by
        # hour. Trim both back to the state's own step so the restored set is
        # consistent regardless of where the write was interrupted. Cheap and a
        # no-op on a clean checkpoint -- this only ever fires after a crash.
        _h_state = s_start * STEP_HOURS       # last hour the STATE accounts for
        _ndrop_ts = sum(1 for h in ts['hours'] if float(h) > _h_state)
        if _ndrop_ts:
            # keep-length computed ONCE: 'hours' is the first key in ts, so
            # reading len(ts['hours']) inside the loop would shrink the target for
            # every later key and skew the series against 'hours'.
            _keep_ts = len(ts['hours']) - _ndrop_ts
            for _k in ts:
                ts[_k] = ts[_k][:_keep_ts]
            print(f"  NOTE: timeseries ckpt ran {_ndrop_ts} record(s) past the "
                  f"state (h>{_h_state}); trimmed to stay consistent.", flush=True)
        _ndrop_fr = sum(1 for h in frame_hours if h > _h_state)
        if _ndrop_fr:
            _keep = len(frame_hours) - _ndrop_fr
            frame_hours = frame_hours[:_keep]
            frames_num = frames_num[:_keep]; frames_mas = frames_mas[:_keep]
            frames_dT = frames_dT[:_keep]; frames_so2 = frames_so2[:_keep]
            frames_h2so4 = frames_h2so4[:_keep]; frames_hr = frames_hr[:_keep]
            print(f"  NOTE: frames ckpt ran {_ndrop_fr} frame(s) past the state "
                  f"(h>{_h_state}); trimmed to stay consistent.", flush=True)
        # the zonal history is trimmed against ITS OWN hours, not against _keep:
        # it can be shorter than frame_hours (see the resume note above), so
        # reusing the probe-slab count here would cut the wrong number of frames
        _ndrop_zm = sum(1 for h in frames_zm_hours if h > _h_state)
        if _ndrop_zm:
            _kz = len(frames_zm_hours) - _ndrop_zm
            frames_zm_hours = frames_zm_hours[:_kz]
            frames_zm_num = frames_zm_num[:_kz]; frames_zm_mas = frames_zm_mas[:_kz]
            frames_zm_dT = frames_zm_dT[:_kz]; frames_zm_so2 = frames_zm_so2[:_kz]
            frames_zm_h2so4 = frames_zm_h2so4[:_kz]; frames_zm_hr = frames_zm_hr[:_kz]
            print(f"  NOTE: zonal frames ran {_ndrop_zm} frame(s) past the state; "
                  f"trimmed.", flush=True)

        # ---- frames history BEHIND the state (a hole, not an overrun) --------
        # The trim above fixes frames that are too NEW. The opposite case is not a
        # crash and so goes unnoticed: a frames ckpt that stops well short of the
        # state leaves a GAP, and the run then appends from the state's hour on top
        # of it. frame_hours records the truth, but the filmstrip and the gifs draw
        # panels in sequence -- so day 90 ends up beside day 335 with nothing in the
        # image saying 245 days were skipped.
        # This is the live situation for prod1yr: the 2026-08-03 truncation destroyed
        # its real frames (hours 2184-8016) and the file was replaced by hand with
        # the prod90d frames, which cover 0-2160 while the state sits at 8016.
        # Warn rather than refuse -- the physics is unaffected and finishing the run
        # matters more than the movie -- but say it before 30 hours of GPU time, not
        # after, and name the two ways out.
        if frame_hours and FRAME_EVERY > 0:
            _gap = _h_state - int(frame_hours[-1])
            if _gap > FRAME_EVERY:
                print(f"  WARNING: the frames history STOPS at h{int(frame_hours[-1])} "
                      f"but the state is at h{_h_state} -- a {_gap} h "
                      f"({_gap/24:.0f} day) HOLE.\n"
                      f"    {len(frame_hours)} frames restored; new ones will be "
                      f"appended from h{_h_state + FRAME_EVERY} onward, so "
                      "frame_hours will jump.\n"
                      "    The filmstrip and gifs draw panels in ORDER and will "
                      "splice across that hole without\n"
                      "    showing it. The physics state and the timeseries are "
                      "unaffected.\n"
                      "    Either accept a gapped filmstrip, or start the frames "
                      "history clean by deleting\n"
                      f"    coupled_frames_{OUT_TAG}_ckpt.npz before resuming "
                      "(frames then cover only the resumed segment).", flush=True)

        print(f"  RESUMED from {_state_f}: continuing at step {s_start+1}/"
              f"{N_STEPS} (h {s_start*STEP_HOURS}), {len(frame_hours)} frames, "
              f"{len(ts['hours'])} logged steps restored", flush=True)
        print(f"    restored M/M0 {float(burdens(num, mas)[1])/M0:.4f}  "
              f"budget sum {sum(cumB.values()):+.3e}  inj_cum {inj_cum:.3e}",
              flush=True)

    # Derive the day count from N_HOURS, NOT from N_DAYS. N_DAYS only sets the
    # DEFAULT for N_HOURS (:382), so any run that sets N_HOURS directly leaves
    # N_DAYS at its default 2 and this banner used to lie -- the 90-day prod run
    # announced itself as "2-day coupled run: 360 steps x 6h = 2160h". Nothing
    # about the run was wrong (N_STEPS comes from N_HOURS alone); only the label.
    print(f"\n{'='*60}\n{N_STEPS*STEP_HOURS/24:g}-day coupled run: "
          f"{N_STEPS} steps x {STEP_HOURS}h "
          f"= {N_STEPS*STEP_HOURS}h, advect+coag every step"
          + (f"  [RESUMING at step {s_start+1}]" if s_start else "")
          + f"\n{'='*60}", flush=True)

    PROFILE = bool(os.environ.get('PROFILE'))
    _ckw = _CkptWriter(CKPT_ASYNC)
    print(f"  CKPT_ASYNC={int(CKPT_ASYNC)}: _ckpt files "
          f"{'snapshotted to host, written by a background thread' if CKPT_ASYNC else 'written synchronously'}"
          f" (every {CKPT_EVERY_FRAMES} frame(s), STATE_CKPT={int(STATE_CKPT)})", flush=True)
    # winds at the START of the first step to be run. u0/v0/w0 is the PREVIOUS
    # step's field, carried forward at the bottom of the loop, so on a resume it
    # must be the checkpoint's hour -- seeding it with winds(0) makes the first
    # resumed step interpolate transport across the wrong time interval and the
    # trajectory then diverges from an uninterrupted run. Identical to winds(0)
    # when s_start == 0.
    u0, v0, w0 = winds(s_start * STEP_HOURS)
    for s in range(s_start, N_STEPS):
        it0 = s * STEP_HOURS; it1 = (s + 1) * STEP_HOURS   # hourly time indices
        tw = time.time()
        #these winds will eventually be used to advect the tracers from it0 to it1 using advection scheme
        u1, v1, w1 = winds(it1)
        if ADV_SL_HOURLY:
            # the hourly snapshots between the two step ends, for the SL
            # sub-steps (sl_lr resamples them onto its K sub-step nodes)
            hourly = ([(u0, v0, w0)] + winds_block(it0 + 1, it1 - 1) + [(u1, v1, w1)])
        t_read = time.time() - tw

        # ---- 0. SAI source: continuous SO2 release into the injection cell(s) ----
        # applied at the step start so the fresh pulse is advected and
        # oxidized within the same step it is emitted; touches ONLY the gas,
        # so the aerosol staged budget is unaffected by the ordering
        if inj_dq is not None:
            so2 = so2 + inj_dq
            inj_cum += float((inj_dq * A_j).sum())
        if inj_dq_h2so4 is not None:
            # ad-hoc gas-phase H2SO4 source (INJ_H2SO4_TG_YR)
            h2so4 = h2so4 + inj_dq_h2so4
            inj_h2so4_cum += float((inj_dq_h2so4 * A_j).sum())

        # budget checkpoint: burden at the start of this step (== end of prev)
        M_start_np = Mbur(mas, 'np'); M_start_pol = Mbur(mas, 'pol')

        # ---- 1. transport (advect all 82 tracers with shared winds) ----
        # (80 aerosol rows plus the 2 gas tracers stacked on the end; winds are
        # shared, so the whole 82-row stack advects in one batch)
        ta = time.time()
        # add all tracers and advect them together
        if ADV_PASSIVE:
            # the test: total sulfate mass as the one aerosol tracer (see ADV_PASSIVE)
            _num0 = num; _mas0 = mas
            _mt0 = mas.sum(axis=0)
            qb = jnp.concatenate([_mt0[None], so2[None], h2so4[None]], axis=0)
            _qfroz_full = qfroz
            qfroz = jnp.concatenate([qfroz[NA:2*NA].sum(axis=0)[None],
                                     qfroz[2*NA][None], qfroz[2*NA + 1][None]], axis=0)
        else:
            qb = jnp.concatenate([num, mas, so2[None], h2so4[None]], axis=0)
        ntr_all = qb.shape[0]
        step = TRACER_CHUNK if (0 < TRACER_CHUNK < ntr_all) else ntr_all
        akw = dict(lat=lat, dp=DP, lat_freeze=LAT_FREEZE, dt_total=STEP_SEC)
        if ADV_VFLUX:
            akw['return_vflux'] = True
        if ADV_SCHEME == 'sl':
            # the SL vertical coordinate is the true level pressure (omega is
            # Pa/s); DP alone does not reconstruct it on log-spaced levels
            akw['plev'] = PLEV_PA
        if ADV_SL_HOURLY:
            akw['hourly_winds'] = hourly
        if step == ntr_all:
            out = advect_step_batch(qb, u0, v0, w0, u1, v1, w1,
                                    qfrozb=qfroz, **akw)
            qb, nsub, vfl = out if ADV_VFLUX else (out[0], out[1], None)
        else:
            # advect the tracer stack in chunks to cap peak memory; winds and
            # therefore nsub are shared across all tracers, so this is exact.
            outs = []; vfs = []
            for a in range(0, ntr_all, step):
                b = min(a + step, ntr_all)
                out = advect_step_batch(qb[a:b], u0, v0, w0, u1, v1, w1,
                                        qfrozb=qfroz[a:b], **akw)
                outs.append(out[0]); nsub = out[1]
                if ADV_VFLUX:
                    vfs.append(out[2])
            qb = jnp.concatenate(outs, axis=0)
            vfl = jnp.concatenate(vfs, axis=0) if ADV_VFLUX else None
        # vertical FACE exchange over this step, in the same burden units as Mbur:
        # [0] = the slab TOP face, [1] = the slab BOTTOM face. These are FACE
        # labels, not directions: each is a SIGNED NET flux (after the sign flip at
        # cumV below, + = into the slab, - = out of it), so 'vf_in' can legitimately
        # print negative when the top face is net outflowing. Do NOT read them as
        # gross inflow/outflow -- fct_lr's f_top/f_bot is one signed number per
        # column per substep, and this sum cancels inflow columns against outflow
        # columns over all 192x288 columns and every substep, so GROSS inflow is
        # invisible here by construction. A net -2e-3 at the bottom is equally
        # consistent with (out 2e-3, in 0) and (out 2.5e-3, in 5e-4). To measure
        # what BC_BOT_AER actually feeds in, A/B it against BC_BOT_AER=0; this
        # diagnostic cannot answer that question.
        # Aerosol MASS rows only (NBINS:2*NBINS) -- gases are budgeted separately.
        if vfl is not None:
            if ADV_PASSIVE:
                vf_in = float((vfl[0, 0] * AF_j).sum()); vf_out = float((vfl[0, 1] * AF_j).sum())
            else:
                vf_in = float((vfl[NA:2*NA, 0] * AF_j).sum())
                vf_out = float((vfl[NA:2*NA, 1] * AF_j).sum())
        else:
            vf_in = vf_out = 0.0
        if ADV_PASSIVE:
            _mt1 = jnp.maximum(qb[0], 0.0)
            so2 = qb[1]; h2so4 = qb[2]
            qfroz = _qfroz_full
            _had = _mt0 > 0.0
            _ratio = jnp.where(_had, _mt1 / jnp.where(_had, _mt0, 1.0), 0.0)
            # shapes for cells that were empty: per-bin fraction of the level's
            # zonal-mean mass (number per unit mass likewise), global-level fallback
            _mz = _mas0.sum(axis=3, keepdims=True); _nz = _num0.sum(axis=3, keepdims=True)
            _mtz = _mz.sum(axis=0, keepdims=True)
            _mg = _mas0.sum(axis=(2, 3), keepdims=True); _ng = _num0.sum(axis=(2, 3), keepdims=True)
            _mtg = _mg.sum(axis=0, keepdims=True)
            _rowok = _mtz > 0.0
            _fm = jnp.where(_rowok, _mz / jnp.where(_rowok, _mtz, 1.0),
                            _mg / jnp.maximum(_mtg, 1e-300))               # (NBINS,nlev,nlat,1)
            _fn = jnp.where(_rowok, _nz / jnp.where(_rowok, _mtz, 1.0),
                            _ng / jnp.maximum(_mtg, 1e-300))               # number per kg sulfate
            mas = jnp.where(_had[None], _mas0 * _ratio[None], _fm * _mt1[None])
            num = jnp.where(_had[None], _num0 * _ratio[None], _fn * _mt1[None])
            del _num0, _mas0
        else:
            num = qb[:NA]; mas = qb[NA:2*NA]
            so2 = qb[2*NA]; h2so4 = qb[2*NA + 1]
        # post-advection, pre-floor: separates transport (non-polar) from the
        # polar-cap overwrite (both happen inside advect_step_batch)
        M_adv = Mbur(mas); M_adv_np = Mbur(mas, 'np'); M_adv_pol = Mbur(mas, 'pol')
        #post-advection floor. The MASS floor is budgeted below (cumB['floor']);
        #the NUMBER floor was not, and it is the bigger one: the ultrafine number
        #field has a ~3000x vertical gradient (the CARMA tropopause mode), which
        #the PPM reconstruction cannot represent without undershoot, so clipping
        #negatives silently CREATES number. Track it so N/N0 can never again be
        #read without knowing how much of it the floor manufactured.
        N_pre = float((num.sum(0) * A_j).sum())
        num = jnp.maximum(num, 0.0); mas = jnp.maximum(mas, 0.0)
        so2 = jnp.maximum(so2, 0.0); h2so4 = jnp.maximum(h2so4, 0.0)
        nfloor_cum += float((num.sum(0) * A_j).sum()) - N_pre
        M_flr = Mbur(mas)                          # floor-only contribution
        num.block_until_ready(); t_adv = time.time() - ta
        u0, v0, w0 = u1, v1, w1

        if os.environ.get('DEBUG') and s == 0:
            fn = np.asarray(jnp.isfinite(num).all(axis=(1,2,3)))
            fm = np.asarray(jnp.isfinite(mas).all(axis=(1,2,3)))
            print(f"  [dbg] after ADVECT: num finite {fn.all()} (bad bins {np.where(~fn)[0]}), "
                  f"mas finite {fm.all()} (bad bins {np.where(~fm)[0]})", flush=True)

        # ---- 2. microphysics forced by the meteorology's T (+ dT_rad only under
        # RAD_T_FEEDBACK=1), p ----
        # One call per step; DT_MICRO spans the whole STEP_HOURS interval. T is
        # the snapshot at the step start (it0), consistent with the winds.
        # RAD_T_FEEDBACK=0 (default): T_met alone -- dT_rad is diagnostic and is
        # seen only by the radiation's emission temperature. =1: plus the
        # accumulated radiative increment (the aerosol -> radiation ->
        # temperature -> microphysics feedback path). T3d also drives the OH
        # number density, the settling and the wet-size diagnostics below.
        tm = time.time()
        T3d = temp(it0) + np.asarray(dT_rad) if RAD_T_FEEDBACK else temp(it0)
        # RH is read here, OUTSIDE the MICRO branch, because it is not only a
        # micro input: wet settling (WET_SETTLING) and every wet size diagnostic
        # (WET_OPTICS) need it too. It used to be read inside the MICRO=full
        # branch, which left it undefined -- a NameError one step in -- for any
        # other micro mode.
        rh3d = relhum(it0)
        #run microphysics on the updated temperature...clip_add and clip_rem over the mass created/destroyed (not truly mass conserving)
        if MICRO_MODE == 'off':
            # transport only: nothing evolves the bins, and with no two-moment
            # consistency clip there is no clip mass to report either. The jnp
            # arrays pass straight through the np/jnp round-trip below.
            num_np, mas_np, clip_add, clip_rem = num, mas, 0.0, 0.0
        elif MICRO_MODE == 'full':
            # full chain: SO2+OH consumes so2 -> h2so4; nucleation and
            # condensation move h2so4 into the bins, so aerosol M genuinely
            # grows here (the 'micro' budget stage is now a real source)
            # OH: SZA-parabola diurnal (per substep) or CESM's field (constant
            # over the step). run_microphysics_full accepts either shape.
            oh3d = oh_sza(it0) if OH_SZA else oh_molec(it0, T3d)
            # DEBUG path: dump the exact micro inputs (post-advection state)
            # and exit, so the detonating cell can be hunted offline without
            # paying for the full-grid micro pass (debug_nucleation/)
            # DUMP_PREMICRO_STEP picks WHICH step's pre-micro state to dump
            # (default 0 = fresh). For step>0 the microphysics runs normally
            # on steps 0..N-1 first, so the dumped state is the evolved (number
            # spun-up) state -- used to isolate the evolved-state OOD component
            # of the mass drift. File is step-tagged so the fresh dump survives.
            _dump_step = int(os.environ.get('DUMP_PREMICRO_STEP', '0'))
            if os.environ.get('DUMP_PREMICRO') and s == _dump_step:
                _fn = ('debug_nucleation/premicro_state.npz' if _dump_step == 0
                       else f'debug_nucleation/premicro_state_step{_dump_step}.npz')
                np.savez(_fn,
                         num=np.asarray(num), mas=np.asarray(mas),
                         so2=np.asarray(so2), h2so4=np.asarray(h2so4),
                         T3d=np.asarray(T3d), pres3d=np.asarray(pres3d),
                         rh3d=np.asarray(rh3d), oh3d=np.asarray(oh3d),
                         lat=lat, lon=lon, plev=PLEV_PA)
                print(f'  [dbg] pre-micro state (step {_dump_step}) dumped to '
                      f'{_fn}, exiting', flush=True)
                return
            num_np, mas_np, so2_np, h2so4_np, clip_add, clip_rem = \
                run_microphysics_full(num, mas, so2, h2so4, T3d, pres3d,
                                      rh3d, oh3d, A)
            so2 = jnp.asarray(so2_np); h2so4 = jnp.asarray(h2so4_np)
        else:
            num_np, mas_np, clip_add, clip_rem = run_microphysics(num, mas, T3d,
                                                                  pres3d, A)
        num = jnp.asarray(num_np); mas = jnp.asarray(mas_np)
        clip_add_cum += clip_add; clip_rem_cum += clip_rem
        #Burden after coagulation (mass change due to coagulation and two-moment clip)
        M_mic = Mbur(mas)                          # coag + two-moment clip (+cond/nuc growth in MICRO=full)
        t_mic = time.time() - tm

        # ---- 2b. gravitational settling (the aerosol's one true sink) ----
        # implicit upwind column sweep (settling.py); mass crossing the bottom
        # face exits the model -- sedimentation across the band bottom toward
        # tropospheric removal. Runs after micro so freshly grown particles
        # settle at their new sizes within the same step.
        ts_set = time.time()
        if SETTLE_ENABLE:
            # WET size for the fall speed. TOMAS carries dry SO4, but the particle
            # that falls is an H2SO4/H2O droplet: it is larger (Dp up to ~1.5x) and
            # less dense (1730 down to ~1300 kg/m3) than the dry core, and v_g
            # depends on both. Using the dry DP_BIN/RHO_AER understated the fall
            # speed and so suppressed the model's ONLY true aerosol sink -- directly
            # relevant to the unbounded burden growth in the 1-year run.
            # WET_SETTLING=0 restores the dry sizing for reproducing older runs.
            if WET_SETTLING:
                _wt3d = settling.equilibrium_wt_field(T3d, rh3d)
                _dpw, _rhow = settling.wet_size_field(MMID, _wt3d, RHO_AER)
            else:
                _dpw, _rhow = jnp.asarray(DP_BIN * 1e-9), RHO_AER
            num, mas, out_n, out_m = aer_settle(
                num, mas, jnp.asarray(T3d), jnp.asarray(pres3d),
                jnp.asarray(DP), STEP_SEC, _dpw, _rhow)
            # out_m is q*dp units per (bin,lat,lon); weighting by the latitude
            # metric puts it in the same burden units as Mbur, so the 'settle'
            # budget stage below should equal -settle_out to roundoff. That
            # invariant only actually holds if the weight is W_LAT -- Mbur uses
            # A_j = DP * W_LAT, so plain cos(lat) here left the two disagreeing
            # by the O(dphi^2) metric difference plus the zeroed +-90 rows.
            settle_out = float((out_m.sum(0)
                                * jnp.asarray(W_LAT)[:, None]).sum())
            settle_cum += settle_out
            num.block_until_ready()                # honest timing (settle_step is lazy)
        M_set = Mbur(mas)                          # after settling (== M_mic if off)
        t_settle = time.time() - ts_set
        if PROFILE:
            print(f"  [prof] s={s} read={t_read:.2f}s advect={t_adv:.2f}s "
                  f"micro={t_mic:.2f}s settle={t_settle:.2f}s (nsub={nsub})", flush=True)

        if os.environ.get('DEBUG') and s == 0:
            fn = np.isfinite(num_np).all(axis=(1,2,3)); fm = np.isfinite(mas_np).all(axis=(1,2,3))
            print(f"  [dbg] after MICRO : num finite {fn.all()} (bad bins {np.where(~fn)[0]}), "
                  f"mas finite {fm.all()} (bad bins {np.where(~fm)[0]})", flush=True)
            if not fn.all():
                b = np.where(~fn)[0][0]
                lev,la,lo = np.where(~np.isfinite(num_np[b]))
                print(f"  [dbg] first bad bin {b}: {len(lev)} bad cells; "
                      f"e.g. lev={lev[:3]} lat={la[:3]} lon={lo[:3]}", flush=True)

        # ---- 3. OPEN vertical BC: refill boundary slabs from MAM4 ----
        # (after coag so the pinned levels are exactly the CESM reservoir at
        # each step boundary; NO mass fixer -- open system, burden may change)
        tb = time.time()
        # refill the top and bottom N_BC_TOP/BOT levels from the reservoir at it1.
        # BC_EDGE=open: SKIP the aerosol reset -- let the edge aerosol levels evolve
        # so settling draws on the real (depleting) concentration and self-limits,
        # instead of a pinned reservoir value that fixes a constant outflow. The
        # gases follow this same choice (see _bc_gas_default0); 'clamp' is the
        # original frozen behaviour and the default only for sealed faces.
        if _bc_edge0 != 'open':
            num_top, mas_top = aer_fill(it1, lev_top)
            num_bot, mas_bot = aer_fill(it1, lev_bot)
            num = num.at[:, :N_BC_TOP].set(jnp.asarray(num_top))
            mas = mas.at[:, :N_BC_TOP].set(jnp.asarray(mas_top))
            num = num.at[:, -N_BC_BOT:].set(jnp.asarray(num_bot))
            mas = mas.at[:, -N_BC_BOT:].set(jnp.asarray(mas_bot))
        # gases get the same open-BC treatment from CESM's SO2/H2SO4 fields
        so2_top, h2so4_top = read_gases(ds_gas, it1, lev_top)
        so2_bot, h2so4_bot = read_gases(ds_gas, it1, lev_bot)
        # BC_GAS=flux: give the gases the SAME flux boundary as the aerosol instead
        # of a Dirichlet clamp. Why this matters: clamping H2SO4 at an edge level
        # while the aerosol there is open (BC_EDGE=open) is an infinite gas SOURCE
        # feeding nucleation with no particle SINK, so ultrafine number piles up
        # without limit. Measured at the 13.3 hPa top level over 24 h: the level
        # went from 0.3% to ~50% of the model's total number, with a grown-looking
        # 6-8 nm mode (bin 0 only 2.1%) -- i.e. continuous nucleation, not
        # transport. 'clamp' keeps the historical CESM-forced gases; 'flux' is the
        # self-consistent choice once the faces are open, at the cost of the gases
        # no longer being pinned to CESM, and is the DEFAULT whenever the aerosol
        # faces are open (2026-07-30 -- it was 'clamp' unconditionally before).
        if _bc_gas0 != 'flux':
            so2   = so2.at[:N_BC_TOP].set(jnp.asarray(so2_top))
            h2so4 = h2so4.at[:N_BC_TOP].set(jnp.asarray(h2so4_top))
            so2   = so2.at[-N_BC_BOT:].set(jnp.asarray(so2_bot))
            h2so4 = h2so4.at[-N_BC_BOT:].set(jnp.asarray(h2so4_bot))
        #burden after the BC refill
        M_bc = Mbur(mas)                           # after open-BC refill (== step-end burden)

        # ---- staged mass budget: accumulate each source (normalized by M0) ----
        # checkpoints telescope, so sum(cumB) == M/M0 - 1 exactly.
        cumB['adv_np']  += (M_adv_np - M_start_np) / M0
        cumB['adv_pol'] += (M_adv_pol - M_start_pol) / M0
        cumB['floor']   += (M_flr - M_adv) / M0
        cumB['micro']   += (M_mic - M_flr) / M0
        cumB['settle']  += (M_set - M_mic) / M0
        cumB['bc']      += (M_bc - M_set) / M0
        # attribution INSIDE adv_np/adv_pol (not a new telescoping stage, so the
        # closure identity above is untouched): how much of the advective change
        # was real exchange through the now-open slab faces. What is left over is
        # the scheme's genuine numerical non-conservation.
        cumV['in'] += vf_in / M0
        cumV['out'] -= vf_out / M0

        # ---- 4. polar-cap refresh: freeze target for NEXT step's advection ----
        if pol_idx.size:
            #polar cells are reset at all vertical levels to MAM4..not just the BC
            num_pol, mas_pol = aer_fill(it1, klevs, lat_idx=pol_idx)
            # gases refresh from CESM SO2/H2SO4 at the same polar rows
            so2_pol, h2so4_pol = read_gases(ds_gas, it1, klevs, lat_idx=pol_idx)
            #updates the frozen polar values for the next step's advection
            qfroz = qfroz.at[:, :, pol_idx].set(
                jnp.concatenate([jnp.asarray(num_pol), jnp.asarray(mas_pol),
                                 jnp.asarray(so2_pol)[None],
                                 jnp.asarray(h2so4_pol)[None]], axis=0))

        # ---- 4a. reservoir refresh for the open vertical faces ----------------
        # With ADV_WCONT the slab's faces are open and qfroz's TOP and BOTTOM
        # levels are the concentration that inflowing air carries (upwelling
        # through ~88 hPa brings tropopause air INTO the band). That exchange is
        # ~10%/day of the slab gross, so the value matters: keep it current with
        # the reservoir instead of leaving it at the day-0 field.
        # Runs AFTER the polar refresh on purpose -- that block rewrites qfroz at
        # ALL levels on polar rows, so it would otherwise undo the BC_BOT_AER
        # scaling on the bottom face at high latitudes.
        if ADV_WCONT:
            num_r, mas_r = aer_fill(it1, lev_top)
            qfroz = qfroz.at[:NA, :N_BC_TOP].set(jnp.asarray(num_r))
            qfroz = qfroz.at[NA:2*NA, :N_BC_TOP].set(jnp.asarray(mas_r))
            num_r, mas_r = aer_fill(it1, lev_bot)
            qfroz = qfroz.at[:NA, -N_BC_BOT:].set(jnp.asarray(num_r) * BC_BOT_AER)
            qfroz = qfroz.at[NA:2*NA, -N_BC_BOT:].set(jnp.asarray(mas_r) * BC_BOT_AER)
            qfroz = qfroz.at[2*NA, :N_BC_TOP].set(jnp.asarray(so2_top))
            qfroz = qfroz.at[2*NA+1, :N_BC_TOP].set(jnp.asarray(h2so4_top))
            qfroz = qfroz.at[2*NA, -N_BC_BOT:].set(jnp.asarray(so2_bot))
            qfroz = qfroz.at[2*NA+1, -N_BC_BOT:].set(jnp.asarray(h2so4_bot))

        if PROFILE:
            print(f"  [prof] s={s} bc+polar={time.time()-tb:.2f}s", flush=True)

        # ---- 5. radiation: evolved bins -> heating -> prognostic dT ----
        # Runs on the end-of-step state (post advect/coag/BC) at hour it1.
        # anomaly mode: heating difference vs reference MAM4 aerosol binned at
        # the same hour, same meteorology -- isolates the radiative effect of
        # the aerosol evolution (see radiation.py docstring). dT feeds the
        # NEXT step's microphysics only under RAD_T_FEEDBACK=1; under 0 it is
        # the radiation's own state (emission temperature of the perturbed call).
        # hr_kday is the heating rate of this call [K/day] (anomaly: hr1 - hr0,
        # i.e. the aerosol heating less the LW damping of dT_rad; full: hr;
        # flux: the absolute hr), kept for the frames until the next call.
        arf_toa = float('nan')
        if rad is not None and (s + 1) % RAD_EVERY == 0:
            tr = time.time()
            #radiation timestep
            dt_rad_sec = STEP_SEC * RAD_EVERY
            #when initial state is perturbed
            # training-data sampler: the evolved bins and the dT_rad the
            # radiation is about to see (T_eff = T_CESM + dT_rad), at hour it1
            if RAD_SAMPLE_EVERY and ((s + 1) // RAD_EVERY) % RAD_SAMPLE_EVERY == 0:
                savez_atomic(f'rad_sample_{OUT_TAG}_s{s:04d}.npz',
                             num=np.asarray(aer_to_bins(num, mas)[0], np.float32),
                             dT_rad=np.asarray(dT_rad, np.float32),
                             hour_abs=np.int64(H0 + it1), step=np.int64(s),
                             klevs=np.array(klevs), plev_pa=PLEV_PA,
                             inj_cfg=INJ_CFG, inj_cfg_keys=INJ_CFG_KEYS)
            if RAD_MODE == 'anomaly':
                # the reference aerosol is re-binned from MAM4 only when the
                # hour's reference solve is NOT already in the cache (a hit
                # serves the stored, bit-identical fields and needs no bins)
                # under MET_SRC=file the band temperature of BOTH calls is the
                # file's (the archive's T would put a (T_file - T_CESM) Planck
                # term into the anomaly); the reference solve is then not an
                # archive-only quantity and is never served from the cache
                T_met_band = temp(it1) if MET_SRC == 'file' else None
                if T_met_band is None and rad.ref_ready(H0 + it1):
                    num_ref = None
                else:
                    num_ref, _ = aer_fill_rad(it1, klevs)
                    num_ref = jnp.asarray(num_ref)
                #compute the heating anomaly with the current aerosol and heating states relative to the reference value
                #hr_anom = HR(num)-HR(num_ref)
                _kw = {} if RAD_T_FEEDBACK else dict(optics_at_met=True)
                hr_anom, drad1, drad0 = rad.heating_anomaly(
                    H0 + it1, aer_to_bins(num, mas)[0], num_ref, dT_rad, T_band=T_met_band, **_kw)
                #update the radiative temperature increment
                dT_rad = dT_rad + hr_anom * dt_rad_sec
                hr_kday = np.asarray(hr_anom, np.float64) * 86400.0
                # aerosol radiative effect at TOA (evolved - reference), W/m2,
                # area-weighted global mean; positive = net warming
                # W_LAT, not plain cos(lat): same area metric as A_j and the
                # burden diagnostics. cos(lat) is the ADV_METRIC=0 legacy weight
                # and gives the +-90 rows EXACTLY zero area. The difference on a
                # global mean is ~0.003% for a tropically-peaked plume, so this
                # is for consistency, not accuracy.
                w_lat = W_LAT[:, None] * np.ones((nlat, nlon))
                dn = np.asarray(drad1['sw_dn_toa'])   # same in both calls
                #total outgoing flux at TOA with evolved state versus baseline for the same timestep
                up1 = np.asarray(drad1['sw_up_toa']) + np.asarray(drad1['olr'])
                up0 = np.asarray(drad0['sw_up_toa']) + np.asarray(drad0['olr'])
                #energy leaving difference at TOA
                #up1>up0 means evolved aerosol is letting more energy escape (our goal)
                arf_toa = float(((up0 - up1) * w_lat).sum() / w_lat.sum())
                #AOD area weighted with the perturbed aerosol
                aod_gm = float((np.asarray(drad1['aod550']) * w_lat).sum()
                               / w_lat.sum())
            elif RAD_MODE == 'flux':
                # no feedback: the evolved aerosol at the meteorology's own
                # temperature, dT_rad untouched (stays 0); see RAD_MODE
                T_met_band = temp(it1) if MET_SRC == 'file' else None
                hr_flux, drad1 = rad.heating(H0 + it1, aer_to_bins(num, mas)[0],
                                             T_band_override=T_met_band, split=True)
                hr_kday = np.asarray(hr_flux, np.float64) * 86400.0
                w_lat = W_LAT[:, None] * np.ones((nlat, nlon))
                aod_gm = float((np.asarray(drad1['aod550']) * w_lat).sum()
                               / w_lat.sum())
            #when initial state is not perturbed, just compute the heating and update the temperature increment
            else:                                     # 'full'
                T_eff_band = temp(it1) + np.asarray(dT_rad)
                # keyword only when used: a radiation engine without T_optics_band
                # (e.g. a radiation emulator's heating) still works
                _kw = {} if RAD_T_FEEDBACK else dict(T_optics_band=temp(it1))
                hr, drad1 = rad.heating(H0 + it1, aer_to_bins(num, mas)[0],
                                        T_band_override=T_eff_band, **_kw)
                dT_rad = dT_rad + hr * dt_rad_sec
                hr_kday = np.asarray(hr, np.float64) * 86400.0
                # W_LAT, not plain cos(lat): same area metric as A_j and the
                # burden diagnostics. cos(lat) is the ADV_METRIC=0 legacy weight
                # and gives the +-90 rows EXACTLY zero area. The difference on a
                # global mean is ~0.003% for a tropically-peaked plume, so this
                # is for consistency, not accuracy.
                w_lat = W_LAT[:, None] * np.ones((nlat, nlon))
                aod_gm = float((np.asarray(drad1['aod550']) * w_lat).sum()
                               / w_lat.sum())
            t_rad = time.time() - tr
            if PROFILE:
                print(f"  [prof] s={s} radiation={t_rad:.2f}s", flush=True)
            # diurnal mean of the instantaneous samples (see ARF_AVG_H). Only
            # full windows are reported as an average; before the window fills,
            # arf_avg is the partial mean and is flagged in the log.
            if np.isfinite(arf_toa):
                arf_hist.append(arf_toa)
        arf_avg = float(np.mean(arf_hist)) if len(arf_hist) else float('nan')
        arf_full = len(arf_hist) == arf_hist.maxlen

        # ---- diagnostics ----
        if (s + 1) % LOG_EVERY == 0 or s == N_STEPS - 1:
            #total number and mass burdens (weighted by A_j) at the end of this step
            Nb, Mb = burdens(num, mas)
            Nbi, Mbi = burdens_int(num, mas)
            #num and mas pulled back to numpy, as TOMAS bins (the size statistics are per bin)
            _nb, _mb = aer_to_bins(num, mas)
            num_np = np.asarray(_nb); mas_np = np.asarray(_mb)
            # number and mass at probe level (selected level)
            n_lev = num_np[:, KPROBE]; m_lev = mas_np[:, KPROBE]
            tot_n = n_lev.sum(0); tot_m = m_lev.sum(0)
            #mean particle mass
            mp = np.where(tot_n > 0, tot_m / np.maximum(tot_n, 1e-300), 0.0)  # kg/particle
            # ---- WET sizes at the probe level -------------------------------
            # Every size diagnostic below is reported for the H2SO4/H2O SOLUTION
            # DROPLET, not the dry SO4 core, because that is the particle the
            # optics integrate and the settling sees. Until 2026-08-03 they all
            # used DP_BIN/RHO_AER (dry) while r_eff was labelled "what the optics
            # sees" -- true when the Mie tables were also dry, false the moment
            # they became wet. It also made the numbers non-comparable to the SAI
            # literature, which quotes WET r_eff (~0.4-0.5 um).
            # Unlike DP_BIN this varies per (bin, lat, lon), since the equilibrium
            # composition follows local T and RH -- hence the double sums below.
            if WET_OPTICS:
                _wt_p = np.asarray(settling.equilibrium_wt_field(
                    T3d[KPROBE], rh3d[KPROBE]))                  # (nlat,nlon)
                _dpw = np.asarray(settling.wet_size(
                    MMID[:, None, None], _wt_p[None], RHO_AER)[0]) * 1e9  # nm
                _dpw_mp = np.asarray(settling.wet_size(
                    np.maximum(mp, 1e-30), _wt_p, RHO_AER)[0]) * 1e9      # nm
            else:
                _dpw = np.broadcast_to(DP_BIN[:, None, None], n_lev.shape)
                _dpw_mp = np.cbrt(np.maximum(mp, 0) / RHO_AER * 6.0/np.pi) * 1e9
            #mass-weighted mean diameter (nm) at probe level: sum_k <M_k> Dp_k / sum_k <M_k>
            dp_mean = np.where(tot_n > 0, _dpw_mp, 0.0)          # nm
            # <> is an AREA-WEIGHTED lat/lon mean, using the same W_LAT metric the
            # burden diagnostics use. A plain .mean() gives every latitude row equal
            # weight, so the |lat|>80 caps -- 11.5% of the rows but 1.5% of the area,
            # and stirred well-mixed with almost no aerosol -- drag the mean down.
            _wll = np.broadcast_to(W_LAT[:, None], dp_mean.shape)
            _wv = _wll * (tot_n > 0)
            meanDp = float((dp_mean * _wv).sum() / max(_wv.sum(), 1e-300))
            # number-mean Dp at probe level: sum_k <N_k> Dp_k / sum_k <N_k>
            # (<> = area-weighted mean over lat/lon). Weighted toward the abundant
            # fine end, so coagulation (ultrafine merging away) makes it rise.
            #geometric mean number conc
            gmn = ((n_lev * W_LAT[None, :, None]).sum(axis=(1, 2))
                   / (W_LAT.sum() * n_lev.shape[2]))           # (NBINS,)
            # Double sum over (bin, cell) rather than gmn . DP_BIN: the wet
            # diameter is cell-dependent, so it cannot be pulled out of the
            # lat/lon average. Reduces EXACTLY to the old one-dimensional dot
            # product when Dp is cell-independent (WET_OPTICS=0).
            _wnum = n_lev * W_LAT[None, :, None]               # (NBINS,nlat,nlon)
            _wmas = m_lev * W_LAT[None, :, None]
            #number-mean Dp (nm) at probe level: sum_k <N_k> Dp_k / sum_k <N_k>
            meanDp_num = float((_wnum * _dpw).sum()
                               / max(_wnum.sum(), 1e-300))
            # TRUE mass-weighted mean Dp: sum_k <M_k> Dp_k / sum_k <M_k>. Note
            # this is NOT what `meanDp` above is -- that one is the diameter of
            # the MEAN-MASS particle, D(M/N), which is dominated by the ultrafine
            # NUMBER and so falls when a nucleation mode appears even while all
            # the mass is moving to larger sizes. Optics care about this one.
            gmm = ((m_lev * W_LAT[None, :, None]).sum(axis=(1, 2))
                   / (W_LAT.sum() * m_lev.shape[2]))           # (NBINS,)
            meanDp_mass = float((_wmas * _dpw).sum()
                                / max(_wmas.sum(), 1e-300))
            # EFFECTIVE RADIUS r_eff = <N Dp^3> / <N Dp^2> / 2 -- the PRIMARY size
            # diagnostic, and the one the dashboard plots.
            # This is the size the RADIATION actually integrates: radiation.py builds
            # its Mie tables on the WET droplet diameter (build_wet_mie_tables) and
            # weights them by the per-bin NUMBER mixing ratios, so the
            # third-over-second number moment on those same wet diameters is the mean
            # size that reproduces the layer's optics, with no extra assumption. The
            # other three diagnostics above are all defensible averages of the
            # distribution but none of them is what the optics sees.
            # It is a WET radius, so it is directly comparable to the SAI literature's
            # 0.4-0.5 um; the pre-2026-08-03 values were dry and ran ~10-50% smaller.
            # It is also the best-conditioned of the four. The sub-10 nm bins are not
            # a converged model quantity (PROCESSES.md 2.6: nucleation-mode number
            # moves ~60x under operator reordering); dropping them shifts r_eff by
            # 2-3% in the mature phase, against 15x for meanDp_num. All NBINS are
            # kept here regardless, because all of them are what radiation.py sums.
            # RADIUS, not diameter, because that is the form the SAI literature
            # quotes (r_eff ~ 0.4-0.5 um for continuous multi-Tg injection) -- so it
            # is comparable on sight, with no factor-of-2 to lose. Everything else
            # on this panel is a diameter; the axis label carries the distinction.
            # Ratio of AREA-WEIGHTED moments, not an area-weighted mean of per-cell
            # r_eff. The layer's optical depth is set by the summed moments, so the
            # ratio of the sums is the size reproducing it; averaging per-cell ratios
            # would instead give a near-empty polar cell the same say as a plume cell.
            reff = float(0.5 * (_wnum * _dpw ** 3).sum()
                         / max((_wnum * _dpw ** 2).sum(), 1e-300))
            ts['hours'].append(it1); ts['Nburden'].append(Nb); ts['Mburden'].append(Mb)
            ts['nsub'].append(int(nsub)); ts['Nmin'].append(float(num.min()))
            ts['Nmax'].append(float(num.max())); ts['meanDp_nm'].append(meanDp)
            ts['meanDp_num_nm'].append(meanDp_num)
            ts['meanDp_mass_nm'].append(meanDp_mass)
            ts['reff_nm'].append(reff)
            ts['clipMadd_cum'].append(clip_add_cum)
            ts['clipMrem_cum'].append(clip_rem_cum)
            for k in ('adv_np', 'adv_pol', 'floor', 'micro', 'settle', 'bc'):
                ts['B_' + k].append(cumB[k])
            ts['B_vf_in'].append(cumV['in']); ts['B_vf_out'].append(cumV['out'])
            ts['Nfloor_cum'].append(nfloor_cum)
            # gas-phase burdens + the two cumulative sulfur bookends
            # (injected SO2 in, settled aerosol mass out)
            ts['SO2burden'].append(gas_burden(so2))
            ts['H2SO4burden'].append(gas_burden(h2so4))
            ts['injSO2_cum'].append(inj_cum)
            ts['settleM_cum'].append(settle_cum)
            dT_np = np.asarray(dT_rad)
            ts['dT_min'].append(float(dT_np.min()))
            ts['dT_max'].append(float(dT_np.max()))
            # AREA-WEIGHTED rms over the full slab. A plain .mean() weights every
            # latitude row equally and biased this ~20% low (0.616 vs 0.769 K at
            # day 203 of prod1yr) because the near-zero |lat|>80 caps are 11.5%
            # of the rows but 1.5% of the area. dT_min/dT_max are single-cell
            # extrema, so no weighting applies to them.
            _wdt = np.broadcast_to(W_LAT[None, :, None], dT_np.shape)
            ts['dT_rms'].append(float(np.sqrt((dT_np ** 2 * _wdt).sum()
                                              / _wdt.sum())))
            # heating rate of the last radiation call, K/day, same weighting
            ts['hr_min'].append(float(hr_kday.min()))
            ts['hr_max'].append(float(hr_kday.max()))
            ts['hr_rms'].append(float(np.sqrt((hr_kday ** 2 * _wdt).sum()
                                              / _wdt.sum())))
            ts['arf_toa'].append(arf_toa)
            ts['arf_toa_avg'].append(arf_avg)
            ts['aod550'].append(aod_gm)
            print(f"  step {s+1:3d}/{N_STEPS} (h{it1:4d}) {time.time()-t0:6.0f}s nsub={nsub:2d}  "
                  f"N/N0 {Nb/N0:.4f} (int {Nbi/N0i:.4f}, floor {nfloor_cum/N0:+.3f})  "
                  f"M/M0 {Mb/M0:.4f} (int {Mbi/M0i:.4f})  "
                  f"r_eff {reff:6.1f}nm  "
                  f"Dp(M/N) {meanDp:6.1f}nm  Dp(massw) {meanDp_mass:6.1f}nm  "
                  f"Dp(num) {meanDp_num:6.1f}nm  "
                  f"clipM/M0 +{clip_add/M0:.1e}/{clip_rem/M0:.1e}  "
                  f"finite {bool(jnp.isfinite(num).all())}",
                  flush=True)
            if rad is not None:
                # report the diurnal mean first (that is the physical number);
                # 'inst' is the raw single-solar-time sample it is built from,
                # and 'partial' marks a window that has not filled a full day yet
                print(f"           rad: dT[K] min {dT_np.min():+.4f} max {dT_np.max():+.4f} "
                      f"rms {ts['dT_rms'][-1]:.5f}  "
                      f"ARF_toa {arf_avg:+.4f} W/m2"
                      + ("" if arf_full else "(partial)")
                      + f" [inst {arf_toa:+.4f}, n={len(arf_hist)}]"
                      f"  AOD550 {aod_gm:.4f}",
                      flush=True)
            # staged mass budget (cumulative, /M0). sum should equal M/M0-1.
            csum = sum(cumB.values())
            print(f"           budget/M0: advNP {cumB['adv_np']:+.2e}  "
                  f"advPol {cumB['adv_pol']:+.2e}  floor {cumB['floor']:+.2e}  "
                  f"micro {cumB['micro']:+.2e}  settle {cumB['settle']:+.2e}  "
                  f"bc {cumB['bc']:+.2e}  |  "
                  f"sum {csum:+.2e} vs M/M0-1 {Mb/M0-1:+.2e} "
                  f"(closure {csum-(Mb/M0-1):+.1e})", flush=True)
            if ADV_VFLUX:
                # advNP+advPol is transport; of that, vfIn-vfOut is REAL exchange
                # through the open slab faces. The remainder is the scheme's
                # numerical non-conservation -- the number to drive to zero.
                _vnet = cumV['in'] + cumV['out']
                _resid = cumB['adv_np'] + cumB['adv_pol'] - _vnet
                print(f"           vface/M0 (net, +=into slab): "
                      f"top {cumV['in']:+.2e}  "
                      f"bot {cumV['out']:+.2e}  net {_vnet:+.2e}  ==> "
                      f"advection numerical residual {_resid:+.2e}", flush=True)
            if MICRO_MODE == 'full' or inj_dq is not None:
                # gas ledger: burdens are in the same Pa*cos(lat) units as M,
                # so SO2/S0 tracks depletion vs the CESM background start
                print(f"           gas: SO2 burden {gas_burden(so2):.3e} "
                      f"({gas_burden(so2)/max(S0,1e-300):.3f}x S0)  "
                      f"H2SO4 {gas_burden(h2so4):.3e}  "
                      f"inj_cum {inj_cum:.3e}  injH2SO4_cum {inj_h2so4_cum:.3e}  "
                      f"settled_cum {settle_cum:.3e} "
                      f"(={settle_cum/M0:+.2e} M0)", flush=True)

        if FRAME3D and ((s + 1) % FRAME3D_EVERY == 0 or s == N_STEPS - 1):
            # Per-cell size statistics of the DRY bin grid (DP_BIN is the bin
            # diameter the moments are defined on): geometric number-mean
            # diameter, its geometric standard deviation, the mass-weighted
            # geometric mean, and the WET effective diameter <N Dw^3>/<N Dw^2>
            # on the H2SO4/H2O droplet (settling.wet_size_field, the same size
            # the optics and the settling see; dry when WET_OPTICS=0) -- the
            # per-cell version of the probe-level reff_nm in the timeseries.
            _n3b, _m3b = aer_to_bins(num, mas)
            _n3 = np.asarray(_n3b, np.float64); _m3 = np.asarray(_m3b, np.float64)
            _ntot = _n3.sum(axis=0); _mtot = _m3.sum(axis=0)
            _lnD = np.log(DP_BIN)[:, None, None, None]                 # ln nm
            _nw = np.maximum(_ntot, 1e-300)
            _lndg = (_n3 * _lnD).sum(axis=0) / _nw
            _sig = np.exp(np.sqrt(np.maximum((_n3 * (_lnD - _lndg) ** 2).sum(axis=0) / _nw, 0.0)))
            _lndm = (_m3 * _lnD).sum(axis=0) / np.maximum(_mtot, 1e-300)
            if WET_OPTICS:
                _wt3 = settling.equilibrium_wt_field(T3d, rh3d)
                _dw3 = np.asarray(settling.wet_size_field(MMID, _wt3, RHO_AER)[0])   # (NBINS,nlev,nlat,nlon) m
            else:
                _dw3 = np.broadcast_to((DP_BIN * 1e-9)[:, None, None, None], _n3.shape)
            _deff = (_n3 * _dw3 ** 3).sum(axis=0) / np.maximum((_n3 * _dw3 ** 2).sum(axis=0), 1e-300)
            _has = _ntot > 0
            _f3 = dict(step=np.int64(s), hour_abs=np.int64(H0 + it1), hour=np.int64(it1),
                       num=np.asarray(num, np.float32), mas=np.asarray(mas, np.float32),
                       so2=np.asarray(so2, np.float32), h2so4=np.asarray(h2so4, np.float32),
                       dT_rad=np.asarray(dT_rad, np.float32),
                       hr_kday=np.asarray(hr_kday, np.float32),
                       ntot=_ntot.astype(np.float32), mtot=_mtot.astype(np.float32),
                       dg_num_nm=np.where(_has, np.exp(_lndg), 0.0).astype(np.float32),
                       sigma_g=np.where(_has, _sig, 0.0).astype(np.float32),
                       dg_mass_nm=np.where(_mtot > 0, np.exp(_lndm), 0.0).astype(np.float32),
                       deff_wet_nm=np.where(_has, _deff * 1e9, 0.0).astype(np.float32),
                       plev_pa=PLEV_PA, lat=lat, lon=lon, dp_bin_nm=DP_BIN,
                       units=np.array('num #/kg per bin; mas kg(SO4)/kg per bin; so2,h2so4 kg/kg; dT_rad K; hr_kday K/day (last radiation call); '
                                      'ntot #/kg; mtot kg/kg; diameters nm (dg_* dry on DP_BIN, deff wet droplet); '
                                      'aod550 -; toa_* W/m2 (net = sw_dn - sw_up - olr of the evolved call, '
                                      'arf = up_ref - up_evolved, positive = net warming by the aerosol)'))
            if rad is not None and (s + 1) % RAD_EVERY == 0:
                _d1 = {k: np.asarray(drad1[k], np.float32) for k in ('sw_dn_toa', 'sw_up_toa', 'olr', 'aod550')}
                _f3.update(aod550=_d1['aod550'], toa_sw_dn=_d1['sw_dn_toa'],
                           toa_net_abs=_d1['sw_dn_toa'] - _d1['sw_up_toa'] - _d1['olr'])
                if RAD_MODE == 'flux':
                    # everything a paired difference needs (W/m2; K/day on the band)
                    _f3.update(toa_sw_up=_d1['sw_up_toa'], toa_olr=_d1['olr'],
                               sfc_sw_dn=np.asarray(drad1['sw_dn_sfc'], np.float32),
                               sfc_lw_dn=np.asarray(drad1['lw_dn_sfc'], np.float32),
                               hr_kday=(np.asarray(hr_flux) * 86400.0).astype(np.float32),
                               hr_sw_kday=(np.asarray(drad1['hr_sw_band']) * 86400.0).astype(np.float32))
                if RAD_MODE == 'anomaly':
                    _f3['toa_arf'] = ((np.asarray(drad0['sw_up_toa'], np.float32) + np.asarray(drad0['olr'], np.float32))
                                      - (_d1['sw_up_toa'] + _d1['olr']))
            os.makedirs('frames3d', exist_ok=True)
            savez_atomic(f'frames3d/step_{s:04d}.npz', **_f3)
            del _n3, _m3, _dw3, _f3

        if (s + 1) % FRAME_EVERY_STEPS == 0 or s == N_STEPS - 1:
            frames_num.append(np.asarray(num[:, KPROBE]).copy())
            frames_mas.append(np.asarray(mas[:, KPROBE]).copy())
            frames_dT.append(np.asarray(dT_rad[KPROBE]).copy())
            frames_hr.append(np.asarray(hr_kday[KPROBE], np.float32).copy())
            frames_so2.append(np.asarray(so2[KPROBE]).copy())
            frames_h2so4.append(np.asarray(h2so4[KPROBE]).copy())
            frame_hours.append(it1)
            _zm_capture(it1)
            # incremental checkpoint: frames/timeseries are otherwise only
            # written once at the very end, so a killed/crashed multi-hour run
            # loses everything. Overwrite a single _ckpt file each frame (cheap:
            # one probe-level slab) so partial results survive an early stop.
            # ALL THREE writes below are atomic (savez_atomic) -- plain np.savez
            # here cost the prod1yr frames history to an OOM kill mid-write.
            if len(frame_hours) % CKPT_EVERY_FRAMES == 0 or s == N_STEPS - 1:
                # thunks: under CKPT_ASYNC=0 each dict is built right before its
                # own write (as the inline calls did); under CKPT_ASYNC=1 all
                # three are built now, on this thread, and written in the
                # background. _h makes the 3-D state a private host copy then.
                _h = np.array if CKPT_ASYNC else np.asarray
                _tck = time.time()
                _ck_batch = [
                    (f'coupled_frames_{OUT_TAG}_ckpt.npz', lambda: dict(
                         frame_hours=np.array(frame_hours),
                         frames_num=np.stack(frames_num), frames_mas=np.stack(frames_mas),
                         frames_dT=np.stack(frames_dT),
                         frames_hr_kday=np.stack(frames_hr),
                         frames_so2=np.stack(frames_so2),
                         frames_h2so4=np.stack(frames_h2so4),
                         probe_hpa=PLEV_PA[KPROBE] / 100, xk=XK_NP,
                         # the vertical/horizontal axes, so a zonal frame is
                         # plottable from this file alone rather than needing the
                         # final state beside it
                         plev_pa=PLEV_PA, lat=lat, lon=lon, **_zm_arrays())),
                    (f'coupled_timeseries_{OUT_TAG}_ckpt.npz', lambda: dict(
                         N0=N0, M0=M0,
                         inj_cfg=INJ_CFG, inj_cfg_keys=INJ_CFG_KEYS,
                         phys_cfg=PHYS_CFG, phys_cfg_keys=PHYS_CFG_KEYS,
                         **{k: np.array(v) for k, v in ts.items()})),
                ]
                # full 3-D prognostic state + every cumulative counter, so RESUME=1
                # can pick the run up from here. Written LAST, so the state is never
                # newer than the frames/ts it is meant to agree with -- a kill between
                # the writes leaves them AHEAD of the state instead, which the RESUME
                # branch trims back (see 'frames/ts ahead of the state' there).
                if STATE_CKPT:
                    _ck_batch.append((f'coupled_state_{OUT_TAG}_ckpt.npz', lambda: dict(
                             s_done=s, nbins=NA, nlev=nlev, nlat=nlat, nlon=nlon,
                             step_hours=STEP_HOURS,
                             num=_h(num), mas=_h(mas),
                             so2=_h(so2), h2so4=_h(h2so4),
                             dT_rad=_h(dT_rad),
                             hr_kday=_h(hr_kday),
                             N0=N0, M0=M0, N0i=N0i, M0i=M0i, S0=S0,
                             clip_add_cum=clip_add_cum, clip_rem_cum=clip_rem_cum,
                             nfloor_cum=nfloor_cum, inj_cum=inj_cum,
                             inj_h2so4_cum=inj_h2so4_cum, settle_cum=settle_cum,
                             aod_gm=aod_gm,
                             cumB_keys=np.array(list(cumB.keys())),
                             cumB_vals=np.array(list(cumB.values())),
                             cumV_keys=np.array(list(cumV.keys())),
                             cumV_vals=np.array(list(cumV.values())),
                             arf_hist=np.array(list(arf_hist)),
                             inj_cfg=INJ_CFG, phys_cfg=PHYS_CFG,
                             phys_cfg_keys=PHYS_CFG_KEYS,
                             grid_cfg=GRID_CFG, grid_cfg_keys=GRID_CFG_KEYS)))
                _ckw.submit(_ck_batch)
                del _ck_batch
                if PROFILE:
                    print(f"  [prof] s={s} ckpt={time.time()-_tck:.2f}s "
                          f"({'async: snapshot+queue' if CKPT_ASYNC else 'sync write'})",
                          flush=True)

    # the last checkpoint must be on disk before the run reports done
    _ckw.join()
    if CKPT_ASYNC and PROFILE:
        print(f"  [prof] ckpt writer: main thread waited {_ckw.t_wait:.2f}s in total",
              flush=True)
    print(f"\n{'='*60}\nComplete in {time.time()-t0:.0f}s\n{'='*60}", flush=True)

    # ---- save ----]
    #full slab
    np.savez(f'coupled_final_{OUT_TAG}.npz',
             num=np.asarray(num), mas=np.asarray(mas),
             so2=np.asarray(so2), h2so4=np.asarray(h2so4),
             dT_rad=np.asarray(dT_rad),
             hr_kday=np.asarray(hr_kday, np.float32),
             inj_cfg=INJ_CFG, inj_cfg_keys=INJ_CFG_KEYS,
             phys_cfg=PHYS_CFG, phys_cfg_keys=PHYS_CFG_KEYS,
             xk=XK_NP, plev_pa=PLEV_PA, lat=lat, lon=lon, klevs=np.array(klevs),
             grid_cfg=GRID_CFG, grid_cfg_keys=GRID_CFG_KEYS)
    # Full slab EXCEPT the four probe-level (KPROBE) size diagnostics:
    #   reff_nm, meanDp_nm, meanDp_num_nm, meanDp_mass_nm
    # (an earlier version of this note omitted meanDp_mass_nm, then the dashboard's
    # primary curve became reff_nm -- whichever one the dashboard plots, it is just
    # as single-level as the others, so keep this list complete).
    # All lat/lon reductions here are W_LAT area-weighted; dT_min/dT_max are
    # single-cell extrema.
    # inj_cfg/inj_cfg_keys record the injection scenario that produced this series,
    # so a sweep's files are self-identifying independent of the TAG in the filename.
    np.savez(f'coupled_timeseries_{OUT_TAG}.npz', N0=N0, M0=M0,
             inj_cfg=INJ_CFG, inj_cfg_keys=INJ_CFG_KEYS,
             phys_cfg=PHYS_CFG, phys_cfg_keys=PHYS_CFG_KEYS,
             **{k: np.array(v) for k, v in ts.items()})
    #single level
    np.savez(f'coupled_frames_{OUT_TAG}.npz',
             inj_cfg=INJ_CFG, inj_cfg_keys=INJ_CFG_KEYS,
             phys_cfg=PHYS_CFG, phys_cfg_keys=PHYS_CFG_KEYS,
             frame_hours=np.array(frame_hours),
             frames_num=np.stack(frames_num), frames_mas=np.stack(frames_mas),
             frames_dT=np.stack(frames_dT),
             frames_hr_kday=np.stack(frames_hr),
             frames_so2=np.stack(frames_so2),
             frames_h2so4=np.stack(frames_h2so4),
             probe_hpa=PLEV_PA[KPROBE] / 100, xk=XK_NP,
             plev_pa=PLEV_PA, lat=lat, lon=lon, **_zm_arrays())
    print(f"saved coupled_final/timeseries/frames_{OUT_TAG}.npz "
          f"({len(frame_hours)} frames)", flush=True)


if __name__ == '__main__':
    main()
