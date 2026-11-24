"""Fully-linked coupled run: the production entry point (run_prod.py execs it).

The microphysics is the `tomas_jax.fast` reduced model of tomas-jax's gpu-fast
branch (models/tomas-jax pinned to 787c991; MICRO_ENGINE=fast, the default, via
microphysics/tomas_fast.py). MICRO_ENGINE=full (dev's make_step) needs a
tomas-jax dev checkout and fails on this pin when its step is first built.
Advection is the same swap coupling.py binds for itself, so the two agree on
transport. Which linking (meteorology source, climatology, submodule pins) is
run is chosen by a preset in configs/, applied by run_prod.py.

The emulator engine (MICRO_ENGINE=emulator) and the engine A/B harness
(MICRO_REF, MICRO_SAMPLE_EVERY) are not part of this repo (yet); an emulator
would fill the same microphysics slot behind MICRO_ENGINE.

IC/BC source is coupling.py's default, AER_SRC=mam4 (per-step dynamic from the
CESM hourly series). Pass AER_SRC=carma for the static CARMA reservoir instead.

Run: use run_prod.py beside this file, which sets the whole production
environment (GPU pin,
libcuda path, memory policy) and execs this file. Direct invocation for a short
test, once tomas-jax and jax-rrtmgp are importable -- `git submodule update
--init`, or set TOMAS_JAX_PATH / RRTMGP_PATH; coupling.py resolves both:

  INJ_SO2_TG_YR=0 N_HOURS=18 DEBUG=1 PROFILE=1 \
      OUT_TAG=smoke python3 driver_fast.py

Do NOT activate the tomas-jax .venv: it carries CPU-only jaxlib and no xarray,
so the run silently loses the GPU. System python3 with GPU jax is the combo.
"""
import os
import sys
import time
import functools

import numpy as np

# --- env defaults for this driver (N_BINS is deliberately NOT set: see below) ---
os.environ.setdefault('RAD', '1')
os.environ.setdefault('RAD_MODE', 'anomaly')
os.environ.setdefault('RAD_EVERY', '1')
os.environ.setdefault('MICRO', 'full')            # main() takes the full-chain path
os.environ.setdefault('STEP_HOURS', '6')
# domain: 1-150 hPa (changed 2026-07-29, was 12-100). 24 native levels =
# 1,327,104 cells, 2.18x the 11-level 12-100 band, so expect ~2.2x the step cost
# (micro is ~94% of it): ~170 -> ~370 s/step, a 90-day run ~17 h -> ~37 h.
# Rationale for each edge:
#   1 hPa top  -- MAM4 so4 there is 3.2e-17 kg/kg, 7 orders below the 13 hPa value,
#                 so the top face is effectively an aerosol-free inflow (correct for
#                 mesospheric subsidence) without needing a knob to zero it.
#   150 hPa bot -- MAM4 at 143 hPa is 0.6x the 88 hPa value, and crucially its
#                 LATITUDE structure is right: 7.5e-11 in the tropics (the TTL,
#                 already convectively scavenged, and where upwelling enters the
#                 band) vs 1.26e-9 at 60N (where STE brings real air in). The old
#                 BC_BOT_AER=0 hack zeroed both, right for the tropics and wrong
#                 for the extratropics.
# CAVEAT: dropping the floor from ~88 to 143 hPa DELAYS settling, which is the
# model's only true aerosol sink -- expect a longer effective lifetime and higher
# steady-state burden. There is still no wet removal anywhere, so SAI aerosol that
# settles into the 100-150 hPa layer lingers there.
os.environ.setdefault('P_LO_HPA', '1.0')
os.environ.setdefault('P_HI_HPA', '150.0')
# AER_SRC deliberately NOT set here (changed 2026-07-29): this driver used to force
# 'carma', which silently overrode coupling.py's own 'mam4' default, so every run
# through this file got a STATIC aerosol reservoir unless AER_SRC was passed
# explicitly. MAM4 is the right default -- it is per-step dynamic (bin_mam4 indexes
# the hourly h1 series; verified 100% of cells change over 24 h) and it is the only
# source that exists past the 1-week CARMA output, which multi-year runs need.
# CARMA is now explicit opt-in: AER_SRC=carma. NOTE all runs before this date
# (zonal90d, zonal4wk, carma_tomas_*) were CARMA/static -- not comparable.
os.environ.setdefault('OUT_TAG', 'tomas_fast')
# the fast engine's production presets (formerly in run_prod.py; see its git
# history for the timing behind them): 50000 cells/chunk is FASTER than the
# 250000 module default, and the stiffness sort is worth ~27% of micro.
os.environ.setdefault('FAST_CELL_CAP', '50000')
os.environ.setdefault('FAST_SORT', '1')
# NOTE: deliberately NOT setting N_BINS -> coupling runs at native NBINS=40,
# the only grid tomas_jax.fast supports.

if os.environ.get('STEP_HOURS', '6') != '6':
    sys.exit("driver_fast: STEP_HOURS must be 6 (parity with 6h step)")
# MICRO=off (transport only, for timing runs) is allowed through: it simply never
# calls a micro engine, so the 40-bin/6h parity constraints below do not apply to
# it, and a benchmark then measures advection through the PRODUCTION entry point
# rather than through a second launcher that can drift from it. The legacy 'coag'
# path is still refused -- this driver exists to swap a micro engine into the full
# chain, and pairing it with the pre-SAI model would silently be neither.
if os.environ.get('MICRO', 'full') not in ('full', 'off'):
    sys.exit("driver_fast: MICRO must be 'full' (or 'off' for a transport-only run)")
if os.environ.get('N_BINS', '0') not in ('0', '40'):
    sys.exit(f"driver_fast: the run is 40-bin -- unset N_BINS or set "
             f"40; got N_BINS={os.environ.get('N_BINS')}")

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import _paths                      # noqa: E402 -- puts src/ subdirs on sys.path

import jax.numpy as jnp

# NB the compat shim that used to live here (aliasing coupling.py's imported
# calc_oh_parabola to tomas-jax's calc_oh_concentration) is GONE and must not come
# back: the two take their arguments in the opposite order and
# calc_oh_concentration defaults to constant mode, so the alias made
# calc_oh_parabola(cos_sza, OH_PEAK) return cos(SZA) -- OH ~ 1 molec/cm3, i.e. no
# SO2 oxidation at all -- in every OH_SZA=1 run. coupling.py now owns the fit.

import coupling as C
from fct_fast import advect_step_batch as _fast_advect
# The engine adapters below MUST be imported BELOW the os.environ.setdefault
# block above and not at the top of the file: they import coupling.py, and
# coupling.py reads the environment at import time, so hoisting them changes the
# configuration the run uses. (This is one of the reasons ruff's import-sorting
# rule I001 is disabled -- see docs/REPO_LAYOUT.md.)
#
# MICRO_ENGINE selects WHICH engine fills the seam:
#   fast      (default) tomas_jax.fast.run_fast via microphysics/tomas_fast.py
#             (models/tomas-jax = gpu-fast 787c991), sulfate + water, 40 bins.
#   full      coupling.run_microphysics_full: tomas-jax dev's make_step. Needs
#             a dev checkout of models/tomas-jax; not usable on this pin.
MICRO_ENGINE = os.environ.get('MICRO_ENGINE', 'fast').lower()
_ENGINES = ('fast', 'full')
_ELSEWHERE = "it is not part of this repo"
if MICRO_ENGINE == 'emulator':
    sys.exit(f"driver_fast: MICRO_ENGINE=emulator -- {_ELSEWHERE}")
if MICRO_ENGINE not in _ENGINES:
    sys.exit(f"driver_fast: MICRO_ENGINE must be one of {_ENGINES}; got {MICRO_ENGINE!r}")
for _k in ('MICRO_REF', 'MICRO_SAMPLE_EVERY'):
    if os.environ.get(_k, '') not in ('', '0'):
        sys.exit(f"driver_fast: {_k} (engine A/B / emulator training samples) -- {_ELSEWHERE}")
_need = {MICRO_ENGINE}

# The adapter imports coupling.py, so it too must stay below the setdefault
# block above.
_ENGINE_FN = {'full': C.run_microphysics_full}   # coupling.py's own chain, captured before any swap
# the PHYS_CFG 'MICRO_ENGINE=full' stamp records the engine THIS driver installs
# (coupling.py stamps its own standalone choice at import)
C.PHYS_CFG[list(C.PHYS_CFG_KEYS).index('MICRO_ENGINE=full')] = float(
    C.MICRO_MODE == 'full' and MICRO_ENGINE == 'full')
if 'fast' in _need:
    from tomas_fast import (run_microphysics_full_fast, N_FAST_STEPS, FAST_DT,
                            FAST_CELL_CAP, FAST_SORT, MICRO_WRAP_DEVICE,
                            FAST_MULTIGPU, _n_devices)
    _ENGINE_FN['fast'] = run_microphysics_full_fast
# OH sampling: the SZA parabola is evaluated at C.OH_SUBSTEPS (default 60)
# substep centres per step; MICRO=full holds each sample over its dev steps.

# ---- the advection swap (same scheme coupling.py binds for itself) ----------
_ADV_CFL = float(os.environ.get('ADV_CFL', '0.5'))
_ADV_DTYPE = jnp.float32 if os.environ.get('ADV_F32', '1') != '0' else jnp.float64
# ADV_SCHEME selects the transport form:
#   ffsl advection/ffsl_lr.py -- fct_lr at large
#        Courant number (Lin & Rood's flux-form semi-Lagrangian extension): the
#        same flux-form sweeps, faces, budgets and limiter at ADV_FFSL_K=24
#        sub-steps instead of 70-250. advect 1.5 s/step vs 7.0; 30 days within
#        0.2 % of lr on the budgets (docs/PROCESSES.md 1.9). 2026-10-14.
#   lr   advection/fct_lr.py -- Lin-Rood FLUX form with air-mass
#        tracking. Tracer mass is conserved to ROUNDOFF (~1e-15/day) for any wind
#        field, divergent or not, so it does not depend on the winds satisfying
#        continuity. Costs ~5x more substeps (no integer-shift FFSL in flux form).
#   fast advection/fct_fast.py -- advective form with the 2026-07-25 fixes
#        (true grid spacing, cos(phi) metric, continuity omega, conserving polar
#        caps). Residual ~3e-4/day; cheaper. Kept as the validated fallback.
#   sl   advection/sl_lr.py -- semi-Lagrangian (dinosaur's operator, vendored in
#        sl_core.py): no CFL limit, ADV_SL_K sub-steps on the hourly winds, cubic
#        gather + quasi-monotone limiter, closed column + per-tracer mass fixer
#        (ADV_SL_FIXER), Dirichlet edge levels (coupling.py sets ADV_WCONT=0).
#        ~49 ms/step vs 6.6 s. docs/TRANSPORT_SL_HANDOFF.md. 2026-10-13.
#        THE DEFAULT (coupling.py ADV_SCHEME='sl', ADV_SL_K=12).
# 2026-11-17: the scheme is coupling.py's (C.ADV_SCHEME), not a second default
# of the driver's own. The two had drifted (coupling sl, driver ffsl): a bare
# run advected with ffsl while its header, its sl-only logic and its checkpoint
# stamp said sl.
_ADV_SCHEME = C.ADV_SCHEME
if _ADV_SCHEME == 'lr':
    from fct_lr import advect_step_batch as _adv_fn
elif _ADV_SCHEME == 'sl':
    import sl_lr
    _adv_fn = sl_lr.advect_step_batch
elif _ADV_SCHEME == 'ffsl':
    import ffsl_lr                 # fct_lr at large Courant number (2026-10-14)
    _adv_fn = ffsl_lr.advect_step_batch
else:
    _adv_fn = _fast_advect
C.advect_step_batch = functools.partial(_adv_fn, cfl=_ADV_CFL, dtype=_ADV_DTYPE)

# ---- the ONE micro swap; advection already swapped above --------------------
_micro_fn = _ENGINE_FN[MICRO_ENGINE]
C.run_microphysics_full = _micro_fn


def _wrap_desc():
    # MICRO_WRAP_DEVICE=1 is not used under FAST_MULTIGPU on >1 device (the
    # numpy wrapper runs); say so, so the header names the path that runs.
    if MICRO_WRAP_DEVICE and FAST_MULTIGPU and _n_devices() > 1:
        return "1 -> 0 (FAST_MULTIGPU on >1 device: the numpy wrapper runs)"
    return str(int(MICRO_WRAP_DEVICE))


def _describe(engine):
    if engine == 'fast':
        return (f"tomas_jax.fast.run_fast  ({N_FAST_STEPS} steps x {FAST_DT:.0f}s, "
                f"cap {FAST_CELL_CAP} cells/chunk, sort={FAST_SORT}, "
                f"MICRO_WRAP_DEVICE={_wrap_desc()})")
    return _describe_dev_chain()


# coupling's run header names the chain via describe_full_chain(); point it at
# the engine actually installed so the log stays self-describing.
_describe_dev_chain = C.describe_full_chain
if MICRO_ENGINE != 'full':
    C.describe_full_chain = lambda: _describe(MICRO_ENGINE)

print("=" * 60, flush=True)
_ENGINE_TITLE = {'fast': 'tomas_jax.fast (GPU-fast reduced model)',
                 'full': 'tomas-jax dev make_step (coupling.py MICRO=full)'}
print(f"COUPLING: {_ENGINE_TITLE[MICRO_ENGINE]} + "
      f"{C.AER_SRC.upper()} IC/BC", flush=True)
print(f"  micro    : {_describe(MICRO_ENGINE)}", flush=True)
print((f"  advection: advection/sl_lr  {sl_lr.describe()}  "
       f"({'f32' if _ADV_DTYPE is jnp.float32 else 'f64'} gathers)")
      if _ADV_SCHEME == 'sl' else
      (f"  advection: advection/ffsl_lr  {ffsl_lr.describe()}  "
       f"({'f32' if _ADV_DTYPE is jnp.float32 else 'f64'})")
      if _ADV_SCHEME == 'ffsl' else
      f"  advection: advection/fct_{_ADV_SCHEME}  (cfl={_ADV_CFL}, "
      f"{'f32' if _ADV_DTYPE is jnp.float32 else 'f64'})", flush=True)
print(f"  OH       : " + (f"SZA curve sampled {C.OH_SUBSTEPS}x/step -> "
                          f"per-inner-step profile into the engine" if C.OH_SZA
                          else "CESM field, constant over step"), flush=True)
      # read from coupling (not os.environ): AER_SRC is no longer forced here, so
      # the env key may be absent entirely and coupling.py owns the 'mam4' default
print(f"  IC/BC    : AER_SRC={C.AER_SRC} "
      + ("(per-step dynamic from CESM h1)" if C.AER_SRC == 'mam4'
         else "(STATIC reservoir, built once)")
      + f"   bins={C.NBINS} (native)", flush=True)
print("=" * 60, flush=True)

if __name__ == '__main__':
    C.main()
