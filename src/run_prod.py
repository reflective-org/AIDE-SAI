#!/usr/bin/env python3
"""Production launcher: apply a linking preset, resolve the run configuration,
then exec driver_fast.py.

    mkdir -p runs/<tag> && cd runs/<tag>
    python3 /path/to/AIDE-SAI/src/run_prod.py --preset cesm_numerical --gpu 0 \
        --inj-so2 6 --inj-lat 30 --inj-mirror 1 --inj-zonal 0 --out-tag <tag>

`--help` is the inventory of every knob; `--dry-run` prints the resolved
environment (and checks the preset's submodule pins) without launching;
`--list-presets` shows the presets in configs/.

PRESETS (configs/<name>.env) -- what a "linking" is. A preset fixes everything
that defines WHICH coupled model runs: the meteorology source and its grid, the
boundary climatology, the physics choices that differ between linkings, and
WHICH CHECKOUT fills each model slot (`#@submodule` lines: the submodule path,
the exact commit it must be at, and any patch that must be applied on top). The
launcher refuses to start if a slot's checkout is not at the pinned commit, and
points TOMAS_JAX_PATH / RRTMGP_PATH (and, for MET_SRC=jcm_online,
JCM_STRAT_PATH) at the checkouts the preset names. A preset does NOT set the scenario (INJ_*), the run length, the tag or RESUME: those vary
run to run and go on the command line. See configs/README.md.

Without --preset the default linking, `cesm_numerical`, is applied. `--preset none` runs
on coupling.py's module defaults alone (BC_SRC=cesm: the hourly h1 archive).

WHY IT STILL EXECS. Two reasons, both load-bearing:
  * LD_LIBRARY_PATH must be set BEFORE the process starts. glibc reads it once
    at startup, so setting os.environ['LD_LIBRARY_PATH'] in THIS process and
    then importing jax here would not help it find libcuda.so.1 -- it would
    silently fall back to CPU, the exact failure this launcher exists to prevent.
  * coupling.py reads every knob once at module import with os.environ.get. The
    environment therefore has to be final before the import, which is what
    handing a fully-built env to execve guarantees.
Flags are translated to environment variables and nothing else: the model's
configuration remains env-driven and self-describing in the run header.

PRECEDENCE, highest first:  --flag  >  an env var already set  >  the preset
file  >  the launcher's run-control defaults in KNOBS below  >  coupling.py's
own default. An exported variable that overrides a different preset value is
reported with a WARNING (the stale-tmux-shell failure mode).

COST. ~22-35 s per 6 h step on one H100 (microphysics ~45 %, radiation ~30 %,
advection ~20 %); flat over a year. One year = 1460 steps = ~9-14 h.

HOW TO READ THE OUTPUT -- standing caveats:
  * the first ~week is cold-start spin-up;
  * there is NO WET REMOVAL anywhere. Settling and transport out of the band are
    the only sinks;
  * the model is not calibrated for absolute radiative forcing (MANIFEST.md).

ADV_VPOS=1 -- VERTICAL POSITIVITY LIMITER, set by every preset. Without it the
  number field is not usable: fct_lr's vertical remap is exactly CONSERVATIVE
  but not POSITIVE, so it undershoots negative on the steep ultrafine gradient
  at the injection ring and coupling.py's floor clips those negatives and
  CREATES number.

TO RESUME after a crash/kill: re-run with the same preset, tag and scenario plus
--resume 1. It picks up coupled_state_<TAG>_ckpt.npz: the aerosol and gas state
of the resumed run is bit-identical to an uninterrupted one (2026-11-25, after the
qfroz fix; MANIFEST.md). A RESUME onto a checkpoint with a different injection
scenario or grid is refused.
"""
import argparse
import os
import re
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent          # src/
REPO = HERE.parent                              # the checkout
DRIVER = HERE / "driver_fast.py"
CONFIGS = REPO / "configs"
DEFAULT_PRESET = "cesm_numerical"

# ---------------------------------------------------------------------------
# The knob table. One row per environment variable the model reads.
#   (group, cli-flag, ENV_VAR, preset, metavar, help)
# preset is None for "do not set it -- let coupling.py's own default stand".
# A non-None preset is this launcher's PRODUCTION choice and is what makes a
# bare run_prod.py the production run rather than a bare coupling.py run.
#
# Every one of these was reachable before only by knowing the variable name;
# several (INJ_*, FAST_SORT, STATE_CKPT, FRAME_EVERY, N_HOURS) were bare
# literals in the shell version at various times, which meant an override was
# accepted by the shell and then SILENTLY DISCARDED. A table makes that class of
# bug structurally impossible: presets and pass-throughs go through one path.
# ---------------------------------------------------------------------------
KNOBS = [
 # ---- run control ----
 ("run", "--n-hours",      "N_HOURS",      "2160", "H",   "run length in hours (2160 = 90 d; 8760 = 1 yr)"),
 ("run", "--n-days",       "N_DAYS",       None,   "D",   "run length in days (N_HOURS wins if both given)"),
 ("run", "--out-tag",      "OUT_TAG",      "prod90d", "TAG", "output tag; ALL outputs and checkpoints are keyed by it"),
 ("run", "--resume",       "RESUME",       "0",    "0|1", "resume from coupled_state_<TAG>_ckpt.npz"),
 ("run", "--h0",           "H0",           None,   "N",   "start hour index into the CESM h1 series (0 = 1996-01-01 00Z)"),
 ("run", "--step-hours",   "STEP_HOURS",   None,   "H",   "coupling step [h]; one advect+coag per step (default 6)"),
 ("run", "--state-ckpt",   "STATE_CKPT",   "1",    "0|1", "write the 3-D restart checkpoint (needed for --resume)"),
 ("run", "--frame-every",  "FRAME_EVERY",  "24",   "H",   "spatial frame every N hours (>= N_HOURS = final frame only)"),
 ("run", "--frame3d",      "FRAME3D",      None,   "0|1", "1 = write the full 3-D state + per-cell size statistics + TOA fields after EVERY step to frames3d/step_<s>.npz (~480 MB/step, ~700 GB/yr)"),
 ("run", "--frame3d-every", "FRAME3D_EVERY", None,  "N",   "write frames3d only every N-th step (output-only; N coprime to 4 rotates the UTC hour)"),
 ("run", "--ckpt-every-frames", "CKPT_EVERY_FRAMES", None, "N", "write the state checkpoint every N frames (output cadence only)"),
 ("run", "--ckpt-async",   "CKPT_ASYNC",   None,   "0|1", "perf: snapshot the _ckpt arrays to host and write them in a background thread (default 1 since 2026-11-23; 0 = synchronous writes, the old default)"),
 ("run", "--frame-levels", "FRAME_LEVELS", None,   "probe|all",
                                                          "'probe' (default) = one PROBE_HPA level; 'all' adds a "
                                                          "zonal-mean frame at every level (+1.1 GB/yr) and unlocks "
                                                          "the lat-height figures and gifs"),
 ("run", "--log-every",    "LOG_EVERY",    None,   "N",   "progress line every N coupling steps"),
 ("run", "--probe-hpa",    "PROBE_HPA",    None,   "hPa", "diagnostic/frame level (default 50)"),
 ("run", "--debug",        "DEBUG",        "1",    "0|1", "verbose per-step diagnostics"),
 ("run", "--profile",      "PROFILE",      "1",    "0|1", "per-stage timing lines"),
 # ---- domain / grid ----
 ("grid", "--p-lo-hpa",    "P_LO_HPA",     None,   "hPa", "top of the band (default 1)"),
 ("grid", "--p-hi-hpa",    "P_HI_HPA",     None,   "hPa", "bottom of the band (coupling.py default 100; driver_fast.py, i.e. every launcher run, sets 150)"),
 ("grid", "--n-lev",       "N_LEV",        None,   "N",   "sub-sample the band to ~N levels (0 = full contiguous band)"),
 ("grid", "--n-bins",      "N_BINS",       None,   "N",   "TOMAS bin count (0 = native 40)"),
 ("grid", "--diag-core-hpa", "DIAG_CORE_HPA", None, "LO,HI", "core window for the 'int' burden diagnostics"),
 # ---- injection scenario ----
 ("inj", "--inj-so2",      "INJ_SO2_TG_YR", "0.0", "Tg/yr", "SO2 injection rate. DEFAULT 0 = no-injection control"),
 ("inj", "--inj-h2so4",    "INJ_H2SO4_TG_YR", "0.0", "Tg/yr", "direct H2SO4(g) injection rate"),
 ("inj", "--inj-hpa",      "INJ_HPA",      "55.0", "hPa", "injection altitude (snapped to the nearest model level)"),
 ("inj", "--inj-lat",      "INJ_LAT",      "0.0",  "deg", "injection latitude (snapped to the nearest row)"),
 ("inj", "--inj-lon",      "INJ_LON",      "180.0", "degE", "injection longitude (ignored when --inj-zonal 1)"),
 ("inj", "--inj-zonal",    "INJ_ZONAL",    "1",    "0|1", "1 = spread around the whole latitude ring, 0 = single cell"),
 ("inj", "--inj-mirror",   "INJ_MIRROR",   "0",    "0|1", "release at BOTH +lat and -lat, total split 50/50 (not doubled)"),
 ("inj", "--inj-so2-sched", "INJ_SO2_SCHED_FILE", None, "PATH", "yearly SO2 schedule CSV (year_offset,tg_so2_yr): TOTAL Tg SO2/yr per 365-day model year from the run start, annual steps, last row held; replaces --inj-so2 (mutually exclusive); see inputs/injection/"),
 # ---- aerosol source / initial condition ----
 ("aer", "--aer-src",      "AER_SRC",      None,   "mam4|carma", "IC/BC reservoir source (default mam4)"),
 ("aer", "--init-bin",     "INIT_BIN",     None,  "so4|dgnum", "how MAM4 modes are binned onto TOMAS"),
 ("aer", "--mam4-bin-device", "MAM4_BIN_DEVICE", None, "0|1", "perf: run bin_mode's erf part (INIT_BIN=so4) on the device as eager jax ops, no host round trip (default 1 since 2026-11-23; 0 = numpy, the old default)"),
 ("aer", "--init-sigma",   "INIT_SIGMA",   None,   "S",   "override the log-normal mode width used for binning"),
 ("aer", "--carma-file",   "CARMA_FILE",   None,   "PATH", "CARMA reservoir file (AER_SRC=carma)"),
 ("aer", "--carma-frame",  "CARMA_FRAME",  None,   "N",   "time index into the CARMA file"),
 ("aer", "--carma-rho",    "CARMA_RHO",    None,   "kg/m3", "CARMA particle density"),
 ("aer", "--carma-subbin", "CARMA_SUBBIN", None,   "N",   "CARMA sub-binning factor"),
 # ---- boundaries ----
 ("bc", "--n-bc-top",      "N_BC_TOP",     None,   "N",   "band levels pinned/served at the top face (default 1)"),
 ("bc", "--n-bc-bot",      "N_BC_BOT",     None,   "N",   "band levels pinned/served at the bottom face (default 1)"),
 ("bc", "--bc-edge",       "BC_EDGE",      None,   "open|clamp", "aerosol edge treatment (derived from ADV_WCONT)"),
 ("bc", "--bc-gas",        "BC_GAS",       None,   "flux|clamp", "gas edge treatment (derived from BC_EDGE)"),
 ("bc", "--bc-bot-aer",    "BC_BOT_AER",   None,   "X",   "scale the bottom-face aerosol inflow (0 = clean upwelling)"),
 # ---- physics stages ----
 ("phys", "--micro",       "MICRO",        None,   "full|coag|off", "microphysics mode; 'off' is transport-only"),
 ("phys", "--micro-dt",    "MICRO_DT",     None,   "s",   "MICRO=full: tomas-jax dev step length (default 60, dev's; must divide the 6 h step)"),
 ("phys", "--micro-ncoag", "MICRO_NCOAG",  None,   "N",   "MICRO=full: forward-Euler coag sub-steps per dev step (default 1; dev's own default 10 is within 5e-4 in total N at 3x the cost)"),
 ("phys", "--micro-coag",  "MICRO_COAG",   None,   "fixed|adaptive", "MICRO=full: dev's coag solver (default fixed = euler_fixed, dev's strat setup; adaptive = euler_adaptive, ~100x slower and over the micro budget)"),
 ("phys", "--micro-fion",  "MICRO_FION",   None,   "cm-3 s-1", "MICRO=full: ion-pair production for ion-induced nucleation (default 0 = off; dev's strat setup uses 10)"),
 ("phys", "--micro-mnfix-trim", "MICRO_MNFIX_TRIM", None, "0|1", "MICRO=full: MNFIX over out-of-range bins only (default 1; bit-identical to 0 and 5x faster here)"),
 ("phys", "--micro-max-wall", "MICRO_MAX_WALL", None, "s", "MICRO=full: abort the run if one step's microphysics takes longer (default 300)"),
 ("phys", "--micro-gpus",  "MICRO_GPUS",   None,   "N",   "MICRO=full: spread the micro cells over the first N visible GPUs (default 1; needs --gpu to expose them)"),
 ("phys", "--settle",      "SETTLE",       None,   "0|1", "gravitational settling (default on)"),
 ("phys", "--wet-settling", "WET_SETTLING", None,  "0|1", "hygroscopic growth in the settling velocity"),
 ("phys", "--wet-optics",  "WET_OPTICS",   None,   "0|1", "hygroscopic growth in the optics"),
 ("phys", "--rad",         "RAD",          None,   "0|1", "radiation on/off"),
 ("phys", "--rad-every",   "RAD_EVERY",    None,   "N",   "coupling steps between radiation calls"),
 ("phys", "--rad-mode",    "RAD_MODE",     None,   "anomaly|full|flux", "heating-rate definition (flux = one solve, no feedback)"),
 ("phys", "--rad-t-feedback", "RAD_T_FEEDBACK", None, "0|1", "0 (default, all presets) = microphysics, settling, RH, OH and the radiation optics all see the met source's temperature; dT_rad and the heating rate are diagnostics, and dT_rad enters only the band temperature of the perturbed radiation call (its emission and gas absorption: the Planck damping). 1 = the opt-in closed column loop: T_met + dT_rad everywhere (winds stay prescribed)"),
 ("phys", "--rad-solar",   "RAD_SOLAR",    None,   "instant|window|window_ins|window2|window2_ins|window_blend|dither|window_gq", "solar geometry per radiation call: window_gq = shortwave at the Gauss nodes of the sunlit part of the interval, longwave once (default); instant = the sun at the call hour held for the interval (the pre-2026-11-17 default, bit-identical to earlier runs; leaves the dusk/dawn stripes in dT_rad); window / window_ins / window2 / window2_ins / window_blend / dither = experimental variants (docs/RADIATION_ARTIFACT_EXPLORATION.md)"),
 ("phys", "--rad-solar-alpha", "RAD_SOLAR_ALPHA", None, "A", "window_blend weight in [0, 1]: cos(zenith) = (1-A) x sunlit-mean + A x energy-weighted (default 0.53, fitted)"),
 ("phys", "--rad-solar-nodes", "RAD_SOLAR_NODES", None, "N", "window_gq solves per radiation call, at the Gauss nodes of the sunlit part of the interval (default 3)"),
 ("phys", "--rad-solar-align", "RAD_SOLAR_ALIGN", None, "trailing|centred", "where window_gq's window sits: trailing = the step that just ended, t-6..t h, so dT_rad at t holds only past heating (default); centred = t-3..t+3 h (window_gq before 2026-11-17)"),
 ("phys", "--arf-avg-h",   "ARF_AVG_H",    None,   "H",   "trailing window for the reported TOA forcing"),
 ("phys", "--alpha-cond",  "ALPHA_COND",   None,   "X",   "H2SO4 accommodation coefficient"),
 # ---- chemistry / nucleation ----
 ("chem", "--oh-sza",      "OH_SZA",       None,   "0|1", "diurnal OH parabola in cos(SZA)"),
 ("chem", "--oh-peak",     "OH_PEAK",      None,   "cm-3", "noon OH peak (default 2.3e6)"),
 ("chem", "--oh-substeps", "OH_SUBSTEPS",  None,   "N",   "OH samples per coupling step"),
 ("chem", "--n-coag-substeps", "N_COAG_SUBSTEPS", None, "N", "forward-Euler coag substeps (legacy MICRO=coag path)"),
 # ---- microphysics engine selection ----
 ("engine", "--micro-engine", "MICRO_ENGINE", None, "fast|full",
                                                          "which engine advances the run (default fast = tomas_jax.fast, "
                                                          "tomas-jax gpu-fast; full = tomas-jax dev make_step, which needs a "
                                                          "preset whose tomas-jax slot is a dev checkout)"),
 ("engine", "--micro-wrap-device", "MICRO_WRAP_DEVICE", None, "0|1", "perf: MICRO_ENGINE=fast adapter (microphysics/tomas_fast.py) does its unit conversion, transposes, consistency clip and OH field on the device, no host round trip for num/mas (not bit-identical: device reductions/ordering give ~1e-12 rel diffs in a few cells, within run-to-run GPU noise; default 1 since 2026-11-23, not used under FAST_MULTIGPU on >1 device; 0 = numpy, the old default)"),
 # ---- meteorology source ----
 ("met", "--met-src",      "MET_SRC",      None,   "cesm|file|jcm|jcm_online", "where U/V/OMEGA/T come from (set by the preset): cesm = the hourly CESM h1 archive; jcm = a jcm-strat run directory (--jcm-dir), 6-hourly, and the model GRID with it (T63 Gaussian 96x192, JCM's native levels in the band; needs --clim-file on that grid, scripts/met/regrid_clim_to_grid.py); jcm_online = the same fields and grid from the jcm-strat AL7h model integrated LIVE beside this one (src/met/jcm_server.py, a subprocess under --jcm-py on the same GPU; code from the jcm-strat slot), started from the --jcm-dir chain's 1 January checkpoint; file = step-end frames in MET_DIR (the seam for an emulated dycore). Everything else (Q, gases, reservoirs, the radiation column below the band) comes from --bc-src"),
 ("met", "--jcm-dir",      "JCM_DIR",      None,   "PATH", "MET_SRC=jcm|jcm_online: the jcm-strat run directory (default /data/JCM_stripped/jcm-strat/runs/p31/p31_AL7h70, the AL7h chain); read-only. jcm_online reads only its 1 January checkpoints, their last output frame and the template segment's output (for the grid)"),
 ("met", "--jcm-template", "JCM_TEMPLATE", None,   "PATH", "MET_SRC=jcm_online: the resolved Hydra config the live JCM is built from (default <jcm-dir>/<chain>_19960101/.hydra/config.yaml); start time, QBO year and init file are set per year"),
 ("met", "--jcm-py",       "JCM_PY",       None,   "PATH", "MET_SRC=jcm_online: the interpreter the JCM server runs under, a Python 3.11 venv with jcm-strat's dependencies (JAX 0.10; default <JCM_STRAT_PATH>/.venv/bin/python). Machine-specific: a submodule checkout has no venv. jcm_strat and jcm themselves are imported from the slot, not from this venv"),
 ("met", "--jcm-cache",    "JCM_CACHE",    None,   "PATH", "MET_SRC=jcm_online: jcm-strat's data cache, holding era5/ (the regridded ERA5 nudging windows, JCM_ERA5_CACHE) and huggingface/ (jcm boundary files, HF_HOME) (default <JCM_STRAT_PATH>/cache). Machine-specific; a missing ERA5 window is downloaded and regridded on first use"),
 ("met", "--jcm-scratch",  "JCM_SCRATCH",  None,   "PATH", "MET_SRC=jcm_online: jcm's scratch directory, $SCRATCH (its JAX compilation cache is $SCRATCH/jcm-jax-cache; default <JCM_STRAT_PATH>/scratch). Machine-specific; cannot change results"),
 ("met", "--jcm-te-correction", "JCM_TE_CORRECTION", None, "PATH", "MET_SRC=jcm_online: the AL7h T_e correction file (default the jcm-strat slot's tracked docs/outputs/25_al7_half/corrections/AL7h.nc; the template's own path names a removed jcm-strat worktree)"),
 ("met", "--jcm-work",     "JCM_WORK",     None,   "DIR",  "MET_SRC=jcm_online: where the live JCM writes its year-end checkpoints (default jcm_online/ in the run directory; a RESUME restarts from them)"),
 ("met", "--met-q",        "MET_Q",        None,   "bc|met", "MET_SRC=jcm|jcm_online: water vapour from the BC source (bc, default: the regridded CESM climatology -- jcm-strat is dry and its q is a passive constant) or from JCM (met)"),
 ("met", "--met-dir",      "MET_DIR",      None,   "PATH", "MET_SRC=file: directory holding u/v/omega/T.npy + meta.json"),
 ("met", "--met-rh",       "MET_RH",       None,   "liq|q|cesm", "RH seen by settling, the microphysics' water uptake (full chain) and the FRAME3D wet diagnostics, recomputed every step from Q, T and p: liq (default) = over liquid water, Tabazadeh 1997 eq. 1. The radiation's wet optics ignore MET_RH and always compute liquid RH from Q themselves; q = over ice (the old MET_SRC=file default; 1.6-1.9x liq in the band); cesm = the archive's RELHUM as is"),
 ("met", "--bc-src",       "BC_SRC",       None,   "cesm|clim", "where every non-wind boundary field comes from (set by the preset): clim = the monthly climatology in CLIM_FILE (scripts/met/make_bc_climatology.py); cesm (module default) = the hourly archive"),
 ("met", "--clim-file",    "CLIM_FILE",    None,   "PATH", "BC_SRC=clim: the climatology, on the MODEL grid (set by the preset; the jcm_numerical preset uses the CESM climatology regridded onto the JCM grid by scripts/met/regrid_clim_to_grid.py)"),
 # ---- advection ----
 ("adv", "--adv-scheme",   "ADV_SCHEME",   None,   "lr|fast|sl|ffsl", "transport form (default sl = semi-Lagrangian split, K=12, the JCM/dinosaur kernel, docs/TRANSPORT_SL_HANDOFF.md: 0.76x lr's CH4-truth error, ~0.9 s/step vs 6-8 s; lr = Lin-Rood flux form at CFL sub-steps; ffsl = lr at large Courant number; fast = advective form)"),
 ("adv", "--adv-ffsl-k",   "ADV_FFSL_K",   None,   "int", "ffsl: sub-steps per coupling step (default 24, the production operating point; 6 = hourly; a convergence knob, each ~1/200 of an lr step)"),
 ("adv", "--adv-ffsl-split", "ADV_FFSL_SPLIT", None, "strang|lie", "ffsl: sweep order (strang = z/2, xy|yx alternating, z/2, default; lie = fct_lr's x y z)"),
 ("adv", "--adv-sl-k",     "ADV_SL_K",     None,   "N",   "SL: sub-steps per coupling step (default 12)"),
 ("adv", "--adv-sl-fixer", "ADV_SL_FIXER", None,   "global|local|none", "SL: per-tracer mass fixer on the closed column (default global; local = Bermejo-Conde)"),
 ("adv", "--adv-sl-hourly", "ADV_SL_HOURLY", None, "0|1", "sl and ffsl: 1 = read the hourly CESM winds between the step snapshots; 0 (default) = the two snapshots interpolated in time, as lr does and as production winds arrive"),
 ("adv", "--adv-sl-vert",  "ADV_SL_VERT",  None,   "flux|sl", "SL: vertical treatment (default flux = fct_lr's open-face flux sweep; sl = closed-column 3-D gather, A/B-failed)"),
 ("adv", "--adv-sl-omega", "ADV_SL_OMEGA", None,   "cont|cesm", "SL flux vertical: face omega from continuity (lr's, default) or CESM centers averaged"),
 ("adv", "--adv-sl-rho",   "ADV_SL_RHO",   None,   "0|1", "SL flux vertical, lie split only: carry the air mass 1 - dt div(v) through the horizontal step (strang always does; default 0)"),
 ("adv", "--adv-sl-split", "ADV_SL_SPLIT", None,   "lie|strang", "SL flux vertical: strang (default) = V/2 H V/2 with the air mass carried (second-order face flux); lie = H then V (first order at the open face)"),
 ("adv", "--adv-sl-jac",   "ADV_SL_JAC",   None,   "0|1", "SL flux vertical: 1 (default) = air mass from the Jacobian of the departure map and the bookkeeping remainder returned through the faces; 0 = Eulerian 1 - dt div(v) and the global column fixer"),
 ("adv", "--adv-sl-rhonorm", "ADV_SL_RHONORM", None, "0|1", "SL: 1 (default since PR #4, merged into dev 2026-11-24) = rescale each level's air mass after the horizontal step so the level total is conserved, which stops the per-level fixer imprinting a drift (uniform q: +2%% at 1.24 hPa in 10 days -> 1e-4); 0 = every run made before PR #4 reached dev (2026-11-24), including the 2-year runs"),
 ("adv", "--adv-sl-fixlog", "ADV_SL_FIXLOG", None,  "0|1", "SL diagnostic: 1 = log what the level and column mass fixers corrected every sub-step (results unchanged; default 0)"),
 ("adv", "--adv-sl-device-resample", "ADV_SL_DEVICE_RESAMPLE", None, "0|1", "perf: SL time-resample of the step winds on the device (same float32 formula, bit-identical; engages for float32 winds, prints ENGAGED/not engaged once; default 1 since 2026-11-23; 0 = numpy, the old default)"),
 ("adv", "--adv-sl-vorder", "ADV_SL_VORDER", None, "linear|cubic", "SL: vertical gather order (default linear)"),
 ("adv", "--adv-cfl",      "ADV_CFL",      None,   "X",   "horizontal CFL target (production 0.5)"),
 ("adv", "--adv-passive",  "ADV_PASSIVE",  None,   "0|1", "TEST: advect ONE total-sulfate tracer (+ the two gases) and rescale every bin by the cell's mass ratio (frozen per-cell shape); score against the same run with 0"),
 ("adv", "--adv-f32",      "ADV_F32",      None,   "0|1", "float32 transport sweeps (production 1)"),
 ("adv", "--adv-vpos",     "ADV_VPOS",     None,    "0|1", "VERTICAL POSITIVITY LIMITER -- see the header; do not turn off"),
 ("adv", "--adv-wcont",    "ADV_WCONT",    None,   "0|1", "rederive omega from discrete continuity (opens the faces)"),
 ("adv", "--adv-metric",   "ADV_METRIC",   None,   "0|1", "cos(phi) area metric in the y-sweep"),
 ("adv", "--adv-dxfix",    "ADV_DXFIX",    None,   "0|1", "true grid spacing in the x-sweep"),
 ("adv", "--adv-polar",    "ADV_POLAR",    None,   "zonal|freeze", "polar cap treatment (zonal = stirred, mass-conserving)"),
 # ---- radiation gases / chunking ----
 ("rad", "--co2-ppm",      "CO2_PPM",      None,   "ppm", "background CO2 for RRTMGP"),
 ("rad", "--n2o-ppb",      "N2O_PPB",      None,   "ppb", "background N2O for RRTMGP"),
 ("rad", "--rad-lat-chunk", "RAD_LAT_CHUNK", None, "N",   "latitude rows per radiation chunk (GPU memory; default 96 for the chunk loop, 192 for the RAD_FAST path; an explicit value sets both)"),
 ("rad", "--rad-fast",     "RAD_FAST",     None,   "auto|0|1", "perf: radiation fast path (one lat chunk, SW on sunlit columns only, g-point batching, exact minor-gas scan, no cloud optics; window_gq too). Default auto (since 2026-11-23): on where validated (1 GPU, wet optics, RAD_SOLAR=window_gq, the normal solve), else the chunk loop, decided per radiation driver and stated in the header; 1 = force (refuses anything else); 0 = the chunk loop (the old default)"),
 # ---- radiation engine / reference cache (docs/RADIATION.md) ----
 ("rad", "--rad-engine",   "RAD_ENGINE",   None,   "rrtmgp", "which engine solves the EVOLVED radiation call (rrtmgp = the physics, the only one in this repo)"),
 ("rad", "--rad-ref-cache", "RAD_REF_CACHE", None, "PATH|0", "per-archive-hour cache of the reference-aerosol solve and the clear-sky fields (default inputs/cache/rad_ref; bit-identical on a hit, computes+stores on a miss; 0 = off; always off under MET_SRC=jcm_online)"),
 ("rad", "--rad-sample-every", "RAD_SAMPLE_EVERY", None, "N", "write rad_sample_<TAG>_s<step>.npz (evolved bins, dT_rad, hour) every N radiation steps -- training data for a radiation emulator, not part of this repo (0 = off)"),
 # ---- memory / performance (numerically irrelevant; the perf switches that are
 #      not bit-identical, e.g. MICRO_WRAP_DEVICE, live in their stage's group) ----
 ("perf", "--cell-chunk",  "CELL_CHUNK",   None,   "N",   "cells per microphysics vmap batch"),
 ("perf", "--tracer-chunk", "TRACER_CHUNK", None,  "N",   "tracers per advection batch (0 = all 82 at once)"),
 # ---- input data locations ----
 ("path", "--cesm-dir",    "CESM_DIR",     None,   "PATH", "root of the CESM h1 tseries archive"),
 ("path", "--cesm-prefix", "CESM_PREFIX",  None,   "STR",  "CESM case-name prefix of the h1 files"),
 ("path", "--cesm-suf",    "CESM_SUF",     None,   "STR",  "date-range suffix of the h1 files"),
 ("path", "--tomas-jax-path", "TOMAS_JAX_PATH", None, "PATH", "tomas-jax checkout (default models/tomas-jax)"),
 ("path", "--rrtmgp-path", "RRTMGP_PATH",  None,   "PATH", "jax-rrtmgp checkout (default models/jax-rrtmgp)"),
 ("path", "--jcm-strat-path", "JCM_STRAT_PATH", None, "PATH", "jcm-strat checkout, recursively initialised (jcm_strat + external/jax-gcm's jcm), that the MET_SRC=jcm_online server imports (default models/jcm-strat)"),
 # ---- debug dumps ----
 ("dbg", "--dump-premicro", "DUMP_PREMICRO", None, "PATH", "dump the pre-microphysics state to this file"),
 ("dbg", "--dump-premicro-step", "DUMP_PREMICRO_STEP", None, "N", "step at which to take that dump"),
]

GROUPS = [
    ("run",  "run control"),
    ("grid", "domain and grid"),
    ("inj",  "injection scenario (the knobs meant to change run to run)"),
    ("aer",  "aerosol source / initial condition"),
    ("bc",   "boundaries"),
    ("phys", "physics stages"),
    ("chem", "chemistry, nucleation and coagulation"),
    ("engine", "microphysics engine selection"),
    ("adv",  "advection"),
    ("met",  "meteorology source and boundary climatology (set by the preset)"),
    ("rad",  "radiation gases, chunking, engine and reference cache (docs/RADIATION.md)"),
    ("perf", "memory / performance (cannot change results)"),
    ("path", "input data locations"),
    ("dbg",  "debug dumps"),
]


# ---------------------------------------------------------------------------
# Presets (configs/<name>.env). Format, one item per line:
#   KEY=VALUE                       an environment variable (a KNOBS name)
#   #@submodule SLOT PATH SHA [patch=FILE]
#                                   the checkout that fills model slot SLOT:
#                                   PATH (relative to the repo) must be at
#                                   commit SHA, with FILE (relative to the
#                                   repo) applied on top if given
#   # ...                           a comment
# No shell evaluation, no includes: a preset is read the same way everywhere.
# ---------------------------------------------------------------------------
SLOT_ENV = {"tomas-jax": "TOMAS_JAX_PATH", "jax-rrtmgp": "RRTMGP_PATH",
            "jcm-strat": "JCM_STRAT_PATH"}
# Slots only some linkings use: slot -> (knob, the values that need it). A preset
# pins such a slot exactly when its knob takes one of those values, and never
# otherwise, so a linking that does not run jcm-strat neither needs the
# submodule checked out nor names a commit it does not use. Every other slot is
# pinned by every preset (scripts/validation/test_configs.py).
SLOT_NEEDED_IF = {"jcm-strat": ("MET_SRC", {"jcm_online"})}


def slots_needed(values):
    """The slots a preset with these KEY=VALUE settings must pin."""
    return {slot for slot in SLOT_ENV
            if slot not in SLOT_NEEDED_IF
            or values.get(SLOT_NEEDED_IF[slot][0], "") in SLOT_NEEDED_IF[slot][1]}


# What a preset may NOT set. The scenario and run control vary run to run and
# belong on the command line. The others are read by PRESENCE or truthiness
# ('X' in os.environ, bool(os.environ.get(X))), so writing out a "default" value
# in a preset would silently change behaviour (DEBUG=0 turns debugging ON).
PRESET_FORBIDDEN = {
    "INJ_SO2_TG_YR", "INJ_H2SO4_TG_YR", "INJ_HPA", "INJ_LAT", "INJ_LON",
    "INJ_ZONAL", "INJ_MIRROR", "INJ_SO2_SCHED_FILE", "N_HOURS", "N_DAYS", "OUT_TAG", "RESUME", "H0",
    "MICRO_SUBSTEPS", "OH_SUBSTEPS", "DEBUG", "PROFILE", "DUMP_PREMICRO",
    "INIT_SIGMA", "TOMAS_JAX_PATH", "RRTMGP_PATH", "JCM_STRAT_PATH",
}
KNOB_NAMES = {row[2] for row in KNOBS}
# Knobs whose value is a path. In a preset a relative one is relative to the
# checkout (e.g. CLIM_FILE=inputs/bc_clim/bc_clim.npz), so a preset does not
# depend on the directory the run is launched from.
PATH_KNOBS = {row[2] for row in KNOBS if row[4] == "PATH"}


def preset_path(name):
    """A preset name (configs/<name>.env) or a path to a .env file."""
    cand = Path(name)
    if cand.suffix == ".env" and cand.exists():
        return cand.resolve()
    path = CONFIGS / f"{name}.env"
    if not path.exists():
        have = ", ".join(sorted(q.stem for q in CONFIGS.glob("*.env"))) or "none"
        sys.exit(f"run_prod.py: no preset {name!r} (looked for {path}; have: {have})")
    return path


def load_preset(path):
    """Parse a preset file -> (env dict, [submodule pins])."""
    values, pins = {}, []
    for n, raw in enumerate(path.read_text().splitlines(), 1):
        line = raw.strip()
        where = f"{path.name}:{n}"
        if line.startswith("#@submodule"):
            parts = line.split()[1:]
            if len(parts) not in (3, 4) or (len(parts) == 4 and not parts[3].startswith("patch=")):
                sys.exit(f"run_prod.py: {where}: expected "
                         f"'#@submodule SLOT PATH SHA [patch=FILE]', got {raw!r}")
            slot, sub, sha = parts[:3]
            if slot not in SLOT_ENV:
                sys.exit(f"run_prod.py: {where}: unknown slot {slot!r} "
                         f"(known: {', '.join(SLOT_ENV)})")
            if not re.fullmatch(r"[0-9a-f]{40}", sha):
                sys.exit(f"run_prod.py: {where}: pin the full 40-character commit, got {sha!r}")
            patch = parts[3].split("=", 1)[1] if len(parts) == 4 else None
            pins.append({"slot": slot, "path": sub, "sha": sha, "patch": patch})
            continue
        if not line or line.startswith("#"):
            continue
        if "=" not in line:
            sys.exit(f"run_prod.py: {where}: expected KEY=VALUE, got {raw!r}")
        key, val = (x.strip() for x in line.split("=", 1))
        if key in PRESET_FORBIDDEN:
            sys.exit(f"run_prod.py: {where}: {key} may not be set in a preset "
                     f"(see PRESET_FORBIDDEN in run_prod.py)")
        if key not in KNOB_NAMES:
            sys.exit(f"run_prod.py: {where}: {key} is not a knob run_prod.py knows; "
                     f"add it to KNOBS first so it is documented and echoed")
        if key in PATH_KNOBS and val and not os.path.isabs(val):
            val = str(REPO / val)
        values[key] = val
    return values, pins


def _git(args, cwd):
    r = subprocess.run(["git", *args], cwd=cwd, capture_output=True, text=True)
    return r.returncode, r.stdout.strip(), r.stderr.strip()


def check_pins(pins, verify=True):
    """Verify each slot's checkout is at its pinned commit (and patched).
    Returns {ENV: absolute path}; exits with instructions on a mismatch."""
    env, problems = {}, []
    for pin in pins:
        sub = (REPO / pin["path"]).resolve()
        env[SLOT_ENV[pin["slot"]]] = str(sub)
        if not verify:
            continue
        if not sub.is_dir():
            problems.append(f"{pin['slot']}: {pin['path']} does not exist -- "
                            f"git submodule update --init {pin['path']}")
            continue
        rc, head, err = _git(["rev-parse", "HEAD"], sub)
        if rc != 0:
            problems.append(f"{pin['slot']}: cannot read the commit of {pin['path']} ({err})")
            continue
        if head != pin["sha"]:
            problems.append(f"{pin['slot']}: {pin['path']} is at {head[:12]}, the preset pins "
                            f"{pin['sha'][:12]} -- git -C {pin['path']} checkout {pin['sha'][:12]} "
                            f"(or git submodule update {pin['path']})")
            continue
        if pin["patch"]:
            patch = (REPO / pin["patch"]).resolve()
            rc, _o, _e = _git(["apply", "--check", "--reverse", str(patch)], sub)
            if rc != 0:
                problems.append(f"{pin['slot']}: {pin['patch']} is not applied in {pin['path']} -- "
                                f"git -C {pin['path']} apply {os.path.relpath(patch, sub)}")
    if problems:
        sys.exit("run_prod.py: the preset's submodule pins do not match the checkout:\n  "
                 + "\n  ".join(problems)
                 + "\n  (--no-verify-pins skips this check; the run is then NOT the preset's model)")
    return env


def list_presets():
    for path in sorted(CONFIGS.glob("*.env")):
        first = next((ln.lstrip("# ").strip() for ln in path.read_text().splitlines()
                      if ln.startswith("# ") and not ln.startswith("#@")), "")
        mark = "  (default)" if path.stem == DEFAULT_PRESET else ""
        print(f"{path.stem:<22} {first}{mark}")


def build_parser():
    p = argparse.ArgumentParser(
        prog="run_prod.py",
        description=("Launch a coupled TOMAS-JAX SAI run. Outputs are written to "
                     "the CURRENT DIRECTORY, so launch from a run directory."),
        epilog=("Every flag below maps to the environment variable named in its "
                "help text; an already-set variable is used when the flag is "
                "omitted. A [ENV; launcher default X] tag is this launcher's run-control "
                "default, applied below any configs/ preset; the stack itself "
                "(meteorology, engines, transport, radiation) comes from --preset."),
        # NOT ArgumentDefaultsHelpFormatter: every knob flag defaults to None,
        # meaning 'not given', and it would append '(default: None)' to all 83.
        # The real default is in the help text as the [ENV; launcher default] tag.
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--gpu", default=os.environ.get("GPU", "0"), metavar="N",
                   help="CUDA device to pin the run to (GPU / CUDA_VISIBLE_DEVICES)")
    p.add_argument("--cuda-driver-lib", default=None, metavar="PATH",
                   help="directory holding libcuda.so.1; empty string disables "
                        "the search (CUDA_DRIVER_LIB)")
    p.add_argument("--preallocate", default=None, metavar="true|false",
                   help="XLA_PYTHON_CLIENT_PREALLOCATE (default false: grow on "
                        "demand, so the run is a good neighbour on a shared box)")
    p.add_argument("--preset", default=None, metavar="NAME|FILE",
                   help=f"linking preset: configs/<NAME>.env or a path to a .env "
                        f"file (default {DEFAULT_PRESET}; 'none' = module defaults only)")
    p.add_argument("--list-presets", action="store_true",
                   help="list the presets in configs/ and exit")
    p.add_argument("--no-verify-pins", action="store_true",
                   help="do not check that the preset's submodules are at their pinned "
                        "commits (the run is then NOT the preset's model)")
    p.add_argument("--dry-run", action="store_true",
                   help="print the resolved environment and exit without launching")
    p.add_argument("--allow-source-tree", action="store_true",
                   help="override the refusal to write into the checkout (you "
                        "almost certainly do not want this -- see the guard)")
    for key, title in GROUPS:
        g = p.add_argument_group(title)
        for grp, flag, env, preset, metavar, helptext in KNOBS:
            if grp != key:
                continue
            g.add_argument(flag, default=None, metavar=metavar,
                           help=f"{helptext}  [{env}"
                                + (f"; launcher default {preset}]" if preset is not None else "]"))
    return p


def resolve_env(args, preset_vals=None, preset_name="", slot_env=None):
    """Build the child environment. See PRECEDENCE in the module docstring."""
    env = dict(os.environ)
    resolved, warnings = {}, []
    preset_vals = dict(preset_vals or {})
    preset_vals.update(slot_env or {})
    for _grp, flag, name, preset, _mv, _h in KNOBS:
        attr = flag.lstrip("-").replace("-", "_")
        val = getattr(args, attr)
        if val is None:                       # no flag given
            if name in env:                   # an exported variable wins over the preset
                resolved[name] = (env[name], "env")
                if name in preset_vals and preset_vals[name] != env[name]:
                    warnings.append(f"WARNING: exported {name}={env[name]} overrides the "
                                    f"preset's {name}={preset_vals[name]}")
                continue
            if name in preset_vals:
                val, src = preset_vals[name], f"preset:{preset_name}"
            elif preset is not None:
                val, src = preset, "launcher"
            else:                             # nothing to say: coupling.py decides
                continue
        else:
            src = "flag"
        env[name] = str(val)
        resolved[name] = (str(val), src)
    return env, resolved, warnings


def find_libcuda(explicit):
    """Locate libcuda.so.1.

    JAX needs it, and under a containerized driver mount it is not on the loader
    path. Without it JAX SILENTLY falls back to CPU, which turns a 3.5-hour run
    into weeks -- so a miss is a loud warning, never a quiet default.
    """
    if explicit == "":                        # explicitly disabled
        return None, None
    cand = explicit or "/run/nvidia/driver/usr/lib/x86_64-linux-gnu"
    if (Path(cand) / "libcuda.so.1").exists():
        return cand, None
    for root in ("/run/nvidia", "/usr/lib"):
        r = Path(root)
        if not r.exists():
            continue
        for hit in r.rglob("libcuda.so.1"):
            return str(hit.parent), f"libcuda.so.1 found at {hit.parent}"
    return None, ("WARNING: libcuda.so.1 not found. If the run reports no GPU it "
                  "has fallen back to CPU -- set --cuda-driver-lib.")


def guard_cwd(allow):
    """Refuse to write into the source tree.

    Outputs go to $PWD and they do not belong in the checkout. This fails
    INVISIBLY otherwise: .npz/.png are gitignored, so a run launched from the
    wrong directory looks completely normal while quietly filling the tree (this
    is how 29 GB accumulated before 2026-08-12). The rule is "anywhere inside the
    checkout except under runs/", which is what makes the documented
    `cd runs/<tag>` workflow legal.
    """
    if allow:
        return
    pwd = Path.cwd().resolve()
    runs = REPO / "runs"
    inside = pwd == REPO or REPO in pwd.parents
    in_runs = pwd == runs or runs in pwd.parents
    if inside and not in_runs:
        sys.exit(
            f"run_prod.py: refusing to run from inside the source tree.\n"
            f"             $PWD = {pwd}\n"
            f"             Outputs are written to $PWD, and they do not belong in\n"
            f"             the source tree. Launch from a run directory instead:\n"
            f"                 mkdir -p {REPO}/runs/<tag> && cd $_\n"
            f"                 python3 {Path(__file__).resolve()} --out-tag <tag>\n"
            f"             (a directory outside the checkout entirely also works)")


def main():
    args = build_parser().parse_args()
    if args.list_presets:
        list_presets()
        return 0
    guard_cwd(args.allow_source_tree)
    name = args.preset or os.environ.get("AIDE_PRESET") or DEFAULT_PRESET
    if name == "none":
        pvals, pins, label = {}, [], "none"
    else:
        ppath = preset_path(name)
        pvals, pins = load_preset(ppath)
        label = ppath.stem
    slot_env = check_pins(pins, verify=not args.no_verify_pins)
    env, resolved, warnings = resolve_env(args, pvals, label, slot_env)
    env["AIDE_PRESET"] = label

    libdir, note = find_libcuda(args.cuda_driver_lib
                                if args.cuda_driver_lib is not None
                                else os.environ.get("CUDA_DRIVER_LIB"))
    if note:
        print(f"run_prod.py: {note}", file=sys.stderr)
    if libdir:
        env["LD_LIBRARY_PATH"] = os.pathsep.join(
            [libdir] + ([env["LD_LIBRARY_PATH"]] if env.get("LD_LIBRARY_PATH") else []))
    env["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    env["XLA_PYTHON_CLIENT_PREALLOCATE"] = (
        args.preallocate if args.preallocate is not None
        else os.environ.get("XLA_PYTHON_CLIENT_PREALLOCATE", "false"))

    print(f"run_prod.py: CUDA_VISIBLE_DEVICES={env['CUDA_VISIBLE_DEVICES']}  "
          f"preallocate={env['XLA_PYTHON_CLIENT_PREALLOCATE']}", file=sys.stderr)
    print(f"run_prod.py: preset {label}"
          + (f" ({os.path.relpath(ppath, REPO) if ppath.is_relative_to(REPO) else ppath})"
             if label != "none" else ""), file=sys.stderr)
    for pin in pins:
        print(f"run_prod.py:   slot {pin['slot']:<11} {pin['path']} @ {pin['sha'][:12]}"
              + (f" + {pin['patch']}" if pin["patch"] else "")
              + ("   (NOT VERIFIED)" if args.no_verify_pins else "   (verified)"),
              file=sys.stderr)
    for w in warnings:
        print(f"run_prod.py: {w}", file=sys.stderr)
    width = max((len(k) for k in resolved), default=0)
    for k in sorted(resolved):
        val, src = resolved[k]
        print(f"run_prod.py:   {k:<{width}} = {val}   ({src})", file=sys.stderr)

    if args.dry_run:
        print("run_prod.py: --dry-run, not launching", file=sys.stderr)
        return 0
    if not DRIVER.exists():
        sys.exit(f"run_prod.py: driver not found at {DRIVER}")
    # exec, not import: see WHY IT STILL EXECS in the module docstring.
    os.execve(sys.executable, [sys.executable, str(DRIVER)], env)


if __name__ == "__main__":
    sys.exit(main())
