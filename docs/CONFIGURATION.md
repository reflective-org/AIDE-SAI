# Configuration reference

Every knob in this model is an environment variable, and every one in
`run_prod.py`'s `KNOBS` table also has a `--flag`. This page lists them and says
which apply to a given run. `python3 src/run_prod.py --help` is the live
inventory: it is generated from the same table the launcher resolves, so it
cannot drift from the code.

> **[`../MANIFEST.md`](../MANIFEST.md) is canonical for this tree.** It records
> every default *and why it is what it is*. This page tells you what the knobs
> are; MANIFEST tells you why. Where the two disagree, MANIFEST is current.

Each variable is read **once at module import** and echoed in the run header, so
any log is self-describing.

## Presets: which linking runs

A run is one **linking** — which implementation fills each model slot, which
commit of each model repo it runs, and the handful of settings that differ
between linkings — plus a **scenario** (injection, length, tag) given on the
command line. The linking is a preset file in `configs/`;
**[`configs/README.md`](../configs/README.md) is the reference for presets**
(file format, forbidden keys, pin verification, how to add one). In short:

| preset | sets | pins |
|---|---|---|
| `cesm_numerical` (default) | `MET_SRC=cesm`, `BC_SRC=clim`, `CLIM_FILE=inputs/bc_clim/bc_clim.npz`, `INIT_BIN=so4`, `ADV_VPOS=1`, `FRAME3D=1`, `FRAME3D_EVERY=21` | tomas-jax `787c991` (gpu-fast), jax-rrtmgp `99d2d71` + `patches/jax-rrtmgp-zenith.patch` |
| `jcm_numerical` | `MET_SRC=jcm`, `BC_SRC=clim`, `CLIM_FILE=inputs/bc_clim/bc_clim_jcm.npz` (the CESM climatology regridded onto the JCM grid), `INIT_BIN=so4`, `ADV_VPOS=1`, `FRAME3D=1`, `FRAME3D_EVERY=21` | the same two commits |
| `jcm_online_numerical` | as `jcm_numerical` but `MET_SRC=jcm_online` (jcm-strat run live in a subprocess), plus the machine paths `JCM_PY`, `JCM_CACHE`, `JCM_SCRATCH` (`/data/JCM_stripped/jcm-strat/...` on the development machine) | the same two commits, plus jcm-strat `f3c2070` (`models/jcm-strat`, with its nested `external/jax-gcm` `35dc199`) |

`jcm_online_numerical` is **not bit-reproducible**: jcm-strat on GPU is not
deterministic run to run, so the linking is validated statistically
(`scripts/validation/validate_jcm_online.py`), never bitwise (MANIFEST.md).

The `CLIM_FILE` paths in the presets are relative to the checkout:
`run_prod.py` resolves a relative value of any PATH-type knob in a preset against
the repo root, so a preset does not depend on where the run is launched. The two
files are gitignored (`inputs/bc_clim/`); build them with `scripts/met/` (see
[BOUNDARY_CONDITIONS.md](./BOUNDARY_CONDITIONS.md) and
[inputs/README.md](../inputs/README.md)) or pass `--clim-file`. Every `/data/...`
default on this page, such as `JCM_DIR`, is a path on the development machine;
elsewhere pass your own.

A launch, from a run directory (the launcher refuses to write into the source
tree anywhere except under `runs/`):

```bash
REPO=/path/to/AIDE-SAI                          # absolute; the next line cd's away
mkdir -p $REPO/runs/inj6_30N && cd $REPO/runs/inj6_30N
python3 $REPO/src/run_prod.py --preset cesm_numerical --gpu 0 \
    --inj-so2 6 --inj-lat 30 --inj-mirror 1 --inj-zonal 0 --out-tag inj6_30N
```

`--preset` may be omitted (it defaults to `cesm_numerical`, or to an exported
`AIDE_PRESET`); `--preset path/to/file.env` uses a file outside `configs/`, and
`--preset none` applies no preset at all, leaving the module defaults — which
are **not** a linking (`BC_SRC=cesm`, every boundary field from the hourly
archive). `--list-presets` lists them; `--dry-run` prints every resolved value
with its source and checks the pins without launching. The run receives
`AIDE_PRESET=<name>`, so its log says which linking produced it.

**Precedence, highest first:** `--flag` > a variable already exported in the
environment > the preset > the launcher's run-control defaults (the **launcher**
column below) > the module default in `coupling.py` / `driver_fast.py` /
`radiation.py`. An exported variable that overrides a *different* preset value
prints `WARNING: exported X=... overrides the preset's X=...` — the stale
`export MET_SRC=...` left in a tmux pane is the failure this catches. Do not
`source` a preset file: that bypasses its validation and pin check, and once
its values are exported they outrank the next preset (configs/README.md explains).

A preset cannot set the scenario and run control (`INJ_*`, `N_HOURS`, `N_DAYS`,
`OUT_TAG`, `RESUME`, `H0`), the variables the code reads by presence
(`DEBUG`, `PROFILE`, `OH_SUBSTEPS`, `MICRO_SUBSTEPS`, `DUMP_PREMICRO`,
`INIT_SIGMA`), or the slot paths (`TOMAS_JAX_PATH`, `RRTMGP_PATH`,
`JCM_STRAT_PATH`, which come only from its `#@submodule` lines, so they cannot escape the pin check). It can
set only knobs in `KNOBS`. `--no-verify-pins` skips the pin check; the header
then says `(NOT VERIFIED)`, and the run is not the preset's model.

Anything not in `KNOBS` passes through from your environment unchanged (the
launcher copies the whole environment into the run), which is how the `FAST_*`
engine knobs below are set.

## Which knobs actually apply to my run?

Most of this page is inert for any given run. `run_prod.py` execs
`driver_fast.py`, which sets its own defaults (`RAD=1`, `RAD_MODE=anomaly`,
`RAD_EVERY=1`, `MICRO=full`, `STEP_HOURS=6`, `P_LO_HPA=1`, `P_HI_HPA=150`,
`FAST_CELL_CAP=50000`, `FAST_SORT=1`), imports `coupling.py`, and installs the
microphysics engine `MICRO_ENGINE` names into the `full` chain. It accepts
`MICRO=full` or `off` and `STEP_HOURS=6` only. The `P_HI_HPA` default is
therefore 150 hPa through the launcher; 100 hPa is `coupling.py`'s standalone
module default.

**Microphysics.** Composition is **sulfate + water** in the aerosol and
**SO2 + H2SO4** in the gas phase, 40 bins; organics, NH4/NH3, nitrate, dust and
sea salt are not carried, and ions are off.

- `MICRO_ENGINE=fast` (**the default, and what every preset runs**):
  `tomas_jax.fast` from the tomas-jax `gpu-fast` commit the presets pin,
  through `src/microphysics/tomas_fast.py`
  ([PROCESSES.md](./PROCESSES.md) §2). Its knobs are read by `tomas_fast.py` and
  are **not** in `KNOBS` — no flag, no preset; export them:

  | variable | default | meaning |
  |---|---|---|
  | `FAST_DT` | `360` | inner step [s]; 60 of them per 6 h coupling step |
  | `FAST_CELL_CAP` | `50000` (set by `driver_fast.py`; the module default is 250000) | cells per chunk. 50000 is faster than 250000 |
  | `FAST_SORT` | `1` | stiffness-sort the chunks, so a chunk's adaptive coagulation is not run to the pace of one stiff cell (~27 % of microphysics time) |
  | `FAST_FN_SCALE` | `1.0` | nucleation-rate scale factor |
  | `FAST_COAG_SUB_CAP`, `FAST_COND_SUB_CAP` | `256`, `40` | caps on the adaptive coagulation / condensation sub-step counts |
  | `FAST_COAG_CMAX` | `0.05` | coagulation sub-step control |
  | `FAST_MULTIGPU` | `0` | `1` = deal the cells over every visible GPU (`--gpu 0,1,...`), one thread per device; exact up to chunk composition |

  Under `fast` the `MICRO_DT`, `MICRO_NCOAG`, `MICRO_COAG`, `MICRO_FION`,
  `MICRO_MNFIX_TRIM`, `MICRO_MAX_WALL`, `MICRO_GPUS` and `CELL_CHUNK` knobs are
  **inert**: they belong to the `full` chain. `OH_SUBSTEPS` defaults to the
  engine's inner-step count (60) unless exported.
- `MICRO_ENGINE=full`: tomas-jax `dev`'s own `make_step` via
  `coupling.run_microphysics_full` — also what a bare `coupling.py` runs. It
  calls `make_step` arguments the pinned `gpu-fast` commit does not have, so it
  needs a preset whose tomas-jax slot is a `dev` checkout (configs/README.md,
  "A preset that needs a different model commit"). Then the `MICRO_*` knobs in
  the physics-stages table apply: `MICRO_DT` (60 s, must divide the step),
  `MICRO_NCOAG` (1), `MICRO_COAG` (`fixed`), `MICRO_FION` (0), `MICRO_MNFIX_TRIM`
  (1), `MICRO_MAX_WALL` (300 s abort), `MICRO_GPUS` (1), `CELL_CHUNK` (50000).
- `ALPHA_COND` (H2SO4 accommodation coefficient, 1.0) is read by both engines.
- `MICRO_SUBSTEPS` is gone and exits if set, so an old command line cannot
  silently mean something else. `N_COAG_SUBSTEPS` serves only the legacy
  `MICRO=coag` path, which `driver_fast.py` refuses.

**Transport.** `ADV_SCHEME=sl` (semi-Lagrangian horizontal, `ADV_SL_K=12`
sub-steps, `fct_lr`'s flux-form vertical sweep) is the default since the CH4
truth test ([PROCESSES.md](./PROCESSES.md) §1.10); the `ADV_SL_*` knobs apply to
it. `lr` (Lin-Rood flux form at CFL sub-steps), `ffsl` (`ADV_FFSL_*`) and `fast`
remain selectable. `ADV_CFL` (0.5) and `ADV_F32` (1) are bound by both entry
points; `ADV_VPOS=1` is set by every preset and must stay on
([VALIDATION.md](./VALIDATION.md)).

**Meteorology and boundaries.** `MET_SRC` and `BC_SRC` are separate seams:
`MET_SRC` supplies U/V/OMEGA/T (and, for `jcm` and `jcm_online`, the grid),
`BC_SRC` every other field. `JCM_DIR` and `MET_Q` apply only under
`MET_SRC=jcm|jcm_online`; `JCM_TEMPLATE`, `JCM_PY`, `JCM_CACHE`, `JCM_SCRATCH`,
`JCM_TE_CORRECTION`, `JCM_WORK` and the `JCM_STRAT_PATH` slot only under
`MET_SRC=jcm_online`; `MET_DIR` only under `MET_SRC=file`, `CLIM_FILE` only
under `BC_SRC=clim`.
[COUPLING_VARIABLES.md](./COUPLING_VARIABLES.md) lists what each one serves.

**Radiation.** Only `RAD_ENGINE=rrtmgp` exists here. `RAD_MODE` is `anomaly`
(heating relative to the reference MAM4 aerosol; default), `full` (absolute)
or `flux` (one solve, no feedback); `RAD_T_FEEDBACK=0` (default) keeps the
CESM temperature driving the microphysics, settling, RH and optics, with
`dT_rad` a diagnostic. `RAD_SOLAR` is `window_gq` with `RAD_SOLAR_NODES=3` and
`RAD_SOLAR_ALIGN=trailing` since 2026-11-17; `RAD_SOLAR_ALPHA` applies only to
`window_blend`. `RAD_REF_CACHE` caches the reference solve per archive hour
(bit-identical on a hit). `RAD_SAMPLE_EVERY` writes evolved-state samples for
emulator training; the labelling and training are not part of this repo (yet). Details:
[RADIATION.md](./RADIATION.md).

## Not in `KNOBS`, read by the code

| variable | read by | default | meaning |
|---|---|---|---|
| `FAST_*` | `tomas_fast.py` | see above | the `fast` engine |
| `ADV_SL_ITER` | `sl_lr.py` | `2` | departure-point iterations of the SL scheme |
| `ADV_FFSL_CMAX` | `ffsl_lr.py` | `64` | largest \|Courant\| the `ffsl` whole-cell sum may span; a guard, not a limit (64 cells is 90° of longitude) |
| `ADV_FFSL_FRAC` | `ffsl_lr.py` | `exact` | `ffsl` sliver fraction: `exact` (area of the sliver), `face`, `cell` ([PROCESSES.md](./PROCESSES.md) §1.10) |
| `AIDE_PRESET` | `run_prod.py` | `cesm_numerical` | preset to apply when `--preset` is not given; set for the run to the preset's label |
| `RAD_GBATCH` | `radiation.py` | `4` | `RAD_FAST` path: g-points per loop iteration (1 when `RAD_FAST=0`) |
| `RAD_DYN_MINOR` | `radiation.py` | `1` | `RAD_FAST` path: exact dynamic-length minor-gas scan (`rrtmgp_batched.py`) |
| `RAD_SPLIT_TROPO` | `radiation.py` | `1` | `RAD_FAST` path: static lower/upper split of that scan |
| `RAD_SW_SKIP_NIGHT` | `radiation.py` | `1` | `RAD_FAST` path: SW solved on sunlit columns only; `0` turns `RAD_FAST=auto` off |
| `RAD_SKIP_CLOUD` | `radiation.py` | `1` | `RAD_FAST` path: no cloud optics (the model has no clouds); `0` turns `RAD_FAST=auto` off |
| `RAD_GQ_FAST` | `radiation.py` | `1` | `RAD_FAST` path: window_gq through the fast solve; `0` turns `RAD_FAST=auto` off |
| `RAD_SW_PAD` | `radiation.py` | `4096` | `RAD_FAST` path: sunlit-column count padded to a multiple of this (bounds recompiles) |
| `RAD_SW_MAXCOLS` | `radiation.py` | `1000000` | `RAD_FAST` path: most sunlit columns per SW solve (larger counts are chunked) |

The `RAD_*` sub-knobs above are read at import and echoed in the
`RAD_FAST=... -> 1 (...)` header line; they only matter where `RAD_FAST`
resolves on. There is no `RAD_LAT_CHUNK_FAST` variable: the fast path's 192-row
chunk is a module constant, and `RAD_LAT_CHUNK` sets both chunk sizes.

**Performance switches (default on since 2026-11-23).** A bare run takes the
radiation fast path (`RAD_FAST=auto`, on with one GPU, wet optics and
`RAD_SOLAR=window_gq`; the header says `RAD_FAST=auto -> 0|1 (reason)`), the
device microphysics adapter (`MICRO_WRAP_DEVICE=1`), device MAM4 binning
(`MAM4_BIN_DEVICE=1`), the device SL wind resample (`ADV_SL_DEVICE_RESAMPLE=1`)
and background checkpoint writes (`CKPT_ASYNC=1`). `MICRO_WRAP_DEVICE` and
`MAM4_BIN_DEVICE` are not bit-identical (<= 2.5e-12 rel in a few cells, within
run-to-run GPU noise). Setting all five to `0` runs the pre-2026-11-23 code
path and, with `ADV_SL_RHONORM=0` as well, reproduces the presets' 2-year runs bit for bit. They are not
stamped in the checkpoint, so a RESUME does not report a change of them.
`MANIFEST.md` § Code state has the measurements.

## Resuming across a changed configuration

`INJ_*` mismatches against the checkpoint are **refused**, and so is a
different grid (`GRID_CFG`, e.g. resuming a `jcm_numerical` checkpoint under `cesm_numerical`).
`jcm_numerical` and `jcm_online_numerical` share a grid, so a RESUME between
them is allowed, with a `PHYS_CFG` warning (`MET_SRC=jcm_online`). The
physics-mode flags (`WET_SETTLING`, `WET_OPTICS`, `SETTLE`, `ADV_VPOS`, and the
rest of `PHYS_CFG`) only **warn** — the state is still valid, the model
integrating it forward just changes at the seam. The arrays are append-only, so
adding a field never locks out an older checkpoint.

## Flag ↔ environment variable cross-reference

Generated from the `KNOBS` and `GROUPS` tables in `src/run_prod.py` and the
preset files (2026-11-20; the perf switches added 2026-11-23); the **meaning** column is the `--help` text. The
**launcher** column is `run_prod.py`'s run-control default; a dash means the
launcher sets nothing and the module default (stated in the help text where it
has one) stands. The **preset** column shows the value both presets set, or
`cesm_numerical` / `jcm_numerical` where they differ. `jcm_online_numerical`
sets what `jcm_numerical` sets everywhere except in the meteorology table, which
has a column of its own.

### run control

| flag | variable | launcher | `cesm_numerical` / `jcm_numerical` preset | meaning |
|---|---|---|---|---|
| `--n-hours` | `N_HOURS` | `2160` | — | run length in hours (2160 = 90 d; 8760 = 1 yr) |
| `--n-days` | `N_DAYS` | — | — | run length in days (N_HOURS wins if both given) |
| `--out-tag` | `OUT_TAG` | `prod90d` | — | output tag; ALL outputs and checkpoints are keyed by it |
| `--resume` | `RESUME` | `0` | — | resume from coupled_state_<TAG>_ckpt.npz |
| `--h0` | `H0` | — | — | start hour index into the CESM h1 series (0 = 1996-01-01 00Z) |
| `--step-hours` | `STEP_HOURS` | — | — | coupling step [h]; one advect+coag per step (default 6) |
| `--state-ckpt` | `STATE_CKPT` | `1` | — | write the 3-D restart checkpoint (needed for --resume) |
| `--frame-every` | `FRAME_EVERY` | `24` | — | spatial frame every N hours (>= N_HOURS = final frame only) |
| `--frame3d` | `FRAME3D` | — | — | 1 = write the full 3-D state + per-cell size statistics + TOA fields after EVERY step to frames3d/step_<s>.npz (~480 MB/step, ~700 GB/yr) |
| `--frame3d-every` | `FRAME3D_EVERY` | — | — | write frames3d only every N-th step (output-only; N coprime to 4 rotates the UTC hour) |
| `--ckpt-every-frames` | `CKPT_EVERY_FRAMES` | — | — | write the state checkpoint every N frames (output cadence only) |
| `--ckpt-async` | `CKPT_ASYNC` | — | — | perf: snapshot the _ckpt arrays to host and write them in a background thread (default 1 since 2026-11-23; 0 = synchronous writes, the old default) |
| `--frame-levels` | `FRAME_LEVELS` | — | — | 'probe' (default) = one PROBE_HPA level; 'all' adds a zonal-mean frame at every level (+1.1 GB/yr) and unlocks the lat-height figures and gifs |
| `--log-every` | `LOG_EVERY` | — | — | progress line every N coupling steps |
| `--probe-hpa` | `PROBE_HPA` | — | — | diagnostic/frame level (default 50) |
| `--debug` | `DEBUG` | `1` | — | verbose per-step diagnostics |
| `--profile` | `PROFILE` | `1` | — | per-stage timing lines |

### domain and grid

| flag | variable | launcher | `cesm_numerical` / `jcm_numerical` preset | meaning |
|---|---|---|---|---|
| `--p-lo-hpa` | `P_LO_HPA` | — | — | top of the band (default 1) |
| `--p-hi-hpa` | `P_HI_HPA` | — | — | bottom of the band (coupling.py default 100; driver_fast.py, i.e. every launcher run, sets 150) |
| `--n-lev` | `N_LEV` | — | — | sub-sample the band to ~N levels (0 = full contiguous band) |
| `--n-bins` | `N_BINS` | — | — | TOMAS bin count (0 = native 40) |
| `--diag-core-hpa` | `DIAG_CORE_HPA` | — | — | core window for the 'int' burden diagnostics |

### injection scenario (the knobs meant to change run to run)

| flag | variable | launcher | `cesm_numerical` / `jcm_numerical` preset | meaning |
|---|---|---|---|---|
| `--inj-so2` | `INJ_SO2_TG_YR` | `0.0` | — | SO2 injection rate. DEFAULT 0 = no-injection control |
| `--inj-h2so4` | `INJ_H2SO4_TG_YR` | `0.0` | — | direct H2SO4(g) injection rate |
| `--inj-hpa` | `INJ_HPA` | `55.0` | — | injection altitude (snapped to the nearest model level) |
| `--inj-lat` | `INJ_LAT` | `0.0` | — | injection latitude (snapped to the nearest row) |
| `--inj-lon` | `INJ_LON` | `180.0` | — | injection longitude (ignored when --inj-zonal 1) |
| `--inj-zonal` | `INJ_ZONAL` | `1` | — | 1 = spread around the whole latitude ring, 0 = single cell |
| `--inj-mirror` | `INJ_MIRROR` | `0` | — | release at BOTH +lat and -lat, total split 50/50 (not doubled) |

### aerosol source / initial condition

| flag | variable | launcher | `cesm_numerical` / `jcm_numerical` preset | meaning |
|---|---|---|---|---|
| `--aer-src` | `AER_SRC` | — | — | IC/BC reservoir source (default mam4) |
| `--init-bin` | `INIT_BIN` | — | `so4` | how MAM4 modes are binned onto TOMAS |
| `--mam4-bin-device` | `MAM4_BIN_DEVICE` | — | — | perf: run bin_mode's erf part (INIT_BIN=so4) on the device as eager jax ops, no host round trip (default 1 since 2026-11-23; 0 = numpy, the old default) |
| `--init-sigma` | `INIT_SIGMA` | — | — | override the log-normal mode width used for binning |
| `--carma-file` | `CARMA_FILE` | — | — | CARMA reservoir file (AER_SRC=carma) |
| `--carma-frame` | `CARMA_FRAME` | — | — | time index into the CARMA file |
| `--carma-rho` | `CARMA_RHO` | — | — | CARMA particle density |
| `--carma-subbin` | `CARMA_SUBBIN` | — | — | CARMA sub-binning factor |

### boundaries

| flag | variable | launcher | `cesm_numerical` / `jcm_numerical` preset | meaning |
|---|---|---|---|---|
| `--n-bc-top` | `N_BC_TOP` | — | — | band levels pinned/served at the top face (default 1) |
| `--n-bc-bot` | `N_BC_BOT` | — | — | band levels pinned/served at the bottom face (default 1) |
| `--bc-edge` | `BC_EDGE` | — | — | aerosol edge treatment (derived from ADV_WCONT) |
| `--bc-gas` | `BC_GAS` | — | — | gas edge treatment (derived from BC_EDGE) |
| `--bc-bot-aer` | `BC_BOT_AER` | — | — | scale the bottom-face aerosol inflow (0 = clean upwelling) |

### physics stages

| flag | variable | launcher | `cesm_numerical` / `jcm_numerical` preset | meaning |
|---|---|---|---|---|
| `--micro` | `MICRO` | — | — | microphysics mode; 'off' is transport-only |
| `--micro-dt` | `MICRO_DT` | — | — | MICRO=full: tomas-jax dev step length (default 60, dev's; must divide the 6 h step) |
| `--micro-ncoag` | `MICRO_NCOAG` | — | — | MICRO=full: forward-Euler coag sub-steps per dev step (default 1; dev's own default 10 is within 5e-4 in total N at 3x the cost) |
| `--micro-coag` | `MICRO_COAG` | — | — | MICRO=full: dev's coag solver (default fixed = euler_fixed, dev's strat setup; adaptive = euler_adaptive, ~100x slower and over the micro budget) |
| `--micro-fion` | `MICRO_FION` | — | — | MICRO=full: ion-pair production for ion-induced nucleation (default 0 = off; dev's strat setup uses 10) |
| `--micro-mnfix-trim` | `MICRO_MNFIX_TRIM` | — | — | MICRO=full: MNFIX over out-of-range bins only (default 1; bit-identical to 0 and 5x faster here) |
| `--micro-max-wall` | `MICRO_MAX_WALL` | — | — | MICRO=full: abort the run if one step's microphysics takes longer (default 300) |
| `--micro-gpus` | `MICRO_GPUS` | — | — | MICRO=full: spread the micro cells over the first N visible GPUs (default 1; needs --gpu to expose them) |
| `--settle` | `SETTLE` | — | — | gravitational settling (default on) |
| `--wet-settling` | `WET_SETTLING` | — | — | hygroscopic growth in the settling velocity |
| `--wet-optics` | `WET_OPTICS` | — | — | hygroscopic growth in the optics |
| `--rad` | `RAD` | — | — | radiation on/off |
| `--rad-every` | `RAD_EVERY` | — | — | coupling steps between radiation calls |
| `--rad-mode` | `RAD_MODE` | — | — | heating-rate definition (flux = one solve, no feedback) |
| `--rad-t-feedback` | `RAD_T_FEEDBACK` | — | — | 0 (default) = microphysics, settling, RH and the radiation optics all see the CESM temperature; dT_rad and the heating rate are diagnostics, and dT_rad enters only the perturbed call's emission temperature (anomaly mode's Planck damping). 1 = the old closed loop: T_met + dT_rad everywhere |
| `--rad-solar` | `RAD_SOLAR` | — | — | solar geometry per radiation call: window_gq = shortwave at the Gauss nodes of the sunlit part of the interval, longwave once (default); instant = the sun at the call hour held for the interval (the pre-2026-11-17 default, bit-identical to earlier runs; leaves the dusk/dawn stripes in dT_rad); window / window_ins / window2 / window2_ins / window_blend / dither = experimental variants (docs/RADIATION_ARTIFACT_EXPLORATION.md) |
| `--rad-solar-alpha` | `RAD_SOLAR_ALPHA` | — | — | window_blend weight in [0, 1]: cos(zenith) = (1-A) x sunlit-mean + A x energy-weighted (default 0.53, fitted) |
| `--rad-solar-nodes` | `RAD_SOLAR_NODES` | — | — | window_gq solves per radiation call, at the Gauss nodes of the sunlit part of the interval (default 3) |
| `--rad-solar-align` | `RAD_SOLAR_ALIGN` | — | — | where window_gq's window sits: trailing = the step that just ended, t-6..t h, so dT_rad at t holds only past heating (default); centred = t-3..t+3 h (window_gq before 2026-11-17) |
| `--arf-avg-h` | `ARF_AVG_H` | — | — | trailing window for the reported TOA forcing |
| `--alpha-cond` | `ALPHA_COND` | — | — | H2SO4 accommodation coefficient |

### chemistry, nucleation and coagulation

| flag | variable | launcher | `cesm_numerical` / `jcm_numerical` preset | meaning |
|---|---|---|---|---|
| `--oh-sza` | `OH_SZA` | — | — | diurnal OH parabola in cos(SZA) |
| `--oh-peak` | `OH_PEAK` | — | — | noon OH peak (default 2.3e6) |
| `--oh-substeps` | `OH_SUBSTEPS` | — | — | OH samples per coupling step |
| `--n-coag-substeps` | `N_COAG_SUBSTEPS` | — | — | forward-Euler coag substeps (legacy MICRO=coag path) |

### microphysics engine selection

| flag | variable | launcher | `cesm_numerical` / `jcm_numerical` preset | meaning |
|---|---|---|---|---|
| `--micro-engine` | `MICRO_ENGINE` | — | — | which engine advances the run (default fast = tomas_jax.fast, tomas-jax gpu-fast; full = tomas-jax dev make_step, which needs a preset whose tomas-jax slot is a dev checkout) |
| `--micro-wrap-device` | `MICRO_WRAP_DEVICE` | — | — | perf: MICRO_ENGINE=fast adapter (microphysics/tomas_fast.py) does its unit conversion, transposes, consistency clip and OH field on the device, no host round trip for num/mas (not bit-identical: device reductions/ordering give ~1e-12 rel diffs in a few cells, within run-to-run GPU noise; default 1 since 2026-11-23, not used under FAST_MULTIGPU on >1 device; 0 = numpy, the old default) |

### advection

| flag | variable | launcher | `cesm_numerical` / `jcm_numerical` preset | meaning |
|---|---|---|---|---|
| `--adv-scheme` | `ADV_SCHEME` | — | — | transport form (default sl = semi-Lagrangian split, K=12, the JCM/dinosaur kernel, docs/TRANSPORT_SL_HANDOFF.md: 0.76x lr's CH4-truth error, ~0.9 s/step vs 6-8 s; lr = Lin-Rood flux form at CFL sub-steps; ffsl = lr at large Courant number; fast = advective form) |
| `--adv-ffsl-k` | `ADV_FFSL_K` | — | — | ffsl: sub-steps per coupling step (default 24, the production operating point; 6 = hourly; a convergence knob, each ~1/200 of an lr step) |
| `--adv-ffsl-split` | `ADV_FFSL_SPLIT` | — | — | ffsl: sweep order (strang = z/2, xy\|yx alternating, z/2, default; lie = fct_lr's x y z) |
| `--adv-sl-k` | `ADV_SL_K` | — | — | SL: sub-steps per coupling step (default 12) |
| `--adv-sl-fixer` | `ADV_SL_FIXER` | — | — | SL: per-tracer mass fixer on the closed column (default global; local = Bermejo-Conde) |
| `--adv-sl-hourly` | `ADV_SL_HOURLY` | — | — | sl and ffsl: 1 = read the hourly CESM winds between the step snapshots; 0 (default) = the two snapshots interpolated in time, as lr does and as production winds arrive |
| `--adv-sl-vert` | `ADV_SL_VERT` | — | — | SL: vertical treatment (default flux = fct_lr's open-face flux sweep; sl = closed-column 3-D gather, A/B-failed) |
| `--adv-sl-omega` | `ADV_SL_OMEGA` | — | — | SL flux vertical: face omega from continuity (lr's, default) or CESM centers averaged |
| `--adv-sl-rho` | `ADV_SL_RHO` | — | — | SL flux vertical, lie split only: carry the air mass 1 - dt div(v) through the horizontal step (strang always does; default 0) |
| `--adv-sl-split` | `ADV_SL_SPLIT` | — | — | SL flux vertical: strang (default) = V/2 H V/2 with the air mass carried (second-order face flux); lie = H then V (first order at the open face) |
| `--adv-sl-jac` | `ADV_SL_JAC` | — | — | SL flux vertical: 1 (default) = air mass from the Jacobian of the departure map and the bookkeeping remainder returned through the faces; 0 = Eulerian 1 - dt div(v) and the global column fixer |
| `--adv-sl-vorder` | `ADV_SL_VORDER` | — | — | SL: vertical gather order (default linear) |
| `--adv-cfl` | `ADV_CFL` | — | — | horizontal CFL target (production 0.5) |
| `--adv-passive` | `ADV_PASSIVE` | — | — | TEST: advect ONE total-sulfate tracer (+ the two gases) and rescale every bin by the cell's mass ratio (frozen per-cell shape); score against the same run with 0 |
| `--adv-f32` | `ADV_F32` | — | — | float32 transport sweeps (production 1) |
| `--adv-vpos` | `ADV_VPOS` | — | `1` | VERTICAL POSITIVITY LIMITER -- see the header; do not turn off |
| `--adv-wcont` | `ADV_WCONT` | — | — | rederive omega from discrete continuity (opens the faces) |
| `--adv-metric` | `ADV_METRIC` | — | — | cos(phi) area metric in the y-sweep |
| `--adv-dxfix` | `ADV_DXFIX` | — | — | true grid spacing in the x-sweep |
| `--adv-polar` | `ADV_POLAR` | — | — | polar cap treatment (zonal = stirred, mass-conserving) |
| `--adv-sl-rhonorm` | `ADV_SL_RHONORM` | — | — | SL: 1 (default since PR #4, merged into dev 2026-11-24) = rescale each level's air mass after the horizontal step so the level total is conserved, which stops the per-level fixer imprinting a drift (uniform q: +2% at 1.24 hPa in 10 days -> 1e-4); 0 = every run made before PR #4 reached dev (2026-11-24), including the 2-year runs |
| `--adv-sl-fixlog` | `ADV_SL_FIXLOG` | — | — | SL diagnostic: 1 = log what the level and column mass fixers corrected every sub-step (results unchanged; default 0) |
| `--adv-sl-device-resample` | `ADV_SL_DEVICE_RESAMPLE` | — | — | perf: SL time-resample of the step winds on the device (same float32 formula, bit-identical; engages for float32 winds, prints ENGAGED/not engaged once; default 1 since 2026-11-23; 0 = numpy, the old default) |

### meteorology source and boundary climatology (set by the preset)

| flag | variable | launcher | `cesm_numerical` / `jcm_numerical` preset | `jcm_online_numerical` preset | meaning |
|---|---|---|---|---|---|
| `--met-src` | `MET_SRC` | — | `cesm` / `jcm` | `jcm_online` | where U/V/OMEGA/T come from (set by the preset): cesm = the hourly CESM h1 archive; jcm = a jcm-strat run directory (--jcm-dir), 6-hourly, and the model GRID with it (T63 Gaussian 96x192, JCM's native levels in the band; needs --clim-file on that grid, scripts/met/regrid_clim_to_grid.py); jcm_online = the same fields and grid from the jcm-strat AL7h model integrated LIVE beside this one (src/met/jcm_server.py, a subprocess under --jcm-py on the same GPU; code from the jcm-strat slot), started from the --jcm-dir chain's 1 January checkpoint; file = step-end frames in MET_DIR (the seam for an emulated dycore). Everything else (Q, gases, reservoirs, the radiation column below the band) comes from --bc-src |
| `--jcm-dir` | `JCM_DIR` | — | — | — | MET_SRC=jcm\|jcm_online: the jcm-strat run directory (default /data/JCM_stripped/jcm-strat/runs/p31/p31_AL7h70, the AL7h chain); read-only. jcm_online reads only its 1 January checkpoints, their last output frame and the template segment's output (for the grid) |
| `--jcm-template` | `JCM_TEMPLATE` | — | — | — | MET_SRC=jcm_online: the resolved Hydra config the live JCM is built from (default <jcm-dir>/<chain>_19960101/.hydra/config.yaml); start time, QBO year and init file are set per year |
| `--jcm-py` | `JCM_PY` | — | — | `/data/JCM_stripped/jcm-strat/.venv/bin/python` (machine path) | MET_SRC=jcm_online: the interpreter the JCM server runs under, a Python 3.11 venv with jcm-strat's dependencies (JAX 0.10; default <JCM_STRAT_PATH>/.venv/bin/python). Machine-specific: a submodule checkout has no venv. jcm_strat and jcm themselves are imported from the slot, not from this venv |
| `--jcm-cache` | `JCM_CACHE` | — | — | `/data/JCM_stripped/jcm-strat/cache` (machine path) | MET_SRC=jcm_online: jcm-strat's data cache, holding era5/ (the regridded ERA5 nudging windows, JCM_ERA5_CACHE) and huggingface/ (jcm boundary files, HF_HOME) (default <JCM_STRAT_PATH>/cache). Machine-specific; a missing ERA5 window is downloaded and regridded on first use |
| `--jcm-scratch` | `JCM_SCRATCH` | — | — | `/data/JCM_stripped/jcm-strat/scratch` (machine path) | MET_SRC=jcm_online: jcm's scratch directory, $SCRATCH (its JAX compilation cache is $SCRATCH/jcm-jax-cache; default <JCM_STRAT_PATH>/scratch). Machine-specific; cannot change results |
| `--jcm-te-correction` | `JCM_TE_CORRECTION` | — | — | — | MET_SRC=jcm_online: the AL7h T_e correction file (default the jcm-strat slot's tracked docs/outputs/25_al7_half/corrections/AL7h.nc; the template's own path names a removed jcm-strat worktree) |
| `--jcm-work` | `JCM_WORK` | — | — | — | MET_SRC=jcm_online: where the live JCM writes its year-end checkpoints (default jcm_online/ in the run directory; a RESUME restarts from them) |
| `--met-q` | `MET_Q` | — | — | — | MET_SRC=jcm\|jcm_online: water vapour from the BC source (bc, default: the regridded CESM climatology -- jcm-strat is dry and its q is a passive constant) or from JCM (met) |
| `--met-dir` | `MET_DIR` | — | — | — | MET_SRC=file: directory holding u/v/omega/T.npy + meta.json |
| `--met-rh` | `MET_RH` | — | — | — | RH seen by settling, the microphysics' water uptake (full chain) and the FRAME3D wet diagnostics, recomputed every step from Q, T and p: liq (default) = over liquid water, Tabazadeh 1997 eq. 1. The radiation's wet optics ignore MET_RH and always compute liquid RH from Q themselves; q = over ice (the old MET_SRC=file default; 1.6-1.9x liq in the band); cesm = the archive's RELHUM as is |
| `--bc-src` | `BC_SRC` | — | `clim` | `clim` | where every non-wind boundary field comes from (set by the preset): clim = the monthly climatology in CLIM_FILE (scripts/met/make_bc_climatology.py); cesm (module default) = the hourly archive |
| `--clim-file` | `CLIM_FILE` | — | `inputs/bc_clim/bc_clim.npz` / `inputs/bc_clim/bc_clim_jcm.npz` | `inputs/bc_clim/bc_clim_jcm.npz` | BC_SRC=clim: the climatology, on the MODEL grid (set by the preset; the jcm_numerical preset uses the CESM climatology regridded onto the JCM grid by scripts/met/regrid_clim_to_grid.py) |

### radiation gases, chunking, engine and reference cache (docs/RADIATION.md)

| flag | variable | launcher | `cesm_numerical` / `jcm_numerical` preset | meaning |
|---|---|---|---|---|
| `--co2-ppm` | `CO2_PPM` | — | — | background CO2 for RRTMGP |
| `--n2o-ppb` | `N2O_PPB` | — | — | background N2O for RRTMGP |
| `--rad-lat-chunk` | `RAD_LAT_CHUNK` | — | — | latitude rows per radiation chunk (GPU memory; default 96 for the chunk loop, 192 for the RAD_FAST path; an explicit value sets both) |
| `--rad-fast` | `RAD_FAST` | — | — | perf: radiation fast path (one lat chunk, SW on sunlit columns only, g-point batching, exact minor-gas scan, no cloud optics; window_gq too). Default auto (since 2026-11-23): on where validated (1 GPU, wet optics, RAD_SOLAR=window_gq, the normal solve), else the chunk loop, decided per radiation driver and stated in the header; 1 = force (refuses anything else); 0 = the chunk loop (the old default) |
| `--rad-engine` | `RAD_ENGINE` | — | — | which engine solves the EVOLVED radiation call (rrtmgp = the physics, the only one in this repo) |
| `--rad-ref-cache` | `RAD_REF_CACHE` | — | — | per-archive-hour cache of the reference-aerosol solve and the clear-sky fields (default inputs/cache/rad_ref; bit-identical on a hit, computes+stores on a miss; 0 = off) |
| `--rad-sample-every` | `RAD_SAMPLE_EVERY` | — | — | write rad_sample_<TAG>_s<step>.npz (evolved bins, dT_rad, hour) every N radiation steps -- training data for a radiation emulator (0 = off) |

### memory / performance (cannot change results)

(The perf switches are listed in their stage's table: `CKPT_ASYNC` under run
control, `MAM4_BIN_DEVICE` under aerosol source, `MICRO_WRAP_DEVICE` under the
microphysics engine, `ADV_SL_DEVICE_RESAMPLE` under advection, `RAD_FAST` under
radiation.)

| flag | variable | launcher | `cesm_numerical` / `jcm_numerical` preset | meaning |
|---|---|---|---|---|
| `--cell-chunk` | `CELL_CHUNK` | — | — | cells per microphysics vmap batch |
| `--tracer-chunk` | `TRACER_CHUNK` | — | — | tracers per advection batch (0 = all 82 at once) |

### input data locations

| flag | variable | launcher | `cesm_numerical` / `jcm_numerical` preset | meaning |
|---|---|---|---|---|
| `--cesm-dir` | `CESM_DIR` | — | — | root of the CESM h1 tseries archive |
| `--cesm-prefix` | `CESM_PREFIX` | — | — | CESM case-name prefix of the h1 files |
| `--cesm-suf` | `CESM_SUF` | — | — | date-range suffix of the h1 files |
| `--tomas-jax-path` | `TOMAS_JAX_PATH` | — | — | tomas-jax checkout (default models/tomas-jax) |
| `--rrtmgp-path` | `RRTMGP_PATH` | — | — | jax-rrtmgp checkout (default models/jax-rrtmgp) |
| `--jcm-strat-path` | `JCM_STRAT_PATH` | — | — (`jcm_online_numerical`: its `jcm-strat` slot) | jcm-strat checkout, recursively initialised (jcm_strat + external/jax-gcm's jcm), that the MET_SRC=jcm_online server imports (default models/jcm-strat) |

### debug dumps

| flag | variable | launcher | `cesm_numerical` / `jcm_numerical` preset | meaning |
|---|---|---|---|---|
| `--dump-premicro` | `DUMP_PREMICRO` | — | — | dump the pre-microphysics state to this file |
| `--dump-premicro-step` | `DUMP_PREMICRO_STEP` | — | — | step at which to take that dump |

### launcher-only

| flag | variable | default | meaning |
|---|---|---|---|
| `--preset` | `AIDE_PRESET` | `cesm_numerical` | linking preset: a name in `configs/`, a path to a `.env` file, or `none` |
| `--list-presets` | — | — | list the presets (name and first comment line) and exit |
| `--no-verify-pins` | — | — | skip the submodule pin check; the header says `(NOT VERIFIED)` |
| `--dry-run` | — | — | print the resolved environment with sources, check the pins, and exit |
| `--gpu` | `GPU` → `CUDA_VISIBLE_DEVICES` | `0` | card(s) to pin the run to |
| `--cuda-driver-lib` | `CUDA_DRIVER_LIB` | *(search)* | directory holding `libcuda.so.1`. The launcher tries `/run/nvidia/driver/usr/lib/x86_64-linux-gnu` (this machine's containerised driver mount), then searches `/run/nvidia` and `/usr/lib`; without it **JAX silently falls back to CPU**. An empty string disables the search, for a normal CUDA install |
| `--preallocate` | `XLA_PYTHON_CLIENT_PREALLOCATE` | `false` | grow GPU memory on demand instead of preallocating 75 % of the card |
| `--allow-source-tree` | — | — | override the refusal to write into the checkout |
