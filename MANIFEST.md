# MANIFEST — AIDE-SAI

The canonical reference for this tree: what each file is, every knob, and **why
each default is what it is**. `README.md` is the orientation; where `docs/`
disagrees with this file, this file is current.

## Scope

AIDE-SAI is a modular dycore–aerosol–radiation coupler for stratospheric aerosol
injection, in JAX on GPU. Each physical process sits in a slot — meteorology
(the dycore), boundary climatology, transport, aerosol microphysics, settling,
radiation — and each slot is filled by a numerical scheme or, in future, a
learned emulator. The end state is model-agnostic coupling: other meteorology
sources (including emulated dycores, which would close the circulation feedback
the current setup lacks) and other aerosol schemes (CARMA, MAM, GLOMAP).

**What ships today is four linkings, all fully numerical** (§ Linkings are
presets): CESM2 meteorology on the f09 grid (`configs/cesm_numerical.env`, the default),
jcm-strat meteorology on the JCM grid read from the AL7h70 chain's output
(`configs/jcm_numerical.env`), and the same jcm-strat model run live beside the
coupler (`configs/jcm_online_numerical.env`), plus CESM2 meteorology with
tomas-jax dev's adaptive-sub-stepping microphysics
(`configs/cesm_adaptive_substep.env`, 2026-11-25). All use the
CESM climatology for every non-wind field, `tomas_jax.fast` microphysics (except
`cesm_adaptive_substep`: tomas-jax `make_step`, `MICRO_ENGINE=full`),
semi-Lagrangian transport and RRTMGP + Mie radiation. Every default,
measurement and caveat below refers to that stack unless it says otherwise.

Not part of this repo (yet): the TOMAS microphysics emulator and the engine A/B
harness, the radiation emulator, the MAM4-JAX driver, their training and
benchmark scripts, the `linking_plots/` site and the operator-study scripts.
An emulator would fill its slot behind the existing selector (table below)
rather than add a new code path.

The seams that make a slot swappable are deliberate and worth preserving:

| slot | selector | what fills it today |
|---|---|---|
| meteorology (dycore) | `MET_SRC=cesm\|jcm\|jcm_online\|file`. Every field goes through `open_var()`, and the met source also declares the GRID (`ds['lat']`, `ds['lon']`, `hyam/hybm`) | `cesm` (h1 archive, f09) and `jcm` (`src/met/met_jcm.py`, T63 Gaussian). `jcm_online` (`src/met/met_jcm_online.py`) serves the same fields from the JCM model *running* in a subprocess (`src/met/jcm_server.py`, its own interpreter), so a live dycore is one more class behind the same surface, and a later one can reuse the server's line protocol. `file` reads step-end frames from `MET_DIR` and is the seam an emulated dycore plugs into; no converter for it ships here |
| boundary climatology | `BC_SRC=cesm\|clim` + `CLIM_FILE` | `clim`: monthly CESM means on the model grid (`src/met/bc_clim.py`) |
| aerosol IC/BC source | `AER_SRC=mam4\|carma` | `mam4` (the default): CESM's MAM4 output fields binned onto the TOMAS bins. This is NOT the MAM4-JAX package |
| transport | `ADV_SCHEME=sl\|lr\|ffsl\|fast` | `sl`, K=12 |
| microphysics | `MICRO_ENGINE=fast\|full`, swapped into `coupling.py` by `driver_fast.py` at import instead of branched inside it — which is why the driver is a separate file | `fast`: `tomas_jax.fast` (tomas-jax gpu-fast `787c991`); `cesm_adaptive_substep`: `full`, tomas-jax `make_step` at PR #49 `0ba8f4b` (`models/tomas-jax-dev`) |
| radiation | `RAD_ENGINE=rrtmgp`; `RAD=0` drops it and the `jax-rrtmgp` dependency entirely; `RAD_REF_CACHE` serves the archive-only half of the stage from disk | RRTMGP + Mie (jax-rrtmgp `99d2d71` + patch) |
| which checkout fills a code slot | `#@submodule` lines in the preset (`TOMAS_JAX_PATH`, `RRTMGP_PATH`, `JCM_STRAT_PATH`) | `models/tomas-jax` (or `models/tomas-jax-dev` for `cesm_adaptive_substep`), `models/jax-rrtmgp`; `models/jcm-strat` (with its `external/jax-gcm`) for `MET_SRC=jcm_online` only |

A new scheme goes in behind one of these selectors; a new combination of
schemes is a new preset (next section), not a branch.

## Linkings are presets

A **linking** — which coupled model runs — is a file in `configs/`, applied by
`src/run_prod.py --preset NAME` (default `cesm_numerical`; `--preset none` runs on the
module defaults alone; `--list-presets` lists them). All linkings live on one
branch, and what distinguishes one from another is one readable file, not
hard-coded launcher presets that are visible only by diffing branches.

| preset | meteorology | grid | `CLIM_FILE` (relative to the repo root) | reproduces |
|---|---|---|---|---|
| `cesm_numerical` (default) | CESM2 h1 hourly U/V/OMEGA/T (`MET_SRC=cesm`) | f09, 192 × 288, 24 levels 1.2–143 hPa | `inputs/bc_clim/bc_clim.npz` | the 2-year CESM runs `fast_2yr_inj6_30NS` / `fast_2yr_ctrl`, bit-identically with `ADV_SL_RHONORM=0` and the perf switches off (`RAD_FAST=0 MICRO_WRAP_DEVICE=0 MAM4_BIN_DEVICE=0 ADV_SL_DEVICE_RESAMPLE=0 CKPT_ASYNC=0`; § Code state); with the perf defaults (on) within run-to-run GPU noise; `ADV_SL_RHONORM=1` (default since PR #4, merged into dev 2026-11-24) is a real change, § sl_lr level air mass |
| `jcm_numerical` | jcm-strat AL7h70, 6-hourly (`MET_SRC=jcm`, `JCM_DIR` at its default `/data/JCM_stripped/jcm-strat/runs/p31/p31_AL7h70`) | T63 Gaussian 96 × 192, 47 levels 1.01–137.8 hPa | `inputs/bc_clim/bc_clim_jcm.npz` (the CESM climatology regridded by `scripts/met/regrid_clim_to_grid.py`) | the 2-year JCM run `jcm_1yr_inj6_30NS`, bit-identically with `ADV_SL_RHONORM=0` and the perf switches off (`RAD_FAST=0 MICRO_WRAP_DEVICE=0 MAM4_BIN_DEVICE=0 ADV_SL_DEVICE_RESAMPLE=0 CKPT_ASYNC=0`); with the perf defaults (on) within run-to-run GPU noise; `ADV_SL_RHONORM=1` (default since PR #4, merged into dev 2026-11-24) is a real change, § sl_lr level air mass |
| `jcm_online_numerical` | jcm-strat AL7h **run live**, 6-hourly (`MET_SRC=jcm_online`; code from the `jcm-strat` slot, `models/jcm-strat` f3c2070; started from the same `JCM_DIR` chain) | as `jcm_numerical` | as `jcm_numerical` | the knob values and pins of the 1-year run `jcm_online_1yr_inj6_30NS` (2026-11-19) — **statistically, not bitwise**: jcm-strat on GPU is not deterministic run to run (§ Code state); that run had `ADV_SL_RHONORM=0` and the perf switches at 0, and their defaults (on) are untested on this linking (`RAD_FAST` adds ~2 GB on the GPU jcm-strat shares) |
| `cesm_adaptive_substep` | as `cesm_numerical` | as `cesm_numerical` | as `cesm_numerical` | no runs reproduced yet (added 2026-11-25). The microphysics slot is tomas-jax `make_step` (`MICRO_ENGINE=full`) from `models/tomas-jax-dev` at the head of tomas-jax PR #49 (`0ba8f4b`, branch `adaptive-coag`): heun_adaptive coagulation and adaptive gas sub-steps (§ Code state, "Adaptive-sub-stepping microphysics"). Re-pin to the merge commit once PR #49 lands |

All four also set `BC_SRC=clim`, `INIT_BIN=so4`, `ADV_VPOS=1` and, since
2026-11-24, `FRAME3D=1 FRAME3D_EVERY=21` (output-only: the full 3-D state every
5.25 days, rotating through the four UTC hours, ~33 GB/yr on f09 and ~22 GB/yr on
T63; the JCM runs above wrote none, and without 3-D frames no time-mean map or
column field can be made afterwards), and pin the
same jax-rrtmgp slot and, except `cesm_adaptive_substep` (`models/tomas-jax-dev`),
the same tomas-jax slot; `jcm_online_numerical` also pins
`jcm-strat` and sets three machine paths (`JCM_PY`, `JCM_CACHE`,
`JCM_SCRATCH`). A dry run of each preset with those runs' scenario resolves the knob
values their logs recorded (the `cesm_numerical` preset additionally states
`MET_SRC=cesm`, which was coupling.py's default when those runs were made).
The climatology files live in the checkout under `inputs/bc_clim/` and are
gitignored build products (`inputs/README.md`): `scripts/met/make_bc_climatology.py`
builds the f09 one from the archive (~3 GB), `regrid_clim_to_grid.py` the JCM
one from it (~20 s). The launcher resolves a relative value of any PATH-type
knob in a preset (`PATH_KNOBS` in `run_prod.py`) against the repo root, so the
presets hold no machine paths and work from any run directory; to use a copy
elsewhere, export `CLIM_FILE` as an absolute path.

**The format** (parsed by `run_prod.load_preset`, no shell evaluation):

- `KEY=VALUE` — only knobs in `run_prod.py`'s `KNOBS` table, so every value is
  documented in `--help` and echoed. **Forbidden** (`PRESET_FORBIDDEN`): the
  scenario (`INJ_*`), the run (`N_HOURS`, `N_DAYS`, `OUT_TAG`, `RESUME`, `H0`),
  and the variables read by presence or truthiness (`DEBUG`, `PROFILE`,
  `OH_SUBSTEPS`, `MICRO_SUBSTEPS`, `DUMP_PREMICRO`, `INIT_SIGMA`), where writing
  out a "default" would change behaviour — `DEBUG=0` turns debugging ON. The
  slot paths `TOMAS_JAX_PATH` / `RRTMGP_PATH` / `JCM_STRAT_PATH` come only from
  `#@submodule`.
- `#@submodule SLOT PATH SHA [patch=FILE]` — the checkout that fills a code
  slot (`tomas-jax` → `TOMAS_JAX_PATH`, `jax-rrtmgp` → `RRTMGP_PATH`,
  `jcm-strat` → `JCM_STRAT_PATH`), the full 40-character commit it must be at,
  and a patch that must be applied on top. Every preset pins tomas-jax and
  jax-rrtmgp; `jcm-strat` is pinned exactly by the presets that select
  `MET_SRC=jcm_online` (`SLOT_NEEDED_IF` in `run_prod.py`), so the other
  linkings need neither the submodule checked out nor a commit they do not use.

**What the launcher does with it.** It refuses to start if a slot's checkout is
not at its pinned commit or the patch is not applied (`git apply --check
--reverse`), and says the command that fixes it; `--no-verify-pins` skips the
check, and the run is then not the preset's model. It points the slot variables
at the pinned checkouts and resolves every knob with the precedence

`--flag` > exported env > preset > launcher run-control defaults (`KNOBS`) >
`driver_fast.py`'s `setdefault`s > `coupling.py` / module defaults

and prints a WARNING when an exported variable overrides a different preset
value — the stale-tmux-shell failure mode. That includes the slot variables: an
exported `TOMAS_JAX_PATH` wins over the preset's slot, with a warning, and the
pin check then describes a checkout the run does not use. Each resolved value
is echoed with its source (`flag`, `env`, `preset:<name>`, `launcher`).

**Where stack-defining values live now.** In `configs/`. The `preset` column of
`KNOBS` holds only run-control and scenario defaults (`N_HOURS=2160`,
`OUT_TAG=prod90d`, `RESUME=0`, `STATE_CKPT=1`, `FRAME_EVERY=24`, `DEBUG=1`,
`PROFILE=1`, `INJ_SO2_TG_YR=0`, `INJ_H2SO4_TG_YR=0`, `INJ_HPA=55`, `INJ_LAT=0`,
`INJ_LON=180`, `INJ_ZONAL=1`, `INJ_MIRROR=0`). Defaults that are the same for
every linking stay as module defaults in code (`ADV_SCHEME=sl`,
`MICRO_ENGINE=fast`, `RAD_SOLAR=window_gq`, …) or as `driver_fast.py`
`setdefault`s (`P_LO_HPA=1`, `P_HI_HPA=150`, `RAD_MODE=anomaly`,
`FAST_CELL_CAP=50000`, `FAST_SORT=1`); a value that differs between linkings
belongs in the presets.

**Which tomas-jax.** A slot is always a git submodule pinned to an exact commit,
never a vendored copy. A different version of a dependency — e.g. tomas-jax
`dev` (`835ee24`) for `MICRO_ENGINE=full` — goes in as a SECOND submodule path
plus a new preset whose `tomas-jax` slot points there; that is how a preset
chooses its submodule. `models/tomas-jax-dev` (tomas-jax PR #49 head `0ba8f4b`,
branch `adaptive-coag`; 2026-11-25) is the first such path, pinned by
`cesm_adaptive_substep`.
`scripts/validation/test_configs.py` (in CI) checks that every preset parses
with the launcher's own loader, that it pins exactly the slots it needs, that
each pin equals the commit the superproject records for that submodule (the
index gitlink, `git ls-files --stage`; HEAD's on a clean checkout), that each
named patch exists, and that the default preset exists.

**Changing a preset or a default** needs an entry in this file (§ Code state)
saying what changed, when and why, because it changes what a bare run means.
A new linking, including a mixed numerical/emulated one, is a new preset plus,
if needed, a new implementation behind an existing selector (`MET_SRC`,
`BC_SRC`, `AER_SRC`, `ADV_SCHEME`, `MICRO_ENGINE`, `RAD_ENGINE`) — never a
long-lived branch. `main` is the one long-lived branch; work happens on
short-lived `feat/`, `fix/`, `docs/` branches merged by PR. A frozen version is
an annotated tag (e.g. one per published run set): reproduce = check out the
tag, `git submodule update --init`, apply the patch, run the preset.

## Layout

Source lives under `src/`, grouped by process, but the **module names stay
flat on purpose** — `coupling.py` does bare `import settling` / `radiation`, and
`src/_paths.py` puts each subdirectory on `sys.path` so those imports keep
resolving. `src/` is not a package; do not add `__init__.py` to a subdirectory
whose name collides with a module (`src/settling/` vs `settling.py`) without
converting every import at the same time. See `docs/REPO_LAYOUT.md`.

```
configs/                    the linkings: cesm_numerical.env (default), jcm_numerical.env,
                            jcm_online_numerical.env; README.md
src/_paths.py               import-path setup; REPO_ROOT / MODELS / INPUTS anchors
src/run_prod.py             the production launcher: applies a preset, verifies the
                            submodule pins, resolves and echoes every knob, execs
                            driver_fast.py (self-documenting header, --help)
src/driver_fast.py          THE production entry point. Imports coupling.py and swaps
                            in the advection and the micro engine: MICRO_ENGINE=fast
                            (default) = tomas_jax.fast.run_fast via tomas_fast.py;
                            full = coupling.run_microphysics_full (tomas-jax dev
                            make_step; needs a dev checkout in the tomas-jax slot)
src/coupling.py             the model: grid, IC/BC, budget, diagnostics, main loop
src/met/met_jcm.py          MET_SRC=jcm: the jcm-strat reader, which also supplies the grid
src/met/met_jcm_online.py   MET_SRC=jcm_online: a JcmMet subclass serving frames from a live
                            jcm-strat run; starts and talks to jcm_server.py
src/met/jcm_server.py       the live JCM, run in jcm-strat's own interpreter (Python 3.11,
                            JAX 0.10) as a subprocess; imports nothing from this repo and
                            asserts jcm_strat/jcm come from the jcm-strat slot
src/met/bc_clim.py          BC_SRC=clim: serves the climatology through open_var()
src/microphysics/tomas_fast.py  the tomas_jax.fast adapter (coupling state <-> FastState)
src/advection/sl_lr.py      semi-Lagrangian split transport, ADV_SCHEME=sl (THE DEFAULT, K=12)
src/advection/sl_core.py    the SL kernels: dinosaur's semi_lagrangian module (Google
                            NeuralGCM, Apache 2.0) vendored without its dependencies and
                            patched to accept rows AT the poles (the f09 grid)
src/advection/fct_lr.py     Lin-Rood flux-form advection at CFL sub-steps (ADV_SCHEME=lr);
                            its vertical sweep, faces, budgets and the ADV_VPOS limiter
                            are what sl and ffsl reuse; the budget reference
src/advection/ffsl_lr.py    fct_lr at large Courant number (ADV_SCHEME=ffsl), selectable
src/advection/fct_fast.py   PPM/Zalesak primitives that fct_lr imports; ADV_SCHEME=fast;
                            lat_edges (cell edges for every grid)
src/advection/fct_core.py   legacy sealed-face advection. NOT on the run path -- kept
                            solely for the bit-identical legacy check in
                            scripts/validation/test_conservation.py
src/settling/settling.py    gravitational settling; also the canonical wet-droplet
                            sizing (tang_density/wet_size), which radiation.py imports
src/radiation/radiation.py  RRTMGP + Mie optics (RAD_ENGINE=rrtmgp)
src/radiation/rad_cache.py  per-archive-hour cache of the reference-aerosol solve and the
                            clear-sky fields (RAD_REF_CACHE; bit-identical on a hit);
                            rad_cache_build.py fills entries
scripts/met/                make_bc_climatology.py (archive -> bc_clim.npz),
                            regrid_clim_to_grid.py (bc_clim.npz -> another met grid)
scripts/radiation/          build_ref_cache.py (batch-fill the reference cache),
                            test_rad_cache.py, plot_solar_artifact.py
scripts/utils/              plot_run.py (the four post-run figures), gif_run.py,
                            run_summary.py (per-stage cost from a log)
scripts/validation/         harnesses -- see below
docs/                       CONFIGURATION (every env var), VALIDATION (the harnesses),
                            PROCESSES, RADIATION, COUPLING_VARIABLES, BOUNDARY_CONDITIONS,
                            REPO_LAYOUT, TRANSPORT_SL_HANDOFF, RADIATION_ARTIFACT_EXPLORATION
.github/workflows/ci.yml    CI: preset check, conservation, settling physics, an import
                            check and a ruff errors-only gate
models/tomas-jax            SUBMODULE, pinned 787c991 (branch gpu-fast). Sectional microphysics
models/jax-rrtmgp           SUBMODULE, pinned 99d2d71. Radiative transfer (needs patches/)
models/jcm-strat            SUBMODULE, pinned f3c2070 (branch dev), with its own nested
                            submodule external/jax-gcm at 35dc199 (the `jcm` package).
                            The live JCM for MET_SRC=jcm_online only
patches/                    jax-rrtmgp-zenith.patch, the one carried dependency fix
inputs/                     static input data (rad_data/: Palmer & Williams 1975 H2SO4
                            indices, as distributed in HITRAN Aerosols-2016) + provenance
                            of the external CESM archive; inputs/cache/ is gitignored
runs/                       raw model output, one directory per OUT_TAG. Gitignored
outputs/                    derived figures and tables. Gitignored
```

**Analysis scripts are flat and read `coupled_*_<TAG>.npz` from the current
working directory**, so run them from wherever the outputs are. Outputs land in
whatever directory you launch from, and `.gitignore` excludes
`*.npz/*.png/*.gif/*.log` — which hides a misplaced run rather than preventing
it, so `run_prod.py` refuses to start anywhere inside the checkout except under
`runs/` (`runs/*` is gitignored; a directory outside the checkout also works).

## Setup from a fresh clone

```bash
git clone --recurse-submodules https://github.com/reflective-org/AIDE-SAI.git
cd AIDE-SAI
git -C models/jax-rrtmgp apply ../../patches/jax-rrtmgp-zenith.patch   # required for RAD=1
pip install -r requirements.txt
pip install --upgrade "jax[cuda12]>=0.6.2"     # GPU wheel; the CPU one is unusable for production
mkdir -p runs/smoke && cd runs/smoke
../../src/run_prod.py --dry-run                 # resolves the cesm_numerical preset and checks the pins
../../src/run_prod.py --n-hours 6 --out-tag smoke   # smoke test (needs the CESM archive + CLIM_FILE)
```

**Neither tomas-jax nor jax-rrtmgp is usable as a plain checkout of its default
branch**, and `run_prod.py` refuses to start until both match the preset. (The
third submodule, `models/jcm-strat`, is needed only by `jcm_online_numerical`;
`--recurse-submodules` also fetches its nested jax-gcm. To skip it:
`git submodule update --init models/tomas-jax models/jax-rrtmgp`.)

| repo | required state | why | symptom if skipped |
|---|---|---|---|
| `models/tomas-jax` | `787c991`, branch `gpu-fast` | the `tomas_jax.fast` reduced model is the microphysics of both linkings (user decision 2026-11-18). `dev`'s `make_step` is built lazily, so `MICRO_ENGINE=full` fails only when used; `settling.h2so4_equilibrium_wt` falls back to `tomas_jax.fast.water` for the Tabazadeh fit | `cannot import tomas_jax.fast` from tomas_fast.py (and the pin check refuses first) |
| `models/tomas-jax-dev` | `0ba8f4b`, branch `adaptive-coag` (tomas-jax PR #49: dev `835ee24` + 8 commits) | `cesm_adaptive_substep` only: `make_step(coag_solver='heun_adaptive', gas_substep='adaptive', cap_warnings=...)` and `TOMAS_SPECIES` exist from PR #49 on; it has no `tomas_jax.fast` | the pin check refuses that preset; the other presets do not need it |
| `models/jax-rrtmgp` | `99d2d71` + `patches/jax-rrtmgp-zenith.patch` | `radiation.py` passes zenith per column, `(nlat, nlon, 1)`; upstream `sw_cell_source` assumes a scalar and broadcasts it against the `(nlat, nlon)` TOA flux | shape blow-up in the SW solve on the first `RAD=1` step |

The patch is one hunk against `99d2d71` and is the only carried modification to
either dependency; keep it that way, and if upstream takes the fix, delete the
patch rather than let the two drift. `git apply` refuses a partial application,
so a failure means the submodule was moved forward and the patch needs
regenerating against the new commit. Because a gitlink records only a commit,
the patch CANNOT live as a submodule working-tree edit — that is exactly where
it was found before the submodule conversion, uncommitted and reproducible only
by whoever had applied it — which is why the preset names it (`patch=`) and the
launcher checks it.

| dependency | not shipped because | how it is found |
|---|---|---|
| `tomas-jax` | separate repo (microphysics) | the preset's slot → `$TOMAS_JAX_PATH`; without a preset, `$TOMAS_JAX_PATH`, else `models/tomas-jax`, else the normal import path |
| `jax-rrtmgp` | separate repo (radiation) | `$RRTMGP_PATH`, same order. `RAD=0` skips radiation and this dependency |
| CESM `h1` hourly meteorology + MAM4 | ~23 TB as archived | **On this machine (the shared H100 box), nothing to set** — the `coupling.py` defaults resolve to the FWHIST archive under `/data`, world-readable. Elsewhere: `$CESM_DIR` (+ `CESM_PREFIX`, `CESM_SUF`). Layout: `$CESM_DIR/hour_1/$CESM_PREFIX.h1.<VAR>$CESM_SUF`. Variables: `docs/COUPLING_VARIABLES.md`. Under the `cesm_numerical` preset only U/V/OMEGA/T are read hourly (one hour × band levels at a time, ~5.3 MB per variable-hour); everything else comes from `CLIM_FILE` |
| boundary climatology (`CLIM_FILE`) | build product, GBs | `inputs/bc_clim/` in the checkout (gitignored; the preset's relative path resolves against the repo root); build with `scripts/met/make_bc_climatology.py` / `regrid_clim_to_grid.py` |
| jcm-strat meteorology (`jcm_numerical`; `jcm_online_numerical` reads its 1 January checkpoints and template config) | separate model's output | `JCM_DIR` (default `/data/JCM_stripped/jcm-strat/runs/p31/p31_AL7h70`, this machine; read-only), `JCM_TEMPLATE` |
| jcm-strat code (`jcm_online_numerical` only) | separate repo (dycore) | the preset's `jcm-strat` slot → `$JCM_STRAT_PATH` (default `models/jcm-strat`), recursively initialised. Never put on this process's `sys.path`: only the server's `PYTHONPATH` |
| jcm-strat interpreter and caches (`jcm_online_numerical` only) | machine state, not in any checkout | `JCM_PY` (a Python 3.11 venv with jcm-strat's dependencies; `models/jcm-strat/scripts/bootstrap_env.sh` builds one), `JCM_CACHE` (`era5/` nudging windows, `huggingface/` boundary files), `JCM_SCRATCH` (JAX compile cache). The preset sets this machine's `/data/JCM_stripped/jcm-strat/...` paths |
| CUDA driver library | site-specific | `run_prod.py` puts `$CUDA_DRIVER_LIB` on `LD_LIBRARY_PATH`. Where libcuda.so.1 is not on the loader path JAX **silently falls back to CPU**; set the variable, or empty it for a normal CUDA install |

An env var that is set but points nowhere is an error, not a silent fallback; a
missing CESM file names the variable that built the path.

Do not activate the tomas-jax `.venv` — it is CPU-only jaxlib with no xarray. The
working combo is **system python3** with GPU jax.

## Running

**The preset sets the linking and the launcher sets the run control; the
SCENARIO is yours.** `INJ_SO2_TG_YR` defaults to **0 = no injection**, so a
forgotten flag gives an obviously unforced baseline rather than a silent SAI
result. State the amount and give the scenario its own tag.

**Launch from a run directory, never the repo root.** Every output path is
relative to the working directory, so that directory is where the run lands.
`run_prod.py` **refuses to start** inside the checkout except under `runs/` —
the mistake is otherwise invisible, since `.npz`/`.png` are gitignored and a
misplaced run looks entirely normal while filling the tree (29 GB accumulated
that way before 2026-08-12). The script still execs `driver_fast.py` from its
own tree, so it runs the code you edited regardless of where you launched it.

```bash
REPO=/path/to/AIDE-SAI
mkdir -p $REPO/runs/inj6_30NS && cd $_
$REPO/src/run_prod.py --preset cesm_numerical --gpu 0 --out-tag inj6_30NS --n-hours 8760 \
    --inj-so2 6 --inj-hpa 55 --inj-lat 30 --inj-mirror 1 --inj-zonal 0   # the 2-year runs' scenario, 1 yr
$REPO/src/run_prod.py --preset jcm_numerical --out-tag ctrl_jcm         # NO-INJECTION control on the JCM linking
$REPO/src/run_prod.py --preset cesm_numerical --out-tag inj6_30NS --n-hours 17520 --resume 1 \
    --inj-so2 6 --inj-hpa 55 --inj-lat 30 --inj-mirror 1 --inj-zonal 0   # continue it to 2 years
```

Two checkouts launched from the *same* run directory with the same `OUT_TAG`
overwrite each other. `OUT_TAG` discipline is what keeps runs apart.

**Narrowing a run to one process.** Each coupling step walks a fixed stage list
(`docs/PROCESSES.md` §0), and each stage has an env flag that switches it off, so
a run can be reduced to the process being measured. Transport only:

```bash
MICRO=off RAD=0 SETTLE=0 STATE_CKPT=0 FRAME_EVERY=48 N_HOURS=48 \
  OUT_TAG=speed-benchmark $REPO/src/run_prod.py          # 8 steps
```

`MICRO=off` (2026-08-27) disables microphysics outright — a benchmark/diagnostic
mode, **not** a physics configuration: the aerosol is advected but never evolves.
It goes through the production step loop rather than a separate harness, so what
is measured is the advection the model actually runs. `MICRO` is validated at
import (`full|coag|off`); it previously fell through to the legacy `coag` path on
a typo. `driver_fast.py` accepts `full` and `off` only — pairing a swapped-in
engine with the pre-SAI `coag` model would silently be neither.

Set `FRAME_EVERY` to the run **length**, not 0: `FRAME_EVERY_STEPS` is
`max(1, FRAME_EVERY/STEP_HOURS)`, so 0 means a frame every step. `STATE_CKPT=0`
suppresses the restart cycle; `coupled_final_<TAG>.npz` is still written
unconditionally and is the bulk of the output.

**Cost.** ~22–35 s per 6-h step on one H100 with the perf switches off, flat
over a year (one year = 1460 steps ≈ 9–14 h). With their defaults (on since
2026-11-23, § Code state) the 48 h tests measured ~18 s/step (CESM, ≈ 7 h/yr)
and ~12 s/step (JCM, ≈ 5 h/yr); a full year on the new defaults has not been
timed yet. A complete 1-year run measured on 2026-08-28
(`runs/prod1yr/`, then `lr` transport and the pre-cache radiation) was
**34.9 s/step** — micro 16.5 s, radiation 9.4 s, advection 7.4 s, rest 1.7 s —
with quarter means 35.8/34.1/35.5/34.2 s: burden growth does not make steps more
expensive. Since then the radiation reference cache (default since 2026-10-13)
cut the radiation stage to one solve (3.1 s/step on a quiet GPU under
`RAD_SOLAR=instant`; `window_gq@3` adds ~+16 % per model step) and `sl`
transport is ~0.9 s/step against `lr`'s 6–8 s at 82 tracers. The `lr` transport
benchmark (production f09 grid, 1,327,104 cells, `ADV_F32=1`, `ADV_CFL=0.5`,
`ADV_VPOS=1`): 7.2 s/step in steady state, ~168 CFL substeps → 43 ms/substep;
per-step cost tracks `nsub`, so quote the per-substep figure. The 340–370 s/step
and "microphysics ~94%" figures carried until 2026-08-28 described the
pre-2026-08-13 tomas-jax engine and are superseded.

`run_prod.py` pins a **single** GPU (`CUDA_VISIBLE_DEVICES=${GPU:-0}`, `--gpu`)
rather than auto-selecting one, so the card a long run lands on is never a
function of what else happened to be idle at launch. Microphysics shards across
all *visible* devices, so exporting more than one card is what makes it
multi-GPU.

**The RESUME env must match the run you are resuming.** `INJ_*` mismatches are
refused outright (the scenario is stamped in the state ckpt), and so is a
`GRID_CFG` mismatch (a state on one grid is not a state on another);
`PHYS_CFG` mismatches — `WET_SETTLING` / `WET_OPTICS` / `SETTLE` / `ADV_VPOS`,
`MET_SRC`, `RAD_SOLAR` and the rest — only WARN, because they change physics
mid-trajectory rather than invalidating the state: a seam, not a corruption.
Both arrays are append-only, so an older checkpoint is never locked out by a new
field. Repeat the preset and scenario flags on the resume command line; they are
not read back from the checkpoint.

Checkpoints are written **atomically** (temp file + `os.replace`) and in a fixed
order — frames, timeseries, then state LAST — so the physics state is never newer
than the diagnostics. A resume trims frames that ran past the state, warns about a
frames history that stops short of it (the filmstrip would splice across the hole
without showing it), and tolerates an unreadable timeseries ckpt by continuing with
an empty plotted history: every cumulative counter the run needs comes from the
state ckpt, not from there.

The full effective config is printed in the run's own header (the launcher's
resolved knobs with their sources and the verified slots, then coupling.py's
own header), so any log is self-describing.

## Validation harnesses

Two kinds, and the difference matters — `docs/VALIDATION.md` has the full
inventory. **Automated tests** are self-contained (fixed seed, no GPU, no CESM,
no submodules, except where noted) and exit non-zero on failure; the first
three run in CI on every push and PR:

```bash
# the linking presets: parse, pins == gitlinks, patches exist (CI)
python3 $REPO/scripts/validation/test_configs.py

# advection conservation (the LR benchmark) (CI)
python3 $REPO/scripts/validation/test_conservation.py

# closed-form settling physics: Tang density, wet growth factor, fall speed,
# and that the implicit sweep changes the burden only by its bottom outflow (CI)
python3 $REPO/scripts/validation/test_physics_math.py

# semi-Lagrangian transport (ADV_SCHEME=sl): pole halo, constant field,
# Williamson case 1 over the poles, mass-fixer closure, bin consistency, driver
python3 $REPO/scripts/validation/test_sl_transport.py

# lr at large Courant number (ADV_SCHEME=ffsl): sweep == fct_lr at |c| < 1,
# conservation/uniform field at |c| up to 8, exact whole-cell shifts, the 3-D
# driver's budget closure, Williamson 1 at hourly sub-steps vs lr (CPU, ~1 min)
JAX_PLATFORMS=cpu python3 $REPO/scripts/validation/test_ffsl_transport.py

# window-integrated solar geometry: interval insolation vs hourly. CPU, no
# archive, ~1 min, but imports radiation.py, so it needs the submodules
python3 $REPO/scripts/validation/test_solar_window.py
```

**Investigations** load a real run or the archive and print diagnostics for a
human to read; there is no pass/fail. Run them from the directory holding the
output, since that is where they look for a checkpoint and where they write any
figure:

```bash
# the ADV_VPOS positivity limiter: positivity, conservation, smooth-field inactivity
python3 $REPO/scripts/validation/validate_vpos_f32.py

# transport against CESM's own transported CH4 over up to a year (GPU + archive)
CUDA_VISIBLE_DEVICES=<free> python3 $REPO/scripts/validation/ch4_truth_test.py --scheme sl --k 12 --days 365 --tag <tag>

# radiation on one real CESM hour; the radiation diagnostics vs CESM (GPU + archive)
CUDA_VISIBLE_DEVICES=<free> python3 $REPO/scripts/validation/test_radiation.py
CUDA_VISIBLE_DEVICES=<free> python3 $REPO/scripts/validation/validate_radiation.py

# radiation reference cache: a served hour is bit-identical to a computed one, the
# SW/LW-split solve equals the default one; GPU + archive (docs/RADIATION.md)
CUDA_VISIBLE_DEVICES=<free> python3 $REPO/scripts/radiation/test_rad_cache.py

# MET_SRC=jcm_online: does the live JCM stay on the AL7h70 track? Band zonal-mean
# U and T, online vs stored, next to the stored year-to-year spread (GPU; the
# settings come from --preset jcm_online_numerical). Statistical by design:
# jcm-strat on GPU is not deterministic run to run (§ Code state)
CUDA_VISIBLE_DEVICES=<free> python3 $REPO/scripts/validation/validate_jcm_online.py --days 60

# RAD_SOLAR against hourly radiation from a saved state (window alignment, the
# window_gq shortwave/longwave split, error of k-day means; docs/RADIATION_ARTIFACT_EXPLORATION.md)
CUDA_VISIBLE_DEVICES=<free> python3 $REPO/scripts/validation/check_solar_align.py --state S.npz --start 1997-01-01 --days 3 --out X
#   also check_solar_window.py, check_solar_gq_sw.py, check_solar_gq_lw.py, check_solar_accum.py
```

The other harnesses of the operator studies (`floor_anatomy.py`,
`compare_hz_operators.py`, `compare_3d_operators.py`, `bench_sl_transport.py`,
`bench_radiation_split.py`, `compare_runs.py`) are not part of this repo.

**A change to advection confirmed only in float64 is not confirmed.** Two bugs
in the positivity limiter were invisible in f64 and fatal in f32: a `1e-300`
guard that underflows to zero below float32's smallest normal (~1.2e-38),
producing inf/NaN; and a reformulation that cost 2.9e-5 relative accuracy on
smooth fields where f64 showed only 1.2e-13.

In practice you set nothing: `validate_vpos_f32.py` already defaults to
`ADV_CFL=0.5` / `ADV_F32=1`, and `coupling.py` and `driver_fast.py` pin the same
values explicitly, so an `ADV_F32=1 ADV_CFL=0.5` prefix is a no-op. The rule
binds in exactly one place: calling `fct_lr.advect_step_batch` **directly**,
e.g. from a new test. Its signature defaults (`cfl=0.2`, `float64`) are a
configuration no run uses, so a bare call passes for reasons that do not
transfer. `dt_total` has no default at all — it is the coupling step
(`STEP_SEC`), and the 1 h default it used to carry made a caller who omitted it
advect a sixth of the interval, silently. See `docs/VALIDATION.md`.

**Run the harnesses by absolute path from wherever the output is, not from the
repo.** Python puts the *script's* directory on `sys.path`, never the cwd, so
each one inserts the repo root and `src/` subdirectories itself and works from
any directory. What they take from the cwd is the checkpoint they read
(`validate_vpos_f32.py`; override with `STATE=<path>`) and where
`validate_radiation.py` drops its figure.

## Code state — why the defaults are what they are

- **Performance switches (2026-11-23; default ON since 2026-11-23, were
  opt-in at 0).** `RAD_FAST`, `MICRO_WRAP_DEVICE`, `MAM4_BIN_DEVICE`,
  `ADV_SL_DEVICE_RESAMPLE`, `CKPT_ASYNC`. Each is read once at import with
  `os.environ.get`, echoed in the run header and listed in `run_prod.py`'s
  KNOBS; no preset sets any of them. **Why on:** in 48 h injected runs on
  both presets (CESM and JCM numerical, `RAD_REF_CACHE=0`) every switch, and
  all five together, stayed within the unmodified code's own run-to-run GPU
  noise (<= 2.5e-12 pointwise; budget closure lines identical), for a step
  wall of 30 -> 18 s (CESM) and 20 -> 12 s (JCM). Test record:
  `runs/perf_speedups/README.md` on branch `perf/numerical-speedups`
  (baselines `base_cesm`/`base_jcm`, all-on `combined_cesm`/`combined_jcm`).
  **To reproduce the pre-switch bit pattern** (the 2-year preset runs, any run
  before 2026-11-23) set all five to 0:
  `RAD_FAST=0 MICRO_WRAP_DEVICE=0 MAM4_BIN_DEVICE=0 ADV_SL_DEVICE_RESAMPLE=0 CKPT_ASYNC=0`
  (and `ADV_SL_RHONORM=0` for any run made before PR #4 reached dev on 2026-11-24, which includes the perf validation runs above; § sl_lr level air mass).
  Each `=0` is exactly the old code path (the off-path code was not changed).
  Validated only over 48 h; the year-scale behaviour of the defaults is not
  yet tested.
  - `RAD_FAST` (default **auto**) — radiation fast path: one latitude chunk
    (192 rows, a module constant -- there is no `RAD_LAT_CHUNK_FAST` variable;
    an explicit `RAD_LAT_CHUNK` sets both chunk sizes), SW on
    sunlit columns only, g-point batching (`rrtmgp_batched.py`), exact
    dynamic minor-gas scan, no cloud optics, window_gq through
    `_solve_fast_gq`. Bit-identical radiation on both presets; radiation
    12.2 -> 6.0 s/step (CESM), 7.9 -> 4.0 (JCM), compile step 117 -> 52 s /
    243 -> 74 s; ~2 GB more device memory. Validated only for one GPU, wet
    optics, `RAD_SOLAR=window_gq` and the normal (non-split) solve, so the
    default is not a plain 1 but **auto**, resolved per `RadiationDriver` at
    construction (device count, optics mode, solar mode, sub-knobs -- none of
    these is known at import): on where validated, otherwise off, and the
    header says `RAD_FAST=auto -> 0|1 (reason)`; under `RAD_MODE=flux` (every
    call split=True) it resolves 0. On an auto-on driver every
    call the fast path does not cover (split=True: RAD_MODE=flux, the
    reference-cache builder, the check harnesses; a one-solve solar mode set
    after construction; `_gq_full`) runs the chunk loop, i.e. the RAD_FAST=0
    path, and says so once per kind of call. Nothing is installed at import
    any more: the `rrtmgp_batched` import and its process-wide
    `jax.lax.scan` / gas-optics minor-gas patches are installed by the first
    driver that resolves on, and the cloud skip is applied to a shallow copy
    of the optics library, so `self.rt` (the chunk loop's) is never patched.
    The patches are pass-throughs outside `rrtmgp_batched`'s own context, so
    an off call in a process that has them still runs the library's graph.
    `RAD_FAST=1` forces it and still **refuses** (SystemExit) anything
    outside the validated path; `RAD_FAST=0` is the chunk loop. Its
    reference-cache key gains `rad_fast=1` where it would resolve on apart from
    the device count (so a one-GPU builder and a multi-GPU run still share a
    directory; the content is bit-identical either way) -- which means a default run does NOT hit a reference
    cache built before 2026-11-23 (it computes and stores on a miss; rebuild
    with `scripts/radiation/build_ref_cache.py`, which now resolves the same
    way), and a `RAD_FAST=0` run still hits the old one.
  - `MICRO_WRAP_DEVICE` (default 1) — the tomas_jax.fast adapter (unit conversion,
    transposes, clip, OH field) on the device. micro ~2.7 s/step faster.
    **Not bit-identical**: clip-sum diagnostics differ at ~1e-15 and a few
    dozen final cells at <= 2.5e-12 rel. Not used under FAST_MULTIGPU on >1
    device (the header says `MICRO_WRAP_DEVICE=1 -> 0 (...)`).
  - `MAM4_BIN_DEVICE` (default 1) — bin_mode's erf part (INIT_BIN=so4) as eager jax ops.
    <= 9e-14 rel in a few final cells. Saves ~1.4-2.4 s/step only with
    `RAD_REF_CACHE=0`; with a warm reference cache (the default) only
    ~0.1-0.3 s/step (bc+polar). Not in the cache key (bits match).
  - `ADV_SL_DEVICE_RESAMPLE` (default 1) — sl_lr's float32 wind time resample on the
    device (separate jits, no FMA). Bit-identical on vs off; advect
    0.9 -> 0.45 s/step (CESM), 0.6 -> 0.3 (JCM). Prints ENGAGED/not engaged
    on its first call.
  - `CKPT_ASYNC` (default 1) — `_ckpt` files snapshotted to host and written by a
    background thread (joined before the next checkpoint and at the end).
    Identical files; takes 1.6-2.6 s per checkpoint off the main thread,
    more late in a year. Costs one extra host copy of the state plus the
    frames stack in RAM while a write is in flight (untested over a year).
  - `RAD_LAT_CHUNK=192` (existing knob, CESM only): bit-identical radiation,
    gain within timing noise; subsumed by `RAD_FAST`.
  - All five together (48 h): CESM step wall 30 -> 18 s, JCM 20 -> 12 s; max
    rel diff vs base 8.5e-14 (CESM final mas), all timeseries/frames identical
    except the clip sums (<= 1e-15), budget closure unchanged.
  - Known, not caused by any switch: the final num/mas of a CESM run differ
    run to run at <= 2.5e-12 in a few columns (GPU nondeterminism in the
    microphysics), and a RESUME does not reproduce an uninterrupted run
    (observed in the perf tests, `ckr*_` runs: Nburden ~3e-4 rel after the
    first resumed step, with or without CKPT_ASYNC; likely some carried
    advection state is not in the checkpoint) -- not yet investigated.

- **The live JCM linking (2026-11-20; `configs/jcm_online_numerical.env`,
  `MET_SRC=jcm_online`; the code dates from 2026-11-18).** `MET_SRC=jcm_online`
  serves exactly what `MET_SRC=jcm` serves, on the same grid, but from jcm-strat
  **integrating alongside** the aerosol model instead of from the AL7h70 chain's
  stored output, with every term of that configuration active: Jucker dry
  relaxation + the AL7h T_e correction, QBO nudging to ERA5, ERA5 nudging of
  u/v/T below 400 hPa (tau 6 h), Hines and Lott-Miller drag, the sponge, the
  production tracers, the omega diagnostic; T63 L95, 12-min step. Still one-way
  (the circulation never sees the aerosol); the server is where a heating
  feedback would go through. Nothing in it is read unless `MET_SRC=jcm_online`:
  the other presets are unchanged bit for bit (their `PHYS_CFG` gains one
  trailing 0).
  - **Why a subprocess.** jcm-strat needs its own interpreter (Python 3.11,
    JAX 0.10.2, jcm, dinosaur) and this repo runs on JAX 0.6.2 / Python 3.10:
    they cannot share a process. `met_jcm_online.JcmOnlineMet` (a
    `met_jcm.JcmMet` subclass, so grid, level flip, calendar mapping and `MET_Q`
    are shared) starts `src/met/jcm_server.py` under `JCM_PY` on the same GPU and
    reads each 6-hourly frame (five L95 fields, 28 MB) over a pipe. The server
    imports nothing from this repo; its log goes to stderr, i.e. into the run log.
  - **Where jcm-strat comes from.** A new model slot, `jcm-strat`
    (`JCM_STRAT_PATH`): the submodule `models/jcm-strat` at f3c2070 (branch
    `dev`), whose own submodule `external/jax-gcm` (35dc199) is the `jcm`
    package. The server's `PYTHONPATH` is exactly those two directories, so
    `jcm_strat` and `jcm` come from the pinned checkout and not from the venv's
    editable installs (which point at whatever checkout the venv was built
    from); the server prints both `__file__`s and refuses to start if either
    resolves outside the slot (`jcm_server.py --check-imports` runs only that
    check). Only presets that select `MET_SRC=jcm_online` pin the slot
    (`SLOT_NEEDED_IF`), so the other linkings need neither the checkout nor its
    ~0.6 GB of history. What a checkout cannot hold is machine state and has
    its own knobs, set by the preset to this machine's paths: `JCM_PY` (the
    interpreter: a venv with jcm-strat's dependencies), `JCM_CACHE` (`era5/` =
    the regridded ERA5 nudging windows, `huggingface/` = jcm boundary files) and
    `JCM_SCRATCH` (JAX compile cache). The AL7h T_e correction is a tracked
    file of the slot (`JCM_TE_CORRECTION` defaults to it; the chain's config
    names a jcm-strat worktree that no longer exists). The template config
    itself belongs to the chain, `JCM_DIR`, and so does one machine path inside
    it that no knob overrides: the QBO nudging reference
    (`qbo.era5_glob`, `/data/JCM_stripped/jcm-strat/cache/era5_ref/...` on this
    machine). On another machine, use a template whose `era5_glob` points at
    your copy (`JCM_TEMPLATE`).
  - **What it builds.** The chain's own resolved Hydra config (`JCM_TEMPLATE`,
    default the `<chain>_19960101` segment's `.hydra/config.yaml`) with only what
    the chain changed per segment: start time, run length, QBO year, `init.file`.
    Before building it installs the same policies `jcm_strat.main` does
    (mass-fixer exclusions, health policy, calendar, dot precision, level table).
    Like the chain, it builds one model per calendar year and, at each year end,
    writes a jcm checkpoint (`JCM_WORK/jcm_<YYYY>0101.ckpt`, default
    `jcm_online/` in the run directory) and starts the next year from it.
  - **Start and RESUME.** The server starts lazily, on the first frame read, in
    the calendar year of that hour, from this run's own `jcm_<YYYY>0101.ckpt` if
    present (so a RESUME continues the online trajectory), else the chain's
    `<YYYY-1>0101/checkpoint.ckpt`. An hour inside the year is reached by
    integrating forward from 1 January (~3 s per simulated day). The frame AT the
    init instant cannot be produced without stepping; it is the donor's last
    output frame (the same state), stored beside our own checkpoints as
    `.frame0.npz`.
  - **Memory: the ERA5 nudging target stays on the host.** jcm keeps the year's
    ERA5 nudging target (±1 day, 29 GiB) on the device, and its compiled step
    asks ~30 GiB more; next to the aerosol model on one card that ran out of
    memory. The server loads the target onto the CPU device and hands each
    6-hour resume a fixed 4-sample slice starting one sample before it (the same
    BY_DATE 'pad' lookups, one compile). The server then holds a few GiB on the
    card; a 6-hour JCM step costs ~0.6 s after compilation.
  - **Not bit-reproducible, and cannot be.** jcm-strat on GPU is not
    deterministic run to run: two identical stock `python -m jcm_strat.main`
    runs from the 1995 checkpoint differ by 2.8 m/s in u after 6 h and 70 m/s
    after 24 h (single upper-level points, NH high latitudes; RMS 0.4 m/s), and
    the stored chain was made with another dinosaur build besides. So this
    linking is validated **statistically, never bitwise**, and a rerun of the
    preset does not reproduce its provenance run bit for bit.
    `scripts/validation/validate_jcm_online.py` (60 days from 1996-01-01, zonal
    means on the 47 band levels 1–138 hPa) gives RMS online − stored of
    0.03 m/s / 0.00 K at day 1, 0.7 m/s / 0.33 K at day 10, 2.1 m/s / 1.2 K at
    day 60 (peak 7.1 m/s / 3.4 K at day 40), against 8–14 m/s / 2.7–5.7 K
    between the stored 1996 and 1997 on the same dates: on average 0.21x (U) and
    0.34x (T) the year-to-year spread. ERA5 and QBO nudging keep it on the
    AL7h70 track; the free polar vortex drifts most. A full year (1996, 18 min
    of JCM on the shared H100) ends 3.6 m/s from the stored chain in band
    zonal-mean U; the 1996 → 1997 rollover (own checkpoint, rebuild) continues
    smoothly (3.8 → 4.1 m/s), but the rebuild took 356 s against 58 s for the
    first build: ~5 % of a year's wall time, not yet understood.
  - **Stamps.** `GRID_CFG` met code 3; a RESUME between `jcm` (2) and
    `jcm_online` (3) is allowed (same grid). `PHYS_CFG` gains
    `MET_SRC=jcm_online` (appended; warns on a switch).
  - **Provenance.** The 1-year run `jcm_online_1yr_inj6_30NS` (2026-11-19; the
    2-year JCM run's scenario: 6 Tg SO2/yr, 55 hPa, 30N/30S mirrored, 180E,
    `--frame-levels all`) ran with these knob values, jcm-strat f3c2070 (jax-gcm
    35dc199) and the same tomas-jax / jax-rrtmgp pins. A dry run of the preset
    with that scenario resolves every value its log header recorded; the only
    differences are the absolute `CLIM_FILE` path (same file, md5-identical),
    the tag, and the slot and machine-path variables the launcher now echoes
    (`JCM_STRAT_PATH`, `JCM_PY`, `JCM_CACHE`, `JCM_SCRATCH`, `TOMAS_JAX_PATH`,
    `RRTMGP_PATH`), which resolve to the interpreter, caches and commits that run
    used. Its server was started with the same template, init checkpoint, frame0
    and a byte-identical T_e correction file.

- **AIDE-SAI starts (2026-11-20).** The tree runs both 2-year linkings, and
  the presets reproduce the 2-year CESM and JCM runs bit-identically (the JCM
  reader leaves the CESM physics path unchanged). The values that used to be
  hard-coded production presets in `run_prod.py` and differed between
  linkings — `MET_SRC`, `BC_SRC`, `CLIM_FILE`, `INIT_BIN`, `ADV_VPOS` — moved
  to `configs/`, and the submodule pins with them (§ Linkings are presets).
  The boundary climatologies live in the checkout (`inputs/bc_clim/`,
  gitignored), and `run_prod.py` resolves a relative PATH-type knob in a
  preset against the repo root (`PATH_KNOBS`). The radiation
  reference-cache key now reads the commit of the rrtmgp checkout actually in
  use (`RRTMGP_PATH`), not always `models/jax-rrtmgp`, so a preset with a
  different rrtmgp slot cannot be served another's cache. Selecting
  `MICRO_ENGINE=emulator`, `MICRO_REF`, `MICRO_SAMPLE_EVERY`,
  `RAD_ENGINE=emulator` or `RAD_REF` exits saying it is not part of this repo; their
  `PHYS_CFG` stamps are kept (append-only), and `RAD_SAMPLE_EVERY` still writes
  radiation training samples.

- **Microphysics: `tomas_jax.fast`, `MICRO_ENGINE=fast` (2026-11-18, user
  decision).** `models/tomas-jax` on gpu-fast `787c991`: the reduced
  sulfate + water model on the native 40 bins (`N_BINS` must be unset or 40),
  called through `src/microphysics/tomas_fast.py`. `driver_fast.py` sets
  `FAST_CELL_CAP=50000` (faster than the 250000 module default) and
  `FAST_SORT=1` (the stiffness sort is worth ~27 % of the micro stage).
  `MICRO_ENGINE=full` remains in code: tomas-jax `dev`'s `make_step`, which
  the full-numerical CESM runs used with `dev` `835ee24`. It
  needs a preset whose tomas-jax slot is a dev checkout; on the gpu-fast pin it
  fails when its step is first built. The `MICRO_*` knobs below are its
  defaults, each measured against dev's own setup (dt 60 s, 10 coag sub-steps)
  on real met:
  * `MICRO_DT=60` — dev's step. 120 s costs ~1e-2 in total N, 300 s ~4e-2.
  * `MICRO_NCOAG=1` (dev: 10) — within 4.7e-4 (max over cells) in total N,
    and the only dt-60 setting under the budget on one GPU.
  * `MICRO_COAG=fixed` — dev's strat setup; `euler_adaptive` is ~100x slower.
  * `MICRO_MNFIX_TRIM=1` — bit-identical to trim=0, 5x faster in the coag loop
    (13.2 vs 0.5 ms per MNFIX at 20k cells). Dev's docstring recommends
    trim=False under vmap; on this problem that is a 5.6x slowdown overall.
  * `MICRO_FION=0` — ions off, user decision. Dev's strat value 10 gives 6.7x
    N growth in one 6 h step vs 1.1x. Stamped append-only; legacy value 10.
  * Composition sulfate + water, gases SO2 + H2SO4: no `org_conc` / `nh3_conc`
    passed (dev defaults 0), nucleation `org_frac=0`, `rho_nuc=1770`. Dev
    hard-codes ICOMP=44, so 42 zero species slots ride along; a sulfate-only
    build (ICOMP=2) is round-off identical and 5.6x faster but needs a
    tomas-jax change (configurable composition).
  * `MICRO_MAX_WALL=300` — microphysics of one full-grid 6 h step <= 5 min; the
    run aborts (`MicroTooSlow`) past it. Measured: 239 s on one H200 (1,327,104
    cells), 44.5 s over six (`MICRO_GPUS=6`, one host thread per device).
  * `CELL_CHUNK=50000` — memory only; batch size does not change throughput.
  The Tabazadeh wt% that settling and the wet optics use is
  `settling.h2so4_equilibrium_wt`: dev's `tabazadeh_h2so4_wt` vmapped over
  cells when dev is the checkout, else gpu-fast's
  `tomas_jax.fast.water.h2so4_equilibrium_wt` — the same arithmetic (max diff
  0.0 wt% over 72k random T/RH), so settling, the optics and the microphysics'
  water uptake agree under either engine. `MICRO_ENGINE=full`, `MICRO_DT`,
  `MICRO_NCOAG`, `MICRO_COAG=adaptive` and `MICRO_FION` are in `PHYS_CFG`.

- **Adaptive-sub-stepping microphysics, preset `cesm_adaptive_substep`
  (2026-11-25, user request).** The `full` chain above with tomas-jax PR #49
  (`models/tomas-jax-dev`, `0ba8f4b`), configured as the PR's "Recommended
  settings" (tomas-jax `docs/gas_substep.md`) and GPU benchmarks
  (`docs/gpu_deployment.md`). New knobs, all default off, so every other
  preset and a bare `MICRO=full` run is unchanged:
  * `MICRO_COAG=heun` → `coag_solver='heun_adaptive'`: second-order, error
    controlled (rtol 3e-3, atol 1e-6 of total N, dev's defaults); the lowest
    L1 N error in the PR's comparison (0.016 vs 0.019 for euler_adaptive).
  * `MICRO_GAS_SUBSTEP=adaptive`, `MICRO_GAS_FRAC=0.1`,
    `MICRO_GAS_MAX_SUBSTEPS` (dev 16384; preset 65536, the PR's GPU value) →
    `gas_substep='adaptive'`. Without it the nucleation gas clamp fires at
    >= 1 ppmv SO2 and N(>3 nm) can come out up to 50x too high, i.e. in the
    plume. `split_chemistry` is not passed: with adaptive gas sub-steps it
    raises (the PR's snippet lists it; the adaptive path always splits).
  * `MICRO_DT=360` in the preset (60 dev steps per 6 h step, one OH sample
    each), the PR's GPU outer step; both the gas chain and the coagulation
    sub-step adaptively inside it.
  * `MICRO_CAP_WARNINGS=0`: dev's per-cell cap callbacks are unrolled per cell
    under vmap (8192 cells: compile 217 vs 9 s, step 5.3 vs 0.06 s). The caps
    are then read from the step's sub-step info (`return_substep_info=True`,
    only on the adaptive gas path, so `0` with an adaptive coag solver and no
    gas sub-steps is refused: a coag cap would truncate silently). Each step
    prints `[full] ... cap hits: gas N coag N cells`, the mean/max gas
    sub-steps and the max `k_nuc dt_sub`; with `MICRO_CAP_ABORT=1` (default) a
    hit aborts the run (`MicroCapHit`).
  * `TOMAS_SPECIES=sulfate`: tomas-jax's reduced SO4/NH4/H2O layout (3 Mk
    columns, not 44), round-off identical for this chain (tomas-jax
    `tests/test_sulfate_only.py`). Read by tomas-jax at its first import;
    `coupling.py` refuses a value the checkout does not run.
  * `MICRO_SORT=prod`, `CELL_CHUNK=4096`: cells ordered by SO2 production
    (k1 OH SO2, half-decade tiers hottest first, T within a tier) and run in
    4096-cell chunks. A vmapped adaptive step runs every cell of a chunk in
    lockstep with its slowest one; in grid order with 50k chunks one sunlit
    injection cell would hold a whole chunk (estimated 600-1000 s, over
    `MICRO_MAX_WALL`). Cost only: per-cell results do not depend on
    chunk-mates (max rel. difference 0 across orderings in the PR), so not
    stamped.
  * `MICRO_FION=0` stays (user decision 2026-11-18); the PR validated at 10.
  * Measured (48 h, 6 Tg SO2/yr mirrored single cells at ±30°, lon 180,
    55 hPa, one H200, 2026-11-25): microphysics 26-38 s per 6 h step, ~33 s
    per step in all, no cap hit, injection cell up to ~17,000 gas sub-steps
    per step (cap 65536), budget closure ~2e-16. Validated by tomas-jax on
    the AFGL tropical grid only.
  Stamped append-only in `PHYS_CFG`: `MICRO_COAG=heun`, `MICRO_GAS_FRAC`
  (-1 = no gas sub-steps), `TOMAS_SPECIES=sulfate`; a RESUME from an older
  checkpoint compares them against their implied values (0, -1, 0).
  Nucleation sensitivity can use `make_step`'s `fn_scale` step argument or
  `NucleationParams`, not yet exposed as knobs.

- **`MICRO_COAG_SCALE`, coagulation-kernel scale (2026-11-25, user request).**
  Default 1 (unchanged). A multiplier on the Brownian coagulation kernel in
  all three of `make_step`'s coag solvers, passed as its runtime step argument
  `coag_kernel_scale` (jit-traced, like `fion`; the name matches tomas-jax
  `39535ea`, PR #33). It exists for coagulation/nucleation sensitivity tests
  (e.g. x0.5, x2) on the adaptive-sub-stepping chain, as `FAST_COAG_SCALE`
  served the fast engine in the frozen AIDE-SAI-link-sens tree. At 1 the
  argument is not passed at all, so a default run is bit-identical and still
  builds on `0ba8f4b`, which lacks it. Otherwise it needs a tomas-jax checkout
  whose `make_step` accepts it (branch `adaptive-coag-kernel-scale`):
  `coupling.py` refuses at startup a checkout without it (dev only warns on an
  unknown step argument and ignores it), a negative value, and `MICRO` other
  than `full`; `driver_fast.py` refuses it under `MICRO_ENGINE=fast` (gpu-fast
  `787c991` has no kernel scale). Echoed in the run header (`coag kernel
  x<scale>`) and stamped append-only in `PHYS_CFG` (`MICRO_COAG_SCALE`; implied
  1 for older checkpoints, so a scaled RESUME warns of the seam).

- **Meteorology from one source, everything else from the climatology
  (2026-11-17; now in the presets).** Both presets set `BC_SRC=clim`: only
  U/V/OMEGA/T come from the met source, and every other field — Q, O3, CH4,
  the MAM4 aerosol reservoirs, the gases, the surface fluxes and the radiation
  column — from the monthly CESM climatology on the model grid (monthly
  1996–2014 means of f.e21.FWHIST.f09, linear in day of year).
  `bc_clim.ClimArchive` runs without a met file, and `coupling.main` routes per
  variable. `coupling.py`'s own default is still `BC_SRC=cesm` (the hourly
  archive for everything); that is what `--preset none` gives, and
  `MET_SRC=jcm` refuses it.

- **The JCM linking (2026-11-18; `configs/jcm_numerical.env`).** `MET_SRC=jcm` takes
  U/V/OMEGA/T from the jcm-strat AL7h70 chain (`JCM_DIR`, read-only),
  6-hourly, linear in time between step ends (the `sl` default), **and the grid
  with them**:
  - **Horizontal**: T63 Gaussian 96 x 192 (rows ±88.57, no pole rows). Cell edges
    come from `fct_fast.lat_edges` — midpoints, plus the poles as outer edges on a
    Gaussian grid (half cells at the pole rows on f09, bit-identical to before) —
    and are shared by `grid_metric`, the burden weight `W_LAT`, the regridder and
    `plot_run.py`. The polar caps (|lat| > 80) are 5 rows each (22 rows total on f09).
    `sl_lr` also keeps the two OUTERMOST rows out of the continuity omega: a
    Gaussian pole-touching cell has a zero-width pole face, so cross-pole flow has
    no face to leave by (a constant field went 4x wrong with `lat_freeze=90`); f09's
    outermost rows are its ±90 rows, already excluded. The caps contain them anyway.
  - **Vertical**: JCM's own 47 native levels with reference p 1.01–137.8 hPa
    (the same `P_LO_HPA=1`, `P_HI_HPA=150` band as on f09, which gives 24
    levels 1.2–143 hPa there). No vertical interpolation of the meteorology:
    above 78 hPa JCM's levels are pure pressure, and the 7 below have
    b ≤ 0.0074, the same reference-pressure convention as the CESM path. Twice
    CESM's resolution in the band; 866,304 cells against 1,327,104.
  - **Calendar**: the model hour stays CESM's noleap hour since 1996-01-01 00Z; it is
    served from the JCM frame with the same Y-M-D-H, so 29 February is skipped
    (one step interpolates across 30 h). JCM segment `<YYYY>0101` starts at 06Z,
    so the indexed years start one before H0's.
  - **Everything JCM does not provide** comes from the CESM climatology regridded
    onto the JCM grid by `scripts/met/regrid_clim_to_grid.py` (the preset's
    `CLIM_FILE`, `inputs/bc_clim/bc_clim_jcm.npz`; rebuild with the command in the script's docstring, ~20 s).
    `install_bc_source` refuses a climatology whose lat/lon/hyam/hybm differ
    from JCM's.

  | field | source | regrid |
  |---|---|---|
  | U, V, OMEGA, T (band and full radiation column) | **JCM** AL7h70, 6-hourly | none (native grid and levels) |
  | **Q** (RH, radiation) | **CESM clim** — JCM's `specific_humidity` is a passive constant 2.32e-3 kg/kg (jcm-strat is dry); `MET_Q=met` would take it | conservative + linear in ln p |
  | **O3, CH4** (radiation) | **CESM clim** | conservative + ln p |
  | **num_a1-3, so4_a1-3** (MAM4 IC, edge/cap reservoirs, radiation reference aerosol) | **CESM clim** | conservative + ln p |
  | **SO2, H2SO4** (gas IC/BC) | **CESM clim** | conservative + ln p |
  | **FLDS, FLNS, FSDS, FSNS** (RRTMGP surface T and albedo) | **CESM clim** | conservative |
  | **OH** (opened; unused under the default `OH_SZA=1`) | **CESM clim** | bilinear + ln p |
  | **RELHUM** (unused under `MET_RH=liq`) | **CESM clim** | bilinear + ln p |
  | **dgnumwet1-3** (only `INIT_BIN=dgnum`) | **CESM clim** | bilinear + ln p |
  | **PS, TS** (served, unused) | **CESM clim** | bilinear |
  | CO2, N2O | constants, as before | — |

  Conservative = first-order area overlap in sin(lat) × lon (global means kept
  to roundoff); ln p is constant beyond the end levels.
  - **Stamps**: `PHYS_CFG` gains `MET_SRC=jcm` and `MET_Q=met` (warn). `GRID_CFG`
    (met code, nlat, nlon, nlev, first/last lat, top/bottom hPa) is in every
    state checkpoint and is **refused** on a RESUME mismatch; a checkpoint
    without it predates the JCM work, is on f09, and is refused under `MET_SRC=jcm`.
  - **Known differences from the CESM linking**: JCM's upper-stratospheric T is
    ~10 K warmer than CESM's at 1 hPa; burdens are in grid units
    (`q·dp·W_LAT` sums), so compare them across grids only in kg (`plot_run.py`
    derives the unit from the run's own grid).

- **Transport: `ADV_SCHEME=sl`, `ADV_SL_K=12` (user decision 2026-11-18).**
  The semi-Lagrangian SPLIT operator (`ADV_SL_VERT=flux`): SL horizontal remap
  per level (K sub-steps, per-level mass fixer, Jacobian air mass + remainder
  sweep) plus `fct_lr`'s own flux-form vertical sweep on the open faces, Strang
  ordered; `lr`'s boundary physics, `B_vf` and residual unchanged. On the two
  bracketing 6-hourly snapshots (`ADV_SL_HOURLY=0`; hourly winds buy nothing:
  0.75x vs 0.76x) it has 0.76x `lr`'s year-mean RMSE against CESM's own
  transported CH4 (`ch4_truth_test.py`; `lr` itself is biased high at
  40–75 hPa), the most accurate scheme in the tree, at ~0.9 s vs 6–8 s per
  step at 82 tracers (advection-comparison study, this machine). Both drivers read
  `C.ADV_SCHEME`, so there is one default — they had drifted once (coupling
  `sl`, driver `ffsl`) and a bare run advected with `ffsl` while its header and
  stamp said `sl`. History: `lr` was the production scheme until 2026-10-15
  and is byte for byte unchanged and the budget reference; `ffsl` K=24 (`lr`
  at large Courant number) was the default 2026-10-15..11-17
  and is no longer recommended — its sliver measure was wrong at the cap edge
  until 2026-11-05 (`ADV_FFSL_FRAC`, now `exact`), and even fixed it is 3.0x
  `lr`'s CH4 error at K=24. The handoff's closed-column design
  (`ADV_SL_VERT=sl`) FAILED its A/B — at day 30 mass/SO2/AOD 9–15 % low,
  entirely from the Dirichlet edge-level reset being ~3× the flux-face exchange
  — and is kept only for the record. `docs/PROCESSES.md` §1.8–1.10.
  `PHYS_CFG` stamps `ADV_SCHEME=sl`, `=ffsl` and `=lr` (warn-only).

- **`ADV_SL_RHONORM=1`: sl_lr level air mass (default since PR #4, merged into dev 2026-11-24).**
  After each horizontal sub-step `sl_lr._horizontal_substep` rescales the
  carried air mass so every level keeps its total, `sum(ac rho_out) =
  sum(ac rho_in)`. A horizontal step cannot change a level's air mass, but the
  rho_out it built did: the Eulerian polar-cap rows lose ~1.2e-4 of the 1.24 hPa
  level per sub-step and the Jacobian rows gain 1–2.5e-5 on every level. The
  per-level mass fixer read that as a tracer error and rescaled the whole level
  every sub-step, so a uniform q = 1 on CESM 1996 winds drifted **+2 % at
  1.24 hPa in 10 days** while the budget still closed (the fixers force the
  closure, so the budget line cannot see it). With the fixer off the same test
  stays within 1e-5, so the transport itself is near-conservative; the drift
  was imprinted by the fixer. It does not shrink with K (K=24 drifts the same)
  or with the cap reset; removing the caps moves it to −1 % everywhere. With
  the renormalisation the per-level fixer factor drops from 2.6e-4 to 3.8e-8
  per sub-step and the uniform field stays within 1e-4 after 10 days
  (`scripts/validation/sl_fixer_cap_diag.py`, `ADV_SL_FIXLOG=1`). The local
  mismatch that is left goes back down the column through `_remainder_sweep`.
  Year-long tracer truth tests (month 12 against CESM's own transported
  tracers, K=12): N2O RMSE 3.71e-9 → 3.15e-9 and bias −2.1e-9 → −1.3e-10
  (now better than `lr`); CH4 (monthly truth) RMSE 2.64e-8 → 3.34e-8 and bias
  +5.8e-9 → +1.7e-8, which is where K=24 lands with or without the switch —
  probably two compensating errors in the old K=12 CH4 score; open.
  **What the switch changes:** a bare run is not bit-identical to runs made before PR #4 reached dev
  (2026-11-24), including the 2-year CESM and JCM runs (set `ADV_SL_RHONORM=0`
  to reproduce them). PHYS_CFG stamps it (warn-only; checkpoints written before
  the stamp existed resume without a warning).

- **`MET_RH=liq`** (2026-11-17): RH is recomputed every step from Q, T_met and
  p, over liquid water (Tabazadeh 1997 eq. 1, `settling.rh_liquid_pct`). It is
  the same arithmetic the wet optics always used, so settling, the optics and
  the microphysics' water uptake agree. The old `q` (over ice) gave 1.6–1.9×
  the liquid RH in the band; `q` and `cesm` (archive RELHUM) remain selectable.
  Nucleation takes no RH. Stamped in `PHYS_CFG`.

- **`RAD_T_FEEDBACK=0`** (2026-11-17): T_met drives microphysics, settling, RH
  and the radiation optics. `dT_rad` is integrated, but it enters only the
  band temperature of the perturbed call (its emission and gas absorption:
  anomaly mode's longwave Planck damping), so the longwave cooling is still in
  the heating rate and in ARF. With it, every preset is one-way throughout:
  nothing radiative reaches the aerosol, the chemistry or the circulation. The heating
  rate of the last radiation call is output as `hr_kday` (K/day): in the 3-D
  frames, in the probe/zonal frames (`frames_hr_kday`, `frames_zm_hr_kday`), as
  `hr_min/max/rms` in the timeseries, and in the state/final files.
  `RAD_T_FEEDBACK=1` restores the closed loop. Stamped in `PHYS_CFG`.

- **`BC_GAS` defaults to `flux`** whenever the aerosol faces are open, derived
  from the resolved `BC_EDGE` so the gas and aerosol boundaries cannot desync — not
  even under an explicit `BC_EDGE=clamp`. An unconditional `clamp` default put the
  production config in the one incoherent corner of the four combinations: an
  unbounded Dirichlet gas source at a level whose particles are free to leave.
  Measured cost at 13.3 hPa over 24 h — that single level went from 0.3% to ~50% of
  the model's **total** number as a 6–8 nm mode, continuous nucleation fed by
  clamped H2SO4 rather than transport. What `flux` costs: the gases are no longer
  pinned to CESM. `BC_GAS=clamp` still works and is required to reproduce runs made
  under the old default.
- **`WET_SETTLING` / `WET_OPTICS` default ON.** TOMAS carries dry SO4 mass, but a
  stratospheric sulfate particle is an H2SO4/H2O solution droplet, and both the
  settling velocity (∝ Dp, ρ_p) and the optics need the wet size. Sizing the Mie
  tables on the dry core while using a 75 wt% *solution* refractive index — a
  droplet already 25% water by mass, sized as if dry — underestimates 550 nm
  extinction by ~47% for this size distribution; with the real RH/T-dependent
  composition (40–55 wt% near the moist 143 hPa band base) the error reaches a
  factor of 2+. Composition and density come from the same parameterizations the
  microphysics engine applies internally (Tabazadeh 1997, Tang 1997), so settling,
  optics and coagulation see one droplet. `=0` restores the dry sizing.
- **`RAD_SOLAR` default `window_gq` at `RAD_SOLAR_NODES=3` (since 2026-11-17;
  `instant` before).** Flag added 2026-11-11.
  One radiation call's heating is held for the whole interval
  (`STEP_HOURS * RAD_EVERY`), and the calls land on 00/06/12/18 UTC every day, so
  `instant` (the sun at the call hour) gives each longitude the same four local
  times all run long: the daily-mean TOA insolation carries a **stationary** wave-4
  error of −24 to +23 % (`scripts/validation/test_solar_window.py`), and columns
  near the terminator get all-or-nothing SW for 6 h — the dusk/dawn artifact that
  accumulates in `dT_rad`. `window` passes the sunlit-mean cos(zenith) of the
  interval centred on the call hour and a per-column TOA irradiance
  TSI × sunlit fraction (Hogan & Hirahara 2016), so each interval's insolation is
  exact (to 0.06 %, the declination held per window) at the same one-solve cost;
  the surface albedo becomes the insolation-weighted albedo of the window's hourly
  records. `window` was not made the default because it changes what the
  radiation diagnostics mean (6-h means, not snapshots), and because, **measured
  on 2026-11-11, `window` biases the global forcing high**: +20 % daily-mean ARF against hourly radiation
  (+17.5..+25.3 % over 5 days; instant +0.1 %), although it nearly halves the
  local per-level error (32 → 18 %) and cuts the artifact's wave-4 6×. The single
  solve at the sunlit-mean cos(zenith) over-reflects, which is the non-linearity
  of SW transfer across the interval. The experimental `window_ins` (−13.9 %)
  and two-solve `window2` / `window2_ins` (+9.6 % / −7.3 %, best local accuracy,
  2× radiation cost) bracket it without removing it. **The follow-up
  `window_gq` (2026-11-13) removes both errors**: `RAD_SOLAR_NODES` shortwave
  solves at the Gauss nodes of the sunlit part of the interval, longwave once;
  per-level error 7.1 / 6.0 % and ARF +4.2 / +1.1 % for 2 / 3 nodes, global
  heating within 1 %, radiation ×1.37 / ×1.73 (+7 % / +16 % per model step).
  **Why it became the default (2026-11-17):** the one-year `instant`/`window_gq@3`
  A/B showed the scheme drives the long-run wave-4 in `dT_rad` — under `instant`
  it grows to 484 mK along the plume by day 365 (0.9 K at worst), under
  `window_gq` it levels off at ~100 mK — with zonal-mean warming within 0.7 %,
  ARF within 0.2–1.2 % and an identical burden, for ~+10 % run time with the
  reference cached. 3 nodes over 2 because the global ARF error is +1.1 % instead
  of +4.2 %; the remaining error is all shortwave. **What the switch changes:** a
  bare run is not bit-identical to runs before 2026-11-17 (set
  `RAD_SOLAR=instant` to reproduce them); the TOA diagnostics and `arf_toa` are
  sunlit-window quadrature means, not call-hour snapshots, so do not compare
  them across the switch; the reference cache is keyed per mode and node count,
  so the first run over a period builds a new one (~3 h per simulated year,
  once); resuming a checkpoint stamped `instant` warns (PHYS_CFG, warn-only).
  The blend and `dither` were tested and are not recommended. The full account
  is `docs/RADIATION_ARTIFACT_EXPLORATION.md`.
  **`RAD_SOLAR_ALIGN` default `trailing` (2026-11-17):** radiation runs at hour t,
  the last hour of each coupling step, on that step's end state, and its heating
  is added to `dT_rad` at t for the next step. `window_gq` first centred its
  window on t (t−3..t+3 h), which put 3 h of future sunlight into `dT_rad` at t;
  `trailing` integrates t−6..t, the step just finished, so `dT_rad` stays causal.
  The cost of that is that the inputs held at t (meteorology, aerosol, reference
  aerosol) sit at the window's end rather than its centre; a 3-day check from
  both day-365 states, each alignment scored against hourly radiation over its
  own hours (`scripts/validation/check_solar_align.py`), shows that costs nothing
  measurable: column error equal (9.3 / 12.0 %), per-level error +0.3..0.6 pt,
  wave-4 −0.5 pt, global ARF +1.14 vs +1.19 %; the two references themselves
  differ by 2–3 % in column heating and 0.00 % in ARF. The one-year A/B and the
  numbers above ran `centred`; `RAD_SOLAR_ALIGN=centred` reproduces them. Only
  `window_gq` reads it.
- **`P_LO_HPA=1`, `P_HI_HPA=150`** (`driver_fast.py` `setdefault`s since
  2026-07-29; `coupling.py`'s standalone default bottom is 100 hPa). 1 hPa top:
  MAM4 so4 there is 3.2e-17 kg/kg, 7 orders below the 13 hPa value, so the top
  face is effectively an aerosol-free inflow. 150 hPa bottom: MAM4 at 143 hPa
  has the right LATITUDE structure — 7.5e-11 in the tropics (the TTL, already
  scavenged, where upwelling enters the band) vs 1.26e-9 at 60N (where STE
  brings real air in). Caveat: the lower floor delays settling, the model's
  only true aerosol sink.
- `INIT_BIN=so4` (module default and set by both presets). The old `dgnum` path
  binned MAM4 *number* by `dgnumwet` and set `mas = num*MMID` without reading
  `so4_a*`, inflating sulfate mass **4.29×** (coarse mode 6.68×).
- `INJ_ZONAL` defaults to **1** (zonal ring). The point source is 5.6× slower and
  drives runaway nucleation. `INJ_MIRROR=1` releases at both ±`INJ_LAT`, splitting
  the same total 50/50 — the total is what you asked for, not doubled. (The
  2-year runs used `INJ_ZONAL=0`, `INJ_MIRROR=1` at 30° and 55 hPa: a
  single-longitude release on both hemispheres.)
- `INJ_SO2_SCHED_FILE` (added 2026-11-24, for the AIDE-eval CCMI-2022 SAI
  validation): a yearly rate schedule in place of the constant `INJ_SO2_TG_YR`.
  Annual steps, because the WACCM controller it reproduces updates the rate once
  a year. The year counts 365-day years from the run start (hour 0 = `H0`), not
  calendar years of the meteorology. A file is a path outside the checkpoint, so
  the parsed rates are stamped into `inj_cfg` (`INJ_SCHED_N`, `INJ_SCHED_CRC`);
  a RESUME under a changed file is refused. A one-row schedule of 6.0 is
  bit-identical to `INJ_SO2_TG_YR=6` (the increment is built by the same
  `_inj_increment`).
- Diagnostic core window is **symmetric**, `-1` per end → 1.6–121.5 hPa, 22/24
  levels on f09 (1.2–125.5 hPa, 45/47 on JCM), excluding exactly the two levels
  the reservoir is written into.
- **`ADV_VPOS`** vertical positivity limiter in `fct_lr.py` (and so in `sl`'s and
  `ffsl`'s vertical sweep), default **ON** and set explicitly by both presets so
  the log records it. `ADV_VPOS=0` is a forensic escape hatch, not a supported
  configuration. Lin-Rood is exactly conservative but not positive; its
  vertical remap undershot on the steep ultrafine gradient at the injection
  ring and the clip turned that into a spurious number source. 100% of the
  negatives came from that one operator — the horizontal sweeps produce exactly
  zero, being Zalesak-bounded. Measured: the floor injected ~3.3e-3 of the
  standing number burden per 6 h step and accumulated to 35% of a day-90
  standing N; 97.4% of it landed below 10 nm and 0.0025% in the optically
  active 150–1200 nm bins, so mass/AOD/ARF were always safe but total number
  and anything under ~10 nm were not. Over 8 steps of pure advection the
  unlimited scheme grew the number burden 0.94% out of nothing; limited it sits
  at 0.99987.
- `vface` diagnostic labels are `top`/`bot`. **They are FACE labels, signed
  `+ = into slab` — not gross inflow/outflow**, and the per-column/per-substep
  cancellation means gross inflow is not recoverable from them.

## Reading the output — standing caveats

- `init N burden(sum num)` and `M burden(sum mas)` in the header are **unweighted
  sums**. Every ratio in the log normalises by the dp·cos(lat)-**weighted** `N0`
  and `M0`, which are ~260× and ~585× larger. Never divide a logged quantity by a
  header value.
- First ~week of any cold start is spin-up; 90 days is a **transient**, not a
  steady state (the sink is not burden-proportional, loss ~ M^0.35).
- `RAD_MODE=anomaly` baseline is **time-varying** under `AER_SRC=mam4` (under
  `BC_SRC=clim` it follows the monthly climatology), so reported forcing is a
  difference against a moving target.
- Under `RAD_SOLAR=window*` and `window_gq` the radiation diagnostics
  (`arf_toa`, the TOA and surface fluxes) are **interval means**, while
  `aod550`, the burdens and `dT_rad` are values at the call hour; do not compare
  them with an `instant` run's snapshots or with instantaneous CESM fluxes.
  `RAD_SOLAR` is a warn-only `PHYS_CFG` flag, so a resume across a change mixes
  the two kinds of sample in the `ARF_AVG_H` window for one day. The reference
  cache has a separate directory per solar mode and width (an `instant` run
  keeps its old key).
- Under `RAD_T_FEEDBACK=0` (the default) `dT_rad` is a diagnostic: the
  microphysics, settling and optics see T_met, not T_met + `dT_rad`.
- No wet removal anywhere; settling and transport out of the band are the only
  sinks, so aerosol settling into the 100–150 hPa layer lingers.
- `AOD550` is stratospheric-only and sulfate-only over the band. It is
  **uncalibrated** — fine for relative work, and to be settled before quoting
  absolute forcing.
- Burdens are in grid units; compare the CESM and JCM linkings only in kg.
- The budget printout closes to roundoff (`sum` vs `M/M0-1`). If a change breaks
  that closure, the change is wrong — that line is the model's own audit.
