# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

AIDE-SAI is a modular dycore–aerosol–radiation coupler for stratospheric
aerosol injection, in JAX on GPU. Each physical process sits in a swappable
slot — meteorology (`MET_SRC`), boundary climatology (`BC_SRC`), transport
(`ADV_SCHEME`), aerosol microphysics (`MICRO_ENGINE`), settling, radiation
(`RAD_ENGINE`) — and each slot is filled by a numerical scheme today, or by a
learned emulator later. The meteorology is prescribed (one-way: the winds are
read and the circulation never responds); aerosol, radiation and microphysics
are interactively coupled. Under `RAD_T_FEEDBACK=1` the radiative heating from
the model's own aerosol accumulates into the temperature the microphysics, the
settling and the next radiation call see; the default is `RAD_T_FEEDBACK=0`,
where T_met drives them and `dT_rad` and the heating rate (`hr_kday`) are
diagnostics (`dT_rad` only damps, via the emission temperature).

**A linking is a preset, not a branch.** `configs/<name>.env` fixes which
coupled model runs: the knob values that differ between linkings and, via
`#@submodule SLOT PATH SHA [patch=FILE]` lines, the exact submodule commit (and
patch) filling each model slot. `src/run_prod.py --preset NAME` (default
`cesm_numerical`; `none` = coupling.py's module defaults) refuses to start if a slot's
checkout is not at its pin, and echoes the preset, the pins and every resolved
value. Three linkings ship, all numerical:

- `cesm_numerical` (default): CESM2 h1 hourly winds/T on the f09 grid, the monthly CESM
  climatology for everything else, `tomas_jax.fast` microphysics
  (`models/tomas-jax` = gpu-fast 787c991), semi-Lagrangian transport (`sl`,
  K=12), RRTMGP + Mie radiation.
- `jcm_numerical`: the same, but jcm-strat AL7h70 6-hourly meteorology on the JCM grid
  (T63 Gaussian 96×192) with the CESM climatology regridded onto it.
- `jcm_online_numerical`: as `jcm_numerical`, but jcm-strat AL7h **runs live**
  beside the coupler (`MET_SRC=jcm_online`): `src/met/jcm_server.py` in a
  subprocess under jcm-strat's own interpreter (`JCM_PY`, Python 3.11 / JAX 0.10;
  this repo runs JAX 0.6), importing `jcm_strat` and `jcm` from the third slot,
  `models/jcm-strat` (f3c2070, nested `external/jax-gcm`). Only this preset pins
  that slot (`SLOT_NEEDED_IF`). It also sets machine paths (`JCM_PY`,
  `JCM_CACHE`, `JCM_SCRATCH`). jcm-strat on GPU is **not deterministic run to
  run**, so this linking is validated statistically
  (`scripts/validation/validate_jcm_online.py`), never bitwise.

A new linking — including a mixed numerical/emulated one — is a new preset,
plus any new slot implementation behind an existing selector. A different
version of a model package is a *second* submodule path (e.g.
`models/tomas-jax-dev`) that the new preset's slot points at; never a vendored
copy and never a long-lived branch. Changing a preset or a default needs a
`MANIFEST.md` entry (see `CONTRIBUTING.md`).

**Direction of travel.** The end goal is a model-agnostic coupler: other
meteorology sources (including emulated dycores, which would close the
circulation loop too — `MET_SRC=file` is that seam) and other aerosol schemes
(CARMA, MAM, GLOMAP). CESM, JCM and TOMAS are what is wired in first, not the
target. At any seam — the meteorology readers (and the live-JCM server's line
protocol), the wind interface, `AER_SRC`,
the engine swap in `driver_fast.py`, the radiation driver, the preset slots —
prefer keeping the seam generic over specialising it, and say so if a change
would make a slot harder to swap. `README.md` is the orientation;
**`MANIFEST.md` is the canonical reference for this tree** — it records every
default and, more importantly, *why* each one is what it is. Where `docs/`
disagrees with MANIFEST, MANIFEST is current. Read it before changing physics.

The TOMAS and radiation emulators, the engine A/B harness and the MAM4-JAX
driver are not part of this repo (yet); an emulator would fill its slot behind
the existing selector (`MICRO_ENGINE`, `RAD_ENGINE`). (`AER_SRC=mam4`, the default, is unrelated: it
bins CESM's MAM4 output fields onto the TOMAS bins.)

## Layout invariants

- **Source lives in `src/`, but the module names are flat on purpose.**
  `coupling.py` does bare `import settling` / `import radiation`, and
  `src/_paths.py` puts each `src/` subdirectory on `sys.path` so they resolve.
  `src/` is NOT a package: do not add `__init__.py` to a subdirectory whose name
  collides with a module (`src/settling/` vs `settling.py`), and do not let an
  import-sorting pass reorder modules that set up `sys.path` before importing
  what that setup makes importable. See `docs/REPO_LAYOUT.md`.
- `run_prod.py` is the launcher: it applies the preset, resolves the
  environment (precedence `--flag` > exported env > preset > launcher
  run-control defaults > coupling.py defaults; an exported variable that
  overrides a preset value is warned about) and execs `driver_fast.py`.
  Presets may only set knobs in its `KNOBS` table, and never the scenario
  (`INJ_*`), run control (`N_HOURS`, `OUT_TAG`, `RESUME`, `H0`) or variables
  read by presence/truthiness (`DEBUG`, `PROFILE`, `OH_SUBSTEPS`, ...): see
  `PRESET_FORBIDDEN`.
- `driver_fast.py` is the production entry point: it imports `coupling.py` and
  selects the microphysics engine (`MICRO_ENGINE=fast`, the default:
  `tomas_jax.fast` via `microphysics/tomas_fast.py`; `full` = tomas-jax dev's
  `make_step`, which needs a preset whose tomas-jax slot is a dev checkout).
  `coupling.py` alone is the standalone/dev path and uses the same advection,
  so the two agree.
- `advection/fct_core.py` is legacy and **not on the run path** — it exists
  only for the bit-identical comparison in
  `scripts/validation/test_conservation.py` (in CI).
- Analysis scripts read `coupled_*_<TAG>.npz` from the *current working
  directory* and write `<TAG>_*.png` beside them.
  Multi-run comparisons (`scripts/analysis/`) take the run directories as
  flags and write into the current directory: run them from under `runs/`.
- Prefer extending `coupling.py` over adding helper scripts; the four
  overlapping plot scripts this repo used to have are why `scripts/utils/plot_run.py` is one
  file.

## Validation happens in AIDE-eval

**Any validation or evaluation of this model happens in
[AIDE-eval](https://github.com/reflective-org/AIDE-eval)**
(`reflective-org/AIDE-eval`): scoring a linking or one of its parts against
observations, reanalyses or reference models, and ranking or scoring linkings
against each other by how well they do. Do not build validation analyses,
skill metrics, scorecards or reference comparisons in this repo. Write the run
outputs AIDE-eval expects (see its evaluation protocol) and evaluate them
there, so every linking is scored the same way in one place.

**Config comparisons are allowed here.** Comparing runs of this model with
each other, such as two presets (`cesm_numerical` vs `jcm_numerical`), a preset
against itself with one knob changed, or a sensitivity sweep, belongs in this
repo. These comparisons show how the output *differs* and why (which slot,
which knob, which process in the budget). They do not say which run is *right*.
The line is the reference: once a comparison brings in observations, a
reanalysis or an external model, or judges a run as better or worse, it is an
evaluation and goes to AIDE-eval.

What also stays here are **code-correctness gates**: the CI tests and the
`scripts/validation/` harnesses that check an implementation does what it
claims, for example conservation, closed-form physics, preset pins and
bit-identity after a refactor. They tell you the code is right, not that the
model is right.

## Working on this code

- **A GPU is not optional** for anything but a smoke test, and JAX will
  *silently* fall back to CPU if it cannot load `libcuda.so.1` — `run_prod.py`
  handles that. If a step takes minutes instead of seconds, check
  `jax.devices()` first.
- **Validate advection changes at `ADV_F32=1 ADV_CFL=0.5`**, the production
  precision (what coupling.py and driver_fast.py bind), not the f64/cfl=0.2
  defaults of the advection functions themselves. Two positivity-limiter bugs
  were invisible in f64 and fatal in f32.
- Changing a default changes what a bare run *means*. Every env var is read once
  at module import with `os.environ.get`, and the resolved config is echoed in
  the run header so any log is self-describing — keep it that way.
- The injection scenario (`INJ_*`) is stamped into the state checkpoint and a
  `RESUME` onto a mismatched checkpoint is refused; physics-mode flags
  (`WET_*`, `SETTLE`, `ADV_VPOS`) only warn. Both arrays are append-only:
  adding a field must not lock out older checkpoints.
- Diagnostics and plots report concentrations at STP; the microphysics kernels
  keep ambient density. Don't mix the two.
- The budget printout closes to roundoff (`sum` vs `M/M0-1`). If a change breaks
  that closure, the change is wrong — that line is the model's own audit.

## Run outputs

**Outputs live in `runs/<TAG>/`**, one directory per `OUT_TAG`, gitignored.
Every output path in `coupling.py` and `scripts/utils/plot_run.py` is relative to the working
directory, so the working directory is the run:

```bash
mkdir -p runs/prod90d && cd runs/prod90d
../../src/run_prod.py --preset cesm_numerical --out-tag prod90d   # --dry-run first to see the resolved env
python3 ../../scripts/utils/plot_run.py prod90d
```

`run_prod.py` **refuses** to start from anywhere inside the checkout except
under `runs/`. That guard exists because the mistake is otherwise invisible: `.npz`/`.png` are gitignored, so a
run launched from the wrong place looks entirely normal while filling the tree
(this is how 29 GB accumulated before 2026-08-12). The scripts resolve their own
tree from `__file__`, so invoking them by absolute path from anywhere is correct
and is the intended usage — do not `cd` into the repo to run them.

A single 90-day run writes GBs of `.npz`, gitignored by broad patterns (`*.npz`,
`*.png`, `*.gif`) — never commit them, and never widen the tracked set to
include them. Nothing untracked belongs outside `runs/`: the checkpoint-reading
validation harnesses take their `STATE` from `$PWD` too, so run those from the
run directory as well.
