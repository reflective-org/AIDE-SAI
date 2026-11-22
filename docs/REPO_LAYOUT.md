# Repository layout

Where everything lives and, more usefully, the rule for where a new file goes.
The tree below is every tracked file (`git ls-files`, as of 2026-11-20) except
the two submodules' contents.

```
AIDE-SAI/
  README.md, MANIFEST.md, CHANGELOG.md   orientation; the canonical record of every default and why; history
  CLAUDE.md, CONTRIBUTING.md, CODE_OF_CONDUCT.md, CITATION.cff, LICENSE, requirements.txt
  .github/         CI (tests + lint), PR template, CODEOWNERS
  configs/         the linking presets -- what a "linking" is (configs/README.md)
    cesm_numerical.env  DEFAULT: CESM2 h1 meteorology, f09 grid, CESM climatology
    jcm_numerical.env   jcm-strat AL7h70 meteorology on the JCM grid, the climatology regridded onto it
  models/          the model repos, as git submodules pinned to exact commits
    tomas-jax/       sectional aerosol microphysics  (gpu-fast, 787c991)
    jax-rrtmgp/      radiative transfer              (99d2d71 + patches/jax-rrtmgp-zenith.patch)
  patches/         patches that must be applied to a submodule
    jax-rrtmgp-zenith.patch
  src/             ALL source code needed to advance the coupled state
    _paths.py        import-path setup; also defines REPO_ROOT, MODELS, INPUTS
    run_prod.py      the launcher: applies a preset, verifies its pins, execs driver_fast.py
    driver_fast.py   production entry point: installs the microphysics engine and the transport
    coupling.py      the orchestrator: forcing, IC/BC, budget, main(); runnable standalone
    advection/       sl_lr.py + sl_core.py (ADV_SCHEME=sl, the default), fct_lr.py (lr),
                     ffsl_lr.py (ffsl, imports fct_lr's sweeps), fct_fast.py (fast, and the
                     primitives fct_lr imports), fct_core.py (legacy, test_conservation.py only)
    met/             bc_clim.py (BC_SRC=clim), met_jcm.py (MET_SRC=jcm: meteorology and grid),
                     met_jcm_online.py + jcm_server.py (MET_SRC=jcm_online: jcm-strat run live;
                     the server runs in jcm-strat's own interpreter, a subprocess)
    microphysics/    tomas_fast.py -- the tomas_jax.fast adapter (MICRO_ENGINE=fast)
    radiation/       radiation.py -- Mie + RRTMGP -> heating, AOD, ARF; rad_cache.py,
                     rad_cache_build.py -- the per-hour reference cache
    settling/        settling.py -- closed-form gravitational settling
  scripts/         everything that prepares inputs for a run or reads one afterwards
    met/             make_bc_climatology.py, regrid_clim_to_grid.py -- build CLIM_FILE
    radiation/       build_ref_cache.py, test_rad_cache.py, plot_solar_artifact.py
    utils/           plot_run.py, gif_run.py, run_summary.py -- the post-run figures and summary
    analysis/        multi-run comparisons (scripts/analysis/README.md): summary_2yr/ (injection
                     vs control), jcm_vs_cesm_2yr.py (one scenario on two linkings)
    validation/      tests and investigation harnesses (docs/VALIDATION.md)
  inputs/          small static input data + provenance of the external archives
    rad_data/palmer_williams_h2so4.dat
    assets/ne_110m/  Natural Earth 1:110m coastlines (public domain) for map figures
  docs/            documentation (docs/README.md is the index); figures/ holds its SVGs
  runs/            raw model output, one directory per OUT_TAG   (only README.md tracked)
  outputs/         derived figures and tables                    (only README.md tracked)
```

The `src/` vs `scripts/` split is the load-bearing one: `src/` is what advances
the coupled state, `scripts/` is what prepares its inputs or inspects it
afterwards. A run must not depend on anything in `scripts/`.

## Where does a new file go?

| the file is… | it goes in |
|---|---|
| needed to advance the coupled state | `src/<process>/` |
| a new implementation of an existing slot (a microphysics engine, a transport scheme, a met source) | `src/<process>/`, selected by that slot's existing selector (`MICRO_ENGINE`, `ADV_SCHEME`, `MET_SRC`, `BC_SRC`, `AER_SRC`, `RAD_ENGINE`) |
| a new physical process | a new `src/<process>/` directory |
| a new linking (a combination of slot implementations, inputs and pins) | a new `configs/<name>.env` preset, plus a MANIFEST.md entry |
| a different version of a model repo | a second submodule path (e.g. `models/tomas-jax-dev`) and a preset whose slot points there — never a vendored copy |
| a check that the physics is right | `scripts/validation/` |
| a script that builds a run's inputs | `scripts/<process>/` |
| a figure or a plot script | `scripts/utils/`; its products in `outputs/` |
| a comparison across several runs (figures + `metrics.json`) | `scripts/analysis/`; run it from a directory under `runs/` |
| data the model reads and that is small | `inputs/` |
| data the model writes | `runs/<TAG>/` — never committed |
| a change to tomas-jax or jax-rrtmgp | that repo, upstream; or `patches/` if it cannot be |

## How imports work inside `src/`

`src/` is deliberately **not** a Python package. Every module is a flat,
top-level name (`coupling`, `settling`, `fct_lr`, `radiation`, `tomas_fast`) and
the subdirectories are organisational only. `src/_paths.py` puts each
subdirectory on `sys.path`, so `import settling` works from anywhere in the tree
with no package prefix.

Two consequences worth knowing before editing:

1. **Do not add `__init__.py`** to a `src/` subdirectory without converting all
   the imports at the same time. `src/settling/` and `settling.py` share a name;
   the module wins today, but an `__init__.py` would turn the directory into a
   regular package and shadow it. `_paths.py` inserts the subdirectories ahead of
   `src/` itself for the same reason.
2. **Do not let an "organize imports" pass reorder these files.** Several modules
   run path setup before importing what that setup makes importable, and
   `driver_fast.py` must set its environment defaults before importing anything
   that reads the environment at import time. This is why `ruff` rule `I001` is
   off in CI.

Converting `src/` into a proper installable package removes all of the above.
That is deliberately deferred — see [DEFERRED.md](./DEFERRED.md).

## The submodules and the presets

`models/tomas-jax` and `models/jax-rrtmgp` are git submodules pinned to exact
commits, so a clone reconstructs the code that produced a result.
`models/jcm-strat` (with its own nested submodule `external/jax-gcm`) is a third,
used only by `jcm_online_numerical`. Each preset
names, per model slot, the checkout that fills it and the commit it must be at
(`#@submodule SLOT PATH SHA [patch=FILE]`); `run_prod.py` refuses to start when a
checkout does not match, and points `TOMAS_JAX_PATH` / `RRTMGP_PATH` (and
`JCM_STRAT_PATH`) at it.
`scripts/validation/test_configs.py` (in CI) checks that every pin equals the
commit the superproject records.

```bash
git clone --recurse-submodules https://github.com/reflective-org/AIDE-SAI.git
# or, in an existing clone:
git submodule update --init --recursive
git -C models/jax-rrtmgp apply ../../patches/jax-rrtmgp-zenith.patch
```

Outside `run_prod.py`, `src/coupling.py` and `src/radiation/radiation.py`
resolve the two repos from `models/`, overridable with `TOMAS_JAX_PATH` /
`RRTMGP_PATH`.
