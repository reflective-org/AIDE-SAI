# Contributing to AIDE-SAI

Thank you for your interest in contributing! This document provides guidelines for contributing to the project.
All contributors are expected to follow our [Code of Conduct](./CODE_OF_CONDUCT.md).

## Getting Started

1. Fork and clone the repository:

```bash
git clone https://github.com/reflective-org/AIDE-SAI.git
cd AIDE-SAI
git submodule update --init
git -C models/jax-rrtmgp apply ../../patches/jax-rrtmgp-zenith.patch
```

2. Install
Follow the installation instructions in [the README](./README.md#installation).

The patch is required setup, not optional: the `cesm_numerical` and `jcm_numerical` presets pin it,
and `run_prod.py` refuses to start without it (see the patch file's header).

## Branch model

There is **one long-lived branch, `main`**. Everything else is short-lived:

```bash
git checkout -b feat/your-feature-name       # new feature or a new linking
git checkout -b fix/your-bug-fix             # bug fix
git checkout -b docs/what-you-documented     # documentation only
```

Open the pull request against `main`; the branch is deleted after merge. Keep
a PR to one change, and say in its description which presets it affects and
how you checked that.

**A new linking is a preset, never a branch.** A linking — which meteorology,
climatology, transport, microphysics and radiation run together, and which
submodule commits fill the model slots — lives in `configs/<name>.env`
(`src/run_prod.py` documents the format). Adding one, including a mixed
numerical/emulated one, means:

- a new `configs/<name>.env`, with a header comment saying what it is and,
  once runs exist, which commit and runs it reproduces;
- any new slot implementation behind an existing selector (`MET_SRC`,
  `BC_SRC`, `AER_SRC`, `ADV_SCHEME`, `MICRO_ENGINE`, `RAD_ENGINE`), keeping the
  seam generic rather than special-casing the new source;
- if it needs a different version of a model package, a **second submodule
  path** for it (e.g. `models/tomas-jax-dev` for tomas-jax dev, which
  `MICRO_ENGINE=full` needs) pinned to an exact commit, and the new preset's
  `#@submodule` line pointing that slot there. Model packages are always git
  submodules pinned to exact commits, never vendored copies; a fix that cannot
  go upstream yet ships as a patch in `patches/` named by the preset.

**Changing a preset or a default needs a `MANIFEST.md` entry**, dated and
saying why. A default is what a bare run *means*, and a preset is what a
linking *is*: changing either silently changes every later run under the same
name, so the record of when and why is what keeps old logs interpretable.

**Frozen versions are annotated tags** (e.g. one per published run set):

```bash
git tag -a <name> -m "what this freezes and which runs it reproduces"
```

To reproduce a tagged run: check out the tag, `git submodule update --init`,
apply the preset's patch, and launch with the same preset and scenario.
`run_prod.py` verifies the pins, so a checkout that does not match fails loudly
instead of producing a different model under the same name.

## Code Style
- PEP 8 with 4-space indentation
- Type hints encouraged for public APIs
- No enforced formatter yet; please match the surrounding style

## Running Tests

**This model does not fail loudly.** A broken run still completes, stays finite,
and draws a plausible-looking figure — four transport bugs once cost a mass leak
that went unnoticed for months, with no crash and no warning. Run these before
you push:

```bash
python3 scripts/validation/test_conservation.py     # transport does not leak mass
python3 scripts/validation/test_physics_math.py     # settling math unchanged
python3 scripts/validation/test_configs.py          # presets parse; pins == submodule gitlinks
```

Seconds each, no GPU and no CESM data needed, and non-zero exit on failure. CI
(`.github/workflows/ci.yml`) runs all three plus an import check and an
errors-only `ruff` pass on every push to and pull request against `main`; a PR
is merged only with CI green. Run the first two whenever you touch transport
(`src/advection/`) or `settling.py`, and `test_configs.py` whenever you touch
`configs/`, a submodule pin or `run_prod.py`'s knob table.

CI cannot run the model: it has no GPU and no CESM archive. A change to physics
or to a preset therefore also needs a short run on the box — at least
`run_prod.py --preset <name> --dry-run` for every affected preset, and a smoke
run (a few 6 h steps) from a `runs/<tag>/` directory — and the PR should say
what was run.

The rest of `scripts/validation/` are investigations rather than pass/fail tests: they
load a real run and print diagnostics for a human to read, and need the CESM
archive and usually a checkpoint. See
[docs/VALIDATION.md](./docs/VALIDATION.md) for what each one asks, how to run
it, and the precision rule to follow if you change advection.

### Writing Tests

There is no pytest in this repo, and no `tests/` directory. Automated tests live
in `scripts/validation/` as standalone scripts following the pattern in
`test_conservation.py`:

- a module-level `check(name, ok, detail)` that prints `PASS`/`FAIL` and appends
  failures to a `FAILED` list;
- one function per group of related assertions;
- a `__main__` block that runs them and ends with `sys.exit(1 if FAILED else 0)`.

The exit code is what makes a script usable in CI, so it is not optional.

To be runnable in CI a test must be **self-contained**: inputs built from a
fixed seed (`np.random.default_rng(0)`), no GPU, no CESM archive, and no
`tomas-jax` / `jax-rrtmgp` import. Anything needing those is an investigation
rather than a test — write it as one, and say so in its docstring. If you add a
self-contained harness, add a step for it to `.github/workflows/ci.yml`.

## Reporting Issues

When reporting a bug, please include:

- Python version
- The commit, and the preset (the run header echoes it with its pins)
- Steps to reproduce the issue
- Full error traceback

## License

By contributing, you agree that your contributions will be licensed under the Apache 2.0 license.
