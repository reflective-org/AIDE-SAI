# configs/ — linking presets

A **preset** is one linking written down: which implementation fills each
model slot, which checkout of each model repo it runs, and the settings that
differ between linkings. `src/run_prod.py --preset NAME` applies
`configs/NAME.env`. Without `--preset` it applies `cesm_numerical`, the default.

| preset | linking |
|---|---|
| `cesm_numerical.env` (default) | CESM2 h1 meteorology on f09, `tomas_jax.fast`, RRTMGP. Reproduces the 2-year CESM runs bit-identically |
| `jcm_numerical.env` | jcm-strat AL7h70 meteorology on the JCM T63 grid, `tomas_jax.fast`, RRTMGP. Reproduces the 2-year JCM run bit-identically |

```bash
python3 src/run_prod.py --list-presets                    # name + first comment line; marks the default
python3 src/run_prod.py --preset jcm_numerical --dry-run ... # resolved environment + pin check, no launch
```

**Why presets, and not branches or shell scripts?** Before 2026-11-20, during
development, each linking lived on its own branch, with its values hard-coded
in that branch's `run_prod.py`. Running a second linking meant a second
worktree, and nothing at launch checked that the model checkouts were the ones
the branch recorded. A preset is a file on `main` that names both the values and the
commits, and the launcher checks the commits before it starts.

## File format

One item per line. The file is parsed by `run_prod.py`'s `load_preset()`, never
by a shell: no variable expansion, no quoting, no includes.

```
# First comment line: a one-line description (shown by --list-presets)
#
# Provenance: which runs this preset reproduces, and with which commit.
#@submodule tomas-jax  models/tomas-jax  787c991e499342df2dc59c96470bd5f167537868
#@submodule jax-rrtmgp models/jax-rrtmgp 99d2d7174d2e6967c9c45e8f4bae22a4994d1c18 patch=patches/jax-rrtmgp-zenith.patch

# a comment saying WHY this value
MET_SRC=cesm
```

| line | meaning |
|---|---|
| `KEY=VALUE` | Sets an environment variable. `KEY` must be a knob in `run_prod.py`'s `KNOBS` table, so that it is documented in `--help` and echoed in the run header. Whitespace around `=` is stripped, and the value is taken literally, except that a relative value of a path knob (`PATH_KNOBS`: the knobs whose `KNOBS` type is `PATH`, e.g. `CLIM_FILE=inputs/bc_clim/bc_clim.npz`) is resolved against the repo root, so a preset does not depend on the launch directory |
| `#@submodule SLOT PATH SHA [patch=FILE]` | The checkout that fills model slot `SLOT`. `PATH` is relative to the repo root. `SHA` is the full 40-character commit that the checkout must be at. `FILE` (relative to the repo root) is a patch that must be applied on top |
| `# ...`, blank | A comment. The first `# ` line is the preset's description |

The slots are the keys of `SLOT_ENV` in `run_prod.py`. Each slot sets the path
variable that `coupling.py` reads:

| slot | sets | what fills it |
|---|---|---|
| `tomas-jax` | `TOMAS_JAX_PATH` | the tomas-jax checkout (microphysics, and the Mie and water-equilibrium routines that radiation and settling import) |
| `jax-rrtmgp` | `RRTMGP_PATH` | the jax-rrtmgp checkout (radiative transfer and its k-distribution data) |

Every preset must pin **every** slot exactly once. `test_configs.py` enforces
this, so no preset silently inherits whatever `models/` happens to contain.

## Forbidden keys

`load_preset()` refuses a preset that sets any key in `PRESET_FORBIDDEN`. The
keys fall into three groups:

- **The scenario and run control**: `INJ_SO2_TG_YR`, `INJ_H2SO4_TG_YR`,
  `INJ_HPA`, `INJ_LAT`, `INJ_LON`, `INJ_ZONAL`, `INJ_MIRROR`, `N_HOURS`,
  `N_DAYS`, `OUT_TAG`, `RESUME`, `H0`. These change from run to run and go on
  the command line. A preset says *which model* runs, not *what experiment*
  runs on it. Keeping them apart is what lets the same scenario flags run
  against `cesm_numerical` and `jcm_numerical` unchanged.
- **Variables read by presence or truthiness**: `DEBUG`, `PROFILE`,
  `OH_SUBSTEPS`, `MICRO_SUBSTEPS`, `DUMP_PREMICRO`, `INIT_SIGMA`. The code
  tests these with `'X' in os.environ` or `bool(os.environ.get(X))`, so writing
  out what looks like the default changes behaviour. For example, `DEBUG=0`
  turns debugging *on*.
- **The slot paths**: `TOMAS_JAX_PATH`, `RRTMGP_PATH`. These come only from
  `#@submodule` lines, so a path can never escape the pin check.

A key that is not in `KNOBS` is also refused. Add it to `KNOBS` first, with
its help text, and the preset can then use it.

## Precedence

For each knob, the highest source wins:

1. a `--flag` on the command line
2. a variable already exported in the environment
3. the preset (including the slot paths from `#@submodule`)
4. `run_prod.py`'s run-control defaults (the non-`None` column of `KNOBS`:
   `N_HOURS=2160`, `OUT_TAG=prod90d`, `STATE_CKPT=1`, `FRAME_EVERY=24`,
   `DEBUG=1`, `PROFILE=1`, `INJ_*` = no injection, ...)
5. the module defaults in `coupling.py` / `driver_fast.py` / `radiation.py`

When an exported variable overrides a *different* preset value, the launcher
prints a `WARNING: exported X=... overrides the preset's X=...`. That warning
catches the classic failure: a tmux shell left with an old `export MET_SRC=...`
silently turns a `--preset cesm_numerical` run into something else. Every resolved value
is echoed with its source (`flag`, `env`, `preset:<name>`, `launcher`), and
`AIDE_PRESET=<name>` is passed to the run, so a log always says which linking
produced it.

Other ways to choose the preset:

- `--preset path/to/file.env` uses a file outside `configs/` (any path ending
  in `.env`).
- An exported `AIDE_PRESET` is used when `--preset` is not given.
- `--preset none` applies no preset, which leaves the run-control defaults and
  the module defaults. Note that the module defaults are not a linking: they
  give `BC_SRC=cesm`, every boundary field from the hourly archive.

## Pin verification

Before anything is launched (and in `--dry-run`), `check_pins()` checks each
`#@submodule` line:

1. `PATH` exists, or else run `git submodule update --init PATH`;
2. `git -C PATH rev-parse HEAD` equals `SHA`;
3. if a patch is named, `git apply --check --reverse FILE` succeeds in `PATH`,
   which means the patch is applied.

Any mismatch stops the launch with the command that fixes it. Then
`TOMAS_JAX_PATH` / `RRTMGP_PATH` are set to the absolute checkout paths, so
the run imports exactly what was verified. The header prints each slot as
`(verified)`.

`--no-verify-pins` skips the check, and the header then says `(NOT VERIFIED)`.
Use it only for local experiments. A run made that way is **not** the preset's
model, and its log says so. An exported `TOMAS_JAX_PATH` / `RRTMGP_PATH` also
wins over the slot path (with the warning above). The pin was checked on the
preset's path, not on yours, so treat that run as unverified as well.

In CI, `scripts/validation/test_configs.py` checks that every preset parses
with `load_preset()` and sets `ADV_VPOS=1`. It also checks that each preset
pins every slot once, that every pinned `PATH` is a registered submodule
pinned to **exactly the gitlink the superproject records** (`git ls-tree
HEAD`), and that every named patch exists. So a preset and
`git submodule update` can never disagree silently: moving a submodule without
updating every preset that points at it fails CI.

## Do not `source` a preset

```bash
source configs/jcm_numerical.env && python3 src/run_prod.py ...   # WRONG
```

This is not supported, and it goes wrong in three ways:

- **It bypasses validation.** The shell does not check for forbidden or unknown
  keys, and to bash every `#@submodule` line is a comment. The pins are never
  checked, and the slot paths are never set.
- **It either does nothing or changes precedence.** A plain `source` sets
  unexported shell variables, which `run_prod.py` never sees: it applies its
  own preset (the default `cesm_numerical`, if you gave none) and the sourced file had no
  effect. With `set -a` (or an `export` in front) the values become *exported
  environment*, which outranks the preset `run_prod.py` applies. You get a mix
  of the two, held together only by warnings.
- **It leaves stale variables.** They stay in that shell, or tmux pane,
  after the run; once exported. The next `--preset cesm_numerical` launched from it still runs with
  `MET_SRC=jcm` and the JCM climatology, because exports outrank presets.

Always pass the preset to the launcher with `--preset NAME`. If a shell already
has exports in it, `unset` them, or start a fresh shell.

## Adding a preset

1. **Write `configs/<name>.env`.** Start with the one-line description and a
   provenance comment that names the runs, the date and the commit. Pin both
   slots, and set only the knobs that make this linking different. Leave
   everything else to the module defaults, so a later change to a default
   reaches every preset that does not override it. Every preset must set
   `ADV_VPOS=1`, because without the vertical positivity limiter the number
   field is not usable. A new implementation goes behind an existing selector
   (`MET_SRC`, `BC_SRC`, `AER_SRC`, `ADV_SCHEME`, `MICRO_ENGINE`,
   `RAD_ENGINE`) and is chosen here. It never goes on a branch.
2. **Check it**: `python3 scripts/validation/test_configs.py`, then
   `run_prod.py --preset <name> --dry-run` from a run directory with your
   scenario flags. Read every resolved value and its source.
3. **Record it.** Add a dated MANIFEST.md entry saying what the linking is and
   why each value differs from the defaults. If the preset reproduces runs,
   give it the run names and the commit, and confirm that the dry run resolves
   the knob values those run logs recorded.

### A preset that needs a different model commit

A slot's checkout is a git submodule pinned to an exact commit, never a
vendored copy. To run a different commit, such as tomas-jax `dev` `835ee24` for
`MICRO_ENGINE=full`, which calls `make_step` arguments that `gpu-fast` does not
have, add it as a **second submodule path**. Do not move the existing one,
because `cesm_numerical` and `jcm_numerical` pin it.

```bash
git submodule add -b dev https://github.com/reflective-org/tomas-jax models/tomas-jax-dev
git -C models/tomas-jax-dev checkout 835ee249c9596af2534090451edd7247ec488dce
git add .gitmodules models/tomas-jax-dev
```

Then point the slot at it in the new preset:

```
#@submodule tomas-jax  models/tomas-jax-dev 835ee249c9596af2534090451edd7247ec488dce
#@submodule jax-rrtmgp models/jax-rrtmgp    99d2d7174d2e6967c9c45e8f4bae22a4994d1c18 patch=patches/jax-rrtmgp-zenith.patch
MICRO_ENGINE=full
```

This is how a config chooses which submodule fills a slot. Several presets can
share a checkout, and two commits of the same model can sit side by side,
because each preset names the path and the commit it runs.

Add test coverage with it:

- `test_configs.py` already checks the new pin against the new gitlink.
- If the new commit changes what the slot must provide, add a check for that
  too. For example, a dry-run or import check that `MICRO_ENGINE=full` resolves
  against the dev checkout. A harness that needs the submodule or a GPU cannot
  run in CI. Write it as a validation script, list it in `docs/VALIDATION.md`,
  and run it before merging.
- If the preset needs a slot that does not exist yet (a third model repo), add
  it to `SLOT_ENV` in `run_prod.py` together with the variable `coupling.py`
  reads for it. Every existing preset must then pin that slot too.

A patch that the new commit needs goes in `patches/` and is named with
`patch=`. Generate it against the pinned commit; `git apply` refuses to
half-apply onto anything else.
