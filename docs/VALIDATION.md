# Validation

## Why bother

This model does not fail loudly. Its characteristic failure is a run that
completes, stays finite, and produces fields that look entirely reasonable while
being wrong.

That is not hypothetical. Two examples from this tree's own history:

- Four separate transport bugs added up to a **mass leak** that went unnoticed
  for months (fixed 2026-07-25): the aerosol burden fell faster than the run's
  own budgeted sinks could account for. Every one was silent — no crash, no NaN,
  no warning. The model just quietly lost aerosol.
- Two positivity-limiter bugs were **invisible in float64 and fatal in float32**.
  Testing at the module defaults would have found nothing; production runs in
  f32.

A figure cannot tell you about either. Both sides of any comparison you draw come
out of the same code, so a plot of a broken model looks like a plot of a working
one. That is what the checks here are for: they assert invariants the physics
must satisfy regardless of what the answer turns out to be — mass in equals mass
out, a droplet is bigger than its dry core, fall speed rises with size.

## Three kinds of thing in `scripts/validation/` and `scripts/radiation/`

They share folders but answer different questions.

### CI tests — "did I break something?"

Self-contained: no GPU, no CESM archive, no submodule checkout; they run in
seconds and **exit non-zero on failure**. CI runs them on every push and pull
request to `main`.

| file | what it asserts |
|---|---|
| `test_conservation.py` | Lin-Rood transport conserves tracer mass to roundoff for any wind field; the cos(φ) area metric, grid spacing, continuity-derived omega and polar caps all behave; the legacy sweeps match `fct_core.py` bit for bit. Guards the four bugs behind the leak |
| `test_physics_math.py` | closed-form settling physics: Tang solution density, wet growth factor (1.087–2.566, cross-checked against the run header), slip-corrected fall speed, and that the implicit sweep changes the burden *only* by its bottom outflow |
| `test_configs.py` | every preset in `configs/` parses with `run_prod.py`'s own loader (known knobs only, no forbidden keys, well-formed `#@submodule` lines), pins exactly the slots it needs (every slot; `jcm-strat` exactly when `MET_SRC=jcm_online`), pins each registered submodule to exactly the commit the superproject records (the index gitlink, `git ls-files --stage`), names patch files that exist, and the default preset exists |

```bash
python3 $REPO/scripts/validation/test_conservation.py     # ~12 s
python3 $REPO/scripts/validation/test_physics_math.py     # ~2 s
python3 $REPO/scripts/validation/test_configs.py          # ~1 s
```

### Gates — "is this component still right?" (CPU, not in CI)

Also pass/fail and runnable on a CPU, but they need more than CI installs: the
transport gates are slower (~30 s to ~2 min), and the solar-geometry gate
imports `radiation.py`, which needs the tomas-jax and jax-rrtmgp checkouts.
Run the relevant one before merging a change to that component.

| file | what it asserts |
|---|---|
| `test_sl_transport.py` | the semi-Lagrangian transport (`sl_core.py`, `sl_lr.py`; the default scheme): constant field preserved across the poles, Williamson case 1 at α = 0° and 90°, mass fixer to 1e-12, bin consistency, the coupling-facing driver, and the split operator's flux-form column budget. `SL_TEST_GRID=jcm` runs the same gates on the JCM grid (reads one JCM file) |
| `test_ffsl_transport.py` | `ffsl_lr.py`: equals `fct_lr` at \|c\| < 1, telescoping budget and positivity at any Courant number, Williamson case 1 at K = 6, the 3-D driver |
| `test_solar_window.py` | `RAD_SOLAR` geometry against brute-force integrals: sunlit fraction and mean cos(zenith) for every window mode, exact insolation, polar day and night, `window_gq`'s quadrature and its trailing window |

### Investigations and GPU gates — "what is actually going on here?"

These load real meteorology (and often a run checkpoint) and need a GPU. Most
print diagnostics for a human to read; there is no fixed right answer. Run them
when chasing a specific question, not routinely.

| file | the question | needs |
|---|---|---|
| `validate_vpos_f32.py` | is the `ADV_VPOS` positivity limiter doing its job at production precision? | CESM archive + checkpoint (`STATE`) |
| `ch4_truth_test.py` | how well does a transport scheme reproduce CESM's own CH4 over a year? The test that made `sl` K=12 the default ([PROCESSES.md](./PROCESSES.md) §1.10) | CESM archive + GPU; writes `outputs/ch4_truth/<tag>.npz` |
| `validate_radiation.py` | how does our radiation compare against CESM's own heating rates and extinction? | CESM archive + `jax-rrtmgp` + GPU. Writes a `.png` |
| `test_radiation.py` | does the radiation driver run on one real CESM hour without NaNs? | CESM archive + `jax-rrtmgp` + GPU |
| `validate_jcm_online.py` | does `MET_SRC=jcm_online` (jcm-strat run live) stay on the stored AL7h70 track? Band zonal-mean U and T, online − stored, next to the stored year-to-year spread at the same dates. **Statistical by design**: jcm-strat on GPU is not deterministic run to run, so this linking is never checked bitwise. Reference: 0.21x (U), 0.34x (T) the year-to-year spread over 60 days (MANIFEST.md) | the jcm-strat slot + `JCM_PY` venv + the AL7h70 chain + GPU; settings from `--preset jcm_online_numerical`; run from `runs/<tag>/` |
| `check_solar_window.py`, `check_solar_accum.py`, `check_solar_gq_sw.py`, `check_solar_gq_lw.py`, `check_solar_align.py` | `RAD_SOLAR` modes against hourly radiation on frozen aerosol states: per day, accumulated over days, the one-longwave path, SW vs LW error, centred vs trailing window ([RADIATION_ARTIFACT_EXPLORATION.md](./RADIATION_ARTIFACT_EXPLORATION.md)) | CESM archive + GPU + a state file |
| `scripts/radiation/test_rad_cache.py` | pass/fail: a cached reference hour is bit-identical to the computed one; the split solve equals the default | CESM archive + GPU + a day-30 state (`runs/chemdump/debug_nucleation/premicro_state_step120.npz` on the development machine) |

`test_radiation.py` is named like an automated test but behaves like an
investigation — it prints, it does not gate.

The harnesses that were used to study operators, emulators and engine A/Bs
(`compare_*`, `bench_*`, `floor_anatomy.py`, `micro_ab_offline.py`, the emulator
tests) are not part of this repo (yet).

## Rules that matter

### Production runs in single precision — validate there

> [!IMPORTANT]
> **A change to advection confirmed only in float64 is not confirmed.** Two bugs
> in the vertical positivity limiter were invisible in f64 and fatal in f32.

Why single precision changes the answer, not just the speed:

The PPM remap in `fct_lr.py` — the vertical sweep of every scheme, `sl`
included — is exactly *conservative* but not *positive*: it can put −δ in one
cell and +δ in its neighbour with the sum untouched. `coupling.py` then clips
those negatives to zero, which silently **creates** particles (mass has a
budgeted `floor` term; number's never had one). `ADV_VPOS` is the limiter that
stops it.

The negatives are rounding-scale, and that scale is the whole problem. In f64
they are ~1e-16 — indistinguishable from zero, so a broken limiter looks
perfectly healthy. In f32 they are ~1e-7, nine orders of magnitude larger, and
the same bug is fatal. A higher CFL compounds it: material moves further per
substep, gradients are steeper, and the limiter is under more stress.

**What to actually set.** In practice, nothing:

- `validate_vpos_f32.py` already defaults to `ADV_CFL=0.5` and `ADV_F32=1`, so
  prefixing those variables is a no-op. It is written for this test.
- Both entry points bind the same values — `coupling.py` and `driver_fast.py`
  each wrap the selected scheme's `advect_step_batch` in a `functools.partial`
  with `cfl=ADV_CFL` (0.5) and `ADV_F32=1` → float32.

The one place the distinction bites is **calling `fct_lr.advect_step_batch`
directly**, e.g. in a new test. Its signature defaults are `cfl=0.2,
dtype=float64`, which no run uses — so a bare call exercises a configuration the
model is never in, and passes for reasons that do not transfer.

Those signature defaults are deliberately left alone. Changing them would not
touch production (both callers override), and it would break
`test_conservation.py`: that harness asserts Lin-Rood conservation as a
*mathematical* property, which is only visible at f64. Measured 2026-08-12,
flipping the default to f32 moves its drop-in check from `+4.9e-16` to
`-9.9e-08` and the test fails — correctly, since that residual is f32 roundoff
rather than a scheme defect.

**Run from your runs directory, not the repo.** The harnesses that read a
checkpoint (`validate_vpos_f32.py`) take it from the working directory by
default, and `validate_radiation.py` writes its figure there. Point them
elsewhere with `STATE=/path/to/coupled_state_<TAG>_ckpt.npz`. They locate the
repo from their own file location, so invoking them by absolute path from
anywhere works.

## What CI covers

CI runs the three CI tests, an import check of `settling.py` and a lint (ruff,
errors only, `I001` off) on every push and pull request to `main`. GitHub's
runners have no GPU, no submodule checkout and not the ~23 TB CESM archive, so
**no gate or investigation runs there** — the badge means transport
conservation, settling math and the presets' pins are intact, and nothing more.
Everything in the second and third tables stays a manual activity.
