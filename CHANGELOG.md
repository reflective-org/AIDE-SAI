# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Fixed

- **`jcm_online`: no radiation reference cache.** The cache is keyed by
  archive hour, but under `MET_SRC=jcm_online` the anomaly's reference solve
  depends on the run's own live T. The key's archive entry was the constant
  `jcm_online:<chain>`, so a second run on the same cache (a control beside an
  injected run, a rerun) took the first run's reference solves. The cache is
  now off under `jcm_online` (`rad_cache.RefCache` itself refuses a live met
  source) and `build_ref_cache.py` refuses it. A run on a cold cache is
  unchanged; against a warm cache about +1 h per simulated year. The 1-year run
  `jcm_online_1yr_inj6_30NS` was hit for hours 6-24, which left a non-aerosol
  term in its first 2-3 weeks of `dT_rad`, `hr_kday` and ARF (MANIFEST.md).

### Changed

- **Docs: the model is one-way in every shipped linking.** README, CLAUDE.md,
  docs/README.md, RADIATION.md, BOUNDARY_CONDITIONS.md, CONFIGURATION.md,
  MANIFEST and the `--rad-t-feedback` help no longer call aerosol, radiation and
  microphysics interactively coupled. Meteorology → aerosol → radiation, with
  no feedback: `jcm_online`'s protocol sends nothing back, and under
  `RAD_T_FEEDBACK=0` (all presets) `dT_rad` enters only the band temperature of
  the radiation's own perturbed solve (emission and gas absorption, so it is in
  `hr_kday` and ARF, not only the Planck term). The README also lists the
  polar-cap/edge resets, the direct H2SO4 and scheduled SO2 sources, and the
  prescribed OH. No code behaviour changes.
- **sl_lr conserves each level's air mass (`ADV_SL_RHONORM=1`, new default).**
  The per-level mass fixer was imprinting a drift from a level air mass that
  the polar caps and the Jacobian did not conserve: uniform q drifted +2 % at
  1.24 hPa in 10 days. Now within 1e-4. Bare runs are no longer bit-identical
  to the 2-year runs; `ADV_SL_RHONORM=0` reproduces them. MANIFEST.md has the
  numbers.
- `test_configs.py` checks that each preset pins exactly the slots it needs,
  and reads the gitlinks from the index (`git ls-files --stage`) instead of
  `git ls-tree HEAD`; on a clean checkout and in CI the two are the same.
- `cesm_numerical` and `jcm_numerical` are unchanged; their dry runs resolve
  exactly as before. They reproduce their runs bit-identically with the
  performance switches below off (`RAD_FAST=0 MICRO_WRAP_DEVICE=0 MAM4_BIN_DEVICE=0 ADV_SL_DEVICE_RESAMPLE=0 CKPT_ASYNC=0`)
  and `ADV_SL_RHONORM=0` (see above).
- **The five performance switches are on by default** (`RAD_FAST`,
  `MICRO_WRAP_DEVICE`, `MAM4_BIN_DEVICE`, `ADV_SL_DEVICE_RESAMPLE`,
  `CKPT_ASYNC`; added opt-in at 0 on 2026-11-23). In 48 h injected runs on
  both presets, each switch and all five together stayed within the
  unmodified code's own run-to-run GPU noise (<= 2.5e-12 pointwise, budget
  closure identical), for a step wall of 30 -> 18 s (CESM) and 20 -> 12 s
  (JCM). `RAD_FAST` defaults to `auto`: on where it was validated (one GPU,
  wet optics, `RAD_SOLAR=window_gq`, the normal solve), the RAD_FAST=0 chunk
  loop everywhere else (multi-GPU, dry optics, other solar modes, split=True
  calls from `RAD_MODE=flux`, the reference-cache builder and the check
  harnesses), resolved per radiation driver and stated in the run header;
  `RAD_FAST=1` still refuses anything outside the validated path. A default
  run's reference-cache key gains `rad_fast=1`, so it does not hit caches
  built before this change. Setting a switch to 0 runs exactly its old code
  path; all five at 0 (plus `ADV_SL_RHONORM=0`) reproduces earlier runs bit for bit. See `MANIFEST.md`.

### Added

- **Yearly SO2 injection schedule, `INJ_SO2_SCHED_FILE` (`--inj-so2-sched`).**
  A CSV (`year_offset,tg_so2_yr`) gives the total Tg SO2/yr per 365-day model
  year from the run start: constant within a year, the last row held after the
  list ends, same geometry knobs as `INJ_SO2_TG_YR`, with which it is mutually
  exclusive. The row count and a CRC32 of the rates are appended to `inj_cfg`
  (`INJ_SCHED_N`, `INJ_SCHED_CRC`), so a RESUME onto a different schedule is
  refused. Ships `inputs/injection/ccmi2022_sai_30NS_2030-2034.csv`, the
  CCMI-2022 SAI rates for 2030-2034 (Tilmes et al. 2025, Fig. 2b). CI:
  `scripts/validation/test_inj_schedule.py`. Runs without a schedule are
  unchanged except for the two appended `inj_cfg` entries (0, 0).
- `ADV_SL_FIXLOG=1`: logs the level and column mass-fixer corrections per
  sub-step (diagnostic, results unchanged), and
  `scripts/validation/sl_fixer_cap_diag.py` / `sl_fixer_cap_summary.py`, the
  uniform-tracer drift test on CESM winds.
- **A third linking, `configs/jcm_online_numerical.env`: jcm-strat run live as
  the meteorology** (`MET_SRC=jcm_online`). The same fields and grid as
  `jcm_numerical`, but from the jcm-strat AL7h model integrating beside the
  coupler with all its nudging, drag and T_e correction active, started from
  the AL7h70 chain's 1 January checkpoint. `src/met/met_jcm_online.py` (a
  `met_jcm.JcmMet` subclass) runs `src/met/jcm_server.py` as a subprocess
  under jcm-strat's own interpreter (Python 3.11 / JAX 0.10; this repo runs
  JAX 0.6) on the same GPU. The server keeps the year's ERA5 nudging target
  on the host. The preset carries the knob values and pins of the 1-year run
  `jcm_online_1yr_inj6_30NS` (2026-11-19). **Not bit-reproducible**: jcm-strat
  on GPU is not deterministic run to run, so the linking is validated
  statistically (`scripts/validation/validate_jcm_online.py`), never bitwise.
- **A third model slot, `jcm-strat`** (`JCM_STRAT_PATH`): the submodule
  `models/jcm-strat` at f3c2070, with its nested `external/jax-gcm` at
  35dc199. The JCM server imports `jcm_strat` and `jcm` from it and refuses to
  start if they resolve anywhere else (`jcm_server.py --check-imports`). Only
  presets selecting `MET_SRC=jcm_online` pin it (`SLOT_NEEDED_IF` in
  `run_prod.py`).
- Knobs `JCM_TEMPLATE`, `JCM_PY`, `JCM_CACHE`, `JCM_SCRATCH`,
  `JCM_TE_CORRECTION`, `JCM_WORK` and `JCM_STRAT_PATH` (read only under
  `MET_SRC=jcm_online`). `GRID_CFG` met code 3 (a RESUME between `jcm` and
  `jcm_online` is allowed); `PHYS_CFG` gains `MET_SRC=jcm_online` (appended).

- Add `scripts/analysis/`: the 2-year injection-vs-control summary
  (`summary_2yr/`) and the JCM-vs-CESM 2-year comparison
  (`jcm_vs_cesm_2yr.py`), with run directories as arguments, plus Natural
  Earth 1:110m coastlines (public domain) under `inputs/assets/ne_110m/`. On
  the original runs they reproduce the earlier metrics exactly and the figures
  pixel for pixel.

## Initial release — 2026-11-20

AIDE-SAI starts with two numerical linkings of a CESM- or JCM-forced TOMAS
sectional aerosol model with RRTMGP radiation, in JAX on GPU. Its presets
reproduce the 2-year CESM runs (`fast_2yr_inj6_30NS`, `fast_2yr_ctrl`,
2026-11-18/19) and the 2-year JCM run (`jcm_1yr_inj6_30NS`, 2026-11-18..20)
bit-identically. The starting configuration — `sl` K=12 transport,
`MET_RH=liq`, `RAD_T_FEEDBACK=0`, climatology boundaries, `tomas_jax.fast`
microphysics, the JCM grid, `window_gq` solar geometry at 3 nodes over the
trailing 6 h — and the reason for each default are recorded in `MANIFEST.md`.

### Included

- **Linkings are presets.** `configs/cesm_numerical.env` (the default) and
  `configs/jcm_numerical.env` set the meteorology source, the boundary
  climatology and the stack values that differ between linkings (`MET_SRC`,
  `BC_SRC`, `CLIM_FILE`, `INIT_BIN`, `ADV_VPOS`), and pin each model slot with
  `#@submodule SLOT PATH SHA [patch=FILE]` to the exact submodule commit and
  patch the runs used: tomas-jax gpu-fast `787c991`, jax-rrtmgp `99d2d71` +
  `patches/jax-rrtmgp-zenith.patch`. A dry run of each preset with those runs'
  scenario resolves the knob values their logs recorded.
- **`run_prod.py --preset NAME`** (default `cesm_numerical`; `none` = module
  defaults only) and `--list-presets`. The launcher refuses to start if a slot's
  checkout is not at its pinned commit or the patch is not applied
  (`--no-verify-pins` overrides), points `TOMAS_JAX_PATH` / `RRTMGP_PATH` at the
  slot's checkout, warns when an exported variable overrides a preset value, and
  refuses presets that set the scenario, the run control or presence-sensitive
  variables. A relative value of a path knob in a preset (e.g. `CLIM_FILE`) is
  resolved against the repo root. Precedence: `--flag` > exported env > preset >
  launcher run-control defaults > `driver_fast.py`'s `setdefault`s > module
  defaults.
- **Boundary climatologies in the checkout**: the presets read
  `inputs/bc_clim/bc_clim.npz` (CESM f09 grid) and
  `inputs/bc_clim/bc_clim_jcm.npz` (JCM grid). They are gitignored and built by
  `scripts/met/make_bc_climatology.py` and `scripts/met/regrid_clim_to_grid.py`
  (see `inputs/README.md`).
- **`scripts/validation/test_configs.py`**, in CI: every preset parses with the
  launcher's own loader, its pins equal the superproject's gitlinks, the patches
  it names exist, and the default preset exists.
- `models/` — `tomas-jax` and `jax-rrtmgp` as git submodules, so a clone
  reconstructs the exact model code.
- The radiation reference cache (`inputs/cache/`), keyed on the commit of the
  rrtmgp checkout actually in use (`RRTMGP_PATH`).

### Not included (yet)

- The TOMAS microphysics and radiation emulators, the in-loop engine A/B
  harness (`MICRO_REF`) and the training samplers. `MICRO_ENGINE` accepts
  `fast` (default) and `full`, `RAD_ENGINE` accepts `rrtmgp`; selecting
  `MICRO_ENGINE=emulator`, `MICRO_REF`, `MICRO_SAMPLE_EVERY`,
  `RAD_ENGINE=emulator` or `RAD_REF` exits with a message that it is not part
  of this repo. Their `PHYS_CFG` stamps are kept, so checkpoints that carry
  them still resume. An emulator would fill its slot behind the existing
  selector. `RAD_SAMPLE_EVERY` still writes radiation training samples.
- The MAM4-JAX driver. `AER_SRC=mam4`, which bins CESM's MAM4 output onto the
  TOMAS bins, is unrelated and included.
- Converters that write `MET_SRC=file` frames. `MET_SRC=file` itself is
  included as the seam for an emulated dycore.
