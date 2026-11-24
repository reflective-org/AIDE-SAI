# Documentation index and design rationale

> **`MANIFEST.md` in the repo root is canonical for the current tree**, and
> `../README.md` is the orientation. This page indexes `docs/` and keeps the
> *design rationale* behind the grid, the MAM4 initialization and the open
> boundaries — the parts that have not changed. Where it disagrees with
> MANIFEST, MANIFEST is current.

## The documents

| document | what it answers |
|---|---|
| [CONFIGURATION.md](./CONFIGURATION.md) | every knob and flag, which ones apply to a run, how presets and precedence work |
| [`../configs/README.md`](../configs/README.md) | the linking presets: file format, pins, forbidden keys, adding one |
| [REPO_LAYOUT.md](./REPO_LAYOUT.md) | where every tracked file lives, where a new one goes, how imports work |
| [PROCESSES.md](./PROCESSES.md) | transport (§1: schemes, polar caps, boundaries, the CH4 truth test) and microphysics (§2: the `tomas_jax.fast` engine) in detail |
| [COUPLING_VARIABLES.md](./COUPLING_VARIABLES.md) | every field crossing the coupling seams, its units and which source serves it |
| [BOUNDARY_CONDITIONS.md](./BOUNDARY_CONDITIONS.md) | the boundary source (`BC_SRC=clim`): why a climatology is enough, how it is built, what each boundary does |
| [RADIATION.md](./RADIATION.md) | the RRTMGP driver, anomaly mode, `RAD_T_FEEDBACK`, the reference cache, solar geometry |
| [RADIATION_ARTIFACT_EXPLORATION.md](./RADIATION_ARTIFACT_EXPLORATION.md) | the dusk/dawn heating artifact and how `RAD_SOLAR=window_gq` became the default |
| [TRANSPORT_SL_HANDOFF.md](./TRANSPORT_SL_HANDOFF.md) | the design record of the semi-Lagrangian transport, now the default |
| [VALIDATION.md](./VALIDATION.md) | the tests, gates and investigation harnesses, and the f32 rule |
| [DEFERRED.md](./DEFERRED.md) | things deliberately not done, and why |

## What the model is

TOMAS sectional aerosol microphysics (`models/tomas-jax`, the `tomas_jax.fast`
engine) and RRTMGP radiation (`models/jax-rrtmgp`) on a prescribed
stratospheric circulation. Every shipped linking is one-way throughout: the
winds and temperature are read and never respond, and the radiative heating of
the model's own aerosol reaches nothing but the radiation's diagnostics. The
only loop is the opt-in `RAD_T_FEEDBACK=1`, which no preset sets (see
[Not included](#not-included-the-one-way-scope)).

Two seams decide where the inputs come from: **`MET_SRC`** supplies U, V, OMEGA
and T (and with `jcm` the grid), **`BC_SRC`** every other field — the aerosol
initial condition and boundary reservoirs, the gases, water vapour, the
radiation column. The two presets in `configs/` are the two linkings wired up
today:

- `cesm_numerical` (default): CESM2 hourly h1 meteorology on the native f09 grid
  (192×288); boundaries from a monthly climatology of the same archive.
- `jcm_numerical`: a jcm-strat run's 6-hourly meteorology on its T63 Gaussian grid
  (96×192); the same climatology regridded onto that grid.

```
MET_SRC ──► U, V, OMEGA ──► transport (sl, K=12; fct_lr's vertical sweep)
        └─► T, p        ──► microphysics (tomas_jax.fast, 40 bins), settling, OH rate
BC_SRC  ──► MAM4 modes, SO2, H2SO4 ──► initial condition, top/bottom reservoirs
        ├─► Q            ──► RH over liquid (MET_RH=liq) for settling, water uptake, optics
        └─► Q, O3, CH4, surface fluxes ──► radiation column
```

## State (what is tracked and advected)

Two moments per size bin — no per-species aerosol chemistry:
- `num[bin]` — number mixing ratio `[#/kg air]`
- `mas[bin]` — total dry-mass mixing ratio `[kg/kg air]`

plus two gas tracers, `SO2` and `H2SO4`. 40 TOMAS bins ⇒ 80 aerosol fields + 2
gases = **82 advected 3-D tracer fields** over the stratospheric band, set by
`P_LO_HPA` / `P_HI_HPA` (1–150 hPa through the launcher). On the CESM grid that
band is 24 model levels, 1,327,104 cells.

## Method and assumptions

1. **Grid and pressure.** The model runs on the meteorology's own grid. On the
   CESM grid, level pressures use the reference-`PS` convention
   (`p = hyam·P0 + hybm·PS_REF`, `PS_REF = 1e5 Pa`), constant in time, and
   winds, T and the MAM4 fields share the h1 70-level grid, so no wind/tracer
   level matching is needed. A run's grid is stamped into its checkpoint
   (`GRID_CFG`) and a resume onto a different grid is refused.
2. **Initialization from MAM4.** The modal aerosol `num_a{1,2,3}` (#/kg) and
   `so4_a{1,2,3}` (kg/kg) at the start hour are binned onto the 40-bin grid via
   a per-mode dry log-normal (σ_g = 1.8 / 1.6 / 1.8 for accumulation / Aitken /
   coarse; `INIT_SIGMA` overrides). With `INIT_BIN=so4` (both presets) the
   per-mode mean particle mass is `so4/num`, number is distributed by the
   log-normal CDF in log-mass, and per-bin mass = number × bin geometric-mean
   mass at one fixed density (1770 kg/m³). Number is conserved exactly, mass to
   ~2 %. The alternative `dgnum` (CESM's wet median diameter) conserves number
   but books the coarse mode's whole dust/sea-salt volume as sulfate — 4.3× the
   MAM4 sulfate at t=0 — which is why `so4` has been the default since
   2026-07-29 (`coupling.py`, at `_INIT_BIN`).
3. **Transport.** All 82 tracers advect together each step, batched over the
   tracer axis with shared winds. The default is `ADV_SCHEME=sl`: a
   semi-Lagrangian horizontal step (12 sub-steps per 6 h) with `fct_lr`'s
   flux-form vertical sweep and a mass fixer; Lin-Rood (`lr`) and its
   large-Courant form (`ffsl`) remain selectable. Polar caps (|lat| > 80°) are
   stirred to one well-mixed cell per level, mass-conservingly
   (`ADV_POLAR=zonal`), rather than overwritten from a reservoir, which
   discarded mass. [PROCESSES.md](./PROCESSES.md) §1.
4. **Microphysics.** Each cell is independent: SO2 + OH chemistry, nucleation,
   coagulation and condensation, batched over cells in chunks. Mixing ratios are
   converted to per-volume concentrations with the local air density
   `ρ = p/(Rd·T)` and back. [PROCESSES.md](./PROCESSES.md) §2.
5. **Two-moment consistency.** Because `num` and `mas` advect independently, a
   bin can arrive with `Nk > 0, mass ≈ 0`. Before the microphysics each bin's
   mean particle mass is clipped into `[xk_k, xk_{k+1}]` (empty bins → 0),
   keeping `Dp` finite. The clip is *not* mass-conserving, so it is monitored:
   the timeseries saves cumulative totals (`clipMadd_cum` / `clipMrem_cum`,
   burden-weighted), and the staged budget books them. These should stay ≪ M0 —
   growth means the limiter is decoupling the two moments faster than the
   physics justifies.
6. **Open vertical boundaries.** The top `N_BC_TOP` and bottom `N_BC_BOT` band
   levels are reset every coupling step from the boundary source's MAM4 binned
   onto the TOMAS grid. These reservoirs carry the net effect of all physics
   outside the band (emissions, wet removal, everything below the tropopause),
   making the band a flux-through system rather than a sealed one. Number and
   mass are always pinned as a consistent `(Nk, Mk)` pair from one binning —
   never rescaled separately. **No global mass fixer**: with open boundaries
   the burden legitimately changes, and forcing `M0` would cancel the
   flux-through. [BOUNDARY_CONDITIONS.md](./BOUNDARY_CONDITIONS.md).

## SAI sources and sinks

- **Injection**: continuous SO2 (or H2SO4) release, `INJ_SO2_TG_YR` at
  `INJ_LAT` / `INJ_LON` / `INJ_HPA`; `INJ_ZONAL=1` spreads it around the
  latitude ring, `INJ_MIRROR=1` splits it between ±lat. The launcher default is
  no injection, so a forgotten flag gives an obviously unforced baseline. The
  scenario is stamped into the checkpoint and a mismatched resume is refused.
- **Chemistry and microphysics**: the SO2 → H2SO4 → SO4 chain, with OH from a
  cos(SZA) parabola by default (`OH_SZA=1`).
- **Settling** (`settling.py`): per-bin slip-corrected Stokes velocity on the
  wet droplet, implicit upwind column sweep, **open bottom face** — mass
  crossing the lowest level exits the model (the aerosol's one true sink). The
  staged budget has a `settle` stage equal to −(bottom outflow).

## Running

Production runs go through `../src/run_prod.py --preset NAME`, which verifies
the model pins and execs `driver_fast.py` — see
[CONFIGURATION.md](./CONFIGURATION.md). `python3 src/coupling.py` directly is
the standalone/dev path: it uses the same transport but its microphysics is the
`MICRO_ENGINE=full` chain (tomas-jax `dev`'s `make_step`), which the pinned
`gpu-fast` checkout does not provide, so on this checkout it is useful with
`MICRO=off` (transport only). Either way, run from a directory under `runs/`:
every output path is relative to the working directory.

## Outputs

- `coupled_final_<tag>.npz` — final `num`, `mas`, grid metadata.
- `coupled_timeseries_<tag>.npz` — burdens, budget terms, mean Dp vs time.
- `coupled_frames_<tag>.npz` — probe-level (`PROBE_HPA`, default 50 hPa)
  snapshots every `FRAME_EVERY` hours.
- `coupled_state_<tag>_ckpt.npz` — full 3-D state + cumulative counters,
  written last in each checkpoint block; this is what `RESUME=1` reads.

`scripts/utils/plot_run.py <tag>` turns them into the dashboard, filmstrip and
size-distribution figures.

## Performance

Microphysics and radiation dominate a step; transport under the `sl` default is
about a second. Measured timings of the current engine are in
`../MANIFEST.md`. Cost is linear in cell count, so `N_LEV` (sub-sample the
band to ~N levels) is the vertical lever. Multi-GPU microphysics is **opt-in**
(`FAST_MULTIGPU=1` for the fast engine, `MICRO_GPUS` for `full`, with
`--gpu 0,1,...` exposing the cards); the launcher pins a single card by default,
because on a shared machine using more is a decision to make explicitly.

## Not included (the one-way scope)

- **Aerosol → circulation feedback.** The winds are prescribed and never
  modified, so there is no self-lofting and no dynamical response. By default
  (`RAD_T_FEEDBACK=0`) radiative heating does not feed back either: the
  meteorology's T drives the microphysics, settling, RH, OH and optics, the
  heating rate is a diagnostic (`hr_kday`), and `dT_rad` enters only the band
  temperature of the radiation's own next perturbed solve. No preset changes
  this. Under the opt-in `RAD_T_FEEDBACK=1` the heating feeds back on the
  temperature the microphysics, settling, RH, OH and the next radiation call
  see, so that column loop is closed; the circulation one never is.
- **Aerosol → chemistry feedback.** OH is prescribed (a solar-zenith curve by
  default), so SO2 oxidation neither depletes OH nor sees the aerosol's
  dimming.
- **Wet removal and dry deposition.** Gravitational settling out of the band
  bottom is the only true aerosol sink; material crossing that face is treated
  as removed by the (unmodelled) tropospheric wet scavenging below it.
- **Per-species aerosol chemistry.** All aerosol mass is one sulfate moment per
  bin; no dust, sea salt, organics, ammonium or nitrate are carried.
