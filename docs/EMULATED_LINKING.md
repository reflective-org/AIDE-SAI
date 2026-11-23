# Emulated linkings

Every slot that has an emulator runs it; transport (semi-Lagrangian `sl`, K=12) and settling
(closed form) stay numerical. Two presets on the f09 grid:

| preset | meteorology | microphysics | radiation |
|---|---|---|---|
| `cesm_met_emulated` | CESM2 h1 hourly (as `cesm_numerical`) | TOMAS emulator | radem |
| `cesm_emulated` | JCM emulator P2@200 **in the loop** (`MET_SRC=emu`) | TOMAS emulator | radem |

`cesm_met_emulated` differs from `cesm_numerical` only in the two engines, so comparing it with
the `cesm_numerical` runs measures the emulators' error and nothing else. `cesm_emulated` adds the
atmosphere; its difference from `cesm_met_emulated` is the meteorology's (P2@200 emulates the
HL-forced JCM dycore, not CESM, so this is a climate difference, not an error).

And two on the JCM grid (T63 Gaussian 96 × 192, 47 band levels):

| preset | meteorology | microphysics | radiation |
|---|---|---|---|
| `jcm_met_emulated` | jcm-strat AL7h70 output (as `jcm_numerical`) | TOMAS emulator | radem, JCM checkpoint |
| `jcm_emulated` | P2@200 **in the loop, on its own grid** (`MET_SRC=emu`) | TOMAS emulator | radem, JCM checkpoint |

P2@200 runs on the JCM grid natively (same T63 Gaussian latitudes, same 95 hybrid levels as the
AL7h70 chain to float32 precision), so under `jcm_emulated` a frame is the emulator's state with its
levels reversed and the band / radiation-column levels selected: no interpolation. Checked against the
f09-regridded frames at the same time: where the levels coincide (30, 100 hPa) zonal-mean T agrees to
0.04-0.07 K and u to 0.07-0.12 m/s (the regrid's horizontal interpolation). radem's nets take
fixed-length flattened columns (f09: 55-level column, 24 band levels; JCM: 94 and 47), so the JCM grid
has its own checkpoint, trained the same way on JCM-grid states; a checkpoint of the wrong grid is
refused at startup.

## The slots

### Meteorology: `MET_SRC=emu` (`src/met/met_emu.py`)

The JCM emulator P2@200 (T63 Gaussian, 95 levels to 0.01 hPa, dry; prognostic u, v, T, omega,
ps) stepped in the coupled run's own process, one 6 h step whenever the loop first asks for a
frame, and regridded on the GPU onto the f09 band and the CESM radiation column with the
`MET_SRC=file` writer's math. Free-running: no nudging, HL QBO wave drag, global-mean ps fix,
u-fixer (latsm 4, levsm 2), generator noise seed `1000 + ATM_SEED`. Everything comes from one
bundle directory (`ATM_BUNDLE`). The next frame is stepped in a worker thread while the aerosol
step runs (`ATM_PREFETCH`, bit-identical). One-way by default; `ATM_FEEDBACK=1` hands the anomaly
radiation's aerosol heating back to the atmosphere (not validated against the dycore).

* Bit-identical to the offline writer `src/atmos/atmos_rollout.py` for the same bundle and seed
  (`scripts/validation/validate_met_emu.py`, also with JAX x64 on as inside the coupled run;
  the fixer's JAX Hines code runs under `jax.experimental.disable_x64()`).
* Cost: network + physics + fixer 0.42 s per step, regrid 0.02 s, overlapped with the aerosol step.
* The bundle has one IC, 1997-01-01T06 (`--h0 8766`); the CESM-met benchmarks start 1996-01-01, so
  P2-met runs are compared by time since the start (same calendar months).
* Not checkpointed: a RESUME re-integrates the emulator from the IC (deterministic).

### Microphysics: `MICRO_ENGINE=emulator` (`src/microphysics/tomas_emulator.py`)

The TOMAS emulator of the AIDE-SAI-link `linked_emulator` branch (6c5094f) with the mic3 speed-ups
(packed coagulation kernel `coag_pack.py`, one scan segment `EMU_PHYS_SPS=60`, both bit-identical;
the packed block-diagonal Triton attention `SM_ATTN=tri`, rounding-level: 16 steps, N burden
5e-5 relative): the cs_b student network (a PackTF transformer over the 40 bins) advances every
cell one 6 h step in one forward pass; `tomas_jax.fast` runs on the routed cells. Model code
vendored in `src/microphysics/tomas_emulator_code/`; weights outside the repo (`EMU_STUDENT`,
`EMU_CKPT`, `EMU_NORM`).

**The routing is NOT the linked_emulator preset's.** That preset was tuned for speed on
P2@200 winds (plume 0.3 % of cells to the physics, no dense class, a widened triage) and drifts
on CESM meteorology. 30 days of the 6 Tg scenario and its control against `fast_2yr_inj6_30NS` /
`fast_2yr_ctrl` (same dates, same everything else; `runs/m30_*`):

| day 30 vs the numerical run | linked settings, inj | linked settings, **ctrl** | shipped (D), inj | shipped (D), ctrl | + `EMU_PHYS_PTOP=10`, ctrl |
|---|---|---|---|---|---|
| number burden | −2.6 % | +8.5 % (+79 % by day 116) | +0.33 % | +4.5 % | +0.65 % |
| mass burden | +0.02 % | +0.02 % | 0.00 % | 0.00 % | 0.00 % |
| H2SO4(g) | +1.5 % | −1.8 % | −2.8 % | −3.8 % | +0.24 % |
| AOD 550 | +1.3 % | −0.2 % | +0.07 % | +0.04 % | +0.03 % |
| r_eff | −0.3 % | −1.5 % | +0.01 % | −0.03 % | −0.03 % |
| microphysics s/step (GPUs shared) | 0.5 | 0.5 | 3.6 | 1.2 | 7.8 |

What each change fixed (control, number by level at day 30):

* **Triage off (`EMU_TRIAGE=0`).** The quiet step holds a cell's number fixed (no coagulation);
  the physics removes ~0.1 % per step there, so the frozen cells gained +10-25 % at 20-60 hPa by
  day 30 (and the 2-year control 2-7x by day 159). With the network on every cell: 0-1 %.
* **Plume 1 % + dense 4 % (`EMU_PHYS_TOP=0.01 EMU_PHYS_NUM=0.04`)**, the adapter's own accuracy
  defaults (docs of the emulator: dense fine modes are under-coagulated by the network).
* **Online correction (`EMU_ONLINE_CORR=0.01`):** a 1 % physics sample per step, moving-average
  per-(level, latitude band, bin) ratio applied to the network; halves the bias at 3-10 hPa.
* **What is left:** above ~10 hPa the network still gains number (+3 % at 8.6 hPa to +27 % at
  1.2 hPa by day 30 in the control): <1 % of its training set is there. Those levels hold almost
  no mass or optical depth (AOD +0.04 %). `EMU_PHYS_PTOP=10` sends every cell above 10 hPa to
  the physics and removes it (control number +0.65 %), at ~6 s/step: the physics costs ~12 µs per
  cell however it is chunked (`EMU_PHYS_CHUNKS` does not help: the cost is TOMAS's per-cell base
  cost, not the stiffest cell). Routing only above 3 hPa does not help: the bias is made at
  3-10 hPa and transported up.
* RH is not a factor: the band's RH is ~0.003 over liquid and ~0.004 over ice, both inside the
  network's training range; one-step errors are the same with either.

### Radiation: `RAD_ENGINE=radem` (`src/radiation/radem.py`)

The window_gq aerosol-increment emulator of AIDE-SAI-link (`docs/RADIATION_EMULATOR.md` there), in
two parts, all per-column MLPs; no RRTMGP call in the loop:

* **INC(aer, T) = HR(aer, T) − HR(no aer, T)** and the TOA terms: an SW net (512 × 4, evaluated at
  each window_gq Gauss node's cos(zenith) and combined with the node weights exactly as the physics
  combines its node solves) and an LW net (512 × 4), on the column T, log q, log o3 (+ log ch4,
  surface T for LW; albedo, mu, log mu, 1/(mu+0.02) for SW) and the aerosol band optics from the
  physics' own Mie code.
* **P(T, dT) = HR(no aer, T+dT) − HR(no aer, T)**, the clear-sky (Planck) damping: one net
  (1024 × 4), amplitude-normalised so P(T, 0) = 0 exactly.

`RAD_MODE=anomaly` (the only mode it emulates here): dT_rad += 6 h × [INC(evolved, T+dT) + P(T, dT) −
INC(MAM4 reference, T)], the physics anomaly term by term. Under AIDE-SAI's `RAD_T_FEEDBACK=0` the
evolved call's wet optics are built at T_met (only its emission sees T_met + dT_rad), as in the
physics call. The MAM4 reference is re-binned daily in a worker thread and interpolated
(`RAD_REF_INTERP_H=24`, 0.001 % rms); with the meteorology on `MET_SRC=file|emu` and `BC_SRC=clim`
the reference increment, P and the evolved call's wet weights are computed in a worker during
advection + microphysics (bit-identical to inline).

**Checkpoint gen3 (f09), not gen2.** gen2 (`radem_gen2_20261119`, trained on P2@200-met states of
the linked campaign) is out of distribution on CESM meteorology: in the CESM-met run its LW
increment net put 3-98 K/day into thin nucleation-burst columns (AOD 0.006, ~1e11 particles/kg in
bins 0-6) at the plume's southern edge, where the physics gives 0.01 K/day; 30 of the first 400
steps had heating spikes > 2 K/day and dT_max reached 27 K against 3.4 K. Anchoring at zero aerosol
does not help (the nets already give exactly 0 there). On P2 met (in distribution) gen2 had none.
gen3 = gen2's recipe (`scripts/radiation/radem/`, ported grid-generic) on gen2's 600 P2-met globes
plus 450 globes on this repo's CESM-met states (the `fast_y1frames_inj6` numerical frames, with and
without their own dT_rad, and MAM4 references; aerosol x logU[0.3, 3] per column, smooth T
perturbations, days 6-16 of every month held out), and the P net on gen2's set plus 360 CESM
globes. `/mnt/nvme_stage/aide_sai_emulated/radem_gen3_f09` (sha256 in its `SHA256SUMS`).

Offline, against RRTMGP window_gq@3 on the numerical CESM-met states (`fast_y1frames_inj6`; days
10.5, 126 and 252 are held-out days):

| | gen2 | gen3 |
|---|---|---|
| INC, all cells, rel. rms error | 4-390 % (max 13.4 K/day) | 0.8-1.8 % (max 0.014 K/day) |
| INC, plume columns (AOD ≥ 0.01) | 2-6 % | 0.7-1.3 % |
| global d(TOA) | 0.1-5 % | 0.1 % |
| P, rel. rms error | 45-66 % | 2.6-3.9 % |
| P/dT median, cells with abs(dT) > 0.5 K, days 26 / 31.5 / 126 / 252 (physics -0.071 / -0.080 / -0.054 / -0.045 per day) | -0.081 / -0.093 / not run / not run | -0.072 / -0.080 / -0.053 / -0.045 |
| anomaly heating, rel. rms error | 27-540 % | 2.4-6.7 % |

The gen2 P net over-damps on CESM states (P/dT -0.093 against -0.080): in a 30-day in-loop A/B
against RRTMGP with the same microphysics (`r30_inj_gen3a` vs `r30_inj_phys`, gen3 INC + gen2 P),
dT_rad rms -3.5 %, dT_max -16 %, the 24 h-mean ARF 4.5 % strong at day 30; AOD agrees to 3e-5 (the
aerosol does not see the radiation under `RAD_T_FEEDBACK=0`), and the heating-rate extremes match
(max 1.18 vs 1.15 K/day). Hence gen3's P.

The nets take fixed-length flattened columns (f09: 55-level column, 24 band levels), so the JCM
grid (94-level column, 47 band levels) gets its own checkpoint, trained the same way on JCM-grid
states (`jcm_met_emulated` frames, the `jcm_1yr_inj6_30NS` final state, MAM4 references).

## Speed

One coupled 6 h step, mature plume (resumed at day 119 of the 6 Tg run, 24 steps; one GPU per run on
a node shared with other runs; `runs/tim_*`):

| preset | s / step | micro | advect | radiation | read | h per simulated year |
|---|---|---|---|---|---|---|
| `cesm_numerical` (its own 2-year log) | ≈ 31 | 16.3 | 1.2 | 13.8 | 0.3 | 12.2 |
| `cesm_met_emulated` | 3.6 | 2.2 | 0.49 | 0.32 (CESM reads in the radem column) | 0.20 | 1.5 |
| `cesm_emulated` (P2@200 in the loop) | 3.1 | 2.0 | 0.50 | 0.06 | 0.12 | 1.3 |

Bit-identical step speed-ups ported from AIDE-SAI-link (checked by 16-step A/B, every timeseries
entry and every final-state array identical): the on-device wind resample in `sl_lr` (and an
`np.asarray` that pulled an 870 MB stack to the host only to read its shape; advect 1.04 → 0.45 s),
the boundary reads in a worker thread (`IO_BC_PREFETCH`, 0.6 → 0.02 s), the atmosphere's next frame in a
worker thread (`ATM_PREFETCH`), the radiation reference/damping/wet weights in a worker
(`RAD_PREFETCH`). What is left is the microphysics' physics share (plume 1 % + dense 4 % + 1 %
sample of the cells through `tomas_jax.fast`, ~12 µs per cell; `EMU_PHYS_CHUNKS=4` sorts them so
the stiff plume no longer sets every routed cell's substep count: 5.5-6.2 → 2.2-2.4 s on a
day-119 stiff step). The linked_emulator speed routing (0.5 s micro) is faster and drifts
(above). A RESUME is not bit-exact: number 1-5e-3 after 24 steps (the online correction's moving
sums are not checkpointed, plus the model's own resume difference); the in-loop atmosphere is
re-integrated from its IC and lands on the same trajectory (SO2 burden 1e-5).

## Benchmarks

The numerical runs' scenario: 6 Tg SO2/yr, half at 30N and half at 30S, one cell each at 180E,
55 hPa, two years, and its no-injection control (`fast_2yr_inj6_30NS` / `fast_2yr_ctrl` on CESM met,
`jcm_1yr_inj6_30NS` extended to two years on JCM met). Emulated runs `runs/fin_*` (2026-11-23),
compared by step index (= lead time; the P2@200 presets start 1997-01-01T06, the others 1996-01-01)
with `scripts/analysis/linking_ab.py`.

**`cesm_met_emulated` vs `cesm_numerical`** (the emulators' error; same dates, same meteorology):

| lead day | N burden | mass | H2SO4(g) | AOD | r_eff | ARF (24 h) | dT_rad rms | settled |
|---|---|---|---|---|---|---|---|---|
| inj 90 | −0.2 % | 0.0 % | +0.1 % | +0.2 % | 0.0 % | +1.1 % | +0.8 % | −0.6 % |
| inj 365 | −0.7 % | +0.3 % | −0.3 % | +1.7 % | −0.2 % | +1.7 % | +2.6 % | −1.7 % |
| inj 730 | −0.5 % | +1.2 % | +4.0 % | +2.8 % | +0.1 % | +3.2 % | +4.2 % | −2.6 % |
| ctrl 365 | +0.5 % | +0.6 % | +0.4 % | +0.8 % | −1.0 % | +7.8 % | −0.2 % | −0.5 % |
| ctrl 730 | +0.9 % | +1.2 % | −0.4 % | +1.6 % | −1.1 % | +9.1 % | −1.3 % | −0.5 % |

(The control's ARF is a small number, the background against the MAM4 reference.) The same
comparison for the linked_emulator settings as ported (`cme_2yr_*`, radem gen2): injection AOD +17 %,
H2SO4 −45 %, r_eff −19 %, ARF +21 % at day 730; control number +131 % and its radiation NaN by day 365.

**The SAI signal (injection − control)**, means over days 335-365 and 700-730:

| | ΔAOD y1 | ΔAOD y2 | ΔARF y1 (W/m²) | ΔARF y2 |
|---|---|---|---|---|
| `cesm_numerical` | 0.0653 | 0.0758 | −2.03 | −2.42 |
| `cesm_met_emulated` | 0.0662 (+1.4 %) | 0.0779 (+2.8 %) | −2.06 (+1.8 %) | −2.50 (+3.5 %) |
| `cesm_emulated` (P2@200 met) | 0.0667 (+2.1 %) | 0.0937 (+24 %) | −2.08 (+2.6 %) | −2.94 (+22 %) |
| linked settings, as ported | 0.0694 (+6 %) | 0.0871 (+15 %) | NaN | NaN |

**`cesm_emulated`** differs from the numerical run mostly by its meteorology: number −56 to −74 %,
AOD +3 to +23 %, r_eff up to +28 % in the injection run (the single-cell plumes are dispersed by a
different circulation; P2@200 emulates the HL-forced JCM dycore, not CESM). The year-1 SAI signal is
nonetheless within 2-3 %. **`jcm_met_emulated` vs `jcm_numerical`** (injection): number −0.5 / −0.9 %,
mass +0.4 / +1.0 %, AOD +1.3 / +2.7 %, r_eff −0.1 / +0.2 %, ARF +2.4 / +1.1 % at days 365 / 730.

**Radiation-emulator failures seen in the 2-year runs** (none reaches the aerosol under
`RAD_T_FEEDBACK=0`; the forcing tables above are unaffected except where marked):

* f09 gen3, CESM met: 10 steps with heating spikes > 3 K/day (max 14 K/day), all in year 2
  (days 370-600): gen3's CESM labels are year-1 states.
* f09 gen3, P2 met: one spike episode (days 324-331, up to 38 K/day) in the injection run; the
  control's radiation went **NaN at day 519.25** with dT_rad ~0.03 K (an abrupt non-finite, not a
  growth; the meteorology was finite: checked by re-integrating the atmosphere). Not reproduced
  yet. coupling.py now stops a radem run with the step and cell on a non-finite heating.
* JCM gen3b: spike clusters from day 95 (up to 132 K/day) with AL7h70 met; with P2 met the
  control's dT_rad wanders to ±20 K and the injection run's diverges after day ~548. The JCM
  checkpoint has no P2-met states and few late-plume states: not production-ready.

The fix is the same in each case: label the emulated runs' own later states (write `frames3d`
every ~41 steps in the 2-year runs) and retrain (one more radem round, ~2 GPU-h of labels plus
~1 h of training per grid).

## Open items

* **An emulator of the linking's own meteorology.** P2@200 emulates the HL-forced p20bt JCM dycore,
  not CESM and not the nudged AL7h70 chain, so `cesm_emulated` / `jcm_emulated` differ from the
  numerical runs mostly by their meteorology (see the benchmark), not by the emulators. The AL7h70
  retrain (M3 in the AL7h70 campaign, `/mnt/nvme_stage2/gen/al/M3`, 8/8 initial conditions alive at
  400 days nudge-off at step 500) emulates exactly `jcm_numerical`'s meteorology on the JCM grid; it
  needs a bundle (its weights, the AL7h torch physics and analytic tables, a 1996-01-01T06 IC from
  the AL7h70 archive) in `src/atmos/` and a bit-identity check against its own free-run tool before
  it can sit behind `MET_SRC=emu` (native mode needs no other change).
* **The microphysics emulator above ~10 hPa** (`EMU_PHYS_PTOP`, above) and a cheaper physics
  for those clean cells.
* **More JCM radiation labels.** The JCM checkpoint has 750 globes and no P2-met base set; its
  in-run anomaly error is 6-11 % against 2.4-6.7 % on f09.
* **Hosting the weights** (all are absolute paths on this machine) and checkpointing the online
  correction's moving sums (RESUME).
* `ATM_FEEDBACK=1` (the aerosol heating handed to the in-loop atmosphere) is implemented and not
  validated: the emulator's response to imposed heating has not been compared with the dycore's.
