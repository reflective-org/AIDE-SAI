# Emulated vs numerical: speed and accuracy

What the emulators buy and cost, measured head to head on the same scenario. Three runs, 10 days (40 six-hour
steps) from 1997-01-01T06, 10 Tg SO2/yr at 30N/30S, same offline P2@200 winds, same boundaries, one H100 each on a
node shared with other jobs (2026-11-26, commit 9afc43b, `FRAME3D=4`):

| run | microphysics | radiation | launcher |
|---|---|---|---|
| **numerical** (reference) | TOMAS-fast physics, every cell | RRTMGP, window-integrated sun (3 nodes) | `RADEM_CKPT= RAD_REF_INTERP_H=0 scripts/presets/linked_emulator.sh TAG GPU 240 --micro-engine fast` |
| emulated microphysics | cs_b student net + TOMAS physics on the top 0.3 % sulfur cells | RRTMGP, as above | `RADEM_CKPT= RAD_REF_INTERP_H=0 scripts/presets/linked_emulator.sh TAG GPU 240` |
| **emulated** (`linked_emulator`) | cs_b + physics on top 0.3 % | radem gen2 emulator | `scripts/presets/linked_emulator.sh TAG GPU 240` |

Meteorology and transport are the same in all three (P2@200 winds, semi-Lagrangian K=12), so the differences are
the aerosol-side emulators only. Compare any two runs with `scripts/validation/compare_to_reference.py runs/REF runs/RUN`.

## Speed

Seconds per 6-h step, steady state (steps 5-40; stage times from the `[prof]` lines, `steptime.py`):

| stage | numerical | emulated microphysics | emulated | speed-up |
|---|---|---|---|---|
| microphysics | 16.77 | 0.69 | 0.69 | 24x |
| radiation | 4.34 | 4.43 | 0.05 | ~90x |
| transport (SL, K=12) | 0.46 | 0.47 | 0.45 | numerical in all |
| **whole step** | **21.97** | **6.39** | **2.42** | **9.1x** |
| per simulated year (1460 steps) | 8.9 h | 2.6 h | 1.0 h | |
| 10-day run incl. start-up and compile | 958 s | 344 s | 146 s | |

In 3-day runs on 2026-11-20 the emulated step was 2.12 s: whole-step times move with the node's load. What is left in the emulated step is
transport (0.45 s) and the microphysics (0.69 s: the network forward on ~0.9 M cells and the physics on the top
0.3 % cells).

`linked_atmosphere` runs the atmosphere live too: the P2@200 emulator adds 0.29-0.31 s per step (network + physics +
fixer) and 0.06 s regrid/write, offset by reading the winds from memory instead of disk (2026-11-20). For scale,
earlier measurements of the numerical alternative, the JCM dycore it emulates (T63 L95, 12-min step): 0.53 s per
6 h with 25 tracers, 0.34 s dynamics only (H200, 2026-11-12), vs 69 ms for the P2@200 network (fp32, idle H100,
2026-11-10). The atmosphere is not where the coupled model's time goes in either form.

## Accuracy: emulated vs numerical

Global numbers (model's own area-weighted diagnostics), emulated minus numerical, relative:

| quantity | day 1 | day 2 | day 5 | day 10 |
|---|---|---|---|---|
| number burden | -7.0 % | -1.7 % | +1.8 % | +0.9 % |
| sulfate mass burden | +0.01 % | 0.00 % | -0.01 % | -0.01 % |
| AOD 550 nm | +0.03 % | +0.04 % | +0.07 % | +0.09 % |
| effective radius | 0.00 % | -0.06 % | -0.15 % | -0.17 % |
| H2SO4 burden | -0.7 % | -0.3 % | +0.8 % | +1.4 % |
| TOA aerosol forcing, W/m2 (numerical / emulated) | -0.0005 / -0.0005 | -0.0007 / -0.0006 | -0.0026 / -0.0029 | -0.0101 / -0.0099 |
| dT_rad rms | -1.5 % | -0.9 % | -0.3 % | -0.4 % |

The day-1 number burden gap is the first nucleation burst (fewest particles, largest relative error); it is under
2 % from day 2. The forcing is still tiny in the first days (-0.0005 W/m2 on day 1), so its relative error swings
(-18 % at day 5, -0.7 % at day 10); read it in W/m2.

Fields, area-weighted relative rms difference over every cell of the band (1.2-143 hPa):

| field | hour 54 (day 2.25) | hour 126 (day 5.25) | hour 240 (day 10) |
|---|---|---|---|
| number concentration | 3.5 % | 7.6 % | 1.4 % (8-9 % at hours 150-222) |
| sulfate mass | 0.1 % | 0.1 % | 0.1 % |
| number median diameter | 3.3 % | 5.8 % | 7.5 % |
| H2SO4 | 1.1 % | 1.4 % | 1.6 % |
| dT_rad (aerosol heating, accumulated) | 3.9 % | 2.6 % | 2.6 % |
| AOD 550 nm (2-D) | 0.1 % | 0.3 % | 0.4 % |

Per cell at day 10: number concentration median |difference| 6 %, 90th percentile 22 %; number median diameter
6 % / 18 %; wet effective diameter 1 % / 8 %. (The rms of the wet effective diameter is not meaningful: it is set by a
few cells in the top three levels, 1.2-2 hPa, where the wet-growth formula is close to saturation and a tiny
temperature difference changes the wet size up to 50x. AOD is unaffected.)

**Which emulator causes what** (the middle run separates them):

- Radiation emulator alone (emulated vs emulated-microphysics run): dT_rad 2.3-5.0 % rms (5.0 % at hour 30,
  2.3-2.5 % from hour 126); forcing within 1.4 % at day 10; burdens within 0.01 %, aerosol fields within 0.5 % rms
  (number median diameter; number concentration 0.4 %), i.e. the emulated heating barely changes the aerosol in 10 days.
- Microphysics emulator alone (emulated-microphysics vs numerical): all of the aerosol differences above;
  dT_rad 0.7-0.9 % rms.

## Longer horizons and what is not covered

- 10 days, one scenario, one seed. Year-long scores from earlier work: the radiation emulator's dT_rad 0.9 % at
  day 30 and 4.1 % at day 365 vs its physics solve (`docs/RADIATION_EMULATOR.md`); the microphysics student with
  physics on the top 0.3 % vs full physics over 15 days: N -0.8 %, AOD +0.2 %, H2SO4 +1.4 %
  (HF `reflective-org/AIDE-Microphysics` model card).
- The atmosphere emulator is not compared with its dycore here; its response to the aerosol heating
  (`linked_atmosphere`) has not been validated against the dycore.
- Timings are from a node shared with other jobs; stage times are more stable than whole-step times.
