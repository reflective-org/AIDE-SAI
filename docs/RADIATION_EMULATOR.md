# Radiation emulator (radem)

`src/radiation/radem.py` replaces the RRTMGP solves of the aerosol-heating modes with
per-column networks trained against Susanne Baur's PR #18 solve: `RAD_SOLAR=window_gq`, 3 Gauss nodes over
the sunlit part of the 6-h window, LW once. Turn it on with `RADEM_CKPT=<dir>` (plus `RADEM_MUFEAT=1` for gen2);
`scripts/presets/linked_emulator.sh` does that. With `RADEM_CKPT` unset the code is the physics, bit-identical.

## What it emulates

Only **differences**, never the full clear-sky heating (several K/day in the stratosphere; a 1 % error on that
would be the size of the whole aerosol signal):

- **INC(aer, T)** = HR(aer, T) - HR(no aer, T), plus d sw_up_toa and d olr. SW net (512 x 4, SiLU) evaluated at
  each window_gq node's cos(zenith) on all columns and combined with the nodes' insolation weights exactly as the
  physics combines its node solves (SW is linear in the irradiance; dark node -> weight 0 -> exactly 0). LW net
  (512 x 4) once. Inputs: the full column T, log q, log o3 (+ log ch4 and surface T for LW; albedo and mu, log mu,
  1/(mu+0.02) for SW) and the aerosol band optics from the physics' own Mie code (log tau, tau-weighted ssa and g).
- **P(T, dT)** = HR(no aer, T+dT) - HR(no aer, T): the clear-sky response to the aerosol-induced warming
  (radiative damping). pl net (1024 x 4), amplitude-normalised (dT / A in, P / A out), so P(T, 0) = 0 exactly.

How the modes use them (`RAD_MODE`):

| mode | dT_rad += 6 h x | physics solves | emulator |
|---|---|---|---|
| `anomaly` (linked_emulator) | INC(evolved, T+dT) + P(T, dT) - INC(MAM4 ref, T) | 2 x window_gq | SW x 3 + LW, twice; pl once |
| `damped` | INC(evolved, T+dT) + P(T, dT) | 2 | INC + P |
| `increment` | INC(evolved, T+dT) (no damping: dT grows without bound) | 2 | INC |

The anomaly identity: HR(aer, T+dT) - HR(ref, T) = INC(aer, T+dT) + P(T, dT) - INC(ref, T), so emulated anomaly
mode is the same quantity the physics anomaly mode computes. In a coupled-atmosphere setup where the atmosphere
does its own damping, only INC is needed.

Not emulated: the aerosol optics (Mie tables, on the device, cheap) and the MAM4 -> TOMAS re-binning of the
anomaly reference aerosol (host, ~2-3.5 s); the latter is amortised by `RAD_REF_INTERP_H=24` (re-bin daily in a
background thread, interpolate linearly in time; 0.001 % rms vs every step).

## Checkpoint

`/data/tomas-jax-output/radem_gen2_20261119/` (not in the repo; 26 MB):

| file | net | sha256 |
|---|---|---|
| `sw.npz` | SW INC, 512 x 4, 150 ep | `e908768ea2415f24a410c551480e347d62614c8e47c0bfedb958804921bed7f9` |
| `lw.npz` | LW INC, 512 x 4, 150 ep | `cf930a380f1e698ea7917619cf805fdd495062299f8a4b6c2daf8d113a089901` |
| `pl.npz` | P, 1024 x 4, 250 ep, amp-norm | `f7f68f7d7a9c25088edcb9cb17fee6ff09c04a2f68ccfe99bd474eb380012fa9` |

Training data (`scripts/radem/label.py`, `label_planck.py`; `train.py`): 600 globes / 7.2 M columns of
window_gq@3 physics labels from linked-model frames (injection x 3 seeds, control, MAM4 reference, damped and
increment trajectories, coupled-atmosphere run), aerosol scaled per column by 0 / 1 / logU[0.3, 3], half the globes
with a smooth T perturbation <= 4 K; days 6-16 of every month held out. P: 240 on-policy globes (columns with
max|dT| > 0.01 K from 7 trajectories). The scripts are the ones that produced gen2, with the paths they ran with;
`scripts/radem/harness.py` is the held-out scorer.

## Accuracy (radiation session, 2026-11-18/19, window centred on the call; vs physics window_gq@3)

- INC, held-out weeks (days 8-14) of every month, injection - control local heating: 0.9-1.9 % per call
  (worst call 3.5 %), 0.5-1.1 % over 8-day means; global forcing within +-0.4 %; no dusk/dawn stripes (<= 0.5 %).
  Unseen coupled-atmosphere seed: 1.2 %, forcing +0.2 %.
- P on run states: 7-11 % of P rms (the weak part; it only matters as dT approaches radiative equilibrium).
- 1-year coupled A/B, emulator vs physics, anomaly mode: dT_rad rel rms 0.9 % at day 30, 4.1 % at day 365;
  zonal-mean max diff 3.6 % of peak; forcing +0.11 % at the end; aerosol burdens / AOD / r_eff < 0.2 %.

The linked_emulator branch runs the window ending at the call hour (`RAD_SOLAR_ALIGN=trailing`, PR #18's final
default). The nets take each node's mu as an input, so the alignment only changes which mu they see; the
trailing-window A/B is in docs/LINKED_VERSIONS.md.
