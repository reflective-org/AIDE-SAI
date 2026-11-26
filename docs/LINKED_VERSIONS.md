# Linked versions

Named, reproducible configurations of the linked model (emulated meteorology +
aerosol microphysics + transport + radiation). Each one is a launcher in
`scripts/presets/` that pins every flag and records the commit, data paths and
checkpoints it ran with in `runs/<TAG>/preset_snapshot.txt`.

| version | branch | launcher | aerosol heating reaches |
|---|---|---|---|
| `linked_noheating` | `linked_noheating` | `scripts/presets/linked_noheating.sh` | nothing (`RAD_MODE=flux`) |
| `linked_aerosolheating` | `linked_aerosolheating` | `scripts/presets/linked_aerosolheating.sh` | the microphysics/settling/optics temperature (`RAD_MODE=anomaly`); not the winds |
| `emulator` | `emulator` | `scripts/presets/linked_aerosolheating.sh` | as `linked_aerosolheating`, plus PR #18's window-integrated sun (`RAD_SOLAR=window_gq`, 3 nodes, trailing) on the fast path; `ADV_SCHEME=sl` K=12 is the code default |
| `linked_emulator` | `linked_emulator` | `scripts/presets/linked_emulator.sh` | as `emulator`, with the window_gq radiation EMULATED (radem gen2, `RADEM_CKPT`): meteorology, microphysics and radiation are all networks; transport stays numerical |
| `linked_atmosphere` | `linked_emulator` | `scripts/presets/linked_atmosphere.sh` | the ATMOSPHERE: the P2@200 emulator runs live beside the aerosol model and adds the aerosol heating to its own temperature (`RAD_MODE=feedback`), so it reaches the winds; first pass, 10 days tested |

## Weights and inputs

`python3 scripts/get_assets.py` downloads everything the launchers need from Hugging Face into `assets/`
(microphysics: `reflective-org/AIDE-Microphysics`, pinned commit; the rest: `AIDE_HF_REPO`, default
`reflective-org/AIDE-SAI-emulator`) and checks each file against the sha256 pinned in the script;
`--met --gpu N` also builds the offline wind file (`assets/met_P2s200`, 92 GB) with `src/atmos/atmos_rollout.py`.
The launchers resolve every path in `scripts/presets/_assets.sh`: an env var you set (`CLIM_FILE`, `MET_DIR`,
`ATM_BUNDLE`, `EMU_CKPT`, `EMU_STUDENT`, `EMU_NORM`, `RADEM_CKPT`), else `assets/`, else the g405 location.
Checked 2026-11-20: `linked_emulator` and `linked_atmosphere` from the staged Hugging Face layout vs the g405
paths, 3 days, WORST 0.000e+00 (and the atmosphere's T and zonal means identical).
The regenerated wind file is not byte-identical to the one the published runs read
(`/data/AIDE-SAI-link/runs/emu_winds_P2s200_1997_1998/met`): frame 0 is equal, frame 1 differs at 1e-7 m/s (the
u-fixer's summation order, `src/atmos/aide_atmos/fixer.py`), 2e-4 m/s by day 2, and it keeps growing as weather does.
Maintainers: `scripts/release/stage_assets.sh` builds the upload folder and, with `UPLOAD=1`, pushes it.

## What every version shares

- **Meteorology.** The JCM emulator P2@200 (T63, 95 levels to 0.01 hPa, dry),
  free-running with no nudging (HL QBO forcing, mass fix, u-fixer), rolled out
  once offline from 1997-01-01T06 and regridded with `scripts/met/emu_to_met.py`
  to a `MET_SRC=file` directory: u, v, omega, T on the aerosol band. The
  circulation never responds to the aerosol in any version listed here.
- **Aerosol band and transport.** 24 levels, 1.2-143 hPa (set by the met
  file's 150 hPa cut). Semi-Lagrangian `sl_lr`, K=12. Open top and bottom
  faces: aerosol leaving through 143 hPa is removed, inflowing air carries the
  CESM MAM4 reservoir (`BC_BOT_AER=1`). Below the tropopause-to-143 hPa gap in
  the extratropics the model has no aerosol; that is a known limitation, not
  tuned here.
- **Microphysics.** `MICRO_ENGINE=emulator` with the cs_b student network
  (`EMU_STUDENT`, packtf d=256) and TOMAS-fast physics on the top 0.3 % sulfur
  cells. The adapter is `src/microphysics/tomas_emulator.py` (+ `student_lib.py`,
  `simple_models.py`); the model code it imports is vendored in
  `src/microphysics/tomas_emulator_code/` (`EMU_CODE=<dir>` selects an outside copy).
- **Boundaries.** CESM monthly climatology (`BC_SRC=clim`): gases, OH, H2O,
  surface, the radiation column below the band, and RH (`MET_RH=cesm`, CESM
  RELHUM; it does not respond to temperature changes).
- **Radiation.** RRTMGP, clear sky, instant solar geometry (the sun of the
  call hour held for the 6-h step).
- **Scenario default.** 10 Tg SO2/yr, 5 at 30N + 5 at 30S, one cell each at
  180E, 55 hPa, all year (`INJ_SO2`, `INJ_LAT`, `INJ_LON`, `INJ_HPA`).

Checkpoints (not in the repo; paths are env-overridable in the launchers):

| env | file | sha256 |
|---|---|---|
| `EMU_STUDENT` | `emulator_h2so4_20261112/cs_b/final.pt` | `87115f7d09309f71030eea23b0c719c217cabe4b5ffec799852807f58af89330` |
| `EMU_CKPT` | `emulator_growth_20261110/main_g03/stage_K24.pt` | `997ab1224ea95121741bd50e906edd1c316ae4a592f2fa776d771ebb711a68fc` |

## `linked_noheating`

`RAD_MODE=flux`: one radiation solve per step at the meteorology's own
temperature. Fluxes, the absolute heating profile and AOD go to `frames3d`;
`dT_rad` stays 0, so the microphysics, settling and optics see the emulator's
unheated temperature. The SAI forcing is a difference taken after the run
against a no-injection control on the same meteorology (`INJ_SO2=0`).

```bash
scripts/presets/linked_noheating.sh myrun 0          # 1 year on GPU 0 -> runs/myrun/
INJ_SO2=0 scripts/presets/linked_noheating.sh myrun_ctl 1
```

## `linked_aerosolheating`

`RAD_MODE=anomaly`: two radiation solves per step, both on the emulator's
temperature: HR(evolved aerosol, T + dT_rad) minus HR(CESM MAM4 background
aerosol re-binned onto the TOMAS grid, T). The difference accumulates into
`dT_rad`, and T + dT_rad is the temperature the microphysics, settling and the
next radiation call see. With `MET_SRC=file` the reference depends on the
emulator's T, so it is solved every step and the CESM-keyed reference cache is
bypassed.

What it does NOT do: `dT_rad` is not advected (it stays in the cell where it
was deposited, so it piles up along the injection longitude) and never
reaches the winds, which are the same offline file as `linked_noheating`.
RH stays the CESM climatology in the microphysics; only the radiation's wet
optics see RH respond to the warming.

```bash
scripts/presets/linked_aerosolheating.sh myrun_heat 0
```

30 days from scratch against `linked_noheating` (same scenario and winds,
2026-11-17): `dT_rad` reaches 5.2 K at the injection cells (zonal-mean max
0.74 K) and is still growing; global N burden -0.19 / -1.35 / -2.75 % and AOD
-0.005 / -0.08 / -0.81 % at days 5 / 15 / 30; mass burden -0.23 %, r_eff
+0.04 % at day 30; TOA forcing +3.0 / +4.0 / +4.6 % over days 0-10 / 10-20 /
20-30. No NaN, budget closes.

Speed: 5.6 s per 6-h step, ~2.3 h per simulated year (two solves 1.5 s, the
per-step MAM4 reference re-binning on the host ~2 s). Faster options that
change results, not applied: re-bin the reference every N steps, or serve it
from a cache built on the emulator's temperatures.

## `emulator`

`linked_aerosolheating` merged with Susanne Baur's PR #18 (`bugfix/radiation-artifact`, d2b95a1): radiation sees the
sun integrated over the 6-h window (3 shortwave solves at Gauss nodes, longwave once) instead of holding the call
hour's sun, which removes the dusk/dawn stripes in dT_rad. window_gq runs on the fast path (one LW, per-node SW on
that node's sunlit columns, g-batched; `RAD_GQ_FAST=0` = the chunk loop). `ADV_SCHEME=sl`, K=12 is now the code
default. Checked against PR #18's own code with the same settings (anomaly mode, 16-step resume): bit-identical
(WORST 0.000e+00), 6.4 vs 20.3 s per 6-h step.

## `linked_emulator`

The `emulator` branch with the radiation emulated as well (`docs/RADIATION_EMULATOR.md`): no RRTMGP call in the
loop. Same anomaly-mode definition, emulated term by term: dT_rad += [INC(evolved, T+dT) + P(T, dT) -
INC(MAM4 reference, T)] x 6 h, INC from the SW net at PR #18's 3 window_gq nodes (trailing window) + the LW net,
P from the damping net. The MAM4 reference is re-binned once a day and interpolated (`RAD_REF_INTERP_H=24`).
Weights `RADEM_CKPT=/data/tomas-jax-output/radem_gen2_20261119` (sha256 in the launcher). Frames every 4th step
by default here (`FRAME3D=4`; `FRAME3D=1` for every step).

```bash
scripts/presets/linked_emulator.sh myrun_emu 0
```

Measured 2026-11-19, one H100, same scenario as above:

| | s per 6-h step | radiation | h per simulated year |
|---|---|---|---|
| `emulator` (physics window_gq anomaly), 10 days from scratch | 6.39 | 4.59 | 2.6 |
| `linked_emulator`, 3-D frames every step | 3.43 | 0.65 | 1.4 |
| `linked_emulator`, `FRAME3D=4` (16-step resume) | 2.64 | 0.70 | 1.1 |
| `linked_emulator`, no 3-D frames (`FRAME3D=0`, 16-step resume) | 2.36 | 0.59 | 0.96 |

With the radiation at 0.65 s, the 3-D frame write (~1.2 s, previously hidden behind the solve) shows up as the
`bc+polar` wait when frames go out every step; `FRAME3D=4` removes most of it. The rest is micro 0.8, advect 0.5.

Accuracy vs the physics (`emulator` code, `RADEM_CKPT` unset), 10 days from scratch, both runs identical except
the radiation: dT_rad rel rms diff 2.3-2.5 % (days 6-10), largest point diff 0.016 K where the physics has 0.35 K,
zonal-mean diff 0.8 % of the peak; forcing vs MAM4 -0.5 % at day 10 (+5-6 % of -0.007..-0.013 W/m2 on days 6-8);
N burden -0.006 %, mass / AOD / r_eff < 0.002 %. Year-long scores of the same checkpoint (centred window, radiation
session) are in docs/RADIATION_EMULATOR.md: dT_rad 0.9 % at day 30, 4.1 % at day 365, drift from the damping net.

Output convention under `RADEM_CKPT`: the TOA fields are relative to the clear sky at T, so frames3d
`toa_sw_dn` is 0 and `toa_net_abs` is the aerosol's net TOA effect, not the absolute flux; `toa_arf` and
`aod550` mean the same as in the physics. With `RADEM_CKPT` unset the code is the `emulator` branch,
bit-identical (16-step resume, WORST 0.000e+00).

## `linked_atmosphere` (first pass, 2026-11-20)

`linked_emulator` with the meteorology run LIVE instead of read from the offline wind file, so the aerosol heating
feeds back into the circulation. Two processes on one GPU, in lockstep through an exchange directory in `/dev/shm`:

- `src/atmos/atm_live.py`: the JCM emulator P2@200 (same recipe and code as the offline file: nudge-off, HL QBO forcing,
  massfix, u-fixer, IC 1997-01-01T06, noise seed `ATM_SEED`), one 6-h step at a time. After each step it adds
  Q x 6 h to its temperature, where Q is the aerosol heating the coupled run diagnosed at the previous frame. The
  weights, IC and fixer terms come from one bundle directory (`ATM_BUNDLE`, 165 MB, not in the repo; sha256 in its
  `SHA256SUMS`). Needs `torch-harmonics` (requirements.txt).
- `RAD_MODE=feedback` in the coupled run: the anomaly mode's two calls at the atmosphere's T with dT_rad held at 0,
  Q = INC(evolved, T) - INC(MAM4 reference, T) (radem; with `RADEM_CKPT=''` the physics window_gq solves), written to
  `q/q_<frame>.npy`. dT_rad stays 0: the heating comes back through the met T, after the atmosphere has advected and
  damped it, and is what the microphysics, settling and the next radiation call see. `MET_LIVE=1` makes every met read
  wait for the frame (`src/met/met_live.py`).
- Q remap f09 band -> emulator grid: conservative in sin(lat)/lon and in pressure (layer overlap), so the column energy
  is conserved (0.99996 on a day-10 field). The 2026-11-16 prototype interpolated in log p and delivered 96.7 %.
- `src/atmos/atmos_rollout.py` regenerates the offline wind file of the other presets from the same bundle.

```bash
scripts/presets/linked_atmosphere.sh myrun_atm 0 2160                  # 90 days on GPU 0 -> runs/myrun_atm/
INJ_SO2=0 scripts/presets/linked_atmosphere.sh myrun_atm_ctl 1 2160    # control on the same atmosphere seed
```

Not supported yet: RESUME (the atmosphere state is not checkpointed, a run goes in one piece); more than the bundle's
one IC; the response of the emulator to imposed heating has NOT been validated against the JCM dycore.

Checks (2026-11-20, H100s, all 2-10 days from scratch):

| check | result |
|---|---|
| `atm_live.py` with Q off vs `atmos_rollout.py`, 2 days | all six met fields bit-identical, 9 frames |
| coupled run on the live atmosphere (Q off) vs the same frames from a finished file, 2 days | WORST 0.000e+00 (8 frames + final state) |
| `RAD_MODE=feedback` with `FB_SCALE=0` vs the atmosphere ignoring Q, 2 days | WORST 0.000e+00 (all but q_kday) |
| injection - control (same seed), 30S-30N at 52 hPa: T carried by the atmosphere / accumulated Q | 0.76 / 0.85 / 1.01 at days 2 / 5 / 10 |
| Q, injection run | zonal-mean max 0.003 / 0.026 / 0.027 K/day at days 1 / 5 / 10 (local max 0.49); control <= 0.001 |
| response at day 10, injection - control | T up to 0.42 K locally (zonal mean 0.05), u up to 0.24 m/s (zonal mean 0.03): one seed, chaos not separated |
| emulated vs physics Q (`RADEM_CKPT=''`), same scenario | 2.5-4.3 % rms after day 1 (12.6 % at frame 2, where Q ~ 0); global mean within 0-5 %; AOD identical |

Speed, 10 days, `FRAME3D=4`, side by side on two GPUs of a node shared with other jobs: 3.65 s per 6-h step live vs
3.78 s for `linked_emulator` reading the offline file (the atmosphere's 0.31 s network + physics + fixer and 0.06 s
regrid/write are offset by reading frames from /dev/shm instead of NFS). With physics radiation (`RADEM_CKPT=''`):
5.9 s. No idle-node timing yet.

## Speed

2026-11-20: the lem2 step speed-ups are merged (commit 7f5fca8; radem feature cache, MAM4 reference on the
device, reference/damping/wet weights prefetched in a worker during advect+micro `RAD_PREFETCH`, async diagnostics
`DIAG_ASYNC`, frame start after the radiation call), for `RAD_MODE=anomaly` and `feedback`. 3 days from scratch vs
2594d7a: both bit-identical (WORST 0.000e+00; for `linked_atmosphere` also the atmosphere's T, u and zonal-mean Q),
radiation stage 0.61-0.63 -> 0.06 s, step ~2.75 -> ~2.1 s (`linked_emulator`) and ~2.9 -> ~2.25 s
(`linked_atmosphere`), FRAME3D=4, node shared with other jobs. Off switches: `RAD_PREFETCH=0`, `RADEM_FAST=0`,
`REF_DEV=0`, `DIAG_ASYNC=0`, `FRAME_BEGIN_EARLY=1`.

The coupled step carries the 2026-11-13 speed-ups, all bit-identical to the
code they replace:

- radiation: one lat chunk, clear-sky optics skipped, SW solved on sunlit
  columns only, g-point batching (`src/radiation/rrtmgp_batched.py`), exact
  minor-gas scan; 2.78 -> 0.77 s per solve. Knobs `RAD_GBATCH`, `RAD_DYN_MINOR`,
  `RAD_SPLIT_TROPO`, `RAD_SKIP_CLOUD`, `RAD_SW_SKIP_NIGHT`, `RAD_LAT_CHUNK`;
  the old chunk loop still serves multi-GPU and dry optics.
- I/O: async `frames3d` writes, device-side wind resample in `sl_lr`.
- MAM4 -> TOMAS binning (`bin_mode`, used every step for the boundary and
  polar-cap refresh and, in `linked_aerosolheating`, the reference aerosol):
  the (41, cells) erf part runs on the device as separate eager ops, ~3x
  faster per mode with the same bits.
- microphysics: fused-qkv SDPA, stream (not device) syncs in the adapter, and
  `EMU_PHYS_EXACT=1` (on in the launchers; `src/microphysics/coag_percell.py`):
  the padding rows of the physics batch run 0 coagulation substeps and the
  Pallas grid is 4096 CTAs, physics -19..-25 % (micro stage 0.90 -> 0.82 s).

Off by default because they change results or outputs:

- `FRAME3D=4` (env of the launchers): 3-D frames every 4th step only; the
  model state is bit-identical and every step's global-mean TOA flux and AOD
  still go to `toa_gm.npz`. Cuts most of the ~0.4 s frame-write cost.
- `EMU_PHYS_PERCELL`, `SM_ATTN=manual`: faster microphysics that changes
  results at the integration-error / rounding level (see the code comments).

Measured on one H100 (16 steps resumed at step 75), `linked_noheating`:
about 3 s per 6-h step, ~1.2-1.3 h per simulated year (2.93-3.21 s on
2026-11-17 with other runs sharing the node's I/O; the boundary reads vary
0.02-0.55 s with that load). The pre-speed-up code took 8-9 s.

## Verification

- `linked_noheating` vs the pre-speed-up code, frame-by-frame over 15 frames +
  final state (16 steps resumed at step 75): every field bit-identical
  (WORST max|diff|/max 0.000e+00); and 30 days from scratch, burdens, AOD,
  r_eff and forcing at days 5/15/30 identical (0.000 %), final state 0.000e+00.
- `linked_aerosolheating`: `EMU_PHYS_EXACT=1` vs off over 30 days from
  scratch, 120 frames + final state bit-identical; the on-device `bin_mode`
  vs the numpy one, 16-step resume bit-identical.
- `EMU_PHYS_EXACT=1` vs off, and the on-device `bin_mode` vs the numpy one:
  bit-identical (16-step resumes, WORST 0.000e+00; `bin_mode` also
  `np.array_equal` offline on 1.3 M cells x 3 mode widths).
