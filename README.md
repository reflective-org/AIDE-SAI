# AIDE-SAI

[![CI](https://github.com/reflective-org/AIDE-SAI/actions/workflows/ci.yml/badge.svg)](https://github.com/reflective-org/AIDE-SAI/actions/workflows/ci.yml)
[![Python](https://img.shields.io/badge/python-3.10-blue.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/license-Apache%202.0-green.svg)](./LICENSE)

AIDE-SAI is a modular **dycore–aerosol–radiation coupler** for stratospheric
aerosol injection (SAI), in JAX on GPU. Each physical process sits in a
swappable slot (meteorology/dycore, boundary climatology, transport, aerosol
microphysics, settling, radiation), and each slot is filled by a numerical
scheme or, in future, a learned emulator. A **linking** is one choice for every
slot plus the exact submodule commits behind it, written down as a preset file
in [`configs/`](./configs/README.md). This repository ships three linkings, all
fully numerical: CESM2 meteorology on the CESM f09 grid (the default),
jcm-strat meteorology on the JCM T63 grid read from a finished run, and the
same jcm-strat model run live beside the coupler. The end goal is model-agnostic
coupling: other dycores, emulated ones included, and other aerosol schemes
(CARMA, MAM, GLOMAP).

> [!NOTE]
> CI covers what a GPU-less runner without the input data can cover: that the
> tree imports, that advection still conserves mass, that the closed-form
> settling physics is unchanged, and that every preset parses and pins the
> submodule commits the superproject records. The rest of the validation needs
> a GPU and the input data. See [docs/VALIDATION.md](./docs/VALIDATION.md).

> [!WARNING]
> This project is under active development, so it may not behave as expected.
> In particular the model is **not calibrated for absolute radiative forcing**.
> Read the standing caveats in [MANIFEST.md](./MANIFEST.md) before you quote any
> number from it.

## One-way and closed-loop coupling

A linking couples two things to different degrees, and the difference matters
when you read the output.

**The meteorology is one-way.** Winds (U, V, OMEGA) and temperature are *read*
from the meteorology source, and nothing the aerosol does ever changes them.
The circulation therefore never responds: no self-lofting, no change in
Brewer-Dobson ascent, nothing sent back to the dycore. Two shipped linkings
read a finished dycore run (CESM2 history files, or a completed jcm-strat run).
The third, `jcm_online_numerical`, runs the jcm-strat model alongside the
coupler (`MET_SRC=jcm_online`), but it still only sends winds and temperature
one way. Closing this loop needs a dycore that runs alongside the coupler and
takes the aerosol heating as a forcing. The live jcm-strat server is where a
heating term would go back; an emulated dycore behind the `MET_SRC` seam is
the other planned way.

**Aerosol, microphysics and radiation are coupled to each other.** Inside the
prescribed meteorology the model's own aerosol sets the optics, and the optics
heat the layer. Under the presets' setting (`RAD_T_FEEDBACK=0`) that heating is
a diagnostic (`hr_kday`, `dT_rad`): the microphysics, settling, RH and the
optics all see the meteorology's T, and the integrated `dT_rad` enters only the
radiation's emission temperature. Under `RAD_T_FEEDBACK=1` the loop closes: the
heating accumulates into the temperature that the microphysics, settling and
the next radiation call all see. In the band, aerosol is made and removed
inside the model:

```
met source ──► U, V, OMEGA ───────► transport (semi-Lagrangian, sl K=12)
 (MET_SRC) └─► T, p ──────────────► microphysics (TOMAS, 40 bins)
BC_SRC=clim├─► Q ─────────────────► RH over liquid from Q, T, p (MET_RH=liq)
 (monthly  ├─► MAM4 num/so4 ──────► IC, open-BC reservoir, polar caps
 CESM clim)├─► SO2, H2SO4 ────────► gas-phase IC/BC
           └─► O3, CH4, surface ──► radiation column

  SOURCE  continuous SO2 injection    SO2 ─OH─► H2SO4 ─► nucleation/condensation
  SINK    gravitational settling out of the band bottom (no wet removal)
  OPTICS  RRTMGP + Mie on the wet H2SO4/H2O droplet ──► heating, AOD550, ARF_toa

  DEFAULT       aerosol ─► optics ─► heating rate (hr_kday), a diagnostic;
  (open)        microphysics, settling, RH and optics see the met source's T
  CLOSED LOOP   only under RAD_T_FEEDBACK=1:
                aerosol ─► optics ─► heating ─► dT accumulates into T
                    ▲                                    │
                    └──── microphysics + settling ◄───────┘
  NOT CLOSED    U, V, OMEGA are the met source's throughout
```

**State.** 40 TOMAS bins × (number, dry SO4 mass) plus 2 gas tracers = 82
advected 3-D fields, with a 6 h coupling step, on whatever grid the met source
declares. On CESM that is f09, 192 × 288, with 24 levels from 1.2 to 143 hPa.
On JCM it is T63 Gaussian, 96 × 192, with 47 levels from 1.0 to 137.8 hPa. The
band is 1–150 hPa in both cases.

## Slots

Each slot is chosen by an environment variable (the selector). A preset sets
the selectors that differ between linkings. Everything else stays at the
`coupling.py` / `driver_fast.py` default, and both are echoed in the run header.

| slot | selector | implementations in this repo | status / planned |
|---|---|---|---|
| meteorology / dycore | `MET_SRC` (+ the `jcm-strat` slot for `jcm_online`) | `cesm`: CESM2 h1 hourly U/V/OMEGA/T on f09 (module default). `jcm`: a jcm-strat run directory (`JCM_DIR`), 6-hourly, and the model grid comes with it (`src/met/met_jcm.py`). `jcm_online`: the same fields and grid from the jcm-strat AL7h model integrating live in a subprocess (`src/met/met_jcm_online.py` + `jcm_server.py`, code from `models/jcm-strat`). `file`: step-end `u/v/omega/T.npy` frames in `MET_DIR` | `cesm_numerical`, `jcm_numerical` and `jcm_online_numerical` are the presets. `file` is the generic seam for an emulated dycore; nothing in this repo writes it yet |
| boundary climatology | `BC_SRC` (+ `CLIM_FILE`) | `clim`: monthly CESM climatology on the model grid (`src/met/bc_clim.py`). `cesm`: the hourly archive (module default) | All presets use `clim`. The climatology is built by `scripts/met/make_bc_climatology.py` and regridded by `scripts/met/regrid_clim_to_grid.py` |
| aerosol IC/BC source | `AER_SRC` | `mam4`: CESM's MAM4 mode fields binned onto the 40 TOMAS bins (default; this is *not* the MAM4-JAX package). `carma`: a CARMA output file | `mam4` in all presets |
| transport | `ADV_SCHEME` | `sl`: semi-Lagrangian split, `ADV_SL_K=12` (default). `lr`: Lin-Rood flux form at CFL sub-steps. `ffsl`: `lr` at large Courant number. `fast`: advective form | `sl` in all presets. Validate changes at `ADV_F32=1 ADV_CFL=0.5` |
| aerosol microphysics | `MICRO_ENGINE` + the `tomas-jax` slot | `fast`: `tomas_jax.fast`, tomas-jax `gpu-fast` 787c991 (default). `full`: tomas-jax dev's `make_step` | `fast` in all presets. `full` needs a dev checkout, so it needs a preset that has one (see [configs/README.md](./configs/README.md)). Planned: a TOMAS emulator (not part of this repo yet; it would fill this slot behind `MICRO_ENGINE`), then CARMA, MAM, GLOMAP |
| settling | `SETTLE` (`WET_SETTLING`) | closed-form gravitational settling (`src/settling/settling.py`) | On in all presets. It is the model's one true aerosol sink |
| radiation | `RAD_ENGINE` + the `jax-rrtmgp` slot (`RAD=0` turns it off) | `rrtmgp`: RRTMGP + Mie optics, solar geometry `window_gq` at 3 nodes over the trailing 6 h, `RAD_MODE=anomaly` | `rrtmgp` in all presets. Planned: a radiation emulator (not part of this repo yet; it would fill this slot behind `RAD_ENGINE`) |

The seams are deliberately generic. `driver_fast.py` swaps the microphysics
engine into `coupling.py` at import rather than branching inside it, and the met
source declares the grid, so nothing downstream assumes f09. Keep it that way.
A change that makes a slot harder to swap needs saying so in its PR.

## The three linkings

| preset | meteorology | grid | climatology (`CLIM_FILE`) | provenance |
|---|---|---|---|---|
| [`cesm_numerical`](./configs/cesm_numerical.env) (default) | CESM2 h1 hourly U/V/OMEGA/T (`MET_SRC=cesm`) | f09, 192 × 288, 24 levels 1.2–143 hPa | `inputs/bc_clim/bc_clim.npz`, monthly 1996–2014 CESM means | reproduces the 2-year runs `fast_2yr_inj6_30NS` and `fast_2yr_ctrl` (2026-11-18/19) bit-identically with `ADV_SL_RHONORM=0` and the perf switches off (`RAD_FAST=0 MICRO_WRAP_DEVICE=0 MAM4_BIN_DEVICE=0 ADV_SL_DEVICE_RESAMPLE=0 CKPT_ASYNC=0`; the perf switches default on since 2026-11-23, within run-to-run GPU noise; `ADV_SL_RHONORM` defaults to 1 since PR #4 (dev, 2026-11-24), which removes a transport drift those runs carry, see `MANIFEST.md`) |
| [`jcm_numerical`](./configs/jcm_numerical.env) | jcm-strat AL7h70, 6-hourly (`MET_SRC=jcm`) | T63 Gaussian, 96 × 192, 47 levels 1.0–137.8 hPa | `inputs/bc_clim/bc_clim_jcm.npz`, the same climatology regridded onto the JCM grid | reproduces the 2-year run `jcm_1yr_inj6_30NS` (2026-11-18..20) bit-identically with `ADV_SL_RHONORM=0` and the perf switches off (`RAD_FAST=0 MICRO_WRAP_DEVICE=0 MAM4_BIN_DEVICE=0 ADV_SL_DEVICE_RESAMPLE=0 CKPT_ASYNC=0`) |
| [`jcm_online_numerical`](./configs/jcm_online_numerical.env) | jcm-strat AL7h **run live** beside the coupler, 6-hourly (`MET_SRC=jcm_online`; `models/jcm-strat` f3c2070) | as `jcm_numerical` | as `jcm_numerical` | the knob values and pins of the 1-year run `jcm_online_1yr_inj6_30NS` (2026-11-19). **Not bit-reproducible**: jcm-strat on GPU is not deterministic run to run, so this linking is validated statistically (`scripts/validation/validate_jcm_online.py`), never bitwise. That run had `ADV_SL_RHONORM=0` and the perf switches at 0; their defaults (on) are untested on this linking |

All three linkings share the rest of the stack. Microphysics is `tomas_jax.fast`
(`MICRO_ENGINE=fast`, tomas-jax `gpu-fast` 787c991). Transport is
semi-Lagrangian, `sl` K=12. Radiation is RRTMGP + Mie (jax-rrtmgp 99d2d71 plus
`patches/jax-rrtmgp-zenith.patch`), with `RAD_SOLAR=window_gq` at 3 nodes,
trailing, `RAD_MODE=anomaly` and `RAD_T_FEEDBACK=0`. RH is computed over liquid
(`MET_RH=liq`), with `INIT_BIN=so4` and `ADV_VPOS=1`. Every preset also writes the
full 3-D state every 21 steps (`FRAME3D=1 FRAME3D_EVERY=21`, output-only). The JCM linkings take
water vapour from the climatology (`MET_Q=bc`, the default), because jcm-strat
is dry. A dry run of each preset with its runs' scenario resolves to the knob
values those run logs recorded.

**`jcm_online_numerical`** builds jcm-strat from the AL7h70 chain's own resolved
Hydra config, with every term active (Jucker relaxation with the AL7h T_e
correction, QBO nudging, ERA5 nudging below 400 hPa, Hines and Lott-Miller drag,
the sponge), and starts it from the chain's 1 January checkpoint. jcm-strat
needs Python 3.11 and JAX 0.10, and this repo runs JAX 0.6, so the model runs as
a subprocess (`src/met/jcm_server.py`) under its own interpreter on the same
GPU and passes 6-hourly frames over a pipe. The server keeps the year's ERA5
nudging target (~29 GiB) in host memory, so it holds only a few GiB on the card
it shares with the coupler. It imports `jcm_strat` and `jcm` from the
`models/jcm-strat` submodule and refuses to start if they resolve anywhere else.
MANIFEST.md has the design and the validation numbers.

The model hour is CESM's no-leap hour since 1996-01-01 00Z under every preset
(`H0=0`). JCM frames are matched by the same Y-M-D-H.

## Quick start

**Install.** The model repos are git submodules under `models/`, pinned to
exact commits: tomas-jax and jax-rrtmgp (every linking), and jcm-strat with its
own nested jax-gcm submodule (only `jcm_online_numerical`):

```bash
git clone --recursive https://github.com/reflective-org/AIDE-SAI.git
cd AIDE-SAI
# in an existing clone instead:  git submodule update --init --recursive
# (to skip jcm-strat, which only jcm_online_numerical needs:
#  git submodule update --init models/tomas-jax models/jax-rrtmgp)

# REQUIRED, not optional: the zenith patch (see below for why it is a separate step)
git -C models/jax-rrtmgp apply ../../patches/jax-rrtmgp-zenith.patch

pip install -r requirements.txt
pip install --upgrade "jax[cuda12]>=0.6.2"   # GPU wheel -- the CPU one cannot do production
```

**Why the patch is a separate step.** `radiation.py` passes a per-column solar
zenith field, shape `(nlat, nlon, 1)`. Upstream jax-rrtmgp's `sw_cell_source`
assumes a scalar and mis-broadcasts it against the `(nlat, nlon)` TOA flux. A
submodule records only a commit, and this one-hunk fix is in no upstream commit,
so `git submodule update` gives you pristine jax-rrtmgp without it. You don't
have to remember to check, though: every preset pins the patch, and `run_prod.py`
refuses to start if it is not applied. The patch is generated against `99d2d71`.
`git apply` refuses to half-apply, so a failure means the submodule moved and
the patch needs regenerating.

Do **not** activate the tomas-jax `.venv`: its jaxlib is CPU-only and it has no
xarray. The working combination is system `python3` with the GPU jax wheel. To
check the install without launching anything:

```bash
REPO=$PWD
grep -q mu_2d $REPO/models/jax-rrtmgp/rrtmgp/rte/monochromatic_two_stream.py && echo "zenith patch applied"
python3 -c "
import sys; sys.path.insert(0, '$REPO/src')
import _paths, radiation      # both submodules populated (rrtmgp + tomas-jax Mie)
import tomas_jax.fast         # tomas-jax is the gpu-fast commit
import jax; print('deps ok:', jax.devices())
"
```

A pass ends with `deps ok: [CudaDevice(id=0)]`. `[CpuDevice(id=0)]` with a
`cuInit` error means jax cannot find `libcuda.so.1` in this bare shell. That is
harmless here, because `run_prod.py` adds the driver directory itself (see
`--cuda-driver-lib`). Inside a run, though, CPU means months instead of hours.

**For `jcm_online_numerical` only: a separate jcm-strat interpreter.** The JCM
server cannot run in this repo's Python (JAX 0.6); it needs a Python 3.11 venv
with jcm-strat's dependencies (JAX 0.10, dinosaur, hydra). jcm-strat's own
`scripts/bootstrap_env.sh` builds one (`models/jcm-strat/.venv`, the default
`JCM_PY`). The server imports `jcm_strat` and `jcm` from the submodule, not
from that venv, so the venv supplies only the dependencies. The preset sets
`JCM_PY`, `JCM_CACHE` (ERA5 nudging windows and jcm boundary files) and
`JCM_SCRATCH` (JAX compile cache) to this machine's paths; elsewhere, pass
`--jcm-py`, `--jcm-cache` and `--jcm-scratch`. To check the import path on CPU:

```bash
S=$REPO/models/jcm-strat
JAX_PLATFORMS=cpu PYTHONPATH=$S:$S/external/jax-gcm /path/to/jcm-venv/bin/python \
  $REPO/src/met/jcm_server.py --check-imports --jcm-strat $S
```

**Run.** Outputs go to the current directory, so make one directory per run
under `runs/` (gitignored). `run_prod.py` refuses to start anywhere else inside
the checkout. Launch long runs in a detached tmux session, one GPU per run:

```bash
REPO=/abs/path/to/AIDE-SAI
mkdir -p $REPO/runs/cesm_inj6_30NS && cd $REPO/runs/cesm_inj6_30NS

# what would run, without running it (also checks the submodule pins)
python3 $REPO/src/run_prod.py --preset cesm_numerical --dry-run --out-tag cesm_inj6_30NS

# the 2-year runs' scenario: 6 Tg SO2/yr at 30N and 30S, 180E, 55 hPa, two years
tmux new-session -d -s eval_cesm_inj6 -c "$PWD" \
  "HDF5_USE_FILE_LOCKING=FALSE python3 $REPO/src/run_prod.py --preset cesm_numerical --gpu 0 \
     --inj-so2 6 --inj-lat 30 --inj-mirror 1 --inj-zonal 0 --inj-lon 180 --inj-hpa 55 \
     --n-hours 17520 --out-tag cesm_inj6_30NS 2>&1 | tee run.log"
```

Use `--preset jcm_numerical` (or `jcm_online_numerical`) for a JCM linking, with the same scenario flags. The 2-year
runs also set `--frame-levels all`, which writes zonal-mean frames at every
level for the lat-height figures. After a kill or crash, re-run the same
command with `--resume 1`. A resume onto a checkpoint with a different
injection scenario or grid is refused. `run_prod.py --help` lists every knob,
and `--list-presets` lists the presets.

Some notes on the launch:

- **A bare run is a no-injection control.** `--inj-so2` defaults to 0, so a
  forgotten flag gives an obviously unforced baseline rather than a silent SAI
  result.
- **Run length is in hours.** `--n-hours` counts forcing hours from `H0`, and
  the default is 2160 (90 days). One step is 6 h, so 8760 is a year and 17520
  is two years.
- **`OUT_TAG` keys every output and checkpoint**, so give every scenario its own
  tag. Reusing a tag overwrites the other scenario's results.
- **`HDF5_USE_FILE_LOCKING=FALSE`.** netCDF4/HDF5 takes an NFS lock when it
  opens an input file. When the lock manager on this machine wedged
  (2026-10-14), a run sat forever at `opening ...nc`. The inputs are
  read-only, so the lock is safe to turn off.
- **Cost.** `run_prod.py`'s header gives ~22–35 s per 6 h step on one H100, so
  one simulated year (1460 steps) takes ~9–14 h. The first run over a period
  also builds the radiation reference cache (`inputs/cache/rad_ref`, roughly
  3 h per simulated year, once).

`run_prod.py` translates flags to environment variables and nothing else, then
execs `driver_fast.py` from its own tree with the environment final. It must
exec rather than import for two reasons: `LD_LIBRARY_PATH` has to be set before
the process starts, and every knob is read once at import. The order of
precedence is `--flag` > an exported variable > the preset > the launcher's
run-control defaults > `coupling.py`'s defaults. See
[configs/README.md](./configs/README.md) for the preset format and the rules.

## Data requirements

None of the input data is in the repository. The paths below are the defaults
**on the shared H100 machine these runs were made on**. Elsewhere, point the
listed variables or flags at your own copies.

| input | needed by | default on this machine | override |
|---|---|---|---|
| CESM2 h1 hourly archive (FWHIST, f09; ~23 TB as archived) | `cesm_numerical`, at run time for U/V/OMEGA/T only (the climatology serves the rest) | `/data/cesm2.1.5_output/histSST/f.e21.FWHIST.f09_f09_mg17.atmos-scale_fixedSST_1996-2014.001/archive/atm/proc/tseries` | `CESM_DIR`, `CESM_PREFIX`, `CESM_SUF`. Layout: `$CESM_DIR/hour_1/$CESM_PREFIX.h1.<VAR>$CESM_SUF` ([inputs/README.md](./inputs/README.md)) |
| CESM climatology `bc_clim.npz` (~3.2 GB) | `cesm_numerical` | `inputs/bc_clim/bc_clim.npz` (in the checkout, gitignored) | `--clim-file`. Build with `scripts/met/make_bc_climatology.py`, a one-time pass over the archive's monthly stream plus an h1 sub-sample for H2SO4 |
| jcm-strat run directory (AL7h70 chain) | `jcm_numerical`; `jcm_online_numerical` (only its 1 January checkpoints, the last output frame before them, and the 1996 segment's resolved Hydra config) | `/data/JCM_stripped/jcm-strat/runs/p31/p31_AL7h70` (read-only) | `--jcm-dir`, `--jcm-template` |
| jcm-strat interpreter (Python 3.11 venv) | `jcm_online_numerical` | `/data/JCM_stripped/jcm-strat/.venv/bin/python` (set by the preset) | `--jcm-py` |
| jcm-strat caches: `era5/` (regridded ERA5 nudging windows, ~29 GiB per year; fetched on first use), `huggingface/` (jcm boundary files) | `jcm_online_numerical` | `/data/JCM_stripped/jcm-strat/cache` (set by the preset) | `--jcm-cache`; the JAX compile cache is `--jcm-scratch` |
| JCM-grid climatology `bc_clim_jcm.npz` (~1.4 GB) | `jcm_numerical`, `jcm_online_numerical` | `inputs/bc_clim/bc_clim_jcm.npz` (in the checkout, gitignored) | `--clim-file`. Build from `bc_clim.npz` with `scripts/met/regrid_clim_to_grid.py` (command in its docstring, ~20 s). The run refuses a climatology whose grid differs from JCM's |
| RRTMGP k-distributions | all | inside `models/jax-rrtmgp` | (comes with the submodule) |

The JCM linkings open no CESM archive file at run time. CESM enters only
through the climatology. If you pass a `CLIM_FILE` by exporting it rather than
with `--clim-file`, the launcher warns that the export overrides the preset.
That is expected; the warning exists to catch an export left over in a shell
from an earlier run.
Variable lists and units are in
[docs/COUPLING_VARIABLES.md](./docs/COUPLING_VARIABLES.md), and the
climatology's design is in
[docs/BOUNDARY_CONDITIONS.md](./docs/BOUNDARY_CONDITIONS.md).

## Outputs and plotting

Each run writes into its launch directory:

| file | what |
|---|---|
| `coupled_final_<TAG>.npz` | full 3-D prognostic state at the end of the run |
| `coupled_frames_<TAG>.npz` | probe-level snapshots every `FRAME_EVERY` hours (24 by default) |
| `coupled_timeseries_<TAG>.npz` | per-step scalar diagnostics (burdens, AOD550, ARF) and budget terms |
| `coupled_state_<TAG>_ckpt.npz` | restart checkpoint, with `_frames_`/`_timeseries_` twins (`--resume 1` picks it up) |
| `frames3d/step_<s>.npz` | full 3-D state (~480 MB per file on f09). Every preset writes it every 21 steps (`FRAME3D=1 FRAME3D_EVERY=21`, ~33 GB/yr on f09); `--frame3d 0` turns it off |

From the run directory:

```bash
python3 $REPO/scripts/utils/plot_run.py <TAG>      # <TAG>_{dashboard,filmstrip,sizedist,zonal}.png
python3 $REPO/scripts/utils/gif_run.py <TAG>       # <TAG>_dTrad.gif, <TAG>_so4.gif
python3 $REPO/scripts/utils/run_summary.py <TAG>   # per-stage cost from the log's [prof] lines
```

All three read `coupled_*_<TAG>.npz` (or the log) from the **current working
directory** and write beside it. `plot_run.py` takes the grid from the run's
own files, so it works for every linking. Compare burdens across the two grids
only in kg. Never commit `.npz`, `.png` or `.gif`.

Multi-run comparisons live in `scripts/analysis/` (see its
[README](./scripts/analysis/README.md)). `summary_2yr/` turns a 2-year
injection run, its year-1 re-run with full 3-D frames, and a control into
figures and a `metrics.json`. It runs in three steps (`reduce3d.py`,
`reduce_daily.py`, `make_figs.py`), about 3 min on CPU. The figures cover
forcing, AOD, burdens, the sulfur budget, heating, lat-pressure sections, maps
and size distributions. `jcm_vs_cesm_2yr.py` compares the same scenario on
`cesm_numerical` and `jcm_numerical`. Both write to the current directory, so
run them from an analysis directory under `runs/`, passing the run directories
as flags (defaults are `runs/<tag>`). They show how runs differ; scoring
against observations belongs in AIDE-eval.

Before you read any number, know these standing caveats, which are in full in
MANIFEST.md:

- The first ~week is cold-start spin-up.
- There is no wet removal anywhere. Settling and transport out of the band are
  the only sinks.
- `AOD550` and the forcing are uncalibrated in absolute terms.
- The budget line (`sum` vs `M/M0-1`) closes to roundoff. If a change breaks
  that closure, the change is wrong.

## Repository layout

```
configs/        linking presets (cesm_numerical.env = default, jcm_numerical.env,
                jcm_online_numerical.env) -- configs/README.md
src/            everything that advances the coupled state; flat module names on purpose
  run_prod.py     the launcher: preset, pins, knobs -> exec driver_fast.py
  driver_fast.py  production entry point; swaps the microphysics engine into coupling.py
  coupling.py     the model: grid, IC/BC, budget, diagnostics, main loop
  _paths.py       puts each src/ subdirectory on sys.path; REPO_ROOT/MODELS/INPUTS anchors
  met/            met_jcm.py (MET_SRC=jcm), met_jcm_online.py + jcm_server.py
                  (MET_SRC=jcm_online; the server runs in jcm-strat's interpreter),
                  bc_clim.py (BC_SRC=clim)
  advection/      sl_lr.py + sl_core.py (sl), fct_lr.py (lr), ffsl_lr.py (ffsl),
                  fct_fast.py, fct_core.py (legacy; only for test_conservation.py)
  microphysics/   tomas_fast.py, the tomas_jax.fast adapter
  settling/       settling.py
  radiation/      radiation.py (RRTMGP + Mie), rad_cache.py / rad_cache_build.py
models/         git submodules: tomas-jax (gpu-fast 787c991), jax-rrtmgp (99d2d71),
                jcm-strat (dev f3c2070, with external/jax-gcm 35dc199; jcm_online only)
patches/        jax-rrtmgp-zenith.patch, applied after `git submodule update --init`
scripts/        everything that reads a run or builds an input; a run never imports it
  met/            make_bc_climatology.py, regrid_clim_to_grid.py
  radiation/      build_ref_cache.py and the radiation-cache / solar-artifact checks
  analysis/       summary_2yr/ (injection vs control), jcm_vs_cesm_2yr.py (two linkings)
  utils/          plot_run.py, gif_run.py, run_summary.py
  validation/     harnesses; test_conservation, test_physics_math, test_configs run in CI
docs/           process docs (PROCESSES, RADIATION, TRANSPORT_SL_HANDOFF, ...) -- MANIFEST wins
inputs/         small static inputs (H2SO4 refractive indices, Natural Earth coastlines)
                and the radiation cache
runs/           one directory per OUT_TAG (gitignored)
outputs/        derived figures and tables (gitignored)
```

`src/` is not a package. `coupling.py` does bare `import settling` /
`import radiation`, so do not add an `__init__.py` to a subdirectory named like
a module, and do not let an import sorter hoist imports above the `sys.path`
setup ([docs/REPO_LAYOUT.md](./docs/REPO_LAYOUT.md)). Every default, and the
reason for it, is in **[MANIFEST.md](./MANIFEST.md), the canonical reference
for this tree**. Where `docs/` disagrees with it, MANIFEST is current. The full
knob reference is [docs/CONFIGURATION.md](./docs/CONFIGURATION.md).

## Not part of this repo (yet)

This repository holds the three numerical linkings. These are not part of it:

- the TOMAS microphysics emulator (`MICRO_ENGINE=emulator`), the engine A/B
  (`MICRO_REF`) and the training samplers;
- the radiation emulator (`RAD_ENGINE=emulator`);
- the MAM4-JAX driver and its submodule;
- the `MET_SRC=file` converters;
- the `linking_plots` site and the operator studies.

Selecting one of these engines exits with a message that it is not part of
this repo. A component comes into this repository when it fills a slot behind an existing
selector and comes with a preset that uses it.

## Contributing and branch model

`main` is the only long-lived branch. Work happens on short-lived `feat/`,
`fix/` or `docs/` branches, which are merged into `main` by PR and then deleted.

- **A new linking is a new preset**, never a branch. That includes mixed
  numerical/emulated linkings. If it needs a new implementation, put that
  behind an existing selector (`MET_SRC`, `BC_SRC`, `AER_SRC`, `ADV_SCHEME`,
  `MICRO_ENGINE`, `RAD_ENGINE`). If it needs a different commit of a model
  repo, add that commit as a second pinned submodule and point the preset's
  slot at it. [configs/README.md](./configs/README.md) gives the recipe.
- **Changing a preset or a default needs a MANIFEST.md entry**, dated, saying
  why. A default defines what a bare run *means*.
- **A frozen version is an annotated tag**, for example one per published run
  set. To reproduce one: check out the tag, run `git submodule update --init`,
  apply the patch, and run with the preset the run used. The pins make
  `run_prod.py` refuse a checkout that does not match.
- Before you push, run `python3 scripts/validation/test_conservation.py`,
  `test_physics_math.py` and `test_configs.py`. CI runs all three. Validate
  advection changes at `ADV_F32=1 ADV_CFL=0.5`.

[CONTRIBUTING.md](./CONTRIBUTING.md) has the details.

## Citations

The parameterizations this model composes are other people's work:

- Lin & Rood (1996), *Mon. Weather Rev.* 124, 2046 — flux-form advection
- Adams & Seinfeld (2002), *J. Geophys. Res.* 107, 4370 — TOMAS sectional aerosol
- Tabazadeh et al. (1997), *Geophys. Res. Lett.* 24, 1931 — H2SO4/H2O equilibrium composition
- Tang (1997), *J. Geophys. Res.* 102, 1883 — solution density
- Palmer & Williams (1975), *Appl. Opt.* 14, 208 — H2SO4 refractive indices (via HITRAN Aerosols-2016)
- Pincus et al. (2019), *J. Adv. Model. Earth Syst.* 11, 3074 — RRTMGP
- Hanisco et al. (2001), *Geophys. Res. Lett.* 28, 4423 — the diurnal OH curve

The semi-Lagrangian kernels in `src/advection/sl_core.py` are adapted from
dinosaur (Google NeuralGCM, Apache 2.0).

## License

This project is released under the Apache 2.0 License. See the
[LICENSE](./LICENSE) file for details.
