#!/bin/bash
# linked_aerosolheating: the AIDE-SAI linked model with aerosol heating fed into the MICROPHYSICS temperature
# (RAD_MODE=anomaly). The circulation does NOT respond: the winds are the same offline emulator file as linked_noheating.
#
#   meteorology   JCM emulator P2@200 free run (nudge-off, HL QBO forcing), rolled out ONCE offline and regridded
#                 to a MET_SRC=file directory (scripts/met/emu_to_met.py): u, v, omega, T on the aerosol band.
#                 The circulation never sees the aerosol.
#   microphysics  TOMAS emulator, cs_b student network + TOMAS-fast physics on the top 0.3 % sulfur cells
#                 (MICRO_ENGINE=emulator, EMU_STUDENT). Temperature = the emulator's T + dT_rad, where dT_rad
#                 accumulates the aerosol heating anomaly every step. dT_rad is NOT advected (it stays where it was
#                 deposited) and never reaches the winds. RH = CESM RELHUM climatology (MET_RH=cesm), so it does not
#                 respond to the warming in the microphysics; the radiation's wet optics (RH from Q and T) do.
#   transport     semi-Lagrangian sl_lr, K=12 sub-steps, aerosol band 1.2-143 hPa (open faces, MAM4 reservoir inflow).
#   radiation     RRTMGP, two solves per 6-h step (instant solar geometry): HR(evolved aerosol, T + dT_rad) minus
#                 HR(binned CESM MAM4 background aerosol, T), both on the emulator's T; the difference accumulates
#                 into dT_rad. With MET_SRC=file the reference depends on the emulator T, so it is solved every step
#                 (the CESM-keyed reference cache is bypassed). arf_toa = the aerosol forcing vs that reference.
#   boundaries    CESM monthly climatology (BC_SRC=clim): gases, OH, H2O, surface, radiation column below the band.
#
# usage:  scripts/presets/linked_aerosolheating.sh TAG [GPU=0] [NHOURS=8760] [extra run_prod.py flags]
# Runs in <repo>/runs/TAG (run_prod.py refuses to run elsewhere in the checkout). Scenario defaults: 10 Tg SO2/yr,
# 5 at 30N + 5 at 30S, one cell each at 180E, 55 hPa, all year, from 1997-01-01T06. Override with INJ_SO2=... etc.
# FRAME3D=N (default 1) writes the 3-D frames every N-th step only (state bit-identical; every step's global-mean TOA
# flux and AOD still go to toa_gm.npz); FRAME3D=4 cuts the frame-write cost by ~3/4.
# Data paths: scripts/presets/_assets.sh (assets/ from scripts/get_assets.py, else the g405 locations; env vars override). Checkpoints (sha256 at release):
#   EMU_STUDENT cs_b/final.pt         87115f7d09309f71030eea23b0c719c217cabe4b5ffec799852807f58af89330
#   EMU_CKPT    main_g03/stage_K24.pt 997ab1224ea95121741bd50e906edd1c316ae4a592f2fa776d771ebb711a68fc
set -eu
TAG=${1:?usage: linked_aerosolheating.sh TAG [GPU] [NHOURS] [flags]}; GPU=${2:-0}; NH=${3:-8760}; shift; [ $# -gt 0 ] && shift; [ $# -gt 0 ] && shift
REPO=$(cd "$(dirname "$(readlink -f "$0")")/../.." && pwd)
. "$REPO/scripts/presets/_assets.sh"   # CLIM_FILE, MET_DIR, ATM_BUNDLE, EMU_CKPT/STUDENT/NORM, RADEM_DEFAULT
[ -e "$MET_DIR" ] || { echo "missing MET_DIR=$MET_DIR -- see README, Quick start" >&2; exit 1; }
[ -n "${EMU_CODE:-}" ] && export EMU_CODE   # unset = the vendored src/microphysics/tomas_emulator_code
export EMU_PHYS_TOP=0.003 EMU_PHYS_COARSE=0 EMU_PHYS_BURST=0 EMU_PHYS_NUM=0 EMU_ONLINE_CORR=0 EMU_LOSTFIX=1 EMU_SDPA=1
export EMU_PHYS_EXACT=1   # bit-identical: padding rows of the physics batch run 0 coag substeps (physics -19..-25 %)
export EMU_TRIAGE_FINE=0.3 EMU_TRIAGE_POT=0.1 EMU_TRIAGE_N=1000 DAG_CAPTURE_MAXCALL=0 H0=${H0:-8766}
export XLA_PYTHON_CLIENT_PREALLOCATE=false HDF5_USE_FILE_LOCKING=FALSE
PY=${PY:-python3}
RUN=$REPO/runs/$TAG; mkdir -p "$RUN"; cd "$RUN"
{ echo "preset linked_aerosolheating  repo $(git -C "$REPO" rev-parse HEAD 2>/dev/null || echo 'not a git checkout')$(git -C "$REPO" rev-parse 2>/dev/null && { git -C "$REPO" diff --quiet || echo ' (dirty)'; })"
  echo "MET_DIR=$MET_DIR CLIM_FILE=$CLIM_FILE"; env | grep -E '^(EMU_|H0=|INJ_|FRAME3D)' | sort; } > preset_snapshot.txt
exec $PY "$REPO/src/run_prod.py" --gpu "$GPU" --out-tag "$TAG" --n-hours "$NH" \
  --frame-every 24 --frame-levels all --frame3d "${FRAME3D:-1}" \
  --adv-scheme sl --adv-sl-k 12 \
  --inj-so2 "${INJ_SO2:-10}" --inj-lat "${INJ_LAT:-30}" --inj-mirror 1 --inj-zonal 0 --inj-lon "${INJ_LON:-180}" --inj-hpa "${INJ_HPA:-55}" \
  --met-src file --met-dir "$MET_DIR" --met-rh cesm \
  --bc-src clim --clim-file "$CLIM_FILE" \
  --micro-engine emulator \
  --rad-mode anomaly "$@" > run.log 2>&1
