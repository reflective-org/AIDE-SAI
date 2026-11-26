#!/bin/bash
# linked_atmosphere: linked_emulator with the atmosphere emulator running LIVE, so the aerosol heating reaches the winds.
#
#   meteorology   JCM emulator P2@200 (src/atmos/atm_live.py, bundle ATM_BUNDLE), stepped 6 h at a time in lockstep with the
#                 aerosol model, same recipe as the offline wind file of the other presets (nudge-off, HL QBO forcing,
#                 massfix, u-fixer, IC 1997-01-01T06, noise seed ATM_SEED). Frames go through an exchange directory
#                 (LIVE_DIR, default /dev/shm) that the coupled run reads with MET_LIVE=1.
#   radiation     RAD_MODE=feedback: Q = INC(evolved aerosol, T) - INC(MAM4 reference, T) at the atmosphere's T (radem gen2,
#                 window_gq@3 trailing; RADEM_CKPT='' = the physics RRTMGP solves). The atmosphere adds Q x 6 h to its own
#                 temperature after each step, so the heating is advected, damped by the emulator's own physics, reaches the
#                 winds, and comes back to the microphysics / settling / radiation through the met T. dT_rad stays 0.
#   microphysics, transport, boundaries: as linked_emulator (cs_b student + TOMAS-fast top 0.3 %, sl K=12, CESM monthly clim).
#
# usage:  scripts/presets/linked_atmosphere.sh TAG [GPU=0] [NHOURS=8760] [extra run_prod.py flags]
# FB_SCALE=0 sends zeros (control of the coupling itself); INJ_SO2=0 = no-injection control on the same atmosphere seed.
# ATM_QMODE=none: the atmosphere ignores Q (then pass e.g. --rad-mode anomaly or flux; the last --rad-mode wins).
# RESUME is not supported (the atmosphere state is not checkpointed): a run goes from the IC in one piece.
# Keeps runs/TAG/{atm_live.log, met_meta.json, atm_zm.npz, met_T.npy, met_u.npy}; the full exchange dir is deleted at the end
# (KEEP_MET=1 moves it to runs/TAG/met instead; ~35 MB per frame).
# Atmosphere bundle (not in the repo; assets/atmos_P2s200 from scripts/get_assets.py, sha256 in assets/SHA256SUMS): net.pt 505ec9a0649caf02390718c08ae15052cadd6727d4f745a97883e01963b3104d
set -eu
TAG=${1:?usage: linked_atmosphere.sh TAG [GPU] [NHOURS] [flags]}; GPU=${2:-0}; NH=${3:-8760}; shift; [ $# -gt 0 ] && shift; [ $# -gt 0 ] && shift
REPO=$(cd "$(dirname "$(readlink -f "$0")")/../.." && pwd)
. "$REPO/scripts/presets/_assets.sh"   # CLIM_FILE, MET_DIR, ATM_BUNDLE, EMU_CKPT/STUDENT/NORM, RADEM_DEFAULT
[ -e "$ATM_BUNDLE" ] || { echo "missing ATM_BUNDLE=$ATM_BUNDLE -- see README, Quick start" >&2; exit 1; }
ATM_SEED=${ATM_SEED:-0}
LIVE_DIR=${LIVE_DIR:-/dev/shm/aide_link_$TAG}
[ -n "${EMU_CODE:-}" ] && export EMU_CODE   # unset = the vendored src/microphysics/tomas_emulator_code
export EMU_PHYS_TOP=0.003 EMU_PHYS_COARSE=0 EMU_PHYS_BURST=0 EMU_PHYS_NUM=0 EMU_ONLINE_CORR=0 EMU_LOSTFIX=1 EMU_SDPA=1
export RADEM_CKPT=${RADEM_CKPT-$RADEM_DEFAULT} RADEM_MUFEAT=1
export RAD_REF_INTERP_H=${RAD_REF_INTERP_H-24}
export EMU_PHYS_EXACT=1
export EMU_TRIAGE_FINE=0.3 EMU_TRIAGE_POT=0.1 EMU_TRIAGE_N=1000 DAG_CAPTURE_MAXCALL=0 H0=${H0:-8766}
export XLA_PYTHON_CLIENT_PREALLOCATE=false HDF5_USE_FILE_LOCKING=FALSE MET_LIVE=1 FB_SCALE=${FB_SCALE:-1}
PY=${PY:-python3}; PY_ATM=${PY_ATM:-$PY}
NF=$((NH / 6 + 1))
RUN=$REPO/runs/$TAG; mkdir -p "$RUN"; cd "$RUN"
rm -rf "$LIVE_DIR"; mkdir -p "$LIVE_DIR"
{ echo "preset linked_atmosphere  repo $(git -C "$REPO" rev-parse HEAD 2>/dev/null || echo 'not a git checkout')$(git -C "$REPO" rev-parse 2>/dev/null && { git -C "$REPO" diff --quiet || echo ' (dirty)'; })"
  echo "ATM_BUNDLE=$ATM_BUNDLE ATM_SEED=$ATM_SEED LIVE_DIR=$LIVE_DIR CLIM_FILE=$CLIM_FILE NFRAME=$NF"
  env | grep -E '^(EMU_|RADEM_|RAD_|H0=|INJ_|FRAME3D|FB_|MET_LIVE)' | sort; } > preset_snapshot.txt
# ---- the atmosphere (same GPU) ----
$PY_ATM -u "$REPO/src/atmos/atm_live.py" --bundle "$ATM_BUNDLE" --xd "$LIVE_DIR" --nframe $NF --t0 1997-01-01T06 --hour0 "$H0" \
  --seed "$ATM_SEED" --q-mode "${ATM_QMODE:-live}" --gpu "$GPU" > atm_live.log 2>&1 &
APID=$!
until [ -e "$LIVE_DIR/meta.json" ] || [ -e "$LIVE_DIR/FAILED" ] || ! kill -0 $APID 2>/dev/null; do sleep 1; done
[ -e "$LIVE_DIR/meta.json" ] || { echo "atmosphere did not start, see $RUN/atm_live.log" >&2; touch "$LIVE_DIR/FAILED"; exit 1; }
# ---- the aerosol model ----
set +e
$PY "$REPO/src/run_prod.py" --gpu "$GPU" --out-tag "$TAG" --n-hours "$NH" --resume 0 \
  --frame-every 24 --frame-levels all --frame3d "${FRAME3D:-4}" \
  --adv-scheme sl --adv-sl-k 12 \
  --inj-so2 "${INJ_SO2:-10}" --inj-lat "${INJ_LAT:-30}" --inj-mirror 1 --inj-zonal 0 --inj-lon "${INJ_LON:-180}" --inj-hpa "${INJ_HPA:-55}" \
  --met-src file --met-dir "$LIVE_DIR" --met-rh cesm \
  --bc-src clim --clim-file "$CLIM_FILE" \
  --micro-engine emulator \
  --rad-mode feedback "$@" > run.log 2>&1
RC=$?
[ $RC -ne 0 ] && echo "coupled run exit $RC" > "$LIVE_DIR/FAILED"
wait $APID; ARC=$?
cp "$LIVE_DIR/meta.json" met_meta.json 2>/dev/null; cp "$LIVE_DIR/atm_zm.npz" "$LIVE_DIR/q_remap.npz" . 2>/dev/null
[ -e "$LIVE_DIR/FAILED" ] && cp "$LIVE_DIR/FAILED" .
if [ "${KEEP_MET:-0}" = 1 ]; then rm -rf met; mv "$LIVE_DIR" met
else cp "$LIVE_DIR/T.npy" met_T.npy; cp "$LIVE_DIR/u.npy" met_u.npy; rm -rf "$LIVE_DIR"; fi
echo "coupled rc $RC, atmosphere rc $ARC" | tee -a preset_snapshot.txt
[ $RC -eq 0 ] && [ $ARC -eq 0 ]
