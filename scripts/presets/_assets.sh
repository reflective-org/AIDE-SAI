# Sourced by the linked_* presets: where the weights and input data come from.
#
# Order, per item: an env var you set  >  $ASSETS/<path> (what scripts/get_assets.py downloads)  >  the g405 location.
# ASSETS defaults to <repo>/assets (gitignored). Nothing here changes what a run computes: the files are the same
# bytes wherever they come from (sha256 pinned in scripts/get_assets.py).
ASSETS=${ASSETS:-$REPO/assets}
_pick() { if [ -e "$ASSETS/$1" ]; then echo "$ASSETS/$1"; else echo "$2"; fi; }
G405=/data/AIDE-SAI-link/runs
CLIM_FILE=${CLIM_FILE:-$(_pick bc_clim/bc_clim.npz $G405/bc_clim/bc_clim.npz)}
MET_DIR=${MET_DIR:-$(_pick met_P2s200 $G405/emu_winds_P2s200_1997_1998/met)}
ATM_BUNDLE=${ATM_BUNDLE:-$(_pick atmos_P2s200 /data/AIDE-SAI-link-perf/atmos_port/bundle_P2s200)}
# EMU_CKPT (the full-size BinTF) is never loaded when EMU_STUDENT is set; it only appears in the run header
export EMU_CKPT=${EMU_CKPT:-unused-EMU_STUDENT-is-set}
export EMU_STUDENT=${EMU_STUDENT:-$(_pick tomas_emulator/cs_b/model.pt /data/tomas-jax-output/emulator_h2so4_20261112/cs_b/final.pt)}
export EMU_NORM=${EMU_NORM:-$(_pick tomas_emulator/norm_stats.npz /data/tomas-jax-output/emulator_v1/foundation_mlp180/norm_stats.npz)}
RADEM_DEFAULT=$(_pick radem_gen2 /data/tomas-jax-output/radem_gen2_20261119)
for _f in "$CLIM_FILE" "$EMU_STUDENT" "$EMU_NORM"; do
  [ -e "$_f" ] || { echo "missing input: $_f -- run python3 scripts/get_assets.py first (see README, Quick start)" >&2; exit 1; }
done
