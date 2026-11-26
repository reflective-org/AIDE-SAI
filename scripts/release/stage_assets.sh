#!/bin/bash
# Maintainers only: build the Hugging Face upload folder for scripts/get_assets.py from the g405 originals, then
# (with UPLOAD=1) push it.   scripts/release/stage_assets.sh STAGE_DIR [HF_REPO]
#   1. symlinks every file under the layout get_assets.py / scripts/presets/_assets.sh expect
#   2. writes STAGE_DIR/SHA256SUMS (README.md, the model card, is kept if present and not hashed)
#   3. UPLOAD=1: hf upload HF_REPO STAGE_DIR . (needs `hf auth login` with a write token)
# Change a source path below = a new release: then update the sha256 list in scripts/get_assets.py to match.
set -eu
H=${1:?usage: stage_assets.sh STAGE_DIR [HF_REPO]}; REPO_ID=${2:-${AIDE_HF_REPO:-reflective-org/AIDE-SAI-emulator}}
mkdir -p "$H"; cd "$H"; find . -type l -delete; rm -f SHA256SUMS
rm -rf tomas_emulator   # microphysics weights live in HF reflective-org/AIDE-Microphysics
mkdir -p radem_gen2 atmos_P2s200 bc_clim
T=/data/tomas-jax-output
for f in sw lw pl; do ln -s $T/radem_gen2_20261119/$f.npz radem_gen2/; ln -s $T/radem_gen2_20261119/${f}_args.json radem_gen2/; done
for f in /data/AIDE-SAI-link-perf/atmos_port/bundle_P2s200/*; do [ "$(basename "$f")" = SHA256SUMS ] || ln -s "$f" atmos_P2s200/; done
ln -s /data/AIDE-SAI-link/runs/bc_clim/bc_clim.npz bc_clim/; ln -s /data/AIDE-SAI-link/runs/bc_clim/build.log bc_clim/
find -L . -type f ! -name SHA256SUMS ! -name README.md | sed 's|^\./||' | sort | xargs sha256sum > SHA256SUMS
echo "staged $(wc -l < SHA256SUMS) files, $(du -shL . | cut -f1) in $H"
[ -e README.md ] || echo "WARNING: no README.md (model card) in $H"
if [ "${UPLOAD:-0}" = 1 ]; then
  hf upload "$REPO_ID" . . --repo-type model --commit-message "assets $(date -u +%Y-%m-%d)"
else
  echo "upload with:  UPLOAD=1 $0 $H $REPO_ID"
fi
