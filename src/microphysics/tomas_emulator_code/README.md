# tomas_emulator_code (vendored)

The six modules of the tomas-emulator research tree (`/data/tomas-emulator` on g405, not a git repo) that the
linked model imports at run time, copied 2026-11-19 so a fork runs without that tree:
`dataset_v8.py` (NormStatsV8, N_STATE/N_COND), `model_v8.py` (ConvMLPStep, the student nets), `model_routed.py`,
`frontier_common.py`, `frontier_bintf_infer.py` (BinTF loader), `frontier_transformers.py` (BinTF source, exec'd).

Changes vs the originals, path-only: the sibling `frontier_transformers.py` and `sys.path` entry resolve from this
directory, and the normalisation file comes from `EMU_NORM` (default `$EMU_ROOT/foundation_mlp180/norm_stats.npz`).
The training-data loaders (`load_v8_set`, `load_inj`, ...) still point at the original data and are not used by a run.
`EMU_CODE=<dir>` still selects an outside copy. Checked: 16-step resume with this copy vs `/data/tomas-emulator`,
final state bit-identical, and the run opens nothing under `/data/tomas-emulator`.
