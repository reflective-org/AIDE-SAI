# runs/

Raw model output, exactly as a run writes it. One directory per `OUT_TAG`.

Everything here is **disposable and gitignored** (only this README is tracked).
A single 90-day production run writes ~5 GB per checkpoint cycle, so nothing in
this tree should ever be committed.

## Launching a run

`src/run_prod.py` writes every output to the **current working directory** and
refuses to run from anywhere inside the checkout except under `runs/`, so
launch from a tag directory here (or from a directory outside the checkout):

```bash
mkdir -p runs/myrun && cd runs/myrun
../../src/run_prod.py --preset cesm_numerical --out-tag myrun --n-hours 6 --inj-so2 10 2>&1 | tee myrun.log
```

`--preset` picks the linking (`configs/<name>.env`; `cesm_numerical` is the default,
`--list-presets` lists them). Add `--dry-run` to print the resolved environment
and check the preset's submodule pins without launching.

Always pair a scenario with its own `OUT_TAG`: outputs and checkpoints are keyed
by it, so reusing a tag overwrites the other scenario's results. To resume,
re-run with the same preset, tag and scenario plus `--resume 1`; a resume onto a
checkpoint with a different injection scenario or grid is refused.

## What a run writes

| file | contents |
|---|---|
| `coupled_final_<TAG>.npz` | full 3-D end state |
| `coupled_timeseries_<TAG>.npz` | per-step scalar diagnostics and budget terms |
| `coupled_frames_<TAG>.npz` | probe-level frames for the filmstrip |
| `coupled_state_<TAG>_ckpt.npz` | restart checkpoint (`--resume 1` picks it up) |
| `coupled_frames_<TAG>_ckpt.npz`, `coupled_timeseries_<TAG>_ckpt.npz` | checkpoint-cycle copies |

Keep the log with the run. The launcher prints the preset, its submodule pins
and every resolved knob with where its value came from (on stderr, hence the
`2>&1` above), and the model's run header echoes its configuration: together
they are what says which model produced these files.

Figures derived from these runs belong in `outputs/`, not here.
