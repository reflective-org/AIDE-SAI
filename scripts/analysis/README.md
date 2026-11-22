# scripts/analysis/

Multi-run analyses: each script reads several finished runs in `runs/` and
writes figures and a `metrics.json`. Both scripts compare runs of this model with
each other (injection vs control; one scenario on two linkings). They show how
the output differs, not which run is right. Scoring against observations,
reanalyses or other models belongs in AIDE-eval (see `CLAUDE.md`).

Every script writes to the **current working directory**, so make an analysis
directory under `runs/` and run it from there, invoking the script by absolute
or relative path. Run directories are arguments (or env vars); each holds
`coupled_*_<TAG>.npz` with `TAG` equal to the directory's name. Defaults are this
checkout's `runs/<name>`. CPU only, no JAX.

## `summary_2yr/`: 2-year injection vs control (cesm_numerical)

Inputs (all three on the same preset and the same 1996-1997 CESM meteorology):

| flag | env | default | what |
|---|---|---|---|
| `--inj` | `RUN_INJ` | `runs/fast_2yr_inj6_30NS` | injection run, 2920 steps (year 2 may be a resume) |
| `--inj-y1` | `RUN_INJ_Y1` | `runs/fast_y1frames_inj6` | year 1 of the same scenario re-run with `FRAME3D=1 FRAME_LEVELS=all`, which supplies the year-1 3-D snapshots and daily zonal frames |
| `--ctrl` | `RUN_CTRL` | `runs/fast_2yr_ctrl` | control (no injection), 2920 steps, with `frames3d/` and zonal frames |

It also reads CESM `TROP_P` from the h1 archive (`CESM_DIR` / `CESM_PREFIX`, the
same variables and defaults as `src/coupling.py`) to keep only the levels at or
above the tropopause, and draws coastlines from `inputs/assets/ne_110m/`.

```bash
mkdir -p runs/summary_2yr && cd runs/summary_2yr
S=../../scripts/analysis/summary_2yr
python3 $S/reduce3d.py 12      # frames3d pairs -> cache/snap_<step>.npz   (12 workers, ~40 s)
python3 $S/reduce_daily.py     # daily zonal frames -> cache/daily.npz     (~20 s)
python3 $S/make_figs.py        # img/fig1..fig9, metrics.json              (~2 min)
```

Pass the same `--inj/--inj-y1/--ctrl` to all three. The reduce steps skip any step
already in `cache/` and any step whose pair is not complete yet, so they can be
re-run while runs are still going. `cache/` is ~0.5 GB for 140 snapshot pairs.
Times are for one CPU node with 12 workers.

Products:

- `fig1`: global-mean net TOA forcing (injection − control, and each against the
  MAM4 reference aerosol of `RAD_MODE=anomaly`), added AOD, band burdens.
- `fig2`: the injection run's cumulative SO4 budget by stage (the model's own
  budget terms), and where the injected sulfur is (injection − control).
- `fig3`/`fig4`: Hovmöller and global series of added stratospheric column SO4,
  particle number and aerosol heating.
- `fig5`/`fig5b`: lat-pressure means of the added SO4, particle number and
  heating, 2-year mean and monthly GIF.
- `fig6`/`fig7`: maps of forcing, AOD, column SO4, number and size (2-year mean,
  last 30 days) and an animated map of column SO4 and AOD.
- `fig8`: global-mean stratospheric column size distribution.
- `fig9`: zonal-mean SO4 at the probe (injection) level, daily.
- `metrics.json`: window means (`all`, `y1`, `y2`, `l30`) of the above, budget
  closure and end-of-run sulfur accounting.

## `jcm_vs_cesm_2yr.py`: one scenario on two linkings

The same 2-year injection scenario on `cesm_numerical` and `jcm_numerical`,
plus the CESM control as context. There is no JCM control, so the budgets are
change since day 0 for both runs.

| flag | env | default |
|---|---|---|
| `--cesm-inj` | `RUN_INJ` | `runs/fast_2yr_inj6_30NS` |
| `--cesm-inj-y1` | `RUN_INJ_Y1` | `runs/fast_y1frames_inj6` |
| `--cesm-ctrl` | `RUN_CTRL` | `runs/fast_2yr_ctrl` |
| `--jcm-inj` | `RUN_JCM_INJ` | `runs/jcm_1yr_inj6_30NS` |

```bash
mkdir -p runs/jcm_vs_cesm_2yr && cd runs/jcm_vs_cesm_2yr
python3 ../../scripts/analysis/jcm_vs_cesm_2yr.py      # ~15 s, reads timeseries + zonal frames only
```

Products: `img/fig_forcing_aod_burden.png` (TOA aerosol effect, AOD, SO4/SO2
burdens), `fig_sulfur_budget.png` (where the injected S is at day 365 and 730),
`fig_column_sulfate_hov.png`, `fig_latitude_share.png` (share of the added SO4 in
the tropics, mid-latitudes and polar caps), `fig_size.png` (added size
distributions, effective and mass-median radius), and `metrics.json`.
