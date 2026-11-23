#!/usr/bin/env python3
"""Compare runs of two or more linkings against a reference run of the same scenario.

A config comparison (CLAUDE.md: allowed here): it shows how the runs DIFFER, which slot
and which process; it does not say which one is right. Typical use: an emulated preset's
run against the numerical preset's run of the same dates, e.g.

  python3 linking_ab.py --ref /path/fast_2yr_inj6_30NS --runs runs/fin_cme_inj6_30NS runs/fin_ce_inj6_30NS \
      --days 30,90,180,365,548,730 --zm-day 365 --out ab_inj

Each directory holds coupled_timeseries_<TAG>.npz (or its _ckpt twin while running) and
coupled_frames_<TAG>.npz; TAG is the directory name. Rows are matched by STEP INDEX (time
since the start), so runs that start at different dates (the P2@200 presets start
1997-01-01T06) are compared at the same lead time and season, not the same date.
Writes <out>.txt (the tables) and <out>.png (relative difference vs the reference over time).
"""
import argparse
import os

import numpy as np

KEYS = ('Nburden', 'Mburden', 'H2SO4burden', 'SO2burden', 'aod550', 'reff_nm',
        'arf_toa_avg', 'dT_rms', 'dT_max', 'settleM_cum')


def load_ts(d):
    tag = os.path.basename(os.path.normpath(d))
    for f in (f'coupled_timeseries_{tag}.npz', f'coupled_timeseries_{tag}_ckpt.npz'):
        p = os.path.join(d, f)
        if os.path.exists(p):
            return tag, np.load(p)
    raise SystemExit(f'no coupled_timeseries_{tag}[_ckpt].npz in {d}')


def load_zm(d, day):
    tag = os.path.basename(os.path.normpath(d))
    for f in (f'coupled_frames_{tag}.npz', f'coupled_frames_{tag}_ckpt.npz'):
        p = os.path.join(d, f)
        if os.path.exists(p):
            z = np.load(p)
            h = z['frames_zm_hours']                              # hours since the run's start (frame 0 = the IC)
            i = int(np.argmin(np.abs(h - day * 24)))
            if abs(h[i] - day * 24) > 24:
                return None
            return dict(num=z['frames_zm_num'][i].sum(0), mas=z['frames_zm_mas'][i].sum(0),
                        plev=z['plev_pa'] / 100, lat=z['lat'], day=h[i] / 24)
    return None


ap = argparse.ArgumentParser()
ap.add_argument('--ref', required=True); ap.add_argument('--runs', nargs='+', required=True)
ap.add_argument('--days', default='30,90,180,365,548,730'); ap.add_argument('--zm-day', type=float, default=365)
ap.add_argument('--out', required=True)
a = ap.parse_args()
rtag, R = load_ts(a.ref)
runs = [load_ts(d) for d in a.runs]
days = [float(x) for x in a.days.split(',')]
lines = []
P = lines.append
P(f'reference: {rtag} ({len(R["hours"])} steps)')
for tag, E in runs:
    n = min(len(E['hours']), len(R['hours']))
    P(f'\n{tag} vs {rtag}: relative difference at lead day (matched by step index; {n} common steps)')
    P('  day ' + ' '.join(f'{k[:11]:>12s}' for k in KEYS))
    for dd in days:
        i = int(round(dd * 4)) - 1
        if i >= n:
            continue
        P(f'{dd:5.0f} ' + ' '.join(f'{(float(E[k][i]) / float(R[k][i]) - 1) * 100:+11.2f}%' for k in KEYS))
zr = load_zm(a.ref, a.zm_day)
if zr is not None:
    w = np.cos(np.deg2rad(zr['lat']))
    av = lambda x: np.average(x, weights=w)
    cols = []
    for d in a.runs:
        z = load_zm(d, a.zm_day)
        if z is not None and z['num'].shape == zr['num'].shape:
            cols.append((os.path.basename(os.path.normpath(d)), z))
    if cols:
        P(f'\nzonal-mean number ratio run/{rtag} by level at lead day {zr["day"]:.0f} (cos-weighted) [mass ratio]')
        P('   hPa  ' + '  '.join(f'{t[:24]:>24s}' for t, _ in cols))
        for k in range(len(zr['plev'])):
            P(f'{zr["plev"][k]:6.1f}  ' + '  '.join(f'{av(z["num"][k]) / av(zr["num"][k]):10.3f} [{av(z["mas"][k]) / av(zr["mas"][k]):6.3f}]      '
                                               for _, z in cols))
txt = '\n'.join(lines)
print(txt)
open(a.out + '.txt', 'w').write(txt + '\n')
try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axs = plt.subplots(2, 4, figsize=(16, 7), sharex=True)
    for ax, k in zip(axs.flat, ('Nburden', 'Mburden', 'H2SO4burden', 'aod550', 'reff_nm', 'arf_toa_avg', 'dT_rms', 'settleM_cum')):
        for tag, E in runs:
            n = min(len(E['hours']), len(R['hours']))
            x = (np.arange(n) + 1) / 4.0
            ax.plot(x, (np.asarray(E[k][:n], float) / np.asarray(R[k][:n], float) - 1) * 100, lw=1, label=tag)
        ax.axhline(0, color='k', lw=0.5); ax.set_title(f'{k}: run/ref - 1 [%]', fontsize=9)
    for ax in axs[-1]:
        ax.set_xlabel('lead day')
    axs[0, 0].legend(fontsize=7)
    fig.suptitle(f'relative to {rtag}')
    fig.tight_layout(); fig.savefig(a.out + '.png', dpi=110)
except ImportError:
    pass
