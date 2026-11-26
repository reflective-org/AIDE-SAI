#!/usr/bin/env python3
"""Accuracy of one run against a reference run of the same scenario (e.g. emulated vs numerical).

    python3 scripts/validation/compare_to_reference.py runs/REF runs/RUN [runs/RUN2 ...]

Global numbers from coupled_timeseries_<TAG>.npz (already area-weighted by the model): relative difference to the
reference at each reported day. Fields from frames3d/: area-weighted (cos lat) relative rms difference,
sqrt(sum w (x - x_ref)^2 / sum w x_ref^2), over every cell of the band, at each frame both runs wrote.
"""
import glob, os, sys
import numpy as np

DAYS = (1, 2, 5, 10)
GLOBAL = [('Nburden', 'N burden'), ('Mburden', 'sulfate mass burden'), ('aod550', 'AOD 550 nm'),
          ('reff_nm', 'r_eff'), ('arf_toa_avg', 'TOA forcing (diurnal mean)'), ('H2SO4burden', 'H2SO4 burden'),
          ('SO2burden', 'SO2 burden'), ('dT_rms', 'dT_rad rms')]
FIELDS = ['ntot', 'mtot', 'dg_num_nm', 'deff_wet_nm', 'h2so4', 'dT_rad', 'aod550']


def ts(run):
    return np.load(glob.glob(os.path.join(run, 'coupled_timeseries_*.npz'))[0])


def frames(run):
    return {os.path.basename(f): f for f in glob.glob(os.path.join(run, 'frames3d', 'step_*.npz'))}


ref, runs = sys.argv[1], sys.argv[2:]
R = ts(ref)
for run in runs:
    X = ts(run)
    print(f'\n== {os.path.basename(run)} vs {os.path.basename(ref)}')
    print(f'{"global":28s}' + ''.join(f'{"day " + str(d):>12s}' for d in DAYS))
    for k, name in GLOBAL:
        row = []
        for d in DAYS:
            i = np.where(R['hours'] == 24 * d)[0]
            j = np.where(X['hours'] == 24 * d)[0]
            if len(i) == 0 or len(j) == 0:
                row.append('-'); continue
            r, x = float(R[k][i[0]]), float(X[k][j[0]])
            row.append(f'{100 * (x - r) / abs(r):+.2f} %' if r != 0 else 'ref=0')
        print(f'{name:28s}' + ''.join(f'{c:>12s}' for c in row))
    fr, fx = frames(ref), frames(run)
    common = sorted(set(fr) & set(fx))
    print(f'{"field rel. rms diff":28s}' + ''.join(f'{"h" + str(int(np.load(fr[c])["hour"])):>9s}' for c in common[1:]))
    for k in FIELDS:
        row = []
        for c in common[1:]:
            a, b = np.load(fr[c]), np.load(fx[c])
            if k not in a.files or k not in b.files:
                row.append('-'); continue
            r, x = a[k].astype(np.float64), b[k].astype(np.float64)
            w = np.cos(np.deg2rad(a['lat']))[:, None] * np.ones(r.shape[-1])
            w = np.broadcast_to(w, r.shape)
            den = (w * r ** 2).sum()
            row.append(f'{100 * np.sqrt((w * (x - r) ** 2).sum() / den):.1f}%' if den > 0 else 'ref=0')
        print(f'{k:28s}' + ''.join(f'{c:>9s}' for c in row))
