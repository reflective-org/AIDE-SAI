"""Reduce every frames3d snapshot pair (injection, control at the same step) to maps and
global numbers. One cache file per step: cache/snap_<s>.npz. Skips steps already done and
steps whose pair is not complete yet, so it can be re-run while the runs are going.

  python3 reduce3d.py [NWORKERS] [--inj DIR] [--inj-y1 DIR] [--ctrl DIR]

Reads frames3d/step_*.npz of the injection (year-1 re-run + year 2) and control runs and
CESM TROP_P from the h1 archive (CESM_DIR); writes cache/ in the current directory.
"""
import argparse
import os
import time
from multiprocessing import Pool

import numpy as np

from lib import CACHE, G, RHO_AER, add_run_args, configure, geo, snap_files, strat_mask

KEYS3 = ('num', 'mas', 'hr_kday')
KEYS2 = ('aod550', 'toa_arf')


def one_run(path, smask, g):
    z = np.load(path)
    w = (g['dp'][:, None, None] / G) * smask                 # kg air / m2 per layer, strat only
    num = z['num']; mas = z['mas']                           # (40, nlev, nlat, nlon), #/kg and kg/kg
    Nk = np.einsum('klyx,lyx->kyx', num, w, dtype=np.float64)   # #/m2 per bin
    Mk = np.einsum('klyx,lyx->kyx', mas, w, dtype=np.float64)   # kg/m2 per bin
    colN = Nk.sum(0); colM = Mk.sum(0)
    area = g['area']; A = area.sum()
    dpb = g['dp_bin'].astype(np.float64)
    # column size statistics (per-bin column-mean particle mass -> diameter)
    Dk = np.where(Nk > 0, np.cbrt(np.maximum(Mk, 0) / np.maximum(Nk, 1e-300) / RHO_AER * 6 / np.pi) * 1e9,
                  dpb[:, None, None])
    lnD = np.log(Dk)
    dmass = np.exp((Mk * lnD).sum(0) / np.maximum(colM, 1e-300))          # mass-weighted diameter, nm
    d = dpb[:, None, None]
    deff_dry = (Nk * d ** 3).sum(0) / np.maximum((Nk * d ** 2).sum(0), 1e-300)
    # heating: per-cell anomaly hr (evolved - MAM4 reference), K/day
    hr = z['hr_kday'].astype(np.float64)
    m3 = w * area[None]                                      # kg air per cell (strat)
    hr_glob = float((hr * m3).sum() / m3.sum())
    zm_w = m3.sum(2)                                         # (nlev, nlat)
    hr_zm = np.where(zm_w > 0, (hr * m3).sum(2) / np.maximum(zm_w, 1e-300), np.nan)
    # WET effective diameter. Tabazadeh water uptake depends on (T, RH) only, so every bin of a
    # cell grows by the same linear factor gf = deff_wet_cell / deff_dry_cell (both on the bin
    # grid). Then S3_wet = gf^3 S3_dry, S2_wet = gf^2 S2_dry per cell, and column/global wet
    # D_eff = sum S3_wet / sum S2_wet exactly.
    nlev = num.shape[1]
    S2 = np.zeros(num.shape[1:]); S3 = np.zeros(num.shape[1:])
    for k in range(num.shape[0]):                            # bin loop keeps memory flat
        nk = num[k].astype(np.float64)
        S2 += nk * dpb[k] ** 2; S3 += nk * dpb[k] ** 3
    dd = S3 / np.maximum(S2, 1e-300)                         # dry D_eff per cell, nm
    dw = z['deff_wet_nm'].astype(np.float64)
    gf = np.where((S2 > 0) & (dw > 0), dw / np.maximum(dd, 1e-300), 1.0)
    S2w = gf ** 2 * S2 * w; S3w = gf ** 3 * S3 * w             # per m2 of column
    deff_wet = S3w.sum(0) / np.maximum(S2w.sum(0), 1e-300)    # column map, nm
    deff_wet_glob = float((S3w * area[None]).sum() / max((S2w * area[None]).sum(), 1e-300))
    gf_med = float(np.median(gf[(S2 > 0) & (w > 0)])) if ((S2 > 0) & (w > 0)).any() else np.nan
    out = dict(
        aod=z['aod550'].astype(np.float32), arf=z['toa_arf'].astype(np.float32),
        colN=colN.astype(np.float32), colM=colM.astype(np.float32),
        dmass=dmass.astype(np.float32), deff_dry=deff_dry.astype(np.float32),
        deff_wet=deff_wet.astype(np.float32), gf_med=gf_med,
        dmass_glob=float(np.exp((Mk * lnD * area[None]).sum() / max((colM * area).sum(), 1e-300))),
        Nk_glob=(Nk * area[None]).sum((1, 2)) / A, Mk_glob=(Mk * area[None]).sum((1, 2)) / A,
        hr_zm=hr_zm.astype(np.float32), hr_glob=hr_glob,
        aod_glob=float((z['aod550'] * area).sum() / A), arf_glob=float((z['toa_arf'] * area).sum() / A),
        colM_glob=float((colM * area).sum() / A), colN_glob=float((colN * area).sum() / A),
        deff_wet_glob=deff_wet_glob,
        so2_col_glob=float(((np.einsum('lyx,lyx->yx', z['so2'], w)) * area).sum() / A),
    )
    return out, int(z['hour_abs']), int(z['step'])


def work(args):
    s, fi, fc = args
    out_f = f'{CACHE}/snap_{s:04d}.npz'
    if os.path.exists(out_f):
        return s, 0.0
    t0 = time.time()
    g = geo()
    hour_abs = int(np.load(fc)['hour_abs'])
    sm = strat_mask(hour_abs)
    ri, hi, si = one_run(fi, sm, g)
    rc, hc, sc = one_run(fc, sm, g)
    assert hi == hc and si == sc == s, (s, hi, hc, si, sc)
    res = {f'inj_{k}': v for k, v in ri.items()}
    res.update({f'ctl_{k}': v for k, v in rc.items()})
    res.update(step=s, hour_abs=hour_abs, strat_frac=float(sm.mean()),
               strat_frac_area=float((sm * g['area'][None]).sum() / (g['area'].sum() * sm.shape[0])))
    np.savez(out_f + '.tmp.npz', **res)
    os.replace(out_f + '.tmp.npz', out_f)
    return s, time.time() - t0


if __name__ == '__main__':
    ap = add_run_args(argparse.ArgumentParser(description=__doc__.split('\n')[0]))
    ap.add_argument('nworkers', nargs='?', type=int, default=12, help='worker processes (default 12)')
    a = ap.parse_args()
    configure(inj=a.inj, inj_y1=a.inj_y1, ctrl=a.ctrl)
    nw = a.nworkers
    fi, fc = snap_files('inj'), snap_files('ctrl')
    # frames3d files are written via a temp name + os.replace, so every listed file is complete
    steps = sorted(set(fi) & set(fc))
    todo = [(s, fi[s], fc[s]) for s in steps if not os.path.exists(f'{CACHE}/snap_{s:04d}.npz')]
    print(f'{len(steps)} paired snapshots, {len(todo)} to reduce', flush=True)
    geo()
    with Pool(nw) as p:
        for s, dt in p.imap_unordered(work, todo):
            print(f'  step {s:4d} {dt:5.1f}s', flush=True)
