"""Daily zonal-mean reductions from the FRAME_LEVELS=all zonal frames (one frame per day at
00 UTC; a zonal mean averages over all local times, so heating is diurnally representative).

Injection = year-1 re-run (--inj-y1, default runs/fast_y1frames_inj6) + year 2 (--inj, default
runs/fast_2yr_inj6_30NS, zonal frames from its resume point). Control = --ctrl, default
runs/fast_2yr_ctrl. Writes cache/daily.npz in the current directory.

  python3 reduce_daily.py [--inj DIR] [--inj-y1 DIR] [--ctrl DIR]

Stratosphere filter: band levels at or above the zonal mean of CESM TROP_P at the frame hour.
"""
import argparse
import os

import numpy as np

from lib import CACHE, G, add_run_args, configure, final_or_ckpt, geo, npz_member, npz_names, trop_p


def zm_series(run_keys):
    """Concatenate zonal frames of one or more runs by hour (later runs win on overlap)."""
    out = {}
    for rk in run_keys:
        f = final_or_ckpt(rk, 'frames')
        if not os.path.exists(f) or 'frames_zm_hours' not in npz_names(f):
            continue
        hrs = np.asarray(npz_member(f, 'frames_zm_hours'))
        mas = npz_member(f, 'frames_zm_mas'); num = npz_member(f, 'frames_zm_num')
        hr = npz_member(f, 'frames_zm_hr_kday'); so2 = npz_member(f, 'frames_zm_so2')
        for i, h in enumerate(hrs):
            out[int(h)] = (f, i, mas, num, hr, so2)
    return dict(sorted(out.items()))


def main():
    g = geo()
    lat, plev, dp = g['lat'], g['plev'], g['dp']
    wlat = g['area'].sum(1); wlat = wlat / wlat.sum()
    zi = zm_series(['inj_y1', 'inj'])
    zc = zm_series(['ctrl'])
    hours = sorted(set(zi) & set(zc))
    hours = [h for h in hours if h > 0]
    print(f'{len(hours)} paired daily zonal frames (inj {len(zi)}, ctrl {len(zc)})', flush=True)
    nd, nl, ny = len(hours), plev.size, lat.size
    res = {k: np.full((nd, ny), np.nan) for k in
           ('inj_colM', 'ctl_colM', 'inj_colN', 'ctl_colN', 'inj_hr_lat', 'ctl_hr_lat', 'inj_so2col', 'ctl_so2col')}
    res.update({k: np.full(nd, np.nan) for k in ('inj_hr_glob', 'ctl_hr_glob')})
    res.update({k: np.zeros((nd, nl, ny), np.float32) for k in ('inj_so4_zm', 'ctl_so4_zm', 'inj_hr_zm', 'ctl_hr_zm',
                                                                 'inj_num_zm', 'ctl_num_zm')})
    ptz = np.zeros((nd, ny))
    for j, h in enumerate(hours):
        pt = trop_p(h)                                   # H0 = 0, so run hour == archive hour
        ptz[j] = np.nanmean(pt, axis=1)
        mask = plev[:, None] <= ptz[j][None]             # (nlev, nlat)
        w = dp[:, None] / G * mask                       # kg air / m2 per layer
        for pre, (f, i, mas, num, hr, so2) in (('inj', zi[h]), ('ctl', zc[h])):
            m = np.asarray(mas[i], np.float64).sum(0)    # (nlev, nlat) kg/kg
            n = np.asarray(num[i], np.float64).sum(0)
            q = np.asarray(hr[i], np.float64)
            res[f'{pre}_colM'][j] = (m * w).sum(0)
            res[f'{pre}_colN'][j] = (n * w).sum(0)
            res[f'{pre}_so2col'][j] = (np.asarray(so2[i], np.float64) * w).sum(0)
            ws = w.sum(0)
            res[f'{pre}_hr_lat'][j] = np.where(ws > 0, (q * w).sum(0) / np.maximum(ws, 1e-300), np.nan)
            wm = w * wlat[None]
            res[f'{pre}_hr_glob'][j] = (q * wm).sum() / wm.sum()
            res[f'{pre}_so4_zm'][j] = m; res[f'{pre}_num_zm'][j] = n; res[f'{pre}_hr_zm'][j] = q
        if j % 50 == 0:
            print(f'  day {h / 24:.0f}', flush=True)
    np.savez(f'{CACHE}/daily.npz', hours=np.array(hours), lat=lat, plev=plev, dp=dp, wlat=wlat,
             trop_zm=ptz, **res)
    print('wrote cache/daily.npz', flush=True)


if __name__ == '__main__':
    a = add_run_args(argparse.ArgumentParser(description=__doc__.split('\n')[0])).parse_args()
    configure(inj=a.inj, inj_y1=a.inj_y1, ctrl=a.ctrl)
    main()
