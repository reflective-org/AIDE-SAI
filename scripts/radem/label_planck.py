#!/usr/bin/env python3
"""Labels for the clear-sky response net P (the damping of RAD_MODE=damped):
    P = HR(no aerosol, T + dT) - HR(no aerosol, T)      (window_gq@3 physics, split SW/LW)
plus d sw_up_toa, d olr. Per globe: a campaign hour (gases/surface of that hour, met T), and per column a dT
profile on the band levels drawn from
  P_SHAPE  a dT_rad column of --dt-run frames (accumulated aerosol heating: realistic vertical shapes),
           rescaled so its peak |dT| ~ U(0, DT_MAX)
  P_SMOOTH a smooth random profile (3 cosine modes), amplitude U(0, DT_MAX)
  rest     zero (structural zero)
Stored per kept column: T q o3 ch4 (nzc), ts, alb, dT (nzb), P_sw P_lw (nzb, K/day), dswup dolr (W/m2)."""
import os, sys, glob, time, argparse
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import radem_lib as L

ap = argparse.ArgumentParser()
ap.add_argument('--steps', required=True); ap.add_argument('--n-globes', type=int, required=True)
ap.add_argument('--start', type=int, default=0); ap.add_argument('--out', required=True)
ap.add_argument('--seed', type=int, default=0); ap.add_argument('--keep', type=int, default=12000)
ap.add_argument('--dt-run', nargs='+', default=['/data/AIDE-SAI-link-perf/runs/rgq_30d_incphys/frames3d'])
ap.add_argument('--dt-max', type=float, default=6.0); ap.add_argument('--dt-min', type=float, default=0.0)
ap.add_argument('--log-amp', action='store_true', help='amplitudes log-uniform [dt-min, dt-max]')
ap.add_argument('--own-dirs', nargs='*', default=[], help='frames3d dirs: whole globes at their own hour with their own dT_rad')
ap.add_argument('--p-own', type=float, default=0.6); ap.add_argument('--holdout', default='')
ap.add_argument('--keep-dt', action='store_true', help='keep columns with max|dT| > 0.01 K first (on-policy globes are mostly dT ~ 0)')
ap.add_argument('--p-shape', type=float, default=0.5); ap.add_argument('--p-smooth', type=float, default=0.4)
args = ap.parse_args()
os.makedirs(args.out, exist_ok=True)
rng = np.random.default_rng(args.seed * 100003 + args.start)
import rad_features as F
t0 = time.time()
drv, klevs = L.make_driver('window_gq', 3); met = L.MetT()
nzb = len(klevs); nlat, nlon = drv.nlat, drv.nlon; ncol = nlat * nlon
a, b = (int(x) for x in args.steps.split(':'))
# dT shapes: columns of the accumulated-dT frames where |dT| is non-trivial
shapes = []
for f in [x for d_ in args.dt_run for x in sorted(glob.glob(os.path.join(d_, 'step_*.npz')))[20::10]]:
    d = np.load(f)['dT_rad'].reshape(nzb, -1)
    pk = np.abs(d).max(0); idx = np.nonzero(pk > 0.005)[0]
    if idx.size:
        sel = rng.choice(idx, min(4000, idx.size), replace=False)
        shapes.append(d[:, sel] / pk[sel])
shapes = np.concatenate(shapes, 1)
print(f'driver ready {time.time()-t0:.0f}s; {shapes.shape[1]} dT shapes from {args.dt_run}', flush=True)
z = np.linspace(0, np.pi, nzb)
OWN = [x for d_ in args.own_dirs for x in sorted(glob.glob(os.path.join(d_, 'step_*.npz')))]
HOLD = tuple(int(x) for x in args.holdout.split(':')) if args.holdout else None
MDAYS = np.cumsum([0, 31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31])
def day_of_month(hour_abs):
    doy = ((hour_abs - 1) // 24) % 365
    m = int(np.searchsorted(MDAYS, doy, side='right')) - 1
    return int(doy - MDAYS[m]) + 1
print(f'{len(OWN)} own-dT frames', flush=True)
for g in range(args.start, args.n_globes):
    tg = time.time()
    own = bool(OWN) and rng.random() < args.p_own
    for _try in range(1000):
        if own:
            st = int(rng.integers(len(OWN))); zf = np.load(OWN[st]); hour = int(zf['hour_abs'])
        else:
            st = int(rng.integers(a, b)); hour = 8766 + 6 * (st + 1)
        if not HOLD or not (HOLD[0] <= day_of_month(hour) <= HOLD[1]):
            break
    T = met(hour)
    r = rng.random(ncol)
    amp = (np.exp(rng.uniform(np.log(args.dt_min), np.log(args.dt_max), ncol)) if args.log_amp
           else rng.uniform(0, args.dt_max, ncol))
    dT = np.zeros((nzb, ncol))
    m1 = r < args.p_shape; m2 = (r >= args.p_shape) & (r < args.p_shape + args.p_smooth)
    dT[:, m1] = shapes[:, rng.integers(0, shapes.shape[1], m1.sum())] * amp[m1]
    c = rng.normal(0, 1, (3, m2.sum())) / np.sqrt(3); ph = rng.uniform(0, 2 * np.pi, (3, m2.sum()))
    sm = sum(c[k] * np.cos((k + 1) * z[:, None] + ph[k]) for k in range(3))
    dT[:, m2] = sm / np.maximum(np.abs(sm).max(0), 1e-6) * amp[m2]
    dT = dT.reshape(nzb, nlat, nlon)
    if own:                                    # the run's own dT field, scaled x logU[0.5, 2] per column
        dT = np.asarray(zf['dT_rad'], np.float64) * np.exp(rng.uniform(np.log(0.5), np.log(2.0), (1, nlat, nlon)))
    h1, d1 = drv.heating(hour, None, T_band_override=T + dT, split=True)
    h0, d0 = drv.heating(hour, None, T_band_override=T, split=True)
    S = L.SEC_PER_DAY
    s1, s0 = np.asarray(d1['hr_sw_band']), np.asarray(d0['hr_sw_band'])
    Psw = (s1 - s0) * S; Plw = ((np.asarray(h1) - s1) - (np.asarray(h0) - s0)) * S
    dup = np.asarray(d1['sw_up_toa']) - np.asarray(d0['sw_up_toa']); dolr = np.asarray(d1['olr']) - np.asarray(d0['olr'])
    Tf = drv._read3d(drv.dT, 'T', hour).astype(np.float64); Tf[drv.band_pos] = T
    q = drv._read3d(drv.dQ, 'Q', hour); o3 = drv._read3d(drv.dO3, 'O3', hour); ch4 = drv._read3d(drv.dCH4, 'CH4', hour)
    ts, alb = drv.surface(hour)
    parts = drv.solar_parts(hour)                       # window_gq nodes: (zen, scale, alb)
    gmu = np.stack([np.where(sc > 0, np.cos(zz), 0.0) for zz, sc, _ in parts]); gsc = np.stack([sc for _, sc, _ in parts])
    if args.keep_dt:
        Acol = np.abs(dT).max(0).reshape(-1); pk = np.where(Acol > 0.01, 1.0, 0.03)
        keep = np.nonzero(rng.random(ncol) < pk)[0]
        if keep.size > args.keep: keep = rng.choice(keep, args.keep, replace=False)
        keep = np.sort(keep)
    else:
        keep = np.sort(rng.choice(ncol, min(args.keep, ncol), replace=False))
    yi, xi = np.unravel_index(keep, (nlat, nlon))
    col = lambda x: np.asarray(x)[..., yi, xi]
    np.savez(os.path.join(args.out, f'g{g:04d}.npz'), T=col(Tf).T.astype(np.float32), q=col(q).T.astype(np.float32),
             o3=col(o3).T.astype(np.float32), ch4=col(ch4).T.astype(np.float32), ts=col(ts).astype(np.float32),
             alb=col(alb).astype(np.float32), dT=col(dT).T.astype(np.float32), P_sw=col(Psw).T.astype(np.float32),
             P_lw=col(Plw).T.astype(np.float32), dswup=col(dup).astype(np.float32), dolr=col(dolr).astype(np.float32),
             gmu=col(gmu).T.astype(np.float32), gsc=col(gsc).T.astype(np.float32), step=st, hour=hour, own=own)
    print(f'[{g+1}/{args.n_globes}] hour {hour}: P_lw rms {np.sqrt((Plw**2).mean()):.4f} max {np.abs(Plw).max():.3f} K/d, '
          f'P_sw rms {np.sqrt((Psw**2).mean()):.5f} max {np.abs(Psw).max():.4f} K/d; dT max {np.abs(dT).max():.2f} K ({time.time()-tg:.1f}s)', flush=True)
print(f'LABEL_DONE {args.n_globes - args.start} globes in {time.time()-t0:.0f}s', flush=True)
