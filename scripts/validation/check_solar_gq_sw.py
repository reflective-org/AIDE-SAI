#!/usr/bin/env python3
"""RAD_SOLAR=window_gq: the one-longwave path equals N full solves, and what it costs.

radiation.RadiationDriver._hr_multi_impl writes RRTMGP.compute_heating_rate's
clear-sky path out so that a window_gq call solves the longwave ONCE and the
shortwave once per node (the longwave does not see the sun). This check:

  1. equality: heating() and its diagnostics under window_gq through that path
     against the reference, the mean of N full _hr_impl solves (drv._gq_full),
     for N = 2 and 3, on a real frozen aerosol state at T_CESM + dT_rad. It must
     agree to roundoff; a jax-rrtmgp change that breaks the copy shows up here.
  2. cost: wall time of one heating() call (one radiation solve set, CESM reads
     included, after a warm-up call) for instant, window, and window_gq at
     N = 2, 3 both ways. One radiation call of the model is heating_anomaly =
     two heating() calls, the reference one free when the reference cache hits.

GPU + CESM archive, a few minutes. Run from runs/<TAG>/:
    CUDA_VISIBLE_DEVICES=0 python3 ../../scripts/validation/check_solar_gq_sw.py \\
        [--state STATE.npz] [--date YYYY-MM-DD] [--hour H] [--reps 3]
"""
import os, sys, time, argparse
import numpy as np
_REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(_REPO, 'src'))
import _paths                      # noqa: E402,F401
import coupling as C               # noqa: E402
import radiation as R              # noqa: E402
import jax, jax.numpy as jnp       # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument('--state', default=os.path.join(_REPO, 'runs/lr1yr_d177_sample/rad_sample_lr1yr_d177_sample_s0837.npz'))
ap.add_argument('--date', default='1996-07-29')
ap.add_argument('--hour', type=int, default=12)
ap.add_argument('--reps', type=int, default=3)
args = ap.parse_args()
assert jax.devices()[0].platform == 'gpu', f'JAX is on {jax.devices()} -- set LD_LIBRARY_PATH to libcuda (see run_prod.py)'

dU = C.open_var('U'); lat = dU['lat'].values; lon = dU['lon'].values
z = np.load(args.state)
num = jnp.asarray(z['num']); dT = np.asarray(z['dT_rad'], np.float64); klevs = [int(k) for k in z['klevs']]
y, m, d = (int(x) for x in args.date.split('-'))
t = int(round((y - 1996) * 8760 + (np.cumsum([0, 31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30])[m - 1] + d - 1) * 24)) + args.hour
drv = R.RadiationDriver(C.open_var, klevs, lat, lon, C.DP_BIN * 1e-9, solar='instant', solar_window_h=6.0, solar_align='centred',
                        mmid_kg=C.MMID, rho_dry=C.RHO_AER)
Tb = drv._read3d(drv.dT, 'T', t)[drv.band_pos] + dT


def call(mode, nodes=2, full=False, split=False):
    drv.solar_mode, drv.nodes, drv._gq_full = mode, nodes, full
    hr, dg = drv.heating(t, num, T_band_override=Tb, split=split)
    hr = np.asarray(jax.block_until_ready(hr), np.float64)
    return hr, {k: np.asarray(v, np.float64) for k, v in dg.items()}


print(f'state {os.path.basename(args.state)}, {drv.time[t]}', flush=True)
print('\n1. window_gq, one-longwave path vs N full solves (max |diff| / max |ref|):')
ok = True
for n in (2, 3):
    for split in (False, True):
        h1, d1 = call('window_gq', n, full=False, split=split)
        h0, d0 = call('window_gq', n, full=True, split=split)
        rel = {'hr': np.abs(h1 - h0).max() / np.abs(h0).max()}
        rel.update({k: np.abs(d1[k] - d0[k]).max() / max(np.abs(d0[k]).max(), 1e-30) for k in d0})
        worst = max(rel.values())
        ok &= worst < 1e-9
        print(f'   N={n} split={int(split)}  worst {worst:.1e}   ' +
              '  '.join(f'{k} {v:.1e}' for k, v in rel.items()))
assert ok, 'the one-longwave window_gq path does not match N full solves'

print(f'\n2. wall time of one heating() call (mean of {args.reps} after a warm-up):')
specs = [('instant', 'instant', 2, False), ('window', 'window', 2, False)]
for n in (2, 3):
    specs += [(f'window_gq@{n} (N full)', 'window_gq', n, True), (f'window_gq@{n} (1 LW + {n} SW)', 'window_gq', n, False)]
base = None
for lab, mode, n, full in specs:
    call(mode, n, full)
    ts = []
    for _ in range(args.reps):
        t0 = time.time(); call(mode, n, full); ts.append(time.time() - t0)
    tm = float(np.mean(ts)); base = base or tm
    print(f'   {lab:28s} {tm:6.2f} s   x{tm/base:4.2f} instant', flush=True)
print('PASS')
