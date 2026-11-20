#!/usr/bin/env python3
"""Does a RAD_SOLAR mode's error AVERAGE OUT over consecutive days, or pile up?

check_solar_window.py scores one day at a time. That is the right test for the
deterministic modes, whose error sits in the same place every day, but not for
RAD_SOLAR=dither: each dithered call is as wrong locally as an instant one, and
the claim is only that the error moves around and so averages out in dT_rad,
which integrates the heating. This harness tests that directly. For K
consecutive days, on ONE frozen aerosol state (num, dT_rad) but each day's own
meteorology, it computes the daily-mean heating anomaly from

  hourly   24 instant calls at c-2 .. c+21, each held 1 h -- the reference
  <mode>   4 calls at c, c+6, c+12, c+18, each held 6 h

and reports, for every k = 1..K, the errors of the k-day MEAN heating against
the k-day mean hourly heating: column rms, per-level rms, zonal-mean bias,
wave-4 amplitude (absolute over |lat| <= 30, over the reference's tropical rms),
global-mean column heating error and daily-mean global ARF error. A mode whose
k-day error falls with k averages out; one whose error stays flat piles up
(dT_rad has no damping here, so this is the worst case for the piling up).

Labels: instant, window, dither, ... (any RAD_SOLAR mode) and window_gq@N.
GPU + CESM archive, about 6 min per day for instant window dither window_gq@2.
Run from runs/<TAG>/:
    CUDA_VISIBLE_DEVICES=0 python3 ../../scripts/validation/check_solar_accum.py \\
        --state STATE.npz --start YYYY-MM-DD --days K [--mode M ...] [--nodes N ...]
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
ap.add_argument('--state', required=True, help='rad_sample_*.npz (num, dT_rad, klevs)')
ap.add_argument('--start', required=True, help='first day YYYY-MM-DD (archive year, noleap)')
ap.add_argument('--days', type=int, default=10)
ap.add_argument('--mode', action='append', default=None,
                help='RAD_SOLAR mode (repeatable; default instant window dither)')
ap.add_argument('--nodes', action='append', type=int, default=None,
                help='also score window_gq with N nodes, as window_gq@N (repeatable)')
ap.add_argument('--out', default='check_solar_accum')
args = ap.parse_args()
MODES = args.mode if args.mode else ['instant', 'window', 'dither']
assert all(m in R.SOLAR_MODES for m in MODES), f'--mode must be among {R.SOLAR_MODES}'
SPECS = [(m, m, None) for m in MODES] + [(f'window_gq@{n}', 'window_gq', n) for n in args.nodes or []]
LABS = [s[0] for s in SPECS]
LW = max([12] + [len(x) for x in LABS])
assert jax.devices()[0].platform == 'gpu', f'JAX is on {jax.devices()} -- set LD_LIBRARY_PATH to libcuda (see run_prod.py)'

t0 = time.time()
log = open(f'{args.out}.log', 'w')
def say(s=''):
    print(s, flush=True); log.write(s + '\n'); log.flush()

dU = C.open_var('U'); lat = dU['lat'].values; lon = dU['lon'].values
nlat, nlon = lat.size, lon.size
ntime = dU['time'].size
_e = np.concatenate([[lat[0] * C.DEG], 0.5 * (lat[:-1] + lat[1:]) * C.DEG, [lat[-1] * C.DEG]])
W_LAT = ((np.sin(_e[1:]) - np.sin(_e[:-1])) / ((lat[1] - lat[0]) * C.DEG) if C.ADV_METRIC
         else np.cos(lat * C.DEG))
w_lat = W_LAT[:, None] * np.ones((nlat, nlon))
gm = lambda x: float((x * w_lat).sum() / w_lat.sum())
TROP = np.abs(lat) <= 30
ds_mam = {f'{p}_a{m}': C.open_var(f'{p}_a{m}') for p in ('num', 'so4') for m in (1, 2, 3)}
z = np.load(args.state)
num = jnp.asarray(z['num']); dT = np.asarray(z['dT_rad'], np.float64); klevs = [int(k) for k in z['klevs']]
y, m, d = (int(x) for x in args.start.split('-'))
c0 = int(round((y - 1996) * 8760 + (np.cumsum([0, 31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30])[m - 1] + d - 1) * 24))
assert c0 - 2 >= 0 and c0 + 24 * (args.days - 1) + 21 < ntime, 'days outside the archive'
drv = R.RadiationDriver(C.open_var, klevs, lat, lon, C.DP_BIN * 1e-9, solar='instant', solar_window_h=6.0, solar_align='centred',
                        mmid_kg=C.MMID, rho_dry=C.RHO_AER)
NODES0 = drv.nodes
say(f'driver ready {time.time()-t0:.0f}s; state {os.path.basename(args.state)}; {args.days} days from '
    f'{args.start}; labels {LABS}')


def anomaly(mode, t, num_ref, nodes=None):
    drv.solar_mode = mode
    drv.nodes = NODES0 if nodes is None else nodes
    hr, d1, d0 = drv.heating_anomaly(t, num, num_ref, dT)
    up1 = np.asarray(d1['sw_up_toa']) + np.asarray(d1['olr'])
    up0 = np.asarray(d0['sw_up_toa']) + np.asarray(d0['olr'])
    return np.asarray(hr, np.float64), up0 - up1


def wave4(col):
    F = np.fft.rfft(col, axis=-1)
    return float((2 * np.abs(F[..., 4]) / col.shape[-1])[TROP].mean())


S = 86400.0
cum = {k: [0.0, 0.0] for k in ['hourly'] + LABS}        # running sums of (hr, arf) daily means
out = {'lat': lat, 'lon': lon, 'klevs': np.array(klevs), 'labels': np.array(LABS)}
rows = []                                                # (k, label, metrics...)
for k in range(args.days):
    td = time.time(); c = c0 + 24 * k
    calls = (c, c + 6, c + 12, c + 18)
    num_ref = {t: jnp.asarray(C.bin_mam4(ds_mam, t - C.H0, klevs)[0]) for t in range(c - 2, c + 22)}
    hourly = {t: anomaly('instant', t, num_ref[t]) for t in range(c - 2, c + 22)}
    res = {'hourly': (np.mean([v[0] for v in hourly.values()], 0) * S, np.mean([v[1] for v in hourly.values()], 0))}
    offs = [R.dither_offset(drv.time[t], 6.0) for t in calls]
    for lab, mode, nodes in SPECS:
        vals = [hourly[t] for t in calls] if mode == 'instant' else [anomaly(mode, t, num_ref[t], nodes) for t in calls]
        res[lab] = (np.mean([v[0] for v in vals], 0) * S, np.mean([v[1] for v in vals], 0))
    for lab, (hr, arf) in res.items():
        cum[lab][0] = cum[lab][0] + hr; cum[lab][1] = cum[lab][1] + arf
        out[f'd{k}_{lab}_col'] = hr.sum(0).astype(np.float32)
    n = k + 1
    ref_hr, ref_arf = cum['hourly'][0] / n, cum['hourly'][1] / n
    rcol = ref_hr.sum(0); rrms = np.sqrt(gm(rcol ** 2))
    trms = np.sqrt((rcol[TROP] ** 2 * w_lat[TROP]).sum() / w_lat[TROP].sum())
    say(f'\n=== {n}-day mean (day {k}: {drv.time[c]}, {time.time()-td:.0f}s; dither offsets '
        + ' '.join(f'{o:+.2f}' for o in offs) + ' h) ===')
    say(f'{"":{LW}s} {"col rms":>8s} {"lev rms":>8s} {"zm bias":>8s} {"w4 err":>7s} {"heat err":>9s} {"ARF err":>8s}')
    for lab in LABS:
        hr, arf = cum[lab][0] / n, cum[lab][1] / n
        e = hr - ref_hr; ecol = e.sum(0)
        col_rms = np.sqrt(gm(ecol ** 2)) / rrms
        lev_rms = np.sqrt(np.mean([gm(e[l] ** 2) for l in range(e.shape[0])])) / \
            np.sqrt(np.mean([gm(ref_hr[l] ** 2) for l in range(e.shape[0])]))
        zm = np.abs(ecol.mean(-1)).max() / np.abs(rcol.mean(-1)).max()
        w4e = wave4(ecol) / trms
        heat = 100 * (gm(hr.sum(0)) / gm(rcol) - 1)
        arf_err = 100 * (gm(arf) / gm(ref_arf) - 1)
        say(f'{lab:{LW}s} {100*col_rms:7.1f}% {100*lev_rms:7.1f}% {100*zm:7.1f}% {100*w4e:6.1f}% '
            f'{heat:+8.1f}% {arf_err:+7.1f}%')
        rows.append((n, lab, col_rms, lev_rms, zm, w4e, heat, arf_err))
    out['rows_n'] = np.array([r[0] for r in rows]); out['rows_label'] = np.array([r[1] for r in rows])
    out['rows'] = np.array([r[2:] for r in rows], np.float64)   # col rms, lev rms, zm, w4 (fractions), heat %, ARF %
    for lab in ['hourly'] + LABS:
        out[f'mean_{lab}_hr'] = (cum[lab][0] / n).astype(np.float32)
    np.savez(f'{args.out}.npz', **out)                          # after every day, so a partial run is usable

say(f'\n=== error of the k-day mean vs k: column rms (wave-4) ===')
say(f'{"k":>3s} ' + ' '.join(f'{lab:>{max(14, len(lab))}s}' for lab in LABS))
for n in range(1, args.days + 1):
    cells = []
    for lab in LABS:
        r = [x for x in rows if x[0] == n and x[1] == lab][0]
        cells.append(f'{100*r[2]:5.1f}% ({100*r[5]:4.1f}%)'.rjust(max(14, len(lab))))
    say(f'{n:3d} ' + ' '.join(cells))
say(f'\nsaved {args.out}.npz in {time.time()-t0:.0f}s')
