#!/usr/bin/env python3
"""Where does window_gq's remaining error come from: SW or LW? And do cheap LW fixes help?

Same frame as scripts/validation/check_solar_accum.py: ONE frozen aerosol state
(num, dT_rad), each day's own meteorology, reference = 24 instant calls/day
(c-2..c+21) each held 1 h; a scheme = 4 calls/day (c, c+6, c+12, c+18) each held
6 h. Every solve is run with split=True, so each anomaly is split into its SW and
LW heating and its SW (TOA SW-up) and LW (OLR) part of the forcing.

Variants (window_gq at --nodes N for the SW unless named @2 / @4):
  instant      context (the hourly calls at the call hours)
  gq           as in the model: 1 LW at the call hour, call-hour Ts
  gq_tsavg     (b) Ts from the window-mean sigma*Ts^4 = mean(FLNS+FLDS) over the
               window's hourly records t-2..t+3 (the albedo's window)
  gq_lw2       (c) gq SW + LW = mean of two LW solves at t-1 and t+2, each with its
               own hour's full meteorology (taken from the hourly instant solves)
  gq_lwhour    upper bound: gq SW + the exact hourly LW (all 6 hours of the window)
(a) is the SW-only / LW-only error rows of each variant against the hourly SW / LW.
All errors are normalised by the hourly TOTAL, so SW + LW parts add up.

GPU + CESM archive, ~18 min per day. Run from runs/<TAG>/:
  CUDA_VISIBLE_DEVICES=0 python3 ../../scripts/validation/check_solar_gq_lw.py --state S.npz --start 1997-01-01 --days 3 --out X
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
ap.add_argument('--state', required=True)
ap.add_argument('--start', required=True)
ap.add_argument('--days', type=int, default=3)
ap.add_argument('--nodes', type=int, default=3)
ap.add_argument('--out', required=True)
args = ap.parse_args()
assert jax.devices()[0].platform == 'gpu', f'JAX is on {jax.devices()}'

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
assert c0 - 2 >= 0 and c0 + 24 * (args.days - 1) + 21 < ntime
drv = R.RadiationDriver(C.open_var, klevs, lat, lon, C.DP_BIN * 1e-9, solar='instant', solar_window_h=6.0, solar_align='centred',
                        mmid_kg=C.MMID, rho_dry=C.RHO_AER)
drv.ref_cache = None          # every reference solve computed with split=True
say(f'driver ready {time.time()-t0:.0f}s; state {os.path.basename(args.state)}; {args.days} days from '
    f'{args.start}; window_gq nodes {args.nodes}')

# ---- (b) window-mean Ts: patch surface() -------------------------------------
_surface0 = drv.surface
TS_AVG = {'on': False}
def _surface(t):
    ts, alb = _surface0(t)
    if not TS_AVG['on']:
        return ts, alb
    w = drv.solar_window_h
    lo = max(0, t + int(np.floor(-w / 2)) + 1); hi = min(drv.time.size - 1, t + int(np.ceil(w / 2)))
    up = np.stack([drv.dFLNS['FLNS'].isel(time=k).values + drv.dFLDS['FLDS'].isel(time=k).values
                   for k in range(lo, hi + 1)]).mean(axis=0)
    return (np.maximum(up, 1.0) / R.SIGMA_SB) ** 0.25, alb
drv.surface = _surface


def solve(t, numx, T_override):
    hr, dg = drv.heating(t, numx, T_band_override=T_override, split=True)
    hr = np.asarray(hr, np.float64); sw = np.asarray(dg['hr_sw_band'], np.float64)
    return hr, sw, np.asarray(dg['sw_up_toa'], np.float64), np.asarray(dg['olr'], np.float64)


def anomaly(mode, t, num_ref, ts_avg=False, nodes=None):
    """dict sw, lw: heating anomaly parts [K/s]; asw, alw: forcing parts [W/m2]."""
    drv.solar_mode = mode; drv.nodes = nodes or args.nodes; TS_AVG['on'] = ts_avg
    Tc = drv._read3d(drv.dT, 'T', t)[drv.band_pos].astype(np.float64)
    h1, s1, u1, o1 = solve(t, num, Tc + dT)
    h0, s0, u0, o0 = solve(t, num_ref, None)
    TS_AVG['on'] = False
    return {'sw': s1 - s0, 'lw': (h1 - s1) - (h0 - s0), 'asw': u0 - u1, 'alw': o0 - o1}


def wave4(col):
    F = np.fft.rfft(col, axis=-1)
    return float((2 * np.abs(F[..., 4]) / col.shape[-1])[TROP].mean())


mean = lambda ds: {k: np.mean([x[k] for x in ds], 0) for k in ds[0]}
LABS = ['instant', 'gq@2', 'gq', 'gq@4', 'gq_tsavg', 'gq_lw2', 'gq_lwhour']
S = 86400.0
cum = {}
out = {'lat': lat, 'lon': lon, 'klevs': np.array(klevs)}
for k in range(args.days):
    td = time.time(); c = c0 + 24 * k
    calls = (c, c + 6, c + 12, c + 18)
    num_ref = {t: jnp.asarray(C.bin_mam4(ds_mam, t - C.H0, klevs)[0]) for t in range(c - 2, c + 22)}
    hourly = {t: anomaly('instant', t, num_ref[t]) for t in range(c - 2, c + 22)}
    gq = {t: anomaly('window_gq', t, num_ref[t]) for t in calls}
    gq2 = {t: anomaly('window_gq', t, num_ref[t], nodes=2) for t in calls}
    gq4 = {t: anomaly('window_gq', t, num_ref[t], nodes=4) for t in calls}
    gqts = {t: anomaly('window_gq', t, num_ref[t], ts_avg=True) for t in calls}
    lw_of = lambda hrs: {'lw': np.mean([hourly[h]['lw'] for h in hrs], 0),
                         'alw': np.mean([hourly[h]['alw'] for h in hrs], 0)}
    res = {'hourly': mean(list(hourly.values())),
           'instant': mean([hourly[t] for t in calls]),
           'gq@2': mean([gq2[t] for t in calls]),
           'gq': mean([gq[t] for t in calls]),
           'gq@4': mean([gq4[t] for t in calls]),
           'gq_tsavg': mean([gqts[t] for t in calls]),
           'gq_lw2': mean([{**gq[t], **lw_of((t - 1, t + 2))} for t in calls]),
           'gq_lwhour': mean([{**gq[t], **lw_of(range(t - 2, t + 4))} for t in calls])}
    out[f'd{k}_hourly_gasw'] = np.array([gm(hourly[t]['asw']) for t in range(c - 2, c + 22)])
    out[f'd{k}_hourly_galw'] = np.array([gm(hourly[t]['alw']) for t in range(c - 2, c + 22)])
    for lab, r in res.items():
        r = {kk: v * (S if kk in ('sw', 'lw') else 1.0) for kk, v in r.items()}
        cum[lab] = r if lab not in cum else {kk: cum[lab][kk] + r[kk] for kk in r}
        out[f'd{k}_{lab}_swcol'] = r['sw'].sum(0).astype(np.float32)
        out[f'd{k}_{lab}_lwcol'] = r['lw'].sum(0).astype(np.float32)
        out[f'd{k}_{lab}_asw'] = r['asw'].astype(np.float32); out[f'd{k}_{lab}_alw'] = r['alw'].astype(np.float32)
    n = k + 1
    ref = {kk: v / n for kk, v in cum['hourly'].items()}
    rtot = ref['sw'] + ref['lw']; rcol = rtot.sum(0)
    rrms = np.sqrt(gm(rcol ** 2))
    rlev = np.sqrt(np.mean([gm(rtot[l] ** 2) for l in range(rtot.shape[0])]))
    trms = np.sqrt((rcol[TROP] ** 2 * w_lat[TROP]).sum() / w_lat[TROP].sum())
    rarf = gm(ref['asw'] + ref['alw'])
    say(f'\n=== {n}-day mean (day {k}: {drv.time[c]}, {time.time()-td:.0f}s) ===')
    say(f'hourly reference: global ARF {rarf:+.4f} W/m2 = SW {gm(ref["asw"]):+.4f} + LW {gm(ref["alw"]):+.4f}; '
        f'column heating rms {rrms:.4g} K/day')
    say(f'{"":14s} {"part":5s} {"col rms":>8s} {"lev rms":>8s} {"w4 err":>7s} {"ARF err":>8s}   (all / hourly TOTAL)')
    for lab in LABS:
        v = {kk: x / n for kk, x in cum[lab].items()}
        for part, e, ea in (('total', (v['sw'] + v['lw']) - rtot, (v['asw'] + v['alw']) - (ref['asw'] + ref['alw'])),
                            ('SW', v['sw'] - ref['sw'], v['asw'] - ref['asw']),
                            ('LW', v['lw'] - ref['lw'], v['alw'] - ref['alw'])):
            ecol = e.sum(0)
            col = np.sqrt(gm(ecol ** 2)) / rrms
            lev = np.sqrt(np.mean([gm(e[l] ** 2) for l in range(e.shape[0])])) / rlev
            say(f'{lab if part == "total" else "":14s} {part:5s} {100*col:7.1f}% {100*lev:7.1f}% '
                f'{100*wave4(ecol)/trms:6.1f}% {100*gm(ea)/rarf:+7.2f}% {gm(ea):+.4f} W/m2')
    np.savez(f'{args.out}.npz', **out)
say(f'\ndone {time.time()-t0:.0f}s')
