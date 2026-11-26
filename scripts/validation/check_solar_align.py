#!/usr/bin/env python3
"""Does it matter where window_gq's 6-h window sits relative to the call hour?

Radiation runs at hour t, the LAST hour of each 6-h step, and its heating is
added to dT_rad there for the next step. window_gq's window is centred on t
(t-3..t+3 h); the alternative is the window that ends at t (t-6..t, the step
just finished), which keeps dT_rad causal but puts the call-hour inputs
(meteorology, aerosol, reference aerosol) at the window's end instead of its
centre.

Same frame as check_solar_gq_lw.py: ONE frozen aerosol state (num, dT_rad),
each day's own meteorology, 4 calls/day (c, c+6, c+12, c+18). Each alignment
is scored against ITS OWN hourly reference, the instant solves at the hours its
windows cover (record k = the mean over [k-1, k], as the albedo window):
  centred   calls' windows t-3..t+3  -> hourly c-2 .. c+21
  trailing  calls' windows t-6..t    -> hourly c-5 .. c+18
Also printed: instant against each reference, and the two references against
each other (how much the 3-h shift itself moves the truth).

  CUDA_VISIBLE_DEVICES=0 python3 check_solar_align.py --state S.npz --start 1997-01-01 --days 3 --out X
"""
import os, sys, time, argparse
import numpy as np
_REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
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
assert c0 - 5 >= 0 and c0 + 24 * (args.days - 1) + 21 < ntime
drv = R.RadiationDriver(C.open_var, klevs, lat, lon, C.DP_BIN * 1e-9, solar='instant', solar_window_h=6.0,
                        mmid_kg=C.MMID, rho_dry=C.RHO_AER)
drv.ref_cache = None
W = drv.solar_window_h
say(f'driver ready {time.time()-t0:.0f}s; state {os.path.basename(args.state)}; {args.days} days from '
    f'{args.start}; window_gq nodes {args.nodes}')


def solve(t, numx, T_override):
    hr, dg = drv.heating(t, numx, T_band_override=T_override, split=True)
    return np.asarray(hr, np.float64), np.asarray(dg['sw_up_toa'], np.float64), np.asarray(dg['olr'], np.float64)


def anomaly(mode, t, num_ref, offset=0.0):
    """hr: heating anomaly [K/s]; arf: TOA forcing [W/m2] (SW-up + OLR)."""
    drv.solar_mode = mode; drv.nodes = args.nodes; drv.solar_offset_h = offset
    Tc = drv._read3d(drv.dT, 'T', t)[drv.band_pos].astype(np.float64)
    h1, u1, o1 = solve(t, num, Tc + dT)
    h0, u0, o0 = solve(t, num_ref, None)
    drv.solar_offset_h = 0.0
    return {'hr': h1 - h0, 'arf': (u0 - u1) + (o0 - o1)}


def wave4(col):
    F = np.fft.rfft(col, axis=-1)
    return float((2 * np.abs(F[..., 4]) / col.shape[-1])[TROP].mean())


mean = lambda ds: {k: np.mean([x[k] for x in ds], 0) for k in ds[0]}
S = 86400.0
# (label, scheme, reference)
ROWS = [('instant vs centred ref', 'instant', 'ref_c'), ('instant vs trailing ref', 'instant', 'ref_t'),
        ('gq centred (now)', 'gq_c', 'ref_c'), ('gq trailing (t-6..t)', 'gq_t', 'ref_t'),
        ('trailing ref vs centred ref', 'ref_t', 'ref_c')]
cum = {}
out = {'lat': lat, 'lon': lon, 'klevs': np.array(klevs)}
for k in range(args.days):
    td = time.time(); c = c0 + 24 * k
    calls = (c, c + 6, c + 12, c + 18)
    hrs = range(c - 5, c + 22)
    num_ref = {t: jnp.asarray(C.bin_mam4(ds_mam, t - C.H0, klevs)[0]) for t in hrs}
    hourly = {t: anomaly('instant', t, num_ref[t]) for t in hrs}
    res = {'ref_c': mean([hourly[t] for t in range(c - 2, c + 22)]),
           'ref_t': mean([hourly[t] for t in range(c - 5, c + 19)]),
           'instant': mean([hourly[t] for t in calls]),
           'gq_c': mean([anomaly('window_gq', t, num_ref[t], 0.0) for t in calls]),
           'gq_t': mean([anomaly('window_gq', t, num_ref[t], -W / 2) for t in calls])}
    out[f'd{k}_hourly_garf'] = np.array([gm(hourly[t]['arf']) for t in hrs])
    for lab, r in res.items():
        r = {'hr': r['hr'] * S, 'arf': r['arf']}
        cum[lab] = r if lab not in cum else {kk: cum[lab][kk] + r[kk] for kk in r}
        out[f'd{k}_{lab}_col'] = r['hr'].sum(0).astype(np.float32)
        out[f'd{k}_{lab}_arf'] = r['arf'].astype(np.float32)
    n = k + 1
    v = {lab: {kk: x / n for kk, x in cum[lab].items()} for lab in cum}
    say(f'\n=== {n}-day mean (day {k}: {drv.time[c]}, {time.time()-td:.0f}s) ===')
    for lab in ('ref_c', 'ref_t'):
        say(f'{lab}: global ARF {gm(v[lab]["arf"]):+.4f} W/m2, column heating rms '
            f'{np.sqrt(gm(v[lab]["hr"].sum(0) ** 2)):.4g} K/day')
    say(f'{"":30s} {"col rms":>8s} {"lev rms":>8s} {"w4 err":>7s} {"ARF err":>8s}   (all / the reference)')
    for lab, a, b in ROWS:
        ref = v[b]; rcol = ref['hr'].sum(0)
        rrms = np.sqrt(gm(rcol ** 2))
        rlev = np.sqrt(np.mean([gm(ref['hr'][l] ** 2) for l in range(ref['hr'].shape[0])]))
        trms = np.sqrt((rcol[TROP] ** 2 * w_lat[TROP]).sum() / w_lat[TROP].sum())
        e = v[a]['hr'] - ref['hr']; ecol = e.sum(0); ea = v[a]['arf'] - ref['arf']
        say(f'{lab:30s} {100*np.sqrt(gm(ecol**2))/rrms:7.1f}% '
            f'{100*np.sqrt(np.mean([gm(e[l]**2) for l in range(e.shape[0])]))/rlev:7.1f}% '
            f'{100*wave4(ecol)/trms:6.1f}% {100*gm(ea)/gm(ref["arf"]):+7.2f}% {gm(ea):+.4f} W/m2')
    np.savez(f'{args.out}.npz', **out)
say(f'\ndone {time.time()-t0:.0f}s')
