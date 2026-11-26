#!/usr/bin/env python3
"""Hourly radiation vs the 6-hourly RAD_SOLAR modes, on frozen aerosol states.

The precise-numbers test for RAD_SOLAR (docs/RADIATION_ARTIFACT_EXPLORATION.md).
For each test day and frozen aerosol state, the daily-mean heating ANOMALY the
model integrates (heating_anomaly: evolved bins at T_CESM + dT_rad minus the MAM4
reference at T_CESM) is computed on identical inputs as

  hourly   24 RAD_SOLAR=instant calls at the archive hours c-2 .. c+21, each held
           1 h (STEP_HOURS=1 radiation; the meteorology changes every hour) --
           the reference
  <mode>   4 calls at c, c+6, c+12, c+18, each held 6 h, for every --mode
           (default: instant window window_ins window2 window2_ins; instant is
           the subset of the hourly calls at those hours, so it is not re-solved;
           window_blend is scored only when asked for, by --mode at the
           driver's RAD_SOLAR_ALPHA, or by --alpha A, labelled window_blend@A;
           window_gq likewise by --nodes N, labelled window_gq@N)

where c is 00 UTC of the day; the four 6-h windows and the 24 hourly records both
tile [c-3 h, c+21 h]. The aerosol state is frozen over the day, so only the
radiation differs. One driver serves every mode (its solar_mode is switched
between calls), and the MAM4 reference bins of each hour are built once.

Reported per day and mode, against hourly: area-weighted rms and max error of
the band-summed anomaly, the per-level rms error, the zonal-mean bias, the
zonal wave-4 amplitude of the ERROR field (the dusk/dawn artifact; absolute
amplitude over |lat| <= 30 divided by the hourly reference's tropical rms -- the
anomaly's zonal mean is ~0, so a ratio to it would be meaningless), and the
daily-mean TOA aerosol radiative forcing (coupling.py's arf_toa definition).

GPU + CESM archive, ~6 min per day for the five default modes. Writes
<out>.npz and <out>.log to the working directory -- run it from runs/<TAG>/:
    CUDA_VISIBLE_DEVICES=0 python3 ../../scripts/validation/check_solar_window.py \
        --day STATE.npz:YYYY-MM-DD [--day ...] [--mode M ...] [--alpha A ...]
STATE is a rad_sample_*.npz (num, dT_rad, klevs); the date's year is the archive
year (noleap). A state may be used on a date other than its own (the aerosol is
then transplanted, which is fine for a geometry test).
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
ap.add_argument('--day', action='append', required=True, help='STATE.npz:YYYY-MM-DD')
ap.add_argument('--mode', action='append', default=None,
                help=f'RAD_SOLAR mode to score (repeatable; default all of {R.SOLAR_MODES[:5]})')
ap.add_argument('--alpha', action='append', type=float, default=None,
                help='also score window_blend at this RAD_SOLAR_ALPHA, as window_blend@A (repeatable)')
ap.add_argument('--nodes', action='append', type=int, default=None,
                help='also score window_gq with N Gauss nodes, as window_gq@N (repeatable)')
ap.add_argument('--out', default='check_solar_window')
args = ap.parse_args()
MODES = args.mode if args.mode else ([] if (args.alpha or args.nodes) else list(R.SOLAR_MODES[:5]))
assert all(m in R.SOLAR_MODES for m in MODES), f'--mode must be among {R.SOLAR_MODES}'
assert all(0.0 <= a <= 1.0 for a in args.alpha or []), '--alpha must be in [0, 1]'
assert all(1 <= n <= 8 for n in args.nodes or []), '--nodes must be in 1..8'
# (label, RAD_SOLAR mode, alpha or None = the driver's, nodes or None = the driver's):
# labels are the npz/log keys
SPECS = ([(m, m, None, None) for m in MODES]
         + [(f'window_blend@{a:g}', 'window_blend', a, None) for a in args.alpha or []]
         + [(f'window_gq@{n}', 'window_gq', None, n) for n in args.nodes or []])
MODES = [lab for lab, _, _, _ in SPECS]
LW = max([12] + [len(m) for m in MODES])        # label column width (12 unless window_blend@A)
assert jax.devices()[0].platform == 'gpu', f'JAX is on {jax.devices()} -- set LD_LIBRARY_PATH to libcuda (see run_prod.py)'

t0 = time.time()
log = open(f'{args.out}.log', 'w')
def say(s=''):
    print(s, flush=True); log.write(s + '\n'); log.flush()

dU = C.open_var('U'); lat = dU['lat'].values; lon = dU['lon'].values
nlat, nlon = lat.size, lon.size
ntime = dU['time'].size
# coupling.py's arf_toa area weights (W_LAT, built inside coupling.main(); same formula)
_e = np.concatenate([[lat[0] * C.DEG], 0.5 * (lat[:-1] + lat[1:]) * C.DEG, [lat[-1] * C.DEG]])
W_LAT = ((np.sin(_e[1:]) - np.sin(_e[:-1])) / ((lat[1] - lat[0]) * C.DEG) if C.ADV_METRIC
         else np.cos(lat * C.DEG))
w_lat = W_LAT[:, None] * np.ones((nlat, nlon))
gm = lambda x: float((x * w_lat).sum() / w_lat.sum())
TROP = np.abs(lat) <= 30
ds_mam = {f'{p}_a{m}': C.open_var(f'{p}_a{m}') for p in ('num', 'so4') for m in (1, 2, 3)}
days = []
for spec in args.day:
    path, date = spec.rsplit(':', 1)
    y, m, d = (int(x) for x in date.split('-'))
    z = np.load(path)
    hour_c = int(round((y - 1996) * 8760 + (np.cumsum([0, 31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30])[m - 1] + d - 1) * 24))
    # the hourly reference reads c-2 .. c+21; a negative index would wrap to the archive's END
    assert hour_c - 2 >= 0 and hour_c + 21 < ntime, f'{date}: hours {hour_c-2}..{hour_c+21} outside the archive'
    days.append(dict(path=path, date=date, hour_c=hour_c, num=jnp.asarray(z['num']), dT=np.asarray(z['dT_rad'], np.float64),
                     klevs=[int(k) for k in z['klevs']]))
klevs = days[0]['klevs']
assert all(d['klevs'] == klevs for d in days), 'all states must share the band levels'
drv = R.RadiationDriver(C.open_var, klevs, lat, lon, C.DP_BIN * 1e-9, solar='instant', solar_window_h=6.0, solar_align='centred',
                        mmid_kg=C.MMID, rho_dry=C.RHO_AER)
say(f'driver ready {time.time()-t0:.0f}s; nzb={len(klevs)}; {len(days)} day(s); modes {MODES}')


ALPHA0, NODES0 = drv.alpha, drv.nodes


def anomaly(mode, t, num, dT, num_ref, alpha=None, nodes=None):
    """(hr_anom (nzb, nlat, nlon) K/s, arf_toa (nlat, nlon) W/m2) exactly as coupling.py forms them."""
    drv.solar_mode = mode
    drv.alpha = ALPHA0 if alpha is None else alpha
    drv.nodes = NODES0 if nodes is None else nodes
    hr, d1, d0 = drv.heating_anomaly(t, num, num_ref, dT)
    up1 = np.asarray(d1['sw_up_toa']) + np.asarray(d1['olr'])
    up0 = np.asarray(d0['sw_up_toa']) + np.asarray(d0['olr'])
    return np.asarray(hr, np.float64), up0 - up1


def wave4(col):
    """absolute zonal wavenumber-4 amplitude [field units], mean over |lat| <= 30"""
    F = np.fft.rfft(col, axis=-1)
    return float((2 * np.abs(F[..., 4]) / col.shape[-1])[TROP].mean())


bands = {'tropics |lat|<30': np.abs(lat) < 30, 'mid 30-60': (np.abs(lat) >= 30) & (np.abs(lat) < 60),
         'polar >60': np.abs(lat) >= 60}
out = {'lat': lat, 'lon': lon, 'klevs': np.array(klevs), 'modes': np.array(MODES)}
summary = []
S = 86400.0
for di, d in enumerate(days):
    td = time.time(); c = d['hour_c']
    calls = (c, c + 6, c + 12, c + 18)
    num_ref = {t: jnp.asarray(C.bin_mam4(ds_mam, t - C.H0, klevs)[0]) for t in range(c - 2, c + 22)}
    hourly = {t: anomaly('instant', t, d['num'], d['dT'], num_ref[t]) for t in range(c - 2, c + 22)}
    res = {'hourly': (np.mean([v[0] for v in hourly.values()], 0) * S, np.mean([v[1] for v in hourly.values()], 0))}
    for lab, mode, alpha, nodes in SPECS:
        if mode == 'instant':
            vals = [hourly[t] for t in calls]
        else:
            vals = [anomaly(mode, t, d['num'], d['dT'], num_ref[t], alpha, nodes) for t in calls]
        res[lab] = (np.mean([v[0] for v in vals], 0) * S, np.mean([v[1] for v in vals], 0))
    for k, (hr, arf) in res.items():
        out[f'd{di}_{k}_hr'] = hr.astype(np.float32); out[f'd{di}_{k}_arf'] = arf.astype(np.float32)
    ref_hr, ref_arf = res['hourly']
    rcol = ref_hr.sum(0); rrms = np.sqrt(gm(rcol ** 2))
    trms = np.sqrt((rcol[TROP] ** 2 * w_lat[TROP]).sum() / w_lat[TROP].sum())
    say(f'\n=== day {di}: {d["date"]} (state {os.path.basename(d["path"])}), {time.time()-td:.0f}s ===')
    say(f'reference (hourly): band-summed anomaly rms {rrms:.4f} K/day, wave-4 {100*wave4(rcol)/trms:.1f} % of '
        f'its tropical rms, daily-mean ARF {gm(ref_arf):+.4f} W/m2')
    say(f'{"":{LW}s} {"col rms":>8s} {"col max":>8s} {"lev rms":>8s} {"zm bias":>8s} {"w4 err":>7s} '
        f'{"ARF":>9s} {"ARF err":>8s}   ' + '   '.join(f'{b} rms' for b in bands))
    for k in MODES:
        hr, arf = res[k]
        e = hr - ref_hr; ecol = e.sum(0)
        col_rms = np.sqrt(gm(ecol ** 2)) / rrms
        col_max = np.abs(ecol).max() / np.abs(rcol).max()
        lev_rms = np.sqrt(np.mean([gm(e[l] ** 2) for l in range(e.shape[0])])) / \
            np.sqrt(np.mean([gm(ref_hr[l] ** 2) for l in range(e.shape[0])]))
        zm = np.abs(ecol.mean(-1)).max() / np.abs(rcol.mean(-1)).max()
        w4e = wave4(ecol) / trms
        arf_err = 100 * (gm(arf) / gm(ref_arf) - 1)
        b_rms = []
        for b, msk in bands.items():
            wm = w_lat[msk]
            b_rms.append(np.sqrt((ecol[msk] ** 2 * wm).sum() / wm.sum()) / np.sqrt((rcol[msk] ** 2 * wm).sum() / wm.sum()))
        say(f'{k:{LW}s} {100*col_rms:7.1f}% {100*col_max:7.1f}% {100*lev_rms:7.1f}% {100*zm:7.1f}% {100*w4e:6.1f}% '
            f'{gm(arf):+9.4f} {arf_err:+7.1f}%   ' +
            '   '.join(f'{100*x:{len(b)+4}.1f}%' for x, b in zip(b_rms, bands)))
        summary.append((d['date'], k, col_rms, col_max, lev_rms, zm, w4e, arf_err))
# per (day, label) metrics, rows in summary order: col rms, col max, lev rms, zm bias, w4 err (fractions), ARF err [%]
out['dates'] = np.array([d['date'] for d in days])
out['summary'] = np.array([s[2:] for s in summary], np.float64)
out['summary_day'] = np.repeat(np.arange(len(days)), len(MODES))      # summary is day-major
out['summary_mode'] = np.array([s[1] for s in summary])
np.savez(f'{args.out}.npz', **out)
say('\n=== mean over days (errors against hourly radiation) ===')
say(f'{"":{LW}s} {"col rms":>8s} {"col max":>8s} {"lev rms":>8s} {"zm bias":>8s} {"w4 err":>7s} {"ARF err (range)":>22s}')
for k in MODES:
    rows = np.array([s[2:] for s in summary if s[1] == k], np.float64)
    say(f'{k:{LW}s} {100*rows[:,0].mean():7.1f}% {100*rows[:,1].mean():7.1f}% {100*rows[:,2].mean():7.1f}% '
        f'{100*rows[:,3].mean():7.1f}% {100*rows[:,4].mean():6.1f}%  {rows[:,5].mean():+6.1f}% '
        f'({rows[:,5].min():+.1f}..{rows[:,5].max():+.1f})')
say(f'\nsaved {args.out}.npz in {time.time()-t0:.0f}s')
