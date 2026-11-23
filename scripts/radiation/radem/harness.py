#!/usr/bin/env python3
"""Score a candidate aerosol-increment radiation against physics window_gq@3 on
held-out campaign frames (csb_yr inj + csb_yr_ctl, P2@200 met T).

  python harness.py --steps A:B --out DIR --cand instant      # discrimination check
  python harness.py --steps A:B --out DIR --cand gq_self      # floor (truth vs itself, re-solved)
  python harness.py --steps A:B --out DIR --cand emu:CKPTDIR  # the emulator

Truth increments are cached per frame in DIR/truth_<run>_<step>.npz.
Metrics (all vs physics window_gq@3, the model's default radiation):
  per call   local  = rms(err)/rms(truth) of SW+LW heating, cells with column AOD >= 0.01
             column = same on the dp-weighted column integral (per column)
  forcing    global = area-weighted mean of -(d swup + d olr): candidate vs truth, %
             inj-ctl = the same for the inj - ctl difference (both runs at each hour)
  time mean  local  = rms(err)/rms(truth) of the mean over all frames (plume cells)
             wave-4 = rms over plume lat/levels of the zonal wave-4 amplitude of the
                      time-mean error / rms of the time-mean truth (the stripe test)
"""
import os, sys, time, argparse
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))), 'src', 'radiation'))
import radem_lib as L

ap = argparse.ArgumentParser()
ap.add_argument('--steps', required=True, help='a:b[:s] frame steps (6-hourly; 4 per day)')
ap.add_argument('--runs', default='inj,ctl')
ap.add_argument('--out', required=True)
ap.add_argument('--cand', required=True)
ap.add_argument('--aod-min', type=float, default=0.01)
args = ap.parse_args()
os.makedirs(args.out, exist_ok=True)
a, b, *s = (int(x) for x in args.steps.split(':')); steps = list(range(a, b, s[0] if s else 1))
runs = args.runs.split(',')

t0 = time.time()
gq, klevs = L.make_driver('window_gq', 3)
met = L.MetT()
assert met.klevs == klevs
lat = gq.lat
dp = gq.dp_band                                      # (nzb,) Pa, band layer thickness

cand = args.cand
if cand == 'instant':
    inst, _ = L.make_driver('instant')
    predict = lambda hour, num, T: L.increment(inst, hour, num, T)
elif cand == 'gq_self':
    predict = lambda hour, num, T: L.increment(gq, hour, num, T)
elif cand.startswith('emu:'):
    import radem as E
    emu = E.Emulator(cand[4:], gq)
    predict = lambda hour, num, T: emu.increment(hour, num, T)
else:
    raise SystemExit(f'unknown --cand {cand}')
if 'mam4' in runs:
    import coupling as C
    ds_mam = {f'{p}_a{m}': C.open_var(f'{p}_a{m}') for p in ('num', 'so4') for m in (1, 2, 3)}
print(f'drivers ready {time.time()-t0:.0f}s; {len(steps)} steps x {runs}; cand {cand}', flush=True)

def truth(run, step, num, hour, T):
    f = os.path.join(args.out, f'truth_{run}_{step:04d}.npz')
    if os.path.exists(f):
        z = np.load(f); return {k: z[k] for k in z.files}
    d = L.increment(gq, hour, num, T)
    np.savez(f, **d); return d

K = ('sw', 'lw', 'swup', 'olr')
acc = {r: {'t': [], 'c': []} for r in runs}
rows = []
for st in steps:
    for r in runs:
        tf = time.time()
        if r == 'mam4':
            hour = 8766 + 6 * (st + 1); num = np.asarray(C.bin_mam4(ds_mam, hour - C.H0, klevs)[0], np.float32)
        else:
            num, hour = L.load_frame(r, st)
        T = met(hour)
        tr = truth(r, st, num, hour, T)
        tc = time.time()
        cd = predict(hour, num, T)
        tc = time.time() - tc
        ht, hc = tr['sw'] + tr['lw'], cd['sw'] + cd['lw']
        m = (tr['aod'] >= args.aod_min)[None] & np.ones_like(ht, bool)
        loc = np.sqrt(((hc - ht)[m] ** 2).mean() / max((ht[m] ** 2).mean(), 1e-30)) if m.any() else np.nan
        colt = np.tensordot(dp, ht, 1); colc = np.tensordot(dp, hc, 1); mc = m[0]
        col = np.sqrt(((colc - colt)[mc] ** 2).mean() / max((colt[mc] ** 2).mean(), 1e-30)) if mc.any() else np.nan
        Ft = -L.gmean(tr['swup'] + tr['olr'], lat); Fc = -L.gmean(cd['swup'] + cd['olr'], lat)
        rows.append((r, st, hour, loc, col, Ft, Fc, tc))
        acc[r]['t'].append({k: tr[k] for k in K}); acc[r]['c'].append({k: cd[k] for k in K})
        acc[r]['aod'] = tr['aod']
        print(f'  {r} step {st} hour {hour}: local {loc*100:6.2f} %  column {col*100:6.2f} %  '
              f'forcing truth {Ft:+.4f} cand {Fc:+.4f} W/m2 ({(Fc/Ft-1)*100 if Ft else np.nan:+.2f} %)  '
              f'cand {tc:.2f}s  [{time.time()-tf:.1f}s]', flush=True)

def wave4(x):
    """(nzb, nlat, nlon) -> zonal wave-4 amplitude (nzb, nlat)."""
    return 2.0 * np.abs(np.fft.rfft(x, axis=-1)[..., 4]) / x.shape[-1]

print('\n==== SUMMARY vs physics window_gq@3 ====', flush=True)
R = np.array([(x[3], x[4], x[5], x[6]) for x in rows], float)
runs_arr = np.array([x[0] for x in rows])
for r in runs:
    sel = runs_arr == r
    loc, col, Ft, Fc = R[sel].T
    tm_t = {k: np.mean([d[k] for d in acc[r]['t']], 0) for k in K}
    tm_c = {k: np.mean([d[k] for d in acc[r]['c']], 0) for k in K}
    ht = tm_t['sw'] + tm_t['lw']; hc = tm_c['sw'] + tm_c['lw']
    plume = np.abs(lat) <= 60
    # plume cells of the time mean: where the time-mean |heating| exceeds 1 % of its max
    m = np.abs(ht) > 0.01 * np.abs(ht).max()
    tl = np.sqrt(((hc - ht)[m] ** 2).mean() / (ht[m] ** 2).mean())
    w4e = wave4(hc - ht)[:, plume]; zm = np.abs(ht.mean(-1))[:, plume]
    big = zm > 0.05 * zm.max()
    w4 = np.sqrt((w4e[big] ** 2).mean() / (zm[big] ** 2).mean())
    w4t = wave4(ht)[:, plume]
    print(f'{r}: per-call local {np.nanmean(loc)*100:.2f} % (max {np.nanmax(loc)*100:.2f})  column {np.nanmean(col)*100:.2f} %  |  '
          f'forcing mean truth {Ft.mean():+.4f} cand {Fc.mean():+.4f} W/m2 ({(Fc.mean()/Ft.mean()-1)*100:+.2f} %)  |  '
          f'time-mean local {tl*100:.2f} %  wave-4 of error {w4*100:.2f} % (truth own wave-4 {np.sqrt((w4t[big]**2).mean()/(zm[big]**2).mean())*100:.2f} %)', flush=True)
if 'inj' in runs and 'ctl' in runs:
    si, sc = runs_arr == 'inj', runs_arr == 'ctl'
    dFt = R[si, 2] - R[sc, 2]; dFc = R[si, 3] - R[sc, 3]
    print(f'inj-ctl forcing: truth {dFt.mean():+.4f} cand {dFc.mean():+.4f} W/m2 ({(dFc.mean()/dFt.mean()-1)*100:+.2f} %); '
          f'per-call rms err {np.sqrt(((dFc-dFt)**2).mean()):.4f} W/m2', flush=True)
    # local heating of the inj - ctl difference (what a delta-T built from the injected aerosol sees)
    hd = lambda d: d['sw'] + d['lw']
    pc = []
    for dti, dci, dtc, dcc in zip(acc['inj']['t'], acc['inj']['c'], acc['ctl']['t'], acc['ctl']['c']):
        t_ = hd(dti) - hd(dtc); c_ = hd(dci) - hd(dcc)
        m = (acc['inj']['aod'] >= args.aod_min)[None] & np.ones_like(t_, bool)
        pc.append(np.sqrt(((c_ - t_)[m] ** 2).mean() / (t_[m] ** 2).mean()))
    tmt = np.mean([hd(a) - hd(b) for a, b in zip(acc['inj']['t'], acc['ctl']['t'])], 0)
    tmc = np.mean([hd(a) - hd(b) for a, b in zip(acc['inj']['c'], acc['ctl']['c'])], 0)
    m = np.abs(tmt) > 0.01 * np.abs(tmt).max()
    print(f'inj-ctl heating: per-call local {np.mean(pc)*100:.2f} % (max {np.max(pc)*100:.2f})  '
          f'time-mean local {np.sqrt(((tmc-tmt)[m]**2).mean()/(tmt[m]**2).mean())*100:.2f} %', flush=True)
print(f'mean candidate time per call {R.shape and np.mean([x[7] for x in rows]):.3f} s; total {time.time()-t0:.0f}s', flush=True)
np.savez(os.path.join(args.out, f'summary_{cand.replace(":", "_").replace("/", "_")}.npz'),
         rows=np.array(rows, dtype=object), steps=steps)
