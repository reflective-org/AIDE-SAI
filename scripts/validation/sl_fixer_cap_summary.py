"""Summarise outputs/sl_fixer_cap_diag/*.npz: drift and fixer factors per config."""
import sys, glob, json, os, numpy as np
out = sys.argv[1]
print(f"{'config':34s} {'min q':>9s} {'max q':>9s} {'|lev mean|max':>13s} {'top lev mean':>12s} "
      f"{'rms caps':>9s} {'rms res':>9s} | {'fix lvl max':>11s} {'fix lvl mean':>12s} {'col max':>9s} "
      f"{'air cap':>9s} {'air res':>9s}  (day 10, fixer stats over all sub-steps)")
for f in sorted(glob.glob(os.path.join(out, '*.npz'))):
    d = np.load(f); r = json.loads(str(d['rows'])); z = r[-1]
    lv = d['log_level']; cl = d['log_column']; ac = d['log_air_cap']; ar = d['log_air_res']
    mx = lambda a: np.abs(a).max() if a.size else 0.0
    print(f"{os.path.basename(f)[:-4]:34s} {z['min']:9.6f} {z['max']:9.6f} {z['lev_mean_absmax']:13.2e} "
          f"{z['lev_mean_top']:+12.2e} {z['rms_caps']:9.2e} {z['rms_resolved']:9.2e} | {mx(lv):11.2e} "
          f"{(np.abs(lv).mean() if lv.size else 0):12.2e} {mx(cl):9.2e} {mx(ac):9.2e} {mx(ar):9.2e}   (day {z['day']})")
    if lv.size:   # per-level: mean signed fixer error vs mean signed cap / resolved air imbalance
        p = d['plev'] / 100
        print("    hPa   mean(M1/M0-1)  mean(air_cap)  mean(air_res)  final mean(q-1)")
        for k in [0, 1, 2, 4, 8, len(p) - 1]:
            print(f"   {p[k]:7.2f}  {lv[:, 0, k].mean():+.2e}      {ac[:, k].mean() if ac.size else 0:+.2e}      "
                  f"{ar[:, k].mean() if ar.size else 0:+.2e}      {d['lev_mean'][-1][k]:+.2e}")
