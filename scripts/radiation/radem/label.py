#!/usr/bin/env python3
"""Labels for the aerosol-increment emulator: physics (RRTMGP, one solve each
with and without aerosol) on run states, at per-column sun angles drawn
like window_gq's Gauss nodes. (AIDE-SAI port of AIDE-SAI-link scripts/radem/label.py;
grid-generic: the linking comes from --preset.)

Per globe:
  state  = a frames3d frame of a run (dir=<frames3d>, at ITS OWN hour; dirdt= adds the
           frame's dT_rad to T), a 3-D state file (state=<npz with num + hour>, at a random
           hour of --steps), or the MAM4 reference binned at a random hour of --steps
           (gases, surface and the met T of that hour)
  aerosol x s_col per column: P_ONE as is, P_ZERO zero (structural zero),
           rest log-uniform [S_MIN, S_MAX] (heavier / lighter burdens)
  T      = met T (+ with prob P_TPERT a smooth random profile, amplitude <= T_AMP K:
           room for the fine-tuned atmosphere)
  mu     = per column: with prob P_NODE one of the hour's window_gq node mu's
           (when that node is sunlit), otherwise U(MU_MIN, 1)
  albedo = the window albedo window_gq uses
Stored per kept column (float32): gas columns T q o3 ch4 (nzc), ts alb mu,
SW and LW band optics on the band levels (tau ssa g), and the targets
dsw dlw (nzb, K/day at unit TSI), dswup dolr (W/m2), aod. Shards <out>/g<NNNN>.npz.
"""
import os, sys, time, argparse, glob
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import radem_lib as L

ap = argparse.ArgumentParser()
ap.add_argument('--preset', required=True, help='the linking whose grid/met/climatology the labels are made on')
ap.add_argument('--steps', default='0:1460', help='a:b 6-h step range (from H0) for the mam4 / state= sources')
ap.add_argument('--runs', required=True,
                help='comma list, repeat to weight: mam4 | dir=<frames3d dir> | dirdt=<frames3d dir> | state=<npz> '
                     '(dirdt: T = met + the frame\'s own dT_rad)')
ap.add_argument('--holdout', default='', help='lo:hi day-of-month range (1-based, inclusive) never drawn, e.g. 6:16')
ap.add_argument('--only-days', default='', help='lo:hi day-of-month range: draw ONLY these days (validation on the holdout)')
ap.add_argument('--n-globes', type=int, required=True)
ap.add_argument('--start', type=int, default=0)
ap.add_argument('--out', required=True)
ap.add_argument('--seed', type=int, default=0)
ap.add_argument('--keep', type=int, default=12000)
ap.add_argument('--s-min', type=float, default=0.3); ap.add_argument('--s-max', type=float, default=3.0)
ap.add_argument('--p-one', type=float, default=0.3); ap.add_argument('--p-zero', type=float, default=0.08)
ap.add_argument('--p-tpert', type=float, default=0.5); ap.add_argument('--t-amp', type=float, default=4.0)
ap.add_argument('--p-node', type=float, default=0.6); ap.add_argument('--mu-min', type=float, default=0.005)
ap.add_argument('--no-aug', action='store_true', help='states as they are, real node mu (validation)')
args = ap.parse_args()
os.makedirs(args.out, exist_ok=True)
rng = np.random.default_rng(args.seed * 100003 + args.start)

L.apply_preset(args.preset)
import jax, jax.numpy as jnp
import radiation as R
import rad_features as F
t0 = time.time()
drv, klevs, C = L.make_driver('instant')
met = L.MetT(drv)
STATES = {}
nzb = len(klevs); nlat, nlon = drv.nlat, drv.nlon; ncol = nlat * nlon; nzc = drv.nzc
a, b = (int(x) for x in args.steps.split(':'))
runs = args.runs.split(',')
DIRFILES = {}
HOLD = tuple(int(x) for x in args.holdout.split(':')) if args.holdout else None
ONLY = tuple(int(x) for x in args.only_days.split(':')) if args.only_days else None
MDAYS = np.cumsum([0, 31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31])
def day_of_month(hour_abs):
    doy = ((hour_abs - 1) // 24) % 365          # the call at hour h stands for the 6 h before it
    m = int(np.searchsorted(MDAYS, doy, side='right')) - 1
    return int(doy - MDAYS[m]) + 1
ds_mam = {f'{p}_a{m}': C.open_var(f'{p}_a{m}') for p in ('num', 'so4') for m in (1, 2, 3)} if 'mam4' in runs else None
plev_band = None
print(f'driver ready {time.time()-t0:.0f}s nzc={nzc} nzb={nzb} nb_sw={drv.nb_sw} nb_lw={drv.nb_lw}', flush=True)


def smooth_profile(n):
    """(nzb,) smooth random profile in [-1, 1]-ish: 3 cosine modes in level index."""
    z = np.linspace(0, np.pi, n)
    c = rng.normal(0, 1, 3) / np.sqrt(3)
    return sum(c[k] * np.cos((k + 1) * z + rng.uniform(0, 2 * np.pi)) for k in range(3))


Cc = R.RAD_LAT_CHUNK
chunks = [(i, min(i + Cc, nlat)) for i in range(0, nlat, Cc)]
for g in range(args.start, args.n_globes):
    tg = time.time()
    run = runs[int(rng.integers(len(runs)))]
    for _try in range(1000):
        if run.startswith('state='):
            st = int(rng.integers(a, b)); hour = C.H0 + 6 * (st + 1)
            if run not in STATES:
                STATES[run] = np.asarray(np.load(run.split('=', 1)[1])['num'], np.float32)
            num0 = STATES[run]; dT0 = None
        elif run == 'mam4':
            st = int(rng.integers(a, b)); hour = C.H0 + 6 * (st + 1)
            num0 = np.asarray(C.bin_mam4(ds_mam, hour - C.H0, klevs)[0], np.float32); dT0 = None
        else:
            kind_, d_ = run.split('=', 1)
            fs_ = DIRFILES.setdefault(d_, sorted(glob.glob(os.path.join(d_, 'step_*.npz'))))
            st = int(rng.integers(len(fs_))); z_ = np.load(fs_[st])
            num0 = np.asarray(z_['num'], np.float32); hour = int(z_['hour_abs'])
            dT0 = np.asarray(z_['dT_rad'], np.float64) if kind_ == 'dirdt' else None
        dm = day_of_month(hour)
        if (not HOLD or not (HOLD[0] <= dm <= HOLD[1])) and (not ONLY or ONLY[0] <= dm <= ONLY[1]):
            break
    Tb = met(hour) + (dT0 if dT0 is not None else 0.0)
    # ---- augmentation ----
    if args.no_aug:
        s_col = np.ones((nlat, nlon), np.float32); tamp = 0.0
    else:
        s_col = np.exp(rng.uniform(np.log(args.s_min), np.log(args.s_max), (nlat, nlon))).astype(np.float32)
        r = rng.random((nlat, nlon))
        s_col[r < args.p_one] = 1.0
        s_col[(r >= args.p_one) & (r < args.p_one + args.p_zero)] = 0.0
        tamp = args.t_amp * rng.random() if rng.random() < args.p_tpert else 0.0
        if tamp > 0:
            # one profile shape per globe, amplitude varying smoothly-ish per column
            prof = smooth_profile(nzb)
            Tb = Tb + tamp * prof[:, None, None] * rng.uniform(0.3, 1.0, (1, nlat, nlon))
    num = num0 * s_col[None, None]
    # ---- geometry: node mu's of this hour's window_gq call, or uniform ----
    nodes = R.solar_gauss(drv.time[hour], drv.lat, drv.lon, drv.solar_window_h, 3)
    mus = np.stack([np.where(sc > 0, np.cos(z), 0.0) for z, sc in nodes])          # (3, nlat, nlon)
    pick = rng.integers(0, 3, (nlat, nlon))
    mu_node = np.take_along_axis(mus, pick[None], 0)[0]
    mu_uni = rng.uniform(args.mu_min, 1.0, (nlat, nlon))
    if args.no_aug:
        mu = np.where(mu_node > 0, mu_node, mu_uni)
    else:
        mu = np.where((rng.random((nlat, nlon)) < args.p_node) & (mu_node > 0), mu_node, mu_uni)
    zen = np.arccos(np.clip(mu, 0.0, 1.0))
    alb = drv._window_albedo(hour, -drv.solar_window_h / 2, drv.solar_window_h / 2)
    geom = (zen, np.ones((nlat, nlon)), alb)
    inc = L.increment(drv, hour, num, Tb, geom=geom)
    # ---- inputs, the way the emulator will build them ----
    h = F.prep_hour(drv, hour)
    T_full = h['T'].copy(); T_full[drv.band_pos] = Tb
    wwt = F.wt_weights_for(drv, T_full, h['q'])
    num_j = jnp.asarray(num)
    opt = []
    for i0, i1 in chunks:
        ww = jnp.asarray(wwt[:, :, i0:i1]) if wwt is not None else None
        opt.append([np.asarray(o) for o in F.optics_chunk(drv, num_j[:, :, i0:i1], ww)])
    opt = [np.concatenate([o[k] for o in opt], axis=2) for k in range(6)]   # each (nb, nzb, nlat, nlon)
    # ---- stratified keep: plume columns preferred ----
    aod = inc['aod'].reshape(-1); sz = (s_col == 0).reshape(-1)
    p = np.full(ncol, 0.10); p[aod >= 1e-3] = 0.4; p[aod >= 1e-2] = 1.0; p[sz] = 0.15
    keep = np.nonzero(rng.random(ncol) < p)[0]
    if keep.size > args.keep:
        keep = np.sort(rng.choice(keep, args.keep, replace=False))
    yi, xi = np.unravel_index(keep, (nlat, nlon))
    col = lambda x: np.asarray(x)[..., yi, xi]
    np.savez(os.path.join(args.out, f'g{g:04d}.npz'),
             T=col(T_full).T.astype(np.float32), q=col(h['q']).T.astype(np.float32),
             o3=col(h['o3']).T.astype(np.float32), ch4=col(h['ch4']).T.astype(np.float32),
             ts=col(h['ts']).astype(np.float32), alb=col(alb).astype(np.float32), mu=col(mu).astype(np.float32),
             tau_sw=np.moveaxis(col(opt[0]), -1, 0).astype(np.float32), ssa_sw=np.moveaxis(col(opt[1]), -1, 0).astype(np.float32),
             g_sw=np.moveaxis(col(opt[2]), -1, 0).astype(np.float32), tau_lw=np.moveaxis(col(opt[3]), -1, 0).astype(np.float32),
             ssa_lw=np.moveaxis(col(opt[4]), -1, 0).astype(np.float32), g_lw=np.moveaxis(col(opt[5]), -1, 0).astype(np.float32),
             dsw=col(inc['sw']).T.astype(np.float32), dlw=col(inc['lw']).T.astype(np.float32),
             dswup=col(inc['swup']).astype(np.float32), dolr=col(inc['olr']).astype(np.float32),
             aod=col(inc['aod']).astype(np.float32), s_col=col(s_col), lat_i=yi.astype(np.int16), lon_i=xi.astype(np.int16),
             run=run, step=st, hour=hour, tamp=tamp)
    print(f'[{g+1}/{args.n_globes}] {run} step {st} hour {hour} tamp {tamp:.1f}: kept {keep.size} '
          f'(aod>=1e-2 {(aod[keep]>=1e-2).sum()}); dsw p99 {np.percentile(np.abs(inc["sw"]),99):.4f} '
          f'max {np.abs(inc["sw"]).max():.3f}; dlw max {np.abs(inc["lw"]).max():.3f} K/d  ({time.time()-tg:.1f}s)', flush=True)
print(f'LABEL_DONE {args.n_globes - args.start} globes in {time.time()-t0:.0f}s', flush=True)
