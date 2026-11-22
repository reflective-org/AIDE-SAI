"""Uniform-field drift + fixer-factor diagnostic for sl_lr on real CESM winds.

q = 1 everywhere (reservoir 1 too) must stay 1 under any consistent transport.
With ADV_SL_RHONORM=0 (the default before 2026-11-22) sl_lr drifts (~1e-3 in 2 days, +2 % level mean at 1.24 hPa
by day 10) while the budget closes to roundoff, and with fixer=none it stays
within ~1e-6, so the drift is something the mass fixers imprint. This runs one
configuration for STEPS 6-h steps (1996-01-01 00Z on, 1-150 hPa band, f32
gather like production) with ADV_SL_FIXLOG=1 and records, per sub-step:
  level   M1/M0 - 1 the horizontal fixer removes, per level
  column  the column fixer's correction
  air_res / air_cap  the level air-mass imbalance from resolved rows / caps
plus per-step level means and rms of q - 1. Compare configurations to see
whether the caps are the cause (lat_freeze=90 removes them, polar_mode=freeze
resets them, fixer=none is the control).

  CUDA_VISIBLE_DEVICES= JAX_PLATFORMS=cpu python3 sl_fixer_cap_diag.py OUTDIR [--steps=40] [kw=val ...]
kw go to sl_lr.advect_step_batch (nsub, fixer, jac, lat_freeze, polar_mode, ...).
Writes OUTDIR/<tag>.npz and prints one JSON row per step.
"""
import os, sys, time, json
os.environ['ADV_SL_FIXLOG'] = '1'
import numpy as np
outdir = sys.argv[1]; os.makedirs(outdir, exist_ok=True)
steps = 40; kw = {}; tag = 'sl'
for a in sys.argv[2:]:
    if a.startswith('--steps='):
        steps = int(a.split('=')[1]); continue
    k, v = a.split('=', 1)
    for cast in (int, float):
        try:
            v = cast(v); break
        except ValueError:
            pass
    if v in ('True', 'False'):
        v = v == 'True'
    kw[k] = v; tag += f'_{k}{v}'
SRC = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'src')
sys.path.insert(0, os.path.abspath(SRC)); import _paths  # noqa
import jax, jax.numpy as jnp
jax.config.update('jax_enable_x64', True)
import xarray as xr
from fct_fast import grid_metric
import sl_lr as M

RUN = 'f.e21.FWHIST.f09_f09_mg17.atmos-scale_fixedSST_1996-2014.001'
TS = f'/data/cesm2.1.5_output/histSST/{RUN}/archive/atm/proc/tseries/hour_1'
F = lambda v: f'{TS}/{RUN}.cam.h1.{v}.1996010100-2014123100.nc'
ds = {v: xr.open_dataset(F(v), decode_times=False) for v in ('U', 'V', 'OMEGA')}
lat = ds['U'].lat.values; lon = ds['U'].lon.values
plev = ds['U'].hyam.values * ds['U'].P0.values + ds['U'].hybm.values * 1e5
band = np.where((plev >= 100.0) & (plev <= 15000.0))[0]
kl = list(range(band.min(), band.max() + 1))
PLEV = plev[kl]; DP = np.gradient(PLEV)
rd = lambda v, t: ds[v][v].isel(time=t, lev=kl).values
nlev, nlat, nlon = len(kl), lat.size, lon.size
_, ac = grid_metric(lat, None)
Wm = DP[:, None, None] * ac[None, :, None] * np.ones((1, 1, nlon))
Mq = lambda q: float((np.asarray(q, np.float64) * Wm).sum())
q0 = np.ones((1, nlev, nlat, nlon), np.float64)
ckw = dict(lat=lat, dp=DP, qfrozb=jnp.asarray(np.ones_like(q0)), dt_total=21600.0, lat_freeze=80.0,
           dtype=jnp.float32, plev=PLEV, return_vflux=True)
ckw.update(kw)
print(f"[{tag}] {M.describe()}; kw={kw}; {nlat}x{nlon}x{nlev} {PLEV[0]/100:.2f}..{PLEV[-1]/100:.1f} hPa; "
      f"{jax.devices()}", flush=True)
q = jnp.asarray(q0); M0 = Mq(q0); cum = 0.0
rows, lev_mean, lev_rms, logs = [], [], [], {k: [] for k in ('level', 'column', 'air_res', 'air_cap')}
w_next = tuple(rd(v, 0) for v in ('U', 'V', 'OMEGA'))
for s in range(steps):
    w_now, w_next = w_next, tuple(rd(v, 6 * (s + 1)) for v in ('U', 'V', 'OMEGA'))
    ta = time.time()
    out = M.advect_step_batch(q, *w_now, *w_next, **ckw)
    qn = np.asarray(out[0]); vfl = np.asarray(out[2])
    L = M.fixlog_take()
    for k in logs:
        logs[k].append(L[k])
    fin = float((vfl[0, 0] * ac[:, None]).sum()); fout = float((vfl[0, 1] * ac[:, None]).sum())
    cum += fin - fout
    d = qn[0] - 1.0
    lm = (d * Wm).sum(axis=(1, 2)) / Wm.sum(axis=(1, 2)); lev_mean.append(lm)
    lev_rms.append(np.sqrt((d ** 2).mean(axis=(1, 2))))
    cap = np.abs(lat) > 80
    row = dict(step=s + 1, day=(s + 1) / 4, sec=round(time.time() - ta, 1),
               min=float(qn.min()), max=float(qn.max()),
               lev_mean_top=float(lm[0]), lev_mean_absmax=float(np.abs(lm).max()),
               rms_caps=float(np.sqrt((d[:, cap] ** 2).mean())),
               rms_resolved=float(np.sqrt((d[:, ~cap] ** 2).mean())),
               fix_level_absmax=float(np.abs(L['level']).max()) if L['level'].size else 0.0,
               fix_level_absmean=float(np.abs(L['level']).mean()) if L['level'].size else 0.0,
               fix_column_absmax=float(np.abs(L['column']).max()) if L['column'].size else 0.0,
               air_cap_absmax=float(np.abs(L['air_cap']).max()) if L['air_cap'].size else 0.0,
               air_res_absmax=float(np.abs(L['air_res']).max()) if L['air_res'].size else 0.0,
               residcum_M0=(Mq(qn) / M0 - 1) - cum / M0)
    rows.append(row); print(json.dumps(row), flush=True)
    q = jnp.asarray(qn)
    np.savez(os.path.join(outdir, f'{tag}.npz'), rows=json.dumps(rows), plev=PLEV, lat=lat,
             lev_mean=np.array(lev_mean), lev_rms=np.array(lev_rms), q_final=qn,
             **{f'log_{k}': np.concatenate(v) if v and v[0].size else np.zeros(0) for k, v in logs.items()})
print(f"[{tag}] done; final level mean(q-1) by hPa:", flush=True)
for kk in range(nlev):
    print(f"   {PLEV[kk]/100:8.2f}  {lev_mean[-1][kk]:+.2e}  rms {lev_rms[-1][kk]:.2e}", flush=True)
