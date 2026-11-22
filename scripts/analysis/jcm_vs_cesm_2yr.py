#!/usr/bin/env python3
"""JCM vs CESM meteorology, 2-year SAI comparison (6 Tg SO2/yr, 3+3 at +-30 deg, 180E, 55 hPa,
1996-1997): the same scenario on the cesm_numerical and jcm_numerical linkings.

Read-only on the run dirs; CPU only, ~1 min. Writes img/*.png and metrics.json in the CURRENT
WORKING DIRECTORY. Each run directory holds coupled_*_<TAG>.npz with TAG = the directory's name.

  CESM injection: timeseries from --cesm-inj (2920 steps, year 2 resumed);
                  zonal frames = --cesm-inj-y1 (days 0..365) + --cesm-inj (days 366..730)
  CESM control:   --cesm-ctrl timeseries (reference only)
  JCM injection:  --jcm-inj (2920 steps, 731 zonal frames; resumed in place)
No JCM control exists, so the budget below is change-from-day-0 for both runs.

  python3 jcm_vs_cesm_2yr.py [--cesm-inj DIR] [--cesm-inj-y1 DIR] [--cesm-ctrl DIR] [--jcm-inj DIR]

Defaults: runs/fast_2yr_inj6_30NS, runs/fast_y1frames_inj6, runs/fast_2yr_ctrl,
runs/jcm_1yr_inj6_30NS of this checkout (env RUN_INJ, RUN_INJ_Y1, RUN_CTRL, RUN_JCM_INJ).
Burden units as scripts/utils/plot_run.py (BURDEN_KG from the grid); dry diameter from the
bin boundary masses xk.
"""
import argparse
import json
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

RUNS = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'runs')
_ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
for _flag, _env, _dflt in (('--cesm-inj', 'RUN_INJ', 'fast_2yr_inj6_30NS'),
                           ('--cesm-inj-y1', 'RUN_INJ_Y1', 'fast_y1frames_inj6'),
                           ('--cesm-ctrl', 'RUN_CTRL', 'fast_2yr_ctrl'),
                           ('--jcm-inj', 'RUN_JCM_INJ', 'jcm_1yr_inj6_30NS')):
    _ap.add_argument(_flag, default=os.environ.get(_env) or os.path.join(RUNS, _dflt),
                     help=f'run directory (env {_env}; default runs/{_dflt})')
_a = _ap.parse_args()
D_INJ, D_Y1, D_CTRL, D_JCM = (os.path.abspath(x).rstrip('/')
                              for x in (_a.cesm_inj, _a.cesm_inj_y1, _a.cesm_ctrl, _a.jcm_inj))
T_INJ, T_Y1, T_CTRL, T_JCM = (os.path.basename(x) for x in (D_INJ, D_Y1, D_CTRL, D_JCM))
OUT = os.path.abspath('.')
IMG = f'{OUT}/img'
os.makedirs(IMG, exist_ok=True)
R_E, GRAV, RHO_AER = 6.371e6, 9.80665, 1770.0
S_SO4, S_SO2 = 32.06 / 96.06, 32.06 / 64.06
C_CESM, C_JCM, C_CTL = '#c4511b', '#1f63b4', '#8a8a8a'
plt.rcParams.update({'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False,
                     'figure.dpi': 110, 'savefig.dpi': 150, 'savefig.bbox': 'tight'})
M = {}


def lat_edges(lat):
    e = np.empty(lat.size + 1)
    e[1:-1] = 0.5 * (lat[:-1] + lat[1:])
    e[0] = lat[0] if abs(lat[0]) >= 89.99 else -90.0
    e[-1] = lat[-1] if abs(lat[-1]) >= 89.99 else 90.0
    return e


def grid(lat, nlon):
    e = np.deg2rad(lat_edges(lat)); dphi = np.deg2rad(lat[1] - lat[0])
    wl = (np.sin(e[1:]) - np.sin(e[:-1])) / dphi
    bkg = R_E ** 2 * dphi * (2 * np.pi / nlon) / GRAV
    area = R_E ** 2 * 2 * np.pi * (np.sin(e[1:]) - np.sin(e[:-1]))
    return wl, bkg, area


def daily(x):
    x = np.asarray(x, float); return x[:len(x) // 4 * 4].reshape(-1, 4).mean(1)


def win(x, a, b):          # mean of a daily series over days [a, b)
    return float(np.mean(np.asarray(x)[a:b]))


# ------------------------------------------------------------------ load
R = {}
for run, tsf, frs in (
        ('cesm', f'{D_INJ}/coupled_timeseries_{T_INJ}.npz',
         [f'{D_Y1}/coupled_frames_{T_Y1}.npz',
          f'{D_INJ}/coupled_frames_{T_INJ}.npz']),
        ('jcm', f'{D_JCM}/coupled_timeseries_{T_JCM}.npz',
         [f'{D_JCM}/coupled_frames_{T_JCM}.npz']),
        ('ctrl', f'{D_CTRL}/coupled_timeseries_{T_CTRL}.npz', [])):
    ts = np.load(tsf)
    assert ts['hours'].size == 2920 and ts['hours'][-1] == 17520, (run, ts['hours'][[0, -1]])
    d = {'ts': {k: np.asarray(ts[k]) for k in ts.files}}
    if frs:
        f0 = np.load(frs[0])
        lat = np.asarray(f0['lat'], float); nlon = len(f0['lon']) if 'lon' in f0.files else 288
        d['lat'] = lat; d['plev'] = np.asarray(f0['plev_pa'], float); d['xk'] = np.asarray(f0['xk'], float)
        hrs, zn, zm = [], [], []
        for i, p in enumerate(frs):
            f = np.load(p)
            h = np.asarray(f['frames_zm_hours']); sl = slice(None) if i == 0 else slice(1, None)
            hrs.append(h[sl]); zn.append(np.asarray(f['frames_zm_num'])[sl]); zm.append(np.asarray(f['frames_zm_mas'])[sl])
        d['fh'] = np.concatenate(hrs); d['zn'] = np.concatenate(zn); d['zm'] = np.concatenate(zm)
        assert d['fh'].size == 731 and np.all(np.diff(d['fh']) == 24), (run, d['fh'][[0, -1]])
        d['wl'], d['bkg'], d['area'] = grid(lat, nlon)
        print(run, 'frames', d['zn'].shape, 'BURDEN_KG', f"{d['bkg']:.6e}")
    R[run] = d
R['ctrl']['bkg'] = R['cesm']['bkg']
day = np.arange(730) + 0.5

# ------------------------------------------------------------------ fig A: forcing, AOD, burden
fig, ax = plt.subplots(3, 1, figsize=(9, 9), sharex=True)
for run, c, lab, ls in (('cesm', C_CESM, 'CESM injection', '-'), ('jcm', C_JCM, 'JCM injection', '-'),
                        ('ctrl', C_CTL, 'CESM control', '--')):
    t = R[run]['ts']; tg = R[run]['bkg'] / 1e9
    arf = daily(t['arf_toa']); aod = daily(t['aod550']); so4 = daily(t['Mburden']) * tg; so2 = daily(t['SO2burden']) * tg
    ax[0].plot(day, arf, color=c, ls=ls, lw=1.4, label=lab)
    ax[1].plot(day, aod, color=c, ls=ls, lw=1.4, label=lab)
    ax[2].plot(day, so4, color=c, ls=ls, lw=1.6, label=f'{lab}: SO$_4$')
    if run != 'ctrl':
        ax[2].plot(day, so2, color=c, ls=':', lw=1.4, label=f'{lab}: SO$_2$')
    for k, x in (('arf', arf), ('aod', aod), ('so4_Tg', so4), ('so2_Tg', so2)):
        M[f'{run}_{k}_y1'] = win(x, 0, 365); M[f'{run}_{k}_y2'] = win(x, 365, 730)
        M[f'{run}_{k}_l30'] = win(x, 700, 730); M[f'{run}_{k}_2yr'] = win(x, 0, 730)
    M[f'{run}_so4_Tg_day0'] = float(t['M0'] * tg)
for a in ax:
    a.axvline(365, color='0.6', lw=0.7)
ax[0].set_ylabel('TOA aerosol effect\nvs MAM4 reference [W m$^{-2}$]'); ax[0].axhline(0, color='0.5', lw=0.5)
ax[1].set_ylabel('550 nm AOD, band, global mean')
ax[2].set_ylabel('band burden [Tg]'); ax[2].set_xlabel('day since 1996-01-01 (line = start of 1997)')
ax[0].legend(frameon=False, fontsize=9); ax[2].legend(frameon=False, fontsize=8, ncol=2)
fig.tight_layout(); fig.savefig(f'{IMG}/fig_forcing_aod_burden.png'); plt.close(fig)

# ------------------------------------------------------------------ budget (change from day 0, both runs)
for run in ('cesm', 'jcm'):
    t = R[run]['ts']; tg = R[run]['bkg'] / 1e9
    for i, e in (('y1', 1459), ('y2', 2919)):
        injS = t['injSO2_cum'][e] * tg * S_SO2
        dso4 = (t['Mburden'][e] - t['M0']) * tg * S_SO4
        dso2 = (t['SO2burden'][e] - t['SO2burden'][0]) * tg * S_SO2
        dh2 = (t['H2SO4burden'][e] - t['H2SO4burden'][0]) * tg * 32.06 / 98.08
        sett = t['settleM_cum'][e] * tg * S_SO4
        M[f'{run}_S_{i}'] = dict(injected=float(injS), so4_added=float(dso4), so2_added=float(dso2),
                                 h2so4_added=float(dh2), settled=float(sett),
                                 left_band_other=float(injS - dso4 - dso2 - dh2 - sett))
    M0 = float(t['M0'])
    for k in ('B_micro', 'B_adv_np', 'B_settle', 'B_adv_pol'):
        M[f'{run}_cum_{k}_TgSO4'] = float(t[k][-1] * M0 * tg)
    bsum = sum(t[k] for k in ('B_adv_np', 'B_adv_pol', 'B_floor', 'B_micro', 'B_bc', 'B_settle') if k in t)
    M[f'{run}_budget_closure'] = float(np.max(np.abs(bsum - (t['Mburden'] / M0 - 1))))

fig, ax = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
parts = [('so4_added', 'added SO$_4$ in band', '#2a78d6'), ('so2_added', 'added SO$_2$ in band', '#e07a5f'),
         ('settled', 'settled out of band bottom', '#edae49'),
         ('left_band_other', 'left through the faces (advection) + H$_2$SO$_4$', '#bdbdbd')]
for a, i, ttl in ((ax[0], 'y1', 'end of 1996 (day 365)'), (ax[1], 'y2', 'end of 1997 (day 730)')):
    for x, run, lab in ((0, 'cesm', 'CESM'), (1, 'jcm', 'JCM')):
        b = 0.0
        for k, l, c in parts:
            v = M[f'{run}_S_{i}'][k]
            a.bar(x, v, bottom=b, color=c, width=0.6, label=l if x == 0 else None)
            if abs(v) > 0.15:
                a.text(x, b + v / 2, f'{v:.2f}', ha='center', va='center', fontsize=8)
            b += v
        a.text(x, b + 0.08, f"injected {M[f'{run}_S_{i}']['injected']:.2f}", ha='center', fontsize=8)
    a.set_xticks([0, 1], ['CESM', 'JCM']); a.set_title(ttl)
ax[0].set_ylabel('Tg S (change since day 0)')
ax[1].legend(frameon=False, fontsize=8, loc='upper left', bbox_to_anchor=(1.0, 1.0))
fig.tight_layout(); fig.savefig(f'{IMG}/fig_sulfur_budget.png'); plt.close(fig)

# ------------------------------------------------------------------ frames: column SO4, shares, size
BANDS = {'tropics |lat|<20': lambda la: np.abs(la) < 20, 'mid-lat 20-60': lambda la: (np.abs(la) >= 20) & (np.abs(la) < 60),
         'polar >60': lambda la: np.abs(la) >= 60}
for run in ('cesm', 'jcm'):
    d = R[run]; lat = d['lat']; DP = np.gradient(d['plev'])
    xk = d['xk']; Dpe = (6 * xk / (np.pi * RHO_AER)) ** (1 / 3) * 1e9
    d['Dp'] = np.sqrt(Dpe[:-1] * Dpe[1:]); d['dlog'] = np.log10(Dpe[1:] / Dpe[:-1])
    coln = np.einsum('fbkj,k->fbj', d['zn'], DP) / GRAV         # #/m2 per bin per lat
    colm = np.einsum('fbkj,k->fbj', d['zm'], DP) / GRAV         # kg/m2
    d['colm'] = colm.sum(1) * 1e6                               # mg/m2 (731, nlat)
    tot = (colm.sum(1) * d['area']).sum(1)                      # kg
    # excess over day 0 (initial MAM4 background)
    ex = (colm.sum(1) - colm[0].sum(0)[None]) * d['area'][None]
    d['share'] = {b: (ex * fn(lat)[None]).sum(1) / ex.sum(1) for b, fn in BANDS.items()}
    w = d['wl']
    cn = (coln * w).sum(-1) / w.sum(); cm = (colm * w).sum(-1) / w.sum()
    d['sdN'] = cn / d['dlog']; d['sdM'] = cm * 1e6 / d['dlog']
    d['reff'] = 0.5 * (cn * d['Dp'] ** 3).sum(1) / (cn * d['Dp'] ** 2).sum(1)
    d['Dgm'] = np.exp((cm * np.log(d['Dp'])).sum(1) / cm.sum(1))
    d['N10'] = cn[:, d['Dp'] > 10].sum(1); d['N100'] = cn[:, d['Dp'] > 100].sum(1)
    M[f'{run}_fr_check_so4_Tg_day730'] = float(tot[-1] / 1e9)
    for b in BANDS:
        M[f'{run}_share_{b}_day365'] = float(d['share'][b][365]); M[f'{run}_share_{b}_day730'] = float(d['share'][b][730])
    for k in ('reff', 'Dgm', 'N10', 'N100'):
        M[f'{run}_{k}_day365'] = float(d[k][365]); M[f'{run}_{k}_day730'] = float(d[k][730])
        M[f'{run}_{k}_y2mean'] = float(d[k][366:].mean())
    del d['zn'], d['zm']

# Hovmoeller: column SO4 vs lat x day
vmax = max(np.percentile(R[r]['colm'], 99.5) for r in ('cesm', 'jcm'))
fig, ax = plt.subplots(2, 1, figsize=(10, 6.5), sharex=True)
for a, run, ttl in ((ax[0], 'cesm', 'CESM'), (ax[1], 'jcm', 'JCM')):
    d = R[run]
    pc = a.pcolormesh(np.arange(731), d['lat'], d['colm'].T, cmap='magma_r', vmin=0, vmax=vmax, shading='auto')
    a.axvline(365, color='w', lw=0.8); a.set_ylabel('latitude'); a.set_yticks([-90, -60, -30, 0, 30, 60, 90])
    a.set_title(f'{ttl}: zonal-mean band-column sulfate')
ax[1].set_xlabel('day since 1996-01-01')
fig.colorbar(pc, ax=ax, label='SO$_4$ column [mg m$^{-2}$]', shrink=0.8)
fig.savefig(f'{IMG}/fig_column_sulfate_hov.png'); plt.close(fig)

# latitude share of added sulfate
fig, ax = plt.subplots(1, 3, figsize=(12, 3.6), sharey=True)
for a, b in zip(ax, BANDS):
    for run, c in (('cesm', C_CESM), ('jcm', C_JCM)):
        a.plot(np.arange(731)[30:], 100 * R[run]['share'][b][30:], color=c, lw=1.5, label=run.upper())
    a.axvline(365, color='0.6', lw=0.7); a.set_title(b); a.set_xlabel('day')
ax[0].set_ylabel('% of added SO$_4$ (vs day 0)'); ax[0].legend(frameon=False)
fig.tight_layout(); fig.savefig(f'{IMG}/fig_latitude_share.png'); plt.close(fig)

# size: distributions at day 365 / 730 + size and number over time
fig, ax = plt.subplots(1, 3, figsize=(13, 3.9))
for run, c in (('cesm', C_CESM), ('jcm', C_JCM)):
    d = R[run]
    for f, ls in ((365, '--'), (730, '-')):
        ax[0].plot(d['Dp'], d['sdN'][f] - d['sdN'][0], color=c, ls=ls, lw=1.4, label=f'{run.upper()} day {f}')
        ax[1].plot(d['Dp'], d['sdM'][f] - d['sdM'][0], color=c, ls=ls, lw=1.4, label=f'{run.upper()} day {f}')
    ax[2].plot(np.arange(731), d['reff'], color=c, lw=1.5, label=f'{run.upper()} effective radius')
    ax[2].plot(np.arange(731), d['Dgm'] / 2, color=c, ls=':', lw=1.3, label=f'{run.upper()} mass-median radius')
for a in ax[:2]:
    a.set_xscale('log'); a.set_xlim(3, 3000); a.set_xlabel('dry diameter [nm]'); a.axhline(0, color='0.5', lw=0.5)
ax[0].set_ylabel('added dN/dlog$_{10}$D [m$^{-2}$], global column'); ax[0].set_yscale('symlog', linthresh=1e10)
ax[1].set_ylabel('added dM/dlog$_{10}$D [mg m$^{-2}$]')
ax[0].legend(frameon=False, fontsize=8); ax[2].axvline(365, color='0.6', lw=0.7)
ax[2].set_xlabel('day'); ax[2].set_ylabel('dry radius [nm], global band column'); ax[2].legend(frameon=False, fontsize=7)
fig.tight_layout(); fig.savefig(f'{IMG}/fig_size.png'); plt.close(fig)

with open(f'{OUT}/metrics.json', 'w') as f:
    json.dump(M, f, indent=1)
print(json.dumps(M, indent=1))
