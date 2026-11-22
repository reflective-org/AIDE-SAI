"""Figures + metrics.json for the 2-year fast-run summary page.

Inputs: coupled_timeseries of both runs, cache/daily.npz (reduce_daily.py), cache/snap_*.npz
(reduce3d.py), the injection run's probe-level frames. Writes img/*.png|gif and metrics.json in
the current directory (where cache/ is).

  python3 make_figs.py [--inj DIR] [--inj-y1 DIR] [--ctrl DIR]
"""
import argparse
import io
import os
from glob import glob

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt                     # noqa: E402
from matplotlib.colors import LogNorm, TwoSlopeNorm  # noqa: E402
import numpy as np                                   # noqa: E402

from lib import (BURDEN_KG, BURDEN_TG, CACHE, EARTH_AREA, IMG, OUT, RUN, S_PER_SO2, S_PER_SO4,  # noqa: E402
                 TAG, add_run_args, configure, coast, geo, load_ts, save_gif, wjson)

_a = add_run_args(argparse.ArgumentParser(description=__doc__.split('\n')[0])).parse_args()
configure(inj=_a.inj, inj_y1=_a.inj_y1, ctrl=_a.ctrl)

plt.rcParams.update({'font.size': 10, 'axes.titlesize': 11, 'figure.dpi': 110, 'savefig.dpi': 110,
                     'axes.spines.top': False, 'axes.spines.right': False})
C_INJ, C_CTL, C_DIF, C_REF = '#2a78d6', '#7a7a7a', '#d1495b', '#1baf7a'
YEAR = 365
W = dict(all=(0, 730), y1=(0, 365), y2=(365, 730), l30=(700, 730))
M = {}                                                # metrics


def rmean(y, w=30):
    y = np.asarray(y, float); w = max(1, min(w, len(y))); k = np.ones(w)
    ok = np.isfinite(y)
    num = np.convolve(np.where(ok, y, 0), k, 'same'); den = np.convolve(ok.astype(float), k, 'same')
    return num / np.maximum(den, 1)


def daily(x):
    x = np.asarray(x, float); n = len(x) // 4 * 4
    return x[:n].reshape(-1, 4).mean(1)


def wmean(day_vals, key, day0=0.5):
    """mean over a window of a daily series indexed by day (0-based day index)."""
    a, b = W[key]
    v = np.asarray(day_vals, float)[a:b]
    return float(np.nanmean(v)) if v.size else np.nan


def yearline(ax):
    ax.axvline(YEAR, color='0.6', lw=0.8, ls=':')


def savefig(fig, name):
    fig.savefig(f'{IMG}/{name}', bbox_inches='tight'); plt.close(fig)


# ============================================================ global time series
ti, tc = load_ts('inj'), load_ts('ctrl')
n = min(len(ti['hours']), len(tc['hours'])) // 4 * 4
ndays = n // 4
M['ndays_paired'] = ndays
M['ndays_inj'] = len(ti['hours']) // 4
day = np.arange(ndays) + 0.5
F_net = daily(ti['arf_toa'][:n] - tc['arf_toa'][:n])          # 4 samples/day = window_gq diurnal mean
F_inj = daily(ti['arf_toa'][:n]); F_ctl = daily(tc['arf_toa'][:n])
dAOD = daily(ti['aod550'][:n] - tc['aod550'][:n])
AOD_i, AOD_c = daily(ti['aod550'][:n]), daily(tc['aod550'][:n])
Mi, Mc = daily(ti['Mburden'][:n]) * BURDEN_TG, daily(tc['Mburden'][:n]) * BURDEN_TG     # Tg SO4, band
Ni, Nc = daily(ti['Nburden'][:n]) * BURDEN_KG / EARTH_AREA, daily(tc['Nburden'][:n]) * BURDEN_KG / EARTH_AREA
SO2i, SO2c = daily(ti['SO2burden'][:n]) * BURDEN_TG, daily(tc['SO2burden'][:n]) * BURDEN_TG
H2i = daily(ti['H2SO4burden'][:n]) * BURDEN_TG
for key in W:
    M[f'F_net_{key}'] = wmean(F_net, key); M[f'dAOD_{key}'] = wmean(dAOD, key)
    M[f'dM_Tg_{key}'] = wmean(Mi - Mc, key); M[f'SO2_inj_Tg_{key}'] = wmean(SO2i, key)
    M[f'F_inj_vs_ref_{key}'] = wmean(F_inj, key); M[f'F_ctl_vs_ref_{key}'] = wmean(F_ctl, key)
    M[f'dN_1e10_{key}'] = wmean((Ni - Nc) / 1e10, key)
    M[f'AOD_inj_{key}'] = wmean(AOD_i, key); M[f'AOD_ctl_{key}'] = wmean(AOD_c, key)
M['M_inj_end_Tg'] = float(Mi[-1]); M['M_ctl_end_Tg'] = float(Mc[-1])
M['M_inj_start_Tg'] = float(Mi[0])
M['SO2_inj_end_Tg'] = float(SO2i[-1])
M['inj_SO2_cum_Tg'] = float(ti['injSO2_cum'][n - 1] * BURDEN_TG)

fig, ax = plt.subplots(3, 1, figsize=(10, 9), sharex=True, gridspec_kw=dict(height_ratios=[3, 2, 2]))
a = ax[0]
a.plot(day, F_net, color=C_DIF, lw=0.5, alpha=0.45)
a.plot(day, rmean(F_net), color=C_DIF, lw=2, label='injection − control (SAI forcing)')
a.plot(day, rmean(F_inj), color=C_INJ, lw=1, ls='--', label='injection vs MAM4 reference')
a.plot(day, rmean(F_ctl), color=C_CTL, lw=1, ls='--', label='control vs MAM4 reference')
a.axhline(0, color='0.3', lw=0.6); yearline(a)
a.set_ylabel('net TOA forcing [W m$^{-2}$]'); a.legend(frameon=False, fontsize=9, loc='lower left')
a.set_title('Global-mean net TOA forcing (thin: daily, thick: 30-day running mean)')
a = ax[1]
a.plot(day, dAOD, color=C_DIF, lw=0.5, alpha=0.45); a.plot(day, rmean(dAOD), color=C_DIF, lw=2, label='added AOD (inj − ctrl)')
a.plot(day, AOD_c, color=C_CTL, lw=1, label='control AOD (background)')
yearline(a); a.set_ylabel('AOD 550 nm'); a.legend(frameon=False, fontsize=9, loc='upper left')
a = ax[2]
a.plot(day, Mi - Mc, color=C_DIF, lw=2, label='added SO$_4$ (inj − ctrl)')
a.plot(day, Mc, color=C_CTL, lw=1, label='control SO$_4$')
a.plot(day, SO2i, color='#e07a5f', lw=1.5, label='injection SO$_2$')
yearline(a); a.set_ylabel('burden in the band [Tg]'); a.set_xlabel('day since 1996-01-01')
a.legend(frameon=False, fontsize=9, loc='upper left')
fig.tight_layout(); savefig(fig, 'fig1_forcing_aod_burden.png')

# ============================================================ mass / sulfur budget (injection run)
B = {k: ti[k][:n] for k in ti if k.startswith('B_')}
M0 = float(ti['M0'])
tg = M0 * BURDEN_TG
dayS = (np.arange(n) + 1) / 4
fig, ax = plt.subplots(1, 2, figsize=(12, 4.2))
a = ax[0]
for k, lab, c in (('B_micro', 'microphysics (SO$_2$→SO$_4$ + nucleation)', '#d1495b'),
                  ('B_adv_np', 'advection (out through the open faces)', '#2e4057'),
                  ('B_settle', 'settling', '#edae49'), ('B_adv_pol', 'polar-cap stirring', '#66a182')):
    if k in B:
        a.plot(dayS, B[k] * tg, color=c, lw=1.6, label=lab)
a.plot(dayS, (ti['Mburden'][:n] / M0 - 1) * tg, 'k--', lw=1.2, label='net (M − M$_0$)')
a.axhline(0, color='0.5', lw=0.6); yearline(a)
a.set_xlabel('day'); a.set_ylabel('cumulative ΔM [Tg SO$_4$]'); a.legend(frameon=False, fontsize=8)
a.set_title('Injection run: cumulative SO$_4$ budget by stage')
bsum = sum(B[k] for k in ('B_adv_np', 'B_adv_pol', 'B_floor', 'B_micro', 'B_bc', 'B_settle') if k in B)
resid = bsum - (ti['Mburden'][:n] / M0 - 1)
M['budget_closure_max'] = float(np.max(np.abs(resid)))
for k in B:
    M[f'cum_{k}_Tg'] = float(B[k][-1] * tg)
# sulfur accounting, injection minus control (Tg S)
injS = ti['injSO2_cum'][:n] * BURDEN_TG * S_PER_SO2
dSO4S = (ti['Mburden'][:n] - tc['Mburden'][:n]) * BURDEN_TG * S_PER_SO4
dSO2S = (ti['SO2burden'][:n] - tc['SO2burden'][:n]) * BURDEN_TG * S_PER_SO2
dH2S = (ti['H2SO4burden'][:n] - tc['H2SO4burden'][:n]) * BURDEN_TG * 32.06 / 98.08
a = ax[1]
a.plot(dayS, injS, 'k', lw=1.5, label='injected S')
a.stackplot(dayS, dSO4S, dSO2S, np.maximum(injS - dSO4S - dSO2S - dH2S, 0),
            colors=['#2a78d6', '#e07a5f', '#cccccc'],
            labels=['added SO$_4$ in band', 'added SO$_2$ in band', 'left the band (bottom face, settling) + H$_2$SO$_4$'])
yearline(a); a.set_xlabel('day'); a.set_ylabel('Tg S'); a.legend(frameon=False, fontsize=8, loc='upper left')
a.set_title('Where the injected sulfur is (injection − control)')
fig.tight_layout(); savefig(fig, 'fig2_budget.png')
M['injS_end_Tg'] = float(injS[-1]); M['dSO4S_end_Tg'] = float(dSO4S[-1]); M['dSO2S_end_Tg'] = float(dSO2S[-1])
M['lost_S_end_Tg'] = float(injS[-1] - dSO4S[-1] - dSO2S[-1] - dH2S[-1])

# ============================================================ daily zonal means
g = geo(); lat, lon, plev = g['lat'], g['lon'], g['plev']
have_daily = os.path.exists(f'{CACHE}/daily.npz')
if have_daily:
    D = np.load(f'{CACHE}/daily.npz')
    dd = D['hours'] / 24.0
    di = np.round(dd - 1).astype(int)                # frame at 00 UTC of day k+1 -> index k
    wl = D['wlat']
    colM_d = (D['inj_colM'] - D['ctl_colM']) * 1e6   # mg/m2
    colN_d = (D['inj_colN'] - D['ctl_colN'])
    hr_d = (D['inj_hr_lat'] - D['ctl_hr_lat']) * 1e3  # mK/day
    hrg_d = (D['inj_hr_glob'] - D['ctl_hr_glob']) * 1e3
    colMg = np.nansum(colM_d * wl[None], 1); colNg = np.nansum(colN_d * wl[None], 1) / 1e10
    ser = {}
    for nm, v in (('dcolM', colMg), ('dcolN', colNg), ('dhr', hrg_d)):
        full = np.full(730, np.nan); ok = (di >= 0) & (di < 730); full[di[ok]] = v[ok]; ser[nm] = full
        for key in W:
            M[f'{nm}_{key}'] = wmean(full, key)
    M['ndays_daily'] = int(len(dd))

    fig, ax = plt.subplots(2, 1, figsize=(11, 7.5), sharex=True)
    vmax = np.nanpercentile(colM_d, 99.5)
    pc = ax[0].pcolormesh(dd, lat, colM_d.T, cmap='viridis', vmin=0, vmax=vmax, shading='auto')
    fig.colorbar(pc, ax=ax[0], label='mg m$^{-2}$', pad=0.01)
    ax[0].set_title('Added stratospheric column SO$_4$, zonal mean (injection − control)')
    v = np.nanpercentile(np.abs(hr_d), 99)
    pc = ax[1].pcolormesh(dd, lat, hr_d.T, cmap='RdBu_r', norm=TwoSlopeNorm(0, -v, v), shading='auto')
    fig.colorbar(pc, ax=ax[1], label='mK day$^{-1}$', pad=0.01)
    ax[1].set_title('Added aerosol heating rate, mass-weighted over stratospheric band levels')
    for a in ax:
        a.set_ylabel('latitude'); a.set_yticks([-90, -60, -30, 0, 30, 60, 90]); yearline(a)
        a.axhline(30, color='w', lw=0.4, ls='--'); a.axhline(-30, color='w', lw=0.4, ls='--')
    ax[1].set_xlabel('day since 1996-01-01')
    fig.tight_layout(); savefig(fig, 'fig3_hovmoller_col_heating.png')

    fig, ax = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    ax[0].plot(np.arange(730) + 0.5, ser['dhr'], color=C_DIF, lw=0.5, alpha=0.5)
    ax[0].plot(np.arange(730) + 0.5, rmean(ser['dhr']), color=C_DIF, lw=2)
    ax[0].axhline(0, color='0.4', lw=0.6); yearline(ax[0])
    ax[0].set_ylabel('mK day$^{-1}$'); ax[0].set_title('Global-mean added aerosol heating (stratospheric band, mass-weighted)')
    ax[1].plot(np.arange(730) + 0.5, ser['dcolM'], color=C_INJ, lw=1.8, label='added column SO$_4$ [mg m$^{-2}$]')
    a2 = ax[1].twinx(); a2.plot(np.arange(730) + 0.5, ser['dcolN'], color='#8f2d56', lw=1.2, label='added particles [10$^{10}$ m$^{-2}$]')
    a2.set_ylabel('10$^{10}$ m$^{-2}$', color='#8f2d56'); a2.spines['right'].set_visible(True)
    ax[1].set_ylabel('mg m$^{-2}$', color=C_INJ); yearline(ax[1]); ax[1].set_xlabel('day since 1996-01-01')
    ax[1].set_title('Global-mean added stratospheric column SO$_4$ and particle number')
    fig.tight_layout(); savefig(fig, 'fig4_heating_column_timeseries.png')

    # lat-pressure means
    def latp(pre, key):
        a_, b_ = W[key]; sel = (di >= a_) & (di < b_)
        return D[f'{pre}'][sel].mean(0) if sel.any() else None
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.3), sharey=True)
    for a, (k, lab, sc, cm) in zip(ax, (('so4_zm', 'added SO$_4$ [µg kg$^{-1}$]', 1e9, 'viridis'),
                                       ('num_zm', 'added particles [10$^{6}$ per kg air]', 1e-6, 'magma'),
                                       ('hr_zm', 'added heating [mK day$^{-1}$]', 1e3, 'RdBu_r'))):
        f = (latp(f'inj_{k}', 'all') - latp(f'ctl_{k}', 'all')) * sc
        if k == 'hr_zm':
            vv = np.nanpercentile(np.abs(f), 99.5)
            pc = a.pcolormesh(lat, plev / 100, f, cmap=cm, norm=TwoSlopeNorm(0, -vv, vv), shading='auto')
        else:
            pc = a.pcolormesh(lat, plev / 100, f, cmap=cm, vmin=0, vmax=np.nanpercentile(f, 99.5), shading='auto')
        a.plot(lat, np.nanmean(D['trop_zm'], 0) / 100, 'w--', lw=1)
        fig.colorbar(pc, ax=a, pad=0.01).set_label(lab)
        a.set_yscale('log'); a.set_ylim(plev.max() / 100, plev.min() / 100); a.set_xlabel('latitude')
        a.set_xticks([-90, -60, -30, 0, 30, 60, 90])
    ax[0].set_ylabel('pressure [hPa]')
    fig.suptitle('Zonal means, 2-year mean, injection − control (dashed: mean CESM tropopause)', y=1.02)
    fig.tight_layout(); savefig(fig, 'fig5_latp_2yr.png')

    # monthly lat-p GIF of added SO4
    frames = []
    vmax = None
    mons = []
    for m in range(24):
        sel = (di >= m * 365 / 12) & (di < (m + 1) * 365 / 12)
        if sel.sum() == 0:
            continue
        mons.append((m, (D['inj_so4_zm'][sel].mean(0) - D['ctl_so4_zm'][sel].mean(0)) * 1e9))
    if mons:
        vmax = np.nanpercentile(np.stack([x for _, x in mons]), 99.5)
        mn = 'Jan Feb Mar Apr May Jun Jul Aug Sep Oct Nov Dec'.split()
        for m, f in mons:
            fig, a = plt.subplots(figsize=(7, 3.8))
            pc = a.pcolormesh(lat, plev / 100, f, cmap='viridis', vmin=0, vmax=vmax, shading='auto')
            fig.colorbar(pc, ax=a, pad=0.01, label='added SO$_4$ [µg kg$^{-1}$]')
            a.set_yscale('log'); a.set_ylim(plev.max() / 100, plev.min() / 100)
            a.set_title(f'{mn[m % 12]} {1996 + m // 12}: zonal-mean added SO$_4$'); a.set_xlabel('latitude'); a.set_ylabel('hPa')
            buf = io.BytesIO(); fig.savefig(buf, format='png', dpi=90, bbox_inches='tight'); plt.close(fig)
            from PIL import Image
            frames.append(Image.open(buf).convert('RGB'))
        save_gif(frames, f'{IMG}/fig5b_latp_so4_monthly.gif', ms=350)

# ============================================================ snapshots: maps
snaps = sorted(glob(f'{CACHE}/snap_*.npz'))
if snaps:
    S = [np.load(f) for f in snaps]
    sd = np.array([int(s['hour_abs']) / 24.0 for s in S])
    M['nsnap'] = len(S)

    def mapmean(key, sel):
        return np.mean([S[i][key] for i in np.where(sel)[0]], 0)
    sel_all = sd <= 730; sel_l30 = (sd > 700) & (sd <= 730)
    M['nsnap_l30'] = int(sel_l30.sum())
    for key, sel in (('all', sel_all), ('l30', sel_l30), ('y1', sd <= 365), ('y2', (sd > 365) & (sd <= 730))):
        if sel.sum() == 0:
            continue
        for k in ('aod_glob', 'arf_glob', 'colM_glob', 'colN_glob', 'hr_glob'):
            M[f'snap_d{k}_{key}'] = float(np.mean([S[i][f'inj_{k}'] - S[i][f'ctl_{k}'] for i in np.where(sel)[0]]))
        for k in ('dmass_glob', 'deff_wet_glob', 'gf_med'):
            M[f'snap_inj_{k}_{key}'] = float(np.mean([S[i][f'inj_{k}'] for i in np.where(sel)[0]]))
            M[f'snap_ctl_{k}_{key}'] = float(np.mean([S[i][f'ctl_{k}'] for i in np.where(sel)[0]]))
    M['strat_frac_excluded'] = float(1 - np.mean([s['strat_frac'] for s in S]))
    M['strat_frac_excluded_area'] = float(1 - np.mean([s['strat_frac_area'] for s in S]))
    CO = coast()

    def mappanel(a, f, cmap, norm, title, unit, fig):
        pc = a.pcolormesh(lon, lat, f, cmap=cmap, norm=norm, shading='auto')
        for c in CO:
            a.plot(c[:, 0], c[:, 1], color='0.25', lw=0.35)
        a.set_xlim(0, 360); a.set_ylim(-90, 90); a.set_xticks([]); a.set_yticks([-60, -30, 0, 30, 60])
        a.set_title(title, fontsize=10)
        fig.colorbar(pc, ax=a, pad=0.01, fraction=0.04).set_label(unit, fontsize=9)

    for key, sel, lab in (('2yr', sel_all, '2-year mean'), ('l30', sel_l30, 'last 30 days')):
        if sel.sum() == 0:
            continue
        dA = mapmean('inj_aod', sel) - mapmean('ctl_aod', sel)
        dF = mapmean('inj_arf', sel) - mapmean('ctl_arf', sel)
        dC = (mapmean('inj_colM', sel) - mapmean('ctl_colM', sel)) * 1e6
        dN = (mapmean('inj_colN', sel) - mapmean('ctl_colN', sel)) / 1e10
        dm = mapmean('inj_dmass', sel); dw = mapmean('inj_deff_wet', sel)
        fig, ax = plt.subplots(3, 2, figsize=(13, 9.5))
        vF = np.nanpercentile(np.abs(dF), 99.5)
        mappanel(ax[0, 0], dF, 'RdBu_r', TwoSlopeNorm(0, -vF, vF), f'net TOA SAI forcing, {lab}', 'W m$^{-2}$', fig)
        mappanel(ax[0, 1], dA, 'magma_r', plt.Normalize(0, np.nanpercentile(dA, 99.5)), f'added AOD 550 nm, {lab}', '', fig)
        mappanel(ax[1, 0], dC, 'viridis', plt.Normalize(0, np.nanpercentile(dC, 99.5)), f'added column SO$_4$, {lab}', 'mg m$^{-2}$', fig)
        vN = np.nanpercentile(np.abs(dN), 99.5)
        mappanel(ax[1, 1], dN, 'PuOr_r', TwoSlopeNorm(0, -vN, vN), f'added particles, {lab}', '10$^{10}$ m$^{-2}$', fig)
        mappanel(ax[2, 0], dm, 'cividis', plt.Normalize(np.nanpercentile(dm, 0.5), np.nanpercentile(dm, 99.5)),
                 f'mass-weighted dry diameter (injection), {lab}', 'nm', fig)
        mappanel(ax[2, 1], dw, 'cividis', plt.Normalize(np.nanpercentile(dw, 0.5), np.nanpercentile(dw, 99.5)),
                 f'wet effective diameter (injection), {lab}', 'nm', fig)
        fig.tight_layout(); savefig(fig, f'fig6_maps_{key}.png')

    # column SO4 + AOD GIF over all snapshots
    from PIL import Image
    vC = np.nanpercentile(np.stack([(S[i]['inj_colM'] - S[i]['ctl_colM']) * 1e6 for i in range(len(S))][-12:]), 99.5)
    vA = np.nanpercentile(np.stack([S[i]['inj_aod'] - S[i]['ctl_aod'] for i in range(len(S))][-12:]), 99.5)
    gser = np.array([(S[i]['inj_colM_glob'] - S[i]['ctl_colM_glob']) * 1e6 for i in range(len(S))])
    frames = []
    for i in range(len(S)):
        fig = plt.figure(figsize=(13, 5.2))
        gs = fig.add_gridspec(2, 2, height_ratios=[3, 1.1])
        a0 = fig.add_subplot(gs[0, 0]); a1 = fig.add_subplot(gs[0, 1]); a2 = fig.add_subplot(gs[1, :])
        mappanel(a0, (S[i]['inj_colM'] - S[i]['ctl_colM']) * 1e6, 'viridis', plt.Normalize(0, vC),
                 f'added column SO$_4$, day {sd[i]:.1f}', 'mg m$^{-2}$', fig)
        mappanel(a1, S[i]['inj_aod'] - S[i]['ctl_aod'], 'magma_r', plt.Normalize(0, vA),
                 f'added AOD 550 nm, day {sd[i]:.1f}', '', fig)
        a2.plot(sd, gser, color='0.75', lw=1); a2.plot(sd[:i + 1], gser[:i + 1], color=C_INJ, lw=2)
        a2.set_xlim(0, 730); a2.set_ylabel('global mg m$^{-2}$'); a2.set_xlabel('day since 1996-01-01'); yearline(a2)
        fig.tight_layout()
        buf = io.BytesIO(); fig.savefig(buf, format='png', dpi=72, bbox_inches='tight'); plt.close(fig)
        frames.append(Image.open(buf).convert('RGB'))
    save_gif(frames, f'{IMG}/fig7_column_aod.gif', ms=160)

    # size distributions (global-mean stratospheric column)
    dpb = g['dp_bin'].astype(float)
    edges = np.sqrt(dpb[1:] * dpb[:-1]); edges = np.concatenate([[dpb[0] ** 2 / edges[0]], edges, [dpb[-1] ** 2 / edges[-1]]])
    dln = np.log(edges[1:] / edges[:-1])
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.3))
    for a, k, unit, sc in ((ax[0], 'Nk_glob', 'dN/dlnD [10$^{10}$ m$^{-2}$]', 1e-10), (ax[1], 'Mk_glob', 'dM/dlnD [mg m$^{-2}$]', 1e6)):
        for sel, ls, lab in ((sel_all, '-', '2-year mean'), (sel_l30, '--', 'last 30 days')):
            if sel.sum() == 0:
                continue
            vi = mapmean(f'inj_{k}', sel) / dln * sc; vc = mapmean(f'ctl_{k}', sel) / dln * sc
            a.plot(dpb, vi, color=C_INJ, ls=ls, lw=1.8, label=f'injection, {lab}')
            a.plot(dpb, vc, color=C_CTL, ls=ls, lw=1.4, label=f'control, {lab}')
            a.plot(dpb, vi - vc, color=C_DIF, ls=ls, lw=1.2, label=f'increment, {lab}')
        a.set_xscale('log'); a.set_xlabel('dry diameter [nm]'); a.set_ylabel(unit)
        if k == 'Nk_glob':
            a.set_yscale('symlog', linthresh=1e-1)
    ax[0].legend(frameon=False, fontsize=8)
    fig.suptitle('Global-mean stratospheric column size distribution (dry, from the 40 bins)', y=1.02)
    fig.tight_layout(); savefig(fig, 'fig8_size_distribution.png')

# ============================================================ probe level (both years, daily)
pf = None
for cand in (f"{RUN['inj']}/coupled_frames_{TAG['inj']}.npz",):
    if os.path.exists(cand):
        pf = cand
if pf:
    from lib import npz_member
    hrs = np.asarray(npz_member(pf, 'frame_hours'))
    mas = npz_member(pf, 'frames_mas')
    zm = np.stack([np.asarray(mas[i]).sum(0).mean(1) for i in range(len(hrs))]) * 1e9
    fig, a = plt.subplots(figsize=(11, 3.6))
    pc = a.pcolormesh(hrs / 24, lat, zm.T, cmap='viridis', norm=LogNorm(max(np.nanpercentile(zm, 1), 1e-2), np.nanpercentile(zm, 99.8)),
                      shading='auto')
    fig.colorbar(pc, ax=a, pad=0.01, label='SO$_4$ [µg kg$^{-1}$]')
    a.set_title('Injection run: zonal-mean SO$_4$ at 51.7 hPa (injection level), daily'); a.set_ylabel('latitude')
    a.set_xlabel('day since 1996-01-01'); yearline(a); a.set_yticks([-90, -60, -30, 0, 30, 60, 90])
    fig.tight_layout(); savefig(fig, 'fig9_probe_hovmoller.png')

wjson(f'{OUT}/metrics.json', M)
print('metrics:', len(M), 'figures in', IMG)
