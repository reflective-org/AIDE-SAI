#!/usr/bin/env python3
"""Figures for docs/RADIATION_ARTIFACT_EXPLORATION.md (the RAD_SOLAR study).

CPU only, ~30 s. Reads finished run outputs and writes SVGs to docs/figures/ (map
panels are rasterised inside the SVG). SVG rather than PNG because *.png is
gitignored repo-wide; these are documentation figures, not run output.

  solar_geometry.svg      why fixed 6-hourly calls alias the sun (geometry only)
  solar_heating_maps.svg  daily heating-anomaly error against hourly radiation,
                          one day, every mode
  solar_tradeoff.svg      local error against global-forcing error; the wave-4 error
  solar_profiles.svg      zonal-mean and per-level errors, mean of the five days
  solar_online30d.svg     the coupled 30-day instant/window A/B
  solar_longrun.svg       wave-4/8 in dT_rad through the 209-day instant run the
                          test states come from
  solar_followup.svg      the follow-ups: Gauss-node solves (window_gq), the
                          dithered sun time and the fitted blend against the 2x2;
                          error of the k-day mean; global heating error

Inputs (defaults are the runs the doc quotes):
  --variants runs/solar_window_variants   check_solar_window.{npz,log}
  --online   runs/solar30d_instant runs/solar30d_window
  --longrun  runs/lr1yr_d177_sample                      coupled_frames_<tag>.npz
  --followup runs/solar_followup    check_gq_dither.log, check_accum.npz
  --blend    runs/solar_blend       test.log

  python3 scripts/radiation/plot_solar_artifact.py [--day 2] [--png DIR]
--png DIR also writes PNG copies there (for a quick look; never commit them).
"""
import os
import re
import argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt                       # noqa: E402
import matplotlib.colors as mcolors                   # noqa: E402
import cartopy.crs as ccrs                            # noqa: E402
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter   # noqa: E402

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
ap = argparse.ArgumentParser()
ap.add_argument('--variants', default=os.path.join(REPO, 'runs', 'solar_window_variants'))
ap.add_argument('--online', nargs=2, default=[os.path.join(REPO, 'runs', 'solar30d_instant'),
                                              os.path.join(REPO, 'runs', 'solar30d_window')])
ap.add_argument('--longrun', default=os.path.join(REPO, 'runs', 'lr1yr_d177_sample'))
ap.add_argument('--followup', default=os.path.join(REPO, 'runs', 'solar_followup'))
ap.add_argument('--blend', default=os.path.join(REPO, 'runs', 'solar_blend'))
ap.add_argument('--day', type=int, default=2, help='test-day index for the map figure (2 = 1996-07-29)')
ap.add_argument('--out', default=os.path.join(REPO, 'docs', 'figures'))
ap.add_argument('--png', default=None, help='also write PNG previews to this directory')
args = ap.parse_args()
os.makedirs(args.out, exist_ok=True)

# ---- style: recessive chrome, one identity per scheme kept across every figure ----
INK, INK2, MUTED = '#0b0b0b', '#52514e', '#898781'
GRID, AXIS, SURF = '#e1e0d9', '#c3c2b7', '#fcfcfb'
BLUE, ORANGE, GRAY = '#2a78d6', '#eb6834', '#7c7b76'
plt.rcParams.update({
    'font.family': 'sans-serif', 'font.sans-serif': ['Inter', 'Helvetica Neue', 'Arial', 'DejaVu Sans'],
    'font.size': 9, 'axes.titlesize': 9.5, 'axes.labelsize': 9, 'xtick.labelsize': 8, 'ytick.labelsize': 8,
    'legend.fontsize': 8, 'axes.edgecolor': AXIS, 'axes.labelcolor': INK2, 'xtick.color': AXIS,
    'ytick.color': AXIS, 'xtick.labelcolor': INK2, 'ytick.labelcolor': INK2, 'text.color': INK,
    'axes.titlecolor': INK, 'axes.titlelocation': 'left', 'axes.spines.top': False,
    'axes.spines.right': False, 'axes.grid': True, 'grid.color': GRID, 'grid.linewidth': 0.6,
    'axes.axisbelow': True, 'figure.facecolor': SURF, 'axes.facecolor': SURF, 'savefig.facecolor': SURF,
    'legend.frameon': False, 'lines.linewidth': 1.6, 'svg.fonttype': 'none', 'svg.hashsalt': 'rad-solar',
})
# colour = sun-angle choice, line/marker = solves per call (secondary encoding, so
# identity is never colour alone); the reference is primary ink
STY = {
    'hourly':      dict(c=INK,    ls='-',  m='*', label='hourly radiation (reference)'),
    'instant':     dict(c=GRAY,   ls='-',  m='o', label='instant (today)'),
    'window':      dict(c=BLUE,   ls='-',  m='o', label='window: sunlit-mean μ'),
    'window_ins':  dict(c=ORANGE, ls='-',  m='o', label='window_ins: energy-weighted μ'),
    'window2':     dict(c=BLUE,   ls='--', m='s', label='window2: sunlit-mean, 2 solves'),
    'window2_ins': dict(c=ORANGE, ls='--', m='s', label='window2_ins: energy-weighted, 2 solves'),
}
MODES = ['instant', 'window', 'window_ins', 'window2', 'window2_ins']
# diverging blue <-> red through a neutral grey (warm = positive heating / error)
DIV = mcolors.LinearSegmentedColormap.from_list(
    'div', ['#0d366b', '#2a78d6', '#86b6ef', '#cde2fb', '#f0efec', '#f9cfcd', '#ef8a89', '#e34948', '#9c2322'])
DIV.set_bad(SURF)
PROJ = ccrs.PlateCarree(central_longitude=180)
DATA = ccrs.PlateCarree()
DEG = np.pi / 180


def save(fig, name):
    path = os.path.join(args.out, name)
    fig.savefig(path, format='svg', dpi=150, metadata={'Date': None}, bbox_inches='tight')
    if args.png:
        os.makedirs(args.png, exist_ok=True)
        fig.savefig(os.path.join(args.png, name.replace('.svg', '.png')), dpi=130, bbox_inches='tight')
    plt.close(fig)
    print('wrote', path)


def nice(x):
    """round x up to 1, 2, 2.5 or 5 x 10^n (colour-scale limits)"""
    e = 10 ** np.floor(np.log10(x))
    return float(next(s * e for s in (1, 1.5, 2, 2.5, 3, 4, 5, 6, 8, 10) if s * e >= x))


def edges(c, lo=None, hi=None):
    e = np.concatenate([[c[0] - (c[1] - c[0]) / 2], (c[:-1] + c[1:]) / 2, [c[-1] + (c[-1] - c[-2]) / 2]])
    return np.clip(e, lo, hi) if lo is not None else e


def map_ax(fig, spec, title=None):
    ax = fig.add_subplot(spec, projection=PROJ)
    ax.set_global()
    ax.coastlines('110m', lw=0.4, color=INK2, alpha=0.7)
    ax.set_xticks([-180, -90, 0, 90, 180], crs=DATA)
    ax.set_yticks([-60, -30, 0, 30, 60], crs=DATA)
    ax.xaxis.set_major_formatter(LongitudeFormatter())
    ax.yaxis.set_major_formatter(LatitudeFormatter())
    ax.tick_params(length=2, labelsize=7)
    ax.grid(False)
    for s in ax.spines.values():
        s.set_edgecolor(AXIS); s.set_linewidth(0.6)
    if title:
        ax.set_title(title, fontsize=8.5, loc='left')
    return ax


def mesh(ax, lat, lon, z, vmax, cmap=DIV):
    return ax.pcolormesh(edges(lon), edges(lat, -90, 90), z, transform=DATA, cmap=cmap,
                         vmin=-vmax, vmax=vmax, rasterized=True, shading='flat')


def cbar(fig, im, axes, label, **kw):
    cb = fig.colorbar(im, ax=axes, orientation='horizontal', fraction=0.05, pad=0.08, aspect=45, **kw)
    cb.outline.set_visible(False)
    cb.ax.tick_params(labelsize=7.5, length=2, color=AXIS)
    cb.set_label(label, color=INK2, fontsize=8.5)
    return cb


# ---- shared inputs ----
Z = np.load(os.path.join(args.variants, 'check_solar_window.npz'))
lat, lon = Z['lat'], Z['lon']
ndays = len({k.split('_')[0] for k in Z.files if re.match(r'd\d+_', k)})
_e = edges(lat, -90, 90) * DEG
W = np.sin(_e[1:]) - np.sin(_e[:-1])                  # exact cell-area weights by latitude
W2 = W[:, None] * np.ones((lat.size, lon.size))
gm = lambda x: float((x * W2).sum() / W2.sum())       # noqa: E731

# per-day metric rows, exactly as check_solar_window.py printed them
row = re.compile(r'^(\w+)\s+(-?[\d.]+)%\s+(-?[\d.]+)%\s+(-?[\d.]+)%\s+(-?[\d.]+)%\s+(-?[\d.]+)%\s+'
                 r'([-+][\d.]+)\s+([-+][\d.]+)%')
M, dates, day = {m: [] for m in MODES}, [], -1
for line in open(os.path.join(args.variants, 'check_solar_window.log')):
    if line.startswith('=== day'):
        day += 1; dates.append(line.split()[3])
    elif line.startswith('=== mean'):
        break
    elif (r := row.match(line)) and r.group(1) in M:
        M[r.group(1)].append([float(x) for x in r.groups()[1:]])
M = {m: np.array(v) for m, v in M.items()}            # (ndays, 7): col rms, col max, lev rms, zm, w4, ARF, ARF err
assert all(v.shape == (ndays, 7) for v in M.values()), {m: v.shape for m, v in M.items()}


# =========================== 1. geometry ===========================
def cosz(la, lo, t, doy0):
    """cos zenith with solar_zenith()'s conventions; t in hours from 00 UTC of day-of-year doy0."""
    doy = doy0 + t / 24.0
    dec = -23.44 * DEG * np.cos(2 * np.pi * (doy + 10.0) / 365.0)
    hang = 2 * np.pi * (t / 24.0 - 0.5) + np.deg2rad(lo)
    return np.sin(np.deg2rad(la)) * np.sin(dec) + np.cos(np.deg2rad(la)) * np.cos(dec) * np.cos(hang)


DOY = 80                                              # 1996-03-21 (noleap day of year)
CALLS = (0, 6, 12, 18)
fig = plt.figure(figsize=(9.2, 6.6))
gs = fig.add_gridspec(2, 2, height_ratios=[1, 1.25], hspace=0.32, wspace=0.18, top=0.88)
t = np.linspace(-3, 21, 24 * 60 + 1)
for j, (lo0, sub) in enumerate([(0.0, 'calls at local 00, 06, 12, 18 h'), (45.0, 'calls at local 03, 09, 15, 21 h')]):
    ax = fig.add_subplot(gs[0, j])
    mu = np.maximum(cosz(0.0, lo0, t, DOY), 0)
    xs, yi, yw = [], [], []
    for c in CALLS:
        w = (t >= c - 3) & (t <= c + 3)
        xs += [c - 3, c + 3]
        yi += [max(cosz(0.0, lo0, c, DOY), 0)] * 2
        yw += [mu[w].mean()] * 2
    true_day = mu.mean()
    err_i = 100 * (np.mean(yi[::2]) / true_day - 1)
    err_w = 100 * (np.mean(yw[::2]) / true_day - 1)
    ax.fill_between(t, mu, color=INK, alpha=0.06, lw=0)
    ax.plot(t, mu, color=INK, lw=1.6, label='real sun, cos(zenith)')
    ax.plot(xs, yi, color=GRAY, lw=2.0, label='instant: cos(zenith) at the call, held 6 h')
    ax.plot(xs, yw, color=BLUE, lw=2.0, ls='-', label='window modes: the window\'s mean insolation')
    ax.plot(CALLS, [max(cosz(0.0, lo0, c, DOY), 0) for c in CALLS], 'o', color=GRAY, ms=6,
            mec=SURF, mew=1.5, zorder=5)
    for c in CALLS:
        ax.axvline(c - 3, color=GRID, lw=0.8, zorder=0)
    ax.set_xlim(-3, 21); ax.set_ylim(0, 1.08)
    ax.set_xticks(range(-3, 22, 3))
    ax.set_xticklabels([f'{h % 24:02d}' for h in range(-3, 22, 3)])
    ax.set_xlabel('UTC hour (dots: the four radiation calls)')
    if j == 0:
        ax.set_ylabel('cos(zenith) at TOA')
    ax.set_title(f'({"ab"[j]}) Equator, {int(lo0)}°E: {sub}', fontsize=9)
    ax.text(0.02, 0.95, f'daily insolation\ninstant {err_i:+.0f} %\nwindow {err_w:+.1f} %', transform=ax.transAxes,
            ha='left', va='top', fontsize=8, color=INK2, linespacing=1.4)
    if j == 0:
        fig.legend(*ax.get_legend_handles_labels(), loc='lower center', bbox_to_anchor=(0.5, 0.905), ncol=3,
                   handlelength=1.6, fontsize=8)
# (c) the stationary daily-insolation error of instantaneous sampling
truth = np.zeros((lat.size, lon.size)); tt = np.arange(-3, 21, 1 / 60) + 1 / 120
for ti in tt:
    truth += np.maximum(cosz(lat[:, None], lon[None, :], ti, DOY), 0)
truth /= tt.size
inst = np.mean([np.maximum(cosz(lat[:, None], lon[None, :], c, DOY), 0) for c in CALLS], 0)
err = np.ma.masked_where(truth < 1e-3, 100 * (inst / np.maximum(truth, 1e-12) - 1))
ax = map_ax(fig, gs[1, :], '(c) Daily TOA insolation under instant calls, error against the real sun (1996-03-21)')
im = mesh(ax, lat, lon, err, 25)
cbar(fig, im, ax, 'error, %   (window modes: ≤ 0.064 % everywhere)', extend='both')
save(fig, 'solar_geometry.svg')

# =========================== 2. heating-error maps, one day ===========================
d = args.day
ref = Z[f'd{d}_hourly_hr'].astype(np.float64).sum(0)
vmax = nice(np.percentile(np.abs(ref), 98))
fig = plt.figure(figsize=(11.0, 4.9))
gs = fig.add_gridspec(2, 3, hspace=0.16, wspace=0.1)
axes = []
for i, k in enumerate(['hourly'] + MODES):
    if k == 'hourly':
        z, ttl = ref, 'hourly radiation: the anomaly (reference)'
    else:
        z = Z[f'd{d}_{k}_hr'].astype(np.float64).sum(0) - ref
        r = M[k][d]
        ttl = f'{k} − hourly  (rms {r[0]:.0f} %, wave-4 {r[4]:.1f} %)'
    ax = map_ax(fig, gs[i // 3, i % 3], ttl)
    im = mesh(ax, lat, lon, z, vmax)
    axes.append(ax)
cbar(fig, im, axes, f'daily-mean heating anomaly summed over the 24 radiation levels, K/day   '
     f'({dates[d]}, mature plume; errors are mode − hourly)', extend='both')
save(fig, 'solar_heating_maps.svg')

# =========================== 3. trade-off ===========================
fig, (ax, bx) = plt.subplots(1, 2, figsize=(10.0, 4.2), gridspec_kw=dict(width_ratios=[1.35, 1], wspace=0.55))
ax.axhline(0, color=AXIS, lw=0.9, zorder=1)
ax.plot(0, 0, marker='*', ms=13, color=INK, mec=SURF, mew=1.2, ls='none', zorder=6)
ax.annotate('hourly radiation\n(the target)', (0, 0), xytext=(8, -26), textcoords='offset points',
            fontsize=8, color=INK2)
LBL_OFF = {'instant': (-10, 10, 'right'), 'window': (10, 2, 'left'), 'window_ins': (10, -4, 'left'),
           'window2': (10, 2, 'left'), 'window2_ins': (10, -4, 'left')}
for k in MODES:
    s, v = STY[k], M[k]
    x, y = v[:, 2].mean(), v[:, 6].mean()
    ax.errorbar(x, y, yerr=[[y - v[:, 6].min()], [v[:, 6].max() - y]], fmt='none', ecolor=s['c'],
                elinewidth=1.4, capsize=0, zorder=3)
    ax.plot(x, y, marker=s['m'], ms=8.5, color=s['c'], mec=SURF, mew=1.8, ls='none', zorder=5)
    dx, dy, ha = LBL_OFF[k]
    ax.annotate(f'{k}\n{y:+.1f} % · {x:.0f} %', (x, y), xytext=(dx, dy), textcoords='offset points',
                ha=ha, va='center', fontsize=8, color=INK)
ax.set_xlim(-1, 36); ax.set_ylim(-19, 27)
ax.set_xlabel('local error: per-level rms of the daily heating anomaly, % of signal')
ax.set_ylabel('global forcing (ARF) error, %')
ax.set_title('(a) No scheme gets both right\n'
             '     labels: ARF error · local error; bars: range over the five days', fontsize=9)
ax.text(35, -18, '○ one solve per call   □ two solves', ha='right', va='bottom', fontsize=7.5, color=INK2)
# (b) the artifact itself
w4 = [M[k][:, 4].mean() for k in MODES]
yy = np.arange(len(MODES))[::-1]
bx.barh(yy, w4, height=0.62, color=[STY[k]['c'] for k in MODES], edgecolor=SURF, linewidth=2)
for y_, v_ in zip(yy, w4):
    bx.text(v_ + 0.8, y_, f'{v_:.1f} %', va='center', fontsize=8, color=INK)
bx.set_yticks(yy); bx.set_yticklabels(MODES, fontsize=8)
bx.tick_params(axis='y', length=0)
bx.set_xlim(0, 52); bx.grid(axis='y', visible=False)
bx.set_xlabel('wave-4 of the error, % of tropical signal')
bx.set_title('(b) The dusk/dawn artifact itself', fontsize=9)
save(fig, 'solar_tradeoff.svg')

# =========================== 4. where the error sits ===========================
plev = None
fin = os.path.join(args.online[0], f'coupled_final_{os.path.basename(args.online[0])}.npz')
if os.path.exists(fin):
    F0 = np.load(fin)
    if list(F0['klevs']) == list(Z['klevs']):
        plev = F0['plev_pa'] / 100.0
fig, (ax, bx) = plt.subplots(1, 2, figsize=(10.0, 4.3), gridspec_kw=dict(width_ratios=[1.5, 1], wspace=0.28))
zm = {k: [] for k in ['hourly'] + MODES}
lv = {k: [] for k in ['hourly'] + MODES}
for di in range(ndays):
    r3 = Z[f'd{di}_hourly_hr'].astype(np.float64)
    zm['hourly'].append(r3.sum(0).mean(-1))
    lv['hourly'].append([np.sqrt(gm(r3[l] ** 2)) for l in range(r3.shape[0])])
    for k in MODES:
        e3 = Z[f'd{di}_{k}_hr'].astype(np.float64) - r3
        zm[k].append(e3.sum(0).mean(-1))
        lv[k].append([np.sqrt(gm(e3[l] ** 2)) for l in range(e3.shape[0])])
yl = plev if plev is not None else np.arange(len(Z['klevs']))
for k in ['hourly'] + MODES:
    s = STY[k]
    lab = s['label'] if k == 'hourly' else s['label'] + ' (error)'
    ax.plot(lat, np.mean(zm[k], 0), color=s['c'], ls=s['ls'], lw=2.0 if k == 'hourly' else 1.6, label=lab)
    bx.plot(np.mean(lv[k], 0), yl, color=s['c'], ls=s['ls'], lw=2.0 if k == 'hourly' else 1.6)
ax.axhline(0, color=AXIS, lw=0.9, zorder=1)
ax.set_xlim(-90, 90); ax.set_xticks(range(-90, 91, 30))
ax.xaxis.set_major_formatter(LatitudeFormatter())
ax.set_ylabel('K/day, summed over the 24 levels')
ax.set_title('(a) Zonal mean: the reference, and each scheme\'s error', fontsize=9)
ax.legend(loc='upper left', bbox_to_anchor=(0, -0.13), ncol=3, handlelength=2.2, fontsize=7.8)
if plev is not None:
    bx.set_yscale('log'); bx.invert_yaxis()
    bx.set_yticks([3, 10, 30, 100]); bx.set_yticklabels(['3', '10', '30', '100'])
    bx.set_ylabel('pressure, hPa')
else:
    bx.set_ylabel('radiation level index')
bx.set_xlabel('global rms, K/day')
bx.set_title('(b) Per level: the reference\'s rms, and each\nscheme\'s rms error', fontsize=9)
bx.set_xlim(left=0)
fig.text(0.125, 0.985, f'Mean of the five test days ({", ".join(dates)})', fontsize=8, color=INK2, va='bottom')
save(fig, 'solar_profiles.svg')

# =========================== 5. coupled 30-day A/B ===========================
tags = [os.path.basename(p) for p in args.online]
TS = [np.load(os.path.join(p, f'coupled_timeseries_{tg}.npz')) for p, tg in zip(args.online, tags)]
FN = [np.load(os.path.join(p, f'coupled_final_{tg}.npz')) for p, tg in zip(args.online, tags)]
fig = plt.figure(figsize=(10.0, 7.4))
gs = fig.add_gridspec(2, 2, height_ratios=[1, 1.15], hspace=0.38, wspace=0.22, width_ratios=[1, 1])
ax = fig.add_subplot(gs[0, 0])
for ts, k in zip(TS, ['instant', 'window']):
    dd = ts['hours'] / 24.0
    ax.plot(dd, ts['arf_toa_avg'], color=STY[k]['c'], lw=1.8, label=f'{k}  (day 30: {ts["arf_toa_avg"][-1]:+.4f})')
ax.axhline(0, color=AXIS, lw=0.9, zorder=1)
ax.set_xlim(0, 30); ax.set_xlabel('day')
ax.set_ylabel('W m⁻²')
ax.set_title('(a) Reported forcing arf_toa_avg (24-h mean)', fontsize=9)
ax.legend(loc='lower left', fontsize=8)
ax.spines['right'].set_visible(False)
# (c) tropical longitude profile of the stationary difference
plv = FN[0]['plev_pa'] / 100.0
kk = int(np.argmin(np.abs(plv - 52.0)))
lat30, lon30 = FN[0]['lat'], FN[0]['lon']
diff = (FN[1]['dT_rad'][kk] - FN[0]['dT_rad'][kk]) * 1e3          # mK
eddy = diff - diff.mean(-1, keepdims=True)
band = (lat30 >= -35) & (lat30 <= -25)                            # the sunlit (austral-summer) plume
wt = np.cos(lat30[band] * DEG)
prof = (eddy[band] * wt[:, None]).sum(0) / wt.sum()
spec = np.abs(np.fft.rfft(prof)) * 2 / prof.size
bx = fig.add_subplot(gs[0, 1])
bx.axhline(0, color=AXIS, lw=0.9, zorder=1)
bx.plot(lon30, prof, color=INK, lw=1.6)
bx.set_xlim(0, 360); bx.set_xticks(range(0, 361, 90))
bx.xaxis.set_major_formatter(LongitudeFormatter(dateline_direction_label=True))
bx.set_ylabel('mK')
bx.set_title(f'(b) window − instant dT_rad at {plv[kk]:.0f} hPa along the plume (25–35°S),\n'
             f'     zonal mean removed: wave-4 {spec[4]:.1f} mK, wave-8 {spec[8]:.1f} mK', fontsize=9)
cx = map_ax(fig, gs[1, :], f'(c) window − instant dT_rad at {plv[kk]:.0f} hPa after 30 days, zonal mean removed')
vm = nice(np.percentile(np.abs(eddy), 99))
im = mesh(cx, lat30, lon30, eddy, vm)
cbar(fig, im, cx, f'mK   (for scale: the plume warming dT_rad peaks at '
     f'{FN[0]["dT_rad"][kk].max():.2f} K at this level)', extend='both')
save(fig, 'solar_online30d.svg')

# =========================== 6. the long instant run ===========================
tg = os.path.basename(args.longrun)
FR = np.load(os.path.join(args.longrun, f'coupled_frames_{tg}.npz'))
latf, fd = FR['lat'], FR['frames_dT']                         # (nframe, nlat, nlon) K at the probe level
lonf = np.arange(fd.shape[-1]) * (360.0 / fd.shape[-1])
days_f = FR['frame_hours'] / 24.0
A = np.abs(np.fft.rfft(fd, axis=-1)) * 2 / fd.shape[-1]       # (nframe, nlat, nwave)
tropf = np.abs(latf) <= 30
zmax = fd.mean(-1).max(-1)
fig = plt.figure(figsize=(10.0, 7.2))
gs = fig.add_gridspec(2, 1, height_ratios=[1, 1.2], hspace=0.32)
ax = fig.add_subplot(gs[0])
VIO, AQUA = '#6250d6', '#1baf7a'
ax.plot(days_f, zmax, color=INK, lw=2.0, label='zonal-mean dT_rad, peak over latitude')
ax.plot(days_f, A[:, :, 4].max(-1), color=VIO, lw=1.6, label='wave-4 amplitude, peak over latitude')
ax.plot(days_f, A[:, tropf, 4].mean(-1), color=VIO, lw=1.6, ls='--', label='wave-4 amplitude, |lat| ≤ 30 mean')
ax.plot(days_f, A[:, :, 8].max(-1), color=AQUA, lw=1.6, label='wave-8 amplitude, peak over latitude')
ax.plot(days_f, A[:, :, 2].max(-1), color=GRAY, lw=1.2, label='wave-2 amplitude, peak over latitude (for comparison)')
ax.set_xlim(0, days_f[-1]); ax.set_ylim(bottom=0)
ax.set_xlabel('day of the run'); ax.set_ylabel('K')
ax.set_title(f'(a) dT_rad at {float(FR["probe_hpa"]):.0f} hPa through a {int(days_f[-1])}-day RAD_SOLAR=instant run '
             f'(30N/S zonal SO₂, 10 Tg/yr):\n     waves 4 and 8, the 6-h sampling\'s fingerprint, outgrow the zonal mean', fontsize=9)
ax.legend(loc='upper left', fontsize=8)
i = len(days_f) - 1
eddy_f = fd[i] - fd[i].mean(-1, keepdims=True)
cx = map_ax(fig, gs[1], f'(b) dT_rad at {float(FR["probe_hpa"]):.0f} hPa on day {int(days_f[i])}, zonal mean removed '
            f'(three of the five test days use this run\'s state from a few hours earlier)')
im = mesh(cx, latf, lonf, eddy_f, nice(np.percentile(np.abs(eddy_f), 99.5)))
cbar(fig, im, cx, f'K   (zonal-mean peak {zmax[i]:.2f} K)', extend='both')
save(fig, 'solar_longrun.svg')


# =========================== 7. the follow-ups ===========================
GREEN, PURPLE = '#1a9e77', '#8a5cc2'
STY.update({
    'window_blend@0.53': dict(c=BLUE, ls=':', m='D', label='window_blend@0.53: fitted blend, 1 solve'),
    'dither':            dict(c=PURPLE, ls='-', m='o', label='dither: instant sun at a shifted time, 1 solve'),
    'window_gq@2':       dict(c=GREEN, ls='--', m='s', label='window_gq@2: 2 Gauss-node SW solves'),
    'window_gq@3':       dict(c=GREEN, ls='-.', m='^', label='window_gq@3: 3 Gauss-node SW solves'),
})
row2 = re.compile(r'^([\w@.]+)\s+(-?[\d.]+)%\s+(-?[\d.]+)%\s+(-?[\d.]+)%\s+(-?[\d.]+)%\s+(-?[\d.]+)%\s+'
                  r'([-+][\d.]+)\s+([-+][\d.]+)%')


def per_day(log, labels):
    out = {k: [] for k in labels}
    for line in open(log):
        if line.startswith('=== mean'):
            break
        if (r := row2.match(line)) and r.group(1) in out:
            out[r.group(1)].append([float(x) for x in r.groups()[1:]])
    return {k: np.array(v) for k, v in out.items() if v}


FU = per_day(os.path.join(args.followup, 'check_gq_dither.log'), ['dither', 'window_gq@2', 'window_gq@3'])
FU.update(per_day(os.path.join(args.blend, 'test.log'), ['window_blend@0.53']))
ALL = dict(M); ALL.update(FU)
assert all(v.shape == (ndays, 7) for v in ALL.values()), {m: v.shape for m, v in ALL.items()}
A = np.load(os.path.join(args.followup, 'check_accum.npz'))
alab, an, arows = A['rows_label'], A['rows_n'], A['rows']

fig = plt.figure(figsize=(12.0, 4.4))
gs = fig.add_gridspec(1, 3, width_ratios=[1.35, 1.05, 0.9], wspace=0.42)
# (a) local error vs forcing error, five test days
ax = fig.add_subplot(gs[0])
ax.axhline(0, color=AXIS, lw=0.9, zorder=1)
ax.plot(0, 0, marker='*', ms=13, color=INK, mec=SURF, mew=1.2, ls='none', zorder=6)
SHOW = ['instant', 'window', 'window2', 'window_blend@0.53', 'dither', 'window_gq@2', 'window_gq@3']
OFF = {'instant': (0, 17, 'center'), 'window': (9, 0, 'left'), 'window2': (9, 0, 'left'),
       'window_blend@0.53': (8, -17, 'left'), 'dither': (0, -17, 'center'), 'window_gq@2': (9, 2, 'left'),
       'window_gq@3': (0, -17, 'center')}
for k in SHOW:
    st, v = STY[k], ALL[k]
    x, y = v[:, 2].mean(), v[:, 6].mean()
    ax.errorbar(x, y, yerr=[[y - v[:, 6].min()], [v[:, 6].max() - y]], fmt='none', ecolor=st['c'],
                elinewidth=1.4, capsize=0, zorder=3, alpha=0.8)
    ax.plot(x, y, marker=st['m'], ms=8, color=st['c'], mec=SURF, mew=1.6, ls='none', zorder=5)
    dx, dy, ha = OFF[k]
    ax.annotate(f'{k}\n{y:+.1f} % · {x:.0f} %', (x, y), xytext=(dx, dy), textcoords='offset points',
                ha=ha, va='center', fontsize=7.5, color=INK)
ax.set_xlim(-2, 48); ax.set_ylim(-6, 27)
ax.set_xlabel('local error: per-level rms of the heating anomaly, %')
ax.set_ylabel('global forcing (ARF) error, %')
ax.set_title('(a) One day at a time: Gauss nodes get both close\n'
             '     five test days; labels ARF error · local error', fontsize=9)
# (b) error of the k-day mean vs k
bx = fig.add_subplot(gs[1])
for k in ['instant', 'window', 'dither', 'window_gq@2']:
    sel = alab == k
    st = STY[k]
    bx.plot(an[sel], 100 * arows[sel, 0], color=st['c'], ls=st['ls'], marker=st['m'], ms=4.5,
            mec=SURF, mew=0.8, label=k)
    bx.annotate(f'{100 * arows[sel, 0][-1]:.0f} %', (an[sel][-1], 100 * arows[sel, 0][-1]), xytext=(5, 0),
                textcoords='offset points', va='center', fontsize=7.5, color=INK)
bx.set_xlim(0.5, an.max() + 1.6); bx.set_ylim(0, 80)
bx.set_xticks(range(1, int(an.max()) + 1))
bx.set_xlabel('days averaged (k)')
bx.set_ylabel('column rms error of the k-day mean, %')
bx.set_title('(b) Does the error average out?\n     10 consecutive days from 1996-07-25', fontsize=9)
bx.legend(loc='upper center', bbox_to_anchor=(0.5, -0.17), fontsize=7.5, ncol=4, handlelength=1.6,
          columnspacing=0.8)
# (c) global-mean heating error of the 10-day mean
cx = fig.add_subplot(gs[2])
labs = ['instant', 'window', 'dither', 'window_gq@2']
nmax = an.max()
he = [arows[(alab == k) & (an == nmax), 4][0] for k in labs]
yy = np.arange(len(labs))[::-1]
CLIP = 12.0
cx.axvline(0, color=AXIS, lw=0.9)
cx.barh(yy, np.clip(he, -CLIP, CLIP), height=0.6, color=[STY[k]['c'] for k in labs], edgecolor=SURF, linewidth=2)
for y_, v_ in zip(yy, he):
    xt = min(max(v_, -CLIP), CLIP)
    cx.text(max(xt, 0) + 0.6, y_, f'{v_:+.1f} %' + (' →' if v_ > CLIP else ''),
            va='center', ha='left', fontsize=8, color=INK)
cx.set_yticks(yy); cx.set_yticklabels(labs, fontsize=8)
cx.tick_params(axis='y', length=0)
cx.set_xlim(-6, CLIP + 6); cx.grid(axis='y', visible=False)
cx.set_xlabel('global-mean column heating error, %')
cx.set_title(f'(c) Global heating, {int(nmax)}-day mean\n     (window clipped)', fontsize=9)
save(fig, 'solar_followup.svg')
