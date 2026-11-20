#!/usr/bin/env python3
"""Gate for RAD_SOLAR=window*: radiation.solar_window() against brute force.

CPU only, no archive, ~1 min. Checks, on the model's 192 x 288 grid:
  1. the closed form equals a fine midpoint-rule integral of the same sun
     (window-mean max(cos zenith, 0) and sunlit fraction) for widths from
     1 h to 30 h (> 24 h exercises the periodic continuation), all dates;
  1b. the energy-weighted mu (window_ins) against brute force, its insolation
     scale, and the two half-windows (window2) tiling the window exactly;
  1c. the blend (window_blend) reduces to sunlit at alpha=0 and to energy at
     alpha=1, keeps the insolation exact and is linear in alpha;
  2. width -> 0 recovers solar_zenith() on the sunlit side;
  3. polar night gives f = 0 and zenith = pi (night for RRTMGP and for the
     emulator's cos(zenith) feature), polar day f = 1;
  4. four 6-h windows centred on the coupling hours tile the day: the daily
     TOA insolation they deliver matches a 1-minute integral of the real sun
     (declination varying through the day), where the instantaneous geometry
     the model used before the flag is off by -24..+23 % (the artifact);
  5. window_gq's insolation is exact and it converges on nonlinear responses;
  6. window_gq's trailing window (RAD_SOLAR_ALIGN=trailing) equals the centred
     window of the call 3 h earlier.
Run: python3 scripts/validation/test_solar_window.py
"""
import os, sys
os.environ.setdefault('JAX_PLATFORMS', 'cpu')
import numpy as np
import cftime
_REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(_REPO, 'src'))
import _paths                      # noqa: E402,F401
import radiation as R              # noqa: E402

lat = np.linspace(-90.0, 90.0, 192)
lon = np.arange(288) * 1.25
DATES = [(1, 1), (3, 21), (5, 5), (6, 21), (9, 23), (12, 21)]


SUB = (slice(None, None, 4), slice(None, None, 8))      # 48 x 36 columns for the brute force


def brute(tv, width_h, n=4000):
    """Midpoint rule over the window with the declination of the centre,
    on the SUB-sampled columns."""
    doy = tv.dayofyr + tv.hour / 24.0
    dec = -23.44 * np.pi / 180.0 * np.cos(2 * np.pi * (doy + 10.0) / 365.0)
    phi = np.deg2rad(lat[SUB[0]])[:, None, None]
    s = (np.arange(n) + 0.5) / n - 0.5                  # fraction of the window
    hc = 2 * np.pi * (tv.hour / 24.0 - 0.5) + np.deg2rad(lon[SUB[1]])[None, :, None]
    h = hc + 2 * np.pi * width_h / 24.0 * s[None, None, :]
    mu = np.sin(phi) * np.sin(dec) + np.cos(phi) * np.cos(dec) * np.cos(h)
    return np.maximum(mu, 0).mean(-1), (mu > 0).mean(-1)


# 1. closed form vs brute force
worst_mu = worst_f = 0.0
for (m, d) in DATES:
    for hr in (0, 7, 13):
        tv = cftime.DatetimeNoLeap(1997, m, d, hr)
        for w in (1.0, 6.0, 12.0, 30.0):
            zen, f = R.solar_window(tv, lat, lon, w)
            mu_bar = (f * np.where(f > 0, np.cos(zen), 0.0))[SUB]
            f = f[SUB]
            mb, fb = brute(tv, w)
            worst_mu = max(worst_mu, np.abs(mu_bar - mb).max())
            worst_f = max(worst_f, np.abs(f - fb).max())
print(f'1. closed form vs brute force: max |mu_bar err| {worst_mu:.1e}, max |f err| {worst_f:.1e}')
assert worst_mu < 1e-5 and worst_f < 1e-3, 'closed form disagrees with the brute-force integral'

# 1b. the energy-weighted variant (window_ins): mu_ins = int(mu^2)/int(mu), and
# scale * cos(zenith) must still be mu_bar; half-windows (window2) must tile the window
def brute_ins(tv, width_h, n=4000, offset_h=0.0):
    hr = tv.hour + offset_h
    doy = tv.dayofyr + hr / 24.0
    dec = -23.44 * np.pi / 180.0 * np.cos(2 * np.pi * (doy + 10.0) / 365.0)
    phi = np.deg2rad(lat[SUB[0]])[:, None, None]
    s = (np.arange(n) + 0.5) / n - 0.5
    hc = 2 * np.pi * (hr / 24.0 - 0.5) + np.deg2rad(lon[SUB[1]])[None, :, None]
    mu = np.maximum(np.sin(phi) * np.sin(dec) + np.cos(phi) * np.cos(dec) *
                    np.cos(hc + 2 * np.pi * width_h / 24.0 * s[None, None, :]), 0)
    m1 = mu.mean(-1)
    return (mu ** 2).mean(-1) / np.where(m1 > 0, m1, 1.0), m1


worst_ins = worst_bar = worst_tile = 0.0
for (m, d) in DATES:
    for hr in (0, 7, 13):
        tv = cftime.DatetimeNoLeap(1997, m, d, hr)
        for w in (1.0, 6.0, 12.0, 30.0):
            zen, sc = R.solar_window(tv, lat, lon, w, mu='energy')
            zs, fs = R.solar_window(tv, lat, lon, w)
            mins, mb = brute_ins(tv, w)
            lit = mb[...] > 1e-6
            worst_ins = max(worst_ins, np.abs(np.cos(zen[SUB])[lit] - mins[lit]).max())
            bar_e = sc * np.where(sc > 0, np.cos(zen), 0.0); bar_s = fs * np.where(fs > 0, np.cos(zs), 0.0)
            worst_bar = max(worst_bar, np.abs(bar_e - bar_s).max())
            # two halves (offsets -w/4, +w/4, width w/2) carry the window's insolation
            halves = sum(R.solar_window(tv, lat, lon, w / 2, offset_h=o)[1] *
                         np.where(R.solar_window(tv, lat, lon, w / 2, offset_h=o)[1] > 0,
                                  np.cos(R.solar_window(tv, lat, lon, w / 2, offset_h=o)[0]), 0.0)
                         for o in (-w / 4, w / 4)) / 2
            worst_tile = max(worst_tile, np.abs(halves - bar_s).max())
print(f'1b. energy-weighted mu vs brute force: max |cos err| {worst_ins:.1e}; scale*cos(zen) = mu_bar to '
      f'{worst_bar:.1e}; two halves tile the window to {worst_tile:.1e} (declination per half)')
# (the halves tile only up to the declination, which each half takes at its own
# centre: < 1e-3 in mu_bar even for the 30-h test windows)
assert worst_ins < 1e-4 and worst_bar < 1e-12 and worst_tile < 2e-3, 'energy-weighted / half-window geometry wrong'

# 1c. the blend (window_blend): alpha = 0 is the sunlit-mean geometry, alpha = 1
# the energy-weighted one, any alpha keeps scale * cos(zenith) = mu_bar and puts
# cos(zenith) between the two (mu_ins >= mu_bar / f), linear in alpha
worst_a0 = worst_a1 = worst_bbar = worst_lin = 0.0
for (m, d) in DATES:
    for hr in (0, 7, 13):
        tv = cftime.DatetimeNoLeap(1997, m, d, hr)
        for w in (1.0, 6.0, 12.0, 30.0):
            zs, fs = R.solar_window(tv, lat, lon, w)
            ze, se = R.solar_window(tv, lat, lon, w, mu='energy')
            bar_s = fs * np.where(fs > 0, np.cos(zs), 0.0)
            z0, s0 = R.solar_window(tv, lat, lon, w, mu='blend', alpha=0.0)
            z1, s1 = R.solar_window(tv, lat, lon, w, mu='blend', alpha=1.0)
            worst_a0 = max(worst_a0, np.abs(np.cos(z0) - np.cos(zs)).max(), np.abs(s0 - fs).max())
            worst_a1 = max(worst_a1, np.abs(z1 - ze).max(), np.abs(s1 - se).max())
            lit = fs > 0
            for a in (0.25, 0.6, 0.9):
                za, sa = R.solar_window(tv, lat, lon, w, mu='blend', alpha=a)
                worst_bbar = max(worst_bbar, np.abs(sa * np.where(sa > 0, np.cos(za), 0.0) - bar_s).max())
                want = (1 - a) * np.cos(zs) + a * np.cos(ze)
                worst_lin = max(worst_lin, np.abs(np.cos(za) - want)[lit].max())
                assert np.all(za[~lit] == np.pi) and np.all(sa[~lit] == 0.0), 'blend: night not dark'
print(f'1c. blend: alpha=0 vs sunlit {worst_a0:.1e}, alpha=1 vs energy {worst_a1:.1e}, '
      f'scale*cos(zen) = mu_bar to {worst_bbar:.1e}, linear in alpha to {worst_lin:.1e}')
assert worst_a0 < 1e-12 and worst_a1 == 0.0 and worst_bbar < 1e-12 and worst_lin < 1e-12, 'blend geometry wrong'

# 2. zero-width limit
worst = 0.0
for (m, d) in DATES:
    for hr in range(0, 24, 5):
        tv = cftime.DatetimeNoLeap(1997, m, d, hr)
        z0 = R.solar_zenith(tv, lat, lon)
        zw, fw = R.solar_window(tv, lat, lon, 1e-6)
        sun = np.cos(z0) > 1e-3
        worst = max(worst, np.abs(np.cos(zw[sun]) - np.cos(z0[sun])).max())
        assert np.all(np.abs(fw[sun] - 1.0) < 1e-6), 'sunlit column not fully lit in a vanishing window'
print(f'2. width -> 0 matches solar_zenith: max |cos err| {worst:.1e}')
assert worst < 1e-6

# 3. polar night / polar day at the NH solstice
tv = cftime.DatetimeNoLeap(1997, 6, 21, 12)
zen, f = R.solar_window(tv, lat, lon, 6.0)
assert np.all(f[lat < -70] == 0.0) and np.all(zen[lat < -70] == np.pi), 'polar night not dark'
assert np.all(np.abs(f[lat > 70] - 1.0) < 1e-9), 'polar day not fully lit'
assert np.all((f >= 0) & (f <= 1)) and np.all(np.isfinite(zen))
print('3. polar night f=0, zen=pi; polar day f=1; f in [0,1], zenith finite')

# 4. daily tiling vs the real sun at 1-minute resolution
def cos_real(m, d, minute):
    """cos zenith of the real sun (declination moving through the day)."""
    doy = cftime.DatetimeNoLeap(1997, m, d, 0).dayofyr + minute / 1440.0
    dec = -23.44 * np.pi / 180.0 * np.cos(2 * np.pi * (doy + 10.0) / 365.0)
    h = 2 * np.pi * (minute / 1440.0 - 0.5) + np.deg2rad(lon)[None, :]
    p = np.deg2rad(lat)[:, None]
    return np.sin(p) * np.sin(dec) + np.cos(p) * np.cos(dec) * np.cos(h)


print('4. daily-mean TOA insolation vs 1-min truth (lat 0-60, where it is > 0.05):')
for (m, d) in DATES:
    truth = np.mean([np.maximum(cos_real(m, d, mi), 0) for mi in range(-180, 1260)], 0)
    inst = np.mean([np.maximum(np.cos(R.solar_zenith(cftime.DatetimeNoLeap(1997, m, d, h), lat, lon)), 0)
                    for h in (0, 6, 12, 18)], 0)
    win = np.zeros_like(truth)
    for h in (0, 6, 12, 18):
        z, f = R.solar_window(cftime.DatetimeNoLeap(1997, m, d, h), lat, lon, 6.0)
        win += f * np.where(f > 0, np.cos(z), 0.0) / 4
    ok = (np.abs(lat) <= 60)[:, None] & (truth > 0.05)
    ei = 100 * (inst[ok] / truth[ok] - 1); ew = 100 * (win[ok] / truth[ok] - 1)
    print(f'   {m:02d}-{d:02d}  instant {ei.min():+6.1f} .. {ei.max():+6.1f} %   '
          f'window {ew.min():+7.3f} .. {ew.max():+7.3f} %')
    assert np.abs(ew).max() < 0.2, 'window geometry does not tile the day'

# 5. window_gq: exact insolation, and convergence of a nonlinear SW response
#    (mu * r(mu), r a transmission-like and a reflection-like curve) to the
#    brute-force window mean as the node count grows; dither offsets
def brute_resp(tv, width_h, r, n=4000):
    doy = tv.dayofyr + tv.hour / 24.0
    dec = -23.44 * np.pi / 180.0 * np.cos(2 * np.pi * (doy + 10.0) / 365.0)
    phi = np.deg2rad(lat[SUB[0]])[:, None, None]
    s_ = (np.arange(n) + 0.5) / n - 0.5
    hc = 2 * np.pi * (tv.hour / 24.0 - 0.5) + np.deg2rad(lon[SUB[1]])[None, :, None]
    mu = np.sin(phi) * np.sin(dec) + np.cos(phi) * np.cos(dec) * np.cos(hc + 2 * np.pi * width_h / 24.0 * s_)
    mu = np.maximum(mu, 0)
    return np.where(mu > 0, mu * r(np.maximum(mu, 1e-9)), 0).mean(-1)


RESP = {'transmission exp(-0.15/mu)': lambda m: np.exp(-0.15 / m),
        'reflection 1/(1+4mu)': lambda m: 1.0 / (1.0 + 4.0 * m)}
worst_ins = 0.0
errs = {(k, n): 0.0 for k in RESP for n in (0, 1, 2, 3, 4)}      # n=0: window (sunlit mean)
for (m, d) in DATES:
    for hr in (0, 7, 13):
        tv = cftime.DatetimeNoLeap(1997, m, d, hr)
        zs, fs = R.solar_window(tv, lat, lon, 6.0)
        mu_bar = np.where(fs > 0, fs * np.cos(zs), 0.0)
        for n in (1, 2, 3, 4):
            parts = R.solar_gauss(tv, lat, lon, 6.0, n)
            assert len(parts) == n
            ins = sum(sc * np.where(sc > 0, np.cos(z), 0.0) for z, sc in parts) / n
            worst_ins = max(worst_ins, np.abs(ins - mu_bar).max())
            for k, r in RESP.items():
                tru = brute_resp(tv, 6.0, r)
                q = (sum(sc * np.where(sc > 0, np.cos(z) * r(np.maximum(np.cos(z), 1e-9)), 0.0)
                         for z, sc in parts) / n)[SUB]
                ok = tru > 0.02
                errs[(k, n)] = max(errs[(k, n)], np.abs(q[ok] / tru[ok] - 1).max())
        for k, r in RESP.items():
            tru = brute_resp(tv, 6.0, r)
            me = np.cos(zs)
            q = np.where(fs > 0, fs * me * r(np.maximum(me, 1e-9)), 0.0)[SUB]
            ok = tru > 0.02
            errs[(k, 0)] = max(errs[(k, 0)], np.abs(q[ok] / tru[ok] - 1).max())
assert worst_ins < 1e-12, f'window_gq insolation not exact: {worst_ins:.2e}'
print(f'5. window_gq insolation exact (max |err| {worst_ins:.1e}); worst column error of a '
      f'nonlinear response vs brute force:')
for k in RESP:
    print(f'   {k:28s} window {100*errs[(k,0)]:6.2f} %   ' +
          '   '.join(f'gq{n} {100*errs[(k,n)]:6.3f} %' for n in (1, 2, 3, 4)))
    assert errs[(k, 3)] < errs[(k, 0)] and errs[(k, 4)] <= errs[(k, 2)] + 1e-6, 'window_gq does not converge'
offs = np.array([R.dither_offset(cftime.DatetimeNoLeap(1997, 1, 1, 0) + __import__('datetime').timedelta(hours=6 * i), 6.0)
                 for i in range(4 * 365)])
assert np.all((offs >= -3.0) & (offs < 3.0)), 'dither offset out of the window'
assert R.dither_offset(cftime.DatetimeNoLeap(1997, 7, 29, 12), 6.0) == \
    R.dither_offset(cftime.DatetimeNoLeap(1997, 7, 29, 12), 6.0)
same_hour = offs[2::4]                                            # 12 UTC every day
for o, nm in ((offs, 'all calls'), (same_hour, '12 UTC calls')):
    hist = np.histogram(o, bins=12, range=(-3, 3))[0] / o.size * 12
    assert np.abs(hist - 1).max() < 0.1, f'dither offsets not uniform ({nm}): {hist}'
print(f'   dither offsets in [-3, 3) h, deterministic, uniform over a year (all calls and per UTC hour)')
# 6. RAD_SOLAR_ALIGN=trailing: window_gq's window ending at the call (offset -W/2) is the
#    centred window of the call 3 h earlier, across midnight and the year boundary too
worst = 0.0
for (y, m, d, hr) in ((1997, 1, 1, 0), (1997, 1, 1, 6), (1997, 6, 21, 18), (1997, 12, 31, 0)):
    tv = cftime.DatetimeNoLeap(y, m, d, hr)
    t3 = tv - __import__('datetime').timedelta(hours=3)
    for n in (2, 3):
        for (za, sa), (zb, sb) in zip(R.solar_gauss(tv, lat, lon, 6.0, n, offset_h=-3.0),
                                      R.solar_gauss(t3, lat, lon, 6.0, n)):
            worst = max(worst, np.abs(za - zb).max(), np.abs(sa - sb).max())
assert worst < 1e-9, f'trailing window != centred window 3 h earlier: {worst:.2e}'
print(f'6. window_gq trailing window = centred window 3 h earlier (max |diff| {worst:.1e})')
print('ALL PASS')
