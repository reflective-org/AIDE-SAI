"""Aerosol-increment radiation emulator: one SW and one LW per-column MLP.

  SW net: (T, log q, log o3 on the full column; albedo; mu; SW band optics of the
           aerosol on the band levels) -> d hr_sw (nzb, K/day) and d sw_up_toa (W/m2)
           at unit irradiance (TSI). window_gq: evaluated at each Gauss node's mu on
           that node's sunlit columns, combined with solar_gauss()'s scales exactly as
           the physics combines its node solves (SW is linear in the irradiance).
  LW net: (T, log q, log o3, log ch4; ts; LW band optics) -> d hr_lw, d olr. Once.

features() is the ONE definition, used with xp=numpy by train.py and xp=jax.numpy here.
"""
import os
import numpy as np

TAU0 = 1e-7
TAU0_COL = 1e-6
TAU_W = 1e-5         # tau weight scale of the ssa/g features (per layer)
MU_FEATS = os.environ.get('RADEM_MUFEAT', '0') == '1'   # v3+: log(mu), 1/(mu+0.02) SW inputs


def features(xp, kind, T, q, o3, ch4, ts, alb, mu, tau, ssa, g):
    """(ncol, NF) float32. T q o3 ch4 (ncol, nzc); ts alb mu (ncol,); tau ssa g (ncol, nb, nzb)."""
    n = T.shape[0]
    f = [T, xp.log(xp.maximum(q, 1e-12)), xp.log(xp.maximum(o3, 1e-15))]
    if kind == 'sw':
        f += [alb[:, None], mu[:, None]]
        if MU_FEATS:
            # the SW increment PEAKS at grazing sun (mu 0.02-0.1) and changes sign there
            # (slant-path shading of lower layers): give the net log(mu) and 1/(mu+0.02)
            m_ = xp.maximum(mu, 1e-3)[:, None]
            f += [xp.log(m_), 1.0 / (m_ + 0.02)]
    else:
        f += [xp.log(xp.maximum(ch4, 1e-15)), ts[:, None]]
    # ssa and g tau-weighted: a layer with almost no optical depth has ssa/g set by
    # numerics of a few particles, which the solve ignores but a network would not
    # (v1: tiny-tau polar layers 6-7 sigma out -> -1.4 K/day SW). w -> 0 below TAU_W.
    w = tau / (tau + TAU_W)
    f += [xp.log1p(tau / TAU0).reshape(n, -1), (ssa * w).reshape(n, -1), (g * w).reshape(n, -1),
          xp.log1p(tau.sum(-1) / TAU0_COL)]
    return xp.concatenate([x.astype(xp.float32) for x in f], axis=1)


_PROF = os.environ.get('RADEM_PROF') == '1'
FAST = os.environ.get('RADEM_FAST', '1') == '1'   # perf (lem2): per-hour cache path; 0 = the original calls
import time as _time


class _Prof:
    """perf: RADEM_PROF=1 -> synced per-part timings of one call, printed as one line."""
    def __init__(self, name):
        self.name, self.t, self.d = name, _time.perf_counter(), []
    def __call__(self, part, *arrs):
        if not _PROF:
            return
        import jax
        jax.block_until_ready([a for a in arrs if a is not None])
        t = _time.perf_counter(); self.d.append((part, t - self.t)); self.t = t
    def done(self):
        if _PROF:
            print(f"  [radem-prof] {self.name} " + " ".join(f"{k}={v*1e3:.0f}ms" for k, v in self.d), flush=True)


AMP_EPS = 1e-6     # K: columns with max|dT| below this get P = 0 exactly (amplitude-normalised P net)


def pl_amp(xp, dT):
    """(ncol,) column amplitude max|dT| used by the amplitude-normalised P net."""
    return xp.maximum(xp.abs(dT).max(1), AMP_EPS)


def features_pl(xp, T, q, o3, ch4, ts, alb, dT, gmu, gsc, amp_norm=False, feat2=False):
    """Clear-sky response net P (RAD_MODE=damped): base column T q o3 ch4 (ncol, nzc), ts alb (ncol,),
    dT on the band levels (ncol, nzb), window_gq node mu and scale (ncol, 3) each.
    amp_norm: dT enters as its shape dT/A plus log A (A = max|dT|); the net then predicts P/A, so the
    amplitude scaling is exact and small-dT columns are as accurate as large ones."""
    if amp_norm:
        A = pl_amp(xp, dT)
        # feat2 (ckpt flag 'feat2', gen3+): also raw dT and dT^2/A so the response's dependence on the actual
        # magnitude (P ~ l dT + m dT^2 -> P/A ~ l s + m A s^2) is a simple function of the inputs
        ex = [dT, dT * dT / A[:, None]] if feat2 else []
        dT = xp.concatenate([dT / A[:, None], xp.log(A)[:, None]] + ex, axis=1)
    f = [T, xp.log(xp.maximum(q, 1e-12)), xp.log(xp.maximum(o3, 1e-15)), xp.log(xp.maximum(ch4, 1e-15)),
         ts[:, None], alb[:, None], dT, gmu, gsc]
    return xp.concatenate([x.astype(xp.float32) for x in f], axis=1)


def mlp_apply(jnp, P, x, nl):
    """Standardise -> [Linear SiLU]*(nl-1) -> Linear -> de-standardise."""
    h = (x - P['x_mu']) / P['x_sd']
    for i in range(nl):
        h = h @ P[f'W{i}'] + P[f'b{i}']
        if i < nl - 1:
            h = h * (1.0 / (1.0 + jnp.exp(-h)))
    return h * P['y_sd'] + P['y_mu']


class Emulator:
    """Drop-in for the physics increment: increment(hour, num, T_band) -> dict like
    radem_lib.increment (sw, lw K/day; swup, olr W/m2; aod), at the driver's solar mode
    (window_gq or instant). drv is a RadiationDriver; only its readers, optics and
    geometry are used -- no RRTMGP solve."""

    def __init__(self, ckpt_dir, drv):
        import jax, jax.numpy as jnp
        import radiation as R
        import rad_features as F
        self.R, self.F, self.jnp, self.drv = R, F, jnp, drv
        self.P = {}; self.nl = {}
        for k in ('sw', 'lw'):
            z = np.load(os.path.join(ckpt_dir, f'{k}.npz'))
            self.nl[k] = int(z['n_layers'])
            self.P[k] = {n: jnp.asarray(z[n]) for n in z.files if n != 'n_layers'}
        if os.path.exists(os.path.join(ckpt_dir, 'pl.npz')):          # damping net (RAD_MODE=damped)
            z = np.load(os.path.join(ckpt_dir, 'pl.npz'))
            self.nl['pl'] = int(z['n_layers'])
            self.P['pl'] = {n: jnp.asarray(z[n]) for n in z.files if n != 'n_layers'}
        self._f = {k: jax.jit(lambda P, x, nl=self.nl[k]: mlp_apply(jnp, P, x, nl)) for k in self.P}
        self.nzb = drv.nzb

    def _increment_ref(self, hour, num, T_band):
        jnp, R, F, drv = self.jnp, self.R, self.F, self.drv
        nlat, nlon, nzb = drv.nlat, drv.nlon, self.nzb
        pf = _Prof('increment')
        h = F.prep_hour(drv, hour) if drv.solar_mode == 'instant' else None
        T = drv._read3d(drv.dT, 'T', hour).astype(np.float64); T[drv.band_pos] = np.asarray(T_band)
        q = drv._read3d(drv.dQ, 'Q', hour); o3 = drv._read3d(drv.dO3, 'O3', hour)
        ch4 = drv._read3d(drv.dCH4, 'CH4', hour)
        ts, alb = drv.surface(hour)
        pf('reads')
        if drv.solar_mode == 'window_gq':
            parts = drv.solar_parts(hour)                       # [(zen, scale, alb)], scales x N
        else:
            parts = [(R.solar_zenith(drv.time[hour], drv.lat, drv.lon), np.ones((nlat, nlon)), alb)]
        pf('solar')
        wwt = F.wt_weights_for(drv, T, q)
        pf('wt', wwt)
        num_d = jnp.asarray(num); pf('h2d_num', num_d)
        tau_s, ssa_s, g_s, tau_l, ssa_l, g_l = F.optics_chunk(
            drv, num_d, None if wwt is None else jnp.asarray(wwt))
        pf('optics', tau_s, tau_l)
        col = lambda a: jnp.asarray(a).reshape(a.shape[0], -1).T          # (nz, nlat, nlon) -> (ncol, nz)
        bcol = lambda a: jnp.moveaxis(jnp.asarray(a).reshape(a.shape[0], a.shape[1], -1), -1, 0)
        Tc, qc, oc, cc = col(T), col(q), col(o3), col(ch4)
        tsc = jnp.asarray(ts).reshape(-1)
        # LW: once, all columns
        xl = features(jnp, 'lw', Tc, qc, oc, cc, tsc, None, None, bcol(tau_l), bcol(ssa_l), bcol(g_l))
        pf('lw_feat', xl)
        yl = self._f['lw'](self.P['lw'], xl)
        lw = yl[:, :nzb].T.reshape(nzb, nlat, nlon); olr = yl[:, nzb].reshape(nlat, nlon)
        pf('lw_net', yl)
        # SW: per node on ALL columns (fixed shapes: no recompiles), scaled; the scale
        # is 0 where the node is dark, so night columns get exactly 0
        sw = jnp.zeros((nzb, nlat * nlon), jnp.float32); up = jnp.zeros(nlat * nlon, jnp.float32)
        ts_, ss_, gs_ = bcol(tau_s), bcol(ssa_s), bcol(g_s)
        N = len(parts)
        for zen, sc, alb_n in parts:
            mu = np.cos(zen).reshape(-1); scf = np.asarray(sc).reshape(-1)
            w = np.where((scf > 0) & (mu > 0), scf / N, 0.0).astype(np.float32)
            xs = features(jnp, 'sw', Tc, qc, oc, None, None, jnp.asarray(np.asarray(alb_n).reshape(-1)),
                          jnp.asarray(np.maximum(mu, 0.0)), ts_, ss_, gs_)
            ys = self._f['sw'](self.P['sw'], xs) * jnp.asarray(w)[:, None]
            sw = sw + ys[:, :nzb].T; up = up + ys[:, nzb]
        pf('sw_x3', sw, up)
        if drv.wet_optics:
            aod = R._aod550_wet(jnp.asarray(num), drv._dp_band_dev[0], drv._sig550_dev[0], jnp.asarray(wwt))
        else:
            aod = R._aod550(jnp.asarray(num), drv._dp_band_dev[0], drv._sig550_dev[0])
        pf('aod', aod)
        r = dict(sw=np.asarray(sw.reshape(nzb, nlat, nlon)), lw=np.asarray(lw),
                 swup=np.asarray(up.reshape(nlat, nlon)), olr=np.asarray(olr), aod=np.asarray(aod))
        pf('d2h'); pf.done()
        return r

    def _planck_ref(self, hour, T_band, dT_band):
        """Clear-sky response P = HR(no aer, T+dT) - HR(no aer, T) under the driver's window_gq call:
        dict sw, lw (nzb, nlat, nlon) K/day; swup, olr (nlat, nlon) W/m2."""
        jnp, drv = self.jnp, self.drv
        nlat, nlon, nzb = drv.nlat, drv.nlon, self.nzb
        pf = _Prof('planck')
        T = drv._read3d(drv.dT, 'T', hour).astype(np.float64); T[drv.band_pos] = np.asarray(T_band)
        q = drv._read3d(drv.dQ, 'Q', hour); o3 = drv._read3d(drv.dO3, 'O3', hour)
        ch4 = drv._read3d(drv.dCH4, 'CH4', hour)
        ts, alb = drv.surface(hour)
        pf('reads')
        parts = drv.solar_parts(hour)
        pf('solar')
        gmu = np.stack([np.where(sc > 0, np.cos(z), 0.0) for z, sc, _ in parts]).reshape(3, -1).T
        gsc = np.stack([sc for _, sc, _ in parts]).reshape(3, -1).T
        col = lambda a: jnp.asarray(a).reshape(a.shape[0], -1).T
        if 'amp_norm' in self.P['pl']:
            dTc = col(np.asarray(dT_band)); A = pl_amp(jnp, dTc)
            x = features_pl(jnp, col(T), col(q), col(o3), col(ch4), jnp.asarray(ts).reshape(-1),
                            jnp.asarray(alb).reshape(-1), dTc, jnp.asarray(gmu), jnp.asarray(gsc), amp_norm=True,
                            feat2='feat2' in self.P['pl'])
            pf('feat', x)
            y = self._f['pl'](self.P['pl'], x) * jnp.where(A > AMP_EPS, A, 0.0)[:, None]
            pf('pl_net', y)
            r = lambda a: np.asarray(a.T.reshape(-1, nlat, nlon))
            out = dict(sw=r(y[:, :nzb]), lw=r(y[:, nzb:2 * nzb]), swup=np.asarray(y[:, 2 * nzb].reshape(nlat, nlon)),
                       olr=np.asarray(y[:, 2 * nzb + 1].reshape(nlat, nlon)))
            pf('d2h'); pf.done()
            return out
        x = features_pl(jnp, col(T), col(q), col(o3), col(ch4), jnp.asarray(ts).reshape(-1),
                        jnp.asarray(alb).reshape(-1), col(np.asarray(dT_band)), jnp.asarray(gmu), jnp.asarray(gsc))
        x0 = features_pl(jnp, col(T), col(q), col(o3), col(ch4), jnp.asarray(ts).reshape(-1),
                         jnp.asarray(alb).reshape(-1), col(np.zeros_like(np.asarray(dT_band))), jnp.asarray(gmu), jnp.asarray(gsc))
        # anchored: P(dT) - P(0) is exactly 0 at dT = 0 (the raw net's offset there accumulated in dT_rad:
        # 2e-3 K/day vs a physics 3e-4 on the 44-81 % of cells with |dT| < 0.01 K)
        y = self._f['pl'](self.P['pl'], x) - self._f['pl'](self.P['pl'], x0)
        r = lambda a: np.asarray(a.T.reshape(-1, nlat, nlon))
        return dict(sw=r(y[:, :nzb]), lw=r(y[:, nzb:2 * nzb]), swup=np.asarray(y[:, 2 * nzb].reshape(nlat, nlon)),
                    olr=np.asarray(y[:, 2 * nzb + 1].reshape(nlat, nlon)))

    # ---------------- perf (lem2, 2026-11-19): per-hour cache, same values ----------------
    # increment() is called twice per step (evolved at T+dT, reference at T) and planck() once, all at the same
    # hour: the background column (T outside the band, q, o3, ch4), the surface, the window_gq geometry and the
    # device copies / log features of the T-independent inputs are the same for all three calls. Read and built
    # once per hour; every value is computed by the same operations as before, so the results are bit-identical.
    def _hc(self, hour):
        c = getattr(self, '_hcache', None)
        if c is not None and c['hour'] == hour:
            return c
        jnp, drv = self.jnp, self.drv
        T0 = drv._read3d(drv.dT, 'T', hour).astype(np.float64)
        q = drv._read3d(drv.dQ, 'Q', hour); o3 = drv._read3d(drv.dO3, 'O3', hour)
        ch4 = drv._read3d(drv.dCH4, 'CH4', hour)
        ts, alb = drv.surface(hour)
        parts = drv.solar_parts(hour) if drv.solar_mode == 'window_gq' else None
        col = lambda a: jnp.asarray(a).reshape(a.shape[0], -1).T
        qc, oc, cc = col(q), col(o3), col(ch4)
        c = dict(hour=hour, T0=T0, q=q, o3=o3, ch4=ch4, ts=ts, alb=alb, parts=parts,
                 qc=qc, oc=oc, cc=cc, tsc=jnp.asarray(ts).reshape(-1), albc=jnp.asarray(alb).reshape(-1),
                 lq=jnp.log(jnp.maximum(qc, 1e-12)), lo3=jnp.log(jnp.maximum(oc, 1e-15)),
                 lch4=jnp.log(jnp.maximum(cc, 1e-15)))
        if parts is not None:
            c['nodes'] = []
            for zen, sc, alb_n in parts:
                mu = np.cos(zen).reshape(-1); scf = np.asarray(sc).reshape(-1)
                c['nodes'].append((mu, scf, jnp.asarray(np.asarray(alb_n).reshape(-1)), jnp.asarray(np.maximum(mu, 0.0))))
        self._hcache = c
        return c

    def prep_wt(self, hour, T_band):
        """The wet-composition weights increment() would build for (hour, T_band); pass back as wwt=."""
        c = self._hc(hour)
        T = c['T0'].copy(); T[self.drv.band_pos] = np.asarray(T_band)
        return self.F.wt_weights_for(self.drv, T, c['q'])

    def _increment_fast(self, hour, num, T_band, wwt=None):
        jnp, R, F, drv = self.jnp, self.R, self.F, self.drv
        nlat, nlon, nzb = drv.nlat, drv.nlon, self.nzb
        pf = _Prof('increment')
        c = self._hc(hour)
        T = c['T0'].copy(); T[drv.band_pos] = np.asarray(T_band)
        pf('reads')
        parts = c['parts']
        if wwt is None:
            wwt = F.wt_weights_for(drv, T, c['q'])
        pf('wt', wwt)
        num_d = jnp.asarray(num)
        tau_s, ssa_s, g_s, tau_l, ssa_l, g_l = F.optics_chunk(
            drv, num_d, None if wwt is None else jnp.asarray(wwt))
        pf('optics', tau_s, tau_l)
        bcol = lambda a: jnp.moveaxis(jnp.asarray(a).reshape(a.shape[0], a.shape[1], -1), -1, 0)
        Tc = jnp.asarray(T).reshape(T.shape[0], -1).T
        f32 = lambda x: x.astype(jnp.float32)

        def opt_feats(tau, ssa, g):          # = the tail of features(), same ops
            n = tau.shape[0]
            w = tau / (tau + TAU_W)
            return [f32(jnp.log1p(tau / TAU0).reshape(n, -1)), f32((ssa * w).reshape(n, -1)),
                    f32((g * w).reshape(n, -1)), f32(jnp.log1p(tau.sum(-1) / TAU0_COL))]
        head = [f32(Tc), f32(c['lq']), f32(c['lo3'])]
        xl = jnp.concatenate(head + [f32(c['lch4']), f32(c['tsc'][:, None])]
                             + opt_feats(bcol(tau_l), bcol(ssa_l), bcol(g_l)), axis=1)
        pf('lw_feat', xl)
        yl = self._f['lw'](self.P['lw'], xl)
        lw = yl[:, :nzb].T.reshape(nzb, nlat, nlon); olr = yl[:, nzb].reshape(nlat, nlon)
        pf('lw_net', yl)
        sw = jnp.zeros((nzb, nlat * nlon), jnp.float32); up = jnp.zeros(nlat * nlon, jnp.float32)
        of_s = opt_feats(bcol(tau_s), bcol(ssa_s), bcol(g_s))
        N = len(parts)
        for mu, scf, albj, muj in c['nodes']:
            w = np.where((scf > 0) & (mu > 0), scf / N, 0.0).astype(np.float32)
            sf = [f32(albj[:, None]), f32(muj[:, None])]
            if MU_FEATS:
                m_ = jnp.maximum(muj, 1e-3)[:, None]
                sf += [f32(jnp.log(m_)), f32(1.0 / (m_ + 0.02))]
            xs = jnp.concatenate(head + sf + of_s, axis=1)
            ys = self._f['sw'](self.P['sw'], xs) * jnp.asarray(w)[:, None]
            sw = sw + ys[:, :nzb].T; up = up + ys[:, nzb]
        pf('sw_x3', sw, up)
        if drv.wet_optics:
            aod = R._aod550_wet(num_d, drv._dp_band_dev[0], drv._sig550_dev[0], jnp.asarray(wwt))
        else:
            aod = R._aod550(num_d, drv._dp_band_dev[0], drv._sig550_dev[0])
        pf('aod', aod)
        r = dict(sw=np.asarray(sw.reshape(nzb, nlat, nlon)), lw=np.asarray(lw),
                 swup=np.asarray(up.reshape(nlat, nlon)), olr=np.asarray(olr), aod=np.asarray(aod))
        pf('d2h'); pf.done()
        return r

    def _planck_fast(self, hour, T_band, dT_band):
        jnp, drv = self.jnp, self.drv
        nlat, nlon, nzb = drv.nlat, drv.nlon, self.nzb
        pf = _Prof('planck')
        c = self._hc(hour)
        T = c['T0'].copy(); T[drv.band_pos] = np.asarray(T_band)
        parts = c['parts']
        gmu = np.stack([np.where(sc > 0, np.cos(z), 0.0) for z, sc, _ in parts]).reshape(3, -1).T
        gsc = np.stack([sc for _, sc, _ in parts]).reshape(3, -1).T
        col = lambda a: jnp.asarray(a).reshape(a.shape[0], -1).T
        pf('reads')
        assert 'amp_norm' in self.P['pl'] and 'feat2' not in self.P['pl'], 'fast planck: amp-norm gen2 pl only'
        f32 = lambda x: x.astype(jnp.float32)
        dTc = col(np.asarray(dT_band)); A = pl_amp(jnp, dTc)
        dTf = jnp.concatenate([dTc / A[:, None], jnp.log(A)[:, None]], axis=1)
        x = jnp.concatenate([f32(col(T)), f32(c['lq']), f32(c['lo3']), f32(c['lch4']), f32(c['tsc'][:, None]),
                             f32(c['albc'][:, None]), f32(dTf), f32(jnp.asarray(gmu)), f32(jnp.asarray(gsc))], axis=1)
        pf('feat', x)
        y = self._f['pl'](self.P['pl'], x) * jnp.where(A > AMP_EPS, A, 0.0)[:, None]
        pf('pl_net', y)
        r = lambda a: np.asarray(a.T.reshape(-1, nlat, nlon))
        out = dict(sw=r(y[:, :nzb]), lw=r(y[:, nzb:2 * nzb]), swup=np.asarray(y[:, 2 * nzb].reshape(nlat, nlon)),
                   olr=np.asarray(y[:, 2 * nzb + 1].reshape(nlat, nlon)))
        pf('d2h'); pf.done()
        return out

    def increment(self, hour, num, T_band, wwt=None):
        if FAST and self.drv.solar_mode == 'window_gq':
            return self._increment_fast(hour, num, T_band, wwt=wwt)
        assert wwt is None
        return self._increment_ref(hour, num, T_band)

    def planck(self, hour, T_band, dT_band):
        if FAST and 'amp_norm' in self.P['pl'] and 'feat2' not in self.P['pl']:
            return self._planck_fast(hour, T_band, dT_band)
        return self._planck_ref(hour, T_band, dT_band)
