"""MET_SRC=emu: the JCM emulator P2@200 run IN THE LOOP as the meteorology.

What it is (2026-11-23, emulated linking). The atmosphere is the PARADIS-style
JCM emulator P2@200 (T63 Gaussian 96x192, 95 levels to 0.01 hPa, dry,
prognostic u/v/T/omega/ps), free-running with no nudging: the recipe of the
AIDE-SAI-link met set and of its atmos_rollout.py / atm_live.py (HL QBO wave
drag, global-mean ps fix, the u-fixer with latsm 4 / levsm 2, generator noise
seed 1000 + ATM_SEED). The code is vendored in src/atmos/aide_atmos/ and every
input -- weights, normalisation, the IC, the fixer terms, the CESM grid -- comes
from one bundle directory (ATM_BUNDLE, sha256 in its SHA256SUMS).

How it is coupled. Not a file and not a subprocess: the network is stepped in
THIS process (torch, on the same GPU as the JAX model -- the microphysics
emulator is torch already), one 6 h step whenever the loop first asks for a
frame it does not have, and the step-end frame is regridded on the GPU onto
the f09 band (u, v, omega, T) and the CESM radiation column (T_full) with
MetWriter's math (log-p linear per column, bilinear lat/lon), the same code
that wrote the MET_SRC=file sets. So MET_SRC=emu is MET_SRC=file with the
frames made on demand: the same open_var surface (bc_clim.ClimArchive), the
same meta keys, and frames bit-identical to atmos_rollout.py's for the same
seed and bundle (scripts/validation/validate_met_emu.py).

Feedback (ATM_FEEDBACK=1). The anomaly radiation's aerosol heating Q [K/s] on
the band at frame j can be handed back with put_q(j, Q): it is remapped
conservatively onto the emulator grid (sin(lat)/lon and pressure overlap, as
atm_live.py) and Q x 6 h is added to the emulator's temperature after step
j -> j+1, so the heating is advected, damped by the emulator's own physics
and reaches the winds. Off by default: the shipped benchmark is one-way, like
every other linking. (Emulator response to imposed heating NOT validated
against the JCM dycore.)

Limits. One IC per bundle (P2s200: 1997-01-01T06 = model hour 8766), so H0
must be that hour. The atmosphere state is not checkpointed: on a RESUME the
emulator is re-integrated from the IC to the restart frame (deterministic on
the same GPU type and software; ~0.1-0.3 s per 6 h step).
"""
import json
import os
import sys
import threading
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ATMOS = os.path.join(os.path.dirname(HERE), 'atmos')
if ATMOS not in sys.path:
    sys.path.insert(0, ATMOS)


class _Field:
    """met[k][i] -> frame i of field k, stepping the emulator up to i first."""
    def __init__(self, src, k):
        self.src, self.k = src, k

    def __getitem__(self, i):
        return self.src.frame(int(i))[self.k]


def _x32():
    """The emulator's JAX parts (the fixer's Hines / dycore terms) are written for JAX's default float32;
    the coupled model runs with x64 on. Built and run inside this context they compute exactly what the
    standalone writers (atmos_rollout.py, atm_live.py) compute."""
    import jax.experimental
    return jax.experimental.disable_x64()


class EmuMet:
    """meta (the MET_SRC=file meta.json keys) + met {u, v, omega, T, T_full} indexable by frame."""

    def __init__(self, *a, **k):
        with _x32():
            self._init(*a, **k)

    def _init(self, bundle, h0, nframe, seed=0, fixer=1, massfix=1, keep=8, feedback=False, native=None):
        """native=None: regrid to the f09 band + CESM radiation column (MetWriter, from the bundle's
        cesm_grid.npz). native=dict(hyam, hybm, P0, lat, lon, p_lo_hpa, p_hi_hpa) (the JCM grid, top-first
        hybrid coefficients of the run's climatology): the emulator's OWN grid -- P2's T63 Gaussian L95 is
        the JCM grid -- so a frame is the emulator state with its levels reversed (surface-first ->
        top-first) and the band / radiation-column levels selected by coupling.py's and radiation.py's
        rules; no interpolation."""
        import torch
        from aide_atmos import core as Cm
        from aide_atmos.met_out import MetWriter, hours_since_archive_t0
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        self.torch, self.Cm = torch, Cm
        self.bundle = os.path.abspath(bundle)
        ics = sorted(f for f in os.listdir(bundle) if f.startswith('ic_') and f.endswith('.npz'))
        if len(ics) != 1:
            raise SystemExit(f'met_emu: expected one ic_<time>.npz in {bundle}, found {ics}')
        self.t0 = np.datetime64(ics[0][3:-4], 'h')
        self.hour0 = hours_since_archive_t0(self.t0)
        if int(h0) != self.hour0:
            raise SystemExit(f'met_emu: the bundle IC is {self.t0} = model hour {self.hour0}; '
                             f'set H0={self.hour0} (got {h0})')
        dev = 'cuda'
        if not torch.cuda.is_available():
            raise SystemExit('met_emu: torch sees no GPU')
        b = Cm.Bundle(bundle, dev)
        self.b, self.dev = b, dev
        self.net = b.build_net(); self.phys = Cm.Physics(b)
        self.WG = Cm.gauss_weights(dev)
        self.FIX = None
        if fixer:
            from aide_atmos.fixer import DycoreTerms, Fixer
            terms = DycoreTerms(os.path.join(bundle, 'fixer_dycore.npz'),
                                os.path.join(bundle, 'fixer_dycore.json'), b.force_npz)
            self.FIX = Fixer(dev, b.FN, b.mean, b.std, np.load(b.coeff), terms, latsm=4, levsm=2)
        self.massfix, self.seed, self.keep = int(massfix), int(seed), int(keep)
        self.nframe = int(nframe)
        prov = (f"src/met/met_emu.py: P2@200 IN THE LOOP, HL (hlr_p), massfix {massfix}, u-fixer {fixer} "
                f"(latsm 4, levsm 2), seed {seed}, IC {self.t0}, feedback {int(bool(feedback))}; bundle {self.bundle}")
        self.native = native
        if native is None:
            self.W = MetWriter(None, self.nframe, b, self.t0, self.hour0, prov,
                               extra_meta=dict(in_loop=True, seed=self.seed, torch=torch.__version__))
            self.meta = self.W.meta
        else:
            self.W = _NativeWriter(b, native, self.nframe, self.t0, self.hour0, prov,
                                   dict(in_loop=True, seed=self.seed, torch=torch.__version__))
            self.meta = self.W.meta
        self.met = {k: _Field(self, k) for k in ('u', 'v', 'omega', 'T', 'T_full', 'ps')}
        self.source = f'emu:{os.path.basename(self.bundle)}'
        ic = b.ic(self.t0)
        self._prev, self._cur = ic[:, :381], ic[:, 381:762]
        self._NG = Cm.NoiseGen(dev, seed=1000 + self.seed)
        self._GM0 = None
        self._t = 0                    # emulator step: the state held is frame self._t
        self._have = -1                # highest frame regridded
        self._lock = threading.RLock()
        self.feedback = bool(feedback)
        self._q = {}
        if self.feedback:
            self._init_qremap()
        self.t_net = self.t_regrid = 0.0; self.n_step = 0
        self.max_u = float('nan')
        # one frame ahead in a worker thread while the aerosol step runs (ATM_PREFETCH, default on;
        # off under feedback, where frame i+1 needs the heating of frame i). Same ops in the same order
        # on the same stream: the frames are bit-identical to stepping inline.
        self.prefetch = (not self.feedback) and os.environ.get('ATM_PREFETCH', '1') != '0'
        self._fut = None
        if self.prefetch:
            from concurrent.futures import ThreadPoolExecutor
            self._pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix='atm')

    def describe(self):
        return (f"JCM emulator P2@200 IN THE LOOP (bundle {self.bundle}, IC {self.t0} = hour {self.hour0}, "
                f"seed {self.seed}, massfix {self.massfix}, u-fixer {int(self.FIX is not None)}, "
                f"feedback {int(self.feedback)}); "
                + ("frames on the emulator's own grid (= the JCM grid), band + radiation-column levels selected"
                   if self.native is not None else "regridded on the GPU to the f09 band + CESM radiation column"))

    # ------------------------------------------------------------------
    def _put(self, i, ph):
        t = time.perf_counter()
        self.W.put(i, ph[0])
        for k in self.W.OUT:
            for j in [j for j in self.W.OUT[k] if j < i - self.keep]:
                del self.W.OUT[k][j]
        self._have = i
        self.t_regrid += time.perf_counter() - t

    def _step(self):
        """state frame t -> t+1 (network + HL physics + mass fix + u-fixer [+ Q]), regrid frame t+1."""
        torch, Cm, b = self.torch, self.Cm, self.b
        mean, std, ips, iu, iT = b.mean, b.std, b.ips, b.iu, b.iT
        t = self._t
        tc = self.t0 + np.timedelta64(6 * t, 'h'); tp = tc - np.timedelta64(6, 'h')
        a1 = time.perf_counter()
        with torch.no_grad():
            x = b.make_x(self._prev, self._cur, tp, tc)
            zz = self._NG.sample(1)
            out = self.phys(self.net(Cm.net_input(x, zz)), x)
            if self.massfix:
                ph_ = out[:, ips] * std[0, ips] + mean[0, ips]
                gm = (self.WG.view(1, -1, 1) * ph_).sum(-2).mean(-1)
                if t == 0:
                    self._GM0 = ((self.WG.view(1, -1, 1) * (x[:, 381 + ips] * std[0, ips] + mean[0, ips]))
                                 .sum(-2).mean(-1))
                out = out.clone(); out[:, ips] = (ph_ * (self._GM0 / gm).view(-1, 1, 1) - mean[0, ips]) / std[0, ips]
            mx = (out * std + mean)[:, iu[16:94]].abs().amax()
            self.max_u = float(mx)
            if not np.isfinite(self.max_u) or self.max_u > 300:
                raise SystemExit(f'met_emu: the atmosphere emulator DIED at step {t} '
                                 f'(max|u| levels 16-93 {self.max_u:.1f} m/s)')
            if self.FIX is not None:
                out = self.FIX.apply(x[:, 381:762], out, tc, tc + np.timedelta64(6, 'h'))
            if self.feedback and t >= 1:
                q = self._q.pop(t, None)
                if q is None:
                    raise SystemExit(f'met_emu: ATM_FEEDBACK=1 but no heating was handed back for frame {t}')
                out = out.clone(); out[:, iT] += self._q_to_emu(q) / std[:, iT]
            torch.cuda.synchronize()
            self.t_net += time.perf_counter() - a1
            self._put(t + 1, out * std + mean)
            self._prev, self._cur = x[:, 381:762].clone(), out
        self._t = t + 1; self.n_step += 1

    def frame(self, i):
        """{k: frame i} (numpy float32 on the host), integrating forward as needed."""
        with self._lock:
            if not 0 <= i < self.nframe:
                raise SystemExit(f'met_emu: frame {i} outside 0..{self.nframe - 1}')
            if self._have < 0:
                with self.torch.no_grad():
                    x = self.b.make_x(self._prev, self._cur, self.t0 - np.timedelta64(6, 'h'), self.t0)
                    self._put(0, x[:, 381:762] * self.b.std + self.b.mean)
            if i < self._have - self.keep:
                raise SystemExit(f'met_emu: frame {i} was evicted (have up to {self._have}, keep {self.keep})')
            while self._have < i:
                self._step_locked()
            out = {k: self.W.OUT[k][i] for k in self.W.OUT}
            if self.prefetch and self._have == i and i + 1 < self.nframe and (self._fut is None or self._fut.done()):
                if self._fut is not None:
                    self._fut.result()            # re-raise a worker failure (e.g. the emulator died)
                self._fut = self._pool.submit(self._step_locked)
            return out

    def _step_locked(self):
        with self._lock:
            with _x32():
                self._step()

    # ---- feedback: band Q (K/s, f09) -> emulator grid, K per 6 h (atm_live.py's remap) ----
    def _init_qremap(self):
        torch = self.torch
        if self.native is not None:
            return
        from aide_atmos.met_out import cesm_grid
        zc = np.load(self.b.coeff)
        lat_e, lon_e = zc['lat'].astype(np.float64), zc['lon'].astype(np.float64)
        plev_e = zc['hyam_pa'] / 100 + zc['hybm'] * 1000
        lat_c, lon_c, klevs, plev_c, krad, hy = cesm_grid(os.path.join(self.bundle, 'cesm_grid.npz'))
        nb, NX = len(klevs), lon_c.size

        def edges_center(c, lo_, hi_):
            e = np.empty(c.size + 1); e[1:-1] = 0.5 * (c[1:] + c[:-1]); e[0] = lo_; e[-1] = hi_; return e

        def cons(src_e, dst_e):
            Wt = np.maximum(0, np.minimum(dst_e[1:, None], src_e[None, 1:]) - np.maximum(dst_e[:-1, None], src_e[None, :-1]))
            return Wt / Wt.sum(1, keepdims=True)

        xq, wq = np.polynomial.legendre.leggauss(96); wq = wq[np.argsort(xq)]
        ge = np.concatenate([[-1.0], -1 + np.cumsum(wq)]); ge[-1] = 1.0
        WY = cons(np.sin(np.deg2rad(edges_center(lat_c, -90, 90))), ge)
        dl = lon_c[1] - lon_c[0]; fle = np.concatenate([lon_c - dl / 2, [lon_c[-1] + dl / 2]])
        dg = lon_e[1] - lon_e[0]; gle = np.concatenate([lon_e - dg / 2, [lon_e[-1] + dg / 2]])
        WX3 = cons(np.concatenate([fle[:-1] - 360, fle[:-1], fle + 360]), gle)
        WX = WX3[:, :NX] + WX3[:, NX:2 * NX] + WX3[:, 2 * NX:]

        def layer_edges(pc):
            return np.concatenate([[pc[0] - (pc[1] - pc[0]) / 2], (pc[1:] + pc[:-1]) / 2, [pc[-1] + (pc[-1] - pc[-2]) / 2]])
        pref_e = plev_e * 1.01325 * 100.0
        oe = np.argsort(pref_e); ee = layer_edges(pref_e[oe])
        eb = layer_edges(np.asarray(plev_c, np.float64))
        VZ = np.zeros((95, nb))
        for j, e in enumerate(oe):
            ov = np.maximum(0.0, np.minimum(ee[j + 1], eb[1:]) - np.maximum(ee[j], eb[:-1]))
            VZ[e] = ov / (ee[j + 1] - ee[j])
        tD = dict(device=self.dev, dtype=torch.float64)
        self._WY, self._WX, self._VZ = (torch.as_tensor(m, **tD) for m in (WY, WX, VZ))

    def _q_to_emu(self, q_band_ks):
        torch = self.torch
        if self.native is not None:      # same grid: band level k (top-first index) -> emulator level 94 - k
            q = torch.zeros((95,) + np.asarray(q_band_ks).shape[1:], device=self.dev, dtype=torch.float64)
            q[[94 - k for k in self.W.klevs]] = torch.as_tensor(np.asarray(q_band_ks), device=self.dev, dtype=torch.float64)
            return (q * 21600.0).float()[None]
        q = torch.as_tensor(np.asarray(q_band_ks), device=self.dev, dtype=torch.float64)
        qh = torch.einsum('yi,kij,xj->kyx', self._WY, q, self._WX)
        return (torch.einsum('ek,kyx->eyx', self._VZ, qh) * 21600.0).float()[None]

    def put_q(self, j, q_band_ks):
        """aerosol heating [K/s] on the band at frame j: applied to the step j -> j+1."""
        if self.feedback:
            with self._lock:
                self._q[int(j)] = np.asarray(q_band_ks, np.float64)


class _NativeWriter:
    """MetWriter's interface (meta, OUT{k: {frame: array}}, put) on the emulator's own grid (the JCM grid)."""

    def __init__(self, bundle, g, nframe, t0, hour0, provenance, extra_meta):
        from aide_atmos.met_out import ARCHIVE_T0
        zc = np.load(bundle.coeff)
        hyam = np.asarray(g['hyam'], np.float64) * float(g['P0']); hybm = np.asarray(g['hybm'], np.float64)
        e_hyam = zc['hyam_pa'][::-1].astype(np.float64); e_hybm = zc['hybm'][::-1].astype(np.float64)
        if hyam.size != 95 or np.abs(e_hyam - hyam).max() > 1e-2 or np.abs(e_hybm - hybm).max() > 1e-6:
            raise SystemExit('met_emu native: the run grid\'s hybrid levels are not the emulator\'s')
        if not (np.allclose(zc['lat'], g['lat'], atol=1e-4) and np.allclose(zc['lon'], g['lon'], atol=1e-4)):
            raise SystemExit('met_emu native: the run grid\'s lat/lon are not the emulator\'s')
        plev = hyam + hybm * 1.0e5                                   # coupling.py / radiation.py reference pressures
        band = np.where((plev >= g['p_lo_hpa'] * 100) & (plev <= g['p_hi_hpa'] * 100))[0]
        self.klevs = list(range(int(band.min()), int(band.max()) + 1))
        self.krad = [int(k) for k in np.where(plev >= 1.005)[0]]
        self.ib = [94 - k for k in self.klevs]; self.ir = [94 - k for k in self.krad]   # emulator (surface-first) indices
        self.b = bundle
        self.OUT = {k: {} for k in ("u", "v", "omega", "T", "T_full", "ps")}
        self.meta = dict(source="jcm-emulator", grid="native (emulator = JCM T63 Gaussian L95)", t0=str(np.datetime64(t0, 'h')),
                         hour0=int(hour0), archive_t0=str(ARCHIVE_T0), calendar="noleap hour index on the CESM h1 axis",
                         step_hours=6, nframe=int(nframe), klevs=self.klevs, plev_pa=[float(p) for p in plev[self.klevs]],
                         p_lo_hpa=float(g['p_lo_hpa']), p_hi_hpa=float(g['p_hi_hpa']),
                         lat=[float(x) for x in g['lat']], lon=[float(x) for x in g['lon']], krad=self.krad,
                         provenance=provenance, level_order="top->bottom (climatology order)", **extra_meta)

    def put(self, i, ph):
        """ph (381, 96, 192) physical, emulator order (surface-first levels)."""
        b = self.b
        for k, ii in (("u", b.iu), ("v", b.iv), ("T", b.iT), ("omega", b.iw)):
            self.OUT[k][i] = ph[ii][self.ib].float().cpu().numpy()
        self.OUT["T_full"][i] = ph[b.iT][self.ir].float().cpu().numpy()
        self.OUT["ps"][i] = (ph[b.ips] * 1e5).float().cpu().numpy()
