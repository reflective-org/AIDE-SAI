"""Write emulator frames as a MET_SRC=file directory (AIDE-SAI-link scripts/met/emu_to_met.py layout), regridded on the
GPU. The math is emu_to_met.py's (log-p linear per column onto the band's fixed pressures and, for T_full, onto the
CESM hybrid levels the radiation column uses at the emulator's own ps; bilinear lat/lon, nearest row beyond the Gaussian
range, periodic lon), as ported in coupled_atm/scripts/atm_server.py and checked there bit-identical to emu_to_met.py.
The CESM grid comes from the bundle's cesm_grid.npz instead of the CESM archive."""
import json, os
import numpy as np
import torch

ARCHIVE_T0 = np.datetime64('1996-01-01T00:00:00')
PS_REF = 1.0e5


def hours_since_archive_t0(t0):
    """emu_to_met.hours_since_archive_t0: noleap hour index on the CESM h1 axis of a real-calendar time."""
    t = np.datetime64(t0)
    h = (t - ARCHIVE_T0) / np.timedelta64(1, 'h')
    leaps = sum(1 for y in range(1996, int(str(t)[:4]) + 1)
                if ((y % 4 == 0 and y % 100 != 0) or y % 400 == 0) and np.datetime64(f'{y}-02-29') < t)
    return int(round(h)) - 24 * leaps


def bilinear_weights(src, dst, periodic=False, period=360.0):
    src = np.asarray(src, np.float64); dst = np.asarray(dst, np.float64)
    if periodic:
        ext = np.concatenate([src, [src[0] + period]])
        d = np.mod(dst - src[0], period) + src[0]
        i0 = np.clip(np.searchsorted(ext, d, side='right') - 1, 0, src.size - 1)
        i1 = (i0 + 1) % src.size
        w1 = (d - ext[i0]) / (ext[i0 + 1] - ext[i0])
        return i0, i1, 1.0 - w1, w1
    i0 = np.clip(np.searchsorted(src, dst, side='right') - 1, 0, src.size - 2)
    i1 = i0 + 1
    w1 = np.clip((dst - src[i0]) / (src[i1] - src[i0]), 0.0, 1.0)
    return i0, i1, 1.0 - w1, w1


def cesm_grid(grid_npz, p_lo_hpa=1.0, p_hi_hpa=150.0):
    """emu_to_met.cesm_grid from the stored grid: lat, lon, klevs (band), plev (band, Pa), krad, hy."""
    z = np.load(grid_npz)
    lat, lon = z['lat'], z['lon']
    plev = z['hyam'] * z['P0'] + z['hybm'] * PS_REF
    band = np.where((plev >= p_lo_hpa * 100) & (plev <= p_hi_hpa * 100))[0]
    klevs = list(range(band.min(), band.max() + 1))
    krad = np.where(plev >= 1.005)[0]
    hy = dict(hyam=z['hyam'], hybm=z['hybm'], P0=float(z['P0']))
    assert np.all(z['hybm'][klevs] == 0.0)
    return lat, lon, klevs, plev[klevs], krad, hy


class MetWriter:
    def __init__(self, out_dir, nframe, bundle, t0, hour0, provenance, extra_meta=None):
        dev = bundle.dev; self.dev = dev; self.out = out_dir; self.nf = nframe
        zc = np.load(bundle.coeff)
        lat_e, lon_e = zc['lat'].astype(np.float64), zc['lon'].astype(np.float64)
        hyam, hybm = zc['hyam_pa'].astype(np.float64), zc['hybm'].astype(np.float64)
        assert np.allclose(bundle.lat, lat_e) and np.allclose(bundle.lon, lon_e)
        lat_c, lon_c, klevs, plev_c, krad, hy = cesm_grid(os.path.join(bundle.path, 'cesm_grid.npz'))
        nb, nkr, NY, NX = len(klevs), krad.size, lat_c.size, lon_c.size
        p_ref = hyam + hybm * PS_REF
        kk = np.where((p_ref >= plev_c[0] * 0.5) & (p_ref <= plev_c[-1] * 1.5))[0]
        self.ksel = np.arange(max(kk.min() - 1, 0), min(kk.max() + 2, 95))[::-1].copy()
        self.ksel_all = np.arange(95)[::-1].copy()
        tD = dict(device=dev, dtype=torch.float64); self.tD = tD
        self.la0, self.la1, wa0, wa1 = [torch.as_tensor(v, device=dev) for v in bilinear_weights(lat_e, lat_c)]
        self.lo0, self.lo1, wo0, wo1 = [torch.as_tensor(v, device=dev) for v in bilinear_weights(lon_e, lon_c, periodic=True)]
        self.wa0, self.wa1, self.wo0, self.wo1 = [w.to(torch.float64) for w in (wa0, wa1, wo0, wo1)]
        self.hyam_t, self.hybm_t = torch.as_tensor(hyam, **tD), torch.as_tensor(hybm, **tD)
        self.plev_c_t = torch.as_tensor(plev_c, **tD)
        self.hyk = torch.as_tensor(hy['hyam'][krad] * hy['P0'], **tD); self.hbk = torch.as_tensor(hy['hybm'][krad], **tD)
        self.nb, self.nkr = nb, nkr
        self.b = bundle
        os.makedirs(out_dir, exist_ok=True)
        self.OUT = {k: np.lib.format.open_memmap(os.path.join(out_dir, f"{k}.npy"), mode="w+", dtype=np.float32, shape=(nframe, nb, NY, NX))
                    for k in ("u", "v", "omega", "T")}
        self.OUT["T_full"] = np.lib.format.open_memmap(os.path.join(out_dir, "T_full.npy"), mode="w+", dtype=np.float32, shape=(nframe, nkr, NY, NX))
        self.OUT["ps"] = np.lib.format.open_memmap(os.path.join(out_dir, "ps.npy"), mode="w+", dtype=np.float32, shape=(nframe, NY, NX))
        self.meta = dict(source="jcm-emulator", t0=str(np.datetime64(t0, 'h')), hour0=int(hour0), archive_t0=str(ARCHIVE_T0),
                         calendar="noleap hour index on the CESM h1 axis", step_hours=6, nframe=int(nframe),
                         klevs=[int(k) for k in klevs], plev_pa=[float(p) for p in plev_c], p_lo_hpa=1.0, p_hi_hpa=150.0,
                         lat=[float(x) for x in lat_c], lon=[float(x) for x in lon_c], krad=[int(k) for k in krad],
                         emu_levels_used=[int(self.ksel.min()), int(self.ksel.max())], provenance=provenance,
                         fields={'u': 'm/s', 'v': 'm/s', 'omega': 'Pa/s', 'T': 'K', 'ps': 'Pa', 'T_full': 'K on krad'},
                         level_order="CESM top->bottom (klevs ascending)", **(extra_meta or {}))

    def _h(self, f):
        g = f[..., self.la0, :] * self.wa0[:, None] + f[..., self.la1, :] * self.wa1[:, None]
        return g[..., :, self.lo0] * self.wo0 + g[..., :, self.lo1] * self.wo1

    @staticmethod
    def _logp(f, p_src, p_dst, clamp):
        lp = torch.log(p_src); lt = torch.log(p_dst)
        if not clamp:
            assert bool((lt[0] >= lp[0]).all()) and bool((lt[-1] <= lp[-1]).all()), "band pressures outside the emulator column"
        k0 = ((lp[None] <= lt[:, None]).sum(1) - 1).clamp(0, lp.shape[0] - 2); k1 = k0 + 1
        l0 = torch.gather(lp, 0, k0); l1 = torch.gather(lp, 0, k1); w = (lt - l0) / (l1 - l0)
        if clamp: w = w.clamp(0.0, 1.0)
        return torch.gather(f, 0, k0) * (1 - w) + torch.gather(f, 0, k1) * w

    def put(self, i, ph):
        """ph (381, 96, 192) physical, emulator order (surface-first levels)."""
        b = self.b; ncol = 96 * 192; ksel, nb = self.ksel, self.nb
        ps = ph[b.ips].to(torch.float64) * 1e5; psf = ps.reshape(1, ncol)
        p_src = self.hyam_t[ksel][:, None] + self.hybm_t[ksel][:, None] * psf
        pd_ = self.plev_c_t[:, None].expand(nb, ncol)
        for k, ii in (("u", b.iu), ("v", b.iv), ("T", b.iT), ("omega", b.iw)):
            f = ph[ii].to(torch.float64)[ksel].reshape(ksel.size, ncol)
            self.OUT[k][i] = self._h(self._logp(f, p_src, pd_, False).reshape(nb, 96, 192)).float().cpu().numpy()
        self.OUT["ps"][i] = self._h(ps).float().cpu().numpy()
        p_all = self.hyam_t[self.ksel_all][:, None] + self.hybm_t[self.ksel_all][:, None] * psf
        p_tgt = self.hyk[:, None] + self.hbk[:, None] * psf
        f = ph[b.iT].to(torch.float64)[self.ksel_all].reshape(95, ncol)
        self.OUT["T_full"][i] = self._h(self._logp(f, p_all, p_tgt, True).reshape(self.nkr, 96, 192)).float().cpu().numpy()

    def close(self, nwritten):
        for a in self.OUT.values():
            a.flush()
        self.meta["nframe_written"] = int(nwritten)
        json.dump(self.meta, open(os.path.join(self.out, "meta.json"), "w"), indent=1)
