"""Standalone P2@200 stepper: everything sai_run.py's step does, rebuilt from a bundle (make_bundle.py) with no
PARADIS trainer, datamodule, corpus or /data path. Each piece names the original it reproduces:

  Net / build_net      g2lib_hl.Net + make_net (PARADIS core from cfg.model + zero-init 1x1 lin skip x tend scale);
                       trainer.LitParadis adds VerticalMixing; clamp_output is off for P2.
  Inputs               gate_mine.FastIO.get: [state t-1 | state t | year-progress forcings (n_in each) | ERA5 slots | year |
                       constants]; after step 0 the state blocks are the run's own, the ERA5 slots carry generator noise and
                       the year channel is zeroed for the net (g2lib_hl.net_input) but read by the physics.
  NoiseGen             g2lib_hl.NoiseGen (torch_harmonics InverseRealSHT, seeded torch.Generator on the device).
  phys                 g2lib_hl.phys with QBO_SRC=hl: JCMPhysicsP20bt (prescribe=False, qbo=False) + hl_delta.
  massfix              sai_run.py --massfix 1.
"""
import json, math, os
import numpy as np
import torch
from . import time_vars
from .paradis import Paradis, VerticalMixing
from .jcm_physics_p20bt import JCMPhysicsP20bt


class _Cfg(dict):
    """dict with attribute access and .get, enough for Paradis(cfg.model.get(...))."""
    def __getattr__(self, k):
        try:
            v = self[k]
        except KeyError as e:
            raise AttributeError(k) from e
        return _Cfg(v) if isinstance(v, dict) else v


class _DS:
    def __init__(self, dims):
        self.num_in_dyn_features = dims["num_in_dyn_features"]; self.num_in_static_features = dims["num_in_static_features"]


class _DM:
    def __init__(self, dims):
        self.dataset = _DS(dims); self.num_out_features = dims["num_out_features"]


class Net(torch.nn.Module):
    """g2lib_hl.Net: PARADIS core + zero-init linear 1x1 skip (identical state-dict layout)."""
    def __init__(s, core, ts, cin=821, cout=381):
        super().__init__()
        s.lin = torch.nn.Conv2d(cin, cout, 1); torch.nn.init.zeros_(s.lin.weight); torch.nn.init.zeros_(s.lin.bias)
        s.core = core; s.register_buffer("ts", ts.clone(), persistent=False)

    def forward(s, x):
        return s.core(x) + s.lin(x) * s.ts


class Bundle:
    def __init__(self, path, dev="cuda"):
        self.path, self.dev = path, dev
        self.meta = json.load(open(os.path.join(path, "model.json")))
        z = np.load(os.path.join(path, "norm.npz"))
        t = lambda a: torch.as_tensor(np.asarray(a), dtype=torch.float32, device=dev)
        self.mean, self.std, self.ts = t(z["mean"]), t(z["std"]), t(z["ts"])
        self.const = t(z["const"])                                     # (6, H, W)
        self.lat, self.lon = np.asarray(z["lat"]), np.asarray(z["lon"])
        self.lat_rad_grid, self.lon_rad_grid = np.asarray(z["lat_rad_grid"]), np.asarray(z["lon_rad_grid"])
        self.FN = self.meta["FN"]; self.layout = self.meta["layout"]
        FN = self.FN
        self.iu = [FN.index(f"u_wind_h{l}") for l in range(95)]; self.iv = [FN.index(f"v_wind_h{l}") for l in range(95)]
        self.iT = [FN.index(f"temperature_h{l}") for l in range(95)]; self.iw = [FN.index(f"omega_h{l}") for l in range(95)]
        self.ips = FN.index("normalized_surface_pressure")
        self.coeff = os.path.join(path, "jcm_analytic_p20bt.npz"); self.force_npz = os.path.join(path, "hlr_p.npz")

    def build_net(self):
        m, d = self.meta["model"], self.meta["dims"]
        cfg = _Cfg(model=dict(m), data=dict(grid=d["grid"]))
        lat_g = torch.as_tensor(self.lat_rad_grid); lon_g = torch.as_tensor(self.lon_rad_grid)
        core = Paradis(_DM(d), cfg, lat_g, lon_g)
        if m["vertical_mixing"]:
            core.vertical_mixing = VerticalMixing(d["num_atm_vars"], d["num_levels"], d["num_sfc_vars"], m["vertical_mixing_hidden_mult"])
        net = Net(core, self.ts).to(self.dev)
        sd = torch.load(os.path.join(self.path, "net.pt"), map_location="cpu", weights_only=False)["net"]
        net.load_state_dict(sd, strict=True)
        return net.eval()

    def build_phys(self):
        idx = {k: torch.tensor(v) for k, v in self.meta["phys_idx"].items()}
        return JCMPhysicsP20bt(self.coeff, idx, nlev=95).to(self.dev)

    def ic(self, t0):
        z = np.load(os.path.join(self.path, f"ic_{np.datetime_as_string(np.datetime64(t0), unit='h')}.npz"))
        return torch.as_tensor(z["x"], device=self.dev)[None]            # (1, 762, H, W) normalised

    # ---- the non-state input channels of sample j (times t_prev = t0 + 6h (j-1), t_cur) ----
    def forcing_channels(self, t_prev, t_cur):
        L = self.layout; n_in = L["n_in"]
        fd = time_vars.time_forcings(np.array([t_prev, t_cur], dtype="datetime64[ns]"), gregorian=L["gregorian"])
        H, W = self.const.shape[-2:]
        fo = [torch.tensor(fd[v], dtype=torch.float32).view(1, 1, -1).expand(H, W, n_in) for v in L["forcing_vars"]]
        f = torch.cat(fo, -1).permute(2, 0, 1)                           # (n_vars*n_in, H, W) as FastIO
        yr = (int(str(np.datetime64(t_cur, "s"))[:4]) - 2000) / 10.0
        return f.to(self.dev), yr

    def make_x(self, state_prev, state_cur, t_prev, t_cur):
        """(1, 821, H, W) DATASET-style input (ERA5 slots 0, year channel set), as FastIO.get returns after the state swap."""
        f, yr = self.forcing_channels(t_prev, t_cur)
        H, W = self.const.shape[-2:]
        e5 = torch.zeros((self.layout["era5"][1] - self.layout["era5"][0], H, W), device=self.dev)
        y = torch.full((1, H, W), yr, device=self.dev)
        x = torch.cat([state_prev[0], state_cur[0], f, e5, y, self.const], 0)[None]
        assert x.shape[1] == 821
        return x


def net_input(x, noise=None):
    x = x.clone(); x[:, 766:814] = 0.0 if noise is None else noise; x[:, 814] = 0.0
    return x


NOISE_SCALES = (4.0, 10.0, 25.0)


class NoiseGen:
    """g2lib_hl.NoiseGen, verbatim."""
    def __init__(s, dev, nlat=96, nlon=192, scales=NOISE_SCALES, nper=16, seed=0):
        import torch_harmonics as th
        s.isht = th.InverseRealSHT(nlat, nlon, grid="legendre-gauss").to(dev)
        s.lmax, s.mmax = s.isht.lmax, s.isht.mmax; s.dev = dev; s.nper = nper; s.scales = scales
        l = torch.arange(s.lmax, device=dev, dtype=torch.float32).view(-1, 1)
        m = torch.arange(s.mmax, device=dev).view(1, -1)
        tri = (m <= l).float()
        s.amp = torch.stack([torch.exp(-(l / lc) ** 2) * tri for lc in scales])
        s.amp[:, 0, 0] = 0.0
        s.g = torch.Generator(device=dev); s.g.manual_seed(seed)
        z = s._raw(64); s.norm = z.std(dim=(0, 2, 3, 4), keepdim=True)[0]

    def _raw(s, B):
        S = len(s.scales)
        re = torch.randn(B, S, s.nper, s.lmax, s.mmax, device=s.dev, generator=s.g)
        im = torch.randn(B, S, s.nper, s.lmax, s.mmax, device=s.dev, generator=s.g); im[..., 0] = 0.0
        C = torch.complex(re, im) * s.amp[None, :, None]
        return s.isht(C)

    def sample(s, B):
        z = s._raw(B) / s.norm[None]
        return z.reshape(B, -1, *z.shape[-2:])


class Physics:
    """g2lib_hl.phys (QBO_SRC='hl', qbo flag False) + hl_delta."""
    def __init__(self, bundle):
        self.b = bundle; self.jp = bundle.build_phys(); self.dev = bundle.dev
        self.mean, self.std = bundle.mean, bundle.std
        L = bundle.layout; self.isin, self.icos = L["isin"], L["icos"]
        self._hl = None

    def split(self, z):
        p = z.float() * self.std + self.mean
        return {k: p[:, getattr(self.jp, f"i_{k}")] for k in ("u", "v", "T", "omega", "ps")}

    def join(self, d, like):
        p = torch.empty(like.shape, dtype=torch.float32, device=like.device)
        for k in ("u", "v", "T", "omega", "ps"):
            p[:, getattr(self.jp, f"i_{k}")] = d[k]
        return ((p - self.mean) / self.std).to(like.dtype)

    def __call__(self, out, x):
        cur = x[:, 381:762]
        sin_y, cos_y = x[:, self.isin], x[:, self.icos]
        tyear = (torch.atan2(sin_y[:, 0, 0], cos_y[:, 0, 0]) / (2 * math.pi)) % 1.0
        yr = x[:, 814, 0, 0] * 10.0 + 2000.0; year = torch.round(yr).to(torch.int64)
        assert float((yr - year).abs().max()) < 1e-3
        d = self.jp(self.split(out), self.split(cur), self.split(out), tyear, prescribe=False, qbo=False, year=year)
        d = dict(d); d["u"] = d["u"] + self.hl_delta(d["u"], tyear)
        return self.join(d, out)

    # ---- HL QBO wave drag: g2lib_hl.hl_params / hl_G / hl_delta verbatim ----
    def hl_params(self):
        dev = self.dev
        z = np.load(self.b.coeff); FP = np.load(self.b.force_npz); th = FP["th"]; assert len(th) == 8
        CS = FP["CS"].astype(np.float64); L0 = int(FP["L0"])
        BE, cE, wE, BW, cW, wW, adamp = (np.exp(th[0]), th[1], np.exp(th[2]), np.exp(th[3]), th[4], np.exp(th[5]), np.exp(th[6])); dl = float(np.exp(th[7]))
        if int(FP["west"]): BE = 0.0
        B = np.where(CS > 0, BE * np.exp(-((CS - cE) / wE) ** 2), 0.0) * 1e-3 + np.where(CS < 0, BW * np.exp(-((CS - cW) / wW) ** 2), 0.0) * 1e-3
        ps = z["sigma_eff"].astype(np.float64) * 101325.0 / 100.0; assert np.all(np.diff(ps) < 0)
        up = np.arange(L0, 95); dz = np.r_[0.0, 7.0 * np.log(ps[up[:-1]] / ps[up[1:]])]; dp = np.r_[1.0, (ps[up[:-1]] - ps[up[1:]]) * 100.0]
        lat = np.asarray(z["lat"], np.float64); eqw = np.where(np.abs(lat) <= 5, np.cos(np.deg2rad(lat)), 0.0); eqw /= eqw.sum()
        t = lambda a_: torch.tensor(np.asarray(a_), dtype=torch.float32, device=dev)
        return dict(CS=t(CS), L0=L0, B=t(B), adamp=float(adamp), dl=dl, dz=t(dz), dp=t(dp), eqw=t(eqw), Gc=t(FP["Gc"]), fscale=float(FP["fscale"]),
                    A=t(FP["A"] if "A" in FP.files else np.zeros((95, 95))), W=t(z["W_QBO"]))

    @torch.no_grad()
    def hl_G(self, P, ueq, tyear):
        U = ueq[:, P["L0"]:]; du = U[:, :, None] - P["CS"][None, None, :]
        s0 = torch.sign(du[:, :1, :]); s0 = torch.where(s0 == 0, torch.ones_like(s0), s0)
        trans = torch.sigmoid(torch.clamp(s0 * du / P["dl"], -30, 30)) * torch.exp(-P["adamp"] * P["dz"][None, :, None] / (du ** 2 + P["dl"] ** 2))
        trans = torch.cat([torch.ones_like(trans[:, :1]), trans[:, 1:]], 1); alive = torch.cumprod(trans, 1)
        flux = (P["B"][None, None, :] * (-s0) * alive).sum(-1)
        X = torch.zeros_like(ueq); X[:, P["L0"] + 1:] = 9.80616 * (flux[:, :-1] - flux[:, 1:]) / P["dp"][None, 1:] * 86400.0
        for _ in range(2): X[:, 1:-1] = 0.25 * X[:, :-2] + 0.5 * X[:, 1:-1] + 0.25 * X[:, 2:]
        ph = 2 * math.pi * tyear; h = torch.stack([torch.ones_like(ph), torch.cos(ph), torch.sin(ph), torch.cos(2 * ph), torch.sin(2 * ph)], 1)
        return P["fscale"] * X + h @ P["Gc"] + ueq @ P["A"].T

    @torch.no_grad()
    def hl_delta(self, u, tyear):
        if self._hl is None:
            self._hl = self.hl_params()
        P = self._hl; ueq = torch.einsum("blyx,y->bl", u.float(), P["eqw"]) / u.shape[-1]
        Gz = self.hl_G(P, ueq, tyear.float())
        return (21600.0 / 86400.0) * (P["W"][None, :, :, None] * Gz[:, :, None, None]).expand_as(u).to(u.dtype)


def gauss_weights(dev):
    xg, wg = np.polynomial.legendre.leggauss(96); wg = wg[np.argsort(xg)]
    return torch.tensor(wg / wg.sum(), device=dev, dtype=torch.float32)
