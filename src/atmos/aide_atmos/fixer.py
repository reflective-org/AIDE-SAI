"""In-process port of the option-1 zonal-mean momentum fixer (g2 code/fixer_lib.py + fixer_phys.py).

fixer_lib.Fixer replaced the m=0 u increment of each 6-h step in 15S-15N x 100-1 hPa (tapered) by
dt * 0.5 * (RHS(s0) + RHS(s1)), RHS = resolved zonal-mean momentum budget (torch, verbatim below) + the DYCORE's HL wave
drag + the DYCORE's Hines GWD zonal-mean u tendencies. The original evaluated those two dycore terms in a separate JAX
process running the full jcm-strat p20bt model (fixer_phys.py serve). Here they are evaluated in-process from the
numbers that model used (fixer_dycore.npz, written once by work/extract_fixer.py from the built model):

  HL    jcm_strat.qbo_nudging.QboNudging.tendency as patched by g2 hl_dyc_patch.install_force_patch (hlr_p.npz, stoch
        sigma 0), called inside JuckerColumns with the Gregorian tyear of the state's time (jcm.date.fraction_of_year_elapsed).
  Hines jcm_strat.gwd.HinesGwdLaunch -> jax-gcm hines_gwd (vendored hines.py), launch level 10, on the moist-air
        diagnostics MoistAirColumnState computes for the fixer's state (q = 0, geopotential = 0 -> height_half = 0,
        air_density = p_full / (rd T)).
Both return the lon-mean of the per-column u tendency, (95, 96 lat asc) surface-first, as fixer_phys.PhysEval.eval does.
"""
import os
import numpy as np

try:
    import torch
except ImportError:   # DycoreTerms alone (JAX) needs no torch
    torch = None
A_E = 6.371e6; OM = 7.292e-5; DT = 21600.0


class DycoreTerms:
    def __init__(self, fx_npz, hines_json, force_npz):
        import jax, jax.numpy as jnp
        from . import hines as H
        import json
        self.jax, self.jnp = jax, jnp
        z = np.load(fx_npz); hp = json.load(open(hines_json))["hines_params"]
        self.params = H.HinesParameters(**{k: jnp.asarray(v, jnp.float32) for k, v in hp.items()})
        self.launch = int(z["hines_launch_level"])
        f = lambda k: jnp.asarray(z[k])
        self.a_full, self.b_full, self.a_half, self.b_half = f("a_full"), f("b_full"), f("a_half"), f("b_half")
        self.w, self.lat_idx, self.counts = f("qbo_w"), f("qbo_lat_idx"), f("qbo_counts")
        assert np.array_equal(z["qbo_lat_idx"], np.arange(z["qbo_lat_idx"].size) % z["qbo_counts"].size), "columns not lon-major"
        self.per_day = float(z["qbo_k"]) * float(z["qbo_tau_days"]); self.nlat = int(z["qbo_counts"].size)
        self.g = {k: f("g3_" + k) for k in ("order", "inv", "eqw", "dz", "dp", "Gc")}
        from . import jcm_constants as JC
        self.JC = JC
        FP = np.load(force_npz)
        self.CS = jnp.asarray(FP["CS"], jnp.float32); self.L0 = int(FP["L0"]); th = FP["th"]
        BE, cE, wE, BW, cW, wW, adamp = (np.exp(th[0]), th[1], np.exp(th[2]), np.exp(th[3]), th[4], np.exp(th[5]), np.exp(th[6]))
        assert len(th) == 8
        self.dl = float(np.exp(th[7]))
        if int(FP["west"]): BE = 0.0
        self.adamp = adamp
        self.Bmag_E = jnp.asarray(np.where(FP["CS"] > 0, BE * np.exp(-((FP["CS"] - cE) / wE) ** 2), 0.0) * 1e-3, jnp.float32)
        self.Bmag_W = jnp.asarray(np.where(FP["CS"] < 0, BW * np.exp(-((FP["CS"] - cW) / wW) ** 2), 0.0) * 1e-3, jnp.float32)
        self.fscale = float(FP["fscale"]) if "fscale" in FP.files else 1.0
        self.Alin = jnp.asarray(FP["A"] if "A" in FP.files else np.zeros((95, 95)), jnp.float32)
        self.MULT1 = jnp.ones((2,), jnp.float32)            # stochastic multipliers at sigma 0 = exactly 1
        self.H = H
        # HL and Hines in SEPARATE jitted graphs and the lon-mean in a third, as fixer_phys's term-by-term capture: Hines is
        # then bit-identical to PhysEval.eval; HL stays within a few ulp (see _hl)
        self._fhl = jax.jit(self._hl); self._fhn = jax.jit(self._hines)
        self._fm = jax.jit(lambda a: a.reshape(a.shape[0], 192, 96).mean(1)[::-1])

    # ---- QboNudging.tendency (hlr_p patch), JAX order in, (nlev, ncols) out ----
    def _hl(self, u_cols, tyear):
        jax, jnp, g = self.jax, self.jnp, self.g
        # QboNudging.zonal_mean is jax.ops.segment_sum over the columns of each latitude row; on the GPU that scatter-add
        # uses atomics and is NOT reproducible run to run (measured: HL moves by a few ulp between identical runs). The
        # columns are lon-major (lat_idx = col % 96), so a reshape-sum over lon is the same sum in a fixed order:
        # deterministic, and within 1-6 ulp (rel <= 4e-7) of the original (whose order no deterministic variant reproduced).
        assert int(u_cols.shape[1]) == 192 * self.nlat
        ubar = u_cols.reshape(u_cols.shape[0], 192, self.nlat).sum(1) / self.counts[None, :]
        return self._hl_from_ubar(ubar, tyear)

    def _hl_from_ubar(self, ubar, tyear):
        jax, jnp, g = self.jax, self.jnp, self.g
        ueq = (ubar @ g["eqw"])[g["order"]]
        U = ueq[self.L0:]
        du = U[:, None] - self.CS[None, :]
        s0 = jnp.sign(du[:1, :]); s0 = jnp.where(s0 == 0, 1.0, s0)
        trans = jax.nn.sigmoid(jnp.clip(s0 * du / self.dl, -30, 30)) * jnp.exp(-self.adamp * g["dz"][:, None] / (du ** 2 + self.dl ** 2))
        trans = trans.at[0].set(1.0)
        alive = jnp.cumprod(trans, axis=0); tau = 0.0
        B = self.Bmag_E * self.MULT1[0] + self.Bmag_W * self.MULT1[1]
        flux = (B[None, :] * (-s0) * alive * jnp.exp(-tau)).sum(-1)
        X = jnp.zeros(95, jnp.float32).at[self.L0 + 1:].set(9.80616 * (flux[:-1] - flux[1:]) / g["dp"][1:] * 86400.0)
        for _ in range(2):
            X = X.at[1:-1].set(0.25 * X[:-2] + 0.5 * X[1:-1] + 0.25 * X[2:])
        ph = 2 * jnp.pi * tyear
        h = jnp.stack([jnp.ones_like(ph), jnp.cos(ph), jnp.sin(ph), jnp.cos(2 * ph), jnp.sin(2 * ph)])
        Gz = self.fscale * X + h @ g["Gc"] + self.Alin @ ueq
        Gj = Gz[g["inv"]]
        tend_lat = self.per_day * self.w * Gj[:, None]
        return tend_lat[:, self.lat_idx]

    def _hines(self, u, v, T, ps):
        """u, v, T (95 top-first, ncols lon-major) float32 physical; ps (ncols,) normalised -> Hines du/dt (95, ncols)."""
        jax, jnp, JC = self.jax, self.jnp, self.JC
        sp = ps * JC.p0
        p_full = self.a_full[:, None] + self.b_full[:, None] * sp[None]
        p_half = self.a_half[:, None] + self.b_half[:, None] * sp[None]
        hgt = jnp.zeros_like(T)                                     # geopotential 0 -> height_full 0
        hhi = (hgt[1:] + hgt[:-1]) / 2
        h_half = jnp.concatenate(((hgt[0] + (hgt[0] - hhi[0]))[None], hhi, (hgt[-1] - (hhi[-1] - hgt[-1]))[None]), axis=0)
        rho = p_full / (JC.rd * T)
        layer_mass = (p_half[1:, :] - p_half[:-1, :]) / JC.grav
        tend, _ = jax.vmap(lambda *a: self.H.hines_gwd(*a, self.params, launch_level=self.launch),
                           in_axes=(1, 1, 1, 1, 1, 1, 1, 1), out_axes=(0, 0))(p_half, p_full, h_half, rho, layer_mass, T, u, v)
        return tend.dudt.T

    @staticmethod
    def tyear(t64):
        """jcm.date.fraction_of_year_elapsed at t (Gregorian), float32 as SolarGeometry.tyear."""
        import jax.numpy as jnp
        t = np.datetime64(t64, "s"); y = int(str(t)[:4])
        doy = int((t.astype("datetime64[D]") - np.datetime64(f"{y}-01-01")).astype(int))
        secs = int((t - t.astype("datetime64[D]")).astype(int))
        diy = 366 if (y % 4 == 0 and y % 100 != 0) or y % 400 == 0 else 365
        return jnp.asarray((jnp.int32(doy) + jnp.int32(secs) / 86400.0) / jnp.int32(diy), jnp.float32)

    def eval(self, u, v, T, ps, t64):
        """As fixer_phys.PhysEval.eval: u, v, T (95, 96 lat, 192 lon) sfc-first float32; ps (96, 192) normalised."""
        jnp = self.jnp
        to_top = lambda a: jnp.asarray(np.ascontiguousarray(np.asarray(a, np.float32).transpose(0, 2, 1)[::-1])).reshape(95, -1)
        uc, vc, Tc = to_top(u), to_top(v), to_top(T)
        hn = self._fhn(uc, vc, Tc, jnp.asarray(np.ascontiguousarray(np.asarray(ps, np.float32).T)).reshape(-1))
        hl = self._fhl(uc, self.tyear(t64))
        return {"HL": np.asarray(self._fm(hl)), "HinesGwdLaunch": np.asarray(self._fm(hn))}


class Fixer:
    """fixer_lib.Fixer (mode 'fix'), with the JAX server replaced by DycoreTerms in-process."""
    def __init__(s, dev, FN, mean, std, z, terms, latsm=4, levsm=2, taper_lat=(10.0, 15.0), p_hi=100.0, p_lo=1.0, nlv_taper=2):
        s.dev = dev; s.mean, s.std = mean, std; s.terms = terms
        s.iu = [FN.index(f"u_wind_h{l}") for l in range(95)]; s.iv = [FN.index(f"v_wind_h{l}") for l in range(95)]; s.iw = [FN.index(f"omega_h{l}") for l in range(95)]
        p = z["hyam_pa"] / 100 + z["hybm"] * 1000; lat = np.asarray(z["lat"], np.float64); phi = np.deg2rad(lat); s.p = p
        eye = np.eye(96)
        s.Dphi = torch.tensor(np.stack([np.gradient(eye[j], phi, edge_order=2) for j in range(96)], 1), dtype=torch.float64, device=dev)
        pl = p * 100.0; I = np.eye(95); s.Dp = torch.tensor(np.stack([np.gradient(I[j], pl, edge_order=2) for j in range(95)], 1), dtype=torch.float64, device=dev)
        s.pl = pl; s.cs = torch.tensor(np.cos(phi), device=dev); s.tn = torch.tensor(np.tan(phi), device=dev); s.f = torch.tensor(2 * OM * np.sin(phi), device=dev)
        s.dpk = torch.tensor(pl[:-1] - pl[1:], device=dev)
        a = np.abs(lat); wl = np.where(a <= taper_lat[0], 1.0, np.where(a >= taper_lat[1], 0.0, np.cos(0.5 * np.pi * (a - taper_lat[0]) / (taper_lat[1] - taper_lat[0])) ** 2))
        idx = np.where((p <= p_hi) & (p >= p_lo))[0]; wv = np.zeros(95); wv[idx] = 1.0
        for j in range(nlv_taper):
            wv[idx[0] + j] = (j + 1) / (nlv_taper + 1); wv[idx[-1] - j] = (j + 1) / (nlv_taper + 1)
        s.W = torch.tensor(wv[:, None] * wl[None, :], dtype=torch.float64, device=dev)
        s.latsm, s.levsm = latsm, levsm
        s.Uo = np.zeros((2, 2, 95, 96), np.float32)

    def _phys(s, p):
        return p[:, s.iu].double(), p[:, s.iv].double(), p[:, s.iw].double()

    def _dyn(s, U, V, W):
        ub, vb, wb = U.mean(-1), V.mean(-1), W.mean(-1)
        Dv = torch.einsum("blj,ij->bli", vb * s.cs, s.Dphi) / (A_E * s.cs)
        c = 0.5 * (Dv[:, :-1] + Dv[:, 1:]) * s.dpk[None, :, None]
        om = torch.zeros_like(Dv); om[:, :-1] = -torch.flip(torch.cumsum(torch.flip(c, [1]), 1), [1])
        uv = ((U - ub[..., None]) * (V - vb[..., None])).mean(-1); uw = ((U - ub[..., None]) * (W - wb[..., None])).mean(-1)
        ddp = lambda F: torch.einsum("bjl,ij->bil", F, s.Dp)
        ddy = lambda F: torch.einsum("blj,ij->bli", F, s.Dphi)
        R = dict(adv_y=-vb * ddy(ub) / A_E, adv_p=-om * ddp(ub), cor=s.f * vb, curv=ub * vb * s.tn / A_E,
                 eddy_y=-ddy(s.cs ** 2 * uv) / (A_E * s.cs ** 2), eddy_p=-ddp(uw))
        return R, ub

    def apply(s, s0n, s1n, t0, t1):
        p0 = s0n.float() * s.std + s.mean; p1 = s1n.float() * s.std + s.mean
        for j, (p, t) in enumerate(((p0, t0), (p1, t1))):
            o = s.terms.eval(p[0, s.iu].cpu().numpy(), p[0, s.iv].cpu().numpy(), p[0, 190:285].cpu().numpy(), p[0, 380].cpu().numpy(), t)
            s.Uo[j, 0] = o["HL"]; s.Uo[j, 1] = o["HinesGwdLaunch"]
        ph = torch.tensor(s.Uo, dtype=torch.float64, device=s.dev)
        R0, ub0 = s._dyn(*s._phys(p0)); R1, ub1 = s._dyn(*s._phys(p1))
        r0 = sum(R0.values()) + ph[0, 0] + ph[0, 1]; r1 = sum(R1.values()) + ph[1, 0] + ph[1, 1]
        fix = DT * 0.5 * (r0 + r1); net = ub1 - ub0
        dlt = fix - net
        for _ in range(s.latsm):
            dlt = torch.cat([dlt[..., :1], 0.25 * dlt[..., :-2] + 0.5 * dlt[..., 1:-1] + 0.25 * dlt[..., 2:], dlt[..., -1:]], -1)
        for _ in range(s.levsm):
            dlt = torch.cat([dlt[:, :1], 0.25 * dlt[:, :-2] + 0.5 * dlt[:, 1:-1] + 0.25 * dlt[:, 2:], dlt[:, -1:]], 1)
        corr = s.W * dlt
        out = s1n.clone(); out[:, s.iu] += (corr[..., None] / s.std[:, s.iu].double()).float()
        return out
