#!/usr/bin/env python3
"""Live atmosphere for the linked model: the JCM emulator P2@200 stepped 6 h at a time IN LOCKSTEP with the coupled run, which
reads the frames with MET_LIVE=1 (src/met/met_live.py) and, under RAD_MODE=feedback, hands back the aerosol heating Q.

Same recipe and code as atmos_rollout.py (one leg): nudge-off, HL QBO wave drag, global-mean ps fix, u-fixer, noise seed
1000 + seed, everything from one bundle directory. With --q-mode none the frames are atmos_rollout.py's for the same seed.

Exchange directory XD (a MET_SRC=file directory being filled):
  u/v/omega/T.npy (nframe, 24, 192, 288), T_full.npy, ps.npy, meta.json     preallocated; meta.json written first
  progress          number of frames written (frames 0..progress-1 are final)
  q/q_<j>.npy       heating K/s on the band (f09 grid), written by the coupled run for frame j (RAD_MODE=feedback)
  FAILED            written by either side on death
  atm_zm.npz        zonal means (95 lev, 96 lat) of u, T and the applied Q (K/6h), emulator grid, every 40 steps
Step j -> j+1 adds Q_j x 6 h to T after network + physics + massfix + fixer. Q_0 = 0 (the coupled run has no radiation at
frame 0). Q remap band f09 -> emulator: conservative in sin(lat) / lon and in pressure onto the emulator's reference
pressures, 0 outside the band. Vertical: pressure-overlap remap, conserves sum Q dp exactly (see VZ below).

  atm_live.py --bundle B --xd XD --nframe N [--t0 1997-01-01T06] [--seed 0] [--q-mode live|none] [--gpu G]
"""
import argparse, ctypes, json, os, sys, time
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
ap = argparse.ArgumentParser()
ap.add_argument("--bundle", required=True); ap.add_argument("--xd", required=True); ap.add_argument("--nframe", type=int, required=True)
ap.add_argument("--t0", default="1997-01-01T06"); ap.add_argument("--hour0", type=int, default=None, help="check: archive hour of t0")
ap.add_argument("--seed", type=int, default=0); ap.add_argument("--fixer", type=int, default=1); ap.add_argument("--massfix", type=int, default=1)
ap.add_argument("--q-mode", dest="q_mode", default="live", choices=["none", "live"])
ap.add_argument("--q-scale", dest="q_scale", type=float, default=1.0)
ap.add_argument("--timeout", type=float, default=1800.0); ap.add_argument("--gpu", default=None)
a = ap.parse_args()
if a.gpu is not None:
    os.environ["CUDA_VISIBLE_DEVICES"] = str(a.gpu)
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
XD = a.xd
os.makedirs(XD, exist_ok=True)
for f in ("FAILED", "progress", "meta.json"):
    if os.path.exists(os.path.join(XD, f)):
        os.remove(os.path.join(XD, f))


def fail(msg):
    open(os.path.join(XD, "FAILED"), "w").write(f"atm_live: {msg}\n"); print("FAILED:", msg, flush=True); sys.exit(1)


import torch
from aide_atmos import core as C
from aide_atmos.met_out import MetWriter, hours_since_archive_t0, cesm_grid
torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
dev = "cuda"
b = C.Bundle(a.bundle, dev)
net = b.build_net(); phys = C.Physics(b)
mean, std, ips, iu, iT = b.mean, b.std, b.ips, b.iu, b.iT
WG = C.gauss_weights(dev)
t0 = np.datetime64(a.t0, "h"); hour0 = hours_since_archive_t0(t0)
if a.hour0 is not None and a.hour0 != hour0:
    fail(f"t0 {t0} is archive hour {hour0}, not --hour0 {a.hour0}")
FIX = None
if a.fixer:
    from aide_atmos.fixer import DycoreTerms, Fixer
    terms = DycoreTerms(os.path.join(a.bundle, "fixer_dycore.npz"), os.path.join(a.bundle, "fixer_dycore.json"), b.force_npz)
    FIX = Fixer(dev, b.FN, mean, std, np.load(b.coeff), terms, latsm=4, levsm=2)
NF = a.nframe
prov = (f"src/atmos/atm_live.py: P2@200 LIVE, HL (hlr_p), massfix {a.massfix}, u-fixer {a.fixer} (latsm 4, levsm 2), seed {a.seed}, "
        f"IC {a.t0}, q-mode {a.q_mode} x {a.q_scale}; bundle {os.path.abspath(a.bundle)}")
W = MetWriter(XD, NF, b, t0, hour0, prov, extra_meta=dict(live=True, seed=a.seed, q_mode=a.q_mode, q_scale=a.q_scale,
                                                        torch=torch.__version__))

# ---- Q: band f09 (K/s) -> emulator (1, 95, 96, 192) K per 6 h ----
zc = np.load(b.coeff)
lat_e, lon_e = zc["lat"].astype(np.float64), zc["lon"].astype(np.float64)
plev_e = zc["hyam_pa"] / 100 + zc["hybm"] * 1000                      # hPa at ps 1000 hPa (emulator level order)
lat_c, lon_c, klevs, plev_c, krad, hy = cesm_grid(os.path.join(a.bundle, "cesm_grid.npz"))
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
# vertical: pressure-overlap (conservative) remap, layer edges midway between level pressures (ends extrapolated half a
# layer), emulator levels at their reference pressures (ps 1013.25 hPa): sum_e Q_e dp_e = sum_k Q_k dp_k exactly. (The
# 2026-11-16 server interpolated linearly in log p, which delivered 96.7 % of the energy of a day-10 Q field.)
def layer_edges(pc):   # pc ascending (Pa) -> edges (n+1)
    return np.concatenate([[pc[0] - (pc[1] - pc[0]) / 2], (pc[1:] + pc[:-1]) / 2, [pc[-1] + (pc[-1] - pc[-2]) / 2]])


pref_e = plev_e * 1.01325 * 100.0                                   # Pa, emulator level order
oe = np.argsort(pref_e); ee = layer_edges(pref_e[oe])
eb = layer_edges(np.asarray(plev_c, np.float64))                    # band edges (Pa), band levels ascending in p
VZ = np.zeros((95, nb))
for j, e in enumerate(oe):
    ov = np.maximum(0.0, np.minimum(ee[j + 1], eb[1:]) - np.maximum(ee[j], eb[:-1]))
    VZ[e] = ov / (ee[j + 1] - ee[j])
tD = dict(device=dev, dtype=torch.float64)
WYt, WXt, VZt = (torch.as_tensor(m, **tD) for m in (WY, WX, VZ))
np.savez(os.path.join(XD, "q_remap.npz"), WY=WY, WX=WX, VZ=VZ)


def q_to_emu(q_band_ks):
    q = torch.as_tensor(np.asarray(q_band_ks), **tD)
    qh = torch.einsum('yi,kij,xj->kyx', WYt, q, WXt)
    return (torch.einsum('ek,kyx->eyx', VZt, qh) * 21600.0).float()[None]


def wait_q(j):
    fn = os.path.join(XD, "q", f"q_{j:05d}.npy"); tw0 = time.time()
    while not os.path.exists(fn):
        if os.path.exists(os.path.join(XD, "FAILED")):
            print("coupled side FAILED; exiting", flush=True); sys.exit(1)
        if time.time() - tw0 > a.timeout:
            fail(f"no Q for frame {j} after {a.timeout:.0f}s")
        time.sleep(0.002)
    return np.load(fn)


def publish(i):
    tmp = os.path.join(XD, ".progress.tmp"); open(tmp, "w").write(str(i + 1)); os.replace(tmp, os.path.join(XD, "progress"))


ZM = {k: np.zeros((NF, 95, 96), np.float32) for k in ("u", "T", "q")}


def put(i, ph):   # ph (1, 381, 96, 192) physical
    W.put(i, ph[0]); publish(i)
    ZM["u"][i] = ph[0, iu].mean(-1).cpu().numpy(); ZM["T"][i] = ph[0, iT].mean(-1).cpu().numpy()


ic = b.ic(t0)
prev, cur = ic[:, :381], ic[:, 381:762]
NG = C.NoiseGen(dev, seed=1000 + a.seed)
tn = tq = tw = 0.0; tt = time.time(); GM0 = None
with torch.no_grad():
    for t in range(NF - 1):
        tc = t0 + np.timedelta64(6 * t, "h"); tp = tc - np.timedelta64(6, "h")
        x = b.make_x(prev, cur, tp, tc)
        if t == 0:
            put(0, x[:, 381:762] * std + mean)
            tmp = os.path.join(XD, ".meta.tmp"); json.dump(W.meta, open(tmp, "w"), indent=1); os.replace(tmp, os.path.join(XD, "meta.json"))
            print(f"frame 0 written; {NF} frames; band {plev_c[0]/100:.2f}..{plev_c[-1]/100:.2f} hPa; q-mode {a.q_mode} x {a.q_scale}; "
                  f"Q onto emulator levels {int(np.where(VZ.any(1))[0].min())}..{int(np.where(VZ.any(1))[0].max())}", flush=True)
        a1 = time.time()
        zz = NG.sample(1)
        out = phys(net(C.net_input(x, zz)), x)
        if a.massfix:
            ph_ = out[:, ips] * std[0, ips] + mean[0, ips]; gm = (WG.view(1, -1, 1) * ph_).sum(-2).mean(-1)
            if t == 0:
                GM0 = ((WG.view(1, -1, 1) * (x[:, 381 + ips] * std[0, ips] + mean[0, ips])).sum(-2).mean(-1))
            out = out.clone(); out[:, ips] = (ph_ * (GM0 / gm).view(-1, 1, 1) - mean[0, ips]) / std[0, ips]
        mx = (out * std + mean)[:, iu[16:94]].abs().amax()
        if not bool(torch.isfinite(mx)) or float(mx) > 300:
            fail(f"emulator DIED at step {t} (max|u| h16-93 {float(mx):.1f})")
        if FIX is not None:
            out = FIX.apply(x[:, 381:762], out, tc, tc + np.timedelta64(6, "h"))
        torch.cuda.synchronize(); tn += time.time() - a1
        # heating for step t -> t+1 = Q diagnosed by the coupled run at frame t (frame 0 has none)
        a2 = time.time()
        if a.q_mode == "live" and t >= 1:
            QS = q_to_emu(wait_q(t)) * a.q_scale
            out = out.clone(); out[:, iT] += QS / std[:, iT]
            ZM["q"][t + 1] = QS[0].mean(-1).cpu().numpy()
        tw += time.time() - a2
        a3 = time.time()
        put(t + 1, out * std + mean); tq += time.time() - a3
        prev, cur = x[:, 381:762].clone(), out
        if t % 40 == 39:
            ctypes.CDLL("libc.so.6").malloc_trim(0)
        if t % 40 == 39 or t == NF - 2:
            np.savez(os.path.join(XD, ".atm_zm.tmp.npz"), plev=plev_e, lat=lat_e, **ZM, nwritten=t + 2)
            os.replace(os.path.join(XD, ".atm_zm.tmp.npz"), os.path.join(XD, "atm_zm.npz"))
            n = t + 1
            print(f"step {n}/{NF-1} day {n/4:.1f}: net+phys+fixer {tn/n*1e3:.0f} ms/step, regrid+write {tq/n*1e3:.0f} ms, "
                  f"waiting for Q {tw/n*1e3:.0f} ms; max|u| h16-93 {float(mx):.1f}; wall {time.time()-tt:.0f}s", flush=True)
W.close(NF)
print("atm_live done", flush=True)
