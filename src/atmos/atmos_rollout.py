#!/usr/bin/env python3
"""Standalone free run of the JCM emulator P2@200 -> a MET_SRC=file directory for AIDE-SAI-link.

Reproduces the recipe of runs/emu_winds_P2s200_1997_1998 (g2 sai_run.py / sai_continue.py): nudge-off, HL QBO wave drag
(hlr_p.npz), global-mean ps fix (--massfix 1), option-1 u-fixer with FIX_LATSM=4 FIX_LEVSM=2 (--fixer 1), generator
noise seed 1000 + seed; everything from one bundle directory (make_bundle.py), no PARADIS trainer / corpus / jcm-strat.

  atmos_rollout.py --bundle B --t0 1997-01-01T06 --segments 1460:0,1464:1 --out MET_DIR [--gpu N]

--segments STEPS:SEED,... : consecutive free-run legs. Each leg after the first restarts the noise generator with its
seed and continues from the previous leg's last two states passed through float32 physical frames (what
sai_continue.py --init-frames did with the saved frames); its first frame replaces the previous leg's last frame
(concat_met.py). The default 1460:0,1464:1 is the 2-year AIDE met set: 2925 frames, 1997-01-01T06 .. 1999-01-02T06.
--days N is the one-leg shortcut (4N steps, seed --seed).
--fixer 0 / --massfix 0 switch those off (the reproduction needs both on).
"""
import argparse, os, sys, time, ctypes
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
ap = argparse.ArgumentParser()
ap.add_argument("--bundle", required=True); ap.add_argument("--t0", default="1997-01-01T06"); ap.add_argument("--out", required=True)
ap.add_argument("--segments", default=""); ap.add_argument("--days", type=int, default=0); ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--fixer", type=int, default=1); ap.add_argument("--massfix", type=int, default=1)
ap.add_argument("--ic-ulp", type=int, default=0, help="TEST ONLY: nudge u of the IC by 1 float32 ulp at one cell (chaos twin)")
ap.add_argument("--init-frames", default="", help="start from a saved emulator-grid frames dir (u/v/T/omega/ps_norm.npy, sai_run --save-frames), frames IDX-1, IDX; --t0 = the time of frame IDX")
ap.add_argument("--init-idx", type=int, default=1460)
ap.add_argument("--gpu", default=None); ap.add_argument("--dump-emu", default="", help="also save emulator-grid frames (npz per frame index list 'i,j,..' -> OUT/emu_<i>.npz)")
a = ap.parse_args()
if a.gpu is not None:
    os.environ["CUDA_VISIBLE_DEVICES"] = str(a.gpu)
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
import torch
from aide_atmos import core as C
from aide_atmos.met_out import MetWriter, hours_since_archive_t0
torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False

segs = [(int(s.split(":")[0]), int(s.split(":")[1])) for s in a.segments.split(",")] if a.segments else [(4 * a.days, a.seed)]
assert all(n > 0 for n, _ in segs), "nothing to run (--segments or --days)"
NF = 1 + sum(n for n, _ in segs)
dev = "cuda"
b = C.Bundle(a.bundle, dev)
net = b.build_net(); phys = C.Physics(b)
mean, std, ips, iu = b.mean, b.std, b.ips, b.iu
WG = C.gauss_weights(dev)
t0 = np.datetime64(a.t0, "h"); hour0 = hours_since_archive_t0(t0)
FIX = None
if a.fixer:
    from aide_atmos.fixer import DycoreTerms, Fixer
    terms = DycoreTerms(os.path.join(a.bundle, "fixer_dycore.npz"), os.path.join(a.bundle, "fixer_dycore.json"), b.force_npz)
    FIX = Fixer(dev, b.FN, mean, std, np.load(b.coeff), terms, latsm=4, levsm=2)
prov = (f"aide_atmos/atmos_rollout.py: P2@200 free run, HL (hlr_p), massfix {a.massfix}, u-fixer {a.fixer} (latsm 4, levsm 2), "
        f"legs {segs} (steps:seed), IC {a.t0}; bundle {os.path.abspath(a.bundle)}")
W = MetWriter(a.out, NF, b, t0, hour0, prov, extra_meta=dict(segments=[list(s) for s in segs], torch=torch.__version__))
dump = {int(i) for i in a.dump_emu.split(",")} if a.dump_emu else set()

def put(i, ph):
    W.put(i, ph[0])
    if i in dump:
        np.savez(os.path.join(a.out, f"emu_{i:05d}.npz"), **{k: ph[0, ii].cpu().numpy() for k, ii in
                 (("u", b.iu), ("v", b.iv), ("T", b.iT), ("omega", b.iw))}, ps_norm=ph[0, ips].cpu().numpy())

if a.init_frames:   # sai_continue.py --init-frames: physical float32 frames -> normalised state
    _F = {k: np.load(f"{a.init_frames}/{k}.npy", mmap_mode="r") for k in ("u", "v", "T", "omega")}; _Fps = np.load(f"{a.init_frames}/ps_norm.npy", mmap_mode="r")
    def _state(i):
        ph = torch.zeros((1, 381, 96, 192), device=dev, dtype=torch.float32)
        for k, ii in (("u", b.iu), ("v", b.iv), ("T", b.iT), ("omega", b.iw)): ph[0, ii] = torch.from_numpy(np.array(_F[k][i])).to(dev)
        ph[0, ips] = torch.from_numpy(np.array(_Fps[i])).to(dev)
        return ((ph - mean[:, :381]) / std[:, :381]).float()
    prev, cur = _state(a.init_idx - 1), _state(a.init_idx)
else:
    ic = b.ic(t0)
    prev, cur = ic[:, :381], ic[:, 381:762]
if a.ic_ulp:
    cur = cur.clone(); j_ = b.iu[40]; cur[0, j_, 50, 0] = torch.nextafter(cur[0, j_, 50, 0], torch.tensor(np.inf, device=dev))
frame = 0; tstep = 0; tt = time.time(); tn = 0.0
with torch.no_grad():
    for leg, (nsteps, seed) in enumerate(segs):
        NG = C.NoiseGen(dev, seed=1000 + seed)
        if leg > 0:   # sai_continue.py: state from float32 physical frames, renormalised
            prev = ((prev_ph - mean[:, :381]) / std[:, :381]).float(); cur = ((cur_ph - mean[:, :381]) / std[:, :381]).float()
            frame -= 1                                   # concat_met.py: this leg's frame 0 replaces the last frame
        GM0 = None
        for t in range(nsteps):
            tc = t0 + np.timedelta64(6 * tstep, "h"); tp = tc - np.timedelta64(6, "h")
            x = b.make_x(prev, cur, tp, tc)
            if t == 0:
                put(frame, x[:, 381:762] * std + mean)
            a1 = time.time()
            zz = NG.sample(1)
            out = phys(net(C.net_input(x, zz)), x)
            if a.massfix:
                ph_ = out[:, ips] * std[0, ips] + mean[0, ips]; gm = (WG.view(1, -1, 1) * ph_).sum(-2).mean(-1)
                if t == 0: GM0 = ((WG.view(1, -1, 1) * (x[:, 381 + ips] * std[0, ips] + mean[0, ips])).sum(-2).mean(-1))
                out = out.clone(); out[:, ips] = (ph_ * (GM0 / gm).view(-1, 1, 1) - mean[0, ips]) / std[0, ips]
            pm = out * std + mean
            mx = pm[:, iu[16:94]].abs().amax()
            if not bool(torch.isfinite(mx)) or float(mx) > 300:
                W.close(frame + 1); sys.exit(f"emulator DIED at leg {leg} step {t} (max|u| h16-93 {float(mx):.1f}); {frame + 1} frames written")
            if FIX is not None:
                out = FIX.apply(x[:, 381:762], out, tc, tc + np.timedelta64(6, "h"))
            torch.cuda.synchronize(); tn += time.time() - a1
            pm = out * std + mean
            frame += 1; tstep += 1
            put(frame, pm)
            prev, cur = x[:, 381:762].clone(), out
            if t == nsteps - 2: prev_ph = pm.clone()
            if t == nsteps - 1: cur_ph = pm.clone()
            if nsteps == 1 and t == 0: prev_ph = x[:, 381:762] * std + mean
            if tstep % 40 == 0:
                ctypes.CDLL("libc.so.6").malloc_trim(0)
            if tstep % 200 == 0:
                el = time.time() - tt
                print(f"step {tstep}/{NF - 1}  day {tstep / 4:.0f}  max|u| h16-93 {float(mx):.1f}  {el:.0f}s ({el / tstep:.3f} s/step, net+fix {tn / tstep:.3f})", flush=True)
W.close(frame + 1)
el = time.time() - tt
print(f"done: {frame + 1} frames -> {a.out}  {el:.0f}s  ({el / max(tstep, 1):.3f} s/step, net+phys+fixer {tn / max(tstep, 1):.3f} s/step)", flush=True)
