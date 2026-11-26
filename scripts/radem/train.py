#!/usr/bin/env python3
"""Train the SW or LW aerosol-increment MLP on label.py shards.

  python train.py --kind sw --train DIR --val DIR --out CKPT [--width 512 --depth 4 --epochs 30]
  python train.py --kind sw --train DIR --overfit            # rung 4: one batch -> loss ~0

Targets: heating (nzb, K/day) + one TOA term (W/m2). Standardisation: ONE scale for all
heating channels and one for the TOA term (so the MSE weights levels by their real
size, like the rms/rms metric); inputs per feature, from the TRAINING shards only.
Writes CKPT/<kind>.npz (x_mu x_sd y_mu y_sd W* b* n_layers) for emulator.py.
"""
import os, sys, glob, time, argparse, json
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'src', 'radiation'))
import radem as E

ap = argparse.ArgumentParser()
ap.add_argument('--kind', choices=('sw', 'lw', 'pl'), required=True)
ap.add_argument('--train', required=True); ap.add_argument('--val', default=None)
ap.add_argument('--out', default=None)
ap.add_argument('--width', type=int, default=512); ap.add_argument('--depth', type=int, default=4)
ap.add_argument('--epochs', type=int, default=30); ap.add_argument('--bs', type=int, default=4096)
ap.add_argument('--lr', type=float, default=1e-3); ap.add_argument('--wd', type=float, default=1e-5)
ap.add_argument('--seed', type=int, default=0)
ap.add_argument('--amp-norm', action='store_true', help='pl: dT shape + log amplitude in, P/amplitude out')
ap.add_argument('--feat2', action='store_true', help='pl amp-norm: also raw dT and dT^2/A features')
ap.add_argument('--init', default='', help='warm start from CKPT/<kind>.npz (same dims; keeps its input/output normalisation)')
ap.add_argument('--relweight', action='store_true', help='per-column loss weight 1/(ms(target heating)+eps): relative error')
ap.add_argument('--overfit', action='store_true'); ap.add_argument('--max-shards', type=int, default=0)
args = ap.parse_args()

import torch
torch.manual_seed(args.seed); np.random.seed(args.seed)
dev = 'cuda'


def load(d, kind, maxn=0):
    fs = sorted(glob.glob(os.path.join(d, 'g*.npz')))
    if maxn: fs = fs[:maxn]
    X, Y, A = [], [], []
    for f in fs:
        z = np.load(f)
        if kind == 'pl':
            x = E.features_pl(np, z['T'], z['q'], z['o3'], z['ch4'], z['ts'], z['alb'], z['dT'], z['gmu'], z['gsc'],
                              amp_norm=args.amp_norm, feat2=args.feat2)
            y = np.concatenate([z['P_sw'], z['P_lw'], z['dswup'][:, None], z['dolr'][:, None]], 1)
            if args.amp_norm:
                keep_ = np.abs(z['dT']).max(1) > 1e-3          # zero-dT columns carry no shape information
                x, y = x[keep_], y[keep_] / E.pl_amp(np, z['dT'][keep_])[:, None]
                X.append(x); Y.append(y.astype(np.float32)); A.append(np.full(len(x), 1.0)); continue
            X.append(x); Y.append(y.astype(np.float32)); A.append(np.abs(z['dT']).max(1) * 0.1)   # "aod" mask: |dT| >= 0.1 K
            continue
        if kind == 'sw':
            x = E.features(np, 'sw', z['T'], z['q'], z['o3'], None, None, z['alb'], z['mu'],
                           z['tau_sw'], z['ssa_sw'], z['g_sw'])
            y = np.concatenate([z['dsw'], z['dswup'][:, None]], 1)
        else:
            x = E.features(np, 'lw', z['T'], z['q'], z['o3'], z['ch4'], z['ts'], None, None,
                           z['tau_lw'], z['ssa_lw'], z['g_lw'])
            y = np.concatenate([z['dlw'], z['dolr'][:, None]], 1)
        X.append(x); Y.append(y.astype(np.float32)); A.append(z['aod'])
    return np.concatenate(X), np.concatenate(Y), np.concatenate(A), len(fs)


t0 = time.time()
Xtr, Ytr, Atr, nf = load(args.train, args.kind, args.max_shards)
NTOA = 2 if args.kind == 'pl' else 1
nzb = Ytr.shape[1] - NTOA                                               # heating channels (pl: SW + LW, 2 x 24)
x_mu = Xtr.mean(0); x_sd = Xtr.std(0); x_sd[x_sd < 1e-6] = 1.0       # constant features pass through
y_mu = np.zeros(Ytr.shape[1], np.float32)                               # increments are 0 at zero aerosol / zero dT
y_sd = np.full(Ytr.shape[1], np.sqrt((Ytr[:, :nzb] ** 2).mean()), np.float32)
for j in range(nzb, Ytr.shape[1]):
    y_sd[j] = np.sqrt((Ytr[:, j] ** 2).mean())
print(f'{args.kind}: {nf} shards, {Xtr.shape[0]} columns, NF {Xtr.shape[1]}, NT {Ytr.shape[1]}; '
      f'heating rms {y_sd[0]:.4g} K/d, toa rms {y_sd[nzb]:.4g} W/m2 ({time.time()-t0:.0f}s)', flush=True)
if args.val:
    Xva, Yva, Ava, nv = load(args.val, args.kind)
    print(f'val: {nv} shards, {Xva.shape[0]} columns', flush=True)

dims = [Xtr.shape[1]] + [args.width] * args.depth + [Ytr.shape[1]]
layers = []
for i in range(len(dims) - 1):
    layers.append(torch.nn.Linear(dims[i], dims[i + 1]))
    if i < len(dims) - 2: layers.append(torch.nn.SiLU())
net = torch.nn.Sequential(*layers).to(dev)
if args.init:
    z0 = np.load(os.path.join(args.init, f'{args.kind}.npz'))
    assert int(z0['n_layers']) == len(dims) - 1 and z0['x_mu'].shape == x_mu.shape, 'init ckpt dims differ'
    x_mu, x_sd, y_sd = z0['x_mu'], z0['x_sd'], z0['y_sd']        # keep the init's normalisation (no refit)
    with torch.no_grad():
        for i, m in enumerate([m for m in net if isinstance(m, torch.nn.Linear)]):
            m.weight.copy_(torch.as_tensor(z0[f'W{i}'].T)); m.bias.copy_(torch.as_tensor(z0[f'b{i}']))
    print(f'warm start from {args.init}', flush=True)
print(f'MLP {dims}: {sum(p.numel() for p in net.parameters())/1e6:.2f} M params', flush=True)

T = lambda a: torch.as_tensor(a, device=dev)
xm, xs, ys = T(x_mu), T(x_sd), T(y_sd)
Xg = T(Xtr); Yg = T(Ytr)
norm_x = lambda x: (x - xm) / xs

if args.overfit:
    idx = torch.arange(min(args.bs, Xg.shape[0]), device=dev)
    xb, yb = norm_x(Xg[idx]), Yg[idx] / ys
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    for it in range(3001):
        loss = ((net(xb) - yb) ** 2).mean()
        opt.zero_grad(); loss.backward(); opt.step()
        if it % 500 == 0:
            print(f'  overfit it {it}: loss {loss.item():.3e} (target var 1 -> 0)', flush=True)
    sys.exit(0)


def evaluate(X, Y, A):
    net.eval(); out = []
    with torch.no_grad():
        for i in range(0, X.shape[0], 65536):
            out.append((net(norm_x(T(X[i:i + 65536]))) * ys).cpu().numpy())
    net.train(); P = np.concatenate(out)
    m = A >= 0.01
    loc = np.sqrt(((P[m, :nzb] - Y[m, :nzb]) ** 2).mean() / (Y[m, :nzb] ** 2).mean())
    toa = np.sqrt(((P[m, nzb] - Y[m, nzb]) ** 2).mean() / (Y[m, nzb] ** 2).mean())
    bias = (P[:, nzb].sum() / Y[:, nzb].sum() - 1)
    return loc, toa, bias


if args.relweight:
    ms = ((Yg[:, :nzb] / ys[:nzb]) ** 2).mean(1)
    Wg = 1.0 / (ms + 1e-4 * ms.mean()); Wg = Wg / Wg.mean()
opt = torch.optim.AdamW(net.parameters(), lr=args.lr, weight_decay=args.wd)
nb = Xg.shape[0] // args.bs
sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, total_steps=args.epochs * nb, pct_start=0.05)
for ep in range(args.epochs):
    te = time.time(); perm = torch.randperm(Xg.shape[0], device=dev); tot = 0.0
    for i in range(nb):
        j = perm[i * args.bs:(i + 1) * args.bs]
        se = (net(norm_x(Xg[j])) - Yg[j] / ys) ** 2
        loss = (se * Wg[j][:, None]).mean() if args.relweight else se.mean()
        opt.zero_grad(set_to_none=True); loss.backward(); opt.step(); sched.step()
        tot += loss.item()
    msg = f'ep {ep+1}/{args.epochs} train loss {tot/nb:.3e}'
    if args.val and ((ep + 1) % 5 == 0 or ep == args.epochs - 1):
        l, t_, b_ = evaluate(Xva, Yva, Ava)
        msg += f' | val (aod>=0.01) heating {l*100:.2f} %  toa {t_*100:.2f} %  toa sum bias {b_*100:+.2f} %'
    print(msg + f'  ({time.time()-te:.0f}s)', flush=True)

if args.out:
    os.makedirs(args.out, exist_ok=True)
    P = dict(x_mu=x_mu, x_sd=x_sd, y_mu=y_mu, y_sd=y_sd, n_layers=np.int64(len(dims) - 1))
    if args.amp_norm:
        P['amp_norm'] = np.int64(1)
    if args.feat2:
        P['feat2'] = np.int64(1)
    lin = [m for m in net if isinstance(m, torch.nn.Linear)]
    for i, m in enumerate(lin):
        P[f'W{i}'] = m.weight.detach().cpu().numpy().T.astype(np.float32)
        P[f'b{i}'] = m.bias.detach().cpu().numpy().astype(np.float32)
    np.savez(os.path.join(args.out, f'{args.kind}.npz'), **P)
    json.dump(vars(args), open(os.path.join(args.out, f'{args.kind}_args.json'), 'w'), indent=1)
    print(f'saved {args.out}/{args.kind}.npz', flush=True)
