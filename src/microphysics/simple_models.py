"""Simple student models for the 40-bin TOMAS 6 h operator. Input: s (B,82) normalised log-state, c16 (B,16) = [logP,T,RH, 13 logOH] normalised.
PackTF: bins packed `pack` per token (40/pack bin tokens + 1 gas token), pre-norm transformer (SDPA), residual output on the state.
MLPR: plain residual MLP (+ optional bin-conv branch = ConvMLPStep)."""
import os, torch, torch.nn as nn, torch.nn.functional as F
_ATTN = os.environ.get('SM_ATTN', 'sdpa')
NB = 40
class Blk(nn.Module):
    def __init__(s, d, h, ff):
        super().__init__(); s.h = h; s.n1 = nn.LayerNorm(d); s.n2 = nn.LayerNorm(d); s.qkv = nn.Linear(d, 3 * d); s.o = nn.Linear(d, d)
        s.f1 = nn.Linear(d, ff); s.f2 = nn.Linear(ff, d)
    def forward(s, x):
        B, T, D = x.shape; q = s.qkv(s.n1(x)).view(B, T, 3, s.h, D // s.h).permute(2, 0, 3, 1, 4)
        if _ATTN == 'manual':
            w = torch.softmax((q[0] @ q[1].transpose(-1, -2)) * ((D // s.h) ** -0.5), -1)
            a = (w @ q[2]).transpose(1, 2).reshape(B, T, D)
        elif _ATTN == 'manual32':   # micro2: logits + softmax in fp32 (as the fused SDPA kernels do), P rounded to the V dtype for P@V
            w = torch.softmax((q[0].float() @ q[1].float().transpose(-1, -2)) * ((D // s.h) ** -0.5), -1)
            a = (w.to(q.dtype) @ q[2]).transpose(1, 2).reshape(B, T, D)
        else:
            a = F.scaled_dot_product_attention(q[0], q[1], q[2]).transpose(1, 2).reshape(B, T, D)
        x = x + s.o(a); return x + s.f2(F.gelu(s.f1(s.n2(x))))
class PackTF(nn.Module):
    def __init__(s, d=256, nl=4, pack=4, heads=4, ff_mult=2, n_cond=16):
        super().__init__(); s.pack = pack; s.nt = NB // pack; s.n_cond = n_cond
        s.tin = nn.Linear(2 * pack + n_cond + 2, d); s.gin = nn.Linear(2 + n_cond + 2, d)   # per-token feats + global (gases, cond)
        s.pos = nn.Parameter(torch.randn(s.nt + 1, d) * 0.02)
        s.blocks = nn.ModuleList([Blk(d, heads, ff_mult * d) for _ in range(nl)]); s.norm = nn.LayerNorm(d)
        s.hb = nn.Linear(d, 2 * pack); s.hg = nn.Linear(d, 2)
        for l in (s.hb, s.hg): nn.init.zeros_(l.weight); nn.init.zeros_(l.bias)
    def forward(s, x, c16):
        B = x.shape[0]; p = s.pack; nk, mk, gas = x[:, :NB], x[:, NB:2 * NB], x[:, 2 * NB:]
        bt = torch.stack([nk.view(B, s.nt, p), mk.view(B, s.nt, p)], -1).reshape(B, s.nt, 2 * p)
        glob = torch.cat([gas, c16], -1)
        tok = torch.cat([bt, glob[:, None].expand(B, s.nt, -1)], -1)
        t = torch.cat([s.tin(tok), s.gin(torch.cat([gas, c16, gas], -1))[:, None]], 1) + s.pos
        for b in s.blocks: t = b(t)
        t = s.norm(t); db = s.hb(t[:, :s.nt]).view(B, s.nt, p, 2); dg = s.hg(t[:, s.nt])
        return x + torch.cat([db[..., 0].reshape(B, NB), db[..., 1].reshape(B, NB), dg], 1)
