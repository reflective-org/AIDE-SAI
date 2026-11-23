"""v8 emulator models. State = 82 (40 Nk + 40 Mk + 2 gases), cond = 6.

ConvMLPStep — AR one-step operator, MLP trunk + conv-over-bins residual branch
  (the seam-free design from the bin-conv lesson, built in from the start).
  forward(state_norm (B,82), cond_norm (B,6)) -> next state_norm (B,82).
  Identity at init (zero-init output heads).

AxialTraj — non-AR trajectory operator (bin+time axial attention), generalized
  from model_axial_traj.py to 2 gas tokens and variable n_out.
"""
import torch
import torch.nn as nn

N_BINS = 40
N_STATE = 82
N_COND = 6


class ConvMLPStep(nn.Module):
    def __init__(self, hidden=768, depth=3, conv_channels=64, conv_kernel=5,
                 conv_layers=2, pos_emb=8, n_cond=N_COND):
        super().__init__()
        self.n_cond = n_cond
        layers, d = [], N_STATE + n_cond
        for _ in range(depth):
            layers += [nn.Linear(d, hidden), nn.GELU()]
            d = hidden
        self.trunk = nn.Sequential(*layers)
        self.head = nn.Linear(hidden, N_STATE)
        nn.init.zeros_(self.head.weight); nn.init.zeros_(self.head.bias)
        # conv branch over the bin axis: local cross-bin structure for every bin
        self.pos = nn.Parameter(torch.randn(pos_emb, N_BINS) * 0.02)
        in_ch = 2 + n_cond + 2 + pos_emb           # Nk,Mk + cond + 2 gases + pos
        conv, c_prev, pad = [], in_ch, conv_kernel // 2
        for _ in range(conv_layers):
            conv += [nn.Conv1d(c_prev, conv_channels, conv_kernel, padding=pad),
                     nn.GELU()]
            c_prev = conv_channels
        self.conv = nn.Sequential(*conv)
        self.conv_out = nn.Conv1d(conv_channels, 2, conv_kernel, padding=pad)
        nn.init.zeros_(self.conv_out.weight); nn.init.zeros_(self.conv_out.bias)

    def forward(self, s, c):
        x = torch.cat([s, c], dim=-1)
        delta = self.head(self.trunk(x))                       # (B,82)
        nk, mk = s[:, :N_BINS], s[:, N_BINS:2 * N_BINS]
        gas = s[:, 2 * N_BINS:]                                # (B,2)
        z = torch.cat([torch.stack([nk, mk], 1),               # (B,2,40)
                       c[:, :, None].expand(-1, -1, N_BINS),
                       gas[:, :, None].expand(-1, -1, N_BINS),
                       self.pos[None].expand(s.shape[0], -1, -1)], dim=1)
        corr = self.conv_out(self.conv(z))                     # (B,2,40)
        zero_g = torch.zeros(s.shape[0], 2, device=s.device, dtype=s.dtype)
        return s + delta + torch.cat([corr[:, 0], corr[:, 1], zero_g], dim=1)


class _MoEFF(nn.Module):
    def __init__(self, d, ff, E):
        super().__init__()
        self.gate = nn.Linear(d, E)
        self.experts = nn.ModuleList(
            [nn.Sequential(nn.Linear(d, ff), nn.GELU(), nn.Linear(ff, d))
             for _ in range(E)])

    def forward(self, x):
        g = torch.softmax(self.gate(x), dim=-1)
        outs = torch.stack([e(x) for e in self.experts], dim=-2)
        return (g.unsqueeze(-1) * outs).sum(-2)


class _AxialBlock(nn.Module):
    def __init__(self, d, nhead, ff, dropout=0.0, ff_experts=1):
        super().__init__()
        self.bin_attn = nn.MultiheadAttention(d, nhead, dropout=dropout,
                                              batch_first=True)
        self.time_attn = nn.MultiheadAttention(d, nhead, dropout=dropout,
                                               batch_first=True)
        self.ln1 = nn.LayerNorm(d); self.ln2 = nn.LayerNorm(d)
        self.ln3 = nn.LayerNorm(d)
        self.ff = (_MoEFF(d, ff, ff_experts) if ff_experts > 1 else
                   nn.Sequential(nn.Linear(d, ff), nn.GELU(), nn.Linear(ff, d)))

    def forward(self, x):                          # (B,T,P,d)
        B, T, P, d = x.shape
        h = self.ln1(x).reshape(B * T, P, d)
        a, _ = self.bin_attn(h, h, h)
        x = x + a.reshape(B, T, P, d)
        h = self.ln2(x).permute(0, 2, 1, 3).reshape(B * P, T, d)
        a, _ = self.time_attn(h, h, h)
        x = x + a.reshape(B, P, T, d).permute(0, 2, 1, 3)
        return x + self.ff(self.ln3(x))


class AxialTraj(nn.Module):
    """state0 (B,82) + cond (B, n_out+1, 6) -> (B, n_out, 82). Gas tokens: 2."""

    def __init__(self, d_model=96, nhead=4, num_blocks=3, dim_ff=256,
                 n_out=12, ff_experts=1, n_cond=N_COND):
        super().__init__()
        self.n_out = n_out
        self.P = N_BINS + 2
        self.tok_in = nn.Linear(2, d_model)
        self.cond_in = nn.Linear(n_cond, d_model)
        self.bin_emb = nn.Parameter(torch.randn(self.P, d_model) * 0.02)
        self.time_emb = nn.Parameter(torch.randn(n_out, d_model) * 0.02)
        self.blocks = nn.ModuleList(
            [_AxialBlock(d_model, nhead, dim_ff, ff_experts=ff_experts)
             for _ in range(num_blocks)])
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, 2)
        nn.init.zeros_(self.head.weight); nn.init.zeros_(self.head.bias)

    def forward(self, s0, cond_seq):
        B = s0.shape[0]
        nk, mk = s0[:, :N_BINS], s0[:, N_BINS:2 * N_BINS]
        gas = s0[:, 2 * N_BINS:]                               # (B,2)
        tok = torch.cat([torch.stack([nk, mk], -1),            # (B,40,2)
                         torch.stack([gas, torch.zeros_like(gas)], -1)], 1)
        x = self.tok_in(tok)                                   # (B,42,d)
        x = x.unsqueeze(1).expand(B, self.n_out, self.P, -1).clone()
        x = x + self.bin_emb.view(1, 1, self.P, -1) \
              + self.time_emb.view(1, self.n_out, 1, -1)
        x = x + self.cond_in(cond_seq[:, 1:1 + self.n_out]).unsqueeze(2)
        for blk in self.blocks:
            x = blk(x)
        d = self.head(self.norm(x))                            # (B,T,42,2)
        base = torch.cat([nk, mk, gas], -1).unsqueeze(1)
        dNk = d[:, :, :N_BINS, 0]
        dMk = d[:, :, :N_BINS, 1]
        dG = d[:, :, N_BINS:, 0]                               # (B,T,2)
        return base + torch.cat([dNk, dMk, dG], dim=-1)
