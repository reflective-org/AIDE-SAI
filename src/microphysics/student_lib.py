"""Deployment wrappers: student checkpoints (train_student.py) and SDPA-fused BinTF -> step6h(s_norm (B,82), c_norm (B,13,6)) -> (B,82) fp32, compiled."""
import json, os, sys, torch, torch.nn as nn, torch.nn.functional as F
sys.path.insert(0, os.environ.get('EMU_CODE', os.path.join(os.path.dirname(os.path.abspath(__file__)), 'tomas_emulator_code')))   # tomas-emulator model code (vendored), as tomas_emulator.py EMU_CODE
from model_v8 import ConvMLPStep
from frontier_bintf_infer import BinTF
import frontier_bintf_infer as FI
dev = 'cuda'

class SdpaBinTF(nn.Module):
    """same math as BinTF (norm_first TransformerEncoder), attention via F.scaled_dot_product_attention with fused qkv (profile_emu_fwd_20261109)"""
    def __init__(s, m):
        super().__init__(); s.m = m; s.H = 4
    def forward(s, s0, cond_seq):
        m = s.m; B = s0.shape[0]; N = FI.N_BINS; T = 12
        nk, mk = s0[:, :N], s0[:, N:2 * N]; gas = s0[:, 2 * N:]
        tok = torch.cat([torch.stack([nk, mk], -1), torch.stack([gas, torch.zeros_like(gas)], -1)], 1)
        glob = torch.cat([gas, cond_seq[:, 1:1 + T].flatten(1)], 1)
        x = m.tok_in(torch.cat([tok, glob.unsqueeze(1).expand(B, N + 2, -1)], -1)) + m.bin_emb
        for Lr in m.enc.layers:
            h = Lr.norm1(x); a = Lr.self_attn
            qkv = F.linear(h, a.in_proj_weight, a.in_proj_bias).view(B, N + 2, 3, s.H, -1).permute(2, 0, 3, 1, 4)
            o = F.scaled_dot_product_attention(qkv[0], qkv[1], qkv[2]).transpose(1, 2).reshape(B, N + 2, -1)
            x = x + F.linear(o, a.out_proj.weight, a.out_proj.bias)
            h = Lr.norm2(x); x = x + Lr.linear2(F.gelu(Lr.linear1(h)))
        d = m.head(m.norm(x)).reshape(B, N + 2, T, 2).permute(0, 2, 1, 3)
        base = torch.cat([nk, mk, gas], -1).unsqueeze(1)
        return (base + torch.cat([d[:, :, :N, 0], d[:, :, :N, 1], d[:, :, N:, 0]], -1))[:, -1]

class ConvAR(nn.Module):
    def __init__(s, net, steps=12): super().__init__(); s.net = net; s.steps = steps
    def forward(s, x, c):
        for j in range(s.steps): x = s.net(x, c[:, j + 1])
        return x
class Mlp6(nn.Module):
    def __init__(s, net): super().__init__(); s.net = net
    def forward(s, x, c): return s.net(x, torch.cat([c[:, 0, :3], c[:, :, 3]], 1))
class Last(nn.Module):
    def __init__(s, net): super().__init__(); s.net = net
    def forward(s, x, c): return s.net(x, c)[:, -1]

def _wrap(core, dtype, mode):
    core = core.to(dev).eval(); fwd = torch.compile(core, mode=mode) if mode else core
    @torch.no_grad()
    def step6h(s, c):
        with torch.autocast('cuda', dtype=dtype, enabled=(dtype != torch.float32)): out = fwd(s, c)
        return torch.nan_to_num(out.float(), nan=0.0, posinf=50.0, neginf=-50.0).clamp(-50, 50)
    return step6h

def load_sdpa(ckpt, d, nl, dtype=torch.float16, mode='reduce-overhead'):
    m = BinTF(d, nl).to(dev).eval(); m.load_state_dict(torch.load(ckpt, map_location=dev)['model']); return _wrap(SdpaBinTF(m), dtype, mode)

def load_student(path, dtype=torch.float16, mode=os.environ.get('EMU_COMPILE_MODE', 'reduce-overhead')):
    ck = torch.load(path, map_location=dev)
    # a training checkpoint carries its args; a published one (HF reflective-org/AIDE-Microphysics, {"model": state_dict})
    # does not, and takes the architecture from config.json beside it (scripts/get_assets.py writes it)
    a = ck['args'] if 'args' in ck else json.load(open(os.path.join(os.path.dirname(path), 'config.json'))); arch = a['arch']
    if arch == 'convar': net = ConvMLPStep(hidden=a['hidden'], depth=a['depth']); net.load_state_dict(ck['model']); core = ConvAR(net, a.get('steps', 12))
    elif arch == 'mlp6': net = ConvMLPStep(hidden=a['hidden'], depth=a['depth'], n_cond=16); net.load_state_dict(ck['model']); core = Mlp6(net)
    elif arch == 'bintf':
        net = BinTF(a['d'], a['nl']); net.load_state_dict(ck['model']); core = SdpaBinTF(net) if a['nl'] > 0 and False else Last(net)
    elif arch == 'packtf':
        import simple_models as SM
        net = SM.PackTF(a['d'], a['nl'], a['pack'], a.get('heads', 4), a.get('ffm', 2)); net.load_state_dict(ck['model']); core = Mlp6(net)
    return _wrap(core, dtype, mode)
