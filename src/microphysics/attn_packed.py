"""mic3: packed block-diagonal attention for short sequences (the student PackTF: T=11 tokens, 4 heads x 64), SM_ATTN=tri.
G sequences (batch x head) share one (G*16)-row tile; each sequence sits in a 16-row slot (rows >= T masked) and the score tile
is masked block-diagonally, so one tl.dot pair does G attentions. Same math as the flash kernels (fp32 scores, max-subtracted
exp2, fp32 row sum, P rounded to fp16 for P@V, fp32 accumulate, divide by the row sum), but NOT bit-identical to the cuDNN SDPA
kernel production uses: ~2e-4 of the attention outputs differ by 1 fp16 ulp (the row-sum order differs; mic3/NOTES.md).
cuDNN pads every 11-token sequence to a 64x128 tile in its own CTA: 1.42 ms per (16384,4,11,64) call vs ~0.1 ms here."""
import torch, triton, triton.language as tl


@triton.jit
def _attn_packed(Q, K, V, O, sqb, sqh, sqt, sob, soh, sot, NSEQ, qk_scale,
                 H: tl.constexpr, T: tl.constexpr, D: tl.constexpr, SLOT: tl.constexpr, G: tl.constexpr):
    R: tl.constexpr = G * SLOT
    pid = tl.program_id(0)
    r = tl.arange(0, R); d = tl.arange(0, D)
    seq = pid * G + r // SLOT; t = r % SLOT
    ok = (t < T) & (seq < NSEQ)
    b = seq // H; h = seq % H
    off_in = b[:, None] * sqb + h[:, None] * sqh + t[:, None] * sqt + d[None, :]
    q = tl.load(Q + off_in, mask=ok[:, None], other=0.0)
    k = tl.load(K + off_in, mask=ok[:, None], other=0.0)
    v = tl.load(V + off_in, mask=ok[:, None], other=0.0)
    s = tl.dot(q, tl.trans(k))                                       # (R, R) fp32
    c = tl.arange(0, R)
    keep = ((r[:, None] // SLOT) == (c[None, :] // SLOT)) & ((c[None, :] % SLOT) < T)
    s = tl.where(keep, s, float('-inf'))
    m = tl.max(s, 1)
    p = tl.math.exp2((s - m[:, None]) * qk_scale)
    l = tl.sum(p, 1)
    acc = tl.dot(p.to(v.dtype), v)
    o = acc / l[:, None]
    off_out = b[:, None] * sob + h[:, None] * soh + t[:, None] * sot + d[None, :]
    tl.store(O + off_out, o.to(O.dtype.element_ty), mask=ok[:, None])


def attn_packed(q, k, v, G=4, num_warps=4):
    """q, k, v (B, H, T, D) views with identical strides (unit stride on D), T <= 16. Returns (B, H, T, D), a view of a
    contiguous (B, T, H, D) buffer (so .transpose(1, 2).reshape(B, T, H*D) is free)."""
    B, H, T, D = q.shape
    o = torch.empty((B, T, H, D), device=q.device, dtype=q.dtype).transpose(1, 2)
    nseq = B * H
    _attn_packed[(triton.cdiv(nseq, G),)](q, k, v, o, q.stride(0), q.stride(1), q.stride(2), o.stride(0), o.stride(1), o.stride(2),
                                         nseq, (D ** -0.5) * 1.4426950408889634, H=H, T=T, D=D, SLOT=16, G=G, num_warps=num_warps)
    return o
