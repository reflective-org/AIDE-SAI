"""mic3 (EMU_COAG_PACK, default on with EMU_PHYS_EXACT): coag_percell with (1) THREE cells per CTA packed in 128 lanes (bins 0-39 | 40-79 |
80-119; lanes 120-127 inert) instead of one cell in 64 lanes, and (2) each lane's 40 frozen kernel entries kij[c, k, :] loaded once per
cell into registers instead of re-gathered from L1/L2 every substep. Per-lane arithmetic is unchanged: bit-identical to coag_percell
(verified np.array_equal on 15 dumps + coupled equiv.py; mic3/NOTES.md). Use num_warps=4. micro2 COPY of tomas_jax/fast/coagulation_pallas.py with a PER-CELL adaptive substep count (n_sub_c = clip(ceil(dt*max_bins(lambda_c)/c_max),1,cap),
dt_sub_c = dt/n_sub_c) instead of the batch max. Everything else unchanged. Original docstring follows.

Fused Pallas (Triton) coagulation substep loop — GPU fast path.

Drop-in alternative to fast/coagulation.coagulation_step with identical
signature and returns. The frozen per-cell kernel kij (nbins x nbins f64,
12.8 KB) is uploaded ONCE per outer step and every Euler substep (TFL
rates + Euler + positivity clamp + full 2-sweep MNFIX) runs inside a
single Pallas kernel per cell. The XLA path instead re-reads the
(C, nbins, nbins) kernel decomposition from HBM every substep via two
GEMMs plus transpose/reduce fusions — the dominant measured cost of the
fast model (docs/gpu_fast_optimization.md).

Architecture (persistent-CTA grid — each CTA processes one cell at a
time, cells pid, pid+P, ...; bin axis padded nbins -> BP = next pow2
>= nbins+1, a Triton shape requirement):

  * All per-bin state lives in (BP,) register vectors. There are NO
    (BP, BP) register tiles: a first fused-kernel attempt that held kij
    and one-hot masks as (64, 64) tiles was 5-25x SLOWER than XLA
    because ~20 tile-sized temporaries per substep spill to local
    memory (see the optimization log). Everything cross-bin is instead:
  * TFL matvecs: a FULLY UNROLLED loop over source bins; each iteration
    loads one kij column ((BP,) contiguous row of the transposed
    kernel, L1/L2-resident after the first substep) and 3 scalars, and
    updates 4 accumulators with masked FMAs. Unrolling lets all the
    loads issue independently — with a fori_loop the ~120 L2-latency
    scalar loads per substep serialize and dominate.
  * shift_right and the MNFIX deposit: a CTA-private GMEM scratch row
    + CTA barrier, using masked gather loads (shift) and f64 atomic_add
    scatter (deposit, lax.cond-skipped on the CTA-uniform "no lane
    shifted" fast path). CTA-private scratch means barriers fully order
    the traffic; it stays L1/L2-resident.
  * MNFIX shift-target bins: exact ceil/floor(log2) via IEEE-754
    exponent-field extraction (no libdevice log); the new average mass
    xnew is gathered from the true xk table so bin boundaries are
    bit-identical to the reference (the grid is NOT assumed to be exact
    powers of two — jnp.power gives 1-ulp deviations on some bins).

Numerical deviations from the XLA path (all reassociation-level,
<< 1e-10 relative; verified in tests/test_fast_coag_pallas.py):
  * matvec accumulation order (serial j vs cuBLAS GEMM order),
  * deposit summation order (atomic_add vs XLA scatter-add — the XLA
    GPU scatter is itself non-bitwise-deterministic),
  * exact integer log2 vs the reference's float ceil(log(r)/log(p)):
    these can pick a different MNFIX target bin only when log(r)/log(p)
    sits within ~1 ulp of an integer (measure-zero; requires p = 2
    mass-doubling grids, which is the only fast-model grid).

Padded lanes are kept inert: kij columns are zero there (masked loads),
their bin boundaries are clones of the top bin so MNFIX never shifts
them, and they are re-zeroed after each Euler update so the mass
promoted past the top bin (lane nbins receives bin nbins-1's TFL shift)
never survives — exactly reproducing the reference, where shift_right
drops that mass and only the analytic overflow term accounts for it.
"""
import functools

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import triton as pl_triton

from tomas_jax.fast.coagulation import _kernel_cell, _loss_frequency_cell
from tomas_jax.fast.config import SRTSO4

# Constants mirrored from the reference implementations (do not retune):
_NEPS_RATES = 1.0e-3   # physics/coagulation_rates.NEPS
_NEPS_MNFIX = 1.0e-5   # fast/mnfix._NEPS
_EPS = 1.0e-40         # fast/mnfix._EPS

_MANT_MASK = (1 << 52) - 1

# micro2 knobs, read at TRACE time (pbench clears run._segment_runner's cache between variants):
FLOOR = 1      # per-cell count is at least min(FLOOR, shared batch-max count): batch max <= FLOOR -> exactly the shared rule
CMAX = None    # Courant factor for the per-cell count; None = the caller's c_max
SKIP_EMPTY = False   # rows whose boxvol == PAD_BOXVOL (the adapter's padding rows, marked in _phys_prep) run 0 substeps.
# NOT inferred from Nk: real correction-sample cells can hold 0 < sum(Nk) < 1 m-3 (22-30 per batch, micro2/NOTES.md).
# Padding rows hold no particles or gas, so their loss frequency never sets the shared count; with FLOOR >= n_sub_cap every
# real cell runs the shared count -> bit-identical to production.
PAD_BOXVOL = 7.0e5   # real cells: coupling.BOXVOL = 1e6


def _next_pow2(n):
    p = 1
    while p < n:
        p *= 2
    return p


def _ceil_log2(r):
    """Exact ceil(log2(r)) for finite r >= 1, via the IEEE-754 exponent."""
    b = jax.lax.bitcast_convert_type(r, jnp.int64)
    e = ((b >> 52) & 0x7FF) - 1023
    frac_nonzero = (b & _MANT_MASK) != 0
    return (e + jnp.where(frac_nonzero, 1, 0)).astype(jnp.int32)


def _floor_log2(r):
    """Exact floor(log2(r)) for finite r >= 1."""
    b = jax.lax.bitcast_convert_type(r, jnp.int64)
    return (((b >> 52) & 0x7FF) - 1023).astype(jnp.int32)



NPACK = 3
BPP = 128


def _make_kernel(nbins, bp, n_sweeps, use_barrier, n_groups, n_prog):
    """cv3 packed kernel: program pid processes cell groups pid, pid+n_prog, ...; group g = cells NPACK*g .. NPACK*g+NPACK-1.
    Lane L: slot = L // nbins (cell in the group), kk = L % nbins (its bin); lanes >= NPACK*nbins are inert padding."""
    assert n_groups % n_prog == 0
    trips = n_groups // n_prog
    W = BPP

    def barrier():
        if use_barrier:
            pl_triton.debug_barrier()

    def kernel(nsub_ref, dtsub_ref, scal_ref, xklo_ref, xkhi_ref, xkgeo_ref, xkg_ref,
               nk_ref, ms_ref, mw_ref, kijT_ref,
               nk_out, ms_out, mw_out, ovf_out, s_ref):
        pid = pl.program_id(0).astype(jnp.int64)
        zeta = scal_ref[1]
        zeta3 = scal_ref[2]
        xk0 = scal_ref[4]
        xk_top = scal_ref[5]

        L = jax.lax.broadcasted_iota(jnp.int32, (W,), 0)
        valid = L < NPACK * nbins
        slot = jnp.minimum(L // nbins, NPACK - 1)          # inert lanes: clamp to the last cell (masked everywhere)
        kk = jnp.where(valid, L - slot * nbins, nbins + (L - NPACK * nbins))   # inert lanes: kk >= nbins (like the old padded lanes)
        base = (slot * nbins).astype(jnp.int64)             # lane offset of this lane's cell in a scratch row
        kk64 = kk.astype(jnp.int64)
        last = kk == nbins - 1

        xk_lo = xklo_ref[...]
        xk_hi = xkhi_ref[...]
        xk_geo = xkgeo_ref[...]

        def _row(row):
            return jnp.asarray(row, jnp.int64)

        def store_row(row, vec):
            pl.store(s_ref, (pid, _row(row), pl.ds(0, W)), vec)

        def load_row(row):
            return pl.load(s_ref, (pid, _row(row), pl.ds(0, W)))

        def shift_row(row):
            # out[k] = row[k-1] within the cell, 0 at the cell's bin 0
            Lm1 = jnp.maximum(L - 1, 0).astype(jnp.int64)
            return pl.load(s_ref, (pid, _row(row), Lm1), mask=valid & (kk > 0), other=0.0)

        def rates(c, kd, N, m, cols):
            empty = N < _NEPS_RATES
            Ns = jnp.where(empty, _NEPS_RATES, N)
            msafe = jnp.where(empty, _NEPS_RATES * xk_geo, m)

            xbar = msafe / Ns
            ratio = xbar / xk_lo
            factor = 2.0 * Ns / xk_lo
            p = xk_hi / xk_lo
            eff = factor * (p - ratio) / (p - 1.0)
            phi = factor * (ratio - 1.0) / (p - 1.0)
            low = xbar < xk_lo
            eff = jnp.where(low, factor, eff)
            phi = jnp.where(low, 0.0, phi)
            high = xbar > xk_hi
            phi = jnp.where(high, factor, phi)
            eff = jnp.where(high, 0.0, eff)

            store_row(0, msafe)
            store_row(1, xbar)
            store_row(2, Ns)
            barrier()

            z = jnp.zeros((W,), jnp.float64)
            k1m, k1mx, k1mx2, in_term = z, z, z, z
            for j in range(nbins):
                src = base + j
                col = cols[j]
                msj = pl.load(s_ref, (pid, _row(0), src))
                xbj = pl.load(s_ref, (pid, _row(1), src))
                nsj = pl.load(s_ref, (pid, _row(2), src))
                c1j = msj * xbj
                c2j = msj * (xbj * xbj)
                lo = jnp.where(j < kk, col, 0.0)
                up = jnp.where(j > kk, col, 0.0)
                k1m = k1m + lo * msj
                k1mx = k1mx + lo * c1j
                k1mx2 = k1mx2 + lo * c2j
                in_term = in_term + up * nsj

            two_xk = 2.0 * xk_lo
            dNdt_curr = (
                -kd * Ns ** 2
                - phi * k1m
                - zeta * (eff - phi) / two_xk * k1mx
                - Ns * in_term
            )
            dMdt_curr = (
                Ns * k1m
                - kd * Ns * msafe
                - msafe * in_term
                - phi * xk_hi * k1m
                - 0.5 * zeta * eff * k1mx
                + zeta3 * (phi - eff) / two_xk * k1mx2
            )
            growthN = (
                0.5 * kd * Ns ** 2
                + phi * k1m
                + zeta * (eff - phi) / two_xk * k1mx
            )
            A = (
                kd * Ns * msafe
                + 0.5 * zeta * eff * k1mx
                - zeta3 * (phi - eff) / two_xk * k1mx2
            )
            B = phi * k1m
            E = A + B * xk_hi
            store_row(3, growthN)
            store_row(4, E)
            barrier()
            dNdt = dNdt_curr + shift_row(3)
            dMdt = dMdt_curr + shift_row(4)
            dOvf = jnp.where(last, E, 0.0)
            return dNdt, dMdt, dOvf

        def sweep(N, m, w):
            dry = m
            avg = dry / (N + _EPS)
            avg = jnp.where((N == 0.0) & (dry <= 0.0), xk_geo, avg)
            fm = m / (dry + _EPS)
            fw = w / (dry + _EPS)

            needs_up = (avg > xk_hi) & (kk < nbins - 1)
            ratio_up = avg * 1.1 / (xk0 + _EPS)
            kk_up_raw = _ceil_log2(jnp.maximum(ratio_up, 1.0)) - 1
            kk_up = jnp.clip(kk_up_raw, kk + 1, nbins - 1)

            needs_down = (avg < xk_lo) & (kk > 0)
            ratio_dn = avg / (1.1 * xk0 + _EPS)
            kk_dn_raw = _floor_log2(jnp.maximum(ratio_dn, 1.0))
            kk_dn = jnp.clip(kk_dn_raw, 0, jnp.maximum(kk - 1, 0))

            shifted = (needs_up | needs_down) & valid
            kt = jnp.where(needs_up, kk_up, jnp.where(needs_down, kk_dn, kk))
            g_idx = jnp.where(needs_up, jnp.minimum(kk_up + 1, nbins), kk_dn)
            g_idx = jnp.where(valid, g_idx, 0)
            xg = pl.load(xkg_ref, (g_idx,))
            xnew = jnp.where(needs_up, xg / 1.1, xg * 1.1)

            nshift = jnp.clip(
                (dry - xk_geo * N) / (xnew - xk_geo + _EPS), 0.0, N
            )
            nshift = jnp.where(shifted, nshift, 0.0)
            mshift = xnew * nshift
            n_remain = N - nshift

            Nb = jnp.where(shifted, n_remain, N)
            mb = jnp.where(shifted, (xk_geo * n_remain) * fm, m)
            wb = jnp.where(shifted, (xk_geo * n_remain) * fw, w)

            def deposit():
                zv = jnp.zeros((W,), jnp.float64)
                barrier()
                store_row(5, zv)
                store_row(6, zv)
                store_row(7, zv)
                barrier()
                tgt = base + jnp.where(valid, kt, 0).astype(jnp.int64)
                amask = shifted if use_barrier else None
                pl.atomic_add(s_ref, (pid, _row(5), tgt), nshift, mask=amask)
                pl.atomic_add(s_ref, (pid, _row(6), tgt), mshift * fm, mask=amask)
                pl.atomic_add(s_ref, (pid, _row(7), tgt), mshift * fw, mask=amask)
                barrier()
                return load_row(5), load_row(6), load_row(7)

            def no_deposit():
                zv = jnp.zeros((W,), jnp.float64)
                return zv, zv, zv

            n_shifting = jnp.sum(shifted.astype(jnp.int32))
            recv_N, recv_m, recv_w = jax.lax.cond(
                n_shifting > 0, deposit, no_deposit
            )
            return Nb + recv_N, mb + recv_m, wb + recv_w

        def mnfix_(N, m, w):
            N = jnp.maximum(N, 0.0)
            m = jnp.maximum(m, 0.0)
            w = jnp.maximum(w, 0.0)
            empty = N < _NEPS_MNFIX
            N = jnp.where(empty, _NEPS_MNFIX, N)
            m = jnp.where(empty, _NEPS_MNFIX * xk_geo, m)
            w = jnp.where(empty, 0.0, w)

            dry = m
            avg = dry / (N + _EPS)
            too_high = avg > xk_top
            scale = (N * xk_top / 1.2) / (dry + _EPS)
            m = jnp.where(too_high, m * scale, m)
            w = jnp.where(too_high, w * scale, w)

            dry = m
            too_low = dry / (N + _EPS) < xk0
            N = jnp.where(too_low, dry / (xk0 * 1.2), N)

            for _ in range(n_sweeps):
                N, m, w = sweep(N, m, w)
            return N, m, w

        def substep_factory(c, kd, dt_sub, nsub_l, cols):
            def body(it, carry):
                N0, m0, w0, ovf0 = carry
                dNdt, dMdt, dOvf = rates(c, kd, N0, m0, cols)
                N = jnp.maximum(N0 + dt_sub * dNdt, 0.0)
                m = jnp.maximum(m0 + dt_sub * dMdt, 0.0)
                w = jnp.maximum(w0, 0.0)
                ovf = ovf0 + dt_sub * dOvf
                N = jnp.where(valid, N, 0.0)
                m = jnp.where(valid, m, 0.0)
                w = jnp.where(valid, w, 0.0)
                N, m, w = mnfix_(N, m, w)
                act = it < nsub_l                      # cells of the group with fewer substeps keep their state
                return (jnp.where(act, N, N0), jnp.where(act, m, m0), jnp.where(act, w, w0), jnp.where(act, ovf, ovf0))
            return body

        def group_body(t, _):
            g = pid + t.astype(jnp.int64) * n_prog
            c = g * NPACK + slot.astype(jnp.int64)        # per-lane cell index
            kd = pl.load(kijT_ref, (c, kk64, kk64), mask=valid, other=0.0)
            N0 = pl.load(nk_ref, (c, kk64), mask=valid, other=0.0)
            m0 = pl.load(ms_ref, (c, kk64), mask=valid, other=0.0)
            w0 = pl.load(mw_ref, (c, kk64), mask=valid, other=0.0)
            z = jnp.zeros((W,), jnp.float64)
            nsub_l = pl.load(nsub_ref, (c,))
            dt_sub = pl.load(dtsub_ref, (c,))
            n_it = jnp.max(nsub_l)
            # cv5: this lane's 40 frozen kernel entries kij[c, kk, :] live in registers for the whole substep loop
            cols = [pl.load(kijT_ref, (c, jnp.asarray(j, jnp.int64), kk64), mask=valid, other=0.0) for j in range(nbins)]
            Nf, mf, wf, ovf_acc = jax.lax.fori_loop(
                0, n_it, substep_factory(c, kd, dt_sub, nsub_l, cols), (N0, m0, w0, z)
            )
            pl.store(nk_out, (c, kk64), Nf, mask=valid)
            pl.store(ms_out, (c, kk64), mf, mask=valid)
            pl.store(mw_out, (c, kk64), wf, mask=valid)
            pl.store(ovf_out, (c,), ovf_acc, mask=valid & last)
            return _

        jax.lax.fori_loop(0, trips, group_body, jnp.zeros((), jnp.int32))

    return kernel


@functools.lru_cache(maxsize=None)
def _get_pallas_fn(nbins, bp, n_sweeps, num_warps, interpret, n_groups, n_prog):
    kernel = _make_kernel(nbins, bp, n_sweeps, use_barrier=not interpret, n_groups=n_groups, n_prog=n_prog)
    kwargs = {}
    if not interpret and num_warps is not None:
        kwargs["compiler_params"] = pl_triton.CompilerParams(num_warps=num_warps)
    n_cells = n_groups * NPACK

    def call(nsub, dtsub, scal, xklo, xkhi, xkgeo, xkg, Nk, ms, mw, kijT):
        out_shape = [
            jax.ShapeDtypeStruct((n_cells, nbins), jnp.float64),
            jax.ShapeDtypeStruct((n_cells, nbins), jnp.float64),
            jax.ShapeDtypeStruct((n_cells, nbins), jnp.float64),
            jax.ShapeDtypeStruct((n_cells,), jnp.float64),
            jax.ShapeDtypeStruct((n_prog, 8, BPP), jnp.float64),
        ]
        return pl.pallas_call(kernel, grid=(n_prog,), out_shape=out_shape, interpret=interpret, **kwargs)(
            nsub, dtsub, scal, xklo, xkhi, xkgeo, xkg, Nk, ms, mw, kijT)

    return call


def _padded_grid_arrays(xk, nbins, bp):
    """Per-lane bin-boundary vectors, padded lanes cloned from the top bin
    (mnfix-inert), plus the gather table (true xk, finite extension)."""
    xk_geo = jnp.sqrt(xk[:-1] * xk[1:])  # same expression as the references
    npad = bp - nbins
    xk_lo = jnp.concatenate([xk[:-1], jnp.full((npad,), xk[nbins - 1])])
    xk_hi = jnp.concatenate([xk[1:], jnp.full((npad,), xk[nbins])])
    xk_geo_p = jnp.concatenate([xk_geo, jnp.full((npad,), xk_geo[-1])])
    p = xk[1] / xk[0]
    ext = xk[-1] * p ** jnp.arange(1, bp - nbins, dtype=jnp.float64)
    xk_g = jnp.concatenate([xk, ext])  # (bp,) — indices 0..nbins are true xk
    return xk_lo, xk_hi, xk_geo_p, xk_g


def coagulation_step_pallas(
    Nk, Mk, xk, temp, pres, boxvol, dt, c_max=0.1, n_sub_cap=64,
    num_warps=2, n_persistent=2640, interpret=False, _floor=None, _skip_empty=None,
):
    """Drop-in replacement for fast/coagulation.coagulation_step.

    Same args and returns: (Nk, Mk, overflow, cap_hit, n_sub). Extra
    static knobs: num_warps (Triton), n_persistent (persistent-CTA grid
    size; ~20 per SM keeps the concurrent kij slices L2-resident),
    interpret (run the kernel as JAX ops — CPU-compatible, for testing).
    GPU-only unless interpret=True.
    """
    nbins = Nk.shape[-1]
    bp = _next_pow2(nbins + 1)
    assert NPACK * nbins <= BPP
    C = Nk.shape[0]
    n_groups0 = -(-C // NPACK)
    n_prog = min(n_persistent, n_groups0)
    n_groups = -(-n_groups0 // n_prog) * n_prog
    C_pad = n_groups * NPACK

    # Frozen kernel + shared adaptive substep count: identical to the
    # reference path (same functions, same XLA ops).
    kij = jax.vmap(_kernel_cell)(Nk, Mk, temp, pres, boxvol)
    lam = jax.vmap(_loss_frequency_cell, in_axes=(0, 0, 0, None))(
        Nk, Mk, kij, xk
    )
    n_raw = jnp.ceil(dt * jnp.max(lam, axis=-1) / (CMAX or c_max)).astype(jnp.int32)    # (C,) per cell
    n_max = jnp.clip(jnp.max(n_raw), 1, n_sub_cap)                  # what the shared (production) rule would run
    n_sub = jnp.clip(jnp.maximum(n_raw, jnp.minimum(FLOOR if _floor is None else _floor, n_max)), 1, n_sub_cap)   # floor never exceeds n_max
    if SKIP_EMPTY if _skip_empty is None else _skip_empty:
        n_sub = jnp.where(boxvol == PAD_BOXVOL, 0, n_sub)
    cap_hit = jnp.any(n_raw > n_sub_cap)
    dt_sub = dt / jnp.maximum(n_sub, 1)

    # The in-kernel matvec streams kij COLUMNS: transpose once per outer
    # step so each column is a contiguous row.
    kijT = jnp.swapaxes(kij, -1, -2)

    def pad(a):
        if C_pad == C:
            return a
        widths = [(0, C_pad - C)] + [(0, 0)] * (a.ndim - 1)
        return jnp.pad(a, widths)

    xk_lo, xk_hi, xk_geo, xk_g = _padded_grid_arrays(xk, nbins, bp)
    npad3 = BPP - NPACK * nbins
    tile = lambda v: jnp.concatenate([v[:nbins]] * NPACK + [jnp.full((npad3,), v[nbins])])   # v[nbins] = the top-bin clone
    xk_lo, xk_hi, xk_geo = tile(xk_lo), tile(xk_hi), tile(xk_geo)

    # Scalars, computed with the same XLA expressions as the references:
    p_ratio = xk[1] / xk[0]
    zeta = 0.5 * (1.0 + (p_ratio + 1.0) ** 2 / (4.0 * p_ratio))
    zeta3 = zeta ** 3
    log_p = jnp.log(xk[1] / xk[0])  # layout slot kept; kernel uses exact log2
    scal = jnp.stack([
        jnp.zeros(()), zeta, zeta3, log_p, xk[0], xk[nbins],
        jnp.zeros(()), jnp.zeros(()),
    ]).astype(jnp.float64)
    nsub_arr = jnp.pad(n_sub, (0, C_pad - C), constant_values=0)
    dtsub_arr = jnp.pad(dt_sub, (0, C_pad - C), constant_values=dt)

    call = _get_pallas_fn(nbins, bp, 2, num_warps, interpret, n_groups,
                          n_prog)
    Nk_f, ms_f, mw_f, ovf, _scratch = call(
        nsub_arr, dtsub_arr, scal, xk_lo, xk_hi, xk_geo, xk_g,
        pad(Nk), pad(Mk[..., 0]), pad(Mk[..., 1]), pad(kijT),
    )
    Nk_f, ms_f, mw_f, ovf = (
        Nk_f[:C], ms_f[:C], mw_f[:C], ovf[:C]
    )
    Mk_f = jnp.stack([ms_f, mw_f], axis=-1)
    overflow = jnp.zeros(Mk.shape[:-2] + (Mk.shape[-1],))
    overflow = overflow.at[..., SRTSO4].set(ovf)
    return Nk_f, Mk_f, overflow, cap_hit, jnp.max(n_sub)   # diag: the stiffest cell's count
