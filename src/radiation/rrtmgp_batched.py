"""g-point-batched clear-sky LW / SW solves for jax-rrtmgp (2026-11-13 perf, rad agent).

jax-rrtmgp's two_stream.solve_lw / solve_sw run a jax.lax.fori_loop over the 128 (LW) / 112 (SW)
g-points, one g-point per iteration. Inside an iteration the vertical recurrences are unrolled over
~57 levels, i.e. hundreds of tiny kernels on 55k-element vectors: the GPU is launch/latency bound, not
bandwidth bound (measured: 1.27 s for LW on 55k columns but 0.25 s on 3.5k columns).

Here the g-point loop runs over BLOCKS of G g-points: the unchanged per-g-point step functions are
jax.vmap'ed over the G g-point indices (G x fewer, G x larger kernels), and the G fluxes are then
accumulated into the running sum ONE AT A TIME in g-point order. Every element is computed by the same
operations as in the library and summed in the same order (0 + f0 + f1 + ...), so the result is
bit-identical to two_stream.solve_lw / solve_sw (verified by md5 of the outputs; see rad/NOTES.md).

The step bodies below are line-for-line the library's step_fn (rrtmgp/rte/two_stream.py), minus the cloud
branches (clear sky: cloud paths are None -> the driver's _skip_clouds override reproduces the zero-cloud
combine bit-exactly). Keep in step with the submodule if it is bumped (check_batched in NOTES.md).
"""
import jax
import jax.numpy as jnp
import numpy as np
from rrtmgp.rte import monochromatic_two_stream
from rrtmgp.rte import two_stream as ts


# ---- dynamic-trip minor-absorber scan ----------------------------------------------------------
# gas_optics._compute_minor_optical_depth accumulates the minor-gas optical depth with a fixed-length
# lax.scan over the WIDEST band's interval range (5-11 steps), each step evaluating a full interpolation
# for every cell and then masking it out (`jnp.where(in_band, c, 0.0)`, i.e. adding exactly 0.0) when the
# g-point's band is narrower -- ~half of the steps are discarded work, and it is ~45 % of all GPU time.
# Since igpt is known per step, the loop can simply run only the band's own steps: identical sum
# (the skipped terms are +0.0), same order. Under vmap over a block of g-points the trip count becomes the
# block's maximum, still <= the library's. Implemented as a pass-through wrapper of jax.lax.scan that
# only engages on the one call flagged by the (wrapped) minor_scan_length; everything else is untouched.
from rrtmgp.optics import gas_optics as _go
_CTX = {}
_orig_msl = _go.minor_scan_length
_orig_scan = jax.lax.scan


def _msl(minor_bnd_start, minor_bnd_end, minor_absorber_intervals):
    if _CTX.get('on'):
        _CTX['pending'] = (np.asarray(minor_bnd_start), np.asarray(minor_bnd_end), int(minor_absorber_intervals))
    return _orig_msl(minor_bnd_start, minor_bnd_end, minor_absorber_intervals)


def _scan(f, init, xs=None, *args, **kw):
    pend = _CTX.pop('pending', None)
    if pend is None or not _CTX.get('on') or args or kw:
        return _orig_scan(f, init, xs, *args, **kw)
    starts, ends, nint = pend
    ibnd = _CTX['g2b'][_CTX['igpt']]
    s = jnp.asarray(starts)[ibnd]; e = jnp.asarray(ends)[ibnd]
    n = jnp.where(s >= 0, jnp.minimum(e - s + 1, nint - s), 0)
    n = jnp.clip(n, 0, xs.shape[0])
    tau = jax.lax.fori_loop(0, n, lambda j, c: f(c, xs[j])[0], init)
    return tau, None


_orig_cmod = _go.compute_minor_optical_depth


def _cmod(lookup, vmr_lib, molecules, temperature, p, igpt, vmr_fields=None):
    """compute_minor_optical_depth evaluates BOTH the lower- and upper-atmosphere minor-gas tables on every
    cell and then selects with jnp.where(p > p_ref_tropo, lower, upper). The pressure is a function of the
    level only (pressure-level model; verified monotone below), so the lower/upper cells are a static
    leading/trailing block of the vertical axis: evaluate each table on its own levels only and
    concatenate. Per-cell arithmetic is unchanged -> bit-identical, half the minor-gas work."""
    pcol = _CTX.get('pcol')
    if pcol is None or not _CTX.get('on'):
        return _orig_cmod(lookup, vmr_lib, molecules, temperature, p, igpt, vmr_fields)
    nz = pcol.shape[0]
    low = pcol > float(lookup.p_ref_tropo)
    k = int(low.sum())
    assert low[:k].all() and not low[k:].any(), 'pressure not monotone in z: cannot split lower/upper statically'
    if k in (0, nz):
        return _orig_cmod(lookup, vmr_lib, molecules, temperature, p, igpt, vmr_fields)
    isz = lambda x: hasattr(x, 'ndim') and x.ndim >= 1 and x.shape[-1] == nz
    cut = lambda sel: (lambda x: x[..., sel] if isz(x) else x)
    out = []
    for sel, is_lower in ((slice(0, k), True), (slice(k, nz), False)):
        c = cut(sel)
        args = (molecules, temperature, p)
        vf = None if vmr_fields is None else {g: c(v) for g, v in vmr_fields.items()}
        out.append(_go._compute_minor_optical_depth(
            lookup, vmr_lib, c(molecules), c(temperature), c(p), igpt, is_lower, vf))
    return jnp.concatenate(out, axis=-1)


def install_dynamic_minor_scan(split_tropo=True):
    _go.minor_scan_length = _msl
    jax.lax.scan = _scan
    if split_tropo:
        _go.compute_minor_optical_depth = _cmod


class dyn_ctx:
    """Context for one g-point step: tells the patched scan which igpt / band table is in effect."""
    def __init__(self, igpt, g2b, on=True, pcol=None):
        self.igpt, self.g2b, self.on, self.pcol = igpt, g2b, on, pcol

    def __enter__(self):
        _CTX.update(igpt=self.igpt, g2b=self.g2b, on=self.on, pcol=self.pcol)

    def __exit__(self, *a):
        _CTX.clear()


def _blocked_sum(step, n_gpt, G, init):
    """sum_{g} step(g) accumulated sequentially in g order, G g-points per loop iteration."""
    assert n_gpt % G == 0, (n_gpt, G)

    def body(ib, cum):
        ig = ib * G + jnp.arange(G)
        fl = jax.vmap(step)(ig)                       # dict of (G, ...)
        for j in range(G):                            # same order as the library's running sum
            cum = jax.tree.map(lambda f, c: f[j] + c, fl, cum)
        return cum

    return jax.lax.fori_loop(0, n_gpt // G, body, init)


def solve_lw(pressure, temperature, molecules, optics_lib, atmos_state, vmr_fields, sfc_temperature,
             aerosol_optics, G, dyn=False, pcol=None):
    vmr_fields = ts._reindex_vmr_fields(vmr_fields, optics_lib.gas_optics_lw)
    g_point_to_bnd = optics_lib.gas_optics_lw.g_point_to_bnd

    def step_fn(igpt):
        ibnd = g_point_to_bnd[igpt]
        aer_slice = {k: v[ibnd] for k, v in aerosol_optics.items()}
        with dyn_ctx(igpt, g_point_to_bnd, bool(dyn), pcol):
            props = ts._compute_local_properties_lw(
                pressure, temperature, molecules, igpt, optics_lib, vmr_fields, sfc_temperature,
                None, None, None, None, aerosol_optics_slice=aer_slice)
        sfc_src = props['sfc_src']
        toa_flux_down_lw = (atmos_state.toa_flux_lw / optics_lib.n_gpt_lw) * jnp.ones_like(sfc_src)
        sfc_emissivity_lw = atmos_state.sfc_emis * jnp.ones_like(sfc_src)
        return monochromatic_two_stream.lw_transport(
            props['t_diff'], props['r_diff'], props['src_up'], props['src_down'],
            toa_flux_down_lw, sfc_src, sfc_emissivity_lw, False)

    init = {k: jnp.zeros_like(temperature) for k in ('flux_up', 'flux_down', 'flux_net')}
    return _blocked_sum(step_fn, optics_lib.n_gpt_lw, G, init)


def solve_sw(pressure, temperature, molecules, optics_lib, atmos_state, vmr_fields, aerosol_optics, G, dyn=False, pcol=None):
    zenith = atmos_state.zenith
    vmr_fields = ts._reindex_vmr_fields(vmr_fields, optics_lib.gas_optics_sw)
    g_point_to_bnd = optics_lib.gas_optics_sw.g_point_to_bnd
    night = zenith >= 0.5 * jnp.pi
    safe_zenith = jnp.where(night, jnp.zeros_like(jnp.asarray(zenith)), zenith)

    def step_fn(igpt):
        with dyn_ctx(igpt, g_point_to_bnd, bool(dyn), pcol):
            sw_optical_props = optics_lib.compute_sw_optical_properties(
                pressure, temperature, molecules, igpt, vmr_fields, None, None, None, None)
        ibnd = g_point_to_bnd[igpt]
        aer_slice = {k: v[ibnd] for k, v in aerosol_optics.items()}
        sw_optical_props = optics_lib.combine_optical_properties(sw_optical_props, aer_slice)
        p2 = monochromatic_two_stream.sw_cell_properties(
            safe_zenith, sw_optical_props['optical_depth'], sw_optical_props['ssa'],
            sw_optical_props['asymmetry_factor'])
        sfc_albedo = atmos_state.sfc_alb * jnp.ones_like(temperature)[:, :, 0]
        solar_flux = atmos_state.irrad * optics_lib.solar_fraction_by_gpt[igpt]
        toa_flux = solar_flux * jnp.ones_like(temperature)[:, :, 0]
        src = monochromatic_two_stream.sw_cell_source(
            t_dir=p2['t_dir'], r_dir=p2['r_dir'], optical_depth=sw_optical_props['optical_depth'],
            toa_flux=toa_flux, sfc_albedo_direct=sfc_albedo, zenith=safe_zenith, use_scan=False)
        return monochromatic_two_stream.sw_transport(
            t_diff=p2['t_diff'], r_diff=p2['r_diff'], src_up=src['src_up'], src_down=src['src_down'],
            sfc_src=src['sfc_src'], sfc_albedo=sfc_albedo, flux_down_dir=src['flux_down_dir'],
            use_scan=False)

    init = {k: jnp.zeros_like(temperature) for k in ('flux_up', 'flux_down', 'flux_net')}
    fl = _blocked_sum(step_fn, optics_lib.n_gpt_sw, G, init)
    return {k: jnp.where(night, jnp.zeros_like(v), v) for k, v in fl.items()}
