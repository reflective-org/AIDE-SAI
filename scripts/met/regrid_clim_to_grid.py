#!/usr/bin/env python3
"""Regrid the CESM boundary climatology (bc_clim.npz) onto a met source's grid.

Added 2026-11-18. With MET_SRC=jcm the model grid IS the JCM
grid (T63 Gaussian 96 x 192, JCM's 95 hybrid levels), and every field JCM does
not provide comes from the CESM climatology that BC_SRC=clim already serves on
the CESM f09 grid (scripts/met/make_bc_climatology.py: monthly 1996-2014 means
of the f.e21.FWHIST.f09 archive). This script puts that climatology on the
target grid and levels, in the same file layout, so bc_clim.ClimArchive serves
it unchanged and coupling.install_bc_source can check that the two agree.

  python3 scripts/met/regrid_clim_to_grid.py \
      --clim inputs/bc_clim/bc_clim.npz \
      --jcm-dir /data/JCM_stripped/jcm-strat/runs/p31/p31_AL7h70 --year 1996 \
      --out inputs/bc_clim/bc_clim_jcm.npz

Methods (the table is also written into the file, key `regrid_method`):
  horizontal, CONSERVATIVE (first-order area overlap, exact for two rectilinear
    lat-lon grids: separable overlap in sin(lat) x longitude; cell edges from
    fct_fast.lat_edges, i.e. half cells at the CESM pole rows and edges at the
    poles for the Gaussian rows) for every field whose area integral means
    something -- the mixing ratios and number concentrations (Q, O3, CH4, SO2,
    H2SO4, num_a1-3, so4_a1-3) and the surface fluxes (FLDS, FLNS, FSDS, FSNS);
  horizontal, BILINEAR (linear in latitude and periodic longitude) for the
    intensive state fields (OH, RELHUM, dgnumwet1-3, PS, TS);
  vertical, LINEAR IN ln(p) between the two grids' reference pressures
    (hyam*P0 + hybm*1e5 -- the convention both the model and the CESM archive
    use), constant beyond the end levels (JCM's top 0.01 hPa and bottom
    996 hPa both lie inside CESM's 6e-6..993 hPa except the last 3 hPa).
    Not conservative: the JCM band is ~2x finer than CESM's, and a first-order
    conservative remap onto finer levels is a staircase (pairs of identical
    levels) -- the climatology is an IC and a reservoir value, not a budget.
  NOT carried: T. The meteorology (U V OMEGA T) is the met source's; leaving the
    climatology's monthly-mean T out means it can never leak in by routing.
Both horizontal operators are convex combinations, so positivity is kept.
The script prints the global mean of each field on the source and the target
grid at three levels (conservative fields agree to roundoff horizontally).
"""
import argparse
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, '..', '..', 'src')
sys.path.insert(0, os.path.join(SRC, 'met'))
sys.path.insert(0, os.path.join(SRC, 'advection'))

import met_jcm                      # noqa: E402
from fct_fast import lat_edges      # noqa: E402  (pure numpy at import? see below)

CONSERVATIVE = ('Q', 'O3', 'CH4', 'SO2', 'H2SO4', 'num_a1', 'num_a2', 'num_a3',
                'so4_a1', 'so4_a2', 'so4_a3', 'FLDS', 'FLNS', 'FSDS', 'FSNS')
BILINEAR = ('OH', 'RELHUM', 'dgnumwet1', 'dgnumwet2', 'dgnumwet3', 'PS', 'TS')
EXCLUDED = ('T',)


def lon_edges(lon):
    d = lon[1] - lon[0]
    return np.concatenate([lon - 0.5 * d, [lon[-1] + 0.5 * d]])


def conservative_lat(src_lat, dst_lat):
    """W (ndst, nsrc): dst value = W @ src values, area (sin lat) overlap."""
    es = np.sin(np.deg2rad(lat_edges(src_lat)))
    ed = np.sin(np.deg2rad(lat_edges(dst_lat)))
    lo = np.maximum(ed[:-1, None], es[None, :-1])
    hi = np.minimum(ed[1:, None], es[None, 1:])
    W = np.clip(hi - lo, 0.0, None)
    return W / W.sum(axis=1, keepdims=True)


def conservative_lon(src_lon, dst_lon):
    es = lon_edges(src_lon); ed = lon_edges(dst_lon)
    W = np.zeros((dst_lon.size, src_lon.size))
    for shift in (-360.0, 0.0, 360.0):
        lo = np.maximum(ed[:-1, None], es[None, :-1] + shift)
        hi = np.minimum(ed[1:, None], es[None, 1:] + shift)
        W += np.clip(hi - lo, 0.0, None)
    return W / W.sum(axis=1, keepdims=True)


def bilinear_lat(src_lat, dst_lat):
    W = np.zeros((dst_lat.size, src_lat.size))
    j = np.clip(np.searchsorted(src_lat, dst_lat) - 1, 0, src_lat.size - 2)
    a = np.clip((dst_lat - src_lat[j]) / (src_lat[j + 1] - src_lat[j]), 0.0, 1.0)
    W[np.arange(dst_lat.size), j] = 1 - a
    W[np.arange(dst_lat.size), j + 1] += a
    return W


def bilinear_lon(src_lon, dst_lon):
    n = src_lon.size; d = src_lon[1] - src_lon[0]
    x = (dst_lon - src_lon[0]) / d
    i = np.floor(x).astype(int); a = x - i
    W = np.zeros((dst_lon.size, n))
    W[np.arange(dst_lon.size), i % n] = 1 - a
    W[np.arange(dst_lon.size), (i + 1) % n] += a
    return W


def vertical_lnp(p_src, p_dst):
    """V (ndst, nsrc): linear in ln p, constant beyond the end levels."""
    ls, ld = np.log(p_src), np.log(p_dst)
    V = np.zeros((p_dst.size, p_src.size))
    k = np.clip(np.searchsorted(ls, ld) - 1, 0, p_src.size - 2)
    a = np.clip((ld - ls[k]) / (ls[k + 1] - ls[k]), 0.0, 1.0)
    V[np.arange(p_dst.size), k] = 1 - a
    V[np.arange(p_dst.size), k + 1] += a
    return V


def gmean(f, lat):
    w = np.diff(np.sin(np.deg2rad(lat_edges(lat))))
    return float((f.mean(axis=-1) * w).sum(axis=-1).mean() / w.sum()) if f.ndim == 2 else \
        float(((f.mean(axis=-1) * w).sum(axis=-1) / w.sum()).mean())


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--clim', default=os.path.join(HERE, '..', '..', 'inputs', 'bc_clim', 'bc_clim.npz'))
    ap.add_argument('--jcm-dir', default=met_jcm.DEFAULT_DIR)
    ap.add_argument('--year', type=int, default=1996, help='a year whose JCM segment defines the grid')
    ap.add_argument('--out', required=True)
    a = ap.parse_args()
    t0 = time.time()
    z = np.load(a.clim, mmap_mode='r')
    tgt = met_jcm.JcmMet(a.jcm_dir, [a.year])
    slat, slon = np.asarray(z['lat']), np.asarray(z['lon'])
    dlat, dlon = tgt.lat, tgt.lon
    p_src = np.asarray(z['hyam']) * float(z['P0']) + np.asarray(z['hybm']) * 1e5
    p_dst = tgt.hyam * tgt.P0 + tgt.hybm * 1e5
    print(f'source {slat.size}x{slon.size} x {p_src.size} lev ({p_src[0]/100:.3g}..{p_src[-1]/100:.0f} hPa) '
          f'-> target {dlat.size}x{dlon.size} x {p_dst.size} lev ({p_dst[0]/100:.3g}..{p_dst[-1]/100:.0f} hPa)',
          flush=True)
    ops = {'conservative': (conservative_lat(slat, dlat), conservative_lon(slon, dlon)),
           'bilinear': (bilinear_lat(slat, dlat), bilinear_lon(slon, dlon))}
    V = vertical_lnp(p_src, p_dst)
    f3 = [str(v) for v in z['fields_3d']]; f2 = [str(v) for v in z['fields_2d']]
    out = {}
    method = {}
    for var in f3 + f2:
        if var in EXCLUDED:
            print(f'  {var}: excluded (meteorology comes from the met source)', flush=True)
            continue
        m = 'conservative' if var in CONSERVATIVE else 'bilinear' if var in BILINEAR else None
        if m is None:
            raise SystemExit(f'{var}: no regrid method assigned')
        Wy, Wx = ops[m]
        f = np.asarray(z[var], np.float64)                   # (12, [lev,] lat, lon)
        h = np.einsum('ja,...ab,ib->...ji', Wy, f, Wx, optimize=True)
        if var in f3:
            g = np.einsum('kq,mqji->mkji', V, h, optimize=True)
            ks = [int(np.argmin(abs(p_src - p))) for p in (5e3, 1e3, 1e5 * 0.5)]
            kd = [int(np.argmin(abs(p_dst - p))) for p in (5e3, 1e3, 1e5 * 0.5)]
            msg = ' '.join(f'{p_src[i]/100:.0f}hPa {gmean(f[:, i], slat):.4g}->{gmean(h[:, i], dlat):.4g}'
                           f'(lev {p_dst[k]/100:.0f}hPa {gmean(g[:, k], dlat):.4g})'
                           for i, k in zip(ks, kd))
        else:
            g = h
            msg = f'{gmean(f, slat):.5g} -> {gmean(g, dlat):.5g}'
        out[var] = g.astype(np.float32)
        method[var] = m + ('' if var in f2 else ' + linear-in-ln(p) vertical')
        print(f'  {var:10s} {m:12s} min {g.min():.3g}  global mean {msg}  [{time.time()-t0:.0f}s]',
              flush=True)
    f3o = [v for v in f3 if v in out]; f2o = [v for v in f2 if v in out]
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    tmp = a.out + '.tmp.npz'
    np.savez(tmp, **out,
             lat=dlat, lon=dlon, hyam=tgt.hyam, hybm=tgt.hybm, hyai=tgt.hyai, hybi=tgt.hybi,
             P0=np.float64(tgt.P0), lev=tgt.lev,
             month_centre_doy=np.asarray(z['month_centre_doy']),
             fields_3d=np.array(f3o), fields_2d=np.array(f2o),
             attrs=np.asarray(z['attrs']),
             provenance=np.array(str(z['provenance']) + f' || regridded to {tgt.source} grid '
                                 f'({dlat.size}x{dlon.size} Gaussian, {p_dst.size} levels) by '
                                 f'regrid_clim_to_grid.py on {time.strftime("%Y-%m-%d")} from '
                                 f'{os.path.basename(a.clim)}'),
             years=np.asarray(z['years']), calendar=np.asarray(z['calendar']),
             regrid_method=np.array([f'{k}: {v}' for k, v in method.items()]),
             regrid_source=np.array(os.path.abspath(a.clim)),
             grid_source=np.array(tgt.source))
    os.replace(tmp, a.out)
    print(f'wrote {a.out} ({os.path.getsize(a.out)/1e9:.2f} GB) in {time.time()-t0:.0f}s', flush=True)


if __name__ == '__main__':
    main()
