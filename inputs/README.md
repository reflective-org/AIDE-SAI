# inputs/

Static input data that the coupler reads, plus pointers to the datasets that
are far too large to live here. The paths below under `/data/...` are the ones
on the machine these results were produced on; a clone elsewhere points the
same variables at its own copies.

## `rad_data/palmer_williams_h2so4.dat` (tracked)

Refractive index of aqueous sulfuric acid: Palmer & Williams (1975), as
distributed in HITRAN Aerosols-2016, tabulated over wavelength at 25, 38, 50,
75, 84.5 and 95.6 wt% H2SO4. `src/radiation/radiation.py` reads it
as `RI_FILE` to build the Mie lookup tables that turn TOMAS bins into RRTMGP
optical properties. 33 KB, so it is tracked in git rather than fetched.

## `assets/ne_110m/ne_110m_coastline.*` (tracked)

Natural Earth 1:110m coastlines (shapefile: `.shp`, `.shx`, `.dbf`, `.prj`,
`.cpg`), used only for drawing coastlines on the map figures of
`scripts/analysis/summary_2yr/`. Made with Natural Earth
(https://www.naturalearthdata.com), which is in the public domain. 95 KB, so it
is tracked; the scripts draw maps without coastlines if it is missing.

## `cache/` (gitignored)

The radiation reference cache (`RAD_REF_CACHE`, default `inputs/cache/rad_ref`;
`src/radiation/rad_cache.py`): the reference-aerosol solve and clear-sky fields
per archive hour, ~24 MB per hour. Bit-identical on a hit, computed and stored
on a miss, so it can be deleted at any time; `RAD_REF_CACHE=0` turns it off.

## `bc_clim/` (gitignored)

The boundary climatologies the presets read as `CLIM_FILE`:
`bc_clim/bc_clim.npz` (CESM f09 grid, ~3 GB; `cesm_numerical`) and
`bc_clim/bc_clim_jcm.npz` (JCM T63 grid, ~1.4 GB; `jcm_numerical`). They are
GBs, so they are gitignored and built locally, once:

1. `scripts/met/make_bc_climatology.py` builds `bc_clim.npz` from the CESM
   archive (a one-time pass over its monthly stream plus an h1 sub-sample for
   H2SO4);
2. `scripts/met/regrid_clim_to_grid.py` regrids it onto the JCM grid as
   `bc_clim_jcm.npz` (~20 s).

The scripts' docstrings give the exact commands. The presets give the paths
relative to the repo root, and `run_prod.py` resolves them against it, so the
run finds them from any launch directory. What they contain is described
under "The boundary climatology" below.

## The CESM forcing archive (NOT here)

The hourly `h1` time series is what the `cesm_numerical` preset reads its winds and
temperature from (`MET_SRC=cesm`), and it is ~23 TB. It is never copied into
this repo. On this machine it lives at:

```
/data/cesm2.1.5_output/histSST/
  f.e21.FWHIST.f09_f09_mg17.atmos-scale_fixedSST_1996-2014.001/archive/atm/proc/tseries
```

`src/coupling.py` expects CESM's own tseries convention, one file per variable:

```
$CESM_DIR/hour_1/$CESM_PREFIX.h1.<VAR>$CESM_SUF
```

Point a clone elsewhere at its own archive with three environment variables
(defaults are the FWHIST run above):

| variable | default |
|---|---|
| `CESM_DIR` | `/data/cesm2.1.5_output/histSST/f.e21.FWHIST.../archive/atm/proc/tseries` |
| `CESM_PREFIX` | `f.e21.FWHIST.f09_f09_mg17.atmos-scale_fixedSST_1996-2014.001.cam` |
| `CESM_SUF` | `.1996010100-2014123100.nc` |

## The boundary climatology

Both presets run `BC_SRC=clim`: every non-wind boundary field (Q, O3, the
gases, the MAM4 aerosol reservoirs, the radiation column) comes from a monthly,
day-of-year-cyclic climatology in `CLIM_FILE`, on the **model** grid.

| preset | `CLIM_FILE` | built by |
|---|---|---|
| `cesm_numerical` | `inputs/bc_clim/bc_clim.npz` (~3 GB, f09) | `scripts/met/make_bc_climatology.py` from the CESM archive |
| `jcm_numerical` | `inputs/bc_clim/bc_clim_jcm.npz` (~1.4 GB, JCM grid) | `scripts/met/regrid_clim_to_grid.py` from the `cesm_numerical` one |

They live in `inputs/bc_clim/` (see below) and are not tracked. To read a copy
elsewhere instead, override `CLIM_FILE` with `--clim-file` or an exported
variable.
The launcher warns when an exported value overrides the preset's, which is
the point: the run is then not byte-for-byte the preset's model.

## The JCM meteorology (NOT here)

The `jcm_numerical` preset (`MET_SRC=jcm`) reads 6-hourly U, V, OMEGA and T from a
jcm-strat run directory, `JCM_DIR`, read-only. Its default (coupling.py's, which
the preset leaves in place) is the AL7h70 chain on this machine,
`/data/JCM_stripped/jcm-strat/runs/p31/p31_AL7h70`.

## RRTMGP spectral data (NOT here either)

The `rrtmgp-gas-{sw,lw}-g*.nc` k-distribution and cloud optics files ship inside
the jax-rrtmgp package and are resolved from it at import
(`RRTMGP_DATA = <rrtmgp package>/optics/rrtmgp_data`), i.e. from whichever
checkout fills the preset's jax-rrtmgp slot (`RRTMGP_PATH`; `models/jax-rrtmgp`
for both shipped presets). Nothing to install separately —
`git submodule update --init` brings them.
