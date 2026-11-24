# Boundary conditions from a climatology

Everything the model reads that is not the wind and temperature — the aerosol
initial condition and reservoirs, the gases, water vapour, the radiation column
— comes from the **boundary source**, `BC_SRC`. Both presets set
`BC_SRC=clim`: a monthly, day-of-year-cyclic climatology of the CESM archive,
one file, on the model grid. This note records why that is enough, how it was
built, and what each boundary does with it. It started as the design note for
CESM-free long runs; the design below is what was implemented.

## What the boundary source supplies

| dependence | consumer | fields |
|---|---|---|
| initial condition | `bin_mam4(t=0)`, `read_gases(t=0)` | num/so4_a{1,2,3}, dgnumwet, SO2, H2SO4 |
| open-BC slabs (per step) | top/bottom band levels | same as IC |
| polar-cap rows of `qfroz` (per step) | the vertical faces only (the caps themselves are stirred, below) | same as IC |
| water vapour | RH (`MET_RH=liq`: over liquid from Q, T, p), wet optics and settling | Q |
| radiation column (per call) | `radiation.py` | Q, O3, CH4 on every native level; FLDS/FLNS/FSDS/FSNS (surface T and albedo), PS, TS |
| radiation reference (per call) | anomaly mode | the MAM4 modes binned onto TOMAS |
| OH, RELHUM | only under `OH_SZA=0` / `MET_RH=cesm` | OH, RELHUM |

The meteorology — U, V, OMEGA, T — is **not** a boundary field: it comes from
`MET_SRC` (`cesm` = the hourly h1 archive, `jcm` = a jcm-strat run directory,
`file` = step-end frames, the seam for an emulated dycore). `bc_clim.ClimArchive`
refuses to serve U/V/OMEGA/T, so the climatology's monthly-mean T can never leak
into a run by routing.

## Why a climatology is enough

The N2O proof-of-concept was a *tracer relaxation* problem: everything the
tracer did came from the boundaries, so the boundaries had to carry the full
CESM signal. With the SO2 → H2SO4 → SO4 chain and settling in the model, the
SAI aerosol is **internally generated and internally removed**: source =
injection, sink = settling through the band bottom. The boundaries only need
to supply a plausible *background* state, not the signal. That downgrades the
BC problem from "assimilate CESM" to "carry a climatology", which is what makes
long runs on any meteorology possible.

## The file

`scripts/met/make_bc_climatology.py` makes one pass over the CESM archive
(1996–2014, the monthly h0 stream; H2SO4, which has no monthly output, from a
sub-sample of the hourly h1 series) and writes `bc_clim.npz`: 12 calendar
months on the 70 native levels of the f09 grid, ~2.6 GB float32, read lazily.
`src/met/bc_clim.py` serves it through the same `open_var()` interface the
archive has, linearly interpolated in day-of-year between month centres and
cyclic, so switching the boundary source is one rebinding in `coupling.main`
and no physics code changes. `BC_SRC=cesm` (the module default, `--preset none`)
reads the hourly archive instead.

For the `jcm_numerical` preset the model grid is the JCM grid (T63 Gaussian 96×192), so
`scripts/met/regrid_clim_to_grid.py` puts the same climatology on it:
conservative in the horizontal for mixing ratios, number and fluxes, bilinear
for intensive fields, linear in ln p in the vertical, T not carried.
`coupling.install_bc_source` checks that the climatology and the model grid
agree. The presets name the files relative to the checkout
(`CLIM_FILE=inputs/bc_clim/bc_clim.npz` and `inputs/bc_clim/bc_clim_jcm.npz`;
`run_prod.py` resolves a relative path knob in a preset against the repo root).
They are gitignored: build them with the two scripts into `inputs/bc_clim/`
(see [inputs/README.md](../inputs/README.md)), or pass `--clim-file`.

## Per-boundary physics

**Bottom (150 hPa on the CESM grid, `P_HI_HPA`).** Two roles, separable:
* *outflow* — settling exits through an open bottom face (`settling.py`),
  independent of any reservoir. Advective export into the pinned bottom slab is
  likewise absorbed by the reset.
* *inflow* — tropical upwelling carries (nearly aerosol-free, SO2-bearing)
  tropospheric air into the band. A climatology-pinned bottom slab represents
  this; `BC_BOT_AER` scales its aerosol (default 1.0). A true flux BC (inflow
  at climatological concentration where OMEGA < 0, free outflow where
  OMEGA > 0) remains a possible upgrade.

**Top (1 hPa).** Effectively aerosol-free (MAM4 so4 there is seven orders below
its 13 hPa value); nothing interesting crosses it except number in the
smallest bins.

**Poles (|lat| > 80°).** The cap treatment is *numerical* (the zonal Courant
number diverges as 1/cos(lat)), not physical. Pinning the caps to a background
climatology would delete the injected aerosol the Brewer–Dobson circulation
carries poleward, every step. The default `ADV_POLAR=zonal` instead stirs each
cap to one well-mixed cell per level, mass-conservingly, with no external data
([PROCESSES.md](./PROCESSES.md) §1.5); every transport scheme uses the same
treatment. A polar filter or a different grid would be the next steps if
structure inside 80–90° ever matters.

**OH.** Not read from the boundary source by default: `OH_SZA=1` replaces it
with a cos(SZA) parabola fitted to Hanisco et al. (2001), sampled through the
step, which captures the diurnal cycle a monthly mean cannot.

**Meteorology.** Two things to watch with any non-CESM source:
* the transport needs *mass-consistent* winds. Emulated winds will not
  satisfy discrete continuity; the flux-form vertical sweep (shared by `lr`,
  `ffsl` and `sl`) keeps tracer mass exact anyway, and the inconsistency shows
  up as the air mass drifting from 1 rather than as lost tracer.
* T drives the coagulation kernel, settling viscosity and the SO2 + OH rate —
  all smooth functions of T, so emulator-grade T is fine.

**Radiation.** Stays physical (RRTMGP + Mie). Its non-aerosol inputs come from
the same climatology, and under `BC_SRC=clim` anomaly mode's *reference
aerosol* is the climatological MAM4 background binned onto TOMAS — so the
anomaly is "SAI against an unperturbed climatological stratosphere" rather than
"against whatever CESM weather did".

## What stays open

- Once a dycore responds to the aerosol's radiative heating (two-way
  coupling, which no linking has today: all are one-way), the pole and bottom questions get re-asked inside its own
  dynamics; the climatology remains valid as the *chemical/background*
  boundary either way.
- A flux-form bottom BC (optional).
- The climatology has no interannual variability and no diurnal cycle; the
  surface fluxes enter only as an albedo ratio and an upward longwave.
