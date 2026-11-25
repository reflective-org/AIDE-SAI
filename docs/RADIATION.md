# Radiation: the RRTMGP driver, anomaly mode, reference cache and solar geometry

How the radiation slot is filled in this repository, what it computes, the
per-hour cache that halves its cost, and the solar-geometry scheme. Numbers
tagged MEASURED name the script or run that produced them; `runs/...` and
`outputs/...` paths are the measurement directories on the development machine
(gitignored, not in the repo).

The radiation slot has one engine here, `RAD_ENGINE=rrtmgp`. The learned column
emulator that was developed for the same slot (`RAD_ENGINE=emulator`, its
`RAD_REF` A/B harness, training scripts and sweeps) is not part of this repo
(yet); selecting it here exits with a message saying so. An emulator would fill
the slot behind the existing selector. The selector, the `PHYS_CFG` stamp and the clear-sky fields the
cache can hold are kept so the slot stays swappable and old checkpoints resume.

## What the stage does

`src/radiation/radiation.py`, called once per `RAD_EVERY` coupling steps (default
every step), on the end-of-step state (stage 5 in
[PROCESSES.md](./PROCESSES.md) §0).

1. **Optics.** Per-bin optical properties from Mie tables built at start-up
   (`bhmie` from tomas-jax, Palmer & Williams (1975) H2SO4 refractive indices
   from `inputs/rad_data/palmer_williams_h2so4.dat`, band-averaged onto the
   14 SW + 16 LW RRTMGP bands, 3 wavelength points per band). Under
   `WET_OPTICS=1` (default) the tables are built on the wet H2SO4/H2O droplet
   over a 10–80 wt% grid and interpolated per cell; dry sizing underestimates
   550 nm extinction by ~47 %.
2. **Solve.** A jax-rrtmgp two-stream solve, clear sky, on the full native
   column of the boundary source (every level with p ≥ 1.005 Pa, the RRTMGP
   table range), aerosol only on the coupled band levels. Gases: the boundary
   source's Q and O3, `CO2_PPM` (380) and `N2O_PPB` (319). Surface albedo from
   FSNS/FSDS, emissivity 0.98. `RAD_LAT_CHUNK` rows per call (96 in the chunk
   loop; the `RAD_FAST` path, on by default since 2026-11-23 where validated,
   uses 192 unless `RAD_LAT_CHUNK` is set, which sets both); bit-identical at
   any value.
3. **Heating.** Returned on the band levels in K/s, plus TOA/surface fluxes and
   the 550 nm AOD.

### `RAD_MODE`: what is integrated

- **`anomaly`** (default). Two calls per radiation step: the evolved TOMAS bins
  at `T_eff = T_met + dT_rad`, and the MAM4 reference aerosol (the boundary
  source's MAM4 fields binned onto the TOMAS grid) at `T_met`. Their difference
  is integrated into the prognostic `dT_rad`. This isolates the radiative effect
  of the aerosol's *evolution*: there is no dynamics to balance the full heating
  rate while the circulation is prescribed, so integrating it would drift the
  band toward radiative equilibrium. Under `AER_SRC=mam4` the reference is
  time-varying.
- **`full`**: the evolved call alone (drifts; only meaningful once a dynamical
  core balances it).
- **`flux`**: one solve per step at the meteorology's own temperature, `dT_rad`
  stays 0 and nothing feeds back; the forcing is a difference taken after the
  run against a paired control on the same meteorology.

### `RAD_T_FEEDBACK`: where `dT_rad` goes

`0` (default since 2026-11-17, and what all three presets run): the
microphysics, the settling, the RH, the OH number density and the aerosol
optics all see `T_met`, so the model is one-way throughout. `dT_rad` enters
**only** the band temperature of the perturbed call (`T_band_override`), which
RRTMGP uses for both the Planck emission and the temperature-dependent gas
absorption. Its longwave response (the Planck damping) still bounds it, `hr_kday`
and ARF include that response, and the heating rate of
every call is written out as `hr_kday` (3-D, probe and zonal frames; `hr_min/max/rms`
in the timeseries). `1` is the earlier closed loop: `T_met + dT_rad` everywhere.
Stamped into `PHYS_CFG`.

Two properties of the physics that matter when reading `dT_rad` (MEASURED with
`check_planck_jacobian.py`, not in this repo):

- RRTMGP's Planck sources at the cell faces come from a WENO5 reconstruction of
  the temperature profile, whose weights move with the local profile. The
  response to a *uniform* band warming is linear (−0.0195 K/day per K at
  52 hPa), but the response to a sharp single-level `dT_rad` profile is not:
  its slope changes by 2× between 0.1 and 1 K. Profiles with sharp vertical
  structure are integrated by a solver whose response to them is non-linear.
- The TOA OLR responds by +0.43 W m⁻² per K of uniform band warming — the same
  order as the −0.06 W m⁻² aerosol forcing a 30-day run reports — so the
  reported forcing already contains the Planck response of the warmed layers.

### The forcing diagnostic

`arf_toa` is an instantaneous TOA flux difference at the call's own solar time;
with 6-h steps it is sampled four times a day, so a single sample's global mean
depends on which longitudes are sunlit. `ARF_AVG_H=24` (default) reports the
mean of the last four samples — four uniformly spaced local times, a proper
diurnal mean — as `arf_toa_avg`; the raw value is always stored. The model is
not calibrated for absolute radiative forcing (MANIFEST.md).

## The reference cache

In anomaly mode half of the stage — the reference-aerosol solve and the MAM4
re-binning that feeds it — depends only on the archive hour, never on the model
state. Measured at 8.5 s per step before the cache (2 × 2.7 s of two-stream
solve on 55,296 columns × 55 native levels, plus ~3 s of re-binning;
`bench_radiation_split.py`, not in this repo). `src/radiation/rad_cache.py` computes
those fields once per archive hour, stores them in float64 and serves them
afterwards. A hit is the same array the loop would have computed, so the run is
bit-identical, and the stage drops to one solve.

Under `MET_SRC=jcm_online` the cache is **off** whatever `RAD_REF_CACHE` says:
the meteorology is integrated live, so the reference solve depends on the run,
not on an archive hour, and a shared cache would hand one run another run's
reference (`MANIFEST.md`, the `jcm_online` entry). `MET_SRC=file` passes its own
band T, which bypasses the cache per call.

`RAD_REF_CACHE` (default `inputs/cache/rad_ref`; `0` disables). One `.npz` per
absolute archive hour under `<root>/rad_ref_<key>/h<hour:06d>.npz`, `key.json`
beside them, gitignored. The key hashes everything that determines the fields:
the archive file, the band and radiation levels, the grid, the optics flags and
tables, the gases, the surface constants, the reference-aerosol source and
binning (`coupling.rad_ref_src()`), the solar geometry (only when it is not
`instant`, so older caches keep their directory) and the commit of the
jax-rrtmgp checkout **actually in use** (`RRTMGP_PATH`, set by the preset's
`jax-rrtmgp` slot) — a change to any of them gets a fresh directory, never a
stale field. Fields per hour:

| field | what | dtype | written by |
|---|---|---|---|
| `hr0`, `d0_*` | the reference-aerosol solve: heating on the band levels, TOA/surface fluxes, AOD | float64 | the loop on a miss, or the builder |
| `hr_clear`, `hr_clear_sw`, `dc_*` | the zero-aerosol solve at `T_met` (total + SW) | float32 | the builder (`--only clear|all`) |
| `hr_dT`, `hr_dT_sw`, `ddT_*` | response to a uniform +1 K on the band levels | float32 | same |

Only `hr0`/`d0_*` are read by a run here; the clear-sky fields were the
emulator's inputs and remain buildable. Entries can be partial; a reader asks
for the fields it needs, and a miss on those computes and stores them. Under
`MET_SRC=file` the reference call takes the file's band temperature and bypasses
the cache. Batch build (optional — a run fills the cache as it goes):

```
CUDA_VISIBLE_DEVICES=N P_LO_HPA=1 P_HI_HPA=150 INIT_BIN=so4 \
  python3 scripts/radiation/build_ref_cache.py --hours 0:721:6 [--only ref|clear|all]
```

Run the builder under the same environment as the run (the band levels and
`INIT_BIN` are part of the key — the first attempt built a 21-level cache because
`driver_fast.py`'s `P_HI_HPA=150` was not in the environment; the BC source and
climatology file of the preset matter too). A 30-day run's 121 hours take
~25 min, a year ~5 h on one GPU; 24 MB per hour with the clear-sky fields.

Gate (MEASURED, `scripts/radiation/test_rad_cache.py`, PASS): a served hour
equals the computed one bit for bit in `hr0` and every diagnostic, the
SW/LW-split solve equals the default solve bit for bit, and the serve costs
0.02–0.03 s against 2.8 s to compute.

### Cost

Per 6-h step, median over 30-day runs on one H200 (2026-10, `instant` geometry):

| configuration | radiation stage | note |
|---|---|---|
| no cache (`bench_phys30d`) | 8.44 s | 2 solves + MAM4 re-binning |
| reference cache (`bench_phys30d_rc`) | **3.1 s** quiet (11.1 s on a shared box) | 1 solve + host work |

`window_gq@3` multiplies the shortwave solves by three (below): +16 % per model
step against `instant`, ~+10 % over a year with the reference cached.

## Solar geometry (RAD_SOLAR)

The full account of the dusk/dawn-artifact work — every approach tried, every test, and the
results — is [RADIATION_ARTIFACT_EXPLORATION.md](./RADIATION_ARTIFACT_EXPLORATION.md). This
section is the short version. The `runs/...` directories it cites are the
measurement runs on the development machine (gitignored, not in the repo).

Added 2026-11-11. One radiation call's heating is held for the whole interval
it stands for (`dT_rad += hr * STEP_SEC * RAD_EVERY`), and the calls land on
00/06/12/18 UTC every day.

**The default is `window_gq` at 3 nodes, trailing window, since 2026-11-17**
(below), and both presets run it. Set `RAD_SOLAR=instant` to reproduce earlier runs bit for bit.

**`instant`** (the default before 2026-11-17) takes the sun at the
call hour (`solar_zenith()`) and holds it for the interval. Each longitude
therefore sees the same four local times all run long:

- The daily-mean TOA insolation carries a **stationary wave-4 error** of −24 to
  +23 % (−24.3 to +22.8, equator to 60°, all seasons; MEASURED,
  `scripts/validation/test_solar_window.py`).
- A column just before dawn at the call hour gets no SW for six hours of
  mostly daylight; one just after dusk gets an afternoon's worth.

That is the dusk/dawn artifact, and it accumulates in `dT_rad`.

**`window`** treats each column over the interval centred on the call hour
(width `STEP_HOURS * RAD_EVERY`, declination held at the centre):

- **Sun angle.** It computes the sunlit fraction f and the sunlit-mean
  μ_eff = mean of cos(zenith) over the sunlit part, in closed form
  (`solar_window()`, any width). RRTMGP gets zenith = arccos(μ_eff) and a
  per-column TOA irradiance TSI·f. The interval's insolation, TSI·f·μ_eff, is
  then exact: 0.06 % in the test above, from the declination being held per
  window. This is the sunlit-mean choice of Hogan & Hirahara (2016, GRL 43,
  482). Cronin's (2014, JAS) insolation-weighted μ is off by −14 to −30 % in the
  thin-layer limit in terminator windows, which is this model's regime, so it is
  not used.
- **Cost.** Still one solve. jax-rrtmgp needs no new patch, because
  `two_stream.py:406` broadcasts an (nx, ny) irradiance.
- **Surface albedo.** It becomes the window's insolation-weighted albedo,
  1 − ΣFSNS/ΣFSDS over the hourly records inside the window. Record k is the
  mean over [k−1 h, k], per `time_bnds`, so for 6 h that is records
  `it1−2 … it1+3`. Without this, a column that is dark at the call hour but
  sunlit later in its window would get `ALB_FALLBACK` = 0.15. That would be
  about 4 % of the insolation: wrong over snow/ice (~0.8) and ocean (~0.06), and
  it matters for aerosol–surface multiple reflection.
- **Night.** Night columns (f = 0) get zenith = π. That is night for RRTMGP
  (`zenith >= π/2`).

**What else follows from `window`:**

- **Reference cache.** The cache key gains `solar = {mode, window_h}` only when
  the mode is not `instant`. Instant runs keep their existing cache
  directories, and each window width gets its own.
- **Diagnostics are interval means.** `arf_toa` and the TOA/surface fluxes
  become 6-h means, while `aod550`, the burdens and `dT_rad` are values at the
  call hour. With `ARF_AVG_H=24` the four samples are an exact daily mean
  (apart from the non-linearity below). Do not compare window diagnostics
  against an instant run's snapshots or instantaneous CESM fluxes.

**MEASURED 2026-11-11 — `window` biases the global forcing high; do not make it
the default yet.** Real RRTMGP, frozen aerosol, two references:

- **Hourly reference** (`scripts/validation/check_solar_window.py`): 24 instant
  calls a day, each held 1 h, on the model's anomaly (evolved minus MAM4
  reference). Five days: 1996-01-31, 03-21, 07-29, 07-31, 09-23. The states are
  the 30N/S plume at day 30, a mature 1-year plume (also transplanted onto both
  equinoxes) and the 15N plume.

  | vs hourly, mean of 5 days | `instant` | `window` |
  |---|---|---|
  | local error, band-summed (rms) | 44.0 % | **30.4 %** |
  | local error, worst column | 46.3 % | **24.8 %** |
  | local error, per level (rms) | 32.0 % | **18.2 %** |
  | wave-4 of the error, tropics (the artifact) | 45.0 % | **7.4 %** |
  | zonal-mean bias | **2.6 %** | 29.3 % |
  | daily-mean global ARF error | **+0.1 %** (−0.4..+0.7) | +20.2 % (+17.5..+25.3) |

  Every day has the same sign and shape. Per-day and per-band numbers are in
  `runs/solar_window_check/check_solar_window.log`. That run's "wave-4" column
  divided by the anomaly's near-zero zonal mean, so it is meaningless; the row
  above is recomputed from the saved fields (absolute wave-4 amplitude ÷ the
  hourly tropical rms), and the script now computes it that way. `instant`'s
  local errors are large (44 % rms for mature plumes) but cancel in the zonal
  and global mean; `window`'s are smaller but share one sign.
- **Geometry-only reference** (`runs/rad_solar_check/`, same day): 12 solves per
  window at 30-min suns, meteorology held at the call hour, evolved minus zero
  aerosol.
  - Global aerosol TOA SW-up: `window` +13.0 %, `instant` +0.1 %.
  - Wave-4 in the band-summed heating: `window` 9.5 % against the reference's
    real 8.3 %, `instant` 27.1 %.
  - The excess is +7 % even in fully sunlit windows, +11–20 % in part-lit ones,
    and largest in the tropics.
  - The window albedo contributes only 0.6 points.

- **Online 30-day A/B** (`runs/solar30d_{instant,window}`, the `bench_phys30d`
  scenario: 30N/S mirrored, 10 Tg SO₂/yr, physics radiation, everything else
  identical). Results at day 30:
  - **Artifact in `dT_rad`:** `window − instant` has a clean wave-4 (+ wave-8)
    signature peaking at **~3 mK at 50–75 hPa**. That is ~0.5 % of the 0.53 K
    zonal-mean plume warming and ~20 % of `dT_rad`'s own wave-4 at 52 hPa
    (15 mK, dominated by the plume's structure). So the artifact is real but
    small in the accumulated temperature after 30 days. It is stationary and
    keeps growing, damped only by the Planck response.
  - **Zonal-mean `dT_rad`:** `window` is +1 % at the peak and +3.5 % in the
    plume-integrated warming.
  - **Forcing diagnostic:** the reported `arf_toa_avg` differs by much more:
    −0.0708 vs −0.0598 W m⁻² (`window` 18 % stronger), and 21 % over days
    10–30. This is consistent with the offline +20 % bias.
  - **Aerosol:** burdens identical (mass ratio 0.999997).

- **The 2×2 follow-up (2026-11-12).** Sun angle {sunlit-mean, energy-weighted}
  × {1, 2 solves per call}, scored against the same hourly reference on the same
  5 days. Global ARF error / per-level rms error / wave-4 of the error:

  | mode | ARF error | per-level rms | wave-4 of error |
  |---|---|---|---|
  | `window` | +20.2 % | 18.2 % | 7.4 % |
  | `window_ins` | −13.9 % | 19.9 % | 2.8 % |
  | `window2` | +9.6 % | 9.6 % | 0.8 % |
  | `window2_ins` | −7.3 % | 13.7 % | 0.6 % |
  | `instant` | +0.1 % | 32.0 % | 45.0 % |

  The two μ choices bias the forcing in opposite directions, and two solves halve
  either bias. None is globally unbiased.
  - **Heating:** every window mode over-heats the plume relative to hourly
    radiation, `window_ins` included. So one tuned μ cannot null both the ARF
    bias and the heating bias.
  - **Long runs:** in the 209-day `instant` run, `dT_rad` at 52 hPa builds a
    stationary wave-4/8 pattern of up to 0.9 K against a 2.9 K zonal mean.
    Whether the scheme causes it is untested; it needs a long A/B.

  Full account with figures (`docs/figures/solar_*.svg`, drawn by
  `scripts/radiation/plot_solar_artifact.py`):
  `docs/RADIATION_ARTIFACT_EXPLORATION.md`.

- **The follow-ups (2026-11-12/13): `window_gq` gets both right.**
  `RAD_SOLAR=window_gq` (the default) solves the shortwave at `RAD_SOLAR_NODES` (default 3)
  Gauss–Legendre nodes of each column's sunlit part of the window (by default the
  window that ends at the call, t−6..t: `RAD_SOLAR_ALIGN=trailing`; `centred` =
  t−3..t+3, as in the results below, which a 3-day check finds equally accurate), each at the
  instantaneous sun with irradiance TSI × quadrature weight, renormalised so the
  insolation is exact (`solar_gauss()`). The longwave is solved once
  (`_hr_multi_impl()`, equal to N full solves to 5e-16,
  `scripts/validation/check_solar_gq_sw.py`). Same 5 days and reference:

  | mode | ARF error | per-level rms | wave-4 of error | radiation cost |
  |---|---|---|---|---|
  | `instant` | +0.1 % | 32.0 % | 45.0 % | 1.00 |
  | `window_blend@0.53` (fitted blend, held out) | +1.0 % | 16.6 % | 2.3 % | 1.00 |
  | `dither` (instant sun at a shifted time) | +1.7 % | 41.7 % | 12.6 % | 1.00 |
  | **`window_gq@2`** | **+4.2 %** | **7.1 %** | 3.1 % | 1.37 |
  | **`window_gq@3`** | **+1.1 %** | **6.0 %** | 1.2 % | 1.73 |

  - **Heating:** on a 10-day mean of a heavily loaded state, the global-mean
    heating error is +0.8 % for `window_gq@2`, −1.7 % for `instant`, and
    +101 % for `window`. The blend nulls the forcing but not the heating.
  - **`dither`** is unbiased, and its local error averages out over days (59 →
    23 % for a 10-day mean, `scripts/validation/check_solar_accum.py`), but
    each 6-h output is as noisy as `instant`.
  - **Coupled 30 days:** `dT_rad`'s wave-4 along the plume is ~43 mK in all four
    runs; the new schemes differ from `instant` by ~6 mK. Forcing vs `instant`:
    `window` +20 %, `window_gq@2` +3.5 %, `dither` +0.4 %.
  - **Cost per model step** (one GPU): `instant` 26.5 s, `window_gq@2` 28.2 s
    (+7 %), `window_gq@3` 30.7 s (+16 %); radiation is a third of a step.
  - The one-year A/B settled it: `instant`'s wave-4 grows to 484 mK by day 365,
    `window_gq@3`'s levels off at ~100 mK, for ~+10 % run time with the reference
    cached. `window_gq@3` is the default since 2026-11-17. (The radiation
    emulator, which is not part of this repo, cannot solve it, so under
    `RAD_ENGINE=emulator` the default stays `instant`; `radiation.py` keeps that
    rule for when an emulator fills the slot.)

So `window` removes the stationary dusk/dawn pattern and halves the local
error, but the single solve at the sunlit-mean μ over-reflects. `instant`'s
large local errors cancel in the global mean almost exactly. The review's
one-layer toy estimate below (+2–5 % global) is too small. Of the candidates
tested since (energy-weighted μ, two solves, a fitted blend, a dithered sun
time, Gauss-node solves), only `window_gq` removes both the local and the
global error (above).

**What `window` does not fix:**

1. **SW transfer is non-linear in μ across the interval.** The direct beam
   decays as exp(−τ/μ), layer reflectance rises at low sun, and the two-stream
   sources carry μ explicitly (`monochromatic_two_stream.py`). So
   mean(μ·X(μ)) ≠ f·μ_eff·X(μ_eff).
   - The single solve is exact for optically thin layers, where absorbed
     energy is TSI·f·τ whatever μ is, and for saturated ones.
   - In between (τ/μ ~ 0.3–1) it errs, mostly in windows that contain a
     terminator.
   - A delta-Eddington toy model (one layer, sulfate-like τ 0.02–0.1) puts the
     aerosol SW effect at **+2 to +10 % in terminator windows, +2 to +5 % in the
     global mean**, against −100…+64 % locally for `instant`. That is a review
     estimate, not RRTMGP.
   - The sign is the same everywhere (μ_eff sits below the energy-weighted μ,
     a slightly too-low sun). `instant`'s large local errors partly cancel in
     the global mean, so the global ARF can get slightly worse even as the
     local heating gets much better.
   - Gas heating mostly cancels between the two calls in anomaly mode; the LW
     part is untouched.
   - Splitting the window into two 3-h solves only took the toy error from
     5.1 to 3.7 %.
   - **To measure it:** RRTMGP at 10-min geometry sub-steps with the
     meteorology fixed at the call hour, averaged over the window, on sample
     hours across seasons, binned by f and latitude. A fitted f-dependent
     correction would be the cheap fix.
2. **Meteorology phase.** The meteorology is the hourly record at the call
   hour, i.e. the mean over [it1−1 h, it1]. It is centred 0.5 h before the
   window centre, and T, Q and O₃ are held across the window (midpoint rule).
3. **Plane-parallel geometry.** At 20 km the sun is still up until it is about
   4.5° below the horizon. Neither mode models that twilight, and very small f
   gives a small μ_eff whose 1/μ overstates the slant path. Small energy,
   shared with `instant`.


## Files

```
src/radiation/radiation.py          RadiationDriver: Mie tables, RRTMGP solve, heating()/heating_anomaly(),
                                    solar geometry (solar_window(), solar_gauss()), cache_key()
src/radiation/rad_cache.py          the per-hour cache (keys, partial entries, atomic writes)
src/radiation/rad_cache_build.py    clear-sky / reference entry builders shared by the scripts
scripts/radiation/build_ref_cache.py           batch-build the cache for a range of hours
scripts/radiation/test_rad_cache.py            gate: bit-identical cache, split solve
scripts/radiation/plot_solar_artifact.py       the docs/figures/solar_*.svg figures
scripts/validation/test_solar_window.py        geometry gate (CPU)
scripts/validation/check_solar_*.py            RAD_SOLAR against hourly radiation (GPU + archive)
scripts/validation/test_radiation.py           smoke test on one real CESM hour
scripts/validation/validate_radiation.py       against CESM's own heating and extinction
inputs/rad_data/palmer_williams_h2so4.dat      the refractive indices
inputs/cache/rad_ref/rad_ref_<key>/h*.npz      the cache (gitignored)
patches/jax-rrtmgp-zenith.patch                applied to models/jax-rrtmgp; the preset's pin checks it
```
