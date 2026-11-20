# Coupling variables — complete reference

Every variable the coupled model consumes or produces, start to finish. Compiled
from `coupling.py`, `radiation.py`, and the production driver `driver_fast.py`.

Grid: the meteorology source's. Under `MET_SRC=cesm` (the `cesm_numerical` preset) the
FWHIST f09 grid, 192 lat × 288 lon, 70 hybrid levels; under `MET_SRC=jcm` (the
`jcm_numerical` preset) the T63 Gaussian 96 × 192 grid with JCM's native levels. The
stratospheric band is selected by pressure (`P_LO_HPA`..`P_HI_HPA`; 1–150 hPa
under `driver_fast.py`). Coupling step = `STEP_HOURS` (6 h). The met source and
grid are stamped into the checkpoint (`GRID_CFG`), and a `RESUME` across grids is
refused.

**Where each field comes from.** U, V, OMEGA and T come from the meteorology
source (`MET_SRC`): the hourly h1 archive (`cesm`), a jcm-strat run directory,
6-hourly (`jcm`), or step-end frames in `MET_DIR` (`file`). Every other field
below comes from the boundary source (`BC_SRC`): the hourly h1 archive (`cesm`,
the module default) or the monthly climatology in `CLIM_FILE` (`clim`, both
presets), which serves the same variable names
([BOUNDARY_CONDITIONS.md](./BOUNDARY_CONDITIONS.md)). Section 1 lists them by
their CESM names.

---

## 1. Input fields (CESM FWHIST `*.cam.h1.*` names)

### Dynamics / thermodynamics (always; from `MET_SRC`)
| var | units (file) | role |
|-----|------|------|
| `U`      | m/s     | zonal wind → advection |
| `V`      | m/s     | meridional wind → advection |
| `OMEGA`  | Pa/s    | vertical (pressure) velocity → advection |
| `T`      | K       | temperature → micro kernels, air density (ρ=p/RₐT), radiation |

### Gas phase (MICRO=full)
| var | units (file) | role |
|-----|------|------|
| `SO2`    | mol/mol (vmr) → kg/kg | gas tracer IC+BC; oxidized to H₂SO₄ |
| `H2SO4`  | mol/mol → kg/kg | gas tracer IC+BC; feeds nucleation + condensation |
| `OH`     | mol/mol → molec/cm³ | oxidant driving SO₂+OH→H₂SO₄ — read only under `OH_SZA=0`; the default replaces it with the SZA curve (§6) |
| `RELHUM` | percent → fraction | read only under `MET_RH=cesm`; the default `MET_RH=liq` recomputes RH over liquid from Q, T, p every step (settling, water uptake, wet diagnostics; the radiation optics always do) |

### Radiation (RAD=1) — read by `radiation.py`
| var | units | role |
|-----|------|------|
| `T`    | K     | atmospheric temperature profile |
| `Q`    | kg/kg | specific humidity (water vapor) |
| `O3`   | mol/mol | ozone (SW/LW absorber) |
| `CH4`  | mol/mol | methane (LW absorber) |
| `FLDS` | W/m²  | surface downwelling longwave (LW boundary) |
| `FLNS` | W/m²  | surface net longwave |
| `FSDS` | W/m²  | surface downwelling shortwave |
| `FSNS` | W/m²  | surface net shortwave → surface albedo = 1 − FSNS/FSDS |

### MAM4 modal aerosol (used if `AER_SRC=mam4`, or as MAM4 IC/BC reservoir)
| var | units | role |
|-----|------|------|
| `num_a1`,`num_a2`,`num_a3` | #/kg | modal number (Aitken/accum/coarse) |
| `so4_a1`,`so4_a2`,`so4_a3` | kg/kg | modal sulfate mass |
| `dgnumwet1/2/3` | m | modal wet diameter — **only if `INIT_BIN=dgnum`** |

## 2. CARMA initial-condition file (used if `AER_SRC=carma`)
File: `cesm2.2_CARMA16node_freerun_1wk_19910601_1deg...nc` (`CARMA_FILE`), frame `CARMA_FRAME`.
| var | units | role |
|-----|------|------|
| `PRSUL01`..`PRSUL20` | kg/kg | pure-sulfate bins (fine/nucleation mode, Dp 0.69 nm–2.59 µm) |
| `MXAER01`..`MXAER20` | kg/kg | mixed-group sulfate bins (accum/coarse, Dp 100 nm–17.4 µm) |

Projected onto TOMAS bins via mp = (4/3)πr³·`CARMA_RHO`.

## 3. Coordinate / structural variables (from the h1 files)
`lat`, `lon`, `lev`, `time`, and the hybrid-sigma coefficients `hyam`, `hybm`, `P0`
→ reference-pressure levels `plev = hyam·P0 + hybm·PS_REF`.

---

## 4. Prognostic state (carried/evolved across steps)
| var | shape | units | notes |
|-----|-------|-------|-------|
| `num` | (NBINS, nlev, nlat, nlon) | #/kg | per-bin number mixing ratio (advected tracer) |
| `mas` | (NBINS, nlev, nlat, nlon) | kg/kg | per-bin **dry sulfate** mass mixing ratio (advected tracer) |
| `so2` | (nlev, nlat, nlon) | kg/kg | SO₂ gas |
| `h2so4` | (nlev, nlat, nlon) | kg/kg | H₂SO₄ vapor |
| `dT_rad` | (nlev, nlat, nlon) | K | accumulated radiative temperature increment. Fed back into T only under `RAD_T_FEEDBACK=1`; by default a diagnostic that enters only the perturbed radiation call's emission temperature (heating rate output as `hr_kday`) |

Advected tracer count = 2·NBINS + 2 (num+mas per bin, plus SO₂, H₂SO₄) = **82 at 40 bins**.

## 5. Derived / setup variables (computed, not read)
| var | meaning |
|-----|---------|
| `klevs`, `band` | native level indices of the stratospheric band |
| `PLEV_PA`, `pres3d` | band pressures [Pa] |
| `DP` | Δp per level [Pa] (vertical advection + burden weighting) |
| `XK` | bin-boundary masses [kg] (`make_grid(NBINS, XK0, 2.0)`) |
| `MMID` | bin geometric-mean mass [kg] |
| `DP_BIN` | per-bin representative diameter [nm] (from MMID, RHO_AER) — used by radiation + meanDp |
| `rho` | air density p/(RD·T) [kg/m³] — per step; converts mixing ratio ↔ per-box concentration |
| `A` / `wgt3d` | burden weight (area×Δp/g) for global integrals |

---

## 6. Configuration knobs (environment variables)

### Run control
`N_DAYS` / `N_HOURS` (length), `H0` (start hour index), `STEP_HOURS` (6),
`OUT_TAG`, `LOG_EVERY`, `FRAME_EVERY`, `DEBUG`, `PROFILE`.

### Domain / grid
`P_LO_HPA`, `P_HI_HPA` (band top/bottom), `N_LEV` (subsample levels; 0=all),
`PROBE_HPA` (diagnostic level), `N_BC_TOP`, `N_BC_BOT` (pinned boundary levels),
`LAT_FREEZE` (=80°, polar-cap latitude, constant).

### Aerosol source & bin grid
`AER_SRC` (`carma`|`mam4`), `N_BINS` (0→native 40), `CARMA_FILE`, `CARMA_FRAME`,
`CARMA_RHO` (1923), `INIT_BIN` (`so4`, the default | `dgnum`, legacy), `INIT_SIGMA`.

### Microphysics
`MICRO` (`full`|`coag`|`off`; `driver_fast.py` accepts `full` and `off`) and
`MICRO_ENGINE`, which engine fills the `full` chain:
- `fast` (default; both presets): `tomas_jax.fast` from tomas-jax `gpu-fast`
  (`787c991`), `FAST_DT` 360 s × 60 inner steps per 6 h step, `FAST_CELL_CAP`
  50000 and `FAST_SORT` 1 (set by `driver_fast.py`), `FAST_FN_SCALE` (1.0),
  `FAST_MULTIGPU` (0). These `FAST_*` variables are read by
  `src/microphysics/tomas_fast.py` and are not in `run_prod.py`'s knob table.
- `full`: tomas-jax dev `make_step` (needs a dev checkout in the tomas-jax
  slot): `MICRO_DT` (60 s), `MICRO_NCOAG` (1), `MICRO_COAG` (`fixed`),
  `MICRO_FION` (0), `MICRO_MNFIX_TRIM` (1), `MICRO_MAX_WALL` (300 s abort),
  `MICRO_GPUS` (1). This is also what a bare `coupling.py` runs.

Shared: `ALPHA_COND` (1.0), `CELL_CHUNK` (50000), `TRACER_CHUNK`, `SETTLE`,
`WET_SETTLING`, `MET_RH` (`liq`); `N_COAG_SUBSTEPS` for the legacy `coag` path
only. Composition: sulfate + water, gases SO2 + H2SO4. (`MICRO_SUBSTEPS`,
`COAG_MAX_SUBSTEPS` and `NUC_*` were removed 2026-11-18; `MICRO_SUBSTEPS` exits
if set.)

### OH diurnal
`OH_SZA` (**default 1** = SZA curve, per-substep diurnal; 0 = constant CESM OH),
`OH_PEAK` (2.3e6 molec/cm³ noon peak), `OH_SUBSTEPS` (samples/step; each driver
sets its own, see below).
The curve is a **parabola in μ = cos(SZA) with no constant term**, `OH = a·μ² + b·μ`,
least-squares fitted at import to `OH_SZA_KNOTS` in `coupling.py` — digitized off
Hanisco et al. 2001 Fig. 1: 2.3 / 2.0 / 1.5 / 0.7 / 0 ×10⁶ molec/cm³ at SZA
0 / 30 / 45 / 60 / 90°. Edit those points to refit; nothing else changes.
Dropping the constant term is what forces OH → 0 at the terminator.
Being least-squares it does **not** pass through the knots: +4.0 / −4.8 / −7.2 /
+19.2 % at SZA 0 / 30 / 45 / 60, exact 0 at 90. So peak OH is 2.393e6, 4% above
`OH_PEAK`. It was `OH_PEAK*max(0,1-(θ/90)²)` before 2026-07-29 — a parabola in θ
rather than μ, which decayed far too slowly (1.22e6 at 60° vs the paper's 0.7e6).
⚠ `μ` is clipped to [0,1] and the **output must not be clipped instead**: `a·μ²+b·μ`
has its second root at μ = −0.652, so it turns positive again past SZA 130.7° and
would emit OH at local midnight. A plot over SZA ∈ [−90, 90] cannot reveal this.
⚠ The curve *replaces* CESM OH, so magnitude changes as well as time resolution.
Measured over the 13–88 hPa band: 24 h-mean OH 0.50× CESM's, tropical SO2 lifetime
18.7 d → 36.3 d. No vertical structure (CESM has a 3.4× gradient across the band),
and polar-night OH is exactly 0.
The cycle is sampled `OH_SUBSTEPS` (60) times per step: under the fast engine
once per 360 s inner step (`tomas_fast.py` sets it to `N_FAST_STEPS` unless
`OH_SUBSTEPS` is exported), under `full` each of the 360 dev steps takes the
sample covering it.

### SAI injection source
`INJ_SO2_TG_YR`, `INJ_H2SO4_TG_YR` (rates), `INJ_HPA` (altitude), `INJ_LAT`,
`INJ_LON`, `INJ_ZONAL`, `INJ_MIRROR`. Both **rates** at 0 = no-injection control
(the default); the geometry knobs are then inert.

### Radiation
`RAD` (on/off), `RAD_MODE` (`anomaly`), `RAD_T_FEEDBACK` (0), `RAD_EVERY`,
`RAD_SOLAR` (`window_gq`), `RAD_SOLAR_NODES` (3), `RAD_SOLAR_ALIGN` (`trailing`),
`WET_OPTICS` (1), `ARF_AVG_H` (24), `RAD_REF_CACHE`, `CO2_PPM` (380),
`N2O_PPB` (319). See [RADIATION.md](./RADIATION.md).

### Advection
`ADV_SCHEME` (`sl`, with `ADV_SL_K` 12; `lr`, `ffsl`, `fast` selectable),
`ADV_CFL` (0.5), `ADV_F32` (float32 transport), `ADV_VPOS` (1), `ADV_POLAR`
(`zonal`). Full list in [CONFIGURATION.md](./CONFIGURATION.md).

---

## 7. Physical constants
| const | value | meaning |
|-------|-------|---------|
| `RD` | 287.05 J/kg/K | dry-air gas constant |
| `GRAV` | 9.80665 m/s² | gravity |
| `BOXVOL` | 1.0e6 cm³ | box volume (=1 m³); per-m³ concentrations |
| `PS_REF` | 1.0e5 Pa | reference surface pressure (level construction) |
| `MW_AIR` | 28.9644 g/mol | air molar mass (vmr→MMR) |
| `RHO_AER` | 1770 kg/m³ | sulfate density (mass↔size for TOMAS) |
| `CARMA_RHO` | 1923 kg/m³ | CARMA-file sulfate density (projection) |
| `NEPS_N` | 1e-10 | empty-bin number threshold |
| From tomas_jax config | | `NBINS` (40), `ICOMP`, `XK0`, `KB`, `MW_H2SO4`, `MW_SO2`, `SRTSO4/SRTSO2/SRTH2O` |
| Radiation | | `TSI_DEFAULT` (solar irradiance), `SFC_EMIS` (surface emissivity) |

---

## 8. Outputs / diagnostics produced

The authoritative list is the `ts = {k: [] for k in (...)}` literal in
`coupling.py` — check it there if this section looks short, since new
diagnostics are added by extending that tuple.

**Timeseries** (`coupled_timeseries_<TAG>.npz`), one value per coupling step:

| group | keys |
|---|---|
| normalizers | `N0`, `M0` (scalars, not series) |
| time | `hours` |
| burdens | `Nburden`, `Mburden`, `SO2burden`, `H2SO4burden` |
| size | `meanDp_nm`, `meanDp_num_nm`, `meanDp_mass_nm`, `reff_nm` |
| budget (cumulative, fractions of M0) | `B_adv_np`, `B_adv_pol`, `B_floor`, `B_micro`, `B_bc`, `B_settle`, `B_vf_in`, `B_vf_out` |
| radiation | `dT_min`, `dT_max`, `dT_rms`, `arf_toa`, `arf_toa_avg`, `aod550` |
| accounting | `nsub`, `Nmin`, `Nmax`, `Nfloor_cum`, `clipMadd_cum`, `clipMrem_cum`, `injSO2_cum`, `settleM_cum` |

**Frames** (`coupled_frames_<TAG>.npz`), probe-level snapshots every
`FRAME_EVERY` hours: `frames_num`, `frames_mas`, `frames_dT`, `frames_so2`,
`frames_h2so4`, plus `frame_hours`, `probe_hpa` and `xk`.

**Both files also carry the run's configuration stamp** — `inj_cfg` /
`inj_cfg_keys` (the injection scenario) and `phys_cfg` / `phys_cfg_keys` (the
physics-mode flags). `RESUME` compares these: an `INJ_*` mismatch is refused
outright, a physics-mode mismatch only warns. Both arrays are append-only, so
adding a field must not lock out older checkpoints.

Older runs may lack the newer keys (`prod90d`, for instance, predates
`reff_nm`); `scripts/utils/plot_run.py` degrades rather than failing when one is absent.
