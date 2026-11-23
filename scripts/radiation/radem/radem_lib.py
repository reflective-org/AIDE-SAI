"""Shared pieces of the radem labelling / training scripts (AIDE-SAI, grid-generic).

Target (what the coupled model consumes): the AEROSOL INCREMENT of one radiation
call at the temperature the model has,
    d = heating(aerosol, T) - heating(no aerosol, T)
split into SW and LW heating on the band levels and the TOA terms (d sw_up, d olr);
and the clear-sky response P(T, dT) = heating(no aerosol, T + dT) - heating(no aerosol, T).
The anomaly mode is INC(evolved, T+dT) + P(T, dT) - INC(reference, T).

The radiation driver is built through coupling.py exactly as a run builds it, for
WHATEVER LINKING THE ENVIRONMENT SELECTS: export a preset's knobs first (MET_SRC,
BC_SRC, CLIM_FILE, ...; e.g. `set -a; . <(grep -v '^#' configs/jcm_met_emulated.env)`)
or use --preset in label.py. The gen2 checkpoint (radem_gen2_20261119) was made on
f09 with the AIDE-SAI-link campaign files; these scripts were ported from that
tree (scripts/radem, 6c5094f) so the JCM grid can get its own nets.
"""
import os
import sys
import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, os.path.join(REPO, 'src'))
import _paths                      # noqa: E402,F401
SEC_PER_DAY = 86400.0


def apply_preset(name):
    """Export a configs/<name>.env's KEY=VALUE lines (relative paths resolved against the repo) unless already set."""
    p = name if name.endswith('.env') else os.path.join(REPO, 'configs', f'{name}.env')
    for raw in open(p):
        line = raw.strip()
        if not line or line.startswith('#') or '=' not in line:
            continue
        k, v = line.split('=', 1)
        if v and not v.startswith('/') and ('/' in v) and os.path.exists(os.path.join(REPO, v)):
            v = os.path.join(REPO, v)
        os.environ.setdefault(k, v)
    os.environ.setdefault('P_LO_HPA', '1')
    os.environ.setdefault('P_HI_HPA', '150')          # driver_fast.py's band, what every launcher run uses
    os.environ.setdefault('XLA_PYTHON_CLIENT_PREALLOCATE', 'false')
    os.environ.setdefault('N_HOURS', '17520')         # the met readers index the run's hours: two years from H0


def make_driver(solar, nodes=3):
    """(drv, klevs, C): a RadiationDriver as coupling.main builds it for the configured linking."""
    import coupling as C
    import radiation as R
    C.install_bc_source()
    dU = C.open_var('U'); lat = dU['lat'].values; lon = dU['lon'].values
    plev = dU['hyam'].values * dU['P0'].values + dU['hybm'].values * C.PS_REF
    band = np.where((plev >= C.P_LO_HPA * 100) & (plev <= C.P_HI_HPA * 100))[0]
    klevs = list(range(band.min(), band.max() + 1))
    drv = R.RadiationDriver(C.open_var, klevs, lat, lon, C.DP_BIN * 1e-9, mmid_kg=C.MMID,
                            rho_dry=C.RHO_AER, solar=solar, solar_nodes=nodes,
                            solar_window_h=C.STEP_HOURS)
    return drv, klevs, C


class MetT:
    """Band temperature of the meteorology at an absolute model hour (what the run's radiation sees)."""
    def __init__(self, drv):
        self.drv = drv

    def __call__(self, hour_abs):
        d = self.drv
        return np.asarray(d._read3d(d.dT, 'T', hour_abs), np.float64)[d.band_pos]


def increment(drv, hour, num, T_band, geom=None):
    """Physics aerosol increment of one call: sw, lw (nzb, nlat, nlon) K/day; swup, olr, aod (nlat, nlon).
    geom=None: the driver's own solar mode; otherwise a one-solve geometry (zen, fsun, alb) for labels."""
    import jax.numpy as jnp
    if geom is None:
        h1, d1 = drv.heating(hour, jnp.asarray(num), T_band_override=T_band, split=True)
        h0, d0 = drv.heating(hour, None, T_band_override=T_band, split=True)
    else:
        h1, d1 = drv._heating(hour, jnp.asarray(num), T_band, split=True, geom=geom)
        h0, d0 = drv._heating(hour, None, T_band, split=True, geom=geom)
    h1 = np.asarray(h1); h0 = np.asarray(h0)
    s1 = np.asarray(d1['hr_sw_band']); s0 = np.asarray(d0['hr_sw_band'])
    return dict(sw=(s1 - s0) * SEC_PER_DAY, lw=((h1 - s1) - (h0 - s0)) * SEC_PER_DAY,
                swup=np.asarray(d1['sw_up_toa']) - np.asarray(d0['sw_up_toa']),
                olr=np.asarray(d1['olr']) - np.asarray(d0['olr']),
                aod=np.asarray(d1['aod550']))


def wlat(lat):
    return np.cos(np.deg2rad(lat))[:, None]


def gmean(x, lat):
    """Area-weighted (cos lat) global mean of (..., nlat, nlon)."""
    w = wlat(lat) * np.ones(x.shape[-2:])
    return (x * w).sum((-2, -1)) / w.sum()
