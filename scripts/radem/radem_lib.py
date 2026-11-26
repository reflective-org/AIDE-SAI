"""Shared pieces of the window_gq aerosol-increment radiation emulator (rgq/radem).

Target (what the linked model consumes): the AEROSOL INCREMENT of one radiation
call at the temperature the model actually has,
    d = heating(aerosol, T) - heating(no aerosol, T)
split into SW and LW heating on the band levels and the TOA terms
(d sw_up, d olr). Q (feedback) and forcing pairs (inj - ctl) are differences of
these at the same T, so the clear-sky part never has to be emulated.

SW is exactly linear in the TOA irradiance, so one SW network evaluated at a
column's cos(zenith) with unit irradiance (TSI) serves any solar mode: window_gq
calls it at each Gauss node's mu and applies solar_gauss()'s scales. LW does not
see the sun: one LW network, evaluated once.

Everything here reads the same archive/driver paths as the coupled model
(rgq/src), with RAD_SOLAR chosen per driver.
"""
import os
import sys
import json
import numpy as np

RGQ = os.environ.get('RADEM_TREE', os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))   # the repo root (was the rgq tree)
sys.path.insert(0, os.path.join(RGQ, 'src'))
import _paths                      # noqa: E402,F401

MET_DIR = '/data/AIDE-SAI-link/runs/emu_winds_P2s200_1997_1998/met'
RUNS = {'inj': '/data/AIDE-SAI-link/runs/csb_1997_yr/csb_yr/frames3d',
        'ctl': '/data/AIDE-SAI-link/runs/csb_1997_yr/csb_yr_ctl/frames3d',
        # coupled-atmosphere feedback runs (aerosol states; their live T is not saved -> met T as proxy)
        'ca_inj0': '/data/AIDE-SAI-link-perf/runs/ca_fb_inj_s0/frames3d', 'ca_ctl0': '/data/AIDE-SAI-link-perf/runs/ca_fb_ctl_s0/frames3d',
        'ca_inj1': '/data/AIDE-SAI-link-perf/runs/ca_fb_inj_s1/frames3d', 'ca_ctl1': '/data/AIDE-SAI-link-perf/runs/ca_fb_ctl_s1/frames3d'}
SEC_PER_DAY = 86400.0


def setup_env():
    """The env the campaign stack runs radiation with (bench.sh / launch.sh)."""
    os.environ.setdefault('H0', '8766')
    os.environ.setdefault('P_LO_HPA', '1')
    os.environ.setdefault('P_HI_HPA', '150')
    os.environ.setdefault('XLA_PYTHON_CLIENT_PREALLOCATE', 'false')


CLIM_FILE = '/data/AIDE-SAI-link/runs/bc_clim/bc_clim.npz'


def make_driver(solar, nodes=3, bc='clim'):
    """A RadiationDriver exactly as coupling.py builds it for the campaign stack
    (BC_SRC=clim: gases/surface from the monthly climatology, T on the whole
    radiation column from the met file), with the given solar mode."""
    setup_env()
    import coupling as C
    import radiation as R
    if bc == 'clim':
        import bc_clim
        C.open_var = bc_clim.ClimArchive(CLIM_FILE, MET_DIR).open_var
    dU = C.open_var('U'); lat = dU['lat'].values; lon = dU['lon'].values
    plev = dU['hyam'].values * dU['P0'].values + dU['hybm'].values * C.PS_REF
    band = np.where((plev >= C.P_LO_HPA * 100) & (plev <= C.P_HI_HPA * 100))[0]
    klevs = list(range(band.min(), band.max() + 1))
    drv = R.RadiationDriver(C.open_var, klevs, lat, lon, C.DP_BIN * 1e-9, mmid_kg=C.MMID,
                            rho_dry=C.RHO_AER, solar=solar, solar_nodes=nodes)
    return drv, klevs


class MetT:
    """Band temperature of the P2@200 met file (what MET_SRC=file radiation sees)."""
    def __init__(self, met_dir=MET_DIR):
        m = json.load(open(os.path.join(met_dir, 'meta.json')))
        self.h0 = int(m['hour0']); self.step = int(m['step_hours']); self.klevs = [int(k) for k in m['klevs']]
        self.T = np.load(os.path.join(met_dir, 'T.npy'), mmap_mode='r')

    def __call__(self, hour_abs):
        i, r = divmod(hour_abs - self.h0, self.step)
        assert r == 0 and 0 <= i < self.T.shape[0], hour_abs
        return np.array(self.T[i], np.float64)


def frame_path(run, step):
    return os.path.join(RUNS[run], f'step_{step:04d}.npz')


def load_frame(run, step):
    """(num (NBINS, nzb, nlat, nlon) float32 #/kg, absolute hour) of one 6-hourly frame:
    the aerosol the radiation call at that hour saw (frames are written after radiation)."""
    z = np.load(frame_path(run, step))
    return np.asarray(z['num'], np.float32), int(z['hour_abs'])


def increment(drv, hour, num, T_band, geom=None):
    """Physics aerosol increment of one call: dict of
    sw, lw (nzb, nlat, nlon) K/day; swup, olr (nlat, nlon) W/m2; aod (nlat, nlon).
    geom=None: the driver's own solar mode (instant or window_gq); otherwise a
    one-solve geometry (zen, fsun, alb) for labels."""
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
