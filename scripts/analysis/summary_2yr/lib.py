"""Shared paths, constants and readers for the 2-year injection-vs-control summary.

Three runs (read-only inputs), all on the cesm_numerical linking (tomas_jax.fast, CESM
1996-1997 meteorology, sl K=12, RAD_MODE=anomaly, RAD_T_FEEDBACK=0):

  inj     injection run, steps 0-2919 (2 years at 6 h)    default runs/fast_2yr_inj6_30NS
  inj_y1  year 1 of the same scenario re-run with          default runs/fast_y1frames_inj6
          FRAME3D=1 FRAME_LEVELS=all (its frames3d/ snapshots and zonal frames are
          used for days 0-365; the injection run supplies year 2)
  ctrl    control, no injection, steps 0-2919              default runs/fast_2yr_ctrl

Each is a run directory holding coupled_*_<TAG>.npz with TAG = the directory's name.
Directories are given by --inj/--inj-y1/--ctrl or the env vars RUN_INJ, RUN_INJ_Y1,
RUN_CTRL; relative defaults resolve against this checkout's runs/. The CESM tropopause
(TROP_P) is read from the h1 archive at CESM_DIR / CESM_PREFIX, the same variables and
defaults as src/coupling.py.

Outputs (cache/, img/, metrics.json) go to the CURRENT WORKING DIRECTORY.
"""
import json
import os
import struct
import zipfile
from glob import glob

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
RUNS = os.path.join(REPO, 'runs')
COAST = os.path.join(REPO, 'inputs', 'assets', 'ne_110m', 'ne_110m_coastline.shp')

# (key, env var, CLI flag, default directory name under runs/)
_RUNSPEC = (('inj', 'RUN_INJ', '--inj', 'fast_2yr_inj6_30NS'),
            ('inj_y1', 'RUN_INJ_Y1', '--inj-y1', 'fast_y1frames_inj6'),
            ('ctrl', 'RUN_CTRL', '--ctrl', 'fast_2yr_ctrl'))
RUN, TAG = {}, {}            # filled by configure(); mutated in place so `from lib import RUN` stays live


def configure(**dirs):
    """Resolve the run directories: argument > env var > runs/<default>. Exported back to the
    environment so worker processes (any multiprocessing start method) see the same values."""
    for key, env, _, dflt in _RUNSPEC:
        v = dirs.get(key) or os.environ.get(env) or os.path.join(RUNS, dflt)
        v = os.path.abspath(v)
        os.environ[env] = v
        RUN[key] = v
        TAG[key] = os.path.basename(v.rstrip('/'))


def add_run_args(p):
    for key, env, flag, dflt in _RUNSPEC:
        p.add_argument(flag, dest=key, default=None,
                       help=f'{key} run directory (env {env}; default runs/{dflt})')
    return p


configure()

CACHE = os.path.abspath('cache')
IMG = os.path.abspath('img')
OUT = os.path.abspath('.')
os.makedirs(CACHE, exist_ok=True)
os.makedirs(IMG, exist_ok=True)

HDIR = os.environ.get('CESM_DIR',
                      '/data/cesm2.1.5_output/histSST/'
                      'f.e21.FWHIST.f09_f09_mg17.atmos-scale_fixedSST_1996-2014.001/'
                      'archive/atm/proc/tseries')
PREFIX = os.environ.get('CESM_PREFIX', 'f.e21.FWHIST.f09_f09_mg17.atmos-scale_fixedSST_1996-2014.001.cam')
H1 = f'{HDIR}/hour_1/{PREFIX}.h1'

G = 9.80616                       # coupling.py GRAV for dp/g
R_E = 6.371e6
RHO_AER = 1770.0
NSTEP = 2920
STEP_H = 6
S_PER_SO4 = 32.06 / 96.06
S_PER_SO2 = 32.06 / 64.06
# burden unit (kg/kg * Pa, area factor divided out) -> kg; plot_run.py BURDEN_KG
BURDEN_KG = R_E ** 2 * (np.pi / (192 - 1)) * (2.0 * np.pi / 288) / 9.80665
BURDEN_TG = BURDEN_KG / 1e9
EARTH_AREA = 4 * np.pi * R_E ** 2


# ---------------------------------------------------------------- npz members as memmaps
def npz_member(path, name):
    """Memory-map one array of an UNCOMPRESSED .npz (np.savez) without reading the rest.
    Returns a read-only np.memmap. Falls back to a full load for compressed members."""
    zf = zipfile.ZipFile(path)
    info = zf.getinfo(name if name.endswith('.npy') else name + '.npy')
    if info.compress_type != zipfile.ZIP_STORED:
        return np.load(path)[name]
    with open(path, 'rb') as f:
        f.seek(info.header_offset)
        h = f.read(30)
        nlen, xlen = struct.unpack('<HH', h[26:30])
        data_off = info.header_offset + 30 + nlen + xlen
        f.seek(data_off)
        ver = np.lib.format.read_magic(f)
        shape, fortran, dtype = np.lib.format._read_array_header(f, ver)
        arr_off = f.tell()
    return np.memmap(path, dtype=dtype, mode='r', offset=arr_off, shape=shape,
                     order='F' if fortran else 'C')


def npz_names(path):
    return [n[:-4] for n in zipfile.ZipFile(path).namelist()]


def final_or_ckpt(run, kind):
    """coupled_<kind>_<TAG>.npz if the run finished, else its _ckpt file."""
    d, t = RUN[run], TAG[run]
    f = f'{d}/coupled_{kind}_{t}.npz'
    return f if os.path.exists(f) else f'{d}/coupled_{kind}_{t}_ckpt.npz'


# ---------------------------------------------------------------- grid / geometry
_geo = {}


def geo():
    """lat, lon, band pressure levels, layer dp, area weights -- from a frames3d file."""
    if not _geo:
        f = sorted(glob(f"{RUN['ctrl']}/frames3d/step_*.npz"))[0]
        z = np.load(f)
        lat, lon, plev = z['lat'], z['lon'], z['plev_pa']
        # layer thickness exactly as coupling.py: DP = np.gradient(PLEV_PA)
        dp = np.abs(np.gradient(plev.astype(np.float64)))
        # cell area (m2) on the f09 grid, poles as half rows
        dlat = np.deg2rad(np.gradient(lat))
        dlon = np.deg2rad(lon[1] - lon[0])
        area = R_E ** 2 * np.cos(np.deg2rad(lat))[:, None] * dlat[:, None] * dlon * np.ones((1, lon.size))
        area *= EARTH_AREA / area.sum()
        _geo.update(lat=lat, lon=lon, plev=plev, dp=dp, area=area, dp_bin=z['dp_bin_nm'])
    return _geo


# ---------------------------------------------------------------- frames3d snapshots
def snap_files(run):
    """{step: path} of frames3d snapshots for 'inj' (year-1 re-run + year-2) or 'ctrl'."""
    if run == 'inj':
        fs = glob(f"{RUN['inj_y1']}/frames3d/step_*.npz") + glob(f"{RUN['inj']}/frames3d/step_*.npz")
    else:
        fs = glob(f"{RUN[run]}/frames3d/step_*.npz")
    out = {}
    for f in fs:
        b = os.path.basename(f)
        if len(b) != 13 or not b[5:9].isdigit():     # step_NNNN.npz only; skip step_NNNN.tmp.npz
            continue
        out[int(b[5:9])] = f
    return dict(sorted(out.items()))


# ---------------------------------------------------------------- CESM tropopause
_trop = {}


def trop_p(hour_abs):
    """CESM TROP_P (Pa) at an absolute archive hour, (nlat, nlon)."""
    import xarray as xr
    if 'ds' not in _trop:
        fs = glob(H1 + '.TROP_P.*.nc')
        if not fs:
            raise FileNotFoundError(f'no {H1}.TROP_P.*.nc: point CESM_DIR / CESM_PREFIX at the CESM h1 archive')
        _trop['ds'] = xr.open_dataset(fs[0], decode_times=False)
    return np.asarray(_trop['ds']['TROP_P'].isel(time=int(hour_abs)).values, np.float64)


def strat_mask(hour_abs):
    """(nlev, nlat, nlon) True where the band level is at or above CESM's tropopause."""
    g = geo()
    pt = trop_p(hour_abs)
    return g['plev'][:, None, None] <= np.where(np.isfinite(pt), pt, 1e9)[None]


# ---------------------------------------------------------------- timeseries
def load_ts(run):
    z = np.load(final_or_ckpt(run, 'timeseries'))
    return {k: z[k] for k in z.files}


def coast():
    """Natural Earth 1:110m coastlines (inputs/assets/ne_110m) -> list of (lon, lat), lon in
    [0, 360). Returns [] (maps drawn without coastlines) if the shapefile is missing."""
    if not os.path.exists(COAST):
        print(f'note: {COAST} not found; maps drawn without coastlines', flush=True)
        return []
    b = open(COAST, 'rb').read(); pos = 100; out = []
    while pos < len(b):
        _, clen = struct.unpack('>ii', b[pos:pos + 8]); pos += 8
        rec = b[pos:pos + 2 * clen]; pos += 2 * clen
        st = struct.unpack('<i', rec[:4])[0]
        if st not in (3, 13, 23):
            continue
        npart, npt = struct.unpack('<ii', rec[36:44])
        parts = list(struct.unpack(f'<{npart}i', rec[44:44 + 4 * npart])) + [npt]
        pts = np.frombuffer(rec[44 + 4 * npart:44 + 4 * npart + 16 * npt], '<f8').reshape(-1, 2)
        for a, c in zip(parts[:-1], parts[1:]):
            seg = pts[a:c].copy()
            seg[:, 0] = seg[:, 0] % 360.0
            # split where the line wraps the dateline seam at 0/360
            br = np.where(np.abs(np.diff(seg[:, 0])) > 180)[0]
            for piece in np.split(seg, br + 1):
                if len(piece) > 1:
                    out.append(piece)
    return out


def save_gif(frames, path, ms=130, colors=200):
    """Animated GIF with one palette for all frames (built from up to 8 sampled frames)."""
    from PIL import Image
    w, h = frames[0].size
    pick = sorted(set(np.linspace(0, len(frames) - 1, min(8, len(frames))).astype(int)))
    strip = Image.new('RGB', (w, h * len(pick)))
    for k, j in enumerate(pick):
        strip.paste(frames[j].resize((w, h)), (0, h * k))
    pal = strip.quantize(colors=colors, method=Image.Quantize.MEDIANCUT)
    q = [f.quantize(palette=pal, dither=Image.Dither.NONE) for f in frames]
    q[0].save(path, save_all=True, append_images=q[1:], duration=ms, loop=0)


def wjson(path, obj):
    with open(path, 'w') as f:
        json.dump(obj, f, indent=1, default=float)
