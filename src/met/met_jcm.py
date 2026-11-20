"""MET_SRC=jcm: JCM (jcm-strat) output served as the model's meteorology AND grid.

Why this exists (2026-11-18). coupling.py reads every
field through `open_var(name)` and indexes the result as
`ds[name].isel(time=<hour>, lev=<native levels>, lat=...).values`, with the grid
taken from `ds['lat']`, `ds['lon']`, `ds['hyam'] * ds['P0'] + ds['hybm'] * 1e5`.
`JcmMet.open_var` returns the same Dataset surface as `bc_clim.ClimArchive`
(it reuses bc_clim's ClimDataset/ClimVar stand-ins), so the model runs on
whatever horizontal grid and level set the met source declares -- here JCM's
T63 Gaussian grid (96 x 192) and its 95 hybrid levels -- with no JCM branch in
the physics. The climatology served beside it (BC_SRC=clim) must be on the
same grid and levels: scripts/met/regrid_clim_to_grid.py writes it, and
coupling.install_bc_source refuses a mismatch.

What it serves (JCM name -> model name), from the jcm-strat run directory
JCM_DIR (default: the AL7h 1970-2022 real-ERA5 chain), read-only:
    u_wind -> U [m/s]   v_wind -> V [m/s]   omega -> OMEGA [Pa/s]
    temperature -> T [K]   specific_humidity -> Q [kg/kg] (see MET_Q below)
Layout: JCM stores (time, level, lon, lat) with level 0 = the SURFACE; the model
wants (lev, lat, lon) with lev 0 = the TOP (CESM's convention), so every read
flips the level axis and transposes. hyam/hybm/hyai/hybi are JCM's
hybrid_a/b (a in Pa, divided by P0 = 1e5) in the same top-first order, so the
model's level pressure a + b * 1e5 is JCM's own nominal pressure of the level;
every level above 78 hPa is pure pressure (b = 0) and the seven band levels
between 86 and 138 hPa carry b <= 0.0074 -- the same reference-pressure
convention the CESM path uses for CESM's hybrid levels (no vertical
interpolation of the meteorology).

Water vapour: JCM's specific_humidity is NOT a moisture field. jcm-strat is a
dry model (Held-Suarez + nudging) and carries q as a passive constant: 2.32e-3
kg/kg at every level and column (checked 1996-01-01 06Z, 0.05-850 hPa, min/max
2.32e-3/2.33e-3), ~600x the stratospheric value. Q is therefore taken from the
BC source (the CESM climatology regridded to this grid) unless MET_Q=met; see
coupling.py MET_Q.

Calendar: the model's hour axis is CESM's (noleap hours since 1996-01-01 00Z,
bc_clim.ORIGIN); JCM runs a Gregorian calendar. Hour h is served from the JCM
frame with the SAME year/month/day/hour, so the dates, the season, the
climatology's day of year and the solar geometry all line up with the CESM
runs; 1996-02-29 (and any other 29 February) is skipped, i.e. the step
1996-02-28 18Z -> 03-01 00Z interpolates the winds across 30 h of JCM time.
JCM output is 6-hourly instantaneous (00/06/12/18Z), so only hours on that grid
are served; anything else is refused (the model reads the meteorology at the
coupling-step ends only, STEP_HOURS=6; ADV_SL_HOURLY is refused upstream).
Each segment's first frame is 06Z on 1 January -- the 00Z frame is the last
frame of the previous year's segment, so the index spans one extra year.

I/O: the files are zlib-compressed in chunks of 14 time steps x 32 levels, so a
single-frame read decompresses 14 frames anyway; a read caches the whole
14-frame block of that variable (~100 MB) and the next 13 steps hit the cache.
"""
import datetime
import glob
import os

import cftime
import netCDF4
import numpy as np

import bc_clim

P0 = 1.0e5
# model name -> JCM variable
VARMAP = {'U': 'u_wind', 'V': 'v_wind', 'OMEGA': 'omega', 'T': 'temperature',
          'Q': 'specific_humidity'}
DEFAULT_DIR = '/data/JCM_stripped/jcm-strat/runs/p31/p31_AL7h70'


def hour_to_datetime64(h):
    """Model hour (noleap, since bc_clim.ORIGIN) -> numpy datetime64[h] with the
    same Y-M-D-H in the Gregorian calendar."""
    d = bc_clim.ORIGIN + datetime.timedelta(hours=int(h))
    return np.datetime64(f'{d.year:04d}-{d.month:02d}-{d.day:02d}T{d.hour:02d}', 'h')


def noleap_hour(year, month, day, hour=0):
    """Inverse of hour_to_datetime64 for a date that exists in noleap."""
    d = cftime.DatetimeNoLeap(year, month, day, hour)
    return int(round((d - bc_clim.ORIGIN).total_seconds() / 3600.0))


class JcmMet:
    """open_var() surface over a jcm-strat chain. years: the calendar years whose
    segments are indexed (each <dir>/*_<YYYY>0101/longrun_day*.nc)."""

    def __init__(self, jcm_dir, years, block_cache=12):
        self.jcm_dir = jcm_dir
        self.source = f'jcm:{os.path.basename(os.path.normpath(jcm_dir))}'
        self.clim_file = self.source          # bc_clim.ClimDataset's encoding fallback
        files = []
        for y in years:
            segs = sorted(glob.glob(os.path.join(jcm_dir, f'*_{y:04d}0101')))
            for s in segs:
                files += glob.glob(os.path.join(s, 'longrun_day*.nc'))
        if not files:
            raise SystemExit(f'met_jcm: no longrun_day*.nc under {jcm_dir} for years '
                             f'{years[0]}..{years[-1]} (JCM_DIR)')
        # time index: datetime64[h] -> (file, frame)
        self.index = {}
        for f in sorted(files):
            with netCDF4.Dataset(f) as nc:
                tv = nc['time']
                ts = cftime.num2date(tv[:], tv.units, getattr(tv, 'calendar', 'standard'),
                                     only_use_cftime_datetimes=False,
                                     only_use_python_datetimes=True)
                if not hasattr(self, 'lat'):
                    self._grid_from(nc)
            for i, t in enumerate(ts):
                key = np.datetime64(t.replace(minute=0, second=0, microsecond=0), 'h')
                if key in self.index:
                    raise SystemExit(f'met_jcm: {key} appears in two files '
                                     f'({self.index[key][0]}, {f})')
                self.index[key] = (f, i)
        keys = sorted(self.index)
        self.t_first, self.t_last = keys[0], keys[-1]
        # the hour axis: every model hour up to the last frame (the loop's range
        # check then says when a run would run past the data)
        last = self.t_last.astype(object)
        self.ntime = noleap_hour(last.year, last.month, last.day, last.hour) + 1
        self.nfiles = len(files)
        self._cache = {}
        self._cache_order = []
        self._cache_max = int(block_cache)
        self._nc = {}

    def _grid_from(self, nc):
        self.lat = np.asarray(nc['lat'][:], np.float64)
        self.lon = np.asarray(nc['lon'][:], np.float64)
        a = np.asarray(nc['hybrid_a_full'][:], np.float64)[::-1]
        b = np.asarray(nc['hybrid_b_full'][:], np.float64)[::-1]
        ai = np.asarray(nc['hybrid_a_half'][:], np.float64)[::-1]
        bi = np.asarray(nc['hybrid_b_half'][:], np.float64)[::-1]
        self.P0 = np.float64(P0)
        self.hyam, self.hybm = a / P0, b
        self.hyai, self.hybi = ai / P0, bi
        self.lev = (a + b * P0) / 100.0                      # reference pressure [hPa], top-first
        self.nlev_native = a.size
        if not (np.all(np.diff(self.lat) > 0) and np.all(np.diff(self.lev) > 0)):
            raise SystemExit('met_jcm: expected increasing latitudes and a top-first '
                             'increasing reference pressure after the level flip')
        # (time, level, lon, lat) is the layout this reader assumes
        dims = nc['u_wind'].dimensions
        if dims != ('time', 'level', 'lon', 'lat'):
            raise SystemExit(f'met_jcm: u_wind dims {dims} != (time, level, lon, lat)')
        self.tchunk = int(nc['u_wind'].chunking()[0]) if nc['u_wind'].chunking() != 'contiguous' else 1

    def describe(self):
        return (f"JCM {self.jcm_dir} ({self.nfiles} files, {str(self.t_first)}..{str(self.t_last)}, "
                f"6-hourly; grid {self.lat.size}x{self.lon.size} Gaussian, {self.nlev_native} hybrid "
                f"levels {self.lev[0]:.3g}..{self.lev[-1]:.0f} hPa)")

    def open_var(self, var):
        if var not in VARMAP:
            raise SystemExit(f'met_jcm: {var!r} is not meteorology JCM provides '
                             f'({", ".join(VARMAP)}); it must come from the BC source')
        return bc_clim.ClimDataset(self, var)

    # ------------------------------------------------------------------
    def _file(self, path):
        nc = self._nc.get(path)
        if nc is None:
            if len(self._nc) >= 4:                 # keep a few handles open
                for p in list(self._nc)[:-3]:
                    self._nc.pop(p).close()
            nc = netCDF4.Dataset(path)
            nc.set_auto_mask(False)
            self._nc[path] = nc
        return nc

    def frame(self, h):
        key = hour_to_datetime64(h)
        if key not in self.index:
            if int(h) % 6:
                raise SystemExit(f'met_jcm: hour {h} ({key}) is not on JCM\'s 6-hourly output grid')
            raise SystemExit(f'met_jcm: hour {h} ({key}) is not in the indexed JCM output '
                             f'({self.t_first}..{self.t_last})')
        return self.index[key]

    def _block(self, path, jvar, i):
        b = i // self.tchunk
        key = (path, jvar, b)
        a = self._cache.get(key)
        if a is None:
            nc = self._file(path)
            v = nc[jvar]
            i0 = b * self.tchunk
            a = np.asarray(v[i0:min(i0 + self.tchunk, v.shape[0])], np.float32)
            self._cache[key] = a
            self._cache_order.append(key)
            while len(self._cache_order) > self._cache_max:
                self._cache.pop(self._cache_order.pop(0), None)
        return a[i - b * self.tchunk]                      # (level, lon, lat), surface-first

    def read(self, var, time, lev, lat):
        if time is None or isinstance(time, slice) or not np.isscalar(time):
            raise SystemExit(f'met_jcm: {var}: one hour per read (got {time!r})')
        path, i = self.frame(int(time))
        a = self._block(path, VARMAP[var], i)              # (L, lon, lat)
        a = a[::-1]                                        # top-first
        if lev is not None:
            a = a[bc_clim.ClimArchive._as_index(lev)]
        a = np.swapaxes(a, -1, -2)                         # (.., lat, lon)
        if lat is not None:
            a = a[..., bc_clim.ClimArchive._as_index(lat), :]
        return np.ascontiguousarray(a)
