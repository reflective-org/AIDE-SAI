"""MET_SRC=jcm_online: the JCM model that produced AL7h, RUN LIVE as the meteorology.

Why this exists (2026-11-18). MET_SRC=jcm (met_jcm.py) reads
the AL7h70 chain's stored 6-hourly output. This source serves the same fields
on the same grid through the same open_var surface, but integrates jcm-strat
itself, step by step, alongside the aerosol model: Jucker relaxation + AL7h
T_e correction, QBO nudging, ERA5 nudging below 400 hPa, Hines + Lott-Miller
drag, sponge, production tracers -- every term of the AL7h configuration,
active (jcm_server.py lists them). Today it is still one-way (the circulation
never sees the aerosol); the point of the seam is that it no longer has to be:
a heating term handed back to the server closes the loop, which stored output
never can.

Process layout. jcm-strat needs its own interpreter (Python 3.11, JAX 0.10)
and AIDE runs on another (JAX 0.6), so JCM runs in a SUBPROCESS
(src/met/jcm_server.py under JCM_PY) on the same GPU, and frames cross a pipe:
~28 MB per 6-hourly frame (five L95 fields), negligible next to a step. The
subprocess is started lazily on the first read and stopped at exit.

Where jcm-strat comes from. The CODE is the jcm-strat model slot
(JCM_STRAT_PATH, models/jcm-strat by default: the preset's #@submodule pin) and
its nested jax-gcm submodule (external/jax-gcm, which provides `jcm`). The
server's PYTHONPATH is exactly those two directories, so it imports jcm_strat
and jcm from the verified checkout and not from the venv's own (editable)
installs, and it asserts that at startup. What a checkout cannot hold is
machine state, and comes from knobs: the interpreter (JCM_PY, a venv with
jcm-strat's dependencies), the ERA5/boundary-file caches (JCM_CACHE) and the
JAX compile cache (JCM_SCRATCH). The AL7h T_e correction is a tracked file of
the slot.

Start. The server is built for the calendar year of the run's first hour from
a jcm checkpoint holding YYYY-01-01 00Z: this run's own (JCM_WORK/jcm_YYYY0101
.ckpt, written by the server at every year end, so a RESUME continues the
online trajectory) if present, else the AL7h70 chain's (JCM_DIR/*_<YYYY-1>0101/
checkpoint.ckpt, the state the chain's YYYY segment started from). A start
hour inside the year is reached by integrating forward from 1 January
(~12 min per simulated year on an H100). The 00Z frame at the init instant
cannot be produced without stepping; it is the donor run's last output frame
(the same state) and is read from there.

Grid, levels, layout, calendar mapping (noleap model hours -> the Gregorian
date with the same Y-M-D-H, 29 Feb skipped), the dry-q caveat (MET_Q): all as
met_jcm.py, which this subclasses -- only where a frame comes from differs.
"""
import atexit
import glob
import json
import os
import subprocess

import numpy as np

import bc_clim
import met_jcm

HERE = os.path.dirname(os.path.abspath(__file__))
# the jcm-strat slot's default checkout (run_prod.py sets JCM_STRAT_PATH from
# the preset's #@submodule line)
DEFAULT_JCM_STRAT = os.path.join(os.path.dirname(os.path.dirname(HERE)), 'models', 'jcm-strat')
# the AL7h T_e correction, relative to the jcm-strat checkout (a tracked file
# there): the chain's config names it under a jcm-strat worktree that has since
# been removed
TE_CORRECTION_REL = 'docs/outputs/25_al7_half/corrections/AL7h.nc'
# jax-gcm (the `jcm` package), a submodule of jcm-strat
JAX_GCM_REL = 'external/jax-gcm'


def _libcuda_dir():
    """The directory holding libcuda.so.1 (run_prod.find_libcuda's search: the
    usual driver mount, else the first hit under /run/nvidia or /usr/lib)."""
    cand = '/run/nvidia/driver/usr/lib/x86_64-linux-gnu'
    if os.path.exists(os.path.join(cand, 'libcuda.so.1')):
        return cand
    from pathlib import Path
    for root in ('/run/nvidia', '/usr/lib'):
        if os.path.isdir(root):
            for hit in Path(root).rglob('libcuda.so.1'):
                return str(hit.parent)
    print('  met_jcm_online: WARNING libcuda.so.1 not found; the JCM server will '
          'refuse to start on CPU', flush=True)
    return None


class JcmOnlineMet(met_jcm.JcmMet):
    """open_var() surface over a live jcm-strat integration.

    h_first, h_last: model hours (noleap, bc_clim.ORIGIN) of the first and last
    frame the run will read; the hour axis spans up to h_last."""

    def __init__(self, chain_dir, template, h_last, workdir, jcm_strat=None,
                 python=None, cache=None, scratch=None, te_correction=None, keep=8):
        self.jcm_dir = chain_dir
        self.workdir = os.path.abspath(workdir)
        self.template = template
        self.jcm_strat = os.path.abspath(jcm_strat or os.environ.get('JCM_STRAT_PATH')
                                         or DEFAULT_JCM_STRAT)
        for sub in ('jcm_strat', os.path.join(JAX_GCM_REL, 'jcm')):
            if not os.path.isdir(os.path.join(self.jcm_strat, sub)):
                raise SystemExit(f'met_jcm_online: {self.jcm_strat} has no {sub}/ -- the '
                                 f'jcm-strat slot (JCM_STRAT_PATH) is not a recursively '
                                 f'initialised jcm-strat checkout: git submodule update '
                                 f'--init --recursive models/jcm-strat')
        self.python = python or os.path.join(self.jcm_strat, '.venv/bin/python')
        if not os.path.exists(self.python):
            raise SystemExit(f'met_jcm_online: no jcm-strat interpreter at {self.python} '
                             f'(JCM_PY: a Python 3.11 venv with jcm-strat\'s dependencies)')
        self.cache = cache or os.path.join(self.jcm_strat, 'cache')
        self.scratch = scratch or os.path.join(self.jcm_strat, 'scratch')
        self.te_correction = te_correction or os.path.join(self.jcm_strat, TE_CORRECTION_REL)
        self.source = f'jcm_online:{os.path.basename(os.path.normpath(chain_dir))}'
        self.clim_file = self.source
        # the grid is declared before the server exists (main() needs it at
        # startup): from the template segment's own output, which is what the
        # server's model writes too
        f = sorted(glob.glob(os.path.join(os.path.dirname(os.path.dirname(template)),
                                          'longrun_day*.nc')))
        if not f:
            raise SystemExit(f'met_jcm_online: no longrun_day*.nc beside {template} (JCM_TEMPLATE)')
        import netCDF4
        with netCDF4.Dataset(f[0]) as nc:
            self._grid_from(nc)
        self.ntime = int(h_last) + 1
        self.t_last = met_jcm.hour_to_datetime64(h_last)
        self.nfiles = 0
        self.year = self.init = None
        self._frames = {}
        self._order = []
        self._keep = int(keep)
        self._p = None

    # ------------------------------------------------------------------
    def _init_for(self, year):
        own = os.path.join(self.workdir, f'jcm_{year:04d}0101.ckpt')
        if os.path.exists(own):
            return own, own + '.frame0.npz'
        segs = sorted(glob.glob(os.path.join(self.jcm_dir, f'*_{year - 1:04d}0101')))
        if not segs:
            raise SystemExit(f'met_jcm_online: no {year}-01-01 state: neither {own} nor a '
                             f'{year - 1} segment under {self.jcm_dir} (JCM_DIR)')
        ck = os.path.join(segs[0], 'checkpoint.ckpt')
        days = sorted(glob.glob(os.path.join(segs[0], 'longrun_day*.nc')),
                      key=lambda f: int(f.rsplit('day', 1)[1].split('.')[0]))
        if not os.path.exists(ck) or not days:
            raise SystemExit(f'met_jcm_online: {segs[0]} lacks checkpoint.ckpt or its output')
        return ck, days[-1]

    def _start(self, h):
        """Start the server in the calendar year of hour h, the first one read
        (H0, or the restart hour of a RESUME)."""
        self.year = met_jcm.hour_to_datetime64(h).astype(object).year
        os.makedirs(self.workdir, exist_ok=True)
        self.init, self.frame0 = self._init_for(self.year)
        env = dict(os.environ)
        # jcm-strat's environment (scripts/env.sh), minus its whole-card memory grab:
        # the server shares the card with this process
        env.update(JCM_STRAT_REPO=self.jcm_strat,
                   HF_HOME=os.path.join(self.cache, 'huggingface'),
                   JCM_ERA5_CACHE=os.path.join(self.cache, 'era5'),
                   SCRATCH=self.scratch,
                   XLA_PYTHON_CLIENT_PREALLOCATE='false')
        env.pop('XLA_PYTHON_CLIENT_MEM_FRACTION', None)
        # jcm_strat and jcm from the slot checkout, ahead of the venv's own
        # (editable) installs; nothing of AIDE's src/ on the server's path.
        # jcm_server.py asserts where both were imported from.
        env['PYTHONPATH'] = os.pathsep.join([self.jcm_strat,
                                             os.path.join(self.jcm_strat, JAX_GCM_REL)])
        for k in ('JAX_ENABLE_X64', 'XLA_FLAGS', 'PYTHONHOME'):
            env.pop(k, None)
        # libcuda.so.1 under the containerised driver mount is off the loader
        # path; without it the server's JAX SILENTLY runs on CPU. A child's
        # LD_LIBRARY_PATH is read at its own start, so setting it here works
        # even when this process was not launched through run_prod.py.
        d = _libcuda_dir()
        if d:
            cur = env.get('LD_LIBRARY_PATH', '')
            if d not in cur.split(':'):
                env['LD_LIBRARY_PATH'] = d + (':' + cur if cur else '')
        cmd = [self.python, '-u', os.path.join(HERE, 'jcm_server.py'),
               '--template', self.template, '--year', str(self.year),
               '--init', self.init, '--frame0', self.frame0,
               '--workdir', self.workdir, '--jcm-strat', self.jcm_strat]
        if self.te_correction:
            cmd += ['--te-correction', self.te_correction]
        print(f'  met_jcm_online: starting {" ".join(cmd)}', flush=True)
        self._p = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                   stderr=None, env=env, cwd=self.workdir)
        atexit.register(self.close)

    def close(self):
        p, self._p = self._p, None
        if p is None or p.poll() is not None:
            return
        try:
            p.stdin.write(b'{"cmd": "quit"}\n'); p.stdin.flush()
            p.wait(timeout=60)
        except Exception:                    # noqa: BLE001 -- shutting down anyway
            p.kill()

    def _request(self, obj):
        p = self._p
        if p is None or p.poll() is not None:
            raise SystemExit(f'met_jcm_online: the JCM server is not running '
                             f'(exit {None if p is None else p.returncode}); see its log above')
        p.stdin.write((json.dumps(obj) + '\n').encode()); p.stdin.flush()
        line = p.stdout.readline()
        if not line:
            raise SystemExit(f'met_jcm_online: the JCM server died (exit {p.wait()})')
        h = json.loads(line)
        if not h.get('ok'):
            raise SystemExit(f'met_jcm_online: server refused {obj}: {h.get("error")}')
        arrs = {}
        if 'shape' in h:
            n = int(np.prod(h['shape']))
            for v in h['vars']:
                buf = p.stdout.read(4 * n)
                if len(buf) != 4 * n:
                    raise SystemExit('met_jcm_online: short read from the JCM server')
                arrs[v] = np.frombuffer(buf, np.float32).reshape(h['shape'])
        return h, arrs

    def describe(self):
        start = (f"started {self.year}-01-01 00Z from {self.init}" if self.year else
                 "starts on the first read, from 1 January of that year")
        return (f"JCM ONLINE (live jcm-strat {self.jcm_strat}, config {self.template}, "
                f"{self.python}, cache {self.cache}, scratch {self.scratch}); {start}; grid {self.lat.size}x{self.lon.size} "
                f"Gaussian, {self.nlev_native} hybrid levels {self.lev[0]:.3g}..{self.lev[-1]:.0f} hPa; "
                f"server workdir {self.workdir}")

    # ------------------------------------------------------------------
    def frame(self, h):
        key = met_jcm.hour_to_datetime64(h)
        if int(h) % 6:
            raise SystemExit(f"met_jcm_online: hour {h} ({key}) is not on JCM's 6-hourly grid")
        return key

    def read(self, var, time, lev, lat):
        if time is None or isinstance(time, slice) or not np.isscalar(time):
            raise SystemExit(f'met_jcm_online: {var}: one hour per read (got {time!r})')
        key = self.frame(int(time))
        fr = self._frames.get(key)
        if fr is None:
            if self._p is None:
                self._start(int(time))
                h, _ = self._request({'cmd': 'hello'})
                g = h['grid']
                for k, mine in (('lat', self.lat), ('lon', self.lon)):
                    if not np.allclose(np.asarray(g[k]), mine):
                        raise SystemExit(f'met_jcm_online: server {k} differs from the template output')
                if not np.allclose(np.asarray(g['hybrid_a_full'])[::-1] / met_jcm.P0, self.hyam):
                    raise SystemExit('met_jcm_online: server levels differ from the template output')
                print(f'  met_jcm_online: server up, model at {h["t0"]}; '
                      f'{self.describe()}', flush=True)
            _, fr = self._request({'cmd': 'frame', 't': str(key) + ':00'})
            self._frames[key] = fr
            self._order.append(key)
            while len(self._order) > self._keep:
                self._frames.pop(self._order.pop(0), None)
        a = fr[met_jcm.VARMAP[var]][::-1]                   # (L, lon, lat), top-first
        if lev is not None:
            a = a[bc_clim.ClimArchive._as_index(lev)]
        a = np.swapaxes(a, -1, -2)
        if lat is not None:
            a = a[..., bc_clim.ClimArchive._as_index(lat), :]
        return np.ascontiguousarray(a)
