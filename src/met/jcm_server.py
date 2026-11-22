"""JCM (jcm-strat) as a live meteorology server for MET_SRC=jcm_online.

Runs in jcm-strat's OWN interpreter (its .venv: Python 3.11, JAX 0.10.x, jcm +
dinosaur), never in AIDE's -- the two JAX versions cannot share a process. It
imports nothing from this repo. src/met/met_jcm_online.py spawns it and talks to
it over stdin/stdout; everything the server prints for humans goes to stderr
(and from there to the AIDE log), stdout carries only the protocol.

What it runs. The model that produced the AL7h70 chain
(jcm-strat runs/p31/p31_AL7h70), configured from that chain's own resolved
Hydra config (TEMPLATE: a segment's .hydra/config.yaml), so every term is
active exactly as there: Jucker dry relaxation with the AL7h T_e correction,
QBO nudging to ERA5, ERA5 nudging of u/v/T below 400 hPa (tau 6 h), Hines and
Lott-Miller drag, the sponge, the production tracers, the omega diagnostic,
T63 L95, 12-minute step. Only what the chain changed per segment is set here:
run.start_time, run.total_time, the QBO year and init.file. Like the chain,
the model is built per calendar year (the QBO forcing year and the ERA5 nudging
window are per segment), started from the previous year's end state, which is
written as a jcm checkpoint and read back exactly as the chain's next segment
read its donor.

Protocol (one JSON line per request on stdin, one JSON line per reply on
stdout, optionally followed by raw float32 bytes):
    {"cmd": "hello"}                    -> {"ok": true, grid..., "t0": ISO}
    {"cmd": "frame", "t": "YYYY-MM-DDTHH"}
        -> {"ok": true, "t": ..., "vars": [...], "shape": [L, lon, lat]}
           then len(vars) * L*lon*lat float32 values, C order, in the layout
           of the jcm-strat output files: (level, lon, lat), level 0 = surface.
    {"cmd": "quit"}
The server only moves forward. It keeps the frames of the last KEEP hours and
refuses anything older; a request for a later hour integrates up to it in
6-hour resumes (the chain's save_interval).
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import logging
import os
import sys
import time
from pathlib import Path

import numpy as np

# stdout is the protocol channel: route every print / log of jcm, hydra, jax to stderr
_PROTO = os.fdopen(os.dup(sys.stdout.fileno()), 'wb', buffering=0)
os.dup2(sys.stderr.fileno(), sys.stdout.fileno())
sys.stdout = sys.stderr

STEP_H = 6
VARS = ('u_wind', 'v_wind', 'omega', 'temperature', 'specific_humidity')
KEEP_H = 48


def log(msg):
    print(f'[jcm_server] {msg}', file=sys.stderr, flush=True)


def check_imports(root):
    """Import jcm_strat and jcm and refuse to run unless both come from the
    jcm-strat checkout ROOT (the preset's verified slot, and its external/jax-gcm
    submodule). jcm-strat's venv carries editable installs of both that point at
    whichever checkout the venv was built from; met_jcm_online puts ROOT and
    ROOT/external/jax-gcm first on PYTHONPATH so the slot wins, and this is the
    check that it did."""
    import jcm
    import jcm_strat
    root = os.path.realpath(root)
    for mod, want in ((jcm_strat, root),
                      (jcm, os.path.join(root, 'external', 'jax-gcm'))):
        f = os.path.realpath(mod.__file__)
        log(f'{mod.__name__} from {f}')
        if os.path.commonpath([f, want]) != want:
            raise SystemExit(f'jcm_server: {mod.__name__} imported from {f}, not from the '
                             f'jcm-strat slot {want} (PYTHONPATH={os.environ.get("PYTHONPATH")})')


def days_in_year(y):
    return (dt.date(y + 1, 1, 1) - dt.date(y, 1, 1)).days


class Server:
    def __init__(self, a):
        from omegaconf import OmegaConf
        self.a = a
        self.tmpl = OmegaConf.load(a.template)
        self.work = Path(a.workdir)
        self.work.mkdir(parents=True, exist_ok=True)
        self.model = None
        self.year = None
        self.t = None                      # datetime of the model's current state
        self.frames = {}                   # datetime -> {var: (L, lon, lat) float32}
        self._installed = False

    # ---------------------------------------------------------------- setup
    def _install_policies(self, cfg):
        """What jcm_strat.main.main does before handing the config to jcm."""
        if self._installed:
            return
        self._installed = True
        from omegaconf import OmegaConf
        from jcm_strat import levels, main as jm, nudging_lat
        jm.install_health_policy(bool(cfg.get('health_ignore_q_max', False)))
        raw = cfg.get('sl_mass_fixer_exclude', None)
        exclude = list(OmegaConf.to_container(raw)) if raw is not None else []
        jm.install_mass_fixer_policy(bool(cfg.get('sl_mass_fixer', True)), exclude)
        di = cfg.get('sl_departure_iterations', None)
        jm.install_sl_options({'departure_iterations': int(di) if di is not None else None})
        jm.install_calendar(cfg.get('calendar', None))
        nudging_lat.install_lat_top(cfg.get('nudging', {}).get('lat_top', None))
        jm.check_jcm3_run_keys(cfg)
        jm.log_dot_precision()
        levels.install(cfg.get('level_table', None))

    def _cfg_for(self, year, init_file):
        from omegaconf import OmegaConf
        cfg = OmegaConf.create(OmegaConf.to_container(self.tmpl, resolve=True))
        cfg.run.start_time = f'{year:04d}-01-01'
        cfg.run.total_time = days_in_year(year)
        cfg.run.checkpoint_path = None
        cfg.run.chunk_days = 0
        cfg.init.kind = 'from_state'
        cfg.init.file = str(init_file)
        hs = cfg.physics.terms.held_suarez
        hs.qbo.year = year
        if self.a.te_correction:
            hs.te_correction_file = self.a.te_correction
        for k in ('te_correction_file',):
            p = hs.get(k)
            if p and not Path(p).exists():
                raise SystemExit(f'jcm_server: {k}={p} does not exist (pass --te-correction)')
        return cfg

    def _build_year(self, year, init_file):
        import jax
        from jcm import runners
        cfg = self._cfg_for(year, init_file)
        self._install_policies(cfg)
        t0 = time.perf_counter()
        runners._apply_log_level(cfg)
        runners.maybe_enable_compilation_cache()
        runners.apply_constants_overrides(cfg)
        self.model = None
        model = runners.build_model(cfg)
        forcing = runners.build_forcing(cfg, model.coords, dycore=getattr(model, 'dycore', None))
        # loaded onto the CPU device: a GPU copy, even freed at once, would stay
        # in the allocator's pool for the life of the process
        with jax.default_device(jax.devices('cpu')[0]):
            forcing = runners._maybe_attach_nudging_target(forcing, cfg, model)
        forcing = self._host_nudging(forcing)
        state, carry = runners._state_from_file(model, cfg)
        self.model, self.forcing, self.cfg = model, forcing, cfg
        self._pending = (state, carry)
        self.year = year
        self.t = dt.datetime(year, 1, 1)
        log(f'built {year}: init {init_file}; {time.perf_counter() - t0:.0f} s; '
            f'devices {jax.devices()}')
        if jax.devices()[0].platform != 'gpu':
            # JAX falls back to CPU silently when it cannot load libcuda.so.1;
            # a CPU JCM would turn a run into weeks
            raise SystemExit(f'jcm_server: no GPU (devices {jax.devices()}); '
                             'libcuda.so.1 not on LD_LIBRARY_PATH?')
        if not hasattr(self, 'grid'):
            self._grid_from_template()

    # The ERA5 nudging target for the whole year (+-1 day) is ~10 GB per field
    # at T63 L95 6-hourly, and jcm keeps it on the device: the compiled step
    # wants ~30 GB more for it, which on a card shared with the aerosol model
    # does not fit. Each 6-hour resume only looks up samples between its two
    # ends (BY_DATE, 'pad'), so the server keeps the year on the HOST and hands
    # every resume a fixed NWIN-sample slice starting one sample before the
    # resume -- the same samples, so the same lookups; a fixed length, so one
    # compile.
    NWIN = 4

    def _host_nudging(self, forcing):
        import jax
        from jcm.forcing import TimeSeries
        nt = getattr(forcing, 'nudging_target', None)
        if nt is None:
            self._nt = None
            return forcing

        def to_host(leaf):
            if isinstance(leaf, TimeSeries):
                return TimeSeries(np.asarray(leaf.values), leaf.times,
                                  leaf.align_mode, leaf.persist)
            return leaf
        nt = jax.tree_util.tree_map(to_host, nt, is_leaf=lambda x: isinstance(x, TimeSeries))
        ts0 = [l for l in jax.tree_util.tree_leaves(nt, is_leaf=lambda x: isinstance(x, TimeSeries))
               if isinstance(l, TimeSeries)][0]
        d = np.asarray(ts0.times.delta.days, np.int64)
        sec = np.asarray(ts0.times.delta.seconds, np.int64)
        self._nt_times = (np.datetime64('1970-01-01T00', 's') + (d * 86400 + sec).astype('timedelta64[s]'))
        self._nt = nt
        nbytes = sum(np.asarray(l.values).nbytes for l in jax.tree_util.tree_leaves(
            nt, is_leaf=lambda x: isinstance(x, TimeSeries)) if isinstance(l, TimeSeries))
        log(f'nudging target on the host: {len(self._nt_times)} samples '
            f'{self._nt_times[0]}..{self._nt_times[-1]}, {nbytes / 2**30:.1f} GiB; '
            f'{self.NWIN}-sample window per resume')
        return forcing.copy(nudging_target=None)

    def _forcing_now(self):
        if self._nt is None:
            return self.forcing
        import jax
        import jax.numpy as jnp
        from jcm.forcing import TimeSeries
        t = np.datetime64(self.t, 's')
        i = int(np.searchsorted(self._nt_times, t, side='right')) - 1   # 'pad' sample of t
        i0 = max(i - 1, 0)
        i1 = i0 + self.NWIN
        if i < 0 or i1 > len(self._nt_times) or \
                self._nt_times[i1 - 1] < t + np.timedelta64(STEP_H * 3600, 's'):
            raise SystemExit(f'jcm_server: nudging window {i0}:{i1} does not cover {t} + {STEP_H} h')

        def cut(leaf):
            if not isinstance(leaf, TimeSeries):
                return leaf
            tm = jax.tree_util.tree_map(lambda x: x[i0:i1], leaf.times)
            return TimeSeries(jnp.asarray(leaf.values[i0:i1]), tm, leaf.align_mode, leaf.persist)
        nt = jax.tree_util.tree_map(cut, self._nt, is_leaf=lambda x: isinstance(x, TimeSeries))
        return self.forcing.copy(nudging_target=nt)

    def _grid_from_template(self):
        """lat/lon/hybrid coefficients exactly as the output files carry them."""
        import glob
        import netCDF4
        seg = Path(self.a.template).parent.parent
        f = sorted(glob.glob(str(seg / 'longrun_day*.nc')))[0]
        with netCDF4.Dataset(f) as nc:
            self.grid = {k: np.asarray(nc[k][:], np.float64).tolist()
                         for k in ('lat', 'lon', 'hybrid_a_full', 'hybrid_b_full',
                                   'hybrid_a_half', 'hybrid_b_half')}

    # ------------------------------------------------------------- stepping
    def _store(self, when, ds):
        fr = {v: np.ascontiguousarray(ds[v].transpose('level', 'lon', 'lat').values,
                                      np.float32) for v in VARS}
        self.frames[when] = fr
        for k in [k for k in self.frames if k < when - dt.timedelta(hours=KEEP_H)]:
            del self.frames[k]

    def _advance(self):
        """Integrate STEP_H hours and store the frame at the new time."""
        secs = f'{STEP_H * 3600} seconds'
        kw = dict(forcing=self._forcing_now(), save_interval=self.cfg.run.save_interval,
                  total_time=secs, output_averages=False)
        if self._pending is not None:
            st, ca = self._pending
            self._pending = None
            preds = self.model.run(initial_state=st, initial_physics_state=ca, **kw)
        else:
            preds = self.model.resume(**kw)
        ds = preds.to_xarray()
        self.t = self.t + dt.timedelta(hours=STEP_H)
        last = ds.isel(time=-1)
        tt = np.datetime64(last['time'].values, 'h') if np.issubdtype(
            last['time'].dtype, np.datetime64) else None
        if tt is not None and tt != np.datetime64(self.t, 'h'):
            raise SystemExit(f'jcm_server: model frame time {tt} != expected {self.t}')
        self._store(self.t, last)

    def _roll_year(self):
        """End of a calendar year: write the state as a checkpoint and rebuild the
        next year's model from it -- what chain_years.sh does between segments."""
        from jcm.checkpoint import save_checkpoint
        ck = self.work / f'jcm_{self.t.year:04d}0101.ckpt'
        save_checkpoint(self.model, str(ck), keep_previous=False, run_state=self.model.run_state)
        # the frame at that instant beside it: a later start from this
        # checkpoint cannot produce it without stepping (see start())
        np.savez(str(ck) + '.frame0.npz', time=np.datetime64(self.t, 'h'), **self.frames[self.t])
        log(f'year end: wrote {ck}')
        self._build_year(self.t.year, ck)

    def frame(self, when):
        if when in self.frames:
            return self.frames[when]
        if self.t is not None and when < self.t:
            raise SystemExit(f'jcm_server: {when} is behind the model ({self.t}); '
                             f'only the last {KEEP_H} h are kept')
        if when.hour % STEP_H or when.minute:
            raise SystemExit(f'jcm_server: {when} is not on the {STEP_H}-hourly grid')
        w0 = time.perf_counter(); n = 0
        while self.t < when:
            if self.t.month == 1 and self.t.day == 1 and self.t.hour == 0 \
                    and self.t.year != self.year:
                self._roll_year()
            self._advance(); n += 1
        if n > 4:
            log(f'advanced {n * STEP_H} h to {when} in {time.perf_counter() - w0:.0f} s')
        if when not in self.frames:
            raise SystemExit(f'jcm_server: no frame at {when} (model at {self.t})')
        return self.frames[when]

    def start(self, year, init_file, frame0):
        self._build_year(year, init_file)
        if frame0 and frame0.endswith('.npz'):
            z = np.load(frame0)
            if np.datetime64(z['time'], 'h') != np.datetime64(self.t, 'h'):
                raise SystemExit(f'jcm_server: {frame0} holds {z["time"]}, not {self.t}')
            self.frames[self.t] = {v: np.asarray(z[v], np.float32) for v in VARS}
            log(f'frame at {self.t} from {frame0}')
        elif frame0:
            # the frame AT the init instant is the donor run's last output frame
            # (same state); the online model cannot emit it without stepping
            import netCDF4
            with netCDF4.Dataset(frame0) as nc:
                nc.set_auto_mask(False)
                tv = nc['time']
                import cftime
                ts = cftime.num2date(tv[:], tv.units, getattr(tv, 'calendar', 'standard'),
                                     only_use_cftime_datetimes=False,
                                     only_use_python_datetimes=True)
                i = [j for j, t in enumerate(ts) if t == self.t]
                if not i:
                    raise SystemExit(f'jcm_server: {frame0} has no frame at {self.t}')
                dims = nc['u_wind'].dimensions
                if dims != ('time', 'level', 'lon', 'lat'):
                    raise SystemExit(f'jcm_server: {frame0} layout {dims}')
                self.frames[self.t] = {v: np.asarray(nc[v][i[0]], np.float32) for v in VARS}
            log(f'frame at {self.t} from {frame0}')


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--template')
    p.add_argument('--year', type=int)
    p.add_argument('--init', help='jcm checkpoint holding YEAR-01-01 00Z')
    p.add_argument('--frame0', default='', help='output file holding the YEAR-01-01 00Z frame')
    p.add_argument('--workdir')
    p.add_argument('--te-correction', default='')
    p.add_argument('--jcm-strat', default='',
                   help='the jcm-strat checkout jcm_strat and jcm must be imported from')
    p.add_argument('--check-imports', action='store_true',
                   help='only check (and print) where jcm_strat and jcm come from, then exit')
    a = p.parse_args()
    logging.basicConfig(stream=sys.stderr, level=logging.INFO)
    if a.jcm_strat:
        check_imports(a.jcm_strat)
    if a.check_imports:
        return
    missing = [k for k in ('template', 'year', 'init', 'workdir') if getattr(a, k) is None]
    if missing:
        p.error('required: ' + ', '.join('--' + k for k in missing))
    s = Server(a)
    s.start(a.year, a.init, a.frame0)

    def reply(obj, payload=None):
        _PROTO.write((json.dumps(obj) + '\n').encode())
        if payload is not None:
            for arr in payload:
                _PROTO.write(memoryview(arr).cast('B'))

    for line in sys.stdin.buffer:
        req = json.loads(line)
        cmd = req.get('cmd')
        try:
            if cmd == 'hello':
                reply({'ok': True, 'grid': s.grid, 't0': s.t.isoformat(), 'vars': VARS})
            elif cmd == 'frame':
                when = dt.datetime.fromisoformat(req['t'])
                fr = s.frame(when)
                shape = list(fr[VARS[0]].shape)
                reply({'ok': True, 't': when.isoformat(), 'vars': VARS, 'shape': shape},
                      [fr[v] for v in VARS])
            elif cmd == 'quit':
                reply({'ok': True}); break
            else:
                reply({'ok': False, 'error': f'unknown cmd {cmd!r}'})
        except SystemExit as e:
            reply({'ok': False, 'error': str(e)})
        except Exception as e:              # noqa: BLE001 -- report, then die
            import traceback
            traceback.print_exc()
            reply({'ok': False, 'error': f'{type(e).__name__}: {e}'})
            raise


if __name__ == '__main__':
    main()
