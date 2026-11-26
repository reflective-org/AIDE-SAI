"""MET_LIVE=1 (2026-11-16): the MET_DIR is being written WHILE the run reads it, by an atmosphere process stepping in lockstep
(coupled_atm/scripts/atm_server.py). The directory has the MET_SRC=file layout (u/v/omega/T/T_full/ps.npy preallocated to
nframe, meta.json); 'progress' holds the number of frames already written (frames 0..progress-1 are final). Every reader of a
frame calls wait(i) first. RAD_MODE=feedback writes the aerosol heating back as q/q_<frame>.npy (K/s on the band)."""
import os, time
import numpy as np
LIVE = os.environ.get('MET_LIVE', '0') != '0'
TIMEOUT = float(os.environ.get('MET_LIVE_TIMEOUT', '1800'))
_n = {}


def _read_progress(d):
    try:
        with open(os.path.join(d, 'progress')) as f:
            return int(f.read().strip() or 0)
    except (FileNotFoundError, ValueError):
        return 0


def wait(met_dir, i):
    """Block until frame i of a live MET_DIR is written (no-op unless MET_LIVE)."""
    if not LIVE or _n.get(met_dir, 0) > i:
        return
    t0 = time.time(); warned = False
    while True:
        n = _read_progress(met_dir)
        if n > i:
            _n[met_dir] = n
            return
        if os.path.exists(os.path.join(met_dir, 'FAILED')):
            raise SystemExit(f'met_live: the atmosphere process FAILED (see {met_dir}/FAILED) while frame {i} was awaited')
        dt = time.time() - t0
        if dt > 60 and not warned:
            print(f'  [met_live] waiting {dt:.0f}s for frame {i} (progress {n})', flush=True); warned = True
        if dt > TIMEOUT:
            raise SystemExit(f'met_live: frame {i} not written after {TIMEOUT:.0f}s (progress {n})')
        time.sleep(0.002)


def put_q(met_dir, i, q):
    """Hand the atmosphere the heating (K/s, band levels, f09) diagnosed at met frame i."""
    d = os.path.join(met_dir, 'q'); os.makedirs(d, exist_ok=True)
    tmp = os.path.join(d, f'.q_{i:05d}.npy.tmp')
    with open(tmp, 'wb') as f:
        np.save(f, np.asarray(q, np.float32))
    os.replace(tmp, os.path.join(d, f'q_{i:05d}.npy'))
