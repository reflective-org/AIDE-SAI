"""MET_SRC=emu (src/met/met_emu.py) against the offline writer src/atmos/atmos_rollout.py.

Both step the JCM emulator P2@200 from the same bundle, IC and noise seed with the same
code; met_emu.py keeps the regridded frames in memory instead of writing a MET_SRC=file
directory. This checks the in-loop frames are BIT-IDENTICAL to the offline file's for
the first N days (u, v, omega, T on the band, T_full on the radiation column).

  python3 scripts/validation/validate_met_emu.py --bundle B --days 2 --gpu G [--work DIR]
GPU, ~1 min. Exit 1 on any difference. The in-loop side runs with JAX x64 ON, as inside coupling.py.
"""
import argparse, os, subprocess, sys, tempfile
import numpy as np
REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
ap = argparse.ArgumentParser()
ap.add_argument('--bundle', required=True); ap.add_argument('--days', type=int, default=2)
ap.add_argument('--gpu', default='0'); ap.add_argument('--work', default=None); ap.add_argument('--seed', type=int, default=0)
a = ap.parse_args()
os.environ['CUDA_VISIBLE_DEVICES'] = str(a.gpu)
work = a.work or tempfile.mkdtemp(prefix='val_met_emu_')
out = os.path.join(work, 'met_offline')
ic = [f for f in os.listdir(a.bundle) if f.startswith('ic_')][0][3:-4]
subprocess.run([sys.executable, os.path.join(REPO, 'src', 'atmos', 'atmos_rollout.py'), '--bundle', a.bundle,
                '--t0', ic, '--days', str(a.days), '--seed', str(a.seed), '--out', out], check=True,
               env={k: v for k, v in os.environ.items() if k != 'JAX_ENABLE_X64'})
# in-loop conditions: the coupled model runs JAX with x64 on (coupling.py)
import jax; jax.config.update('jax_enable_x64', True)
sys.path.insert(0, os.path.join(REPO, 'src')); import _paths  # noqa: F401
import met_emu
from aide_atmos.met_out import hours_since_archive_t0
nf = 4 * a.days + 1
E = met_emu.EmuMet(a.bundle, hours_since_archive_t0(np.datetime64(ic, 'h')), nf, seed=a.seed)
bad = 0
for k in ('u', 'v', 'omega', 'T', 'T_full'):
    F = np.load(os.path.join(out, f'{k}.npy'), mmap_mode='r')
    for i in range(nf):
        x, y = np.asarray(F[i]), np.asarray(E.met[k][i])
        d = float(np.abs(x.astype(np.float64) - y).max())
        if d != 0.0:
            bad += 1
            print(f'  DIFF {k} frame {i}: max|diff| {d:.3e}')
print(f'met_emu vs atmos_rollout, {nf} frames x 5 fields: ' + ('BIT-IDENTICAL' if bad == 0 else f'{bad} field-frames differ'))
print(f'  in-loop cost: network+physics+fixer {E.t_net / max(E.n_step, 1) * 1e3:.0f} ms/step, '
      f'regrid {E.t_regrid / max(E.n_step + 1, 1) * 1e3:.0f} ms/frame')
sys.exit(1 if bad else 0)
