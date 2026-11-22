#!/usr/bin/env python3
"""MET_SRC=jcm_online vs MET_SRC=jcm: does the live JCM stay on the AL7h70 track?

Bit-for-bit agreement with the stored chain is NOT the test, and cannot be:
jcm-strat on GPU is not deterministic run to run (two identical stock
`python -m jcm_strat.main` runs from the 1995 checkpoint differ by 2.8 m/s in u
after 6 h and 70 m/s after 24 h, at single upper-level points; checked
2026-11-18), and the chaotic flow amplifies that. What must hold is that the
large-scale state AIDE sees -- zonal means on the band levels, which the ERA5
and QBO nudging pin -- tracks the stored run far more closely than two
different years of the stored run agree.

For DAYS days from 1 January of YEAR it reads U and T on the band (P_LO..P_HI
hPa) through met_jcm_online.JcmOnlineMet (the same client coupling.py uses) and
through met_jcm.JcmMet (stored output), and prints, per day, the RMS over band
levels x latitudes of the zonal-mean difference, next to the same RMS between
the stored YEAR and the stored YEAR+1 at the same date (the yardstick).

    mkdir -p runs/val_jcm_online && cd runs/val_jcm_online
    python3 ../../scripts/validation/validate_jcm_online.py --days 60

Needs a GPU (the JCM server refuses to run on CPU) and starts the server
exactly as a run of the preset would: the jcm-strat slot, JCM_PY, JCM_CACHE,
JCM_SCRATCH, JCM_DIR and JCM_TEMPLATE are taken from --preset (default
jcm_online_numerical; its #@submodule pins are verified as run_prod.py does),
then from exported variables, then from the flags below, highest last.
Reference (2026-11-18, 60 days from 1996-01-01, 47 band levels 1-138 hPa): RMS
online - stored 0.03 m/s / 0.00 K at day 1, 0.7 m/s / 0.33 K at day 10,
2.1 m/s / 1.2 K at day 60; mean ratio to the year-to-year spread 0.21 (U), 0.34 (T).
Writes jcm_online_validation.npz in the working directory.
"""
import argparse
import os
import sys
import time

import numpy as np

_REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(_REPO, 'src'))
import _paths                      # noqa: E402,F401 -- puts src/ subdirs on sys.path
import met_jcm                     # noqa: E402
import met_jcm_online              # noqa: E402
import run_prod                    # noqa: E402
from pathlib import Path           # noqa: E402

# coupling.py's default JCM_DIR (the AL7h70 chain); not imported from coupling,
# which would pull in JAX, tomas-jax and the radiation stack
CHAIN = '/data/JCM_stripped/jcm-strat/runs/p31/p31_AL7h70'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--year', type=int, default=1996)
    ap.add_argument('--days', type=int, default=30)
    ap.add_argument('--p-lo', type=float, default=1.0)
    ap.add_argument('--p-hi', type=float, default=140.0)
    ap.add_argument('--preset', default='jcm_online_numerical',
                    help="where the JCM settings come from ('none' = env/flags only)")
    ap.add_argument('--no-verify-pins', action='store_true')
    ap.add_argument('--chain', default=None, help='JCM_DIR')
    ap.add_argument('--template', default=None, help='JCM_TEMPLATE')
    ap.add_argument('--jcm-strat', default=None, help='JCM_STRAT_PATH (the jcm-strat checkout)')
    ap.add_argument('--jcm-py', default=None, help='JCM_PY')
    ap.add_argument('--jcm-cache', default=None, help='JCM_CACHE')
    ap.add_argument('--jcm-scratch', default=None, help='JCM_SCRATCH')
    ap.add_argument('--jcm-te-correction', default=None, help='JCM_TE_CORRECTION')
    a = ap.parse_args()
    if os.path.abspath(os.getcwd()).startswith(_REPO) and \
            '/runs/' not in os.path.abspath(os.getcwd()) + '/':
        raise SystemExit('run from a runs/<tag>/ directory (it writes the JCM work files to $PWD)')
    cfg = {}
    if a.preset != 'none':
        vals, pins = run_prod.load_preset(Path(run_prod.preset_path(a.preset)))
        cfg.update(vals)
        cfg.update(run_prod.check_pins(pins, verify=not a.no_verify_pins))
    for k in ('JCM_DIR', 'JCM_TEMPLATE', 'JCM_STRAT_PATH', 'JCM_PY', 'JCM_CACHE',
              'JCM_SCRATCH', 'JCM_TE_CORRECTION'):
        if os.environ.get(k):
            cfg[k] = os.environ[k]
    for k, v in (('JCM_DIR', a.chain), ('JCM_TEMPLATE', a.template),
                 ('JCM_STRAT_PATH', a.jcm_strat), ('JCM_PY', a.jcm_py),
                 ('JCM_CACHE', a.jcm_cache), ('JCM_SCRATCH', a.jcm_scratch),
                 ('JCM_TE_CORRECTION', a.jcm_te_correction)):
        if v:
            cfg[k] = v
    chain = cfg.get('JCM_DIR', CHAIN)
    name = os.path.basename(chain.rstrip('/'))
    tmpl = cfg.get('JCM_TEMPLATE') or f'{chain}/{name}_19960101/.hydra/config.yaml'
    h0 = met_jcm.noleap_hour(a.year, 1, 1)
    h1 = h0 + 24 * a.days
    dy = met_jcm.noleap_hour(a.year + 1, 1, 1) - h0
    on = met_jcm_online.JcmOnlineMet(chain, tmpl, h1, 'jcm_online',
                                     jcm_strat=cfg.get('JCM_STRAT_PATH'),
                                     python=cfg.get('JCM_PY'), cache=cfg.get('JCM_CACHE'),
                                     scratch=cfg.get('JCM_SCRATCH'),
                                     te_correction=cfg.get('JCM_TE_CORRECTION'))
    print(f'online: {on.describe()}', flush=True)
    st = met_jcm.JcmMet(chain, list(range(a.year - 1, a.year + 3)))
    band = np.where((on.lev >= a.p_lo) & (on.lev <= a.p_hi))[0]
    print(f'band: {band.size} levels {on.lev[band[0]]:.3g}..{on.lev[band[-1]]:.0f} hPa', flush=True)
    rows = []
    w0 = time.time()
    for d in range(a.days + 1):
        h = h0 + 24 * d
        r = []
        for var in ('U', 'T'):
            zo = on.read(var, h, band, None).mean(-1)
            zs = st.read(var, h, band, None).mean(-1)
            zn = st.read(var, h + dy, band, None).mean(-1)      # same date next year
            r += [np.sqrt(((zo - zs) ** 2).mean()), np.sqrt(((zn - zs) ** 2).mean())]
        rows.append([d] + r)
        print(f'day {d:3d}  zonal-mean RMS  U: online-stored {r[0]:6.2f} m/s  '
              f'(year-to-year {r[1]:6.2f})   T: online-stored {r[2]:5.2f} K '
              f'(year-to-year {r[3]:5.2f})   [{time.time() - w0:.0f} s]', flush=True)
    rows = np.array(rows)
    np.savez('jcm_online_validation.npz', rows=rows,
             cols=np.array(['day', 'U_online_vs_stored', 'U_year_to_year',
                            'T_online_vs_stored', 'T_year_to_year']),
             band_hpa=on.lev[band], year=a.year)
    ratio_u = rows[1:, 1].mean() / rows[1:, 2].mean()
    ratio_t = rows[1:, 3].mean() / rows[1:, 4].mean()
    print(f'mean ratio online-vs-stored / year-to-year: U {ratio_u:.2f}, T {ratio_t:.2f} '
          f'(well below 1 = on the AL7h70 track)', flush=True)
    on.close()


if __name__ == '__main__':
    main()
