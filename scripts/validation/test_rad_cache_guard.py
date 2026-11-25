#!/usr/bin/env python3
"""CI gate: the radiation reference cache refuses a live met source.

Under MET_SRC=jcm_online the reference solve depends on the run, not on the
archive hour, so a per-hour cache would serve one run another run's reference
(MANIFEST.md, jcm_online entry, 2026-11-25). coupling.py never attaches one
there; rad_cache.RefCache refuses on its own as well, so no other script can
bring the bug back. rad_cache imports only the standard library and numpy, so
this runs without a GPU or the submodules. Exits 1 on failure.
"""
import os
import sys
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                '..', '..', 'src', 'radiation'))
import rad_cache  # noqa: E402

failed = []
with tempfile.TemporaryDirectory() as root:
    try:
        rad_cache.RefCache(root, {'archive': 'jcm_online:p31_AL7h70'})
        failed.append('RefCache accepted archive jcm_online:p31_AL7h70')
    except ValueError:
        pass
    if os.listdir(root):
        failed.append(f'refused RefCache still created {os.listdir(root)}')
    try:
        c = rad_cache.RefCache(root, {'archive': 'f.e21.FWHIST.cam.h1.T.nc'})
        if not os.path.exists(os.path.join(c.dir, 'key.json')):
            failed.append('archive RefCache wrote no key.json')
    except Exception as e:  # noqa: BLE001
        failed.append(f'RefCache refused an archive source: {e!r}')

if failed:
    print('FAILED:\n  ' + '\n  '.join(failed))
    sys.exit(1)
print('rad_cache guard: live met source refused, archive source accepted -- OK')
