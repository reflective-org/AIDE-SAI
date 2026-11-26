#!/usr/bin/env python3
"""Download the weights and input data the linked_* presets need, from Hugging Face, into <repo>/assets/.

    python3 scripts/get_assets.py                 # everything (~3.4 GB)
    python3 scripts/get_assets.py --skip-clim     # weights only (~200 MB)
    python3 scripts/get_assets.py --met --gpu 0   # also build assets/met_P2s200 (offline winds, 92 GB, GPU, ~35 min)

Sources (each file pinned to an HF commit and a sha256 below; a mismatch stops with an error):

    reflective-org/AIDE-Microphysics   tomas40/cs_b/model.pt, tomas40/norm_stats.npz   microphysics student net
    AIDE_HF_REPO (reflective-org/AIDE-SAI-emulator, https://huggingface.co/reflective-org/AIDE-SAI-emulator)
                                       radem_gen2/   radiation emulator (SW, LW, clear-sky damping nets)
                                       atmos_P2s200/ atmosphere emulator P2@200: weights, IC, QBO forcing, fixer
                                       bc_clim/      CESM monthly climatology of the boundary/chemistry fields

What lands in assets/ (the presets find it there by themselves, scripts/presets/_assets.sh):
    tomas_emulator/cs_b/{model.pt,config.json}, tomas_emulator/norm_stats.npz, radem_gen2/, atmos_P2s200/,
    bc_clim/bc_clim.npz, and with --met: met_P2s200/ (regenerated locally from atmos_P2s200).
A private HF repo needs `hf auth login` first.
"""
import argparse, hashlib, os, shutil, subprocess, sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MICRO = ('reflective-org/AIDE-Microphysics', '899026119d190f3893752f0625884e6d2922ab7a')
_SAI_REPO = os.environ.get('AIDE_HF_REPO', 'reflective-org/AIDE-SAI-emulator')   # the pin applies to the default repo only
SAI = (_SAI_REPO, os.environ.get('AIDE_HF_REVISION') or ('80e40014e2e0ba9ff5f65e8960940f415b1429bf' if _SAI_REPO == 'reflective-org/AIDE-SAI-emulator' else None))

# (hf repo, revision, path in the HF repo, path under assets/, sha256)
FILES = [
    (*MICRO, 'tomas40/cs_b/model.pt', 'tomas_emulator/cs_b/model.pt',
     '43603ad2c70ecb66f1c6a75d8dc809678955e999df0c5541084779cb79ae8de0'),
    (*MICRO, 'tomas40/norm_stats.npz', 'tomas_emulator/norm_stats.npz',
     'c7bd095b1fc02438782e5dcbcfff74d7cb853e15996c6bdd930c4ab105ee199d'),
] + [(*SAI, p, p, h) for p, h in [
    ('radem_gen2/sw.npz', 'e908768ea2415f24a410c551480e347d62614c8e47c0bfedb958804921bed7f9'),
    ('radem_gen2/lw.npz', 'cf930a380f1e698ea7917619cf805fdd495062299f8a4b6c2daf8d113a089901'),
    ('radem_gen2/pl.npz', 'f7f68f7d7a9c25088edcb9cb17fee6ff09c04a2f68ccfe99bd474eb380012fa9'),
    ('radem_gen2/sw_args.json', '28bcd8031df8dd51cc393f5db562d84f72570c4bfb4bde69c409557e0f04ecf4'),
    ('radem_gen2/lw_args.json', '0a5f2f6e0a24ea40283b835eb0e15b807171e6c0a4583a23765daa013f1b41b7'),
    ('radem_gen2/pl_args.json', '6fbb0e7bd3656be176719c99eed4c90a49f7317ada0ee4ca7a741915882e30f3'),
    ('atmos_P2s200/net.pt', '505ec9a0649caf02390718c08ae15052cadd6727d4f745a97883e01963b3104d'),
    ('atmos_P2s200/model.json', '51673a96448677c1535e35fdf9ac26875f633e31d5655bf8f67ed3b7ecb10fbe'),
    ('atmos_P2s200/norm.npz', '7ea3ac61b1c8b1d724fa65cb711ce6fa4573534ece9a69f416a98a09cc7ed457'),
    ('atmos_P2s200/ic_1997-01-01T06.npz', '1231d025dfb148c161a78bb085ef3a8d0b2520bb44c49d2e1e80522be76ed885'),
    ('atmos_P2s200/hlr_p.npz', 'd6ea507bbc4a73ea61527106c0e7476a244004aea923f188dc94186a47652b9f'),
    ('atmos_P2s200/fixer_dycore.npz', '9f3b74c423e6c1909c176884609c07a519c005c9c81652507fe8448760a01a87'),
    ('atmos_P2s200/fixer_dycore.json', 'c9f69db2002095b2e83dde7868860e41bae5b42538732d0b0240a219a640be5f'),
    ('atmos_P2s200/jcm_analytic_p20bt.npz', 'a52acf6eb8623d85521217b7598a5c42d748ea1704b3e352794f106cf9ffe97f'),
    ('atmos_P2s200/cesm_grid.npz', 'b581f9c325c2844f972a4345b80c12ee0c46b4ab4c6974e491d0c6b7619652f8'),
    ('bc_clim/bc_clim.npz', '9e1ef7043a9a3bec5894b0eb6f952db7c263d34be32bc40b4a8efadb26b7b867'),
]]

ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument('--dest', default=os.environ.get('ASSETS', os.path.join(REPO, 'assets')))
ap.add_argument('--skip-clim', action='store_true', help='skip bc_clim.npz (3.2 GB)')
ap.add_argument('--met', action='store_true', help='also generate assets/met_P2s200 with src/atmos/atmos_rollout.py (GPU)')
ap.add_argument('--gpu', default='0')
ap.add_argument('--verify-only', action='store_true', help='only check what is already in --dest')
a = ap.parse_args()


def sha256(fn):
    h = hashlib.sha256()
    with open(fn, 'rb') as f:
        for b in iter(lambda: f.read(1 << 24), b''):
            h.update(b)
    return h.hexdigest()


todo = [f for f in FILES if not (a.skip_clim and f[3].startswith('bc_clim/'))]
bad = 0
for repo, rev, src, dst, want in todo:
    out = os.path.join(a.dest, dst)
    if not (os.path.exists(out) and sha256(out) == want) and not a.verify_only:
        from huggingface_hub import hf_hub_download   # pip install huggingface_hub
        print(f'  get      {repo}/{src}', flush=True)
        got = hf_hub_download(repo, src, revision=rev)          # into the HF cache, then copied
        os.makedirs(os.path.dirname(out), exist_ok=True)
        shutil.copyfile(got, out)
    ok = os.path.exists(out) and sha256(out) == want
    bad += not ok
    print(f'  {"ok      " if ok else "BAD     "} {dst}', flush=True)
# the published student checkpoint is {"model": state_dict}; student_lib.load_student reads the architecture here
cfg = os.path.join(a.dest, 'tomas_emulator/cs_b/config.json')
if os.path.isdir(os.path.dirname(cfg)):
    shutil.copyfile(os.path.join(REPO, 'src/microphysics/configs/cs_b.json'), cfg)
if bad:
    sys.exit(f'{bad} file(s) missing or with the wrong sha256 -- delete them and run again')
print(f'{len(todo)} files verified in {a.dest}')

if a.met:
    out = os.path.join(a.dest, 'met_P2s200')
    if os.path.exists(os.path.join(out, 'meta.json')):
        print(f'{out} exists, not regenerating')
    else:
        cmd = [sys.executable, '-u', os.path.join(REPO, 'src/atmos/atmos_rollout.py'), '--bundle',
               os.path.join(a.dest, 'atmos_P2s200'), '--out', out, '--gpu', a.gpu]
        print('generating the offline winds (2 years, 2925 frames):', ' '.join(cmd), flush=True)
        subprocess.run(cmd, check=True)
