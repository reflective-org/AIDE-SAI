#!/usr/bin/env python3
"""CI gate for the linking presets in configs/ (no GPU, no submodules needed).

For every configs/*.env:
  * it parses with run_prod.py's own loader (known knobs only, no forbidden
    keys, well-formed #@submodule lines);
  * every #@submodule PATH that is a registered submodule is pinned to exactly
    the commit the superproject records for it (git ls-tree), so a preset and
    `git submodule update` can never disagree silently;
  * every patch file it names exists;
  * the default preset exists.
"""
import os
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
import run_prod  # noqa: E402

failed = 0


def check(label, ok, detail=""):
    global failed
    print(f"  [{'ok' if ok else 'FAILED'}] {label}" + (f" -- {detail}" if detail and not ok else ""))
    failed += not ok


def gitlink(path):
    out = subprocess.run(["git", "ls-tree", "HEAD", path], cwd=REPO,
                         capture_output=True, text=True).stdout.split()
    return out[2] if len(out) >= 3 and out[1] == "commit" else None


presets = sorted((REPO / "configs").glob("*.env"))
check("configs/ holds at least one preset", bool(presets))
check(f"default preset configs/{run_prod.DEFAULT_PRESET}.env exists",
      (REPO / "configs" / f"{run_prod.DEFAULT_PRESET}.env").exists())
for path in presets:
    print(f"{path.relative_to(REPO)}")
    values, pins = run_prod.load_preset(path)      # exits non-zero on a bad file
    check("parses", True)
    check("sets ADV_VPOS=1", values.get("ADV_VPOS") == "1", f"got {values.get('ADV_VPOS')!r}")
    slots = [p["slot"] for p in pins]
    check("pins every model slot exactly once", sorted(slots) == sorted(run_prod.SLOT_ENV),
          f"slots {slots}")
    for pin in pins:
        link = gitlink(pin["path"])
        check(f"{pin['slot']}: {pin['path']} is a registered submodule", link is not None)
        if link:
            check(f"{pin['slot']}: pin {pin['sha'][:12]} == gitlink {link[:12]}", link == pin["sha"])
        if pin["patch"]:
            check(f"{pin['slot']}: patch {pin['patch']} exists", (REPO / pin["patch"]).is_file())

print("ALL PASSED" if not failed else f"{failed} FAILED")
sys.exit(1 if failed else 0)
