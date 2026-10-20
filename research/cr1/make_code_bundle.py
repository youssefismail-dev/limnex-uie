"""Package the current `uie` code for CR1's Kaggle generation.

The E7 code package (`uie-e7-code`) predates `uie/presence.py`, so CR1 ships
its own: the `uie/` package at a clean, committed HEAD, plus
`CR1_CODE_MANIFEST.json` -- the commit and a SHA-256 for every file, which the
notebook checks before generating anything. The data bundle is not re-uploaded.

    python research/cr1/make_code_bundle.py

Refuses to build from a dirty tree: a manifest whose commit does not describe
its contents would be a lie.
"""
from __future__ import annotations

import hashlib
import json
import subprocess
import sys
import zipfile
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "data" / "kaggle" / "uie-cr1-code.zip"


def git(*args: str) -> str:
    return subprocess.run(["git", *args], cwd=ROOT, capture_output=True,
                          text=True, check=True).stdout.strip()


def main() -> None:
    dirty = git("status", "--porcelain", "--", "uie")
    if dirty:
        sys.exit(f"refusing: uncommitted changes under uie/:\n{dirty}")
    files = sorted(p for p in (ROOT / "uie").rglob("*.py")
                   if "__pycache__" not in p.parts)
    manifest = {
        "experiment": "CR1",
        "registration": "research/cr1/PREREG_CR1.md",
        "branch": git("rev-parse", "--abbrev-ref", "HEAD"),
        "commit": git("rev-parse", "HEAD"),
        "built": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "attach_with": "uie-corpus (the data bundle, unchanged) and the E7 "
                       "training notebook's output (best.pt)",
        "files": {p.relative_to(ROOT).as_posix():
                  hashlib.sha256(p.read_bytes()).hexdigest() for p in files},
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(OUT, "w", zipfile.ZIP_DEFLATED) as z:
        for p in files:
            z.write(p, p.relative_to(ROOT).as_posix())
        z.writestr("CR1_CODE_MANIFEST.json", json.dumps(manifest, indent=1))
    print(f"wrote {OUT}  ({OUT.stat().st_size / 1024:.0f} KB, {len(files)} files)")
    print(f"commit {manifest['commit'][:12]} on {manifest['branch']}")


if __name__ == "__main__":
    main()
