"""Package the E7 `uie` code as a small Kaggle dataset.

The data bundle (`uie-corpus2`) is NOT re-uploaded -- its streams are unchanged,
because presence is derived from each URL at encode time. Only the code moves,
as a separate dataset attached beside it. The manifest records the git commit
and a hash of every file, so the notebook can prove it is running exactly this
code and not the older `uie` the data bundle carries.

    python research/e7/make_code_bundle.py

Refuses to build from a dirty tree: a bundle whose commit does not describe its
contents would make the manifest a lie.
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
OUT = ROOT / "data" / "kaggle" / "uie-e7-code.zip"


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
        "experiment": "E7",
        "registration": "research/e7/PREREG_E7.md",
        "branch": git("rev-parse", "--abbrev-ref", "HEAD"),
        "commit": git("rev-parse", "HEAD"),
        "built": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "attach_with": "uie-corpus2 (the data bundle, unchanged)",
        "files": {p.relative_to(ROOT).as_posix():
                  hashlib.sha256(p.read_bytes()).hexdigest() for p in files},
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(OUT, "w", zipfile.ZIP_DEFLATED) as z:
        for p in files:
            z.write(p, p.relative_to(ROOT).as_posix())
        z.writestr("E7_CODE_MANIFEST.json", json.dumps(manifest, indent=1))
    print(f"wrote {OUT}  ({OUT.stat().st_size / 1024:.0f} KB, {len(files)} files)")
    print(f"commit {manifest['commit'][:12]} on {manifest['branch']}")


if __name__ == "__main__":
    main()
