"""Package the code and tables for C3-QRY's Kaggle generation (arm F).

Ships `uie/` at a clean, committed HEAD (with the adopted `uie/controlled.py`),
`c3_common.py`, `free_matched.py` and the adopted tables
(data/state_suffix_tables.json), with `C3_CODE_MANIFEST.json` -- the commit and a
SHA-256 per file, which the notebook checks before it generates anything.

    python research/c3/make_code_bundle.py

Refuses a dirty `uie/` or `research/c3/`, and tables that are not the adopted
file (sha256 in c3_common.TABLES_SHA256).
"""
from __future__ import annotations

import hashlib
import json
import subprocess
import sys
import zipfile
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))
OUT = ROOT / "data" / "kaggle" / "uie-c3-code.zip"
TABLES = ROOT / "data" / "state_suffix_tables.json"


def git(*args: str) -> str:
    return subprocess.run(["git", *args], cwd=ROOT, capture_output=True,
                          text=True, check=True).stdout.strip()


def main() -> None:
    from c3_common import TABLES_SHA256

    dirty = git("status", "--porcelain", "--", "uie", "research/c3")
    if dirty:
        sys.exit(f"refusing: uncommitted changes under uie/ or research/c3/:\n{dirty}")
    if hashlib.sha256(TABLES.read_bytes()).hexdigest() != TABLES_SHA256:
        sys.exit("refusing: data/state_suffix_tables.json is not the adopted file")
    members = {p.relative_to(ROOT).as_posix(): p
               for p in sorted((ROOT / "uie").rglob("*.py")) if "__pycache__" not in p.parts}
    members["c3_common.py"] = HERE / "c3_common.py"
    members["free_matched.py"] = HERE / "free_matched.py"
    members["state_suffix_tables.json"] = TABLES
    manifest = {
        "experiment": "C3-QRY",
        "registration": "research/c3/PREREG_C3_QRY.md",
        "branch": git("rev-parse", "--abbrev-ref", "HEAD"),
        "commit": git("rev-parse", "HEAD"),
        "built": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "attach_with": "the E7 training notebook's output (best.pt); no data bundle",
        "files": {name: hashlib.sha256(p.read_bytes()).hexdigest()
                  for name, p in members.items()},
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(OUT, "w", zipfile.ZIP_DEFLATED) as z:
        for name, p in members.items():
            z.write(p, name)
        z.writestr("C3_CODE_MANIFEST.json", json.dumps(manifest, indent=1))
    print(f"wrote {OUT}  ({OUT.stat().st_size / 1024:.0f} KB, {len(members)} files)")
    print(f"commit {manifest['commit'][:12]} on {manifest['branch']}; "
          f"tables sha256 {TABLES_SHA256[:16]}")


if __name__ == "__main__":
    main()
