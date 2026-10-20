"""Package the code and tables for SC1's Kaggle generation.

Ships `uie/` at a clean, committed HEAD, `sc1_common.py` (the registered arms,
states and seeds) and `tables.json` (data/sc1/tables.json, built by
build_tables.py), with `SC1_CODE_MANIFEST.json` -- the commit and a SHA-256 for
every file, which the notebook checks before it generates anything. No data
bundle is needed on Kaggle: the tables travel here.

    python research/sc1/make_code_bundle.py

Refuses a dirty `uie/` or `research/sc1/` (a manifest whose commit does not
describe its contents would be a lie), and tables whose row totals are not the
registered ones.
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
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE))
OUT = ROOT / "data" / "kaggle" / "uie-sc1-code.zip"
TABLES = ROOT / "data" / "sc1" / "tables.json"


def git(*args: str) -> str:
    return subprocess.run(["git", *args], cwd=ROOT, capture_output=True,
                          text=True, check=True).stdout.strip()


def main() -> None:
    from build_tables import totals_failures

    dirty = git("status", "--porcelain", "--", "uie", "research/sc1")
    if dirty:
        sys.exit(f"refusing: uncommitted changes under uie/ or research/sc1/:\n{dirty}")
    bad = totals_failures(json.loads(TABLES.read_text(encoding="utf-8")))
    if bad:
        sys.exit("refusing: the tables are not the registered ones:\n  " + "\n  ".join(bad))
    members = {p.relative_to(ROOT).as_posix(): p
               for p in sorted((ROOT / "uie").rglob("*.py")) if "__pycache__" not in p.parts}
    members["sc1_common.py"] = HERE / "sc1_common.py"
    members["tables.json"] = TABLES
    manifest = {
        "experiment": "SC1",
        "registration": "research/sc1/PREREG_SC1.md",
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
        z.writestr("SC1_CODE_MANIFEST.json", json.dumps(manifest, indent=1))
    print(f"wrote {OUT}  ({OUT.stat().st_size / 1024:.0f} KB, {len(members)} files)")
    print(f"commit {manifest['commit'][:12]} on {manifest['branch']}; "
          f"tables sha256 {manifest['files']['tables.json'][:16]}")


if __name__ == "__main__":
    main()
