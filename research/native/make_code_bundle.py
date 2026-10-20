"""Package the E8 code (and the two suffix tables generation needs) as a Kaggle
dataset, `uie-e8-code`.

The data bundle (`uie-corpus2`) is not re-uploaded. The manifest records the git
commit and a hash of every file, so each notebook proves it runs exactly this
code -- not the older `uie` inside the data bundle.

    python research/native/make_code_bundle.py

Refuses a dirty tree under uie/ or research/native/, and refuses tables whose
hash is not the registered one.
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
OUT = ROOT / "data" / "kaggle" / "uie-e8-code.zip"
TABLES = {
    "suffix_table.json": (ROOT / "data" / "bundle2" / "suffix_table.json",
                          "aaa6fb853ea538d41ba4c10661b89faad2a687b869208772c096c7e507ee866d"),
    "state_suffix_tables.json": (ROOT / "data" / "state_suffix_tables.json",
                                 "0c6f2760c033c532d271b360a14f2976139d5957e4aaf64977c8746504213cc9"),
}


def git(*args: str) -> str:
    return subprocess.run(["git", *args], cwd=ROOT, capture_output=True,
                          text=True, check=True).stdout.strip()


def sha256(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main() -> None:
    dirty = git("status", "--porcelain", "--", "uie", "research/native")
    if dirty:
        sys.exit(f"refusing: uncommitted changes:\n{dirty}")
    for name, (path, want) in TABLES.items():
        if sha256(path) != want:
            sys.exit(f"refusing: {path} is not the registered {name} ({want[:12]}...)")
    files = sorted(p for p in (ROOT / "uie").rglob("*.py") if "__pycache__" not in p.parts)
    files.append(ROOT / "research" / "native" / "e8_common.py")   # the shared generation calls
    manifest = {
        "experiment": "E8 (N1)",
        "registration": "research/native/PREREG_E8.md",
        "branch": git("rev-parse", "--abbrev-ref", "HEAD"),
        "commit": git("rev-parse", "HEAD"),
        "built": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "attach_with": "uie-corpus2 (the data bundle, unchanged)",
        "files": {p.relative_to(ROOT).as_posix(): sha256(p) for p in files},
        "tables": {name: want for name, (_, want) in TABLES.items()},
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(OUT, "w", zipfile.ZIP_DEFLATED) as z:
        for p in files:
            z.write(p, p.relative_to(ROOT).as_posix())
        for name, (path, _) in TABLES.items():
            z.write(path, name)
        z.writestr("E8_CODE_MANIFEST.json", json.dumps(manifest, indent=1))
    print(f"wrote {OUT}  ({OUT.stat().st_size / 1024:.0f} KB, {len(files)} code files + "
          f"{len(TABLES)} tables)")
    print(f"commit {manifest['commit'][:12]} on {manifest['branch']}")
    print(f"sha256 {sha256(OUT)}")


if __name__ == "__main__":
    main()
