"""Package the code, settings and tables for DC1's Kaggle generation.

Ships `uie/` at a clean, committed HEAD (with DC1's sampler options),
`dc1_common.py`, the fit's `settings.json`, the adopted state tables and the
shipped suffix table, with `DC1_CODE_MANIFEST.json` -- the commit and a SHA-256
per file, which the notebook checks before it generates anything.

    python research/dc1/make_code_bundle.py [--settings data/dc1/settings.json]

Refuses a dirty `uie/` or `research/dc1/`, and tables that are not the adopted file.
"""
import argparse
import hashlib
import json
import subprocess
import sys
import zipfile
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
STATE_TABLES = ROOT / "data" / "state_suffix_tables.json"
SHIPPED = ROOT / "data" / "bundle2" / "suffix_table.json"
TABLES_SHA256 = "0c6f2760c033c532d271b360a14f2976139d5957e4aaf64977c8746504213cc9"


def git(*args: str) -> str:
    return subprocess.run(["git", *args], cwd=ROOT, capture_output=True,
                          text=True, check=True).stdout.strip()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--settings", type=Path, default=ROOT / "data" / "dc1" / "settings.json")
    ap.add_argument("--out", type=Path, default=ROOT / "data" / "kaggle" / "uie-dc1-code.zip")
    ap.add_argument("--allow-dirty", action="store_true", help="simulation only")
    args = ap.parse_args()

    dirty = git("status", "--porcelain", "--", "uie", "research/dc1")
    if dirty and not args.allow_dirty:
        sys.exit(f"refusing: uncommitted changes under uie/ or research/dc1/:\n{dirty}")
    if hashlib.sha256(STATE_TABLES.read_bytes()).hexdigest() != TABLES_SHA256:
        sys.exit("refusing: data/state_suffix_tables.json is not the adopted file")
    settings = json.loads(args.settings.read_text(encoding="utf-8"))
    if settings.get("registration") != "research/dc1/PREREG_DC1.md":
        sys.exit(f"refusing: {args.settings} is not a DC1 fit output")
    members = {p.relative_to(ROOT).as_posix(): p
               for p in sorted((ROOT / "uie").rglob("*.py")) if "__pycache__" not in p.parts}
    members["dc1_common.py"] = HERE / "dc1_common.py"
    members["settings.json"] = args.settings
    members["state_suffix_tables.json"] = STATE_TABLES
    members["suffix_table.json"] = SHIPPED
    manifest = {
        "experiment": "DC1",
        "registration": "research/dc1/PREREG_DC1.md",
        "branch": git("rev-parse", "--abbrev-ref", "HEAD"),
        "commit": git("rev-parse", "HEAD"),
        "dirty": bool(dirty),
        "built": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "attach_with": "the E7 training notebook's output (best.pt); no data bundle",
        "files": {name: hashlib.sha256(p.read_bytes()).hexdigest()
                  for name, p in members.items()},
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(args.out, "w", zipfile.ZIP_DEFLATED) as z:
        for name, p in members.items():
            z.write(p, name)
        z.writestr("DC1_CODE_MANIFEST.json", json.dumps(manifest, indent=1))
    print(f"wrote {args.out}  ({args.out.stat().st_size / 1024:.0f} KB, {len(members)} files)")
    print(f"commit {manifest['commit'][:12]}; settings sha256 "
          f"{manifest['files']['settings.json'][:16]}")


if __name__ == "__main__":
    main()
