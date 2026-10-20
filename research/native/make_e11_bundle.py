"""Package the E11 code as a Kaggle dataset, `uie-e11-code`
(research/native/PREREG_E11.md). One bundle serves both notebooks.

    python research/native/make_e11_bundle.py

The same rules as the E8-E10 bundles: refuses a dirty tree under uie/ or
research/native/ and tables that are not the registered ones; the manifest records
the commit and a hash of every file, so each notebook proves it runs exactly this
code. E11 adds one table: the presence table free mode's supplied plan draws from
(PREREG_E11 D6; written by e11_targets.py).
"""
from __future__ import annotations

import json
import sys
import zipfile
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from make_code_bundle import ROOT, TABLES, git, sha256  # noqa: E402

OUT = ROOT / "data" / "kaggle" / "uie-e11-code.zip"
NATIVE = ("e8_common.py", "e11_common.py")
#: PREREG_E11 D6: the presence table, by its registered hash.
E11_TABLES = {**TABLES,
              "presence_table.json": (ROOT / "data" / "e11" / "presence_table.json",
                                      "7a74345241af3b8acaaa15e6229920ec43dac2497c3aa2ee6dbfd849b62783c6")}


def main() -> None:
    dirty = git("status", "--porcelain", "--", "uie", "research/native")
    if dirty:
        sys.exit(f"refusing: uncommitted changes:\n{dirty}")
    for name, (path, want) in E11_TABLES.items():
        if sha256(path) != want:
            sys.exit(f"refusing: {path} is not the registered {name} ({want[:12]}...)")
    files = sorted(p for p in (ROOT / "uie").rglob("*.py") if "__pycache__" not in p.parts)
    files += [ROOT / "research" / "native" / name for name in NATIVE]
    manifest = {
        "experiment": "E11 (presence and the root slash)",
        "registration": "research/native/PREREG_E11.md",
        "branch": git("rev-parse", "--abbrev-ref", "HEAD"),
        "commit": git("rev-parse", "HEAD"),
        "built": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "attach_with": "uie-corpus2 (the data bundle, unchanged)",
        "files": {p.relative_to(ROOT).as_posix(): sha256(p) for p in files},
        "tables": {name: want for name, (_, want) in E11_TABLES.items()},
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(OUT, "w", zipfile.ZIP_DEFLATED) as z:
        for p in files:
            z.write(p, p.relative_to(ROOT).as_posix())
        for name, (path, _) in E11_TABLES.items():
            z.write(path, name)
        z.writestr("E11_CODE_MANIFEST.json", json.dumps(manifest, indent=1))
    print(f"wrote {OUT}  ({OUT.stat().st_size / 1024:.0f} KB, {len(files)} code files + "
          f"{len(E11_TABLES)} tables)")
    print(f"commit {manifest['commit'][:12]} on {manifest['branch']}")
    print(f"sha256 {sha256(OUT)}")


if __name__ == "__main__":
    main()
