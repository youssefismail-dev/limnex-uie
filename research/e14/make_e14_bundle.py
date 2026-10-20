"""Package the E14 code as a Kaggle dataset, `uie-e14-code`
(research/e14/PREREG_E14.md section 6, step 2). One bundle serves all three notebooks.

    python research/e14/make_e14_bundle.py

The same rules as the E8-E13 bundles: refuses a dirty tree under uie/,
research/native/, research/e13/ or research/e14/, and any table that is not the
registered one; the manifest records the commit and a hash of every file, so each
notebook proves it runs exactly this code. It carries e12_final's final.pt (the
D1' arm's weights), checked against E12's records. Nothing inside is gzipped:
Kaggle decompresses .gz files inside an uploaded dataset (E13's lesson), so the
builder refuses one.
"""
from __future__ import annotations

import json
import sys
import zipfile
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "research" / "native"))
sys.path.insert(0, str(HERE))

from make_code_bundle import TABLES, git, sha256  # noqa: E402
import e14_common as E14  # noqa: E402

OUT = ROOT / "data" / "kaggle" / "uie-e14-code.zip"
NATIVE = ("e8_common.py", "e11_common.py", "e12_common.py")
E13_CODE = ("e13_common.py",)
E14_CODE = ("e14_common.py",)
E12_FINAL = (ROOT / "data" / "e12" / "train" / "e12_final" / "final.pt", E14.E12_FINAL_SHA256)
ARTIFACTS = {
    "presence_slash_table.json": (ROOT / "data" / "e12" / "presence_slash_table.json",
                                  "5a0eeac991b8343267d6f88f4c590bd6c208ace975dbf2218f8c5a277b71c050"),
}


def main() -> None:
    dirty = git("status", "--porcelain", "--", "uie", "research/native", "research/e13",
                "research/e14")
    if dirty:
        sys.exit(f"refusing: uncommitted changes:\n{dirty}")
    for name, (path, want) in {**TABLES, **ARTIFACTS, "e12_final.pt": E12_FINAL}.items():
        if sha256(path) != want:
            sys.exit(f"refusing: {path} is not the registered {name} ({want[:12]}...)")
    files = sorted(p for p in (ROOT / "uie").rglob("*.py") if "__pycache__" not in p.parts)
    files += [ROOT / "research" / "native" / name for name in NATIVE]
    files += [ROOT / "research" / "e13" / name for name in E13_CODE]
    files += [HERE / name for name in E14_CODE]
    tables = {name: (path, want) for name, (path, want) in {**TABLES, **ARTIFACTS}.items()}
    names = [p.name for p in files] + list(tables) + ["e12_final.pt"]
    if any(n.endswith(".gz") for n in names):
        sys.exit("refusing: a .gz file inside a Kaggle dataset is decompressed on upload")
    manifest = {
        "experiment": "E14 (query exposure N2b; D1' the slash rule on plans with a path)",
        "registration": E14.REGISTRATION,
        "branch": git("rev-parse", "--abbrev-ref", "HEAD"),
        "commit": git("rev-parse", "HEAD"),
        "built": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "attach_with": "uie-corpus2 (the data bundle, unchanged) for E14_train",
        "files": {p.relative_to(ROOT).as_posix(): sha256(p) for p in files},
        "tables": {name: want for name, (_, want) in tables.items()},
        "e12_final": E12_FINAL[1],
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(OUT, "w", zipfile.ZIP_DEFLATED) as z:
        for p in files:
            z.write(p, p.relative_to(ROOT).as_posix())
        for name, (path, _) in tables.items():
            z.write(path, name)
        z.write(E12_FINAL[0], "e12_final.pt")
        z.writestr("E14_CODE_MANIFEST.json", json.dumps(manifest, indent=1))
    print(f"wrote {OUT} ({OUT.stat().st_size / 2**20:.1f} MB), sha256 {sha256(OUT)}")
    print(f"commit {manifest['commit'][:12]}; {len(files)} files; tables {list(tables)}")


if __name__ == "__main__":
    main()
