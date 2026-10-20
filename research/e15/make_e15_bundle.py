"""Package the E15-B code and the training stream as a Kaggle dataset, `uie-e15-code`
(research/e15/PREREG_E15.md section 9.12, step 3). One bundle serves both notebooks.

    python research/e15/make_e15_bundle.py

The same rules as the E8-E14 bundles: refuses a dirty tree under uie/, research/native/,
research/e13/ or research/e15/, any table that is not the registered one, and any stream
that is not Amendment 3's. The manifest records the commit and a hash of every file, so
each notebook proves it runs exactly this code on exactly this stream. Nothing inside is
gzipped: Kaggle decompresses .gz files inside an uploaded dataset.
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
import e15_common as E15  # noqa: E402

OUT = ROOT / "data" / "kaggle" / "uie-e15-code.zip"
NATIVE = ("e8_common.py", "e11_common.py", "e12_common.py")
E13_CODE = ("e13_common.py",)
E15_CODE = ("e15_common.py",)
E12_FINAL = (ROOT / "data" / "e12" / "train" / "e12_final" / "final.pt", E15.E12_FINAL_SHA256)
STREAM_DIR = ROOT / "data" / "e15" / "qry_extra"
ARTIFACTS = {
    "presence_slash_table.json": (ROOT / "data" / "e12" / "presence_slash_table.json",
                                  "5a0eeac991b8343267d6f88f4c590bd6c208ace975dbf2218f8c5a277b71c050"),
}


def main() -> None:
    dirty = git("status", "--porcelain", "--", "uie", "research/native", "research/e13",
                "research/e15")
    if dirty:
        sys.exit(f"refusing: uncommitted changes:\n{dirty}")
    for name, (path, want) in {**TABLES, **ARTIFACTS, "e12_final.pt": E12_FINAL}.items():
        if sha256(path) != want:
            sys.exit(f"refusing: {path} is not the registered {name} ({want[:12]}...)")
    stream = STREAM_DIR / E15.STREAM_FILE
    info = E15.build_stream(STREAM_DIR, stream)          # checks Amendment 3's hashes
    if info["rows"] != sum(E15.N_NEW.values()):
        sys.exit(f"refusing: the stream has {info['rows']} rows, Amendment 3 {sum(E15.N_NEW.values())}")
    files = sorted(p for p in (ROOT / "uie").rglob("*.py") if "__pycache__" not in p.parts)
    files += [ROOT / "research" / "native" / name for name in NATIVE]
    files += [ROOT / "research" / "e13" / name for name in E13_CODE]
    files += [HERE / name for name in E15_CODE]
    tables = {name: (path, want) for name, (path, want) in {**TABLES, **ARTIFACTS}.items()}
    names = [p.name for p in files] + list(tables) + ["e12_final.pt", E15.STREAM_FILE]
    if any(n.endswith(".gz") for n in names):
        sys.exit("refusing: a .gz file inside a Kaggle dataset is decompressed on upload")
    manifest = {
        "experiment": "E15-B (new same-population benign query rows, injected; query-only targets)",
        "registration": E15.REGISTRATION,
        "branch": git("rev-parse", "--abbrev-ref", "HEAD"),
        "commit": git("rev-parse", "HEAD"),
        "built": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "attach_with": "uie-corpus2 (the data bundle, unchanged) for E15B_train",
        "files": {p.relative_to(ROOT).as_posix(): sha256(p) for p in files},
        "tables": {name: want for name, (_, want) in tables.items()},
        "e12_final": E12_FINAL[1],
        "stream": {"file": E15.STREAM_FILE, "rows": info["rows"], "sha256": info["sha256"],
                   "strata": E15.N_NEW, "stratum_files": {k: v[1] for k, v in E15.STRATUM_FILES.items()}},
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(OUT, "w", zipfile.ZIP_DEFLATED) as z:
        for p in files:
            z.write(p, p.relative_to(ROOT).as_posix())
        for name, (path, _) in tables.items():
            z.write(path, name)
        z.write(E12_FINAL[0], "e12_final.pt")
        z.write(stream, E15.STREAM_FILE)
        z.writestr("E15_CODE_MANIFEST.json", json.dumps(manifest, indent=1))
    print(f"wrote {OUT} ({OUT.stat().st_size / 2**20:.1f} MB), sha256 {sha256(OUT)}")
    print(f"commit {manifest['commit'][:12]}; {len(files)} files; tables {list(tables)}; "
          f"stream {info['rows']:,} rows, sha256 {info['sha256'][:12]}...")


if __name__ == "__main__":
    main()
