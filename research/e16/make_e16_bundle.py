"""Package the E16 code, the stream and the baseline's checkpoint as a Kaggle dataset,
`uie-e16-code` (research/e16/PREREG_E16.md section 11, step 2). One bundle serves both
notebooks.

    python research/e16/make_e16_bundle.py

The same rules as the E8-E15 bundles: refuses a dirty tree under uie/, research/native/,
research/e13/, research/e15/ or research/e16/, any table that is not the registered one,
a stream that is not amendment 3's, and a baseline checkpoint that is not the adopted
`e15_new`. The manifest records the commit and a hash of every file, so each notebook
proves it runs exactly this code. Nothing inside is gzipped: Kaggle decompresses .gz
files inside an uploaded dataset.

`e15_new.pt` rides along for two reasons: its tokenizer rebuilds the start before any
training (V-start', amendment 2), and its recorded settings are what E16_train's
readback compares the run against.
"""
from __future__ import annotations

import json
import sys
import zipfile
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
for p in (ROOT / "research" / "native", ROOT / "research" / "e15", HERE):
    sys.path.insert(0, str(p))

from make_code_bundle import TABLES, git, sha256  # noqa: E402
import e15_common as E15  # noqa: E402
import e16_common as E16  # noqa: E402

OUT = ROOT / "data" / "kaggle" / "uie-e16-code.zip"
NATIVE = ("e8_common.py", "e11_common.py", "e12_common.py")
E15_NEW = (ROOT / "data" / "e15" / "train" / "e15_new" / "final.pt", E16.E15_NEW_SHA256)
STREAM_DIR = ROOT / "data" / "e15" / "qry_extra"
ARTIFACTS = {
    "presence_slash_table.json": (ROOT / "data" / "e12" / "presence_slash_table.json",
                                  "5a0eeac991b8343267d6f88f4c590bd6c208ace975dbf2218f8c5a277b71c050"),
}


def main() -> None:
    dirty = git("status", "--porcelain", "--", "uie", "research/native", "research/e13",
                "research/e15", "research/e16")
    if dirty:
        sys.exit(f"refusing: uncommitted changes:\n{dirty}")
    for name, (path, want) in {**TABLES, **ARTIFACTS, "e15_new.pt": E15_NEW}.items():
        if sha256(path) != want:
            sys.exit(f"refusing: {path} is not the registered {name} ({want[:12]}...)")
    stream = STREAM_DIR / E15.STREAM_FILE
    info = E15.build_stream(STREAM_DIR, stream)          # checks amendment 3's hashes
    if info["rows"] != sum(E15.N_NEW.values()):
        sys.exit(f"refusing: the stream has {info['rows']} rows, amendment 3 {sum(E15.N_NEW.values())}")
    files = sorted(p for p in (ROOT / "uie").rglob("*.py") if "__pycache__" not in p.parts)
    files += [ROOT / "research" / "native" / name for name in NATIVE]
    files += [ROOT / "research" / "e13" / "e13_common.py", ROOT / "research" / "e15" / "e15_common.py",
              HERE / "e16_common.py"]
    tables = dict({**TABLES, **ARTIFACTS})
    names = [p.name for p in files] + list(tables) + ["e15_new.pt", E15.STREAM_FILE]
    if any(n.endswith(".gz") for n in names):
        sys.exit("refusing: a .gz file inside a Kaggle dataset is decompressed on upload")
    manifest = {
        "experiment": "E16 (the capacity probe: e16_cap = e15_new's recipe at d_model 512)",
        "registration": E16.REGISTRATION,
        "branch": git("rev-parse", "--abbrev-ref", "HEAD"),
        "commit": git("rev-parse", "HEAD"),
        "built": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "attach_with": "uie-corpus2 (the data bundle, unchanged) for E16_train",
        "files": {p.relative_to(ROOT).as_posix(): sha256(p) for p in files},
        "tables": {name: want for name, (_, want) in tables.items()},
        "e15_new": E15_NEW[1],
        "stream": {"file": E15.STREAM_FILE, "rows": info["rows"], "sha256": info["sha256"],
                   "strata": E15.N_NEW},
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(OUT, "w", zipfile.ZIP_DEFLATED) as z:
        for p in files:
            z.write(p, p.relative_to(ROOT).as_posix())
        for name, (path, _) in tables.items():
            z.write(path, name)
        z.write(E15_NEW[0], "e15_new.pt")
        z.write(stream, E15.STREAM_FILE)
        z.writestr("E16_CODE_MANIFEST.json", json.dumps(manifest, indent=1))
    print(f"wrote {OUT} ({OUT.stat().st_size / 2**20:.1f} MB), sha256 {sha256(OUT)}")
    print(f"commit {manifest['commit'][:12]}; {len(files)} files; tables {list(tables)}; "
          f"stream {info['rows']:,} rows, sha256 {info['sha256'][:12]}...; e15_new.pt {E15_NEW[1][:12]}...")


if __name__ == "__main__":
    main()
