"""Package the E17 code and the pooled stream as a Kaggle dataset, `uie-e17-code`
(research/e17/PREREG_E17.md section 7, step 2). One bundle serves all four notebooks.

    python research/e17/make_e17_bundle.py

The same rules as the E8-E16 bundles: refuses a dirty tree under uie/ or research/
native, e13, e15, e16, e17; any table that is not the registered one; and a pooled
stream that does not rebuild from the registered stream-1 and stream-2 files. The
manifest records the commit and a hash of every file, so each notebook proves it runs
exactly this code. Nothing inside is gzipped (Kaggle decompresses .gz on upload).

No checkpoint rides along: E17_decode and E17_train attach E16_train's output (e16_cap,
checked against `e17_common.E16_CAP_SHA256`), and E17_generate attaches E17_train's.
"""
from __future__ import annotations

import json
import sys
import zipfile
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
for p in (ROOT / "research" / "native", ROOT / "research" / "e15", ROOT / "research" / "e16", HERE):
    sys.path.insert(0, str(p))

from make_code_bundle import TABLES, git, sha256  # noqa: E402
import e17_common as E17  # noqa: E402

OUT = ROOT / "data" / "kaggle" / "uie-e17-code.zip"
NATIVE = ("e8_common.py", "e11_common.py", "e12_common.py")
ARTIFACTS = {
    "presence_slash_table.json": (ROOT / "data" / "e12" / "presence_slash_table.json",
                                  "5a0eeac991b8343267d6f88f4c590bd6c208ace975dbf2218f8c5a277b71c050"),
}


def main() -> None:
    dirty = git("status", "--porcelain", "--", "uie", "research/native", "research/e13", "research/e15",
                "research/e16", "research/e17")
    if dirty:
        sys.exit(f"refusing: uncommitted changes:\n{dirty}")
    for name, (path, want) in {**TABLES, **ARTIFACTS}.items():
        if sha256(path) != want:
            sys.exit(f"refusing: {path} is not the registered {name} ({want[:12]}...)")
    stream = E17.STREAM2_DIR / E17.STREAM_FILE
    info = E17.build_stream(stream)                      # checks every registered hash
    if info["per_stratum"] != E17.N_NEW:
        sys.exit(f"refusing: the pooled stream has {info['per_stratum']}, amendment 1 {E17.N_NEW}")
    files = sorted(p for p in (ROOT / "uie").rglob("*.py") if "__pycache__" not in p.parts)
    files += [ROOT / "research" / "native" / name for name in NATIVE]
    files += [ROOT / "research" / "e13" / "e13_common.py", ROOT / "research" / "e15" / "e15_common.py",
              ROOT / "research" / "e16" / "e16_common.py", HERE / "e17_common.py"]
    tables = dict({**TABLES, **ARTIFACTS})
    names = [p.name for p in files] + list(tables) + [E17.STREAM_FILE]
    if any(n.endswith(".gz") for n in names):
        sys.exit("refusing: a .gz file inside a Kaggle dataset is decompressed on upload")
    manifest = {
        "experiment": "E17 (B5 on e16_cap; option A: streams 1 + 2 on e16_cap's recipe)",
        "registration": E17.REGISTRATION,
        "branch": git("rev-parse", "--abbrev-ref", "HEAD"),
        "commit": git("rev-parse", "HEAD"),
        "built": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "attach_with": "E17_decode: E16_train's output. E17_train: uie-corpus2 and E16_train's output. "
                       "E17_generate: E17_train's output.",
        "files": {p.relative_to(ROOT).as_posix(): sha256(p) for p in files},
        "tables": {name: want for name, (_, want) in tables.items()},
        "e16_cap": E17.E16_CAP_SHA256,
        "stream": {"file": E17.STREAM_FILE, "rows": info["rows"], "sha256": info["sha256"],
                   "per_stratum": info["per_stratum"]},
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(OUT, "w", zipfile.ZIP_DEFLATED) as z:
        for p in files:
            z.write(p, p.relative_to(ROOT).as_posix())
        for name, (path, _) in tables.items():
            z.write(path, name)
        z.write(stream, E17.STREAM_FILE)
        z.writestr("E17_CODE_MANIFEST.json", json.dumps(manifest, indent=1))
    print(f"wrote {OUT} ({OUT.stat().st_size / 2**20:.1f} MB), sha256 {sha256(OUT)}")
    print(f"commit {manifest['commit'][:12]}; {len(files)} files; stream {info['rows']:,} rows, "
          f"sha256 {info['sha256'][:12]}...")


if __name__ == "__main__":
    main()
