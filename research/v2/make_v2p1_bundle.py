"""Package the V2-P1 code, stream 1 and e16_cap's record as a Kaggle dataset,
`uie-v2p1-code` (research/v2/PREREG_V2_P1_MEMORY.md section 10, step 2). One bundle serves
both notebooks.

    python research/v2/make_v2p1_bundle.py

The same rules as the E8-E17 bundles: refuses a dirty tree under uie/ or the research
modules it ships; any table that is not the registered one; a stream that is not stream 1
(`637accab...`); and an e16_cap checkpoint that is not the adopted one. The manifest
records the commit and a hash of every file, so each notebook proves it runs exactly this
code. Nothing inside is gzipped (Kaggle decompresses .gz on upload).

No checkpoint rides along. `e16_cap_record.json` carries what the training notebook needs
of e16_cap's checkpoint: its tokenizer (to rebuild the start before training) and its
settings (what the readback compares the run against). V2P1_generate attaches
V2P1_train's output.
"""
from __future__ import annotations

import json
import sys
import zipfile
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
for p in (ROOT / "research" / "native", HERE):
    sys.path.insert(0, str(p))

from make_code_bundle import TABLES, git, sha256  # noqa: E402
import v2_p1_common as P1  # noqa: E402

OUT = ROOT / "data" / "kaggle" / "uie-v2p1-code.zip"
RECORD = ROOT / "data" / "v2" / "p1" / P1.E16_CAP_RECORD
E16_CAP = ROOT / "data" / "e16" / "train" / "e16_cap" / "final.pt"
STREAM = ROOT / "data" / "e15" / "qry_extra" / P1.STREAM_FILE
NATIVE = ("e8_common.py", "e11_common.py", "e12_common.py")
ARTIFACTS = {
    "presence_slash_table.json": (ROOT / "data" / "e12" / "presence_slash_table.json",
                                  "5a0eeac991b8343267d6f88f4c590bd6c208ace975dbf2218f8c5a277b71c050"),
}
SHIPPED_RESEARCH = ("research/native", "research/e13", "research/e15", "research/e16", "research/e17",
                    "research/v2")


def main() -> None:
    import torch
    dirty = git("status", "--porcelain", "--", "uie", *SHIPPED_RESEARCH)
    if dirty:
        sys.exit(f"refusing: uncommitted changes:\n{dirty}")
    for name, (path, want) in {**TABLES, **ARTIFACTS, "stream 1": (STREAM, P1.STREAM_SHA256),
                               "e16_cap": (E16_CAP, P1.E16_CAP_SHA256)}.items():
        if sha256(path) != want:
            sys.exit(f"refusing: {path} is not the registered {name} ({want[:12]}...)")
    rows = sum(1 for line in STREAM.read_text(encoding="utf-8").splitlines() if line.strip())
    if rows != P1.STREAM_ROWS:
        sys.exit(f"refusing: stream 1 has {rows} rows, not {P1.STREAM_ROWS}")
    blob = torch.load(E16_CAP, map_location="cpu", weights_only=False)
    RECORD.parent.mkdir(parents=True, exist_ok=True)
    RECORD.write_text(json.dumps(P1.e16_cap_record(blob), indent=1, sort_keys=True), encoding="utf-8",
                      newline="\n")
    files = sorted(p for p in (ROOT / "uie").rglob("*.py") if "__pycache__" not in p.parts)
    files += [ROOT / "research" / "native" / name for name in NATIVE]
    files += [ROOT / "research" / "e13" / "e13_common.py", ROOT / "research" / "e15" / "e15_common.py",
              ROOT / "research" / "e16" / "e16_common.py", ROOT / "research" / "e17" / "e17_common.py",
              HERE / "v2_mem.py", HERE / "v2_p1_common.py", HERE / "v2_p1_train.py"]
    tables = dict({**TABLES, **ARTIFACTS})
    names = [p.name for p in files] + list(tables) + [P1.STREAM_FILE, P1.E16_CAP_RECORD]
    if any(n.endswith(".gz") for n in names):
        sys.exit("refusing: a .gz file inside a Kaggle dataset is decompressed on upload")
    manifest = {
        "experiment": "V2-P1 (p1_mem = e16_cap + a product-key memory layer after block 4)",
        "registration": P1.REGISTRATION,
        "branch": git("rev-parse", "--abbrev-ref", "HEAD"),
        "commit": git("rev-parse", "HEAD"),
        "built": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "attach_with": "V2P1_train: uie-corpus2. V2P1_generate: V2P1_train's output.",
        "files": {p.relative_to(ROOT).as_posix(): sha256(p) for p in files},
        "tables": {name: want for name, (_, want) in tables.items()},
        "stream": {"file": P1.STREAM_FILE, "rows": rows, "sha256": P1.STREAM_SHA256},
        "e16_cap_record": sha256(RECORD),
        "e16_cap": P1.E16_CAP_SHA256,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(OUT, "w", zipfile.ZIP_DEFLATED) as z:
        for p in files:
            z.write(p, p.relative_to(ROOT).as_posix())
        for name, (path, _) in tables.items():
            z.write(path, name)
        z.write(STREAM, P1.STREAM_FILE)
        z.write(RECORD, P1.E16_CAP_RECORD)
        z.writestr(P1.CODE_MANIFEST, json.dumps(manifest, indent=1))
    print(f"wrote {OUT} ({OUT.stat().st_size / 2**20:.1f} MB), sha256 {sha256(OUT)}")
    print(f"commit {manifest['commit'][:12]}; {len(files)} files; tables {list(tables)}; "
          f"stream 1 {rows:,} rows; e16_cap record {manifest['e16_cap_record'][:12]}...")


if __name__ == "__main__":
    main()
