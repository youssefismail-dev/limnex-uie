"""Package the V2-P2 code, both piece tables, stream 1 and e16_cap's record as a Kaggle
dataset, `uie-v2p2-code` (research/v2/PREREG_V2_P2.md section 15, step 1). One bundle serves
all four notebooks (train and generate, arms A and B).

    python research/v2/make_v2p2_bundle.py

P1b's rules (make_v2p1b_bundle.py), plus P2's:
- it refuses a dirty tree under uie/ or the research modules it ships, and a uie/ that is
  not 422cb23's (V-iso);
- it refuses any table that is not the registered one: the suffix and presence tables, and
  both piece tables against their pinned hashes (Amendment 3; V-T3);
- it refuses a stream that is not stream 1, an e16_cap checkpoint that is not the adopted one,
  and an arm whose C is not pinned yet.
The manifest records the commit, every file's hash, the piece tables' and the per-field
rules' hashes. Nothing inside is gzipped. No checkpoint rides along.
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
import v2_p2_common as C  # noqa: E402
from make_v2p1_bundle import ARTIFACTS, E16_CAP, NATIVE, SHIPPED_RESEARCH, STREAM  # noqa: E402

OUT = ROOT / "data" / "kaggle" / "uie-v2p2-code.zip"
RECORD = ROOT / "data" / "v2" / "p2" / C.E16_CAP_RECORD
PIECE_TABLES = ROOT / "data" / "v2" / "p2" / "tables"
SHIPPED_V2 = ("v2_mem.py", "v2_bpe.py", "v2_p1_common.py", "v2_p2_common.py", "v2_p2_tokenizer.py",
              "v2_p2_train.py")


def main() -> None:
    import argparse
    import torch
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry", action="store_true",
                    help="the notebook pre-flight only: an uncommitted tree and an unpinned C are "
                         "allowed; writes uie-v2p2-code.dry.zip, marked dry, never for Kaggle")
    args = ap.parse_args()
    global OUT
    if args.dry:
        OUT = OUT.with_name("uie-v2p2-code.dry.zip")
    dirty = git("status", "--porcelain", "--", "uie", *SHIPPED_RESEARCH)
    if dirty and not args.dry:
        sys.exit(f"refusing: uncommitted changes:\n{dirty}")
    if git("diff", "--name-only", C.ISO_COMMIT, "HEAD", "--", "uie") or git("status", "--porcelain", "--", "uie"):
        sys.exit(f"refusing: uie/ is not {C.ISO_COMMIT[:12]}'s -- V-iso")
    for name, (path, want) in {**TABLES, **ARTIFACTS, "stream 1": (STREAM, C.STREAM_SHA256),
                               "e16_cap": (E16_CAP, C.E16_CAP_SHA256)}.items():
        if sha256(path) != want:
            sys.exit(f"refusing: {path} is not the registered {name} ({want[:12]}...)")
    if any(v is None for v in C.TABLE_SHA256.values()) or (
            any(v is None for v in C.LOSS_CONSTANT_P2.values()) and not args.dry):
        sys.exit("refusing: the tables' hashes and both arms' C must be pinned first (Amendment 3)")
    pieces = {}
    for name in ("rest", "host"):
        state = json.loads((PIECE_TABLES / f"{name}.json").read_text(encoding="utf-8"))
        if C.table_sha256(state) != C.TABLE_SHA256[name]:
            sys.exit(f"refusing: the {name} table is not the registered one ({C.TABLE_SHA256[name][:12]}...)")
        pieces[C.TABLE_FILES[name]] = PIECE_TABLES / f"{name}.json"
    rows = sum(1 for line in STREAM.read_text(encoding="utf-8").splitlines() if line.strip())
    if rows != C.STREAM_ROWS:
        sys.exit(f"refusing: stream 1 has {rows} rows, not {C.STREAM_ROWS}")
    blob = torch.load(E16_CAP, map_location="cpu", weights_only=False)
    RECORD.parent.mkdir(parents=True, exist_ok=True)
    RECORD.write_text(json.dumps(C.e16_cap_record(blob), indent=1, sort_keys=True), encoding="utf-8",
                      newline="\n")
    files = sorted(p for p in (ROOT / "uie").rglob("*.py") if "__pycache__" not in p.parts)
    files += [ROOT / "research" / "native" / name for name in NATIVE]
    files += [ROOT / "research" / "e13" / "e13_common.py", ROOT / "research" / "e15" / "e15_common.py",
              ROOT / "research" / "e16" / "e16_common.py", ROOT / "research" / "e17" / "e17_common.py"]
    files += [HERE / name for name in SHIPPED_V2]
    tables = dict({**TABLES, **ARTIFACTS})
    names = [p.name for p in files] + list(tables) + list(pieces) + [C.STREAM_FILE, C.E16_CAP_RECORD]
    if any(n.endswith(".gz") for n in names):
        sys.exit("refusing: a .gz file inside a Kaggle dataset is decompressed on upload")
    manifest = {
        "experiment": "V2-P2 (p2_a: path, query and fragment in pieces; p2_b: the host too)",
        **({"dry_run": "pre-flight only -- NOT for Kaggle (uncommitted tree or unpinned C)"} if args.dry else {}),
        "registration": C.REGISTRATION,
        "branch": git("rev-parse", "--abbrev-ref", "HEAD"),
        "commit": git("rev-parse", "HEAD"),
        "uie_iso_commit": C.ISO_COMMIT,
        "built": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "attach_with": "V2P2_train_a/b: uie-corpus2. V2P2_generate_a/b: that arm's V2P2_train output.",
        "files": {p.relative_to(ROOT).as_posix(): sha256(p) for p in files},
        "tables": {name: want for name, (_, want) in tables.items()},
        "piece_tables": {name: sha256(path) for name, path in pieces.items()},
        "piece_tables_registered": dict(C.TABLE_SHA256),
        "rules_sha256": C.rules_sha256(),
        "loss_constants": dict(C.LOSS_CONSTANT_P2),
        "stream": {"file": C.STREAM_FILE, "rows": rows, "sha256": C.STREAM_SHA256},
        "e16_cap_record": sha256(RECORD),
        "e16_cap": C.E16_CAP_SHA256,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(OUT, "w", zipfile.ZIP_DEFLATED) as z:
        for p in files:
            z.write(p, p.relative_to(ROOT).as_posix())
        for name, (path, _) in tables.items():
            z.write(path, name)
        for name, path in pieces.items():
            z.write(path, name)
        z.write(STREAM, C.STREAM_FILE)
        z.write(RECORD, C.E16_CAP_RECORD)
        z.writestr(C.CODE_MANIFEST, json.dumps(manifest, indent=1))
    print(f"wrote {OUT} ({OUT.stat().st_size / 2**20:.1f} MB), sha256 {sha256(OUT)}")
    print(f"commit {manifest['commit'][:12]}; uie/ = {C.ISO_COMMIT[:12]}; {len(files)} files; "
          f"piece tables {[h[:12] for h in C.TABLE_SHA256.values()]}; stream 1 {rows:,} rows; "
          f"e16_cap record {manifest['e16_cap_record'][:12]}...")


if __name__ == "__main__":
    main()
