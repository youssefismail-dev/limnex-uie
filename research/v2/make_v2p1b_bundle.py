"""Package the V2-P1b code, stream 1 and e16_cap's record as a Kaggle dataset,
`uie-v2p1b-code` (research/v2/PREREG_V2_P1B.md section 7, step 2). One bundle serves both
notebooks.

    python research/v2/make_v2p1b_bundle.py

P1's rules (make_v2p1_bundle.py): it refuses a dirty tree under uie/ or the research modules
it ships, any table that is not the registered one, a stream that is not stream 1, an
e16_cap checkpoint that is not the adopted one, and -- P1b's V-iso -- a uie/ that differs
from the commit adding T2's hook. The manifest records the commit and every file's hash.
Nothing inside is gzipped. No checkpoint rides along.
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
import v2_p1b_common as P1B  # noqa: E402
from make_v2p1_bundle import ARTIFACTS, E16_CAP, NATIVE, SHIPPED_RESEARCH, STREAM  # noqa: E402

OUT = ROOT / "data" / "kaggle" / "uie-v2p1b-code.zip"
RECORD = ROOT / "data" / "v2" / "p1b" / P1B.E16_CAP_RECORD
SHIPPED_V2 = ("v2_mem.py", "v2_p1_common.py", "v2_p1b_common.py", "v2_p1b_train.py")


def main() -> None:
    import torch
    dirty = git("status", "--porcelain", "--", "uie", *SHIPPED_RESEARCH)
    if dirty:
        sys.exit(f"refusing: uncommitted changes:\n{dirty}")
    if not P1B.ISO_COMMIT or git("diff", "--name-only", P1B.ISO_COMMIT, "HEAD", "--", "uie"):
        sys.exit(f"refusing: uie/ is not the hook commit's ({P1B.ISO_COMMIT}) -- V-iso")
    for name, (path, want) in {**TABLES, **ARTIFACTS, "stream 1": (STREAM, P1B.STREAM_SHA256),
                               "e16_cap": (E16_CAP, P1B.E16_CAP_SHA256)}.items():
        if sha256(path) != want:
            sys.exit(f"refusing: {path} is not the registered {name} ({want[:12]}...)")
    rows = sum(1 for line in STREAM.read_text(encoding="utf-8").splitlines() if line.strip())
    if rows != P1B.STREAM_ROWS:
        sys.exit(f"refusing: stream 1 has {rows} rows, not {P1B.STREAM_ROWS}")
    blob = torch.load(E16_CAP, map_location="cpu", weights_only=False)
    RECORD.parent.mkdir(parents=True, exist_ok=True)
    RECORD.write_text(json.dumps(P1B.e16_cap_record(blob), indent=1, sort_keys=True), encoding="utf-8",
                      newline="\n")
    files = sorted(p for p in (ROOT / "uie").rglob("*.py") if "__pycache__" not in p.parts)
    files += [ROOT / "research" / "native" / name for name in NATIVE]
    files += [ROOT / "research" / "e13" / "e13_common.py", ROOT / "research" / "e15" / "e15_common.py",
              ROOT / "research" / "e16" / "e16_common.py", ROOT / "research" / "e17" / "e17_common.py"]
    files += [HERE / name for name in SHIPPED_V2]
    tables = dict({**TABLES, **ARTIFACTS})
    names = [p.name for p in files] + list(tables) + [P1B.STREAM_FILE, P1B.E16_CAP_RECORD]
    if any(n.endswith(".gz") for n in names):
        sys.exit("refusing: a .gz file inside a Kaggle dataset is decompressed on upload")
    manifest = {
        "experiment": "V2-P1b (p1b_mem = P1's memory engine with the query LayerNorm; T1, T2, T4)",
        "registration": P1B.REGISTRATION,
        "branch": git("rev-parse", "--abbrev-ref", "HEAD"),
        "commit": git("rev-parse", "HEAD"),
        "uie_iso_commit": P1B.ISO_COMMIT,
        "built": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "attach_with": "V2P1B_train: uie-corpus2. V2P1B_generate: V2P1B_train's output.",
        "files": {p.relative_to(ROOT).as_posix(): sha256(p) for p in files},
        "tables": {name: want for name, (_, want) in tables.items()},
        "stream": {"file": P1B.STREAM_FILE, "rows": rows, "sha256": P1B.STREAM_SHA256},
        "e16_cap_record": sha256(RECORD),
        "e16_cap": P1B.E16_CAP_SHA256,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(OUT, "w", zipfile.ZIP_DEFLATED) as z:
        for p in files:
            z.write(p, p.relative_to(ROOT).as_posix())
        for name, (path, _) in tables.items():
            z.write(path, name)
        z.write(STREAM, P1B.STREAM_FILE)
        z.write(RECORD, P1B.E16_CAP_RECORD)
        z.writestr(P1B.CODE_MANIFEST, json.dumps(manifest, indent=1))
    print(f"wrote {OUT} ({OUT.stat().st_size / 2**20:.1f} MB), sha256 {sha256(OUT)}")
    print(f"commit {manifest['commit'][:12]}; uie/ = {P1B.ISO_COMMIT[:12]}; {len(files)} files; "
          f"stream 1 {rows:,} rows; e16_cap record {manifest['e16_cap_record'][:12]}...")


if __name__ == "__main__":
    main()
