"""Package the N2a code as a Kaggle dataset, `uie-n2a-code`
(research/native/PREREG_N2A.md, Amendments 3 and 4).

    python research/native/make_n2a_bundle.py

The same rules as the E8 bundle (make_code_bundle.py): refuses a dirty tree under
uie/ or research/native/ and a suffix table that is not the registered one; the
manifest records the commit and a hash of every file, so the notebook proves it
runs exactly this code.
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

OUT = ROOT / "data" / "kaggle" / "uie-n2a-code.zip"
NATIVE = ("e8_common.py", "n2a_train.py")      # the resting decode; the arms


def main() -> None:
    dirty = git("status", "--porcelain", "--", "uie", "research/native")
    if dirty:
        sys.exit(f"refusing: uncommitted changes:\n{dirty}")
    path, want = TABLES["suffix_table.json"]
    if sha256(path) != want:
        sys.exit(f"refusing: {path} is not the registered suffix table ({want[:12]}...)")
    files = sorted(p for p in (ROOT / "uie").rglob("*.py") if "__pycache__" not in p.parts)
    files += [ROOT / "research" / "native" / name for name in NATIVE]
    manifest = {
        "experiment": "N2a (diagnostic)",
        "registration": "research/native/PREREG_N2A.md (Amendments 2-4)",
        "branch": git("rev-parse", "--abbrev-ref", "HEAD"),
        "commit": git("rev-parse", "HEAD"),
        "built": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "attach_with": "uie-corpus2 (the data bundle) and E7's best.pt",
        "files": {p.relative_to(ROOT).as_posix(): sha256(p) for p in files},
        "tables": {"suffix_table.json": want},
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(OUT, "w", zipfile.ZIP_DEFLATED) as z:
        for p in files:
            z.write(p, p.relative_to(ROOT).as_posix())
        z.write(path, "suffix_table.json")
        z.writestr("N2A_CODE_MANIFEST.json", json.dumps(manifest, indent=1))
    print(f"wrote {OUT}  ({OUT.stat().st_size / 1024:.0f} KB, {len(files)} code files + 1 table)")
    print(f"commit {manifest['commit'][:12]} on {manifest['branch']}")
    print(f"sha256 {sha256(OUT)}")


if __name__ == "__main__":
    main()
