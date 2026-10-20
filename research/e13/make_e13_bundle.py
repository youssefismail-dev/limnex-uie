"""Package the E13 code as a Kaggle dataset, `uie-e13-code`
(research/e13/PREREG_E13.md section 6). One bundle serves all three notebooks.

    python research/e13/make_e13_bundle.py

The same rules as the E8-E12 bundles: refuses a dirty tree under uie/,
research/native/ or research/e13/, and any table or artifact that is not the
registered one; the manifest records the commit and a hash of every file, so each
notebook proves it runs exactly this code. E13 adds:
  - e12_final's final.pt (the decoder arms' weights, N's reference and N's
    tokenizer), checked against E12's records;
  - the prefix artifact (C) and the D2 settings, each as frozen on the laptop.
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
import e13_common as E13  # noqa: E402

OUT = ROOT / "data" / "kaggle" / "uie-e13-code.zip"
NATIVE = ("e8_common.py", "e11_common.py", "e12_common.py")
E13_CODE = ("e13_common.py", "e13n_train.py")
#: Fixed at registration or by the laptop steps it names (PREREG_E13 Status).
E12_FINAL = (ROOT / "data" / "e12" / "train" / "e12_final" / "final.pt",
             "968ef622d0a6e9ba3d237778635d4930e0bbed02d240a88c0796567f7d686a17")
ARTIFACTS = {
    "presence_slash_table.json": (ROOT / "data" / "e12" / "presence_slash_table.json",
                                  "5a0eeac991b8343267d6f88f4c590bd6c208ace975dbf2218f8c5a277b71c050"),
    # the decompressed bytes of the frozen prefixes.json.gz (Kaggle decompresses .gz)
    E13.PREFIX_FILE: (ROOT / "data" / "e13" / E13.PREFIX_FILE, None),       # recorded in Status
    E13.SETTINGS_FILE: (ROOT / "data" / "e13" / E13.SETTINGS_FILE, None),   # recorded in Status
}


def main(expected: dict) -> None:
    dirty = git("status", "--porcelain", "--", "uie", "research/native", "research/e13")
    if dirty:
        sys.exit(f"refusing: uncommitted changes:\n{dirty}")
    for name, (path, want) in {**TABLES, **ARTIFACTS, "e12_final.pt": E12_FINAL}.items():
        want = want or expected.get(name)
        if not want:
            sys.exit(f"refusing: no registered hash for {name}")
        if sha256(path) != want:
            sys.exit(f"refusing: {path} is not the registered {name} ({want[:12]}...)")
    files = sorted(p for p in (ROOT / "uie").rglob("*.py") if "__pycache__" not in p.parts)
    files += [ROOT / "research" / "native" / name for name in NATIVE]
    files += [HERE / name for name in E13_CODE]
    tables = {name: (path, want or expected[name])
              for name, (path, want) in {**TABLES, **ARTIFACTS}.items()}
    manifest = {
        "experiment": "E13 (the context arm, the decoder override, N2a re-measured)",
        "registration": E13.REGISTRATION,
        "branch": git("rev-parse", "--abbrev-ref", "HEAD"),
        "commit": git("rev-parse", "HEAD"),
        "built": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "attach_with": "uie-corpus2 (the data bundle, unchanged) for E13_train",
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
        z.writestr("E13_CODE_MANIFEST.json", json.dumps(manifest, indent=1))
    print(f"wrote {OUT} ({OUT.stat().st_size / 2**20:.1f} MB), sha256 {sha256(OUT)}")
    print(f"commit {manifest['commit'][:12]}; {len(files)} files; tables {list(tables)}")


if __name__ == "__main__":
    reg = json.loads((ROOT / "data" / "e13" / "registered_hashes.json").read_text(encoding="utf-8"))
    main(reg)
