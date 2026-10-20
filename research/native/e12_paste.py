"""E12_RESULTS.md -- paste Appendix A from e12_tables.py's output (data/e12/tables.md,
written by e12_readout.py), then list every number in the prose that the appendix
does not contain, so each can be checked against its declared source (the E9
practice: no number is transcribed by hand).

    python research/native/e12_paste.py
"""
from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DOC = ROOT / "research" / "native" / "E12_RESULTS.md"
TABLES = ROOT / "data" / "e12" / "tables.md"
MARK = "## Appendix A — the rendered tables"


def main() -> None:
    tables = TABLES.read_text(encoding="utf-8").rstrip("\n")
    if tables.startswith("**DRY RUN"):
        raise SystemExit("STOP: data/e12/tables.md is a dry run; it is never pasted")
    doc = DOC.read_text(encoding="utf-8")
    head = doc.split(MARK)[0].rstrip("\n")
    DOC.write_text(f"{head}\n\n{MARK}\n\n`python research/native/e12_tables.py`, verbatim.\n\n"
                   f"{tables}\n", encoding="utf-8")
    body = head.replace("−", "-")
    appendix = tables.replace("−", "-")
    numbers = sorted(set(re.findall(r"[+-]?\d+\.\d+", body)))
    missing = [n for n in numbers if n not in appendix and n.lstrip("+-") not in appendix]
    print(f"pasted {len(tables.splitlines())} lines; prose numbers {len(numbers)}; "
          f"not in the appendix ({len(missing)}): {missing}")


if __name__ == "__main__":
    main()
