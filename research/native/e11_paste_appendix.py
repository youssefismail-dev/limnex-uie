"""E11_RESULTS.md -- paste Appendix A from e11_tables.py's output, then list every
number in the prose that the appendix does not contain, so each can be checked
against its declared source (the E9 practice: no number is transcribed by hand).

    python research/native/e11_tables.py > data/e11/tables.md
    python research/native/e11_paste_appendix.py
"""
from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DOC = ROOT / "research" / "native" / "E11_RESULTS.md"
TABLES = ROOT / "data" / "e11" / "tables.md"
MARK = "## Appendix A — the rendered tables"


def main() -> None:
    doc = DOC.read_text(encoding="utf-8")
    head = doc.split(MARK)[0].rstrip("\n")
    tables = TABLES.read_text(encoding="utf-8").rstrip("\n")
    DOC.write_text(f"{head}\n\n{MARK}\n\n`python research/native/e11_tables.py`, verbatim.\n\n"
                   f"{tables}\n", encoding="utf-8")
    body = head.replace("−", "-")
    appendix = tables.replace("−", "-")
    numbers = sorted(set(re.findall(r"[+-]?\d+\.\d+", body)))
    missing = [n for n in numbers if n not in appendix and n.lstrip("+-") not in appendix]
    print(f"pasted {len(tables.splitlines())} lines; prose numbers {len(numbers)}; "
          f"not in the appendix ({len(missing)}): {missing}")


if __name__ == "__main__":
    main()
