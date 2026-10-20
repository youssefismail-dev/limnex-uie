"""E14_RESULTS.md -- paste the appendix from e14_tables.py's output, then list every
number in the prose that the appendix does not contain, so each can be checked
against its declared source (the E9 / E13 practice).

    python research/e14/e14_paste.py
"""
from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DOC = ROOT / "research" / "e14" / "E14_RESULTS.md"
MARK = "## Appendix A — the registered readout"
TABLES = ROOT / "data" / "e14" / "tables.md"


def main() -> None:
    text = TABLES.read_text(encoding="utf-8").rstrip("\n")
    part = f"{MARK}\n\n`python research/e14/e14_tables.py`, verbatim.\n\n{text}"
    doc = DOC.read_text(encoding="utf-8")
    head = doc.split(MARK)[0].rstrip("\n")
    DOC.write_bytes((head + "\n\n" + part + "\n").encode("utf-8"))
    body, appendix = head.replace("−", "-"), part.replace("−", "-")
    numbers = sorted(set(re.findall(r"[+-]?\d[\d,]*\.\d+|\d{2,}[\d,]*", body)))
    missing = [n for n in numbers if n not in appendix and n.lstrip("+-") not in appendix]
    print(f"pasted the appendix; prose numbers {len(numbers)}; not in it ({len(missing)}): {missing}")


if __name__ == "__main__":
    main()
