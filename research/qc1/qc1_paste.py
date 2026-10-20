"""QC1_RESULTS.md -- paste Appendix A (qc1_read.py, the registered readout) and
Appendix B (qc1_describe.py, descriptive) verbatim, then list every number in the
prose that neither appendix contains, so each can be checked against its declared
source (the E9 practice: no number is transcribed by hand).

    python research/qc1/qc1_read.py > data/qc1/tables.md
    python research/qc1/qc1_describe.py > data/qc1/describe.md
    python research/qc1/qc1_paste.py
"""
from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DOC = ROOT / "research" / "qc1" / "QC1_RESULTS.md"
MARK_A = "## Appendix A — the registered readout"
MARK_B = "## Appendix B — descriptive"


def main() -> None:
    head = DOC.read_text(encoding="utf-8").split(MARK_A)[0].rstrip("\n")
    a = (ROOT / "data" / "qc1" / "tables.md").read_text(encoding="utf-8").rstrip("\n")
    b = (ROOT / "data" / "qc1" / "describe.md").read_text(encoding="utf-8").rstrip("\n")
    DOC.write_text(f"{head}\n\n{MARK_A}\n\n`python research/qc1/qc1_read.py`, verbatim.\n\n{a}\n\n"
                   f"{MARK_B}\n\n`python research/qc1/qc1_describe.py`, verbatim.\n\n{b}\n",
                   encoding="utf-8")
    body, appendix = head.replace("−", "-"), (a + "\n" + b).replace("−", "-")
    numbers = sorted(set(re.findall(r"[+-]?\d+\.\d+", body)))
    missing = [n for n in numbers if n not in appendix and n.lstrip("+-") not in appendix]
    print(f"pasted {len(a.splitlines())} + {len(b.splitlines())} lines; prose numbers "
          f"{len(numbers)}; not in the appendices ({len(missing)}): {missing}")


if __name__ == "__main__":
    main()
