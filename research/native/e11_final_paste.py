"""E11_STEP20K_RESULTS.md -- paste Appendix A (e11_tables.py at step 20,000) and
Appendix B (e11_side_by_side.py) verbatim, then list every number in the prose
that neither appendix contains, so each can be checked against its declared
source (the E9 practice: no number is transcribed by hand).

    python research/native/e11_tables.py --e11 data/e11_final --train data/e11/train > data/e11_final/tables.md
    python research/native/e11_side_by_side.py > data/e11_final/side_by_side.md
    python research/native/e11_final_paste.py
"""
from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DOC = ROOT / "research" / "native" / "E11_STEP20K_RESULTS.md"
FINAL = ROOT / "data" / "e11_final"
MARK_A = "## Appendix A — the rendered tables at step 20,000"
MARK_B = "## Appendix B — side by side with best.pt, and amendment 3's predictions"


def main() -> None:
    doc = DOC.read_text(encoding="utf-8")
    head = doc.split(MARK_A)[0].rstrip("\n")
    a = (FINAL / "tables.md").read_text(encoding="utf-8").rstrip("\n")
    b = (FINAL / "side_by_side.md").read_text(encoding="utf-8").rstrip("\n")
    DOC.write_text(
        f"{head}\n\n{MARK_A}\n\n`python research/native/e11_tables.py --e11 data/e11_final "
        f"--train data/e11/train`, verbatim.\n\n{a}\n\n{MARK_B}\n\n"
        f"`python research/native/e11_side_by_side.py`, verbatim.\n\n{b}\n", encoding="utf-8")
    body = head.replace("−", "-")
    appendix = (a + "\n" + b).replace("−", "-")
    numbers = sorted(set(re.findall(r"[+-]?\d+\.\d+", body)))
    missing = [n for n in numbers if n not in appendix and n.lstrip("+-") not in appendix]
    print(f"pasted {len(a.splitlines())} + {len(b.splitlines())} lines; prose numbers "
          f"{len(numbers)}; not in the appendices ({len(missing)}): {missing}")


if __name__ == "__main__":
    main()
