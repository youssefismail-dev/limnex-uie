"""E13_RESULTS.md -- paste the appendices from their scripts' outputs, then list every
number in the prose that no appendix contains, so each can be checked against its
declared source (the E9 practice).

    python research/e13/e13_paste.py
"""
from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DOC = ROOT / "research" / "e13" / "E13_RESULTS.md"
APPENDICES = (
    ("## Appendix A — the registered readout", "e13_tables.py", ROOT / "data" / "e13" / "tables.md"),
    ("## Appendix B — P (paths and subdomains)", "e13p_read.py", ROOT / "data" / "e13p" / "tables.md"),
)
MARK = APPENDICES[0][0]


def main() -> None:
    parts = []
    for head, script, path in APPENDICES:
        text = path.read_text(encoding="utf-8").rstrip("\n")
        if text.startswith("**DRY RUN"):
            raise SystemExit(f"STOP: {path} is a dry run; it is never pasted")
        parts.append(f"{head}\n\n`python research/e13/{script}`, verbatim.\n\n{text}")
    doc = DOC.read_text(encoding="utf-8")
    head = doc.split(MARK)[0].rstrip("\n")
    DOC.write_text(head + "\n\n" + "\n\n".join(parts) + "\n", encoding="utf-8", newline="\n")
    body = head.replace("−", "-")
    appendix = "\n".join(parts).replace("−", "-")
    numbers = sorted(set(re.findall(r"[+-]?\d[\d,]*\.\d+|\d{2,}[\d,]*", body)))
    missing = [n for n in numbers if n not in appendix and n.lstrip("+-") not in appendix]
    print(f"pasted {len(parts)} appendices; prose numbers {len(numbers)}; "
          f"not in any appendix ({len(missing)}): {missing}")


if __name__ == "__main__":
    main()
