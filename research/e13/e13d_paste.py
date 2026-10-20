"""E13D_RESULTS.md -- paste the four appendices from their scripts' outputs, then list
every number in the prose that no appendix contains, so each can be checked against
its declared source (the E9 practice).

    python research/e13/e13d_paste.py

A, B and C are the registered readouts; D is the post-hoc descriptive script. Each
output is pasted verbatim; a dry run is never pasted.
"""
from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DOC = ROOT / "research" / "e13" / "E13D_RESULTS.md"
OUT = ROOT / "data" / "e13d"
APPENDICES = (
    ("## Appendix A — the rendered tables", "e13d_read.py", "tables.md"),
    ("## Appendix B — DX7 (amendment 2)", "e13d_read_dx7.py", "tables_dx7.md"),
    ("## Appendix C — DX8 (amendment 3)", "e13d_read_dx8.py", "tables_dx8.md"),
    ("## Appendix D — post-hoc, descriptive (not registered; licenses nothing)", "e13d_posthoc.py",
     "posthoc.md"),
)
MARK = APPENDICES[0][0]


def main() -> None:
    parts = []
    for head, script, name in APPENDICES:
        text = (OUT / name).read_text(encoding="utf-8").rstrip("\n")
        if text.startswith("**DRY RUN"):
            raise SystemExit(f"STOP: data/e13d/{name} is a dry run; it is never pasted")
        parts.append(f"{head}\n\n`python research/e13/{script}`, verbatim.\n\n{text}")
    doc = DOC.read_text(encoding="utf-8")
    head = doc.split(MARK)[0].rstrip("\n")
    DOC.write_text(head + "\n\n" + "\n\n".join(parts) + "\n", encoding="utf-8", newline="\n")
    body = head.replace("−", "-")
    appendix = "\n".join(parts).replace("−", "-")
    numbers = sorted(set(re.findall(r"[+-]?\d+\.\d+", body)))
    missing = [n for n in numbers if n not in appendix and n.lstrip("+-") not in appendix]
    print(f"pasted {len(parts)} appendices, {sum(len(p.splitlines()) for p in parts)} lines; "
          f"prose numbers {len(numbers)}; not in any appendix ({len(missing)}): {missing}")


if __name__ == "__main__":
    main()
