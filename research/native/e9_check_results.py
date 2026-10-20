"""Check E9_RESULTS.md's numbers against its Appendix A.

Appendix A is e9_tables.py's output, pasted by script (--paste), so the check
first confirms it still equals that output. Then every number in the text must
- appear in the appendix exactly (signs aside: the text says "rises by 0.37"
  where a table says "change -0.37"), or
- be declared in RULES, inside the passage that states it, with its source: an
  appendix number it rounds from, a formula over appendix or cited numbers, a
  committed document, or the registered code.
Tables are also checked by row: each row's numbers, signs included, must run in
order inside one appendix row. The honouring table must equal R4's row plus the
min-max of the spread table's row. Tables built from several appendix rows are
listed in EYE and were read against the appendix by eye.

Two limits remain. A right number in the wrong place in the prose passes, and
integers under 100 that are not percentages (counts of cells, positions, section
numbers) are not checked.

    python research/native/e9_check_results.py            # check
    python research/native/e9_check_results.py --paste    # (re)paste Appendix A, then check
"""
from __future__ import annotations

import argparse
import math
import re
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE))

import e9_common as K  # noqa: E402

DOC = HERE / "E9_RESULTS.md"
MARK = "`python research/native/e9_tables.py`, verbatim.\n"
STRIP = re.compile(r"`[^`\n]*`|\d{4}-\d{2}-\d{2}|\d\.\d+e[+-]\d+")
NUM = re.compile(r"(?<![\w.§])(\d[\d,]*(?:\.\d+)?)(?!\w)")
SIGNED = re.compile(r"(?<![\w.§])([-−+]?)(\d[\d,]*(?:\.\d+)?)(?!\w)")
#: tables whose rows combine several appendix rows: read by eye (header text)
EYE = ("| check | what was checked |", "| path cell, x |", "| reading | spread (max − min) |",
       "| prediction | result | scored |")
HONOUR = "range of the three old-recipe trainings"
PCT = re.compile(r"%|[–-]\d[\d.,]*%")          # "27%", or "19" in "19–21%"
EXEMPT = {"95", "1.0"}                          # the CI level; "1.0 is perfect"
F = "research/native/FINDING_LENGTH_WEIGHTING.md"
P = "research/native/PREREG_E9.md"


def z(a: float, b: float, n: int = 40_000) -> float:
    """Two rates in percent, as independent samples of n: the rise in standard errors."""
    return (b - a) / math.sqrt((a * (100 - a) + b * (100 - b)) / n)


def rnd(src):
    return ("round", src)


def der(expr):
    return ("derive", expr)


def cite(rel, passage):
    return ("cite", rel, passage)


def code(fn):
    return ("code", fn)


#: (the passage as the text states it, {number: its source})
RULES = [
    # in plain words
    ("from 58.7% to **75.6%** (benign) and from 29.6% to **38.2%** (phishing)",
     {"58.7": rnd("58.67"), "75.6": rnd("75.58"), "29.6": rnd("29.64"), "38.2": rnd("38.20")}),
    ("from 3.8% to **4.6%** (benign) and from 2.0% to **2.9%** (phishing). "
     "The targets were 6.0–7.3% and 4.2–5.2%",
     {"3.8": rnd("3.81"), "4.6": rnd("4.58"), "2.0": rnd("2.04"), "2.9": rnd("2.94"),
      "6.0": rnd("5.96"), "7.3": rnd("7.28"), "4.2": rnd("4.21"), "5.2": rnd("5.15")}),
    ("from 0.063 to **0.035**, under the 0.05 bar",
     {"0.063": rnd("0.0630"), "0.035": rnd("0.0351"),
      "0.05": cite("uie/corpus/run_eval.py", "the 0.05 bar")}),
    ("from 27–28% to 19–21% of generated queries",
     {"27": rnd("27.23"), "28": rnd("27.73"), "19": rnd("19.26"), "21": rnd("20.53")}),
    ("most 0.011 in any free gate cell", {"0.011": rnd("0.0108")}),
    # 1
    ("all 160 registered files", {"160": code(lambda: str(len(K.jobs())))}),
    ("(registered minimum 250)", {"250": cite(P, "at least 250 URLs")}),
    # 2
    ("C = 14,397.9", {"14397.9": code(lambda: f"{K.LOSS_CONSTANT:.1f}")}),
    ("by 0.0057 on the mean", {"0.0057": der("1.6017 - 1.5960")}),
    ("more than 2.4% past 1", {"2.4": der("100 * (1 - 0.976)")}),
    # 3
    ("(positions 50–100)", {"100": cite(P, "positions 50–100")}),
    ("each at least 0.5", {"0.5": cite(P, "closes ≥ 0.5")}),
    ("differ by 0.020, 0.003 and 0.047",
     {"0.020": der("1.184 - 1.164"), "0.003": der("1.061 - 1.058"),
      "0.047": der("0.892 - 0.845")}),
    ("moves 0.164, 0.040 and 0.105",
     {"0.164": der("1.164 - 1.000"), "0.040": der("1.058 - 1.018"),
      "0.105": der("0.997 - 0.892")}),
    # 4
    ("| data rate | 73.3% | 6.53% | 39.4% | 4.51% |",
     {"73.3": cite(F, "| benign | 0.733 | 0.574"), "6.53": cite(F, "| 0.0653 | 0.0445"),
      "39.4": cite(F, "| phishing | 0.394 | 0.310"), "4.51": cite(F, "| 0.0451 | 0.0251")}),
    ("| share of the gap closed | 116% | 39% | 88% | 44% |",
     {"116": der("100 * (75.58 - 58.97) / (73.3 - 58.97)"),
      "39": der("100 * (4.58 - 3.33) / (6.53 - 3.33)"),
      "88": der("100 * (38.20 - 29.75) / (39.4 - 29.75)"),
      "44": der("100 * (2.94 - 1.72) / (4.51 - 1.72)")}),
    ("116% means benign paths", {"116": der("100 * (75.58 - 58.97) / (73.3 - 58.97)")}),
    ("(75.58% against 73.3%)", {"73.3": cite(F, "| benign | 0.733 | 0.574")}),
    ("about 70% of its query shortfall", {"70": cite(F, "about 70% of its query shortfall")}),
    ("Queries closed 39% (benign) and 44% (phishing) of the gap",
     {"39": der("100 * (4.58 - 3.33) / (6.53 - 3.33)"),
      "44": der("100 * (2.94 - 1.72) / (4.51 - 1.72)")}),
    ("at the resting decode the figures are 44% and 61%",
     {"44": der("100 * (5.01 - 3.81) / (6.53 - 3.81)"),
      "61": der("100 * (3.55 - 2.04) / (4.51 - 2.04)")}),
    ("by about 0.3 to 0.6 points", {"0.3": der("2.04 - 1.72"), "0.6": der("3.55 - 2.94")}),
    ("at least 2.8 standard errors",
     {"2.8": der("min(z(3.33, 3.81), z(1.72, 2.04), z(4.58, 5.01), z(2.94, 3.55))")}),
    ("samples of 40,000 URLs", {"40000": code(lambda: str(K.G * K.N_FREE))}),
    # 5
    ("rising more than 0.0141", {"0.0141": cite(P, "by more than 0.0141")}),
    ("CLEARED → UNDECIDED, x +0.0069", {"0.0069": der("0.0473 - 0.0404")}),
    ("phishing P0Q0 LBL, +0.0115", {"0.0115": der("0.0336 - 0.0221")}),
    # 6
    ("path x (by 0.0105) and left phishing's flat (+0.0006)",
     {"0.0105": der("0.0453 - 0.0348"), "0.0006": der("0.0533 - 0.0527")}),
    ("benign by 0.0048, and phishing by 0.0138",
     {"0.0048": der("0.0126 - 0.0078"), "0.0138": der("0.0453 - 0.0315")}),
    # 7
    ("is 5.2 to 7.3 points above", {"5.2": der("86.76 - 81.56"), "7.3": der("94.95 - 87.67")}),
    ("in benign P0Q0 (by 0.96 points) and phishing P0Q0 (by 0.20). Benign P1Q1 is 0.07 below",
     {"0.96": der("97.85 - 96.89"), "0.20": der("98.15 - 97.95"),
      "0.07": der("93.74 - 93.67")}),
    # 9
    ("Q-TRAIN closure ≥ 0.5 while Q-VAL closure < 0.25",
     {"0.5": cite(P, "Q-TRAIN closure ≥ 0.5"), "0.25": cite(P, "Q-VAL closure < 0.25")}),
    ("Q-TRAIN closure 0.24; Q-VAL closure 0.30",
     {"0.24": der("1 - 0.0410 / 0.0537"), "0.30": der("1 - 0.0351 / 0.0504")}),
    ("CI above 1.10 while F2 passes", {"1.10": cite(P, "CI excluding 1.10, while F2 passes")}),
    ("requests up to 1,200", {"1200": cite("research/native/n2a_qry.py", '"benign": 1_200')}),
    ("reached the 250 queries", {"250": cite("uie/evaluation.py", "MIN_N = 250")}),
    # 10
    ("honouring +5.24 points", {"5.24": der("89.52 - 84.28")}),
    ("The 0.0141 gate-cell allowance", {"0.0141": cite(P, "by more than 0.0141")}),
    # 11, the predictions as registered
    ("F1 about 1.16, F2 about 1.06, F3 about 0.85",
     {"1.16": cite(P, "F1 plain about 1.16"), "1.06": cite(P, "F2 about 1.06"),
      "0.85": cite(P, "F3 about 0.85")}),
    ("near E7's 59% / 3.6%",
     {"59": cite(P, "E7's 59% / 3.6%"), "3.6": cite(P, "E7's 59% / 3.6%")}),
    ("near 73% / 6.6% (benign) and 39.5% / 4.7%",
     {"73": cite(P, "near 73% / 6.6%"), "6.6": cite(P, "near 73% / 6.6%"),
      "39.5": cite(P, "39.5% / 4.7%"), "4.7": cite(P, "39.5% / 4.7%")}),
]


def rendered() -> str:
    out = subprocess.run([sys.executable, str(HERE / "e9_tables.py")], cwd=ROOT,
                         capture_output=True, check=True)
    return out.stdout.decode("utf-8").replace("\r\n", "\n")


def flat(text: str) -> str:
    return " ".join(STRIP.sub(" ", text).split())


def numbers(text: str) -> list:
    """(position, token) for every checkable number in flattened text, signs dropped."""
    out = []
    for m in NUM.finditer(text):
        tok = m.group(1).rstrip(",").replace(",", "")
        pct = bool(PCT.match(text, m.end()))
        if tok in EXEMPT or ("." not in tok and not pct and int(tok) < 100):
            continue
        out.append((m.start(), tok))
    return out


def decimals(tok: str) -> int:
    return len(tok.split(".")[1]) if "." in tok else 0


def tables(text: str) -> list:
    """(header line, [(first cell, [signed numbers])]) for every markdown table."""
    out, lines = [], text.split("\n")
    for i, line in enumerate(lines[:-1]):
        if line.startswith("|") and lines[i + 1].startswith("|---") and (
                i == 0 or not lines[i - 1].startswith("|")):
            rows = []
            for row in lines[i + 2:]:
                if not row.startswith("|"):
                    break
                cells = STRIP.sub(" ", row).split("|")
                body, nums = "|".join(cells[2:]), []
                for m in SIGNED.finditer(body):
                    tok = m.group(2).rstrip(",").replace(",", "")
                    if "." in tok or int(tok) >= 100 or body[m.end():m.end() + 1] == "%":
                        nums.append(("-" if m.group(1) in ("-", "−") and m.group(1) else "") + tok)
                rows.append((cells[1].strip(), nums))
            out.append((line, rows))
    return out


def check_rows(head: str, appendix: str, appx: set) -> tuple:
    """Each doc table row against the appendix rows; returns (counts, failures)."""
    app = tables(appendix)
    every = [nums for _, rows in app for _, nums in rows]
    r4 = next(dict(rows) for hdr, rows in app if hdr.startswith("| state | e9_base_resting"))
    spread = next(dict(rows) for hdr, rows in app if hdr.startswith("| reading | E7 |"))
    counts, fails = {"in order": 0, "honouring": 0, "by eye": 0}, []
    for hdr, rows in tables(head):
        for label, nums in rows:
            nums = [n for n in nums if n.lstrip("-") in appx]    # declared-only numbers drop out
            if HONOUR in hdr:
                s = spread[f"honour {label}"][:3]
                want = r4[label] + [min(s, key=float), max(s, key=float)]
                counts["honouring"] += 1
                if nums != want:
                    fails.append(f"honouring row {label}: {nums} != {want}")
            elif any(hdr.startswith(e) for e in EYE):
                counts["by eye"] += 1
            elif len(nums) >= 2:
                counts["in order"] += 1
                if not any(nums == a[j:j + len(nums)] for a in every for j in range(len(a))):
                    fails.append(f"row {label!r}: {nums} is not in order in any appendix row")
    return counts, fails


#: (right text, planted error): all but the last must be caught (--self-test)
PLANTS = [
    ("+0.0069.", "+0.0079."),                                   # a derived value
    ("| **75.58%** |", "| **75.85%** |"),                       # a table cell from nowhere
    ("| 116% |", "| 126% |"),                                   # a derived table cell
    ("1,036 of 1,036", "1,063 of 1,063"),                       # a validity count
    ("by 0.96", "by 0.69"),                                     # derived, equal to another appendix number
    ("19.26% of 3,011", "19.62% of 3,011"),                     # a guardrail cell
    ("from 58.7% to", "from 57.8% to"),                         # a rounded plain-words figure
    ("C = 14,397.9", "C = 14,379.9"),                           # a code constant
    ("at least 2.8 standard", "at least 3.8 standard"),         # a z value
    ("(registered minimum 250)", "(registered minimum 200)"),   # a citation
    ("| +0.0073 CLEARED | +0.0068 CLEARED |", "| +0.0068 CLEARED | +0.0073 CLEARED |"),
    ("| −0.0004 CLEARED |", "| +0.0004 CLEARED |"),             # a sign flip
    ("| 93.74–96.11% |", "| 93.74–95.57% |"),                   # an honouring range
    ("Copying falls by 7.97 points", "Copying falls by 7.20 points"),  # the stated limit
]


def check(head: str, appendix: str, fresh: str) -> tuple:
    """(report lines, failures) for the text above the marker and the appendix."""
    failures = []
    if appendix.strip() != fresh.strip():
        failures.append("Appendix A differs from e9_tables.py's output (run with --paste)")
    appx = {tok for _, tok in numbers(flat(appendix))}
    text = flat(head)
    cited = set()

    def verify(tok: str, src: tuple) -> str:
        """'' if the source gives tok, else why not."""
        kind = src[0]
        if kind == "round":
            if src[1] not in appx:
                return f"{src[1]} is not in the appendix"
            return "" if f"{float(src[1]):.{decimals(tok)}f}" == tok else f"{src[1]} does not round to it"
        if kind == "cite":
            body = " ".join((ROOT / src[1]).read_text(encoding="utf-8").split())
            if src[2] not in body:
                return f"{src[2]!r} is not in {src[1]}"
            cited.add(tok)
            return ""
        if kind == "code":
            got = src[1]()
            return "" if got == tok else f"the code gives {got}"
        lits = re.findall(r"\d+\.\d+", src[1])
        missing = [x for x in lits if x not in appx and x not in cited]
        if missing:
            return f"{src[1]}: {missing} from nowhere"
        got = abs(eval(src[1], {"z": z, "min": min}))
        return "" if f"{got:.{decimals(tok)}f}" == tok else f"{src[1]} = {got:.4f}"

    ordered = sorted(RULES, key=lambda r: [s[0] for s in r[1].values()] != ["cite"] * len(r[1]))
    explained = set()
    toks = numbers(text)
    for passage, sources in ordered:                   # cites first: formulas use them
        starts = [m.start() for m in re.finditer(re.escape(passage), text)]
        if not starts:
            failures.append(f"passage not found: {passage!r}")
            continue
        for s in starts:
            inside = [(p, t) for p, t in toks if s <= p < s + len(passage)]
            for tok, src in sources.items():
                hits = [p for p, t in inside if t == tok]
                if not hits:
                    failures.append(f"{tok!r} is not in {passage!r}")
                    continue
                why = verify(tok, src)
                if why:
                    failures.append(f"{tok} in {passage!r}: {why}")
                explained.update(hits)
    for p, tok in toks:
        if p not in explained and tok not in appx:
            failures.append(f"unsourced {tok}: ...{text[max(0, p - 40):p + 20]}...")
    counts, row_fails = check_rows(head, appendix, appx)
    failures += row_fails
    report = [f"{len(toks)} numbers in the text, {len(explained)} declared; appendix "
              f"{'matches' if appendix.strip() == fresh.strip() else 'DIFFERS'}",
              f"table rows: {counts['in order']} in order in an appendix row, "
              f"{counts['honouring']} honouring rows against R4 and the spread, "
              f"{counts['by eye']} read by eye"]
    return report, failures


def self_test(head: str, appendix: str, fresh: str) -> int:
    """Plant each error in turn; every one but the last (the stated limit) must fail."""
    caught = []
    for right, wrong in PLANTS:
        if head.count(right) != 1:
            print(f"  plant target not unique in the text: {right!r}")
            return 1
        caught.append(bool(check(head.replace(right, wrong), appendix, fresh)[1]))
        print(f"  {'caught' if caught[-1] else 'missed'}  {wrong!r}")
    print(f"self-test: caught {sum(caught)} of {len(PLANTS)} planted errors")
    return 0 if all(caught[:-1]) and not caught[-1] else 1


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--paste", action="store_true", help="(re)paste Appendix A first")
    ap.add_argument("--self-test", action="store_true",
                    help="plant known errors in a copy of the text and report which are caught")
    args = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")   # a redirected Windows console is cp1252
    doc = DOC.read_text(encoding="utf-8")
    head, sep, appendix = doc.partition(MARK)
    if not sep:
        sys.exit(f"no Appendix A marker in {DOC.name}")
    fresh = rendered()
    if args.paste:
        appendix = "\n" + fresh
        DOC.write_text(head + sep + appendix, encoding="utf-8", newline="\n")
    report, failures = check(head, appendix, fresh)
    print("\n".join(report))
    print("failures:", "none" if not failures else "")
    for f in failures:
        print("  " + f)
    code = 1 if failures else 0
    if args.self_test and not failures:
        code = self_test(head, appendix, fresh)
    sys.exit(code)


if __name__ == "__main__":
    main()
