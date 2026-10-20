"""E12 -- the presence-slash table (research/native/PREREG_E12.md D2). READ-ONLY: no
model is loaded.

    python research/native/e12_targets.py

For each class and suffix: how many domain-frame rows (`*name_train`, every row)
are in each of the six presence-slash states -- the gate's field extractors for
P and Q, and S = a "/" follows the host (the tokenizer's slash-plan definition,
`bool(parse(url).path)`). Free mode's supplied plan and control mode's S split
draw from it.

Checks, before anything uses it:
1. collapsing S gives E11's presence table exactly (`data/e11/presence_table.json`);
2. no row has a path without the slash (P = 1 implies S = 1);
3. the slash rates it implies -- free mode (shipped-weighted) and control mode (SC1's
   state tables) -- equal E11's registered targets (`data/e11/targets.json`).

Writes data/e12/presence_slash_table.json.
"""
from __future__ import annotations

import collections
import gzip
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from uie.evaluation import FIELDS  # noqa: E402
from uie.factorization import parse  # noqa: E402
from uie.suffixes import SuffixTable  # noqa: E402

OUT = ROOT / "data" / "e12"
STEM = {"benign": "name_train", "phishing": "phishing_name_train"}
#: the six presence-slash states, in plan order
PS_STATES = ("P0Q0S0", "P0Q0S1", "P0Q1S0", "P0Q1S1", "P1Q0S1", "P1Q1S1")
E11_STATES = ("P0Q0", "P0Q1", "P1Q0", "P1Q1")
SC1_KEY = {"P0Q0": "P0_Q0", "P0Q1": "P0_Q1"}


def rows(label):
    with gzip.open(ROOT / "data" / "bundle2" / f"{STEM[label]}.jsonl.gz", "rt",
                   encoding="utf-8") as fh:
        for line in fh:
            yield json.loads(line)["url"]


def state_of(url: str) -> str:
    p, q = bool(FIELDS["PTH"](url)), bool(FIELDS["QRY"](url))
    s = bool(parse(url).path)
    return f"P{int(p)}Q{int(q)}S{int(s)}"


def main() -> None:
    e11 = json.loads((ROOT / "data" / "e11" / "presence_table.json").read_text(encoding="utf-8"))
    targets = json.loads((ROOT / "data" / "e11" / "targets.json").read_text(encoding="utf-8"))
    shipped = SuffixTable.load(ROOT / "data" / "bundle2" / "suffix_table.json").counts
    sc1 = json.loads((ROOT / "data" / "state_suffix_tables.json").read_text(encoding="utf-8"))
    table = {}
    for label in ("benign", "phishing"):
        n = collections.defaultdict(lambda: [0] * len(PS_STATES))
        for u in rows(label):
            k = state_of(u)
            if k not in PS_STATES:
                raise SystemExit(f"STOP: {label} row in state {k} (a path without the slash): {u}")
            n[parse(u).public_suffix][PS_STATES.index(k)] += 1
        table[label] = {s: c for s, c in sorted(n.items())}
        # 1. collapsing S gives E11's presence table
        for s, c in n.items():
            collapsed = [c[0] + c[1], c[2] + c[3], c[4], c[5]]
            if collapsed != e11[label].get(s):
                raise SystemExit(f"STOP: {label} {s}: {collapsed} != E11's {e11[label].get(s)}")
        if set(n) != set(e11[label]):
            raise SystemExit(f"STOP: {label} suffix sets differ from E11's presence table")
        # 3. the implied slash rates equal E11's targets
        wsum = sum(shipped[label].values())
        for k, (i0, i1) in (("P0Q0", (0, 1)), ("P0Q1", (2, 3))):
            num = den = cnum = cden = 0.0
            for s, c in shipped[label].items():
                row = n.get(s)
                if row and (row[i0] + row[i1]):
                    w = c / wsum * (row[i0] + row[i1]) / sum(row)
                    num += w * row[i1] / (row[i0] + row[i1])
                    den += w
            for s, c in sc1[label][SC1_KEY[k]].items():
                row = n.get(s)
                if row and (row[i0] + row[i1]):
                    cnum += c * row[i1] / (row[i0] + row[i1])
                    cden += c
            free, ctrl = num / den, cnum / cden
            want = targets[label][f"slash_{k}"]
            if abs(free - want["free"]) > 1e-12 or abs(ctrl - want["control"]) > 1e-12:
                raise SystemExit(f"STOP: {label} {k} slash free {free} / control {ctrl} != "
                                 f"E11's {want['free']} / {want['control']}")
            print(f"  {label:<9} {k}: slash free {100 * free:.3f}%, control {100 * ctrl:.3f}% "
                  f"= E11's targets")
        print(f"== {label}: {sum(sum(c) for c in n.values()):,} rows, {len(n)} suffixes; "
              "collapses to E11's presence table exactly")
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "presence_slash_table.json").write_text(
        json.dumps({"states": list(PS_STATES), "table": table}), encoding="utf-8")
    print("wrote", OUT / "presence_slash_table.json")


if __name__ == "__main__":
    main()
