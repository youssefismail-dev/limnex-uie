"""CR1 check S0: the CR1 scorer, given the canonical gate's own inputs, must
reproduce the gate v2 report exactly (PREREG_CR1.md section 3).

Compares every field's aggregates, per-draw excesses and acceptance block, and
the pooled rows, between `cr1_score.py --canonical` and the gate's report.
Two NaNs (a field insufficient in both) count as equal -- NaN != NaN in Python.

    python research/cr1/s0_compare.py data/cr1/s0/s0_report.json \
        data/hc1_c2_e7_phishing/eval_report.json phishing
"""
import json
import math
import sys

KEYS = ("auc", "floor", "excess", "excess_sd", "excess_min", "excess_max", "n",
        "draws", "draws_attempted", "excess_draws", "insufficient", "acceptance")


def same(a, b) -> bool:
    if isinstance(a, float) and isinstance(b, float) and math.isnan(a) and math.isnan(b):
        return True
    if isinstance(a, list) and isinstance(b, list):
        return len(a) == len(b) and all(same(x, y) for x, y in zip(a, b))
    if isinstance(a, dict) and isinstance(b, dict):
        return a.keys() == b.keys() and all(same(a[k], b[k]) for k in a)
    return a == b


def main() -> None:
    s0 = json.load(open(sys.argv[1], encoding="utf-8"))
    gate = json.load(open(sys.argv[2], encoding="utf-8"))[sys.argv[3]]
    bad, checked = [], 0
    for f, g in gate["fields"].items():
        for k in KEYS:
            if k in g or k in s0["fields"][f]:
                checked += 1
                if not same(g.get(k), s0["fields"][f].get(k)):
                    bad.append((f, k))
    for f, g in (gate.get("pooled_fields") or {}).items():
        for k in KEYS + ("d3_verdict",):
            if k in g:
                checked += 1
                if not same(g.get(k), s0["pooled_fields"][f].get(k)):
                    bad.append(("pooled " + f, k))
    print(f"S0: {checked} values compared; differences: {bad or 'none'}")
    print("S0:", "PASS" if not bad else "FAIL")
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
