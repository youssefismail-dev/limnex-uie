"""E11 -- the data side, before registration (research/native/PREREG_E11.md). READ-ONLY:
no model is loaded.

    python research/native/e11_targets.py

1. The presence table: for each class and suffix, how many domain-frame rows
   (`*name_train`, every row) are in each presence state -- no path / path, no
   query / query, by the gate's own field extractors (the tokenizer's definition:
   FINDING_LENGTH_WEIGHTING 2.2 found 0 disagreements). Free mode's supplied plan
   draws from it (option C, approved 2026-09-29).
2. The supplied-plan state tables: w_k(s) = shipped(s) * n(s, k) / n(s). Drawing a
   state k with probability sum_s w_k(s) and then a suffix from w_k is drawing the
   suffix from the shipped table and the state from the data's rate for it.
3. The root-slash targets: among rows with no path body, the share whose path is
   the bare root "/" -- bare hosts (no query) and host-route queries -- weighted as
   the generated population will be: by the shipped table times the state's rate
   (free mode), or by SC1's state table (control mode).

Writes data/e11/presence_table.json and data/e11/targets.json.
"""
from __future__ import annotations

import collections
import gzip
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from uie.evaluation import FIELDS  # noqa: E402
from uie.factorization import parse  # noqa: E402
from uie.ngram import path_body  # noqa: E402
from uie.suffixes import SuffixTable  # noqa: E402

OUT = ROOT / "data" / "e11"
STEM = {"benign": "name_train", "phishing": "phishing_name_train"}
#: presence states in plan order: (path, query)
STATES = {"P0Q0": (False, False), "P0Q1": (False, True), "P1Q0": (True, False),
          "P1Q1": (True, True)}
#: SC1's state-table keys (data/state_suffix_tables.json)
SC1_KEY = {"P0Q0": "P0_Q0", "P0Q1": "P0_Q1"}


def rows(label):
    with gzip.open(ROOT / "data" / "bundle2" / f"{STEM[label]}.jsonl.gz", "rt",
                   encoding="utf-8") as fh:
        for line in fh:
            yield json.loads(line)["url"]


def tolerance(p: float, n: float) -> float:
    """DECISION_2026-09-29 section 2: two standard errors at the gate's scale."""
    return 2 * math.sqrt(p * (1 - p) / n)


def main() -> None:
    shipped = SuffixTable.load(ROOT / "data" / "bundle2" / "suffix_table.json").counts
    sc1 = json.loads((ROOT / "data" / "state_suffix_tables.json").read_text(encoding="utf-8"))
    presence, targets = {}, {}
    for label in ("benign", "phishing"):
        n = collections.defaultdict(lambda: [0, 0, 0, 0])      # suffix -> counts by state
        slash = {k: collections.Counter() for k in ("P0Q0", "P0Q1")}   # rows with "/"
        for u in rows(label):
            parts = parse(u)
            p, q = bool(FIELDS["PTH"](u)), bool(FIELDS["QRY"](u))
            k = f"P{int(p)}Q{int(q)}"
            s = parts.public_suffix
            n[s][list(STATES).index(k)] += 1
            if not p:
                # the tokenizer's root slash: a path with an empty body
                slash[k][s] += bool(parts.path) and not path_body(parts.path)
        presence[label] = {s: c for s, c in sorted(n.items())}
        table = shipped[label]
        wsum = sum(table.values())
        miss = sum(c for s, c in table.items() if s not in n) / wsum
        # 2. state shares under the shipped table
        share = {k: sum(c / wsum * n[s][i] / sum(n[s]) for s, c in table.items() if s in n)
                 for i, k in enumerate(STATES)}
        # 3. root-slash targets
        t = {"shipped_mass_without_rows": miss, "state_share_free": share,
             "path_rate_free": share["P1Q0"] + share["P1Q1"],
             "query_rate_free": share["P0Q1"] + share["P1Q1"]}
        for k in ("P0Q0", "P0Q1"):
            i = list(STATES).index(k)
            num = den = 0.0
            for s, c in table.items():
                if s in n and n[s][i]:
                    w = c / wsum * n[s][i] / sum(n[s])            # shipped x the state's rate
                    num += w * slash[k][s] / n[s][i]
                    den += w
            free = num / den
            st = sc1[label][SC1_KEY[k]]
            cnum = cden = 0.0
            for s, c in st.items():
                if s in n and n[s][i]:
                    cnum += c * slash[k][s] / n[s][i]
                    cden += c
            ctrl = cnum / cden
            t[f"slash_{k}"] = {
                "free": free, "free_n_per_generation": 8000 * share[k],
                "free_T": tolerance(free, 8000 * share[k]),
                "control": ctrl, "control_state_table_mass_with_rows": cden / sum(st.values()),
                "rows": sum(n[s][i] for s in n), "with_slash": sum(slash[k].values())}
        targets[label] = t
        print(f"== {label}: {sum(sum(c) for c in n.values()):,} rows, {len(n)} suffixes; "
              f"shipped mass without rows {miss:.2e}")
        print("   free state shares " + ", ".join(f"{k} {100 * v:.3f}%" for k, v in share.items())
              + f"; path {100 * t['path_rate_free']:.3f}%, query {100 * t['query_rate_free']:.3f}%")
        for k in ("P0Q0", "P0Q1"):
            x = t[f"slash_{k}"]
            print(f"   root slash, {k}: free {100 * x['free']:.3f}% (n/gen {x['free_n_per_generation']:.0f},"
                  f" T ±{100 * x['free_T']:.2f}); control {100 * x['control']:.3f}%;"
                  f" raw {x['with_slash']:,} of {x['rows']:,}")
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "presence_table.json").write_text(json.dumps(presence), encoding="utf-8")
    (OUT / "targets.json").write_text(json.dumps(targets, indent=1), encoding="utf-8")
    print("wrote", OUT / "presence_table.json", "and", OUT / "targets.json")


if __name__ == "__main__":
    main()
