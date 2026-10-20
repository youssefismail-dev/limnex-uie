"""Evaluability under gate v2 (NATIVE_REPAIR_PLAN section 2) -- READ-ONLY.

For every class, free mode and every joint control state, counts the distinct
real hosts that carry each field -- exactly the pool `discriminate_with_floor`
builds (one URL per host, non-empty field) -- and the largest n it supports,
pool // 3. A cell is unevaluable when that is below MIN_N (250). Also measures
the share of generated URLs (E7 + DC1, free mode) that fall in each joint state.

    python research/native/evaluability.py
"""
import collections
import json
import sys
from pathlib import Path

R = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(R))
sys.path.insert(0, str(R / "research" / "cr1"))
from cr1_score import pools  # noqa: E402
from uie.evaluation import (FIELD_POPULATION_BY_FRAME, FIELDS, MIN_N,  # noqa: E402
                            _one_per_host)

SEED = 20260806
FIELD_NAMES = ("LBL", "SUF", "SUB", "PTH", "QRY")
STATES = {"P1Q1": {"PTH": True, "QRY": True}, "P1Q0": {"PTH": True, "QRY": False},
          "P0Q1": {"PTH": False, "QRY": True}, "P0Q0": {"PTH": False, "QRY": False}}


def field_pool(real, field):
    return sum(1 for u in _one_per_host(real, SEED) if FIELDS[field](u))


def state_of(u):
    return f"P{int(bool(FIELDS['PTH'](u)))}Q{int(bool(FIELDS['QRY'](u)))}"


def main() -> None:
    bundle = R / "data" / "bundle2"
    out = {"min_n": MIN_N, "classes": {}}
    for label in ("benign", "phishing"):
        allp = pools(bundle, label)
        row = {"real_name_urls": len(allp["name"]), "free": {}, "states": {}}
        for f in FIELD_NAMES:
            k = field_pool(allp[FIELD_POPULATION_BY_FRAME["domain"][f]], f)
            row["free"][f] = {"hosts": k, "max_n": k // 3}
        for st, spec in STATES.items():
            p = pools(bundle, label, spec)["name"]
            row["states"][st] = {"urls": len(p), "share": len(p) / len(allp["name"]),
                                 "cells": {f: {"hosts": (k := field_pool(p, f)), "max_n": k // 3}
                                           for f in FIELD_NAMES}}
        gen = collections.Counter()
        total = 0
        for g in range(5):
            path = R / "data" / "dc1" / "gen" / f"free_dc1_{label}_g{g}.json"
            for u in json.loads(path.read_text(encoding="utf-8")):
                st = state_of(u)
                gen[st] += 1
                gen[st + "_with_sub"] += bool(FIELDS["SUB"](u))
                total += 1
        row["generated_e7_dc1_free"] = {k: v / total for k, v in sorted(gen.items())}
        out["classes"][label] = row
        print(label, json.dumps(row, indent=1))
    (R / "data" / "native").mkdir(parents=True, exist_ok=True)
    (R / "data" / "native" / "evaluability.json").write_text(json.dumps(out, indent=1),
                                                             encoding="utf-8")


if __name__ == "__main__":
    main()
