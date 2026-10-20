"""Q3 -- what makes generated queries distinguishable from real ones?

Pre-registered in research/qry/PREREGISTRATION.md. Runs on QUERY-BEARING ROWS
ONLY, both sides. A URL without a query is not evidence about query content, and
a URL with one is not evidence about emission rate; the two axes are never pooled
into one number.

Q4 established that this defect is NOT a selection artifact of the decoder --
untruncated queries are just as distinguishable (+0.1124 against +0.1325, paired
t = -1.64 on 2 df). So content is a genuine modelling failure and nothing is
known about its composition. This decomposes it.

Two halves, deliberately independent:

    A  DESCRIPTIVE   distributions of each registered component, real vs
                     generated. No discriminator, no floor, no thresholds --
                     it cannot be gamed by a transform and it says WHAT differs.

    B  ABLATION      destroy one component at a time and re-measure excess,
                     WITH THE FLOOR RECOMPUTED UNDER THE SAME TRANSFORM. Says
                     which components the tell NEEDS.

Neither half is sufficient alone. A property can differ sharply (A) and carry no
discriminative weight (B), and a collapse under ablation (B) is only interpretable
if the transform conserved everything else -- which is what
tests/test_qry_transforms.py pins.

Nothing here modifies the generator, the checkpoint, the decoder or the gate.
"""
from __future__ import annotations

import argparse
import collections
import gzip
import json
import math
import random
import statistics as st
import sys
from pathlib import Path
from typing import Dict, List, Sequence

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from research.qry.qtransforms import (COMPONENT, TRANSFORMS,  # noqa: E402
                                      split_params)
from uie.evaluation import FIELDS, MIN_N, _one_per_host  # noqa: E402
from uie.evaluation import real_vs_fake_auc              # noqa: E402

#: Inherited verbatim from research/tell/ablate.py. A valid floor sits near 0.5:
#: two disjoint REAL samples should be hard to tell apart. Outside these bounds
#: the transform has changed the measurement rather than the property, and the
#: row is reported VOID rather than interpreted -- this is the rule that caught
#: four contaminated path transforms.
FLOOR_HIGH = 0.55
FLOOR_LOW = 0.42
DEGENERATE = 0.50

SPLIT_SEEDS = (20260806, 20260807, 20260808)


# --------------------------------------------------------------------------- #
# A -- descriptive
# --------------------------------------------------------------------------- #

def entropy(s: str) -> float:
    if not s:
        return 0.0
    c = collections.Counter(s)
    n = len(s)
    return -sum((k / n) * math.log2(k / n) for k in c.values())


def describe(queries: Sequence[str]) -> Dict[str, object]:
    """Every registered component, measured. No thresholds applied here."""
    parts = [split_params(q) for q in queries]
    flat = [p for ps in parts for p in ps]
    names = [k for k, _, _ in flat]
    vals = [v for _, s, v in flat if s]

    def frac(pred, seq):
        seq = list(seq)
        return round(100 * sum(1 for x in seq if pred(x)) / max(len(seq), 1), 2)

    counts = [len(ps) for ps in parts]
    return {
        "n_queries": len(queries),
        "n_params": len(flat),
        # parameter count
        "param_count_mean": round(st.mean(counts), 3),
        "param_count_median": st.median(counts),
        "param_count_1_pct": frac(lambda c: c == 1, counts),
        "param_count_ge3_pct": frac(lambda c: c >= 3, counts),
        # key/value structure and separators
        "bare_flag_pct": frac(lambda p: p[1] == "", flat),
        "empty_value_pct": frac(lambda p: p[1] == "=" and p[2] == "", flat),
        # parameter-name identity
        "name_len_mean": round(st.mean([len(k) for k in names]), 3),
        "name_ttr": round(len(set(names)) / max(len(names), 1), 4),
        "top_names": collections.Counter(names).most_common(12),
        # parameter-value length
        "value_len_mean": round(st.mean([len(v) for v in vals]), 3) if vals else None,
        "value_len_median": st.median([len(v) for v in vals]) if vals else None,
        "value_len_p90": (sorted(len(v) for v in vals)[int(.9 * len(vals))]
                          if vals else None),
        # character-class composition
        "value_numeric_pct": frac(lambda v: v.isdigit(), vals),
        "value_alpha_pct": frac(lambda v: v.isalpha(), vals),
        "value_has_slash_pct": frac(lambda v: "/" in v, vals),
        "value_has_dot_pct": frac(lambda v: "." in v, vals),
        "value_has_hyphen_pct": frac(lambda v: "-" in v, vals),
        # percent-encoding presence
        "pct_encoded_queries_pct": frac(lambda q: "%" in q, queries),
        # ordering and repetition
        "repeated_name_queries_pct": frac(
            lambda ps: len({k for k, _, _ in ps}) != len(ps), parts),
        # value entropy
        "value_entropy_mean": round(st.mean([entropy(v) for v in vals]), 4) if vals else None,
    }


# --------------------------------------------------------------------------- #
# B -- ablation
# --------------------------------------------------------------------------- #

def scored(real_q: Sequence[str], fake_q: Sequence[str], transform,
           seed: int, n_req: int = 1800) -> Dict[str, float]:
    """discriminate_with_floor, with the transform applied AFTER extraction.

    Re-implemented rather than wrapped, for the same reason ablate.py does it:
    the transform has to land between extraction and the three-way split, so the
    FLOOR (real_a vs real_b) is computed on transformed data too. Wrapping would
    leave the floor on untransformed strings and every collapse would be partly
    an artifact of that mismatch.
    """
    rng = random.Random(seed)
    n = min(n_req, len(real_q) // 3, len(fake_q))
    if n < MIN_N or n < 1:
        return {"insufficient": True, "n": n}
    picked = rng.sample(list(real_q), 3 * n)
    a, b, c = picked[:n], picked[n:2 * n], picked[2 * n:]
    f = rng.sample(list(fake_q), n)
    a, b, c, f = ([transform(x) for x in a], [transform(x) for x in b],
                  [transform(x) for x in c], [transform(x) for x in f])
    floor = real_vs_fake_auc(a, b, seed=seed)["auc_mean"]
    auc = real_vs_fake_auc(c, f, seed=seed)["auc_mean"]
    return {"auc": auc, "floor": floor, "excess": auc - floor, "n": n,
            "distinct_frac": len(set(c + f)) / max(len(c) + len(f), 1)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--class", dest="cls", default="benign")
    ap.add_argument("-n", type=int, default=1800)
    args = ap.parse_args()

    pre = "" if args.cls == "benign" else f"{args.cls}_"
    with gzip.open(Path("data/bundle2") / f"{pre}name_val.jsonl.gz", "rt",
                   encoding="utf-8") as fh:
        real_urls = [json.loads(l)["url"] for l in fh]
    real_q = [x for x in (FIELDS["QRY"](u)
                          for u in _one_per_host(real_urls, SPLIT_SEEDS[0])) if x]

    arms = {}
    for arm, path in (("typical", f"data/qry/q4_{args.cls}_typical_queries.json"),
                      ("temperature", f"data/qry/q4_{args.cls}_temperature_queries.json")):
        urls = json.loads(Path(path).read_text())
        arms[arm] = [x for x in (FIELDS["QRY"](u) for u in urls) if x]

    print(f"=== Q3 {args.cls} ===  real {len(real_q):,} query strings; "
          + "; ".join(f"{k} {len(v)}" for k, v in arms.items()), flush=True)

    # ---------------- A: descriptive ----------------
    desc = {"real": describe(real_q)}
    for arm, q in arms.items():
        desc[arm] = describe(q)

    print("\n-- A. component distributions (query-bearing rows only) --")
    keys = [k for k in desc["real"] if k not in ("top_names",)]
    print(f"  {'metric':<28}{'real':>12}{'typical':>12}{'temperature':>14}")
    for k in keys:
        r, t, a = desc["real"][k], desc["typical"][k], desc["temperature"][k]
        print(f"  {k:<28}{str(r):>12}{str(t):>12}{str(a):>14}")
    for side in ("real", "typical"):
        print(f"\n  top names [{side}]: "
              + ", ".join(f"{n}({c})" for n, c in desc[side]["top_names"][:10]))

    # ---------------- B: ablation ----------------
    print("\n-- B. ablation ladder; floor RECOMPUTED under every transform --")
    ladder = {}
    for arm, fake_q in arms.items():
        rows = {}
        print(f"\n  [{arm}]")
        print(f"  {'transform':<16}{'component':<34}{'auc':>8}{'floor':>8}"
              f"{'excess':>9}{'R_T':>7}{'dist':>7}   status")
        base = None
        for name, fn in TRANSFORMS.items():
            draws = [scored(real_q, fake_q, fn, s, args.n) for s in SPLIT_SEEDS]
            ok = [d for d in draws if not d.get("insufficient")]
            if not ok:
                print(f"  {name:<16}{COMPONENT[name][0]:<34}"
                      f"{'--':>8}{'--':>8}{'--':>9}{'--':>7}{'--':>7}   INSUFFICIENT")
                continue
            auc = st.mean([d["auc"] for d in ok])
            floor = st.mean([d["floor"] for d in ok])
            ex = st.mean([d["excess"] for d in ok])
            dist = st.mean([d["distinct_frac"] for d in ok])
            if name == "identity":
                base = ex
            status = "ok"
            if not (FLOOR_LOW <= floor <= FLOOR_HIGH):
                status = "VOID floor"
            elif dist < DEGENERATE:
                status = "VOID degenerate"
            r_t = (ex / base) if base else float("nan")
            rows[name] = {"auc": round(auc, 4), "floor": round(floor, 4),
                          "excess": round(ex, 4), "R_T": round(r_t, 3),
                          "distinct_frac": round(dist, 3), "status": status,
                          "component": COMPONENT[name][0],
                          "direction": COMPONENT[name][1],
                          "n": ok[0]["n"], "draws": len(ok),
                          "excess_draws": [round(d["excess"], 4) for d in ok]}
            print(f"  {name:<16}{COMPONENT[name][0]:<34}{auc:>8.4f}{floor:>8.4f}"
                  f"{ex:>+9.4f}{r_t:>7.2f}{dist:>7.3f}   {status}", flush=True)
        ladder[arm] = rows

    out = Path(f"data/qry/q3_{args.cls}.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"class": args.cls, "descriptive": desc,
                               "ablation": ladder,
                               "split_seeds": list(SPLIT_SEEDS),
                               "floor_bounds": [FLOOR_LOW, FLOOR_HIGH],
                               "degenerate_bound": DEGENERATE},
                              indent=1, default=str) + "\n", encoding="utf-8")
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
