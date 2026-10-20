"""C3-QRY, C3-C -- emission and structure, descriptive (PREREG_C3_QRY section 5).

1. E7's free query rate under the Q:1 table: arm F's round-0 draws, which are
   proportional to the table's own group sizes, so rate = round-0 hits / round-0
   draws estimates sum_s p(s) P(query | s, <Q:?>). Per class and per top-10
   suffix; set against E7's rate under the shipped table (HC1 C4: 3.62% benign,
   1.74% phishing) and the real rate (6.40%, 4.44%).
2. Path presence among query-bearing URLs -- real (the Q:1 state's held-out
   pool), Q and F -- with two-proportion 95% CIs.
3. Query shape, parsed with Q3.1's own `split_params`: parameters per query,
   name length, value length p50 / p90, the share of queries with percent-
   encoding, and the numeric-value, bare-flag and hyphenated-value shares.

    python research/c3/c3_describe.py --c3 data/c3 --sc1 data/sc1 --bundle data/bundle2
"""
import argparse
import json
import statistics
import sys
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "sc1"))
sys.path.insert(0, str(HERE.parent / "cr1"))

from c3_common import G, STATE, tag  # noqa: E402

#: HC1 C4 (research/hc1/HC1_RESULTS.md section 6), quoted, not recomputed.
E7_FREE_RATE_SHIPPED = {"benign": 0.0362, "phishing": 0.0174}
REAL_RATE = {"benign": 0.0640, "phishing": 0.0444}


def quantile(xs, q):
    xs = sorted(xs)
    if not xs:
        return None
    k = (len(xs) - 1) * q
    lo, hi = int(k), min(int(k) + 1, len(xs) - 1)
    return xs[lo] + (xs[hi] - xs[lo]) * (k - lo)


def query_shape(queries) -> dict:
    """Q3.1's motivating statistics, computed with its own parser."""
    from research.qry.qtransforms import _PCT, split_params
    params = [p for q in queries for p in split_params(q)]
    values = [v for _, sep, v in params if sep == "="]
    return {"queries": len(queries),
            "params_per_query": len(params) / len(queries) if queries else None,
            "name_length": statistics.mean(len(k) for k, _, _ in params) if params else None,
            "value_length_p50": quantile([len(v) for v in values], 0.5),
            "value_length_p90": quantile([len(v) for v in values], 0.9),
            "percent_encoded_queries": sum(bool(_PCT.search(q)) for q in queries) / len(queries),
            "numeric_values": sum(v.isdigit() for v in values if v) / max(1, sum(1 for v in values if v)),
            "bare_flags": sum(1 for _, sep, _ in params if sep == "") / max(1, len(params)),
            "hyphenated_values": sum("-" in v for v in values) / max(1, len(values))}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--c3", type=Path, required=True)
    ap.add_argument("--sc1", type=Path, required=True)
    ap.add_argument("--bundle", type=Path, required=True)
    args = ap.parse_args()

    from cr1_score import pools
    from sc1_compare import prop_diff
    from uie.evaluation import FIELDS

    prov = json.loads((args.c3 / "F" / "gen" / "PROVENANCE.json").read_text(encoding="utf-8"))
    out = {"registration": "research/c3/PREREG_C3_QRY.md", "classes": {}}
    for label in ("benign", "phishing"):
        t = tag(label)
        recs = [r for r in prov["files"] if r["file"].startswith(t + "_g")]
        # K1 (PREREG section 8): the rate from the files that finished, and the
        # failing suffix, are reported; the failed arm's URLs are not described
        k1 = [f"{r['file']}: {r['error']}" for r in recs if r.get("error")]
        recs = [r for r in recs if not r.get("error")]
        d0 = sum(r["stats"]["round0_draws"] for r in recs)
        h0 = sum(r["stats"]["round0_with_query"] for r in recs)
        by = Counter(), Counter()
        for r in recs:
            for s, (d, h) in r["stats"]["by_suffix"].items():
                by[0][s] += d
                by[1][s] += h
        top = [s for s, _ in by[0].most_common(10)]
        rate = {"F_round0": h0 / d0 if d0 else None, "draws": sum(r["stats"]["draws"] for r in recs),
                "round0_draws": d0,
                "max_rounds": max((r["stats"]["rounds"] for r in recs), default=None),
                "E7_free_shipped_table (HC1 C4)": E7_FREE_RATE_SHIPPED[label],
                "real": REAL_RATE[label],
                "top10_suffix_rate": {s: by[1][s] / by[0][s] for s in top}}

        real = pools(args.bundle, label, STATE)["name"]
        urls = {"real": real}
        arms = [("Q", args.sc1 / "T" / "gen")] + ([] if k1 else [("F", args.c3 / "F" / "gen")])
        for arm, d in arms:
            urls[arm] = [u for g in range(G)
                         for u in json.loads((d / f"{t}_g{g}.json").read_text(encoding="utf-8"))]
        path = {a: sum(bool(FIELDS["PTH"](u)) for u in us) for a, us in urls.items()}
        n = {a: len(us) for a, us in urls.items()}
        paths = {"share": {a: path[a] / n[a] for a in urls}}
        pairs = [("Q", "F"), ("Q", "real"), ("F", "real")]
        paths.update({f"{a} - {b}": prop_diff(path[a], n[a], path[b], n[b])
                      for a, b in pairs if a in urls and b in urls})
        shape = {a: query_shape([FIELDS["QRY"](u) for u in us]) for a, us in urls.items()}
        out["classes"][label] = {"query_rate": rate, "path_among_query_urls": paths,
                                 "query_shape": shape}
        if k1:
            out["classes"][label].update({"k1": k1, "F": "not described: its arm failed (K1)"})
    (args.c3 / "c3c_report.json").write_text(json.dumps(out, indent=1, default=float),
                                             encoding="utf-8")
    for label, c in out["classes"].items():
        r = c["query_rate"]
        rate = "--" if r["F_round0"] is None else f"{r['F_round0']:.4f}"
        print(f"{label}: free query rate under the Q:1 table {rate} "
              f"(shipped table {r['E7_free_shipped_table (HC1 C4)']:.4f}, real {r['real']:.4f})"
              + (f"  K1: {c['k1']}" if "k1" in c else ""))
        print(f"   path among query URLs: " + ", ".join(
            f"{a} {v:.3f}" for a, v in c["path_among_query_urls"]["share"].items()))
        for a, s in c["query_shape"].items():
            print(f"   {a:<5} params {s['params_per_query']:.3f} name {s['name_length']:.2f} "
                  f"value p50/p90 {s['value_length_p50']}/{s['value_length_p90']} "
                  f"%enc {s['percent_encoded_queries']:.4f} numeric {s['numeric_values']:.3f} "
                  f"bare {s['bare_flags']:.4f} hyphen {s['hyphenated_values']:.4f}")


if __name__ == "__main__":
    main()
