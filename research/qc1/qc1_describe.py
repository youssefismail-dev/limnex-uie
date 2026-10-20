"""QC1 -- DESCRIPTIVE (not a registered reading; licenses nothing). Two facts that
QC1_RESULTS.md section 5 uses to interpret the registered readings:

1. the head's composition: among each cell's queries found word for word in the
   training queries, which strings, generated (e11_plan) against real;
2. the training corpora's structure: rows per host, and each frame's share of the
   query-bearing training rows.

    python research/qc1/qc1_describe.py > data/qc1/describe.md
"""
from __future__ import annotations

import collections
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import qc1_common as C  # noqa: E402


def host(url: str) -> str:
    from uie.factorization import parse
    p = parse(url)
    return ".".join(tuple(p.subdomains) + (p.registrable_label, p.public_suffix))


SUBSAMPLES = 30          # Q3.1's equal-n practice: counts of distinct strings fall with n


def _tv(a: list, b: list) -> float:
    ca, cb = collections.Counter(a), collections.Counter(b)
    return 0.5 * sum(abs(ca[k] / len(a) - cb[k] / len(b)) for k in set(ca) | set(cb))


def _top8(a: list) -> float:
    return sum(c for _, c in collections.Counter(a).most_common(8)) / len(a)


def equal_n(gen: list, real: list) -> dict:
    """Distinct strings, top-8 cover and total variation at equal n (the smaller
    side), averaged over seeded subsamples; the TV baseline is real against real
    (two disjoint halves of the real copies, each of n // 2, against generated
    at the same n // 2), so sampling noise is not read as a difference."""
    import random
    import statistics as st
    n = min(len(gen), len(real))
    h = n // 2
    rows = []
    for s in range(SUBSAMPLES):
        rng = random.Random(20260930 + s)
        g, r = rng.sample(gen, n), rng.sample(real, n)
        rr = rng.sample(real, 2 * h) if len(real) >= 2 * h else real
        rows.append({"distinct_g": len(set(g)), "distinct_r": len(set(r)),
                     "top8_g": _top8(g), "top8_r": _top8(r),
                     "tv_gen_real": _tv(rng.sample(gen, h), rr[:h]), "tv_real_real": _tv(rr[:h], rr[h:])})
    return {"n": n, **{k: st.mean(x[k] for x in rows) for k in rows[0]}}


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    print("**DESCRIPTIVE: not a registered reading; it licenses nothing.**\n")
    print("### The head: queries found word for word in training, generated (`e11_plan`) "
          f"against real, at equal n ({SUBSAMPLES} seeded subsamples)\n\n"
          "| cell | n (copies per side) | distinct strings, generated / real | top 8 cover, "
          "generated / real | total variation, generated vs real / real vs real (at n/2) |\n"
          "|---|---|---|---|---|")
    fav = []
    for label in ("benign", "phishing"):
        counts = C.train_counts(label)
        for cell in [c for c in C.CELLS if c[1] == label]:
            gq = [C.query(u) for u in C.generated(C.PRIMARY, label, cell[2])]
            rq = [C.query(u) for u in C.real_pool(label, cell[2])]
            gl = [q for q in gq if q in counts]
            rl = [q for q in rq if q in counts]
            e = equal_n(gl, rl)
            print(f"| {C.cell_name(*cell)} | {e['n']:,} | {e['distinct_g']:.0f} / {e['distinct_r']:.0f} | "
                  f"{e['top8_g']:.0%} / {e['top8_r']:.0%} | {e['tv_gen_real']:.2f} / {e['tv_real_real']:.2f} |")
            gc, rc = collections.Counter(gl), collections.Counter(rl)
            gn, rn = len(gl), len(rl)
            tg, tr = gc.most_common(8), rc.most_common(8)
            fav.append((C.cell_name(*cell), [(q, c / gn, rc[q] / rn) for q, c in tg],
                        [(q, c / rn, gc[q] / gn) for q, c in tr]))
        del counts
    print("\nThe favourites, each as a share of the copied strings (generated | real):\n")
    for cell, tg, tr in fav:
        print(f"- **{cell}**, generated's top: " + ", ".join(
            f"`{q[:28]}` {g:.1%} | {r:.1%}" for q, g, r in tg))
        print(f"  - real's top: " + ", ".join(f"`{q[:28]}` {r:.1%} | {g:.1%}" for q, r, g in tr))
    print("\n### The training corpora\n\n| corpus | rows | hosts | rows per host | largest host "
          "| rows with a query |\n|---|---|---|---|---|---|")
    for pre in ("", "phishing_"):
        rows_q = {}
        for kind in ("name", "structure"):
            hosts = collections.Counter()
            k = 0
            for u in C._jsonl(f"{pre}{kind}_train"):
                hosts[host(u)] += 1
                k += bool(C.query(u))
            rows = sum(hosts.values())
            rows_q[kind] = k
            print(f"| {pre}{kind}_train | {rows:,} | {len(hosts):,} | {rows / len(hosts):.2f} | "
                  f"{hosts.most_common(1)[0][1]:,} | {k:,} |")
        print(f"| {pre or 'benign '}page-frame share of query rows | | | | | "
              f"{rows_q['structure'] / sum(rows_q.values()):.1%} |")


if __name__ == "__main__":
    main()
