"""E15-B -- post-hoc, DESCRIPTIVE checks run after E15-B's reading, for the E16 diagnosis.
Not registered, and they license nothing. E13-D's post-hoc measures
(research/e13/e13d_posthoc.py, parts A and C), run on the old baseline (`e12_d1p`)
and the new one (`e15_new`), beside the real validation pool.

    python research/e15/e15_posthoc.py > data/e15/posthoc.md

A. Every query string against the cell's OLD training strings (the same reference for
   both arms, so the columns compare): exact copy, digit variant, never seen; and,
   at equal size, the number of distinct strings and the share the top 8 cover.
   Beside it, for `e15_new` only: never seen against old + stream (section 13.2).
   Also the mean length in characters and the mean number of parameters.
C. Free phishing: for URLs that carry a query, whether the path before it is an
   exact training path, and how the query depends on that (E13-D's part C).

Material: E14's and E15's registered files and the gate's pools. Nothing is trained
or generated.
"""
from __future__ import annotations

import collections
import json
import random
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
for p in (ROOT, HERE, ROOT / "research" / "e13", ROOT / "research" / "native", ROOT / "research" / "qc1",
          ROOT / "research" / "cr1", ROOT / "research" / "dc1"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import e13d_common as D  # noqa: E402
import e15_common as E15  # noqa: E402

GEN = {"e12_d1p": ROOT / "data" / "e14" / "decode" / "gen", E15.ARM: ROOT / "data" / "e15" / "generate" / "gen"}
STREAM = ROOT / "data" / "e15" / "qry_extra" / E15.STREAM_FILE
EQUAL_SEED = 20261011
_DIGIT = re.compile(r"\d")


def _norm(s: str) -> str:
    return _DIGIT.sub("0", s)


def _strings(label, state, field) -> set:
    f = D.field_of(field)
    return {s for s in (f(u) for u in D._urls(label, "train", state)) if s}


def stream_queries() -> set:
    f = D.field_of("QRY")
    rows = (json.loads(x) for x in STREAM.read_text(encoding="utf-8").splitlines() if x.strip())
    return {q for q in (f(r["url"]) for r in rows) if q}


def concentration(xs, k, seed) -> tuple:
    s = random.Random(seed).sample(xs, k)
    c = collections.Counter(s)
    return len(c), sum(v for _, v in c.most_common(8)) / k


def part_a() -> None:
    print("### A. Query strings against the cell's old training strings\n")
    print("| cell | source | n | exact copy | digit variant | never seen | never seen, old + stream "
          "| distinct at equal n | top-8 share at equal n | mean chars | mean params |")
    print("|---|---|---|---|---|---|---|---|---|---|---|")
    sq = stream_queries()
    for cell in D.QUERY_CELLS:
        name, _, label, state, field = cell
        train = _strings(label, state, field)
        ntrain = {_norm(s) for s in train}
        sources = [("real validation", D.pool(cell, "val"))] + [
            (arm, D.generated(cell, arm, GEN[arm])) for arm in ("e12_d1p", E15.ARM)]
        k = min(len(xs) for _, xs in sources)
        for src, xs in sources:
            n = len(xs)
            e = sum(x in train for x in xs) / n
            d = sum(x not in train and _norm(x) in ntrain for x in xs) / n
            ns = 1 - e - d
            nso = (f"{sum(x not in train and x not in sq and _norm(x) not in ntrain for x in xs) / n:.1%}"
                   if src == E15.ARM else "")
            dist, top8 = concentration(xs, k, EQUAL_SEED)
            chars = sum(len(x) for x in xs) / n
            params = sum(x.count("&") + 1 for x in xs) / n
            print(f"| {name} | {src} | {n:,} | {e:.1%} | {d:.1%} | {ns:.1%} | {nso} | {dist:,} (n {k:,}) "
                  f"| {top8:.1%} | {chars:.1f} | {params:.2f} |", flush=True)
        del train, ntrain


def part_c() -> None:
    import qc1_common as C
    from uie.evaluation import _one_per_host
    P, Q = D.field_of("PTH"), D.field_of("QRY")
    train = D._urls("phishing", "train", None)
    tp = {P(u) for u in train if P(u)}
    tq = {Q(u) for u in train if Q(u)}
    pairs = {(P(u), Q(u)) for u in train if Q(u)}
    val = _one_per_host(D._urls("phishing", "val", None), C.POOL_SEED)
    print("\n### C. Free phishing: the path before a query\n")
    print("| source | URLs | with a query | of which with a path | all paths: exact | paths before a query: exact "
          "| query: exact | (path, query) pair: exact | query exact, after an exact path | "
          "query exact, after a never-seen path |\n|---|---|---|---|---|---|---|---|---|---|")
    srcs = [("real validation (one URL per host)", val)]
    for arm in ("e12_d1p", E15.ARM):
        srcs.append((f"{arm}, free mode", [u for g in range(E15.G) for u in json.loads(
            (GEN[arm] / f"free_{arm}_phishing_g{g}.json").read_text(encoding="utf-8"))]))
    for src, urls in srcs:
        wq = [u for u in urls if Q(u)]
        wp = [u for u in wq if P(u)]
        allp = [u for u in urls if P(u)]
        seen = [u for u in wp if P(u) in tp]
        new = [u for u in wp if P(u) not in tp]
        r = lambda xs, S, f: sum(f(x) in S for x in xs) / max(1, len(xs))  # noqa: E731
        pq = sum((P(u), Q(u)) in pairs for u in wq) / len(wq)
        print(f"| {src} | {len(urls):,} | {len(wq):,} ({len(wq) / len(urls):.2%}) | {len(wp) / len(wq):.1%} | "
              f"{r(allp, tp, P):.1%} | {r(wp, tp, P):.1%} | {r(wq, tq, Q):.1%} | {pq:.1%} | "
              f"{r(seen, tq, Q):.1%} (n {len(seen):,}) | {r(new, tq, Q):.1%} (n {len(new):,}) |", flush=True)


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    print("**Post-hoc and descriptive: not a registered reading; licenses nothing.**\n")
    part_a()
    part_c()


if __name__ == "__main__":
    main()
