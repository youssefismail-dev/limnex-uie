"""E13-D -- post-hoc, DESCRIPTIVE checks run after DX8's reading. Not registered, and
they license nothing. They answer DECISION_2026-10-01's diagnosis target 1 ("copying
moved further below the natural rate") on E12, and test a caveat on DX3's learner.

    python research/e13/e13d_posthoc.py > data/e13d/posthoc.md

A. Every string against the cell's training strings: an exact copy; a digit variant
   (equal once every digit is mapped to 0); one edit away (the two share a string
   with one character deleted: a substitution, insertion or deletion, and some
   transpositions); or never seen. Real validation (the gate's pool) is the natural
   rate; the transformer is e12_final's five generations; the 6-gram is DX3's
   learner on the full training twin, 3,000 samples.
B. Of the strings that are not copies of the cell's own training strings, the share
   that copies another population's: other states of the class, or the other class.
C. Free phishing: for URLs that carry a query, whether the path before it is an
   exact training path, and how the query depends on that.

Material: E12's registered files and the gate's pools only. Nothing is trained or
generated.
"""
from __future__ import annotations

import gc
import json
import random
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import e13d_common as D  # noqa: E402

NGRAM_SAMPLES, NGRAM_SEED = 3000, D.SEEDS[0] + 17
_DIGIT = re.compile(r"\d")


def _norm(s: str) -> str:
    return _DIGIT.sub("0", s)


def _dels(s: str) -> set:
    return {s[:i] + s[i + 1:] for i in range(len(s))}


def _strings(label, state, field) -> set:
    f = D.field_of(field)
    return {s for s in (f(u) for u in D._urls(label, "train", state)) if s}


def _classes(xs, train, ntrain, dindex) -> list:
    c = [0, 0, 0, 0]
    for s in xs:
        if s in train:
            c[0] += 1
        elif _norm(s) in ntrain:
            c[1] += 1
        elif s in dindex or any(d in train or d in dindex for d in _dels(s)):
            c[2] += 1
        else:
            c[3] += 1
    return [v / len(xs) for v in c]


def part_a() -> None:
    print("### A. Exact copies, digit variants, one-edit near misses, never seen (query cells)\n")
    print("| cell | source | n | exact copy | digit variant | one edit | never seen |\n|---|---|---|---|---|---|---|")
    for cell in D.QUERY_CELLS:
        name, _, label, state, field = cell
        train = _strings(label, state, field)
        ntrain = {_norm(s) for s in train}
        dindex = set()
        for t in train:
            dindex |= _dels(t)
        sources = (("real validation", D.pool(cell, "val")),
                   ("transformer (e12_final)", D.generated(cell, D.PRIMARY)),
                   ("6-gram (DX3's learner)", D.CharNGram().fit(D.pool(cell, "train"))
                    .sample(NGRAM_SAMPLES, random.Random(NGRAM_SEED))))
        for src, xs in sources:
            e, d, o, n = _classes(xs, train, ntrain, dindex)
            print(f"| {name} | {src} | {len(xs):,} | {e:.1%} | {d:.1%} | {o:.1%} | {n:.1%} |", flush=True)
        del train, ntrain, dindex
        gc.collect()


def part_b() -> None:
    print("\n### B. Copies of another population's training strings\n")
    print("| cell | source | own cell | other states | other class | none of these |\n|---|---|---|---|---|---|")
    for cell in D.QUERY_CELLS:
        name, _, label, state, field = cell
        own = _strings(label, state, field)
        same = _strings(label, None, field) - own
        other = _strings("benign" if label == "phishing" else "phishing", None, field) - own - same
        for src, xs in (("real validation", D.pool(cell, "val")),
                        ("transformer (e12_final)", D.generated(cell, D.PRIMARY))):
            n = len(xs)
            e = sum(x in own for x in xs)
            s = sum(x not in own and x in same for x in xs)
            o = sum(x not in own and x in other for x in xs)
            print(f"| {name} | {src} | {e / n:.1%} | {s / n:.1%} | {o / n:.1%} | {(n - e - s - o) / n:.1%} |",
                  flush=True)
        del own, same, other
        gc.collect()


def part_c() -> None:
    import e12_common as K
    import qc1_common as C
    from uie.evaluation import _one_per_host
    P, Q = D.field_of("PTH"), D.field_of("QRY")
    train = D._urls("phishing", "train", None)
    tp = {P(u) for u in train if P(u)}
    tq = {Q(u) for u in train if Q(u)}
    pairs = {(P(u), Q(u)) for u in train if Q(u)}
    val = _one_per_host(D._urls("phishing", "val", None), C.POOL_SEED)
    gen = [u for g in range(K.G) for u in
           json.loads((D.E12 / "gen" / f"free_{D.PRIMARY}_phishing_g{g}.json").read_text(encoding="utf-8"))]
    print("\n### C. Free phishing: the path before a query\n")
    print("| source | URLs | with a query | of which with a path | all paths: exact | paths before a query: exact "
          "| query: exact | (path, query) pair: exact | query exact, after an exact path | "
          "query exact, after a never-seen path |\n|---|---|---|---|---|---|---|---|---|---|")
    for src, urls in (("real validation (one URL per host)", val), (f"{D.PRIMARY}, free mode", gen)):
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
    part_b()
    part_c()


if __name__ == "__main__":
    main()
