"""The ablation ladder: which signal is the PTH discriminator actually using?

Five models spanning 24 to 32 characters of mean path length -- straddling the
real target in both directions -- never moved PTH excess outside the 0.0141
retraining noise floor. Path length is therefore not the tell, and this asks
what is.

Runs entirely on CACHED generations, so no model is retrained and the retraining
noise floor does not apply. The live variance is generation seed and
discriminator seed, and both are suppressed by pairing: every transform is
scored on the SAME generations with the SAME discriminator seeds, so a
difference in excess can only come from the transform.

THE FLOOR IS RECOMPUTED UNDER EVERY TRANSFORM. `excess = auc - floor`, and a
transform changes real-vs-real separability too -- lowercasing makes real data
harder to tell from real data. Holding the untransformed floor would manufacture
collapses that are nothing but the floor moving, and that is the single most
likely way to get a confidently wrong answer here.
"""
from __future__ import annotations

import argparse
import gzip
import json
import random
import statistics
import sys
from pathlib import Path
from typing import Callable, Dict, List, Sequence

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from uie.evaluation import FIELDS, MIN_N, _one_per_host, real_vs_fake_auc

from research.tell.transforms import (FAMILY, TRANSFORMS, _rng, match_ttr,
                                      novel_paths, type_token_ratio)

B = ROOT / "data/bundle2"
#: DURABLE cache. The previous one lived under AppData\Local\Temp and was reaped
#: between sessions, taking every cached generation with it -- this ablation was
#: designed around reusing them and had to regenerate from scratch. Generations
#: are expensive and reproducible, so they belong in data/, not in temp.
CACHE = ROOT / "data/gen_cache"
CACHE.mkdir(parents=True, exist_ok=True)
OUT = ROOT / "data/tell"
OUT.mkdir(parents=True, exist_ok=True)
POOL = {"benign": "name_val", "phishing": "phishing_name_val"}
DELTA = {"benign": "0.22", "phishing": "0.3"}
SEEDS = [30000 + 17 * i for i in range(5)]
#: A VALID floor sits near 0.5: two disjoint real samples should be close to
#: inseparable. Departures in EITHER direction invalidate the excess computed
#: against them, and the first version of these checks only guarded the upper
#: side -- which let `rank` report excess +0.1658 against a floor of 0.294,
#: a real-vs-real AUC well BELOW chance and therefore not a baseline at all.
FLOOR_HIGH = 0.55      # floor inflated: the transform manufactured a signal
FLOOR_LOW = 0.42       # floor below chance: the fit is pathological, not a floor
DEGENERATE = 0.50      # fraction of distinct values that must survive


def _mean_depth(bodies) -> float:
    d = [len([x for x in b.split("/") if x]) for b in bodies]
    return sum(d) / len(d) if d else 0.0


def rows(stem: str) -> List[str]:
    with gzip.open(B / f"{stem}.jsonl.gz", "rt", encoding="utf-8") as fh:
        return [json.loads(line)["url"] for line in fh]


def scored(real_pool: Sequence[str], fake: Sequence[str],
           transform: Callable[[str], str], seed: int, n: int = 1800,
           set_level: str = "") -> Dict[str, float]:
    """`discriminate_with_floor`, with a transform applied after extraction.

    Deliberately a re-implementation rather than a wrapper: the transform has to
    land between field extraction and the three-way split, so that the FLOOR
    (real_a vs real_b) is computed on transformed data too. Wrapping the
    original would leave the floor measured on untransformed strings and every
    collapse would be partly an artifact of that mismatch.
    """
    extract = FIELDS["PTH"]
    rng = random.Random(seed)
    pool = [x for x in (extract(u) for u in _one_per_host(real_pool, seed)) if x]
    fakes = [x for x in (extract(u) for u in fake) if x]

    n = min(n, len(pool) // 3, len(fakes))
    if n < MIN_N or n < 1:
        return {"insufficient": True}

    picked = rng.sample(pool, 3 * n)
    a, b, c = picked[:n], picked[n:2 * n], picked[2 * n:]
    f = rng.sample(fakes, n)
    extra: Dict[str, float] = {}

    if set_level == "nmatch":
        # POWER CONTROL for `novel`. The repetition filter drops ~half the rows
        # (benign 1112 -> 597, phishing ~570 -> 274, the latter barely above
        # MIN_N). Excess depends on n, so a collapse under `novel` could be lost
        # power rather than a removed signal.
        #
        # This keeps the SAME row count `novel` would retain, chosen at random
        # and WITHOUT the repetition filter, so repetition stays intact. If
        # excess survives here but dies under `novel`, repetition was carrying
        # it; if it dies here too, the collapse was sample size all along.
        m = min(len(novel_paths(a)), len(novel_paths(b)),
                len(novel_paths(c)), len(novel_paths(f)))
        if m < MIN_N:
            return {"insufficient": True}
        pick = random.Random(seed * 7919)
        a, b, c, f = (pick.sample(a, m), pick.sample(b, m),
                      pick.sample(c, m), pick.sample(f, m))
        extra = {"kept": m}
    elif set_level == "novel":
        # F2 NECESSITY, CORRECTED. Rows are DROPPED, never rewritten, so every
        # survivor keeps its depth, delimiters, segment lengths, body length,
        # character composition, edges and words verbatim. Only repetition goes.
        #
        # This replaces `types`, which emitted bare SEGMENTS and so removed F4,
        # F5 and F6 at the same time -- a four-family ablation reported as one.
        na, nb, nc, nf = (novel_paths(a), novel_paths(b), novel_paths(c),
                          novel_paths(f))
        m = min(len(na), len(nb), len(nc), len(nf))
        if m < MIN_N:
            return {"insufficient": True, "retain_real": len(nc) / max(len(c), 1),
                    "retain_fake": len(nf) / max(len(f), 1)}
        depth_before = (_mean_depth(c), _mean_depth(f))
        a, b, c, f = na[:m], nb[:m], nc[:m], nf[:m]
        extra = {"retain_real": len(nc) / max(m and len(nc) or 1, 1),
                 "depth_real_before": depth_before[0],
                 "depth_fake_before": depth_before[1],
                 "depth_real_after": _mean_depth(c),
                 "depth_fake_after": _mean_depth(f)}
    elif set_level == "rank":
        # GATE 2, SECOND ATTEMPT -- F2 ISOLATION BY DESTRUCTION, NOT SUBSTITUTION.
        #
        # Each segment becomes "w<rank>", its frequency rank WITHIN ITS OWN SIDE.
        # Identity dies (ranks are relative, so the most common real segment and
        # the most common generated segment are both "w0"), lengths die, and
        # character composition dies. What survives is the repetition structure:
        # the shape of the frequency distribution and which documents share ranks.
        #
        # Two earlier attempts failed and both failed the same way. `match_ttr`
        # and `splice_lex` SUBSTITUTED content -- real words swapped in by hash --
        # and each manufactured a signal instead of removing one (+0.324 and
        # +0.370 against a +0.081 baseline). `freq` destroyed too much: raw
        # counts collapsed the space to a handful of strings, distinct 0.13
        # against a 0.50 floor. Ranks destroy identity while keeping one
        # distinct token per type, so the space stays wide.
        from collections import Counter

        def rank_encode(rows_):
            c = Counter(x for b in rows_ for x in b.split("/") if x)
            order = {seg: i for i, (seg, _) in
                     enumerate(sorted(c.items(), key=lambda kv: (-kv[1], kv[0])))}
            return ["/".join(f"w{order[x]}" for x in b.split("/") if x)
                    for b in rows_]

        a, b, c, f = (rank_encode(a), rank_encode(b), rank_encode(c),
                      rank_encode(f))
    elif set_level == "freq":
        # GATE 2 -- F2 ISOLATION. Every segment becomes "f<count>", its own
        # frequency inside its own set. All lexical content, all segment
        # lengths and all character composition are destroyed; the ONLY
        # surviving information is the repetition profile. Depth is kept
        # because it is the carrier the counts ride on.
        #
        # Necessity (types) says removing repetition collapses excess.
        # Sufficiency asks the converse: does repetition ALONE separate them?
        # A family is named only when both agree.
        from collections import Counter

        def freq_encode(rows_):
            c = Counter(x for b in rows_ for x in b.split("/") if x)
            return ["/".join(f"f{c[x]}" for x in b.split("/") if x)
                    for b in rows_]

        a, b, c, f = (freq_encode(a), freq_encode(b), freq_encode(c),
                      freq_encode(f))
    elif set_level == "splice_lex":
        # GATE 3 -- COUNTERFACTUAL SPLICE for F1. The GENERATED side keeps its
        # own depth, its own segment lengths and its own repetition profile,
        # but every segment is swapped for a REAL word of identical length.
        # The real side is untouched, so the floor stays the genuine
        # real-vs-real baseline, which is what a counterfactual needs.
        #
        # The donor vocabulary comes from `a` -- a real sample DISJOINT from
        # `c`, the real side actually scored. Donating out of `c` would hand
        # the generated set the very strings it is compared against and
        # collapse excess by leakage rather than by mechanism.
        #
        # Deterministic per segment, so a segment the generator repeats maps to
        # the SAME real word every time and F2 is preserved exactly. This
        # therefore isolates lexical identity and nothing else.
        from collections import defaultdict

        donor = defaultdict(list)
        for body in a:
            for x in body.split("/"):
                if x:
                    donor[len(x)].append(x)
        donor = {k: sorted(set(v)) for k, v in donor.items()}

        def relex(body):
            out_ = []
            for x in body.split("/"):
                if not x:
                    continue
                pool_ = donor.get(len(x))
                out_.append(pool_[_rng(x, "splice").randrange(len(pool_))]
                            if pool_ else x)
            return "/".join(out_) + ("/" if body.endswith("/") else "")

        f = [relex(x) for x in f]
    elif set_level == "types":
        # F2, CORRECTED. Documents become the DISTINCT segment types of each
        # side, so every type appears exactly once and document-frequency
        # asymmetry -- the whole of F2 -- is removed from both sides at once.
        # No word is altered, so F1 is untouched and a collapse here is
        # attributable to repetition alone.
        #
        # The first attempt degraded the real side's type-token ratio by
        # SUBSTITUTING repeated segments with other real segments. That changes
        # which words appear, which is an F1 change wearing an F2 label: excess
        # rose from +0.079 to +0.324 because the transform manufactured a
        # lexical signal instead of removing a repetition one. The conservation
        # tests missed it because they pinned lengths and depth but never pinned
        # the word multiset.
        def types_of(rows_):
            seen, out_ = set(), []
            for body in rows_:
                for seg in body.split("/"):
                    if seg and seg not in seen:
                        seen.add(seg)
                        out_.append(seg)
            return out_

        ta, tb, tc, tf = (types_of(a), types_of(b), types_of(c), types_of(f))
        m = min(len(ta), len(tb), len(tc), len(tf))
        if m < MIN_N:
            return {"insufficient": True}
        a, b, c, f = ta[:m], tb[:m], tc[:m], tf[:m]
    elif set_level == "match_ttr":
        # F2: degrade the REAL side to the generated side's repetition profile.
        #
        # THE THREE REAL SUBSETS SHARE ONE VOCABULARY, collapsed jointly. An
        # earlier version degraded a, b and c independently; each then drew from
        # its OWN restricted vocabulary, so the three became mutually
        # distinguishable and the floor shot to 0.907 -- the transform was
        # manufacturing the very signal it existed to remove. Collapsing the
        # concatenation and splitting it back keeps one vocabulary across all
        # three, so only repetition changes and the floor stays interpretable.
        target = type_token_ratio(f)
        joined = match_ttr(a + b + c, target)
        a, b, c = joined[:len(a)], joined[len(a):len(a) + len(b)], \
            joined[len(a) + len(b):]
    else:
        a, b, c, f = ([transform(x) for x in a], [transform(x) for x in b],
                      [transform(x) for x in c], [transform(x) for x in f])

    floor = real_vs_fake_auc(a, b, seed=seed)["auc_mean"]
    auc = real_vs_fake_auc(c, f, seed=seed)["auc_mean"]
    distinct = len(set(c + f)) / max(len(c) + len(f), 1)
    out = {"auc": auc, "floor": floor, "excess": auc - floor, "n": n,
           "kept": len(c), "distinct_frac": distinct,
           "ttr_real": type_token_ratio(c), "ttr_fake": type_token_ratio(f)}
    out.update(extra)
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--label", default="benign")
    ap.add_argument("--model", default="best_e5")
    ap.add_argument("-n", type=int, default=1800)
    ap.add_argument("-D", type=int, default=3)
    ap.add_argument("--only", default="", help="comma-separated transform names")
    a = ap.parse_args()

    d = DELTA[a.label]
    arm = []
    for s in SEEDS:
        p = CACHE / f"{a.model}_{a.label}_n{a.n}_b{d}_s{s}.json"
        if not p.is_file():
            raise FileNotFoundError(f"missing cached generation: {p}")
        arm.append(json.loads(p.read_text(encoding="utf-8")))
    pool = rows(POOL[a.label])
    print(f"PTH tell ablation  {a.model}/{a.label}  delta={d}  "
          f"G={len(arm)} D={a.D} n={a.n}", flush=True)

    names = ([x for x in a.only.split(",") if x] or list(TRANSFORMS))
    names = names + ["novel", "nmatch"]
    names = list(dict.fromkeys(names))   # --only may already name it
    results: Dict[str, List[Dict]] = {}
    for name in names:
        fn = TRANSFORMS.get(name, lambda x: x)
        per_gen = []
        for urls in arm:
            vals = [scored(pool, urls, fn, seed=ds, n=a.n,
                           set_level=(name if name in ("novel", "nmatch", "types", "rank",
                                           "freq", "splice_lex",
                                           "match_ttr") else ""))
                    for ds in range(1, a.D + 1)]
            vals = [v for v in vals if not v.get("insufficient")]
            if vals:
                per_gen.append({k: statistics.mean([v[k] for v in vals])
                                for k in vals[0]})
        results[name] = per_gen
        if per_gen:
            e = statistics.mean([v["excess"] for v in per_gen])
            fl = statistics.mean([v["floor"] for v in per_gen])
            df = statistics.mean([v["distinct_frac"] for v in per_gen])
            fam, direction = FAMILY.get(name, ("F2", "destructive  set-level"))
            print(f"  {name:<12}{fam:<10}{direction:<26}"
                  f"excess {e:>+8.4f}  floor {fl:.3f}  distinct {df:.2f}",
                  flush=True)

    path = OUT / f"{a.model}_{a.label}.json"
    path.write_text(json.dumps({"model": a.model, "label": a.label,
                                "delta": d, "n": a.n, "D": a.D,
                                "results": results}, indent=1), encoding="utf-8")
    print("wrote", path, flush=True)
