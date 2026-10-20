"""D2: is the short-segment deficit a SELECTION effect rather than truncation?

Arithmetic to explain: real segments average 15.92 characters, generated 11.65
(-27%), while DEPTH matches exactly (1.78 vs 1.79).

D2 says the model over-samples the frequent head of the segment vocabulary, and
head words are inherently short (`tag`, `en`, `blog`), so mean length falls with
P("/") perfectly calibrated and no truncation anywhere. Supporting evidence
already measured: generated segments are 56.4% in-vocabulary against a 48.5%
real floor -- the model over-uses known segments.

METHOD, pure counting, no generation:
  1. bucket the real segment vocabulary by frequency rank
  2. measure the mean real segment length within each bucket
  3. measure which buckets the MODEL draws from
  4. predicted = sum_b P_model(b) * mean_len_real(b)

  predicted ~ observed  ->  D2 explains it; the deficit is selection, and the
                            problem belongs with uppercase mode collapse
  observed  < predicted ->  the model ALSO truncates; residual is for D1/D3
"""
from __future__ import annotations

import collections
import gzip
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from uie.factorization import parse

B = str(ROOT / "data/bundle2")
TMP = ROOT / "data/scratch"
gen = json.loads((TMP / "e5_domain_gen.json").read_text(encoding="utf-8"))


def segs(u):
    return [s for s in (parse(u).path or "").split("/") if s]


def rows(stem, limit=200000):
    out = []
    with gzip.open(f"{B}/{stem}.jsonl.gz", "rt", encoding="utf-8") as fh:
        for line in fh:
            out.append(json.loads(line)["url"])
            if len(out) >= limit:
                break
    return out


# The vocabulary is built from TRAIN and the real side measured from VAL.
# Measuring the real side against a vocabulary built from itself makes it 0%
# novel by construction while the model shows 39.1% -- not a comparison. Real
# held-out data contains novel segments too, and they are long, so that flaw
# inflates the apparent model-vs-real gap in exactly the direction being tested.
for label, vstem, stem in (("benign", "name_train", "name_val"),
                           ("phishing", "phishing_name_train",
                            "phishing_name_val")):
    vocab_segs = [s for u in rows(vstem) for s in segs(u)]
    real_segs = [s for u in rows(stem) for s in segs(u)]
    gen_segs = [s for u in gen[label] for s in segs(u)]
    freq = collections.Counter(s.lower() for s in vocab_segs)

    # Frequency buckets over the real vocabulary, by rank.
    ranked = [w for w, _ in freq.most_common()]
    rank_of = {w: i for i, w in enumerate(ranked)}
    EDGES = [(0, 100), (100, 1000), (1000, 10000), (10000, 100000),
             (100000, 10 ** 9)]
    NAMES = ["top 100", "100-1k", "1k-10k", "10k-100k", "100k+"]

    def bucket(w):
        r = rank_of.get(w.lower())
        if r is None:
            return "novel"
        for (lo, hi), nm in zip(EDGES, NAMES):
            if lo <= r < hi:
                return nm
        return NAMES[-1]

    real_by, gen_by = collections.defaultdict(list), collections.defaultdict(list)
    for s in real_segs:
        real_by[bucket(s)].append(len(s))
    for s in gen_segs:
        gen_by[bucket(s)].append(len(s))

    rn, gn = len(real_segs), len(gen_segs)
    print(f"\n{'='*86}\n=== {label} ===   real segments {rn:,}   "
          f"generated {gn:,}   vocab {len(freq):,} types")
    print(f"  {'bucket':<12}{'real share':>12}{'real len':>10}"
          f"{'model share':>13}{'model len':>11}{'len ratio':>11}")
    order = NAMES + ["novel"]
    for b in order:
        r, g = real_by.get(b, []), gen_by.get(b, [])
        if not r and not g:
            continue
        rl = sum(r) / len(r) if r else 0
        gl = sum(g) / len(g) if g else 0
        print(f"  {b:<12}{len(r)/rn:>12.1%}{rl:>10.2f}{len(g)/gn:>13.1%}"
              f"{gl:>11.2f}{(gl/rl if rl else 0):>11.2f}")

    obs_real = sum(map(len, real_segs)) / rn
    obs_gen = sum(map(len, gen_segs)) / gn
    # D2 prediction: model's bucket mix, real lengths within each bucket.
    pred = sum((len(gen_by.get(b, [])) / gn) *
               (sum(real_by[b]) / len(real_by[b]) if real_by.get(b) else 0)
               for b in order)
    # Control: real bucket mix, model's lengths within each bucket.
    ctrl = sum((len(real_by.get(b, [])) / rn) *
               (sum(gen_by[b]) / len(gen_by[b]) if gen_by.get(b) else 0)
               for b in order)

    print(f"\n  observed real segment length            {obs_real:>7.2f}")
    print(f"  observed model segment length           {obs_gen:>7.2f}"
          f"   deficit {obs_gen-obs_real:>+6.2f}")
    print(f"  D2 PREDICTION (model mix, real lengths) {pred:>7.2f}"
          f"   explains {(obs_real-pred)/(obs_real-obs_gen)*100 if obs_real!=obs_gen else 0:>5.1f}% of it")
    print(f"  control (real mix, model lengths)       {ctrl:>7.2f}")
