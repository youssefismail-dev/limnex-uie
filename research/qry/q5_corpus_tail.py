"""Q5 -- is the TRAINING corpus tail-poor, or is the model failing to learn it?

READ-ONLY. Opens corpus files and the already-harvested generated queries. No
generation, no model load, no training, no gate. Nothing is modified.

Q3.1 established that generated query content under-produces the long tail, in
parameter names (TTR 0.6023 real vs 0.3626 generated at equal n) and in value
lengths (p90 25 vs 15 at an identical median). That is a statement about the
model's OUTPUT versus HELD-OUT data. It does not say which of two very different
worlds we are in:

    corpus tail-RICH, model tail-poor  -> a genuine learning/decoding failure,
                                          and a generator change is meaningful
    corpus tail-POOR                   -> the model is reproducing its training
                                          distribution faithfully and NO
                                          generator change is warranted at all

Q1 set the precedent for this check: train and validation query RATES were
verified to agree (6.361% vs 6.403%) before the emission deficit was called a
model defect, and that verification is why the finding survived.

EVERY METRIC HERE IS SAMPLE-SIZE DEPENDENT, so every metric is computed at a
COMMON n with repeated subsamples. That is not a precaution in the abstract: the
raw-TTR comparison in Q3 pointed the wrong way for exactly this reason and had
to be withdrawn. Populations here differ by three orders of magnitude
(906,214 training rows against 476 generated queries), so an uncontrolled
comparison would be measuring corpus size.
"""
from __future__ import annotations

import collections
import gzip
import json
import random
import statistics as st
import sys
from pathlib import Path
from typing import Dict, List, Sequence

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from research.qry.qtransforms import split_params   # noqa: E402
from uie.evaluation import FIELDS, _one_per_host    # noqa: E402

DRAWS = 30
SEED = 20260806
B = Path("data/bundle2")


def queries_from(stem: str, dedup: bool = True, cap: int = 0) -> List[str]:
    """Query strings from a corpus file, host-deduped like the canonical path.

    Dedup matters: a multi-page site contributing forty near-identical query
    strings would inflate concentration and deflate TTR for reasons that have
    nothing to do with the corpus's tail. `discriminate_with_floor` dedups by
    host for the same reason, so this matches the treatment the val numbers in
    Q3.1 already received.
    """
    with gzip.open(B / f"{stem}.jsonl.gz", "rt", encoding="utf-8") as fh:
        urls = [json.loads(l)["url"] for i, l in enumerate(fh)
                if not cap or i < cap]
    if dedup:
        urls = _one_per_host(urls, SEED)
    return [x for x in (FIELDS["QRY"](u) for u in urls) if x]


def shape(queries: Sequence[str]) -> Dict[str, object]:
    parts = [split_params(q) for q in queries]
    return {
        "names": [k for ps in parts for k, _, _ in ps],
        "values": [v for ps in parts for _, s, v in ps if s],
        "counts": [len(ps) for ps in parts],
        "n_queries": len(queries),
    }


def at_n(pool: Dict[str, object], n_names: int, n_queries: int,
         rng: random.Random) -> Dict[str, float]:
    """Every statistic on a subsample of the SAME size, so sizes cannot leak in."""
    nm = rng.sample(pool["names"], n_names)
    c = collections.Counter(nm)
    tot = len(nm)
    qi = rng.sample(range(pool["n_queries"]), n_queries)
    counts = [pool["counts"][i] for i in qi]
    vals = rng.sample(pool["values"], min(n_names, len(pool["values"])))
    lens = sorted(len(v) for v in vals)
    return {
        "name_ttr": len(set(nm)) / tot,
        "top1": 100 * c.most_common(1)[0][1] / tot,
        "top5": 100 * sum(v for _, v in c.most_common(5)) / tot,
        "top20": 100 * sum(v for _, v in c.most_common(20)) / tot,
        "value_p90": lens[int(0.90 * len(lens))],
        "value_p99": lens[int(0.99 * len(lens))],
        "value_median": lens[len(lens) // 2],
        "value_mean": st.mean(lens),
        "param_count_mean": st.mean(counts),
        "param_ge3_pct": 100 * sum(1 for x in counts if x >= 3) / len(counts),
    }


def main() -> None:
    pools: Dict[str, Dict[str, object]] = {}

    print("reading corpora (read-only) ...", flush=True)
    for label, stem in (("TRAIN name", "name_train"),
                        ("TRAIN structure", "structure_train"),
                        ("VAL name", "name_val")):
        q = queries_from(stem)
        pools[label] = shape(q)
        print(f"  {label:<18}{len(q):>9,} query strings, "
              f"{len(pools[label]['names']):>9,} parameters", flush=True)

    for arm in ("typical", "temperature"):
        urls = json.loads(Path(f"data/qry/q4_benign_{arm}_queries.json").read_text())
        q = [x for x in (FIELDS["QRY"](u) for u in urls) if x]
        pools[f"GEN {arm}"] = shape(q)
        print(f"  {'GEN ' + arm:<18}{len(q):>9,} query strings, "
              f"{len(pools[f'GEN ' + arm]['names']):>9,} parameters")

    n_names = min(len(p["names"]) for p in pools.values())
    n_q = min(p["n_queries"] for p in pools.values())
    print(f"\ncommon subsample: {n_names} parameters / {n_q} queries, "
          f"{DRAWS} draws  (mean +- sd)\n", flush=True)

    rng = random.Random(SEED)
    rows: Dict[str, Dict[str, tuple]] = {}
    for label, pool in pools.items():
        draws = [at_n(pool, n_names, n_q, rng) for _ in range(DRAWS)]
        rows[label] = {k: (st.mean([d[k] for d in draws]),
                           st.pstdev([d[k] for d in draws]))
                       for k in draws[0]}

    METRICS = [("name_ttr", "name TTR", 4), ("top1", "top-1 name %", 2),
               ("top5", "top-5 name %", 2), ("top20", "top-20 name %", 2),
               ("value_median", "value len median", 1),
               ("value_p90", "value len p90", 1),
               ("value_p99", "value len p99", 1),
               ("param_count_mean", "param count mean", 3),
               ("param_ge3_pct", "param count >=3 %", 2)]

    hdr = f"  {'metric':<20}" + "".join(f"{lbl:>20}" for lbl in rows)
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))
    for key, name, prec in METRICS:
        line = f"  {name:<20}"
        for lbl in rows:
            m, s = rows[lbl][key]
            line += f"{m:>14.{prec}f}+-{s:<4.{prec}f}"[:20].rjust(20)
        print(line)

    out = Path("data/qry/q5_corpus_tail.json")
    out.write_text(json.dumps(
        {"common_n_parameters": n_names, "common_n_queries": n_q,
         "draws": DRAWS, "seed": SEED,
         "raw_sizes": {k: {"queries": v["n_queries"], "parameters": len(v["names"])}
                       for k, v in pools.items()},
         "at_common_n": {k: {m: [round(a, 6), round(b, 6)] for m, (a, b) in v.items()}
                         for k, v in rows.items()}},
        indent=1) + "\n", encoding="utf-8")
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
