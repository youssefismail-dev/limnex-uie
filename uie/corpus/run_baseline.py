"""
uie/corpus/run_baseline.py

Fit the E1 baseline on the frozen train split and publish its numbers.

These become the bar. E2 must beat them on held-out NLL *and* on sample quality,
with novelty reported alongside — a model that wins by memorising has not won.

Why the training sample is capped
---------------------------------
A 5-gram over 6.5M paths would hold billions of context entries. The cap is a
MEMORY bound, not a statistical one: n-gram counts saturate quickly on short
strings, so the marginal value of the millionth path is negligible while its
memory cost is not. The cap is recorded in the results so the bar is honest
about what produced it.

Evaluation is stratified, per class, on the held-out split — never on training
data, which would reward memorisation rather than measure it.
"""
from __future__ import annotations

import glob
import json
import random
import time
from pathlib import Path
from typing import Dict, List, Tuple

from uie.factorization import parse
from uie.gate import Gate
from uie.metrics import BY_SUFFIX_KIND, compare, describe
from uie.ngram import NgramURLModel
from uie.splits import load_split

#: Memory bound on the fit (see module docstring), per class.
FIT_CAP = 150_000
#: Held-out URLs scored per class. NLL is stable well below this.
EVAL_CAP = 20_000
#: Samples drawn per class for the generation-quality bar.
N_GENERATE = 20_000


def _load(corpus: Path) -> Tuple[List[str], List[str]]:
    urls: List[str] = []
    labels: List[str] = []
    for path in sorted(glob.glob(str(corpus / "shards" / "*.jsonl"))):
        with open(path, encoding="utf-8") as handle:
            for line in handle:
                urls.append(json.loads(line)["url"])
                labels.append("benign")
    phishing = corpus / "phishing.jsonl"
    if phishing.exists():
        with open(phishing, encoding="utf-8") as handle:
            for line in handle:
                urls.append(json.loads(line)["url"])
                labels.append("phishing")
    return urls, labels


def main(corpus: Path, out: Path, *, order: int = 5, seed: int = 20260803) -> None:
    rng = random.Random(seed)
    split = load_split(corpus / "split_random.json.gz")
    urls, labels = _load(corpus)
    print(f"{len(urls):,} URLs loaded")

    partitioned: Dict[str, Dict[str, List[str]]] = {}
    for url, label in zip(urls, labels):
        part = split.partition_of(url)
        if part:
            partitioned.setdefault(part, {}).setdefault(label, []).append(url)
    for part, by_class in sorted(partitioned.items()):
        print(f"  {part:<6} " + "  ".join(f"{k}={len(v):,}" for k, v in sorted(by_class.items())))

    # --- fit -------------------------------------------------------------
    fit_urls: List[str] = []
    fit_labels: List[str] = []
    for label, pool in sorted(partitioned["train"].items()):
        sample = pool if len(pool) <= FIT_CAP else rng.sample(pool, FIT_CAP)
        fit_urls.extend(sample)
        fit_labels.extend([label] * len(sample))
    started = time.time()
    model = NgramURLModel(order=order).fit(fit_urls, fit_labels)
    fit_seconds = time.time() - started
    print(f"\nfitted order-{order} on {len(fit_urls):,} URLs in {fit_seconds:.0f}s")

    results: Dict[str, object] = {
        "order": order, "seed": seed, "fit_cap": FIT_CAP,
        "n_fit": len(fit_urls), "fit_seconds": round(fit_seconds, 1),
        "held_out": {}, "generation": {}, "discrimination": {},
    }

    # --- held-out perplexity, per class and per stratum -------------------
    print("\nheld-out perplexity (per character, lower is better)")
    for label, pool in sorted(partitioned["test"].items()):
        held = pool if len(pool) <= EVAL_CAP else rng.sample(pool, EVAL_CAP)
        overall = model.perplexity(held, label)
        strata: Dict[str, float] = {}
        buckets: Dict[str, List[str]] = {}
        for url in held:
            buckets.setdefault(BY_SUFFIX_KIND(parse(url)), []).append(url)
        for name, group in sorted(buckets.items()):
            if len(group) >= 100:
                strata[name] = round(model.perplexity(group, label), 4)
        results["held_out"][label] = {"n": len(held),
                                      "perplexity": round(overall, 4),
                                      "by_suffix_kind": strata}
        print(f"  {label:<9} n={len(held):>6,}  ppl={overall:7.3f}   " +
              "  ".join(f"{k}={v:.2f}" for k, v in strata.items()))

    # --- discrimination: the generative-classifier signal ------------------
    print("\ndiscrimination via log-likelihood ratio")
    correct = total = 0
    for label, pool in sorted(partitioned["test"].items()):
        held = pool if len(pool) <= 5_000 else rng.sample(pool, 5_000)
        hits = sum(1 for u in held
                   if (model.log_prob_ratio(u) > 0) == (label == "benign"))
        results["discrimination"][label] = {"n": len(held),
                                            "accuracy": round(hits / len(held), 4)}
        correct += hits
        total += len(held)
        print(f"  {label:<9} n={len(held):>6,}  accuracy={hits/len(held):.4f}")
    results["discrimination"]["balanced_accuracy"] = round(correct / total, 4)

    # --- generation quality ------------------------------------------------
    print("\ngeneration quality")
    for label in model.classes:
        generated = model.generate(label, N_GENERATE, seed=seed)
        gate = Gate(known_urls=partitioned["train"].get(label, []))
        kept, report = gate.filter(generated)
        real = partitioned["test"].get(label, [])
        real_sample = real if len(real) <= EVAL_CAP else rng.sample(real, EVAL_CAP)
        diff = compare(describe(real_sample), describe(generated))
        results["generation"][label] = {
            "n": N_GENERATE,
            "gate_pass_rate": round(report.pass_rate, 4),
            "novelty_rate": round(report.novelty_rate, 4),
            "rejections": dict(report.by_reason),
            "distribution_vs_real": diff,
            "examples": generated[:8],
        }
        print(f"  {label:<9} valid={100*report.pass_rate:5.1f}%  "
              f"novel={100*report.novelty_rate:5.1f}%  "
              f"tv_suffix={diff['tv_suffix']:.3f}  "
              f"d_depth={diff['delta_mean_path_depth']:+.2f}")

    out.write_text(json.dumps(results, indent=1) + "\n", encoding="utf-8")
    print(f"\nwrote {out}")


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", type=Path, default=Path("data/corpus"))
    ap.add_argument("--out", type=Path, default=Path("data/corpus/baseline_e1.json"))
    ap.add_argument("--order", type=int, default=5)
    args = ap.parse_args()
    main(args.corpus, args.out, order=args.order)
