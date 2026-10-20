"""
uie/corpus/eval_e2.py

Measure the trained transformer against the E1 bar.

Comparability
-------------
E1 reported per-CHARACTER perplexity as `exp(-total_nats / total_url_chars)`.
The same denominator is used here — the raw URL length — even though E2's
sequence contains field markers as well as characters. Both models place a
distribution over the same object (a URL string, via a lossless bijection), so
total nats are directly comparable; normalising both by the same character count
keeps the ratio meaningful and stops either model from winning by tokenising
differently.

Novelty is not optional
-----------------------
The samples look convincing, which is exactly when memorisation must be checked
rather than assumed. Generated URLs are gated against the FULL training corpus,
and the novelty rate is reported beside every quality number.
"""
from __future__ import annotations

import gzip
import json
import math
import random
from pathlib import Path
from typing import Dict, List, Sequence

import torch

from uie.gate import Gate
from uie.metrics import compare, describe
from uie.model import DECODERS, generate
from uie.train import load_checkpoint

EVAL_CAP = 8_000
N_GENERATE = 4_000


def _load_split_rows(bundle: Path, name: str) -> Dict[str, List[str]]:
    path = bundle / f"{name}.jsonl.gz"
    opener = gzip.open if path.exists() else open
    if not path.exists():
        path = bundle / f"{name}.jsonl"
    out: Dict[str, List[str]] = {}
    with opener(path, "rt", encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            out.setdefault(row["label"], []).append(row["url"])
    return out


@torch.no_grad()
def perplexity(model, tokenizer, urls: Sequence[str], label: str,
               device: torch.device, batch: int = 64) -> float:
    """Per-character perplexity, using the same denominator as E1."""
    total_nats = 0.0
    total_chars = 0
    for i in range(0, len(urls), batch):
        chunk = [u for u in urls[i:i + batch]]
        encoded = [tokenizer.encode(u, label) for u in chunk]
        keep = [(u, e) for u, e in zip(chunk, encoded) if len(e) <= model.cfg.max_len]
        if not keep:
            continue
        width = max(len(e) for _, e in keep)
        ids = torch.tensor([e + [tokenizer.pad_id] * (width - len(e))
                            for _, e in keep], dtype=torch.long, device=device)
        logp = model.sequence_log_prob(ids, tokenizer.pad_id)
        total_nats += float(logp.sum())
        total_chars += sum(len(u) for u, _ in keep)
    return math.exp(-total_nats / total_chars) if total_chars else float("nan")


def main(checkpoint: Path, bundle: Path, corpus: Path, out: Path,
         *, seed: int = 20260803) -> None:
    rng = random.Random(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, tokenizer, cfg = load_checkpoint(checkpoint, device=device)
    model.to(device).eval()
    print(f"loaded {checkpoint}  ({sum(p.numel() for p in model.parameters())/1e6:.2f}M "
          f"params, vocab {cfg.vocab_size}) on {device}")

    val = _load_split_rows(bundle, "val")
    train = _load_split_rows(bundle, "train")
    print("val:", {k: len(v) for k, v in val.items()})

    results: Dict[str, object] = {"checkpoint": str(checkpoint),
                                  "held_out": {}, "generation": {}}

    print("\nheld-out perplexity (per character, lower is better)")
    print(f"  {'class':<10}{'E2':>10}{'E1 bar':>10}{'change':>12}")
    e1_bar = {"benign": 22.050, "phishing": 18.834}
    for label, urls in sorted(val.items()):
        sample = urls if len(urls) <= EVAL_CAP else rng.sample(urls, EVAL_CAP)
        ppl = perplexity(model, tokenizer, sample, label, device)
        bar = e1_bar[label]
        results["held_out"][label] = {"n": len(sample), "perplexity": round(ppl, 4),
                                      "e1_baseline": bar,
                                      "ratio": round(ppl / bar, 4)}
        print(f"  {label:<10}{ppl:>10.3f}{bar:>10.3f}"
              f"{100*(ppl/bar-1):>11.1f}%")

    print("\ngeneration quality")
    for label in tokenizer.classes:
        samples = generate(model, tokenizer, label=label, n=N_GENERATE,
                           decoder="typical", seed=seed, device=device)
        gate = Gate(known_urls=train.get(label, []))
        kept, report = gate.filter(samples)
        real = val.get(label, [])
        real_sample = real if len(real) <= EVAL_CAP else rng.sample(real, EVAL_CAP)
        diff = compare(describe(real_sample), describe(samples))
        results["generation"][label] = {
            "n": N_GENERATE,
            "gate_pass_rate": round(report.pass_rate, 4),
            "novelty_rate": round(report.novelty_rate, 4),
            "rejections": dict(report.by_reason),
            "distribution_vs_real": diff,
            "examples": samples[:20],
        }
        print(f"  {label:<10} valid={100*report.pass_rate:5.1f}%  "
              f"novel={100*report.novelty_rate:5.1f}%  "
              f"tv_suffix={diff['tv_suffix']:.3f}  "
              f"d_depth={diff['delta_mean_path_depth']:+.2f}  "
              f"d_entropy={diff['delta_mean_label_entropy']:+.3f}")

    Path(out).write_text(json.dumps(results, indent=1) + "\n", encoding="utf-8")
    print(f"\nwrote {out}")


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", type=Path, default=Path("data/best.pt"))
    ap.add_argument("--bundle", type=Path, default=Path("data/kaggle_bundle"))
    ap.add_argument("--corpus", type=Path, default=Path("data/corpus"))
    ap.add_argument("--out", type=Path, default=Path("data/corpus/e2_eval.json"))
    args = ap.parse_args()
    main(args.checkpoint, args.bundle, args.corpus, args.out)
