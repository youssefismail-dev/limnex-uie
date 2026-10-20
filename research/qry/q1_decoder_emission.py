"""Q1 -- is the QRY emission deficit the model's, or the decoder's?

Pre-registered in research/qry/PREREGISTRATION.md before this ran.

`decode_temperature` at temperature 1.0 is pure ancestral sampling: it draws from
the model's distribution with NO truncation, so its emission rate is the model's
implied marginal rate. `decode_typical` is what production uses and truncates by
distance-from-entropy. Comparing the two on identical suffix draws separates a
model defect from a decoder artifact without needing to track where in the path
the query decision was taken -- which matters, because that decision is renewed
at EVERY path character step, not taken once.

Nothing here modifies the generator. It calls the frozen decoders by name.
"""
from __future__ import annotations

import json
import math
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from uie.evaluation import FIELDS                            # noqa: E402
from uie.suffixes import SuffixTable, generate_with_suffixes  # noqa: E402
from uie.train import load_checkpoint                        # noqa: E402

N = 4000
SEED = 4242                      # fixed in advance, shared across decoders
OUT = Path("data/qry/q1_decoder_emission.json")

#: Corpus rates the predictions are stated against (E3_REPORT 30.6).
CORPUS = {"benign": {"train": 6.361, "val": 6.403, "typical_expected": 2.49},
          "phishing": {"train": 4.294, "val": 4.435, "typical_expected": 1.30}}

CONDITIONS = [
    ("typical", {}, "production control"),
    ("temperature", {"temperature": 1.0}, "TREATMENT - model's true marginal"),
    ("top_p", {"p": 0.9}, "second truncating decoder"),
    ("greedy", {}, "degenerate reference"),
    ("typical", {}, "null control - same decoder, different seed"),
]


def wilson(k: int, n: int, z: float = 1.96):
    """Wilson interval -- correct near zero, where normal approximation is not.

    Query rates here run near 1%, and at n=4000 a normal interval on 1% reaches
    below zero. That would be an instrument error reported as a measurement.
    """
    if n == 0:
        return (float("nan"), float("nan"), float("nan"))
    p = k / n
    d = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / d
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (100 * p, 100 * max(0.0, centre - half), 100 * min(1.0, centre + half))


def main() -> None:
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, tok, _ = load_checkpoint(Path("data/best_e5.pt"), device=dev)
    model.to(dev).eval()
    sfx = SuffixTable.load(Path("data/bundle2/suffix_table.json"))
    OUT.parent.mkdir(parents=True, exist_ok=True)

    results = {"n": N, "seed": SEED, "checkpoint": "data/best_e5.pt",
               "delta": 0.0, "corpus": CORPUS, "cells": []}

    for label in ("benign", "phishing"):
        print(f"\n=== {label} ===  corpus train {CORPUS[label]['train']}% "
              f"val {CORPUS[label]['val']}%", flush=True)
        for i, (decoder, kwargs, role) in enumerate(CONDITIONS):
            seed = SEED + (9999 if role.startswith("null") else 0)
            t = time.time()
            urls = generate_with_suffixes(
                model, tok, label=label, n=N, table=sfx, decoder=decoder,
                seed=seed, device=dev, decoder_kwargs=kwargs or None)
            q = sum(1 for u in urls if FIELDS["QRY"](u))
            p, lo, hi = wilson(q, len(urls))
            # Reported alongside, because a decoder that changes query rate by
            # changing PATH behaviour is telling a different story than one that
            # changes the query decision directly.
            pth = sum(1 for u in urls if FIELDS["PTH"](u))
            sub = sum(1 for u in urls if FIELDS["SUB"](u))
            row = {"class": label, "decoder": decoder, "kwargs": kwargs,
                   "role": role, "seed": seed, "n": len(urls),
                   "qry_count": q, "qry_pct": round(p, 3),
                   "qry_lo": round(lo, 3), "qry_hi": round(hi, 3),
                   "pth_pct": round(100 * pth / len(urls), 2),
                   "sub_pct": round(100 * sub / len(urls), 2),
                   "distinct": len(set(urls)),
                   "seconds": round(time.time() - t, 1)}
            results["cells"].append(row)
            print(f"  {decoder:<12} {str(kwargs):<20} QRY {q:>4}/{len(urls)} = "
                  f"{p:5.2f}% [{lo:4.2f}, {hi:4.2f}]   PTH {row['pth_pct']:5.2f}%  "
                  f"SUB {row['sub_pct']:5.2f}%  distinct {row['distinct']:>5}  "
                  f"({row['seconds']:.0f}s)  {role}", flush=True)
            OUT.write_text(json.dumps(results, indent=1) + "\n", encoding="utf-8")

    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
