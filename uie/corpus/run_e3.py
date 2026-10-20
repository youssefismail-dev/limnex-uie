"""
uie/corpus/run_e3.py

Run the E3 measurement suite and write the report.

Order matters: the decoder ablation runs first, because every later measurement
depends on a sampling strategy and choosing one before measuring them would
prejudge the result the ablation exists to establish.
"""
from __future__ import annotations

import csv
import gzip
import json
import random
import time
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch

from uie.bloom import BloomFilter
from uie.evaluation import (diversity, embed, human_panel, precision_recall,
                            probe_representation, real_vs_fake_auc)
from uie.factorization import parse, strip_scheme
from uie.gate import Gate
from uie.metrics import BY_SUFFIX_KIND, compare, describe
from uie.model import DECODERS, generate
from uie.train import load_checkpoint

N_ABLATION = 400        # per decoder per class
N_MAIN = 1_200          # for the headline measurements
N_PANEL = 60            # per class, per side — a human has to actually do this


def _rows(path: Path) -> Dict[str, List[str]]:
    opener = gzip.open if str(path).endswith(".gz") else open
    out: Dict[str, List[str]] = {}
    with opener(path, "rt", encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            out.setdefault(row["label"], []).append(row["url"])
    return out


def _find(bundle: Path, stem: str) -> Path:
    for name in (f"{stem}.jsonl.gz", f"{stem}.jsonl"):
        if (bundle / name).is_file():
            return bundle / name
    raise FileNotFoundError(f"no {stem}.jsonl[.gz] in {bundle}")


def main(checkpoint: Path, bundle: Path, out_dir: Path, *,
         seed: int = 20260803) -> None:
    rng = random.Random(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, tokenizer, cfg = load_checkpoint(checkpoint, device=device)
    model.to(device).eval()
    out_dir.mkdir(parents=True, exist_ok=True)

    val = _rows(_find(bundle, "val"))
    train = _rows(_find(bundle, "train"))

    # Novelty is checked against EVERY training URL, not just the ones sampled
    # into the bundle. Benign is capped per label (~1.0M of 5.28M kept), so a
    # gate built from bundle rows alone would score a verbatim copy of any
    # unsampled training URL as an invention — inflating a pre-registered
    # criterion as a side effect of a sampling change.
    # Both names, for the same reason `resolve()` accepts both JSONL names:
    # Kaggle decompresses gzip members on upload, so the `.gz` suffix survives
    # locally but not there.
    bloom_path = next((bundle / name for name in
                       ("known_urls.bloom.gz", "known_urls.bloom")
                       if (bundle / name).is_file()), None)
    membership = BloomFilter.load(bloom_path) if bloom_path is not None else None
    if membership is not None:
        print(f"novelty checked against {membership.n_added:,} training URLs "
              f"(bloom, fp={membership.expected_error_rate:.4%})")
    else:
        print("WARNING: no known_urls.bloom.gz in the bundle; novelty is "
              "checked against bundle rows only and will be OVERSTATED")
    gates = {label: Gate(known_urls=urls, known_membership=membership)
             for label, urls in train.items()}
    print(f"model {sum(p.numel() for p in model.parameters())/1e6:.2f}M on {device}")
    print("val:", {k: len(v) for k, v in val.items()})

    report: Dict[str, object] = {"checkpoint": str(checkpoint), "seed": seed}

    # --- 1. decoder ablation ------------------------------------------------
    print(f"\n[1] decoder ablation ({N_ABLATION}/decoder/class)")
    print(f"  {'decoder':<14}{'class':<10}{'valid%':>8}{'novel%':>8}"
          f"{'distinct':>10}{'4gram':>8}{'tv_suf':>8}{'d_depth':>9}")
    ablation: Dict[str, Dict[str, dict]] = {}
    for name in DECODERS:
        ablation[name] = {}
        for label in tokenizer.classes:
            started = time.time()
            samples = generate(model, tokenizer, label=label, n=N_ABLATION,
                               decoder=name, seed=seed, device=device)
            _, gate_report = gates[label].filter(samples)
            div = diversity(samples)
            real = val[label]
            diff = compare(describe(rng.sample(real, min(len(real), 4000))),
                           describe(samples))
            ablation[name][label] = {
                "gate_pass_rate": round(gate_report.pass_rate, 4),
                "novelty_rate": round(gate_report.novelty_rate, 4),
                "diversity": div,
                "tv_suffix": diff["tv_suffix"],
                "delta_mean_path_depth": diff["delta_mean_path_depth"],
                "seconds": round(time.time() - started, 1),
                "examples": samples[:5],
            }
            print(f"  {name:<14}{label:<10}"
                  f"{100*gate_report.pass_rate:>7.1f}%{100*gate_report.novelty_rate:>7.1f}%"
                  f"{div['distinct_rate']:>10.3f}{div['distinct_4gram_ratio']:>8.3f}"
                  f"{diff['tv_suffix']:>8.3f}{diff['delta_mean_path_depth']:>+9.2f}",
                  flush=True)
    report["decoder_ablation"] = ablation

    # --- 2. headline measurements on the chosen decoder ---------------------
    print(f"\n[2] real-vs-fake, precision/recall, probe ({N_MAIN}/class)")
    headline: Dict[str, dict] = {}
    panel_rows: List[dict] = []
    for label in tokenizer.classes:
        fake = generate(model, tokenizer, label=label, n=N_MAIN,
                        decoder="typical", seed=seed + 1, device=device)
        real = rng.sample(val[label], min(len(val[label]), N_MAIN))

        # The tokenizer no longer emits the scheme (E2.3), so generated URLs
        # never carry one. Comparing them against real URLs that do would hand
        # the discriminator a perfect separator unrelated to sample quality.
        auc = real_vs_fake_auc([strip_scheme(u) for u in real],
                               [strip_scheme(u) for u in fake], seed=seed)
        emb_real = embed(model, tokenizer, real, label, device)
        emb_fake = embed(model, tokenizer, fake, label, device)
        pr = precision_recall(emb_real, emb_fake, k=5)

        headline[label] = {"real_vs_fake": auc, "precision_recall": pr,
                           "diversity": diversity(fake)}
        print(f"  {label:<10} real-vs-fake AUC={auc['auc_mean']:.4f}"
              f" (+/-{auc['auc_std']:.4f})   precision={pr['precision']:.3f}"
              f"  recall={pr['recall']:.3f}", flush=True)

        chosen_real = rng.sample(real, N_PANEL)
        chosen_fake = rng.sample(fake, N_PANEL)
        urls, key = human_panel([strip_scheme(u) for u in chosen_real],
                                [strip_scheme(u) for u in chosen_fake], seed=seed)
        panel_rows.extend({"class": label, "url": u, "is_generated": k}
                          for u, k in zip(urls, key))
    report["headline"] = headline

    # --- 3. representation probe -------------------------------------------
    #
    # Every URL is embedded under a SINGLE FIXED control token, so the attribute
    # being probed for was withheld from the model. Probing for something the
    # model was told — its class, its language, or the suffix it read character
    # by character — measures only that the representation retains its own
    # input, which is trivially true and would report ~98% as if it were a
    # discovery.
    print("\n[3] representation probe (linear; control tokens WITHHELD)")
    probes: Dict[str, dict] = {}

    n_probe = min(1500, len(val["benign"]), len(val["phishing"]))
    mixed = (rng.sample(val["benign"], n_probe) +
             rng.sample(val["phishing"], n_probe))
    true_class = ["benign"] * n_probe + ["phishing"] * n_probe
    # Fixed token for every row: the class is now absent from the input, so
    # recovering it means the model inferred it from the URL text itself.
    neutral = embed(model, tokenizer, mixed, "benign", device)

    checks = {
        "class (withheld)": (neutral, true_class, "inferred from text alone"),
        "suffix_kind (given)": (neutral,
                                [BY_SUFFIX_KIND(parse(u)) for u in mixed],
                                "TRIVIAL - suffix is in the input"),
    }
    for name, (vectors, values, note) in checks.items():
        result = probe_representation(vectors, values, seed=seed)
        result["note"] = note
        probes[name] = result
        print(f"  {name:<22} accuracy={result['accuracy']:.4f}  "
              f"baseline={result['baseline']:.4f}  lift={result['lift']:+.4f}"
              f"  ({result['n_classes']} classes)  [{note}]", flush=True)
    report["probes"] = probes

    # --- 4. human panel export ---------------------------------------------
    panel_path = out_dir / "human_panel.csv"
    with open(panel_path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["class", "url", "is_generated"])
        writer.writeheader()
        writer.writerows(panel_rows)
    report["human_panel"] = {"path": str(panel_path), "n": len(panel_rows),
                             "per_class_per_side": N_PANEL}
    print(f"\n[4] human panel: {len(panel_rows)} rows -> {panel_path}")

    (out_dir / "e3_report.json").write_text(
        json.dumps(report, indent=1) + "\n", encoding="utf-8")
    print(f"\nwrote {out_dir/'e3_report.json'}")


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", type=Path, default=Path("data/best.pt"))
    ap.add_argument("--bundle", type=Path, default=Path("data/kaggle_bundle"))
    ap.add_argument("--out", type=Path, default=Path("data/corpus"))
    args = ap.parse_args()
    main(args.checkpoint, args.bundle, args.out)
