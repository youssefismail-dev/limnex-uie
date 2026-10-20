"""Variance-decomposed paired evaluation harness.

WHY THIS EXISTS. Every comparison in this project so far varied ONE source of
randomness and reported its spread as the total. Three wrong conclusions trace to
that single gap:

  1. analytic Hanley-McNeil SE (0.0167) omitted pool-resampling and CV-fold
     variance; the true seed-to-seed sd is 0.018-0.050
  2. single-seed discriminator comparisons manufactured two phantom regressions
     (benign LBL, phishing SUB), both 0.5 sigma once measured over 7 seeds
  3. single-GENERATION comparisons reported a 3.6 sigma regression on benign SUB
     from a delta that cannot causally touch subdomains

DESIGN.
  G generation seeds x D discriminator seeds per arm.
  per-generation estimate = mean over D          (averages out fit noise)
  arm estimate            = mean over G
  SE(arm)                 = sd(per-generation means) / sqrt(G)
  paired difference       = per-index differences; SE = sd(diffs) / sqrt(G)

Both variance components are reported, because on the evidence so far they are
comparable and omitting either is what went wrong.

SELF-TEST. `--null` compares an arm against ITSELF under different generation
seeds. Every field must return "no change". A harness that reports a regression
between two draws of the same distribution is measuring its own noise, and no
experiment run on it counts.
"""
from __future__ import annotations

import argparse
import gzip
import json
import statistics
import sys
from pathlib import Path
from typing import Dict, List, Optional

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from uie.evaluation import discriminate_with_floor
from uie.suffixes import SuffixTable, generate_with_suffixes
from uie.train import load_checkpoint

B = ROOT / "data/bundle2"
CACHE = ROOT / "data/scratch/harness_cache"
CACHE.mkdir(parents=True, exist_ok=True)
POOL = {"benign": "name_val", "phishing": "phishing_name_val"}
FIELDS = ("LBL", "SUF", "SUB", "PTH")

_loaded: Dict[str, tuple] = {}


def _model(ckpt: str):
    if ckpt not in _loaded:
        m, t, c = load_checkpoint(ckpt)
        m.eval()
        _loaded[ckpt] = (m, t, c)
    return _loaded[ckpt]


def generate_arm(ckpt: str, label: str, gen_seeds: List[int], n: int,
                 bias: Optional[float] = None, tag: str = "") -> List[List[str]]:
    """G independent generations. Cached on disk keyed by every input that
    affects the output, so an arm is never regenerated across experiments."""
    model, tok, cfg = _model(ckpt)
    table = SuffixTable.load(str(B / "suffix_table.json"))
    fb = {"PTH": {tok.eos_id: -bias}} if bias else None
    out = []
    for gs in gen_seeds:
        key = (f"{Path(ckpt).stem}_{label}_n{n}_b{bias or 0}_s{gs}.json")
        p = CACHE / key
        if p.is_file():
            out.append(json.loads(p.read_text(encoding="utf-8")))
            continue
        urls = generate_with_suffixes(model, tok, label=label, n=n, table=table,
                                      decoder="typical", frame="domain",
                                      field_bias=fb, seed=gs)
        p.write_text(json.dumps(urls), encoding="utf-8")
        out.append(urls)
        print(f"    generated {tag} seed {gs}", flush=True)
    return out


def per_generation(arm: List[List[str]], pool: List[str], field: str,
                   disc_seeds: List[int], n: int) -> List[float]:
    """One estimate per generation, each averaged over D discriminator seeds."""
    means = []
    for urls in arm:
        xs = []
        for ds in disc_seeds:
            r = discriminate_with_floor(pool, urls, field, n=n, seed=ds)
            if not r.get("insufficient"):
                xs.append(r["excess"])
        if xs:
            means.append((statistics.mean(xs),
                          statistics.pstdev(xs) if len(xs) > 1 else 0.0))
    return means


def compare(a: List[List[str]], b: List[List[str]], pool: List[str],
            label: str, names=("A", "B"), disc_seeds=(1, 2, 3), n=600):
    print(f"\n  {'field':<6}{names[0]:>18}{names[1]:>18}{'diff':>10}"
          f"{'sd_gen':>9}{'sd_disc':>9}{'sigma':>8}  verdict")
    worst = 0.0
    for field in FIELDS:
        ma = per_generation(a, pool, field, list(disc_seeds), n)
        mb = per_generation(b, pool, field, list(disc_seeds), n)
        if len(ma) < 3 or len(mb) < 3 or len(ma) != len(mb):
            print(f"  {field:<6}{'insufficient':>18}")
            continue
        va, vb = [x[0] for x in ma], [x[0] for x in mb]
        # paired on generation index; both arms share the same seed schedule
        diffs = [y - x for x, y in zip(va, vb)]
        d = statistics.mean(diffs)
        se = (statistics.pstdev(diffs) / (len(diffs) - 1) ** 0.5
              if len(diffs) > 1 else 0.0) or 1e-9
        sd_gen = statistics.pstdev(va)
        sd_disc = statistics.mean([x[1] for x in ma])
        sig = d / se
        worst = max(worst, abs(sig))
        v = "no change" if abs(sig) < 2 else ("B BETTER" if d < 0 else "B WORSE")
        print(f"  {field:<6}{statistics.mean(va):>+18.4f}"
              f"{statistics.mean(vb):>+18.4f}{d:>+10.4f}"
              f"{sd_gen:>9.4f}{sd_disc:>9.4f}{sig:>8.1f}  {v}")
    return worst


def rows(stem):
    with gzip.open(B / f"{stem}.jsonl.gz", "rt", encoding="utf-8") as fh:
        return [json.loads(line)["url"] for line in fh]


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--null", action="store_true",
                    help="self-test: one arm against itself, different seeds")
    ap.add_argument("--ckpt", default=str(ROOT / "data/best_e5.pt"))
    ap.add_argument("--label", default="benign")
    ap.add_argument("-G", type=int, default=5)
    ap.add_argument("-D", type=int, default=3)
    ap.add_argument("-n", type=int, default=600)
    args = ap.parse_args()

    pool = rows(POOL[args.label])
    SA = [7000 + 13 * i for i in range(args.G)]
    SB = [9000 + 13 * i for i in range(args.G)]

    if args.null:
        print(f"NULL SELF-TEST  {args.label}  G={args.G} D={args.D} n={args.n}")
        print("  same checkpoint, same config, DIFFERENT generation seeds.")
        print("  every field must read 'no change', or the harness is broken.\n")
        a = generate_arm(args.ckpt, args.label, SA, args.n, tag="A")
        b = generate_arm(args.ckpt, args.label, SB, args.n, tag="A'")
        worst = compare(a, b, pool, args.label, ("A", "A'"),
                        tuple(range(1, args.D + 1)), args.n)
        print(f"\n  worst |sigma| on a KNOWN NULL: {worst:.1f}")
        print(f"  -> harness {'PASSES' if worst < 2 else 'FAILS'}")
