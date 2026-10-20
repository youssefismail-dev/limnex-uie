"""How much does a field move from the GENERATION draw alone?

The delta field evaluation used ONE generation per arm and resampled only the
discriminator across 7 seeds, so its error bars capture discriminator noise and
not generation noise. It reported benign/SUB worse by 3.6 sigma -- but SUB cannot
change causally: the bias applies only inside a PTH field and subdomains are
emitted BEFORE the path. What does change is the shared RNG stream, so the biased
arm is a different DRAW.

This measures the spread of each field across generation seeds at delta=0. If
SUB's spread is comparable to the 0.0106 "regression", that number is the luck of
a single draw and the comparison must be redone with generation seeds varied.
"""
from __future__ import annotations

import gzip
import json
import statistics
import sys

from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from uie.evaluation import discriminate_with_floor
from uie.suffixes import SuffixTable, generate_with_suffixes
from uie.train import load_checkpoint

B = str(ROOT / "data/bundle2")
N, GEN_SEEDS = 1200, [101, 202, 303, 404]
model, tok, cfg = load_checkpoint(str(ROOT / "data/best_e5.pt"))
model.eval()
table = SuffixTable.load(B + r"/suffix_table.json")


def rows(stem):
    with gzip.open(f"{B}/{stem}.jsonl.gz", "rt", encoding="utf-8") as fh:
        return [json.loads(line)["url"] for line in fh]


pool = rows("name_val")
print(f"delta=0, benign, {len(GEN_SEEDS)} independent GENERATION seeds\n")
res = {f: [] for f in ("LBL", "SUB", "PTH")}
for gs in GEN_SEEDS:
    urls = generate_with_suffixes(model, tok, label="benign", n=N, table=table,
                                  decoder="typical", frame="domain", seed=gs)
    for f in res:
        r = discriminate_with_floor(pool, urls, f, n=N, seed=20260806)
        if not r.get("insufficient"):
            res[f].append(r["excess"])
    print(f"  gen seed {gs}: " + "  ".join(
        f"{f} {res[f][-1]:+.4f}" for f in res if res[f]), flush=True)

print(f"\n  {'field':<8}{'mean':>10}{'sd across draws':>18}"
      f"{'reported delta effect':>24}")
REPORTED = {"LBL": -0.0083, "SUB": +0.0106, "PTH": None}
for f, xs in res.items():
    if len(xs) < 3:
        continue
    sd = statistics.pstdev(xs)
    rep = REPORTED.get(f)
    tag = "" if rep is None else f"{rep:+.4f}  ({abs(rep)/sd:.1f}x sd)"
    print(f"  {f:<8}{statistics.mean(xs):>+10.4f}{sd:>18.4f}{tag:>24}")
