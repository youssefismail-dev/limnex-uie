"""E6 vs E5, 7 seeds, paired.

Single-seed excess has an observed seed-to-seed sd of 0.018-0.050 -- larger than
the analytic Hanley-McNeil SE the harness prints -- and single-seed comparison
has already manufactured two phantom regressions in this project. Every field is
therefore resampled across 7 seeds, with the SAME seeds applied to both
checkpoints so the comparison is paired.

Generation happens ONCE per checkpoint; the seeds resample the real pools, the
fake subsample and the CV folds, which is where the variance lives.
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
TMP = ROOT / "data/scratch"
N = 1200
SEEDS = [20260806 + 7 * i for i in range(7)]
FIELDS = ["LBL", "SUF", "SUB", "PTH"]


def rows(stem):
    with gzip.open(f"{B}/{stem}.jsonl.gz", "rt", encoding="utf-8") as fh:
        return [json.loads(line)["url"] for line in fh]


def gen_for(tag, ckpt):
    cache = TMP / f"{tag}_domain_gen.json"
    if cache.is_file():
        return json.loads(cache.read_text(encoding="utf-8"))
    model, tok, cfg = load_checkpoint(ckpt)
    model.eval()
    table = SuffixTable.load(B + r"/suffix_table.json")
    out = {}
    for label in ("benign", "phishing"):
        out[label] = generate_with_suffixes(
            model, tok, label=label, n=N, table=table, decoder="typical",
            frame="domain", seed=20260806 + 1)
        print(f"  generated {tag}/{label}", flush=True)
    cache.write_text(json.dumps(out), encoding="utf-8")
    return out


gens = {t: gen_for(t, p) for t, p in (
    ("e5", str(ROOT / "data/best_e5.pt")),
    ("e6", str(ROOT / "data/best_e6.pt")))}

POOL = {"benign": "name_val", "phishing": "phishing_name_val"}
pools = {k: rows(v) for k, v in POOL.items()}

print(f"\n{'field':<16}{'E5':>19}{'E6':>19}{'delta':>10}{'sigma':>8}"
      f"  verdict")
for label in ("benign", "phishing"):
    for field in FIELDS:
        vals = {}
        for tag in ("e5", "e6"):
            xs = []
            for sd in SEEDS:
                r = discriminate_with_floor(pools[label], gens[tag][label],
                                            field, n=N, seed=sd)
                if not r.get("insufficient"):
                    xs.append(r["excess"])
            vals[tag] = xs
        if len(vals["e5"]) < 4 or len(vals["e6"]) < 4:
            print(f"{label+'/'+field:<16}{'insufficient':>19}")
            continue
        m5, s5 = statistics.mean(vals["e5"]), statistics.pstdev(vals["e5"])
        m6, s6 = statistics.mean(vals["e6"]), statistics.pstdev(vals["e6"])
        d = m6 - m5
        se = ((s5 ** 2 + s6 ** 2) / len(SEEDS)) ** 0.5 or 1e-9
        sig = d / se
        verdict = ("no change" if abs(sig) < 2 else
                   ("E6 BETTER" if d < 0 else "E6 WORSE"))
        print(f"{label+'/'+field:<16}{m5:>+11.4f}+-{s5:<6.4f}"
              f"{m6:>+11.4f}+-{s6:<6.4f}{d:>+10.4f}{sig:>8.1f}  {verdict}")
