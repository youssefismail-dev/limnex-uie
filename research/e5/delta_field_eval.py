"""Gate on shipping delta: full field evaluation, E5 vs E5+delta, 7 seeds paired.

Structural columns are clean (benign +2.30 chars at 5.5 sigma, depth 1.68->1.77,
P(path) flat). That is NOT sufficient. E6 also looked clean structurally and then
regressed SUB at 3.9 sigma and phishing PTH at 4.3 sigma.

A longer path changes the PTH field's CONTENT distribution, so a regression there
is plausible and must be measured. Any field worse by >2 sigma rejects delta.

delta = 0.22 benign / 0.30 phishing -- the empirical optima, applied ONLY inside
a path body.
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
DELTA = {"benign": 0.22, "phishing": 0.30}
POOL = {"benign": "name_val", "phishing": "phishing_name_val"}

model, tok, cfg = load_checkpoint(str(ROOT / "data/best_e5.pt"))
model.eval()
table = SuffixTable.load(B + r"/suffix_table.json")

base = json.loads((TMP / "e5_domain_gen.json").read_text(encoding="utf-8"))
cache = TMP / "e5_delta_gen.json"
if cache.is_file():
    biased = json.loads(cache.read_text(encoding="utf-8"))
else:
    biased = {}
    for label in ("benign", "phishing"):
        biased[label] = generate_with_suffixes(
            model, tok, label=label, n=N, table=table, decoder="typical",
            frame="domain", seed=20260806 + 1,
            field_bias={"PTH": {tok.eos_id: -DELTA[label]}})
        print(f"  generated {label} at delta {DELTA[label]}", flush=True)
    cache.write_text(json.dumps(biased), encoding="utf-8")


def rows(stem):
    with gzip.open(f"{B}/{stem}.jsonl.gz", "rt", encoding="utf-8") as fh:
        return [json.loads(line)["url"] for line in fh]


pools = {k: rows(v) for k, v in POOL.items()}
print(f"\n{'field':<16}{'E5':>19}{'E5+delta':>19}{'delta':>10}{'sigma':>8}"
      f"  verdict")
worst = 0.0
for label in ("benign", "phishing"):
    for field in ("LBL", "SUF", "SUB", "PTH"):
        got = {}
        for tag, src in (("base", base), ("bias", biased)):
            xs = []
            for sd in SEEDS:
                r = discriminate_with_floor(pools[label], src[label], field,
                                            n=N, seed=sd)
                if not r.get("insufficient"):
                    xs.append(r["excess"])
            got[tag] = xs
        if len(got["base"]) < 4 or len(got["bias"]) < 4:
            print(f"{label+'/'+field:<16}{'insufficient':>19}")
            continue
        m0, s0 = statistics.mean(got["base"]), statistics.pstdev(got["base"])
        m1, s1 = statistics.mean(got["bias"]), statistics.pstdev(got["bias"])
        d = m1 - m0
        se = ((s0 ** 2 + s1 ** 2) / len(SEEDS)) ** 0.5 or 1e-9
        sig = d / se
        worst = max(worst, sig)
        v = ("no change" if abs(sig) < 2 else
             ("BETTER" if d < 0 else "WORSE"))
        print(f"{label+'/'+field:<16}{m0:>+11.4f}+-{s0:<6.4f}"
              f"{m1:>+11.4f}+-{s1:<6.4f}{d:>+10.4f}{sig:>8.1f}  {v}")

print(f"\nworst regression: {worst:+.1f} sigma  ->  "
      f"{'ADOPT' if worst < 2 else 'REJECT'}")
