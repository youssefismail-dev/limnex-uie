"""Regenerate E5+delta arms into the durable cache. Same seeds, same settings."""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from uie.suffixes import SuffixTable, generate_with_suffixes
from uie.train import load_checkpoint

B = ROOT / "data/bundle2"
CACHE = ROOT / "data/gen_cache"
CACHE.mkdir(parents=True, exist_ok=True)
DELTA = {"benign": 0.22, "phishing": 0.30}
SEEDS = [30000 + 17 * i for i in range(5)]

ap = argparse.ArgumentParser()
ap.add_argument("--label", required=True)
ap.add_argument("--ckpt", default=str(ROOT / "data/best_e5.pt"))
ap.add_argument("-n", type=int, default=1800)
a = ap.parse_args()

m, t, _ = load_checkpoint(a.ckpt); m.eval()
table = SuffixTable.load(str(B / "suffix_table.json"))
d = DELTA[a.label]
stem = Path(a.ckpt).stem
for s in SEEDS:
    p = CACHE / f"{stem}_{a.label}_n{a.n}_b{d}_s{s}.json"
    if p.is_file():
        print(f"  cached {s}", flush=True); continue
    urls = generate_with_suffixes(m, t, label=a.label, n=a.n, table=table,
                                  decoder="typical", frame="domain",
                                  field_bias={"PTH": {t.eos_id: -d}}, seed=s)
    p.write_text(json.dumps(urls), encoding="utf-8")
    print(f"  generated {s}", flush=True)
print("done", flush=True)
