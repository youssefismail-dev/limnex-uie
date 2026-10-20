"""Definitive calibration sweep: 5 seeds x 5 delta per class.

The single-seed version was non-monotonic (benign delta 0.08 -> 23.46 but the
derived 0.15 -> 22.21), and biased configs averaged +1.98 chars over baseline at
~1.9 sigma. That is under-powered: one seed, when single-seed measurement has
already produced two phantom regressions in this project.

Each (delta, seed) is an independent generation, so the reported sd is the true
seed-to-seed spread rather than an analytic approximation. Results are flushed
per row so a partial run is still usable.
"""
from __future__ import annotations

import gzip
import json
import statistics
import sys

from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from uie.factorization import parse
from uie.suffixes import SuffixTable, generate_with_suffixes
from uie.train import load_checkpoint

B = str(ROOT / "data/bundle2")
N = 500
SEEDS = [20260810 + 11 * i for i in range(5)]
GRID = {"benign": (0.0, 0.08, 0.15, 0.22, 0.30),
        "phishing": (0.0, 0.06, 0.12, 0.20, 0.30)}
DERIVED = {"benign": 0.15, "phishing": 0.06}

model, tok, cfg = load_checkpoint(str(ROOT / "data/best_e5.pt"))
model.eval()
table = SuffixTable.load(B + r"/suffix_table.json")
FIT = {"benign": "name_train", "phishing": "phishing_name_train"}


def pb(u):
    return "/".join(s for s in (parse(u).path or "").split("/") if s)


def m(urls):
    nz = [b for b in (pb(u) for u in urls) if b]
    d = [len([s for s in b.split("/") if s]) for b in nz]
    return (sum(map(len, nz)) / len(nz) if nz else 0,
            sum(d) / len(d) if d else 0,
            len(nz) / len(urls))


for label in ("benign", "phishing"):
    urls = []
    with gzip.open(f"{B}/{FIT[label]}.jsonl.gz", "rt", encoding="utf-8") as fh:
        for line in fh:
            urls.append(json.loads(line)["url"])
            if len(urls) >= 40000:
                break
    tl, td, tp = m(urls)
    print(f"\n=== {label} ===  target len {tl:.2f}  depth {td:.2f}  "
          f"P(path) {tp:.1%}   derived delta {DERIVED[label]}", flush=True)
    print(f"  {'delta':<8}{'len|path':>18}{'depth':>16}{'P(path)':>16}", flush=True)
    base = None
    for d in GRID[label]:
        L, D, P = [], [], []
        for sd in SEEDS:
            fb = {"PTH": {tok.eos_id: -d}} if d else None
            out = generate_with_suffixes(model, tok, label=label, n=N,
                                         table=table, decoder="typical",
                                         frame="domain", field_bias=fb, seed=sd)
            a, b_, c = m(out)
            L.append(a); D.append(b_); P.append(c)
        ml, sl = statistics.mean(L), statistics.pstdev(L)
        if base is None:
            base = (ml, sl)
        sig = (ml - base[0]) / (((sl**2 + base[1]**2) / len(SEEDS))**0.5 or 1e-9)
        mark = " <- DERIVED" if abs(d - DERIVED[label]) < 1e-9 else ""
        print(f"  {d:<8.2f}{ml:>10.2f}+-{sl:<6.2f}"
              f"{statistics.mean(D):>10.2f}+-{statistics.pstdev(D):<5.2f}"
              f"{statistics.mean(P):>10.1%}+-{statistics.pstdev(P):<5.1%}"
              f"   {sig:>+5.1f}s{mark}", flush=True)
