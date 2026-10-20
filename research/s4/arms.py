"""Calibrated generation harness across E5+delta and S4 w=1 / w=2 / w=4.

ONE DECODE CONFIG FOR EVERY MODEL. delta is the adopted path-scoped EOS bias
(27.2: benign 0.22, phishing 0.30) and is held FIXED across arms rather than
re-tuned per model. Re-tuning it would vary two things at once and the
comparison would no longer attribute anything to the training change. It also
means each S4 arm is measured against the baseline as actually shipped, not
against a stripped-down E5 that flatters it.

Everything else follows 26.1 unchanged: G=5 generation seeds x D=3 discriminator
seeds, n=1800 forced (SUB is unmeasurable below it), paired on generation index,
SE from the spread of per-generation means, and verdicts read against the
FAMILY-WISE null thresholds measured in 26.3 rather than an assumed 2 sigma.

Run per (model, class) so the arms parallelise; `report.py` joins them.
"""
from __future__ import annotations

import argparse
import gzip
import json
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from uie.evaluation import discriminate_with_floor
from uie.factorization import parse
from uie.suffixes import SuffixTable, generate_with_suffixes
from uie.train import load_checkpoint

B = ROOT / "data/bundle2"
CACHE = ROOT / "data/scratch/harness_cache"
OUT = ROOT / "data/s4_arms"
OUT.mkdir(parents=True, exist_ok=True)

MODELS = {"E5": "data/best_e5.pt", "w1": "data/s4_w1_best.pt",
          "w2": "data/s4_w2_best.pt", "w4": "data/s4_w4_best.pt"}
DELTA = {"benign": 0.22, "phishing": 0.30}          # adopted, 27.2
POOL = {"benign": "name_val", "phishing": "phishing_name_val"}
FIT = {"benign": "name_train", "phishing": "phishing_name_train"}
FIELDS = ("LBL", "SUF", "SUB", "PTH")
# family-wise null thresholds, 26.3. Measured on benign; phishing has ~half the
# path rate and no null of its own, so for that class these are BORROWED and
# ANTI-CONSERVATIVE. Stated at the point of use, not buried.
THRESH = {"LBL": 6.03, "SUF": 4.30, "SUB": 5.04, "PTH": 5.37}
SEEDS = [30000 + 17 * i for i in range(5)]


def rows(stem, limit=None):
    out = []
    with gzip.open(B / f"{stem}.jsonl.gz", "rt", encoding="utf-8") as fh:
        for line in fh:
            out.append(json.loads(line)["url"])
            if limit and len(out) >= limit:
                break
    return out


def structural(urls):
    parts = [parse(u) for u in urls]
    bodies = ["/".join(s for s in (p.path or "").split("/") if s) for p in parts]
    nz = [b for b in bodies if b]
    segs = [s for b in nz for s in b.split("/") if s]
    hosts = [p.host_raw or "" for p in parts]
    labels = [p.registrable_label for p in parts if p.registrable_label]
    mean = lambda xs: statistics.mean(xs) if xs else 0.0     # noqa: E731
    return {"seg_len": mean([len(s) for s in segs]),
            "len|path": mean([len(b) for b in nz]),
            "depth": mean([len(b.split("/")) for b in nz]),
            "P(path)": len(nz) / len(urls) if urls else 0.0,
            "host_len": mean([len(h) for h in hosts]),
            "lbl_len": mean([len(x) for x in labels])}


def generate_arm(ckpt, label, n, bias, tag):
    model, tok, _ = load_checkpoint(ckpt)
    model.eval()
    table = SuffixTable.load(str(B / "suffix_table.json"))
    fb = {"PTH": {tok.eos_id: -bias}} if bias else None
    out = []
    for gs in SEEDS:
        key = f"{Path(ckpt).stem}_{label}_n{n}_b{bias or 0}_s{gs}.json"
        p = CACHE / key
        if p.is_file():
            out.append(json.loads(p.read_text(encoding="utf-8")))
            continue
        urls = generate_with_suffixes(model, tok, label=label, n=n, table=table,
                                      decoder="typical", frame="domain",
                                      field_bias=fb, seed=gs)
        p.write_text(json.dumps(urls), encoding="utf-8")
        out.append(urls)
        print(f"    {tag}: generated seed {gs}", flush=True)
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=sorted(MODELS))
    ap.add_argument("--label", required=True, choices=("benign", "phishing"))
    ap.add_argument("-n", type=int, default=1800)
    ap.add_argument("-D", type=int, default=3)
    ap.add_argument("--delta", type=float, default=None,
                    help="override; default is the adopted value for the class")
    a = ap.parse_args()

    d = DELTA[a.label] if a.delta is None else a.delta
    tag = f"{a.model}/{a.label}/d{d}"
    print(f"{tag}  n={a.n} G={len(SEEDS)} D={a.D}", flush=True)

    arm = generate_arm(MODELS[a.model], a.label, a.n, d, tag)
    pool = rows(POOL[a.label])

    per_gen = {f: [] for f in FIELDS}
    for urls in arm:
        for f in FIELDS:
            xs = [r["excess"] for ds in range(1, a.D + 1)
                  for r in [discriminate_with_floor(pool, urls, f, n=a.n,
                                                    seed=ds)]
                  if not r.get("insufficient")]
            if xs:
                per_gen[f].append(statistics.mean(xs))
        print(f"    {tag}: scored a generation", flush=True)

    res = {"model": a.model, "label": a.label, "delta": d, "n": a.n,
           "G": len(SEEDS), "D": a.D,
           "excess": {f: per_gen[f] for f in FIELDS},
           "structural": [structural(u) for u in arm],
           "target": structural(rows(FIT[a.label], 40_000))}
    # DELTA BELONGS IN THE FILENAME. Without it a delta=0 run silently
    # overwrote the adopted-delta result for the same (model, class) -- the
    # output looked complete and was simply a different experiment.
    p = OUT / f"{a.model}_{a.label}_d{d}.json"
    p.write_text(json.dumps(res, indent=1), encoding="utf-8")
    print("wrote", p, flush=True)
