"""Two questions, one generation pass.

Q1  Are the E4->E5 "regressions" real? Benign LBL moved +0.0320 -> +0.0392 and
    phishing SUB +0.0721 -> +0.1099. A single AUC at n=1200 carries SE ~ 0.0167,
    so a difference of two carries ~0.0236 and the deltas are 0.3s and 1.6s.
    A cause cannot be diagnosed before the effect is shown to exist, so this
    re-measures each excess under K seeds -- resampling the real pools, the fake
    subsample and the CV folds -- and reports mean +- sd across seeds.

Q2  Is the uppercase deficit EXPOSURE BIAS? Probe U established the conditionals
    are calibrated on REAL prefixes (ratio 1.00 / 1.07 / 0.98). The remaining
    explanation is that self-generated prefixes differ from real ones and carry
    less uppercase mass. That is directly measurable: run the SAME probe on the
    model's OWN generated paths.

      mass on own prefixes ~ mass on real prefixes  -> exposure bias REFUTED,
                                                       cause still unknown
      mass on own prefixes << real                  -> exposure bias CONFIRMED
"""
from __future__ import annotations

import json
import statistics
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import gzip

from uie.evaluation import discriminate_with_floor
from uie.suffixes import SuffixTable, generate_with_suffixes
from uie.tokenizer import _MARKERS
from uie.train import load_checkpoint

B = str(ROOT / "data/bundle2")
TMP = ROOT / "data/scratch"
N, SEEDS = 1200, 7

e4_cache = json.loads((TMP / "gen_cache.json").read_text(encoding="utf-8"))
table = SuffixTable.load(B + r"/suffix_table.json")


def rows(stem):
    with gzip.open(f"{B}/{stem}.jsonl.gz", "rt", encoding="utf-8") as fh:
        return [json.loads(line)["url"] for line in fh]


# --- generate E5 in the domain frame, once ---------------------------------
m5, t5, c5 = load_checkpoint(str(ROOT / "data/best_e5.pt"))
m5.eval()
e5 = {}
for label in ("benign", "phishing"):
    e5[label] = generate_with_suffixes(m5, t5, label=label, n=N, table=table,
                                       decoder="typical", frame="domain",
                                       seed=20260807)
    print(f"  generated e5/{label}", flush=True)
(TMP / "e5_domain_gen.json").write_text(json.dumps(e5), encoding="utf-8")

# --- Q1: multi-seed significance -------------------------------------------
print("\n" + "=" * 74)
print(f"Q1  excess over floor, {SEEDS} seeds (mean +- sd)")
print(f"  {'field':<22}{'E4':>18}{'E5':>18}{'delta':>10}{'sigma':>8}")
for label, field, stem in (("benign", "LBL", "name_val"),
                           ("phishing", "SUB", "phishing_name_val"),
                           ("benign", "PTH", "name_val"),
                           ("phishing", "PTH", "phishing_name_val")):
    pool = rows(stem)
    out = {}
    for tag, fake in (("E4", e4_cache[f"e4/{label}"]), ("E5", e5[label])):
        vals = []
        for s in range(SEEDS):
            r = discriminate_with_floor(pool, fake, field, n=N,
                                        seed=20260806 + 977 * s)
            if not r.get("insufficient"):
                vals.append(r["excess"])
        out[tag] = vals
    if not out["E4"] or not out["E5"]:
        print(f"  {label+'/'+field:<22} insufficient")
        continue
    m4, s4 = statistics.mean(out["E4"]), statistics.pstdev(out["E4"])
    m5_, s5 = statistics.mean(out["E5"]), statistics.pstdev(out["E5"])
    se = (s4 ** 2 / len(out["E4"]) + s5 ** 2 / len(out["E5"])) ** 0.5
    sig = (m5_ - m4) / se if se else float("nan")
    print(f"  {label+'/'+field:<22}{m4:>+11.4f}+-{s4:<5.4f}"
          f"{m5_:>+11.4f}+-{s5:<5.4f}{m5_-m4:>+10.4f}{sig:>8.1f}")

# --- Q2: uppercase mass on own vs real prefixes -----------------------------
PTH = t5.marker_id("PTH")
ALL_END = [t5.marker_id(m) for m in _MARKERS] + [t5.eos_id]
UPPER = [i for ch, i in t5.stoi.items() if len(ch) == 1 and ch.isupper()]


@torch.no_grad()
def upper_mass(urls, label):
    seqs = []
    for u in urls:
        ids = t5.encode(u, label, frame="domain")
        if PTH not in ids or len(ids) > c5.max_len:
            continue
        i = ids.index(PTH)
        j = i + 1
        while j < len(ids) and ids[j] not in ALL_END:
            j += 1
        if j > i + 1:
            seqs.append((ids, i + 1, j))
    tot = real = n = 0.0
    for s in range(0, len(seqs), 32):
        ch = seqs[s:s + 32]
        w = max(len(x[0]) for x in ch)
        x = torch.full((len(ch), w), t5.pad_id, dtype=torch.long)
        for r, (ids, _, _) in enumerate(ch):
            x[r, :len(ids)] = torch.tensor(ids)
        p = torch.softmax(m5(x).float(), dim=-1)
        for r, (ids, a, b) in enumerate(ch):
            for pos in range(a, b):
                tot += float(p[r, pos - 1, UPPER].sum())
                real += 1.0 if t5.itos[ids[pos]].isupper() else 0.0
                n += 1
    return tot / n, real / n, int(n)


print("\n" + "=" * 74)
print("Q2  uppercase: predicted mass on REAL prefixes vs the model's OWN")
print(f"  {'source':<28}{'predicted':>11}{'actual':>10}{'n':>10}")
for label, stem in (("benign", "name_val"), ("phishing", "phishing_name_val")):
    pr, ar, nr = upper_mass(rows(stem)[:1200], label)
    pg, ag, ng = upper_mass(e5[label], label)
    print(f"  {label+' REAL prefixes':<28}{pr:>11.2%}{ar:>10.2%}{nr:>10,}")
    print(f"  {label+' OWN prefixes':<28}{pg:>11.2%}{ag:>10.2%}{ng:>10,}")
    print(f"  {'':<28}{'ratio own/real ':>11}{pg/pr if pr else 0:>10.2f}")
