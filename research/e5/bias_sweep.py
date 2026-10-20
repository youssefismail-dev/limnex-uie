"""Stage 1: does a decode-time logit bias move the two targets, and what else moves?

HYPOTHESIS (not a conclusion): the path-length and uppercase deficits are
calibration errors at decode time, so a per-token-group log-odds shift should
correct them without touching the model.

The measured causes:
  EOS   mid-path leakage ~1.8% (benign) / 2.5% (phishing) per position, which
        compounds to 0.982**28 = 0.60 survival over a 28-character path.
  UPPER on the model's OWN prefixes the predicted uppercase mass is 0.40x
        (benign) / 0.72x (phishing) of the mass on real prefixes.

Swept ONE AT A TIME. Sweeping jointly would confound which knob moved what, and
the second knob's optimum depends on the first.

Every config reports the two TARGETS and a set of NON-TARGETS that must not move:
label/subdomain length, host composition, depth, P(path), digit rate, novelty
proxy. A bias that fixes length by making hosts weird is a failure, and only the
non-target columns can reveal that.

Stage 2 (separate) runs the full field evaluation on the winner.
"""
from __future__ import annotations

import collections
import gzip
import json
import sys

from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from uie.factorization import parse
from uie.suffixes import SuffixTable, generate_with_suffixes
from uie.tokenizer import _MARKERS
from uie.train import load_checkpoint

B = str(ROOT / "data/bundle2")
N = 500

model, tok, cfg = load_checkpoint(str(ROOT / "data/best_e5.pt"))
model.eval()
table = SuffixTable.load(B + r"/suffix_table.json")
UPPER = [i for ch, i in tok.stoi.items() if len(ch) == 1 and ch.isupper()]

# Fitting targets come from the TRAIN corpora; the eval split is never touched
# here, so a value chosen now is not chosen against the numbers it is judged by.
FIT = {"benign": "name_train", "phishing": "phishing_name_train"}


def pb(u):
    return "/".join(s for s in (parse(u).path or "").split("/") if s)


def stats(urls):
    bodies = [pb(u) for u in urls]
    nz = [b for b in bodies if b]
    chars = sum(map(len, nz)) or 1
    hosts = [".".join(tuple(parse(u).subdomains) +
                      (parse(u).registrable_label, parse(u).public_suffix))
             for u in urls]
    return {
        "len|path": sum(map(len, nz)) / len(nz) if nz else 0,
        "upper%": sum(sum(c.isupper() for c in b) for b in nz) / chars,
        "P(path)": len(nz) / len(bodies),
        "depth": (sum(len([s for s in b.split("/") if s]) for b in nz)
                  / len(nz)) if nz else 0,
        "digit%": sum(sum(c.isdigit() for c in b) for b in nz) / chars,
        # Robust to burstiness: one 40-character uppercase token swings a
        # per-character mean, which made the n=300 sweep unusable for uppercase.
        "any_up": (sum(1 for b in nz if any(c.isupper() for c in b)) / len(nz)
                   if nz else 0),
        "lbl_len": sum(len(parse(u).registrable_label or "") for u in urls) / len(urls),
        "host_len": sum(map(len, hosts)) / len(urls),
        "distinct": len(set(urls)) / len(urls),
    }


def target(label):
    urls = []
    with gzip.open(f"{B}/{FIT[label]}.jsonl.gz", "rt", encoding="utf-8") as fh:
        for line in fh:
            urls.append(json.loads(line)["url"])
            if len(urls) >= 40000:
                break
    return stats(urls)


KEYS = ["len|path", "upper%", "any_up", "P(path)", "depth", "digit%",
        "lbl_len", "host_len"]


def row(tag, s, t):
    cells = []
    for k in KEYS:
        v = s[k]
        f = ".1%" if k.endswith("%") or k in ("P(path)", "distinct") else ".1f"
        cells.append(f"{v:>9{f}}")
    print(f"  {tag:<16}" + "".join(cells))


for label in ("benign", "phishing"):
    t = target(label)
    print(f"\n{'='*96}\n=== {label} / domain frame ===")
    print(f"  {'config':<16}" + "".join(f"{k:>9}" for k in KEYS))
    row("REAL TARGET", t, t)

    # --- PATH-SCOPED eos bias ------------------------------------------------
    # Applied only inside a path body. P(path) is decided BEFORE <PTH> is
    # emitted and host length earlier still, so if the scoping works those
    # columns must not move -- that is the falsifiable prediction.
    for d in (0.0, 0.3, 0.6, 0.9, 1.2):
        fb = {"PTH": {tok.eos_id: -d}} if d else None
        u = generate_with_suffixes(model, tok, label=label, n=N, table=table,
                                   decoder="typical", frame="domain",
                                   field_bias=fb, seed=20260808)
        row(f"PTH eos -{d:.1f}", stats(u), t)
