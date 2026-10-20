"""The targeted correction: delta DERIVED from the measured mis-calibration.

The earlier sweep tested delta in {0.3, 0.6, 0.9, 1.2} and was rejected for
overshooting depth (1.8 -> 2.1) and distorting digits. The boundary-hazard
measurement since shows the model over-predicts the path-exit hazard by only
1.06-1.15x, which implies

    delta = ln(1 / 1.15) ~ 0.15   benign
    delta = ln(1 / 1.06) ~ 0.06   phishing

so that sweep ran 2x to 8x too strong. Its rejection was unfounded: the correct
range was never tested. This brackets the derived value instead of guessing.

Applied ONLY inside a path body. P(path) is decided before <PTH> exists and host
length earlier still, so those columns cannot move if the scoping is sound --
verified over delta 0-1.2 in the previous sweep (`lbl_len` flat at 11.4,
`host_len` 17.1-17.3).

PRE-REGISTERED PREDICTIONS
  len|path  rises ~2-3 chars (benign 23.8 -> ~26; perfect calibration bounds the
            gain at 25.4 -> 28.6, and only ~half the deficit is calibration)
  depth     UNCHANGED within noise. Overshoot beyond +0.1 rejects, as before.
  P(path), lbl_len, host_len, digit%   unchanged
  any_up    unchanged (this intervention does not address uppercase)

n=600 with a shared seed per config, so configs differ only by delta.
"""
from __future__ import annotations

import gzip
import json
import sys

from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from uie.factorization import parse
from uie.suffixes import SuffixTable, generate_with_suffixes
from uie.train import load_checkpoint

B = str(ROOT / "data/bundle2")
N = 600
DERIVED = {"benign": 0.15, "phishing": 0.06}
GRID = {"benign": (0.0, 0.08, 0.15, 0.22, 0.30),
        "phishing": (0.0, 0.06, 0.12, 0.20, 0.30)}

model, tok, cfg = load_checkpoint(str(ROOT / "data/best_e5.pt"))
model.eval()
table = SuffixTable.load(B + r"/suffix_table.json")
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
        "depth": (sum(len([s for s in b.split("/") if s]) for b in nz)
                  / len(nz)) if nz else 0,
        "P(path)": len(nz) / len(bodies),
        "digit%": sum(sum(c.isdigit() for c in b) for b in nz) / chars,
        "any_up": (sum(1 for b in nz if any(c.isupper() for c in b)) / len(nz)
                   if nz else 0),
        "lbl_len": sum(len(parse(u).registrable_label or "") for u in urls) / len(urls),
        "host_len": sum(map(len, hosts)) / len(urls),
    }


KEYS = ["len|path", "depth", "P(path)", "digit%", "any_up", "lbl_len", "host_len"]


def show(tag, s):
    print(f"  {tag:<20}" + "".join(
        f"{s[k]:>10{'.1%' if k in ('P(path)','digit%','any_up') else '.2f'}}"
        for k in KEYS))


for label in ("benign", "phishing"):
    urls = []
    with gzip.open(f"{B}/{FIT[label]}.jsonl.gz", "rt", encoding="utf-8") as fh:
        for line in fh:
            urls.append(json.loads(line)["url"])
            if len(urls) >= 40000:
                break
    print(f"\n{'='*92}\n=== {label} / domain frame ===   derived delta = "
          f"{DERIVED[label]:.2f}")
    print(f"  {'config':<20}" + "".join(f"{k:>10}" for k in KEYS))
    show("REAL TARGET", stats(urls))
    for d in GRID[label]:
        fb = {"PTH": {tok.eos_id: -d}} if d else None
        out = generate_with_suffixes(model, tok, label=label, n=N, table=table,
                                     decoder="typical", frame="domain",
                                     field_bias=fb, seed=20260809)
        mark = "  <- DERIVED" if abs(d - DERIVED[label]) < 1e-9 else ""
        show(f"delta {d:.2f}{mark}", stats(out))
