"""E5 pre-registered validation (E3_REPORT.md 15.3).

Order matters. Prediction 2 is the FALSIFICATION TEST and is reported first: if
uppercase does not move, the content-starvation account of issue 2 is wrong no
matter how the other numbers look.

  P2  uppercase rate per path character moves from 0.47 / 0.51 toward 1.0
  P1  PRIMARY GATE: len|path under frame=page matches structure (39.7 / 29.1),
      under frame=domain matches name (28.3 / 15.9)
  P3  phishing len|path in-frame closes from 11.6 toward 15.9
  P5  CONTAMINATION: frame=page depth must not collapse toward the domain
      frame's 1.29 -- that is the failure the masking existed to prevent

Generated in BOTH frames, since the whole claim is that they now differ.
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
OUT = ROOT / "data/scratch/e5_gen.json"
N = 800
LIMIT = 40000

model, tok, cfg = load_checkpoint(str(ROOT / "data/best_e5.pt"))
model.eval()
table = SuffixTable.load(B + r"/suffix_table.json")
assert tok.include_frame, "checkpoint is not frame-conditioned"

CORPUS = {("benign", "domain"): "name_train",
          ("benign", "page"): "structure_train",
          ("phishing", "domain"): "phishing_name_train",
          ("phishing", "page"): "phishing_structure_train"}


def pb(u):
    return "/".join(s for s in (parse(u).path or "").split("/") if s)


def rows(stem):
    out = []
    with gzip.open(f"{B}/{stem}.jsonl.gz", "rt", encoding="utf-8") as fh:
        for line in fh:
            out.append(json.loads(line)["url"])
            if len(out) >= LIMIT:
                break
    return out


def stats(urls):
    bodies = [pb(u) for u in urls]
    nz = [b for b in bodies if b]
    chars = sum(map(len, nz))
    return {
        "p_path": len(nz) / len(bodies),
        "len_given_path": (sum(map(len, nz)) / len(nz)) if nz else 0.0,
        "depth_given_path": (sum(len([s for s in b.split("/") if s])
                                 for b in nz) / len(nz)) if nz else 0.0,
        "upper_rate": (sum(sum(c.isupper() for c in b) for b in nz) / chars
                       if chars else 0.0),
        "digit_rate": (sum(sum(c.isdigit() for c in b) for b in nz) / chars
                       if chars else 0.0),
    }


gen, real = {}, {}
for (label, frame), stem in CORPUS.items():
    urls = generate_with_suffixes(model, tok, label=label, n=N, table=table,
                                  decoder="typical", frame=frame, seed=20260807)
    gen[(label, frame)] = stats(urls)
    real[(label, frame)] = stats(rows(stem))
    gen[(label, frame)]["_sample"] = urls[:4]
    print(f"  generated {label}/{frame}", flush=True)

OUT.write_text(json.dumps({f"{k[0]}/{k[1]}": v for k, v in gen.items()},
                          indent=1), encoding="utf-8")

E4 = {"benign": 0.47, "phishing": 0.51}

print("\n" + "=" * 76)
print("P2  FALSIFICATION TEST -- uppercase rate per path character")
print(f"  {'class/frame':<20}{'real':>9}{'E5':>9}{'ratio':>8}{'E4 ratio':>10}")
for label in ("benign", "phishing"):
    for frame in ("domain", "page"):
        r, g = real[(label, frame)], gen[(label, frame)]
        ratio = g["upper_rate"] / r["upper_rate"] if r["upper_rate"] else 0
        e4 = f"{E4[label]:.2f}" if frame == "domain" else "-"
        print(f"  {label+'/'+frame:<20}{r['upper_rate']:>9.2%}"
              f"{g['upper_rate']:>9.2%}{ratio:>8.2f}{e4:>10}")

print("\n" + "=" * 76)
print("P1  PRIMARY GATE -- len|path must match the frame it was generated in")
print(f"  {'class/frame':<20}{'real':>9}{'E5':>9}{'gap':>9}")
for label in ("benign", "phishing"):
    for frame in ("domain", "page"):
        r, g = real[(label, frame)], gen[(label, frame)]
        print(f"  {label+'/'+frame:<20}{r['len_given_path']:>9.1f}"
              f"{g['len_given_path']:>9.1f}"
              f"{g['len_given_path']-r['len_given_path']:>+9.1f}")

print("\n" + "=" * 76)
print("P5  CONTAMINATION -- the frames must SEPARATE, not collapse together")
print(f"  {'class':<12}{'metric':<18}{'real dom':>10}{'E5 dom':>9}"
      f"{'real page':>11}{'E5 page':>9}")
for label in ("benign", "phishing"):
    for key, name in (("depth_given_path", "depth|path"),
                      ("p_path", "P(path)"),
                      ("digit_rate", "digit rate")):
        rd, gd = real[(label, "domain")][key], gen[(label, "domain")][key]
        rp, gp = real[(label, "page")][key], gen[(label, "page")][key]
        fmt = ".1%" if key in ("p_path", "digit_rate") else ".2f"
        print(f"  {label:<12}{name:<18}{rd:>10{fmt}}{gd:>9{fmt}}"
              f"{rp:>11{fmt}}{gp:>9{fmt}}")

print("\nsamples")
for k, v in gen.items():
    print(f"  {k[0]}/{k[1]}: " + " | ".join(v["_sample"]))
