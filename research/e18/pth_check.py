"""The benign P1Q1 PTH read-only check (owner's decision, 2026-10-08).

NOTHING IS REGISTERED HERE. This measures only. It asks one question:

    do the paths this model writes under a P1Q1 plan do anything that REAL benign
    P1Q1 paths never do?

A "yes" with a clear, enforceable structure would license a B5-style decode rule
(forbid exactly that, by construction, as D1 and B5 forbid a broken plan). A "no"
closes the book on benign P1Q1 PTH: there is then no honest inference-time fix,
because the remaining tell is a matter of degree, and a knob set to close it would be
tuned to the metric (E13's D2 tried exactly that on exactly this cell and moved it
-0.0018; it is closed).

    python research/e18/pth_check.py --part gen    # fresh e16_b5 generations (CPU, slow)
    python research/e18/pth_check.py --part read   # the battery; writes pth_check.json

Honesty rules, as the owner approved:
- **Fresh generations, never the gate's 50 files.** `control_urls` at g >= GEN_G0, far
  outside the gate's g 0-4, so no seed overlaps `data/e17/decode/gen`.
- **"What real paths never do" is read from the TRAINING pool**, which the model
  already saw, so a rule justified by it is not fitted to the gate. The validation
  pool is read too, and only confirms.
- Every feature below is a well-formedness or degeneracy property of the string. None
  is a content choice, and none is chosen by looking at the gate's score.
"""
from __future__ import annotations

import argparse
import collections
import json
import re
import sys
import time
import urllib.parse
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
for p in (ROOT, HERE, ROOT / "research" / "e17", ROOT / "research" / "e16", ROOT / "research" / "e15",
          ROOT / "research" / "e13", ROOT / "research" / "native", ROOT / "research" / "cr1"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import e12_common as K  # noqa: E402
import e17_common as E17  # noqa: E402

OUT = ROOT / "data" / "e18"
FRESH = OUT / "fresh_benign_P1Q1.json"
E16_CAP = ROOT / "data" / "e16" / "train" / "e16_cap" / "final.pt"
#: fresh seeds: the gate uses g 0-4, so these share no seed with data/e17/decode/gen
GEN_G0, GEN_BLOCKS, GEN_N = 100, 2, 800
STATE = "P1Q1"
LABEL = "benign"
_PCT = re.compile(r"%(?![0-9A-Fa-f]{2})")


# --------------------------------------------------------------------------- #
# part gen -- fresh generations from the adopted baseline
# --------------------------------------------------------------------------- #

def part_gen() -> None:
    import hashlib
    import torch
    from uie.controlled import StateSuffixTables
    from uie.suffixes import SuffixTable
    from uie.train import load_checkpoint
    if hashlib.sha256(E16_CAP.read_bytes()).hexdigest() != E17.E16_CAP_SHA256:
        raise SystemExit("STOP: the laptop's e16_cap final.pt is not the adopted baseline")
    model, tok, _ = load_checkpoint(E16_CAP, device=torch.device("cpu"))
    model.eval()
    shipped = SuffixTable.load(ROOT / "data" / "bundle2" / "suffix_table.json")
    tables = StateSuffixTables.load(ROOT / "data" / "state_suffix_tables.json")
    ps = json.loads((ROOT / "data" / "e12" / "presence_slash_table.json").read_text(
        encoding="utf-8"))["table"]
    urls = []
    for b in range(GEN_BLOCKS):
        g = GEN_G0 + b
        t0 = time.time()
        got = E17.control_urls(model, tok, LABEL, STATE, g, tables, ps, b5=True,
                               device=torch.device("cpu"), n=GEN_N)
        urls.extend(got)
        print(f"  block g={g}: {len(got)} urls ({time.time() - t0:.0f}s)", flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    FRESH.write_text(json.dumps({"arm": "e16_b5", "label": LABEL, "state": STATE,
                                 "decode": E17.decode_kwargs(True),
                                 "g": list(range(GEN_G0, GEN_G0 + GEN_BLOCKS)),
                                 "n_per_block": GEN_N, "urls": urls}, indent=1), encoding="utf-8")
    print(f"wrote {FRESH}: {len(urls)} fresh urls")


# --------------------------------------------------------------------------- #
# part read -- the structural battery
# --------------------------------------------------------------------------- #

def real_paths() -> dict:
    """Real benign P1Q1 paths, from the training pool (the reference) and the
    validation pool (confirmation only)."""
    from cr1_score import in_state
    from n2a_reference import rows
    from uie.evaluation import FIELDS
    want = K.STATES[STATE]
    out = {}
    for split, stems in (("train", ("name_train", "structure_train")),
                         ("val", ("name_val", "structure_val"))):
        paths = []
        for stem in stems:
            for u in rows(stem):
                if in_state(u, want):
                    p = FIELDS["PTH"](u)
                    if p:
                        paths.append(p)
        out[split] = paths
        print(f"  real {split}: {len(paths):,} P1Q1 paths", flush=True)
    return out


def segments(p: str) -> list:
    return [s for s in p.split("/")]


def repeated_run(p: str, unit: int = 4, times: int = 3) -> bool:
    """A substring of at least `unit` characters repeated `times` times in a row."""
    n = len(p)
    if n > 600:
        p, n = p[:600], 600
    for k in range(unit, n // times + 1):
        for i in range(n - k * times + 1):
            block = p[i:i + k]
            if all(p[i + j * k:i + (j + 1) * k] == block for j in range(1, times)):
                return True
    return False


def features(p: str, charset: frozenset, caps: dict) -> dict:
    """Well-formedness and degeneracy properties of one path string. Each is a
    property of the string alone; none looks at any score."""
    segs = segments(p)
    inner = segs[1:-1] if len(segs) > 2 else []
    try:
        decoded_ok = True
        if "%" in p:
            urllib.parse.unquote(p, errors="strict")
    except Exception:
        decoded_ok = False
    return {
        "double_slash": "//" in p,
        "empty_inner_segment": any(s == "" for s in inner),
        "dot_segment": any(s in (".", "..") for s in segs),
        "malformed_percent": bool(_PCT.search(p)),
        "percent_not_utf8": not decoded_ok,
        "whitespace": any(c.isspace() for c in p),
        "control_char": any(ord(c) < 32 or ord(c) == 127 for c in p),
        "non_ascii": any(ord(c) > 127 for c in p),
        "char_unseen_in_real": any(c not in charset for c in p),
        "backslash": "\\" in p,
        "at_sign": "@" in p,
        "repeated_block": repeated_run(p),
        "same_char_run_6": any(p[i:i + 6] == p[i] * 6 for i in range(max(0, len(p) - 5))),
        "longer_than_any_real": len(p) > caps["max_len"],
        "segment_longer_than_any_real": any(len(s) > caps["max_seg"] for s in segs),
        "deeper_than_any_real": len([s for s in segs if s]) > caps["max_depth"],
    }


def ngram_scan(gen: list, real: dict, k: int = 4) -> dict:
    """Data-driven, so it does not depend on which features the battery happens to
    name: the character k-grams present in a path but in NONE of the real training
    paths. Real VALIDATION paths are the control -- held-out real text also contains
    k-grams the training pool never had, so the question is only whether the model
    does it MORE than real held-out data does."""
    def grams(p):
        return {p[i:i + k] for i in range(len(p) - k + 1)}
    seen = set()
    for p in real["train"]:
        seen |= grams(p)
    out = {"k": k, "train_kgrams": len(seen)}
    counts = collections.Counter()
    for who, paths in (("generated", gen), ("val", real["val"])):
        hit = 0
        for p in paths:
            novel = grams(p) - seen
            if novel:
                hit += 1
                if who == "generated":
                    counts.update(novel)
        out[who] = {"n": len(paths), "with_unseen_kgram": hit, "rate": hit / len(paths)}
    out["top_unseen_in_generated"] = counts.most_common(15)
    return out


def part_read() -> None:
    from uie.evaluation import FIELDS
    blob = json.loads(FRESH.read_text(encoding="utf-8"))
    gen = [p for p in (FIELDS["PTH"](u) for u in blob["urls"]) if p]
    print(f"  generated: {len(gen):,} P1Q1 paths (of {len(blob['urls']):,} fresh urls)", flush=True)
    real = real_paths()
    charset = frozenset(c for p in real["train"] for c in p)
    caps = {"max_len": max(len(p) for p in real["train"]),
            "max_seg": max((len(s) for p in real["train"] for s in segments(p)), default=0),
            "max_depth": max(len([s for s in segments(p) if s]) for p in real["train"])}
    print(f"  real train caps: {caps}; charset {len(charset)} chars", flush=True)
    names = list(features("/a", charset, caps))
    counts = {k: {"generated": 0, "train": 0, "val": 0} for k in names}
    any_viol = {"generated": 0, "train": 0, "val": 0}
    examples = collections.defaultdict(list)
    for who, paths in (("generated", gen), ("train", real["train"]), ("val", real["val"])):
        for p in paths:
            f = features(p, charset, caps)
            hit = False
            for k, v in f.items():
                if v:
                    counts[k][who] += 1
                    hit = True
                    if who == "generated" and len(examples[k]) < 5:
                        examples[k].append(p[:160])
            any_viol[who] += hit
    n = {"generated": len(gen), "train": len(real["train"]), "val": len(real["val"])}
    rows = {}
    for k in names:
        rows[k] = {w: counts[k][w] / n[w] for w in n}
        rows[k]["counts"] = dict(counts[k])
    rep = {
        "what": "benign P1Q1 PTH read-only check (owner 2026-10-08); nothing registered",
        "arm": blob["arm"], "decode": blob["decode"], "fresh_g": blob["g"],
        "n": n, "real_caps": caps,
        "features": rows,
        "any_violation": {w: any_viol[w] / n[w] for w in n},
        "examples_generated": {k: v for k, v in examples.items() if v},
        "shape": {
            who: {"mean_len": sum(len(p) for p in ps) / len(ps),
                  "mean_depth": sum(len([s for s in segments(p) if s]) for p in ps) / len(ps)}
            for who, ps in (("generated", gen), ("train", real["train"]), ("val", real["val"]))},
        "ngram_scan": ngram_scan(gen, real),
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "pth_check.json").write_text(json.dumps(rep, indent=1, default=float), encoding="utf-8")
    print(f"\n{'feature':<32} {'generated':>10} {'real train':>11} {'real val':>10}")
    for k in names:
        r = rows[k]
        print(f"  {k:<30} {r['generated']:>10.4%} {r['train']:>11.4%} {r['val']:>10.4%}")
    print(f"  {'ANY of the above':<30} {rep['any_violation']['generated']:>10.4%} "
          f"{rep['any_violation']['train']:>11.4%} {rep['any_violation']['val']:>10.4%}")
    print(f"\nshape: {rep['shape']}")
    ng = rep["ngram_scan"]
    print(f"\n{ng['k']}-grams unseen in {len(real['train']):,} real train paths "
          f"({ng['train_kgrams']:,} distinct seen):")
    print(f"  generated      {ng['generated']['with_unseen_kgram']:>5} / {ng['generated']['n']:<6} "
          f"= {ng['generated']['rate']:.4%}")
    print(f"  real val (ctrl){ng['val']['with_unseen_kgram']:>5} / {ng['val']['n']:<6} "
          f"= {ng['val']['rate']:.4%}")
    print("wrote", OUT / "pth_check.json")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--part", required=True, choices=("gen", "read"))
    args = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")
    {"gen": part_gen, "read": part_read}[args.part]()


if __name__ == "__main__":
    main()
