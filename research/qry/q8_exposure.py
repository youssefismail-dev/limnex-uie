"""Q8 -- effective exposure (H3) and field-mask visibility (H6).

READ-ONLY. Counts corpus statistics and runs the FROZEN `field_mask` to see what
the objective actually trains. No training, no generation, no weight touched,
nothing under uie/ modified.

Pre-registered in PREREG_Q7.md.

H6 exists because `CORPUS_SCOPE` says query CONTENT is trained only from the
structure corpus:

    NAME_FIELDS      = ("SUF", "LBL", "SUB", "USR")
    STRUCTURE_FIELDS = ("PTH", "QRY", "PRT", "FRG")

If that holds in practice, the name corpus -- 906,214 rows, and the population
QRY is SCORED against in the domain frame -- contributes zero gradient to query
content. That is a deliberate design choice (DESIGN section 9), not a bug, but
its consequence for the query tail has never been quantified.

H3 asks whether what remains is enough. The checkpoint trained for
20,000 x 256 = 5.12M exposures with class_alpha 1.0 and within-class sampling
WITHOUT replacement, so every row is seen roughly the same small number of
times. The question is how many of those exposures land on tail structure.

NOTHING HERE IS A PROPOSAL. S4/S4b stays closed; measuring the mask is not
proposing to change it.
"""
from __future__ import annotations

import argparse
import collections
import gzip
import json
import random
import statistics as st
import sys
from pathlib import Path

import torch
from typing import Dict, List

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from research.qry.qtransforms import split_params          # noqa: E402
from uie.evaluation import FIELDS                          # noqa: E402
from uie.train import (CORPUS_FRAME, CORPUS_SCOPE,         # noqa: E402
                       field_mask, load_checkpoint)

CAP = 200_000
MASK_ROWS = 60_000
BOOT = 500
B = Path("data/bundle2")
#: Read from the checkpoint's own train_config, not from defaults.
STEPS, BATCH_SIZE = 20_000, 256
BANDS = [(1, 1), (2, 4), (5, 19), (20, 99), (100, 999), (1000, 10 ** 9)]


def band_label(b) -> str:
    return f"{b[0]}-{'inf' if b[1] > 10 ** 8 else b[1]}"


def band_of(c: int):
    for lo, hi in BANDS:
        if lo <= c <= hi:
            return (lo, hi)
    return BANDS[-1]


def rows_of(stem: str, cap: int = CAP) -> List[str]:
    with gzip.open(B / f"{stem}.jsonl.gz", "rt", encoding="utf-8") as fh:
        return [json.loads(line)["url"] for i, line in enumerate(fh) if i < cap]


def h6_mask_visibility(tok, max_len: int) -> Dict[str, object]:
    """What share of trained target positions does each field actually get?"""
    markers = ("SUF", "LBL", "SUB", "PRT", "PTH", "QRY", "FRG", "USR")
    marker_of = {tok.marker_id(m): m for m in markers}
    out: Dict[str, object] = {}
    for stem, corpus in (("name_train", "name"),
                         ("structure_train", "structure")):
        frame = CORPUS_FRAME.get(corpus, "domain")
        chars: collections.Counter = collections.Counter()
        trained: collections.Counter = collections.Counter()
        for url in rows_of(stem, MASK_ROWS):
            ids = tok.encode(url, "benign", "und", frame=frame)
            if len(ids) > max_len:
                continue
            weights = field_mask(ids, tok, corpus)
            cur = None
            for pos, tid in enumerate(ids[1:], start=1):
                if tid in marker_of:
                    cur = marker_of[tid]
                    continue
                if tid in (tok.eos_id, tok.bos_id, tok.pad_id) or cur is None:
                    continue
                chars[cur] += 1
                if weights[pos - 1] > 0:
                    trained[cur] += 1
        tot_c = sum(chars.values()) or 1
        tot_t = sum(trained.values()) or 1
        out[stem] = {
            "scope": list(CORPUS_SCOPE[corpus]),
            "rows": MASK_ROWS,
            "fields": {
                f: {"chars": chars[f],
                    "char_share_pct": round(100 * chars[f] / tot_c, 3),
                    "trained": trained[f],
                    "trained_share_pct": round(100 * trained[f] / tot_t, 3),
                    "trained_frac_of_own_chars":
                        round(trained[f] / chars[f], 4) if chars[f] else None}
                for f in sorted(chars)}}
    return out


def h3_spectrum(tag: str = "") -> Dict[str, object]:
    """Frequency spectrum of parameter names, and what the model reproduces.

    `tag` selects WHICH generation to score. The samples are produced by
    q4_selection.py, not here, so scoring a new checkpoint requires generating
    its samples first -- the band profile is a property of the generation, and
    reading the baseline's files while claiming a new checkpoint's result would
    be the worst kind of silent error.
    """
    # Query CONTENT is taught by the structure corpus only (H6 verifies this),
    # so the learnability question is asked against THAT distribution.
    struct = rows_of("structure_train")
    names: List[str] = []
    for u in struct:
        q = FIELDS["QRY"](u)
        if q:
            names.extend(k for k, _, _ in split_params(q))
    freq = collections.Counter(names)
    total = sum(freq.values())

    gen: Dict[str, List[str]] = {}
    for arm in ("typical", "temperature"):
        src = Path(f"data/qry/q4_benign_{arm}_queries{tag}.json")
        if not src.is_file():
            raise SystemExit(
                f"missing {src}\n"
                f"  generate it first:\n"
                f"    python research/qry/q4_selection.py --class benign"
                f" --checkpoint data/<ckpt>.pt")
        urls = json.loads(src.read_text())
        g: List[str] = []
        for u in urls:
            q = FIELDS["QRY"](u)
            if q:
                g.extend(k for k, _, _ in split_params(q))
        gen[arm] = g

    corpus_mass: collections.Counter = collections.Counter()
    for nm, c in freq.items():
        corpus_mass[band_of(c)] += c

    hapax_types = sum(1 for c in freq.values() if c == 1)
    result: Dict[str, object] = {
        "structure_rows_read": len(struct),
        "distinct_names": len(freq),
        "total_params": total,
        "hapax_names": hapax_types,
        "hapax_name_pct": round(100 * hapax_types / max(len(freq), 1), 2),
        "hapax_token_pct": round(
            100 * sum(c for c in freq.values() if c == 1) / max(total, 1), 3),
        "bands": {}, "arms": {},
    }
    for b in BANDS:
        result["bands"][band_label(b)] = {
            "distinct_names": sum(1 for c in freq.values() if band_of(c) == b),
            "corpus_token_pct": round(100 * corpus_mass[b] / max(total, 1), 3)}

    # The NULL matters more than the observation. At 758 parameters a rare
    # name's absence is unreadable on its own -- it could simply be sampling.
    # This draws the same number of parameters from the training frequency
    # distribution, which is exactly what a faithful sampler of this size
    # would produce.
    rng = random.Random(20260806)
    pool = list(freq.elements())
    for arm, g in gen.items():
        obs = collections.Counter(
            band_of(freq[n]) if n in freq else "OOV" for n in g)
        n_draw = len(g)
        null: List[Dict] = []
        for _ in range(BOOT):
            draw: collections.Counter = collections.Counter(
                band_of(freq[pool[rng.randrange(len(pool))]])
                for _ in range(n_draw))
            null.append({b: 100 * draw[b] / n_draw for b in BANDS})
        arm_out: Dict[str, object] = {
            "n_params": n_draw,
            "oov_pct": round(100 * obs.get("OOV", 0) / max(n_draw, 1), 2),
            "bands": {}}
        for b in BANDS:
            vals = sorted(x[b] for x in null)
            arm_out["bands"][band_label(b)] = {
                "observed_pct": round(100 * obs.get(b, 0) / max(n_draw, 1), 3),
                "null_pct": round(st.mean(vals), 3),
                "null_ci": [round(vals[int(.025 * len(vals))], 3),
                            round(vals[int(.975 * len(vals))], 3)]}
        result["arms"][arm] = arm_out
    return result


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default="data/best_e5.pt")
    args = ap.parse_args()
    stem = Path(args.checkpoint).stem
    tag = "" if stem == "best_e5" else f"_{stem}"

    model, tok, _ = load_checkpoint(Path(args.checkpoint), device="cpu")
    max_len = model.cfg.max_len
    # Read the exposure count from the checkpoint rather than a constant. A
    # hard-coded 20,000 would silently misreport any run that trained longer,
    # which is exactly the comparison this file exists to support.
    tc = torch.load(Path(args.checkpoint), map_location="cpu",
                    weights_only=False)["train_config"]
    steps = int(tc.get("steps", STEPS))
    batch = int(tc.get("batch_size", BATCH_SIZE))
    print(f"checkpoint {stem}   steps {steps:,} x batch {batch}\n")

    print("=== H6: field-mask visibility (frozen field_mask) ===", flush=True)
    h6 = h6_mask_visibility(tok, max_len)
    for corpus_stem, blk in h6.items():
        print(f"\n  {corpus_stem}   scope = {blk['scope']}")
        print(f"  {'field':<8}{'chars':>11}{'char %':>9}{'trained':>11}"
              f"{'trained %':>11}{'trained/own':>13}")
        for f, v in blk["fields"].items():
            print(f"  {f:<8}{v['chars']:>11,}{v['char_share_pct']:>9.2f}"
                  f"{v['trained']:>11,}{v['trained_share_pct']:>11.2f}"
                  f"{str(v['trained_frac_of_own_chars']):>13}")

    print("\n=== H3: frequency spectrum and reproduction ===", flush=True)
    h3 = h3_spectrum(tag)
    print(f"  structure rows {h3['structure_rows_read']:,}   "
          f"distinct names {h3['distinct_names']:,}   "
          f"parameters {h3['total_params']:,}")
    print(f"  hapax {h3['hapax_names']:,} names "
          f"({h3['hapax_name_pct']}% of types, "
          f"{h3['hapax_token_pct']}% of tokens)")
    print(f"  checkpoint exposures {steps * batch:,} "
          f"({steps:,} steps x {batch})\n")
    print(f"  {'train count':<13}{'distinct':>10}{'corpus%':>10}"
          f"{'typ obs%':>10}{'typ null% [95% CI]':>24}{'temp obs%':>11}")
    for lab, b in h3["bands"].items():
        t = h3["arms"]["typical"]["bands"][lab]
        a = h3["arms"]["temperature"]["bands"][lab]
        null = f"{t['null_pct']:.2f} [{t['null_ci'][0]:.2f},{t['null_ci'][1]:.2f}]"
        print(f"  {lab:<13}{b['distinct_names']:>10,}"
              f"{b['corpus_token_pct']:>10.2f}{t['observed_pct']:>10.2f}"
              f"{null:>24}{a['observed_pct']:>11.2f}")
    for arm in ("typical", "temperature"):
        print(f"  {arm} OOV (name never seen in structure training): "
              f"{h3['arms'][arm]['oov_pct']}%")

    out = Path(f"data/qry/q8_exposure{tag}.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"h6": h6, "h3": h3,
                               "checkpoint": stem,
                               "exposures": steps * batch}, indent=1)
                   + "\n", encoding="utf-8")
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
