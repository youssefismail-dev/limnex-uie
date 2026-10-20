"""E15-B -- post-hoc, DESCRIPTIVE: how much of each query tell is the length mix?
Not registered; licenses nothing. QC1's harness (qc1_common.harness: char 2-5-gram
TF-IDF + logistic regression, x = AUC - real floor), run twice per cell and arm at the
same n and seeds: against the real validation pool as it is, and against the real
pool re-sampled (without replacement) to the GENERATED length-bin mix. The difference
is the part of the tell carried by the length mix.

    python research/e15/e15_posthoc_length.py > data/e15/posthoc_length.md
"""
from __future__ import annotations

import random
import statistics as st
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
for p in (ROOT, HERE, ROOT / "research" / "e13", ROOT / "research" / "native", ROOT / "research" / "qc1",
          ROOT / "research" / "cr1", ROOT / "research" / "dc1"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import e13d_common as D  # noqa: E402
import qc1_common as C  # noqa: E402

GEN = {"e12_d1p": ROOT / "data" / "e14" / "decode" / "gen", "e15_new": ROOT / "data" / "e15" / "generate" / "gen"}
BINS = (16, 32, 64, 128)
MATCH_SEED = 20261012


def bin_of(s: str) -> int:
    return sum(len(s) > b for b in BINS)


def matched(real, gen, rng):
    """The real pool re-sampled, without replacement, to the generated length-bin mix,
    as large as the scarcest bin allows."""
    rb = {}
    for s in real:
        rb.setdefault(bin_of(s), []).append(s)
    gp = {}
    for s in gen:
        gp[bin_of(s)] = gp.get(bin_of(s), 0) + 1
    share = {b: c / len(gen) for b, c in gp.items()}
    total = int(min(len(rb.get(b, [])) / sh for b, sh in share.items() if sh > 0))
    out = []
    for b, sh in share.items():
        out += rng.sample(rb[b], int(round(total * sh)))
    rng.shuffle(out)
    return out


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    print("**Post-hoc and descriptive: not a registered reading; licenses nothing.**\n")
    print("| cell | arm | n | seeds | x, real as is | x, real matched to the generated length mix | carried by length |")
    print("|---|---|---|---|---|---|---|")
    for name in ("free_phishing_QRY", "control_benign_P0Q1_QRY", "control_benign_P1Q1_QRY", "free_benign_QRY"):
        cell = D.BY_NAME[name]
        real = D.pool(cell, "val")
        for arm in GEN:
            gen = D.generated(cell, arm, GEN[arm])
            rm = matched(real, gen, random.Random(MATCH_SEED))
            n = min(C.N_REQ, len(rm) // 3, len(gen))
            xs, xm = [], []
            for s in C.SEEDS:
                f = random.Random(s + 101).sample(gen, n)
                xs.append(C.harness(real, {"g": f}, s, n)["g"])
                xm.append(C.harness(rm, {"g": f}, s, n)["g"])
            a, b = st.mean(xs), st.mean(xm)
            print(f"| {name} | {arm} | {n:,} | {len(C.SEEDS)} | {a:+.4f} | {b:+.4f} | {a - b:+.4f} ({(a - b) / a:.0%}) |",
                  flush=True)


if __name__ == "__main__":
    main()
