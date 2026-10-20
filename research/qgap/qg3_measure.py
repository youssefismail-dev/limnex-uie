"""QG3 -- the per-token ledger for one model (research/qgap/PREREG_QG3.md section 2):
over QG1's 40,000 domain rows per class in free mode's encoding, for every region
and token, the summed probability and the true-next count, in 400 blocks of 100
rows. Diagnostic; laptop; the baseline is only read.

    python research/qgap/qg3_measure.py --model e9_ln          # then e9_base, e7

Writes data/qgap/qg3_<model>_<label>.npz and .json (vocabulary, rows, VR2).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE))

import numpy as np  # noqa: E402

import qg_common as Q  # noqa: E402
from qg1_measure import CKPT, sample_rows, vq1_model  # noqa: E402

BLOCK = 100


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=sorted(CKPT), required=True)
    ap.add_argument("--limit", type=int, help="dry runs only: rows per class")
    args = ap.parse_args()
    try:                                    # the laptop stays usable (UIE practice)
        import psutil
        psutil.Process().nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
    except Exception:
        pass
    model, tok, cfg = vq1_model(args.model)
    print(f"VQ1: {args.model} verified", flush=True)
    Q.DATA.mkdir(parents=True, exist_ok=True)
    for label in Q.LABELS:
        t0 = time.time()
        urls, seqs, pool, dropped = sample_rows(Q.STREAMS[label][0], Q.N_TRAIN, tok, label,
                                                cfg.max_len, args.limit)
        P, N, POS, read = Q.ledger(model, tok, seqs, max_len=cfg.max_len, block=BLOCK)
        dest = Q.DATA / f"qg3_{args.model}_{label}"
        np.savez_compressed(dest.with_suffix(".npz"), P=P, N=N, POS=POS)
        vr2 = {"max_abs_events_minus_positions": float(np.abs(N.sum(2) - POS).max()),
               "max_rel_prob_minus_positions": float(
                   (np.abs(P.sum(2) - POS) / np.maximum(POS, 1)).max())}
        meta = {"registration": "research/qgap/PREREG_QG3.md", "model": args.model,
                "label": label, "stem": Q.STREAMS[label][0], "rows": len(urls),
                "read": read, "block": BLOCK, "limit": args.limit,
                "itos": list(tok.itos), "chars": sorted(tok._chars), "vr2": vr2,
                "regions": list(Q.REGIONS)}
        dest.with_suffix(".json").write_text(json.dumps(meta), encoding="utf-8")
        print(f"  {args.model:<8} {label:<9} rows {read:>6,}  positions "
              + ", ".join(f"{r} {int(POS[:, i].sum()):,}" for i, r in enumerate(Q.REGIONS))
              + f"  VR2 {vr2['max_abs_events_minus_positions']:.0f} / "
              f"{vr2['max_rel_prob_minus_positions']:.1e}  {time.time() - t0:5.0f}s", flush=True)


if __name__ == "__main__":
    main()
