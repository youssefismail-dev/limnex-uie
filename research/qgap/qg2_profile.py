"""QG2-D probes 1 and 2 (research/qgap/PREREG_QG2D.md sections 2-4): the token
profile of the baseline on five slices. Diagnostic; laptop; no training, no
generation; the baseline is only read.

    python research/qgap/qg2_profile.py            # all slices, both classes

  S0  QG1's domain rows, frame domain, P ?, Q ?         (free mode, the reference)
  S1  structure_train rows, frame page, P ?, Q ?         (multi-page sites)
  S2  S0's rows, frame PAGE, P ?, Q ?                    (the frame swapped)
  S3  S0's rows, frame domain, P TRUE, Q ?               (the path state told)
  S4  S0's rows, frame domain, P ?, Q TRUE               (the query state told)

Writes data/qgap/qg2_<slice>_<label>.npz plus a .json of its URLs and checks.
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
from qg1_measure import sample_rows, vq1_model  # noqa: E402

N_ROWS = 40_000
#: slice -> (row source, frame, hidden controls)
SLICES = {"S0": ("domain", "domain", ("PTH", "QRY")),
          "S1": ("page", "page", ("PTH", "QRY")),
          "S2": ("domain", "page", ("PTH", "QRY")),
          "S3": ("domain", "domain", ("QRY",)),
          "S4": ("domain", "domain", ("PTH",))}


def encode(tok, url, label, frame, hide):
    try:
        return tok.encode(url, label, frame=frame, hide_fields=hide)
    except Exception:
        return None


def vp3(tok, seqs, hide) -> int:
    """Rows whose control tokens are not as registered: `?` where hidden, the
    row's own presence (its markers) where shown."""
    pos = tok.field_control_positions()
    bad = 0
    for s in seqs:
        if s is None:
            continue
        for field, marker in (("PTH", "PTH"), ("QRY", "QRY")):
            want = tok.field_control_id(field, None if field in hide
                                        else tok.marker_id(marker) in s)
            if s[pos[field]] != want:
                bad += 1
                break
    return bad


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--slices", default=",".join(SLICES))
    ap.add_argument("--limit", type=int, help="dry runs only: rows per slice")
    args = ap.parse_args()
    try:                                    # the laptop stays usable (UIE practice)
        import psutil
        psutil.Process().nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
    except Exception:
        pass
    model, tok, cfg = vq1_model("e9_ln")
    print("VQ1: e9_ln verified", flush=True)
    Q.DATA.mkdir(parents=True, exist_ok=True)
    for label in Q.LABELS:
        rows = {}
        for source in {SLICES[s][0] for s in args.slices.split(",")}:
            stem = Q.STREAMS[label][0] if source == "domain" else Q.PAGE[label]
            urls, _, pool, dropped = sample_rows(stem, N_ROWS, tok, label, cfg.max_len,
                                                 args.limit)
            rows[source] = (urls, stem, pool, dropped)
        for name in args.slices.split(","):
            t0 = time.time()
            source, frame, hide = SLICES[name]
            urls, stem, pool, dropped = rows[source]
            seqs = [encode(tok, u, label, frame, hide) for u in urls]
            recs = Q.profile(model, tok, seqs, max_len=cfg.max_len)
            ok = [r for r in recs if r is not None]
            n, k = len(ok), len(Q.CATS)
            arr = {key: np.array([r[key] for r in ok], dtype=np.float64)
                   for key in ("hp", "hy", "pp", "py", "hep")}
            arr["pep"] = np.array([r["pep"] if r["pep"] is not None else [np.nan] * k
                                   for r in ok], dtype=np.float64)
            arr["hey"] = np.array([r["hey"] for r in ok], dtype=np.int64)
            arr["pey"] = np.array([r["pey"] if r["pey"] is not None else -1 for r in ok],
                                  dtype=np.int64)
            arr["has_path"] = np.array([r["has_path"] for r in ok], dtype=np.int64)
            arr["row"] = np.array([i for i, r in enumerate(recs) if r is not None],
                                  dtype=np.int64)
            dest = Q.DATA / f"qg2_{name}_{label}"
            np.savez_compressed(dest.with_suffix(".npz"), **arr)
            meta = {"registration": "research/qgap/PREREG_QG2D.md", "slice": name,
                    "label": label, "source": stem, "frame": frame, "hidden": list(hide),
                    "pool": pool, "skipped_too_long": dropped, "limit": args.limit,
                    "rows": len(urls), "readable": n, "cats": list(Q.CATS),
                    "vp2_max_dev": max((r["dev"] for r in ok), default=0.0),
                    "vp3_bad_controls": vp3(tok, seqs, hide), "urls": urls}
            dest.with_suffix(".json").write_text(json.dumps(meta), encoding="utf-8")
            qi = Q.CATS.index("QRY")      # counts only: nothing is read before validity
            print(f"  {name} {label:<9} rows {n:>6,}  host QRY events {int(arr['hy'][:, qi].sum()):>5,}"
                  f"  path QRY events {int(arr['py'][:, qi].sum()):>5,}  VP2 {meta['vp2_max_dev']:.1e}"
                  f"  VP3 bad {meta['vp3_bad_controls']}  {time.time() - t0:6.0f}s", flush=True)


if __name__ == "__main__":
    main()
