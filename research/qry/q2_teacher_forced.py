"""Q2 -- does the model under-assign the query decision, or drift away from it?

Pre-registered in research/qry/PREREGISTRATION.md. Exact: no sampling anywhere.

Q1 measured P(query) under the model's OWN prefixes -- 4.25% benign against a
6.403% corpus rate. That is a marginal. It cannot distinguish two very different
failures:

    mismodelled decision   at a given prefix the model assigns too little
                           probability to opening a query
    prefix drift           the conditional is fine, but the model's own
                           trajectories wander into states where a query is
                           unlikely, so the marginal falls anyway
                           (this is exposure bias)

Teacher forcing separates them, because it holds the prefix FIXED at the real
one. For every real validation URL we walk its true token sequence and, at every
position where the query marker is a legal next token, read p(QRY). The
counterfactual "would this URL have opened a query" is then

    P_open = 1 - PROD over eligible t of (1 - p_t(QRY))

which is exact under teacher forcing, because the trajectory being conditioned on
is the real one at every step.

    If mean P_open ~= the corpus rate  -> conditional is healthy, Q1's gap is
                                          DRIFT
    If mean P_open ~= the Q1 marginal  -> the decision itself is mismodelled

Also recorded, per position: whether `decode_typical`'s filter would have
RETAINED the query marker at that position. That gives the decoder's truncation
rate directly, at the exact positions where it matters, rather than inferring it
from output counts.

Nothing here modifies the generator or the checkpoint.
"""
from __future__ import annotations

import argparse
import gzip
import json
import statistics as st
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from uie.evaluation import FIELDS            # noqa: E402
from uie.train import load_checkpoint        # noqa: E402

BATCH = 64


def typical_keeps(logits: torch.Tensor, target: int, mass: float = 0.95) -> bool:
    """Would decode_typical retain `target` in its candidate set here?

    Mirrors uie.model.decode_typical exactly -- same ordering by distance from
    entropy, same cumulative-mass cutoff. Reimplemented rather than imported
    because the shipped function RETURNS A SAMPLE; we need the membership test,
    and the production code is frozen.
    """
    logp = F.log_softmax(logits, dim=-1)
    probs = logp.exp()
    entropy = -(probs * logp).sum()
    deviation = (-logp - entropy).abs()
    _, order = torch.sort(deviation)
    cumulative = torch.cumsum(probs[order], dim=-1)
    keep = int(torch.searchsorted(cumulative, torch.tensor(mass)).item()) + 1
    keep = min(max(keep, 1), order.numel())
    return bool((order[:keep] == target).any().item())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--class", dest="cls", default="benign",
                    choices=["benign", "phishing"])
    ap.add_argument("--n", type=int, default=2000)
    args = ap.parse_args()

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, tok, _ = load_checkpoint(Path("data/best_e5.pt"), device=dev)
    model.to(dev).eval()

    pre = "" if args.cls == "benign" else f"{args.cls}_"
    path = Path("data/bundle2") / f"{pre}name_val.jsonl.gz"
    with gzip.open(path, "rt", encoding="utf-8") as fh:
        rows = [json.loads(l)["url"] for i, l in enumerate(fh) if i < args.n]

    qry_id = tok.marker_id("QRY")
    host_markers = {tok.marker_id(m) for m in ("SUF", "LBL", "SUB", "PRT")}
    struct_markers = {tok.marker_id(m) for m in ("PTH", "QRY", "FRG")}

    print(f"=== Q2 {args.cls} ===  {len(rows):,} real URLs, teacher forced",
          flush=True)

    open_probs, kept_flags, at_true = [], [], []
    with_q = sum(1 for u in rows if FIELDS["QRY"](u))
    t0 = time.time()

    for start in range(0, len(rows), BATCH):
        chunk = rows[start:start + BATCH]
        enc = [tok.encode(u, args.cls)[:model.cfg.max_len] for u in chunk]
        width = max(len(e) for e in enc)
        ids = torch.tensor([e + [tok.pad_id] * (width - len(e)) for e in enc],
                           dtype=torch.long, device=dev)
        with torch.no_grad():
            logits = model(ids)                       # [B, T, V]

        for b, seq in enumerate(enc):
            # ELIGIBLE POSITIONS: those from which <QRY> is a legal next token.
            # That is every position inside the path body, plus the last token
            # of the host section when there is no path -- the query decision is
            # renewed at each path step, it is not taken once.
            eligible, in_struct, last_host = [], False, None
            for t, tid in enumerate(seq):
                if tid in host_markers:
                    in_struct = False
                if tid in struct_markers:
                    in_struct = True
                if not in_struct:
                    last_host = t
                elif tid != tok.marker_id("QRY"):
                    eligible.append(t)
            if last_host is not None:
                eligible.append(last_host)
            eligible = sorted(set(t for t in eligible if t < len(seq) - 1))
            if not eligible:
                continue

            survive = 1.0
            for t in eligible:
                p = float(F.softmax(logits[b, t], dim=-1)[qry_id].item())
                survive *= (1.0 - p)
                kept_flags.append(typical_keeps(logits[b, t], qry_id))
            open_probs.append(1.0 - survive)

            # Where the real URL DOES open a query, record p at that exact step.
            if qry_id in seq:
                j = seq.index(qry_id)
                if j > 0:
                    at_true.append(
                        float(F.softmax(logits[b, j - 1], dim=-1)[qry_id].item()))

        if start % (BATCH * 8) == 0:
            print(f"  {start + len(chunk):>5}/{len(rows)}  "
                  f"[{time.time() - t0:.0f}s]", flush=True)

    res = {
        "class": args.cls, "n_urls": len(rows),
        "empirical_rate_pct": round(100 * with_q / len(rows), 3),
        "teacher_forced_P_open_pct": round(100 * st.mean(open_probs), 3),
        "teacher_forced_P_open_median_pct": round(100 * st.median(open_probs), 3),
        "p_at_true_QRY_mean": round(st.mean(at_true), 5) if at_true else None,
        "p_at_true_QRY_median": round(st.median(at_true), 5) if at_true else None,
        "n_true_QRY": len(at_true),
        "typical_retention_pct": round(100 * sum(kept_flags) / len(kept_flags), 3),
        "n_eligible_positions": len(kept_flags),
        "seconds": round(time.time() - t0, 1),
    }
    out = Path(f"data/qry/q2_{args.cls}.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, indent=1) + "\n", encoding="utf-8")

    print(f"\n  empirical query rate          {res['empirical_rate_pct']:.3f}%")
    print(f"  teacher-forced P(open)  mean  {res['teacher_forced_P_open_pct']:.3f}%"
          f"   median {res['teacher_forced_P_open_median_pct']:.3f}%")
    print(f"  p(QRY) at the TRUE marker     mean {res['p_at_true_QRY_mean']}"
          f"  median {res['p_at_true_QRY_median']}  (n={res['n_true_QRY']})")
    print(f"  typical() retains <QRY> at     {res['typical_retention_pct']:.2f}% "
          f"of {res['n_eligible_positions']:,} eligible positions")
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
