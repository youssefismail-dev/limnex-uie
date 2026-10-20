"""R1 sizing -- the READ-ONLY half. What would a larger length budget recover?

Pre-registered in REPAIR_PLAN.md R1 step 7: "Read-only first: recompute Q6's
drop table at candidate budgets to choose the smallest value recovering most of
the tail. Only then a single training run at that budget."

NOTHING IS MODIFIED. This encodes corpus rows with the frozen tokenizer and asks
a counterfactual question about a constant. No training, no generation, no
weight, checkpoint, gate or corpus touched. It does not change `max_len`; it
reports what changing it would buy and cost.

The costs are stated alongside the benefits on purpose. `max_len` is not free:
the positional embedding grows linearly in it and attention grows QUADRATICALLY,
so a budget chosen only on recovered tail would be half an answer. The point of
this measurement is to find the SMALLEST budget that recovers most of the tail,
not the largest one that recovers all of it.
"""
from __future__ import annotations

import collections
import gzip
import json
import statistics as st
import sys
from pathlib import Path
from typing import Dict, List

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from research.qry.qtransforms import split_params          # noqa: E402
from uie.evaluation import FIELDS                          # noqa: E402
from uie.train import CORPUS_FRAME, load_checkpoint        # noqa: E402

CAP = 400_000
CANDIDATES = [256, 320, 384, 448, 512, 640, 768]
B = Path("data/bundle2")


def pct(vals: List[int], p: float):
    return vals[min(int(p * len(vals)), len(vals) - 1)] if vals else None


def main() -> None:
    model, tok, _ = load_checkpoint(Path("data/best_e5.pt"), device="cpu")
    current = model.cfg.max_len
    d_model = model.cfg.d_model
    print(f"current max_len = {current}   d_model = {d_model}\n", flush=True)

    report: Dict[str, object] = {"current_max_len": current,
                                 "candidates": CANDIDATES, "corpora": {}}

    for stem, corpus in (("name_train", "name"),
                         ("structure_train", "structure")):
        frame = CORPUS_FRAME.get(corpus, "domain")
        with gzip.open(B / f"{stem}.jsonl.gz", "rt", encoding="utf-8") as fh:
            urls = [json.loads(l)["url"] for i, l in enumerate(fh) if i < CAP]

        # ONE encode pass; every candidate budget is then a comparison against
        # the stored length, so the candidates cost nothing extra and cannot
        # disagree with each other through re-encoding.
        recs = []
        for u in urls:
            n = len(tok.encode(u, "benign", "und", frame=frame))
            q = FIELDS["QRY"](u)
            if not q:
                recs.append((n, None, None, None))
                continue
            parts = split_params(q)
            vals = [len(v) for _, s, v in parts if s]
            recs.append((n, vals, [k for k, _, _ in parts], len(parts)))

        all_q = [r for r in recs if r[1] is not None]
        corpus_vals = sorted(v for r in all_q for v in r[1])
        corpus_names = {k for r in all_q for k in r[2]}
        tail_rows = [r for r in all_q if r[1] and max(r[1]) >= 100]

        print(f"=== {stem} ===  rows {len(recs):,}  query rows {len(all_q):,}  "
              f"rows with a 100+ char value {len(tail_rows):,}")
        print(f"  corpus value p99 {pct(corpus_vals, .99)}  "
              f"max {corpus_vals[-1] if corpus_vals else None}  "
              f"distinct names {len(corpus_names):,}")
        print(f"  {'max_len':>8}{'rows kept':>12}{'drop%':>8}{'qry drop%':>11}"
              f"{'100+ drop%':>12}{'p99 kept':>10}{'names kept':>12}"
              f"{'attn cost':>11}{'pos emb':>10}")

        rows_out = {}
        for cand in CANDIDATES:
            kept = [r for r in recs if r[0] <= cand]
            kept_q = [r for r in kept if r[1] is not None]
            kept_vals = sorted(v for r in kept_q for v in r[1])
            kept_names = {k for r in kept_q for k in r[2]}
            tail_dropped = sum(1 for r in tail_rows if r[0] > cand)
            attn = (cand / current) ** 2
            pos = cand / current
            line = {
                "rows_kept": len(kept),
                "drop_pct": round(100 * (len(recs) - len(kept)) / len(recs), 3),
                "qry_drop_pct": round(100 * (len(all_q) - len(kept_q))
                                      / max(len(all_q), 1), 3),
                "tail100_drop_pct": round(100 * tail_dropped
                                          / max(len(tail_rows), 1), 2),
                "value_p99_kept": pct(kept_vals, .99),
                "value_max_kept": kept_vals[-1] if kept_vals else None,
                "names_kept": len(kept_names),
                "names_kept_pct": round(100 * len(kept_names)
                                        / max(len(corpus_names), 1), 2),
                "attn_cost_x": round(attn, 2),
                "pos_emb_x": round(pos, 2),
            }
            rows_out[cand] = line
            mark = "  <- current" if cand == current else ""
            print(f"  {cand:>8}{line['rows_kept']:>12,}{line['drop_pct']:>8.3f}"
                  f"{line['qry_drop_pct']:>11.3f}{line['tail100_drop_pct']:>12.2f}"
                  f"{str(line['value_p99_kept']):>10}"
                  f"{line['names_kept_pct']:>11.1f}%"
                  f"{line['attn_cost_x']:>11.2f}x{line['pos_emb_x']:>9.2f}x{mark}")
        report["corpora"][stem] = {
            "rows": len(recs), "query_rows": len(all_q),
            "tail100_rows": len(tail_rows),
            "corpus_value_p99": pct(corpus_vals, .99),
            "corpus_value_max": corpus_vals[-1] if corpus_vals else None,
            "corpus_distinct_names": len(corpus_names),
            "by_budget": rows_out}
        print(flush=True)

    out = Path("data/qry/r1_sizing.json")
    out.write_text(json.dumps(report, indent=1) + "\n", encoding="utf-8")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
