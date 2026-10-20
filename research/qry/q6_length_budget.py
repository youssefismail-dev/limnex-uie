"""Q6 -- do long query values survive the length budget into the loss?

READ-ONLY. Encodes corpus rows with the frozen tokenizer to count what training
would keep or drop. No training, no generation, no weight touched, no gate.

Q5 established the training corpus is tail-RICH (value p99 ~110 characters)
while the model emits p99 40-56. The corpus is not the problem, so something
between the corpus and the loss is. `encode_corpus` is the obvious suspect and
it does NOT truncate:

    if len(ids) > max_len:
        dropped += 1
        continue

Rows over the budget are DELETED. ModelConfig defends that choice and the
defence is sound -- a truncated URL teaches a URL that ends mid-path with no end
token. But deletion is not neutral with respect to the tail: the rows it removes
are exactly the longest ones, and at character level a long query value is one
of the few things that pushes a URL over 256 tokens.

So the question is not "is the tail truncated" but "is the tail DELETED", and
the decisive measurement is the value-length distribution the loss actually
sees, against the distribution present in the corpus.

If surviving p99 collapses toward the model's emitted p99, the length budget is
the mechanism. If survivors keep the corpus tail, it is not, and the learning
failure lies elsewhere.
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
from research.qry.qtransforms import split_params   # noqa: E402
from uie.evaluation import FIELDS                   # noqa: E402
from uie.train import CORPUS_FRAME, load_checkpoint  # noqa: E402

CAP = 400_000          # rows read per corpus; keeps the pass a few minutes
B = Path("data/bundle2")


def pct(sorted_vals: List[int], p: float):
    return sorted_vals[min(int(p * len(sorted_vals)), len(sorted_vals) - 1)] \
        if sorted_vals else None


def main() -> None:
    # The tokenizer and max_len must come from the CHECKPOINT, not from
    # defaults: encoding with a different vocabulary or frame flag would answer
    # a question about some other model.
    model, tok, cfg = load_checkpoint(Path("data/best_e5.pt"), device="cpu")
    max_len = model.cfg.max_len
    print(f"checkpoint max_len = {max_len}   include_frame="
          f"{getattr(tok, 'include_frame', False)}   "
          f"include_language={getattr(tok, 'include_language', True)}\n",
          flush=True)

    report: Dict[str, object] = {"max_len": max_len, "cap": CAP, "corpora": {}}

    for stem, corpus in (("name_train", "name"),
                         ("structure_train", "structure")):
        frame = CORPUS_FRAME.get(corpus, "domain")
        with gzip.open(B / f"{stem}.jsonl.gz", "rt", encoding="utf-8") as fh:
            urls = [json.loads(l)["url"] for i, l in enumerate(fh) if i < CAP]

        kept_vals: List[int] = []       # value lengths the LOSS sees
        all_vals: List[int] = []        # value lengths in the CORPUS
        kept_names: List[str] = []
        all_names: List[str] = []
        kept_counts: List[int] = []
        all_counts: List[int] = []
        n_dropped = n_q = n_q_dropped = 0
        # drop rate bucketed by the longest value in the row -- the quantity
        # under suspicion
        buckets = [(0, 10), (10, 25), (25, 50), (50, 100), (100, 10 ** 9)]
        bucket_tot = collections.Counter()
        bucket_drop = collections.Counter()

        for u in urls:
            ids = tok.encode(u, "benign", "und", frame=frame)
            over = len(ids) > max_len
            n_dropped += over
            q = FIELDS["QRY"](u)
            if not q:
                continue
            parts = split_params(q)
            vals = [len(v) for _, s, v in parts if s]
            names = [k for k, _, _ in parts]
            n_q += 1
            all_vals.extend(vals)
            all_names.extend(names)
            all_counts.append(len(parts))
            longest = max(vals) if vals else 0
            for lo, hi in buckets:
                if lo <= longest < hi:
                    bucket_tot[(lo, hi)] += 1
                    if over:
                        bucket_drop[(lo, hi)] += 1
                    break
            if over:
                n_q_dropped += 1
                continue
            kept_vals.extend(vals)
            kept_names.extend(names)
            kept_counts.append(len(parts))

        a, k = sorted(all_vals), sorted(kept_vals)
        row = {
            "rows_read": len(urls),
            "rows_dropped": n_dropped,
            "drop_pct": round(100 * n_dropped / len(urls), 3),
            "query_rows": n_q,
            "query_rows_dropped": n_q_dropped,
            "query_drop_pct": round(100 * n_q_dropped / max(n_q, 1), 3),
            "value_p50_corpus": pct(a, .50), "value_p50_kept": pct(k, .50),
            "value_p90_corpus": pct(a, .90), "value_p90_kept": pct(k, .90),
            "value_p99_corpus": pct(a, .99), "value_p99_kept": pct(k, .99),
            "value_max_corpus": (a[-1] if a else None),
            "value_max_kept": (k[-1] if k else None),
            "value_mean_corpus": round(st.mean(a), 2) if a else None,
            "value_mean_kept": round(st.mean(k), 2) if k else None,
            "param_count_corpus": round(st.mean(all_counts), 3) if all_counts else None,
            "param_count_kept": round(st.mean(kept_counts), 3) if kept_counts else None,
            "name_types_corpus": len(set(all_names)),
            "name_types_kept": len(set(kept_names)),
            "drop_by_longest_value": {f"{lo}-{hi if hi < 10**9 else 'inf'}":
                                      {"rows": bucket_tot[(lo, hi)],
                                       "dropped": bucket_drop[(lo, hi)],
                                       "pct": round(100 * bucket_drop[(lo, hi)]
                                                    / max(bucket_tot[(lo, hi)], 1), 2)}
                                      for lo, hi in buckets},
        }
        report["corpora"][stem] = row

        print(f"=== {stem} (frame={frame}) ===")
        print(f"  rows read {row['rows_read']:,}   dropped over max_len "
              f"{row['rows_dropped']:,} ({row['drop_pct']}%)")
        print(f"  query-bearing rows {row['query_rows']:,}   of those dropped "
              f"{row['query_rows_dropped']:,} ({row['query_drop_pct']}%)")
        print(f"  {'value length':<18}{'corpus':>10}{'kept by loss':>14}")
        for p in ("p50", "p90", "p99", "max", "mean"):
            ck, kk = f"value_{p}_corpus", f"value_{p}_kept"
            print(f"  {p:<18}{str(row[ck]):>10}{str(row[kk]):>14}")
        print(f"  {'param count mean':<18}{row['param_count_corpus']:>10}"
              f"{row['param_count_kept']:>14}")
        print(f"  {'distinct names':<18}{row['name_types_corpus']:>10,}"
              f"{row['name_types_kept']:>14,}")
        print("  drop rate by the row's LONGEST value:")
        for kk, vv in row["drop_by_longest_value"].items():
            print(f"    {kk:>10} chars  {vv['rows']:>7,} rows  "
                  f"{vv['dropped']:>6,} dropped  {vv['pct']:>6.2f}%")
        print(flush=True)

    out = Path("data/qry/q6_length_budget.json")
    out.write_text(json.dumps(report, indent=1) + "\n", encoding="utf-8")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
