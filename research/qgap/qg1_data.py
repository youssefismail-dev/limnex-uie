"""QG1 section 5.4, the data side (research/qgap/QG_PLAN.md): query rates over the
whole pools -- no model involved.

- D under the shipped table over all of `name_train`, for all rows and for rows
  that fit 256 tokens. The shipped-weighted sum is computed two ways:
  FINDING_LENGTH_WEIGHTING's (divided by the whole table's mass, as
  e8_length_weighting.py e7_mode does), and renormalised over the suffixes the pool
  has.
- The same on the validation pool.
- The page frame's (`structure_train`) plain query rate, all rows and rows that fit.

    python research/qgap/qg1_data.py        # writes data/qgap/qg1_data.json
"""
from __future__ import annotations

import collections
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE))

import torch  # noqa: E402

import qg_common as Q  # noqa: E402
from uie.baseline import E9_LN as BASELINE  # built on e9_ln  # noqa: E402
from uie.evaluation import FIELDS  # noqa: E402
from uie.suffixes import SuffixTable  # noqa: E402
from uie.train import load_checkpoint  # noqa: E402


def pool_counts(stem: str, label: str, tok, max_len: int):
    """suffix -> [rows, queries, rows that fit, queries among them]."""
    per = collections.defaultdict(lambda: [0, 0, 0, 0])
    for u in Q.rows(stem):
        s = FIELDS["SUF"](u) or ""
        q = int(bool(FIELDS["QRY"](u)))
        fit = int(len(tok.encode(u, label, frame="domain")) <= max_len)
        c = per[s]
        c[0] += 1
        c[1] += q
        c[2] += fit
        c[3] += q * fit
    return per


def shipped_rate(per, shipped, rows_i: int, q_i: int) -> dict:
    wsum = sum(shipped.values())
    present = {s: c for s, c in shipped.items() if per.get(s, [0] * 4)[rows_i]}
    finding = sum(c / wsum * per[s][q_i] / per[s][rows_i] for s, c in present.items())
    psum = sum(present.values())
    renorm = sum(c / psum * per[s][q_i] / per[s][rows_i] for s, c in present.items())
    return {"finding": finding, "renormalised": renorm, "table_mass_present": psum / wsum}


def main() -> None:
    _, tok, cfg = load_checkpoint(ROOT / "data" / "e9" / "train" / "e9_ln" / "best.pt",
                                  device=torch.device("cpu"))
    shipped = SuffixTable.load(Q.BUNDLE / "suffix_table.json").counts
    out = {"registration": "research/qgap/QG_PLAN.md (QG1, section 5.4)",
           "tokenizer": f"e9_ln ({BASELINE.file_sha256[:12]}...)"}
    for label in Q.LABELS:
        res = {}
        for which, stem in (("train", Q.STREAMS[label][0]), ("val", Q.STREAMS[label][1]),
                            ("page", Q.PAGE[label])):
            per = pool_counts(stem, label, tok, cfg.max_len)
            tot = [sum(c[i] for c in per.values()) for i in range(4)]
            res[which] = {
                "stem": stem, "rows": tot[0], "rows_fit": tot[2],
                "plain_all": tot[1] / tot[0], "plain_fit": tot[3] / tot[2],
                "shipped_all": shipped_rate(per, shipped[label], 0, 1),
                "shipped_fit": shipped_rate(per, shipped[label], 2, 3)}
            r = res[which]
            print(f"  {label:<9} {which:<5} rows {r['rows']:>9,}  plain {100 * r['plain_all']:.2f}% "
                  f"(fit {100 * r['plain_fit']:.2f}%)  shipped {100 * r['shipped_all']['finding']:.2f}% "
                  f"(fit {100 * r['shipped_fit']['finding']:.2f}%)  renormalised "
                  f"{100 * r['shipped_all']['renormalised']:.2f}% "
                  f"(fit {100 * r['shipped_fit']['renormalised']:.2f}%)", flush=True)
        out[label] = res
    Q.DATA.mkdir(parents=True, exist_ok=True)
    (Q.DATA / "qg1_data.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    print("wrote", Q.DATA / "qg1_data.json")


if __name__ == "__main__":
    main()
