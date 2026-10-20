"""Q2b -- p(QRY) at the ONE position where a query would naturally attach,
plus the eligible-position count that Q2's drift hypothesis turns on.

Two corrections to Q2, both recorded rather than silently applied:

1. Q2 counted positions inside a FRAGMENT as eligible. `<QRY>` cannot legally
   follow `<FRG>` (marker order is ... PTH QRY FRG), so those positions inflated
   P_open. Fragments are rare here so the bias was small, but it was a bias.

2. Q2's compounding statistic, P_open = 1 - PROD(1 - p_t), depends on an
   eligibility definition that is genuinely ambiguous. The grammar lets a marker
   end the CURRENT field at any point, so `<QRY>` is technically legal
   mid-hostname -- emitting it after "exa" in "example" would be well-formed and
   absurd. Counting those inflates the set; excluding them is a judgement call
   with no obvious line.

   The ATTACHMENT POINT has no such ambiguity. Every URL has exactly one
   position where its path body ends and a query would naturally attach, and
   p(QRY) there is a single well-defined number per URL, directly comparable to
   the empirical rate. That is the primary statistic here; Q2's compounding
   figure is kept only as a secondary.

The drift hypothesis from Q2 says the marginal falls because the model produces
FEWER AND SHORTER PATHS, so there are fewer positions at which a query could
open, even though the per-step conditional is healthy. That predicts a specific,
checkable thing: generated URLs should have materially fewer eligible positions
per URL than real ones. This measures it directly on both sides.
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
from uie.evaluation import FIELDS                            # noqa: E402
from uie.suffixes import SuffixTable, generate_with_suffixes  # noqa: E402
from uie.train import load_checkpoint                        # noqa: E402

BATCH = 64


def spans(seq, tok):
    """Return (path_positions, attachment_index) for one encoded sequence.

    `path_positions` are indices INSIDE the path field (marker included), which
    is where a query may legally open. `attachment_index` is the last of them --
    the position from which the real URL leaves its path. With no path, the
    attachment point is the last host token.
    """
    pth, qry, frg = (tok.marker_id(m) for m in ("PTH", "QRY", "FRG"))
    host = {tok.marker_id(m) for m in ("SUF", "LBL", "SUB", "PRT")}
    state, path_pos, last_host = None, [], None
    for t, tid in enumerate(seq):
        if tid in host:
            state = "host"
        elif tid == pth:
            state = "path"
        elif tid in (qry, frg):
            state = "tail"          # FRG interior is NOT eligible -- fix (1)
        if state == "host":
            last_host = t
        elif state == "path":
            path_pos.append(t)
    attach = path_pos[-1] if path_pos else last_host
    return path_pos, attach


def probe(model, tok, urls, cls, dev):
    qry_id = tok.marker_id("QRY")
    at_attach, n_elig = [], []
    for start in range(0, len(urls), BATCH):
        chunk = urls[start:start + BATCH]
        enc = [tok.encode(u, cls)[:model.cfg.max_len] for u in chunk]
        width = max(len(e) for e in enc)
        ids = torch.tensor([e + [tok.pad_id] * (width - len(e)) for e in enc],
                           dtype=torch.long, device=dev)
        with torch.no_grad():
            logits = model(ids)
        for b, seq in enumerate(enc):
            path_pos, attach = spans(seq, tok)
            n_elig.append(len(path_pos))
            if attach is not None and attach < len(seq) - 1:
                at_attach.append(
                    float(F.softmax(logits[b, attach], dim=-1)[qry_id].item()))
    return at_attach, n_elig


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--class", dest="cls", default="benign",
                    choices=["benign", "phishing"])
    ap.add_argument("--n", type=int, default=1200)
    args = ap.parse_args()

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, tok, _ = load_checkpoint(Path("data/best_e5.pt"), device=dev)
    model.to(dev).eval()

    pre = "" if args.cls == "benign" else f"{args.cls}_"
    with gzip.open(Path("data/bundle2") / f"{pre}name_val.jsonl.gz", "rt",
                   encoding="utf-8") as fh:
        real = [json.loads(l)["url"] for i, l in enumerate(fh) if i < args.n]

    # The model's OWN sample, untruncated, which is the distribution whose
    # marginal Q1 measured at 4.25% / 1.95%.
    sfx = SuffixTable.load(Path("data/bundle2/suffix_table.json"))
    t0 = time.time()
    gen = generate_with_suffixes(model, tok, label=args.cls, n=args.n, table=sfx,
                                 decoder="temperature", seed=88001, device=dev,
                                 decoder_kwargs={"temperature": 1.0})
    print(f"=== Q2b {args.cls} ===  {len(real):,} real, {len(gen):,} generated "
          f"(temperature, untruncated) [{time.time()-t0:.0f}s]", flush=True)

    r_attach, r_elig = probe(model, tok, real, args.cls, dev)
    g_attach, g_elig = probe(model, tok, gen, args.cls, dev)

    res = {
        "class": args.cls, "n": args.n,
        "real_query_pct": round(100 * sum(1 for u in real if FIELDS["QRY"](u)) / len(real), 3),
        "gen_query_pct": round(100 * sum(1 for u in gen if FIELDS["QRY"](u)) / len(gen), 3),
        "p_attach_real_mean": round(st.mean(r_attach), 5),
        "p_attach_real_median": round(st.median(r_attach), 5),
        "p_attach_gen_mean": round(st.mean(g_attach), 5),
        "p_attach_gen_median": round(st.median(g_attach), 5),
        "eligible_real_mean": round(st.mean(r_elig), 2),
        "eligible_gen_mean": round(st.mean(g_elig), 2),
        "eligible_real_median": st.median(r_elig),
        "eligible_gen_median": st.median(g_elig),
        "pathless_real_pct": round(100 * sum(1 for x in r_elig if x == 0) / len(r_elig), 2),
        "pathless_gen_pct": round(100 * sum(1 for x in g_elig if x == 0) / len(g_elig), 2),
    }
    out = Path(f"data/qry/q2b_{args.cls}.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, indent=1) + "\n", encoding="utf-8")

    print(f"  query rate            real {res['real_query_pct']:6.3f}%   "
          f"generated {res['gen_query_pct']:6.3f}%")
    print(f"  p(QRY) at attachment  real {res['p_attach_real_mean']:.5f} "
          f"(med {res['p_attach_real_median']:.5f})   "
          f"generated {res['p_attach_gen_mean']:.5f} "
          f"(med {res['p_attach_gen_median']:.5f})")
    print(f"  eligible positions    real {res['eligible_real_mean']:6.2f} "
          f"(med {res['eligible_real_median']})   "
          f"generated {res['eligible_gen_mean']:6.2f} "
          f"(med {res['eligible_gen_median']})")
    print(f"  pathless URLs         real {res['pathless_real_pct']:5.2f}%   "
          f"generated {res['pathless_gen_pct']:5.2f}%")
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
