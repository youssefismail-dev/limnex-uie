"""Q4 -- is the query CONTENT defect a selection artifact of the decoder?

Pre-registered in research/qry/PREREGISTRATION.md. Q1 established the condition
this experiment needs: production emits 2.30% queries out of a 4.25% model
marginal in benign, so the queries that reach output ARE a survivorship-biased
subsample -- the ones that happened to survive truncation.

If that selection is what makes them distinguishable, then queries drawn WITHOUT
truncation should be markedly less distinguishable from real ones. If both arms
are equally distinguishable, the content defect is a genuine modelling failure
and is separable from the emission defect.

    arm T   queries from `typical`      -- production, truncated
    arm A   queries from `temperature`  -- untruncated, the model's own sample

Both are scored against the SAME real comparator with the floor recomputed, and
both are subsampled to the SAME n, because AUC is not comparable across sample
sizes and the two arms would otherwise differ in n by construction.

Nothing here modifies the generator; it calls frozen decoders by name.
"""
from __future__ import annotations

import argparse
import json
import statistics as st
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from uie.evaluation import FIELDS, discriminate_with_floor      # noqa: E402
from uie.suffixes import SuffixTable, generate_with_suffixes    # noqa: E402
from uie.train import load_checkpoint                           # noqa: E402

import gzip  # noqa: E402

CHUNK = 4000
SPLIT_SEEDS = (20260806, 20260807, 20260808)

ARMS = [("typical", {}, "T  production, truncated"),
        ("temperature", {"temperature": 1.0}, "A  untruncated model sample")]


def real_rows(bundle: Path, cls: str) -> list:
    pre = "" if cls == "benign" else f"{cls}_"
    with gzip.open(bundle / f"{pre}name_val.jsonl.gz", "rt", encoding="utf-8") as fh:
        return [json.loads(l)["url"] for l in fh]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--class", dest="cls", default="benign",
                    choices=["benign", "phishing"])
    ap.add_argument("--target", type=int, default=400,
                    help="query-bearing rows wanted per arm")
    ap.add_argument("--max-gen", type=int, default=60000,
                    help="hard cap on generated URLs per arm")
    # Defaults to the frozen baseline so every published Q4 artifact stays
    # bit-identical. A non-default checkpoint TAGS its output files rather than
    # overwriting the baseline's -- q8_exposure reads those files by name, and
    # silently replacing them would re-point the published band profile at a
    # different model with no record of it.
    ap.add_argument("--checkpoint", default="data/best_e5.pt")
    args = ap.parse_args()
    stem = Path(args.checkpoint).stem
    tag = "" if stem == "best_e5" else f"_{stem}"

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, tok, _ = load_checkpoint(Path(args.checkpoint), device=dev)
    model.to(dev).eval()
    sfx = SuffixTable.load(Path("data/bundle2/suffix_table.json"))
    bundle = Path("data/bundle2")
    real = real_rows(bundle, args.cls)
    out_dir = Path("data/qry")
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"=== Q4 {args.cls} ===  real pool {len(real):,}  "
          f"target {args.target} query rows/arm  ckpt={stem}", flush=True)

    harvest = {}
    for decoder, kwargs, role in ARMS:
        got, seen, seed = [], 0, 70000
        t0 = time.time()
        while len(got) < args.target and seen < args.max_gen:
            urls = generate_with_suffixes(
                model, tok, label=args.cls, n=CHUNK, table=sfx, decoder=decoder,
                seed=seed, device=dev, decoder_kwargs=kwargs or None)
            seen += len(urls)
            seed += 1
            got.extend(u for u in urls if FIELDS["QRY"](u))
            print(f"  {decoder:<12} {len(got):>4} query rows from {seen:>6,} "
                  f"generated  ({100*len(got)/seen:.2f}%)  "
                  f"[{time.time()-t0:.0f}s]", flush=True)
        harvest[decoder] = {"urls": got, "generated": seen, "role": role,
                            "rate_pct": round(100 * len(got) / max(seen, 1), 3),
                            "seconds": round(time.time() - t0, 1)}
        (out_dir / f"q4_{args.cls}_{decoder}_queries{tag}.json").write_text(
            json.dumps(got, indent=0), encoding="utf-8")

    # EQUAL n ACROSS ARMS. AUC is not comparable across sample sizes, and the
    # two arms differ in yield by construction -- arm T is rarer. Scoring them at
    # their natural sizes would confound selection with sample size, which is the
    # single mistake that would invalidate this experiment.
    n_common = min(len(v["urls"]) for v in harvest.values())
    print(f"\n  scoring BOTH arms at n_fake = {n_common} (equal by construction)",
          flush=True)

    results = {"class": args.cls, "n_common": n_common,
               "split_seeds": list(SPLIT_SEEDS), "arms": {}}
    for decoder, h in harvest.items():
        rows = []
        for s in SPLIT_SEEDS:
            r = discriminate_with_floor(real, h["urls"][:n_common], "QRY",
                                        n=n_common, seed=s)
            rows.append(r)
            print(f"  {decoder:<12} seed {s}  AUC {r.get('auc')}  "
                  f"floor {r.get('floor')}  excess {r.get('excess'):+.4f}  "
                  f"n {r.get('n')}", flush=True)
        ex = [r["excess"] for r in rows if not r.get("insufficient")]
        results["arms"][decoder] = {
            "role": h["role"], "generated": h["generated"],
            "rate_pct": h["rate_pct"], "n_query_rows": len(h["urls"]),
            "excess_mean": round(st.mean(ex), 4) if ex else None,
            "excess_sd": round(st.pstdev(ex), 4) if len(ex) > 1 else 0.0,
            "excess_min": round(min(ex), 4) if ex else None,
            "excess_max": round(max(ex), 4) if ex else None,
            "auc_mean": round(st.mean([r["auc"] for r in rows]), 4) if ex else None,
            "floor_mean": round(st.mean([r["floor"] for r in rows]), 4) if ex else None,
            "draws": rows}

    a = results["arms"].get("typical", {}).get("excess_mean")
    b = results["arms"].get("temperature", {}).get("excess_mean")
    if a is not None and b is not None:
        results["delta_excess_A_minus_T"] = round(b - a, 4)
        print(f"\n  arm T (typical)     excess {a:+.4f}")
        print(f"  arm A (temperature) excess {b:+.4f}")
        print(f"  A - T = {b - a:+.4f}")

    p = out_dir / f"q4_{args.cls}{tag}.json"
    p.write_text(json.dumps(results, indent=1, default=str) + "\n",
                 encoding="utf-8")
    print(f"\nwrote {p}")


if __name__ == "__main__":
    main()
