"""HC1 -- H3: what masking costs, against the exact conditional.

Registered in research/hc1/PREREG_HC1.md, section 4, on the two query-on cells.

  masked arm  H1's <Q:1> URLs (data/hc1/h1/<class>_Q1.json)
  exact arm   generated here by REJECTION, suffix by suffix: H1's seed, suffix
              draws and groups; each group first drawn without enforcement at
              seed + i (exactly the masked arm's group seed), and every URL that
              does not honour <Q:1> redrawn with the SAME suffix at
              seed + i + 1,000,003*k on retry k (at most 100 retries per group).
              Accepted URLs are exact draws from p(x | suffix, rules hold), and
              both arms carry the same suffix mix by construction.

Both arms are scored against the real pool with the gate's own discriminator
(`discriminate_with_floor`: the field's gate population, one URL per host,
n = 1,200 per draw) on QRY and FULL, with D = 10 split seeds SHARED by the arms,
so each seed yields a paired difference. Delta = x_mask - x_exact:

    equivalent   95% CI of Delta inside (-0.02, +0.02)
    distorts     Delta >= 0.02 and the CI excludes 0
    unresolved   otherwise

Also: the exact arm's FIRST-PASS rate -- one unconstrained draw per slot, the
third estimate of P(honour) beside C1 and H2's mean w -- and its accepted/draws
rate, which with the suffix held is the HARMONIC mean of the per-suffix rates
(amendment 2026-09-24); the arms' mean field lengths. gc.collect() follows every
discriminator call, as in the gate.

    python research/hc1/hc1_h3.py --checkpoint data/best_e7.pt --bundle data/bundle2 \
        --h1 data/hc1/h1 --c1 data/e7_eval/c1/c1_report.json --out data/hc1/h3
"""
import argparse
import collections
import gc
import hashlib
import json
import math
import random
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import torch  # noqa: E402
from scipy.stats import t as student_t  # noqa: E402

from uie.corpus.run_eval import _find, _rows  # noqa: E402
from uie.evaluation import FIELD_POPULATION, FIELDS, discriminate_with_floor  # noqa: E402
from uie.model import generate  # noqa: E402
from uie.suffixes import SuffixTable  # noqa: E402
from uie.train import load_checkpoint  # noqa: E402

DELTA = {"benign": 0.22, "phishing": 0.30}
DECODER, DECODER_KWARGS = "temperature", {"temperature": 1.0}
SEED = 20260925                     # H1's base; cell i uses SEED + 10,000*i
CELL_INDEX = {"benign": 2, "phishing": 6}   # the Q:1 cells, in C1's order
SPLIT_BASE = SEED + 500
RETRY_STRIDE = 1_000_003
MAX_RETRIES = 100
TOL = 0.02


def generate_rejection(model, tok, *, label, n, table, seed, device):
    """The exact arm: rejection with the suffix held, group by group."""
    rng = random.Random(seed)
    drawn = table.sample(label, n, rng)
    groups = sorted(collections.Counter(zip(["domain"] * n, drawn)).items())
    d = DELTA[label]
    fb = {"PTH": {tok.eos_id: -d}} if d else None
    out, draws, shortfall, retries, first_pass = [], 0, 0, 0, 0
    for i, ((frame, suffix), count) in enumerate(groups):
        kept, k = [], 0
        while len(kept) < count and k <= MAX_RETRIES:
            urls = generate(model, tok, label=label, n=count - len(kept), suffix=suffix,
                            decoder=DECODER, seed=seed + i + RETRY_STRIDE * k,
                            frame=frame, field_bias=fb, batch_size=32, device=device,
                            decoder_kwargs=DECODER_KWARGS, fields={"QRY": True})
            draws += len(urls)
            ok = [u for u in urls if FIELDS["QRY"](u)]
            if k == 0:
                first_pass += len(ok)        # one unconstrained draw per slot
            kept.extend(ok)
            k += 1
        retries += k - 1
        shortfall += max(0, count - len(kept))
        out.extend(kept[:count])
    return out, {"draws": draws, "accepted": len(out), "shortfall": shortfall,
                 "retries": retries, "groups": len(groups), "first_pass": first_pass}


def parsed_tv(a, b) -> float:
    """Total variation between the suffixes the parser reads in two URL lists."""
    ca = collections.Counter(FIELDS["SUF"](u) for u in a)
    cb = collections.Counter(FIELDS["SUF"](u) for u in b)
    return 0.5 * sum(abs(ca[k] / len(a) - cb[k] / len(b)) for k in set(ca) | set(cb))


def pools(bundle: Path, label: str) -> dict:
    prefix = "" if label == "benign" else f"{label}_"
    name = _rows(_find(bundle, f"{prefix}name_val"))
    structure = _rows(_find(bundle, f"{prefix}structure_val"))
    return {"name": name, "structure": structure, "mixed": name + structure}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--bundle", type=Path, required=True)
    ap.add_argument("--h1", type=Path, required=True, help="H1's output directory")
    ap.add_argument("--c1", type=Path, required=True, help="C1's c1_report.json")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--n", type=int, default=1200)
    ap.add_argument("--splits", type=int, default=10)
    ap.add_argument("--classes", nargs="+", default=["benign", "phishing"])
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    device = torch.device("cpu")
    model, tok, _ = load_checkpoint(args.checkpoint, device=device)
    model.to(device).eval()
    table = SuffixTable.load(args.bundle / "suffix_table.json")
    h1 = json.loads((args.h1 / "h1_h2_report.json").read_text(encoding="utf-8"))
    c1 = {c["label"]: c["per_field"]["QRY"]["rate"]
          for c in json.loads(args.c1.read_text(encoding="utf-8"))["cells"]
          if c["kind"] == "single" and "QRY" in c["per_field"]
          and c["per_field"]["QRY"]["want"] == 1}

    report = {"test": "HC1 H3 (research/hc1/PREREG_HC1.md section 4)",
              "checkpoint_sha256": hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
              "split_seeds": [SPLIT_BASE + d for d in range(args.splits)],
              "tolerance": TOL, "classes": {}}
    for label in args.classes:
        t0 = time.time()
        seed = SEED + 10_000 * CELL_INDEX[label]
        masked = json.loads((args.h1 / f"{label}_Q1.json").read_text(encoding="utf-8"))
        cell = next(c for c in h1["cells"] if c["tag"] == f"{label}_Q1")
        assert cell["seed"] == seed and len(masked) == args.n, "H1's Q:1 cell does not match"
        exact, rej = generate_rejection(model, tok, label=label, n=args.n, table=table,
                                        seed=seed, device=device)
        (args.out / f"{label}_Q1_exact.json").write_text(json.dumps(exact), encoding="utf-8")
        # With the suffix held, a slot needs 1/P_s draws on average, so
        # accepted/draws is the HARMONIC mean of the per-suffix rates. The
        # marginal rate -- comparable with C1 and H2 -- is the FIRST pass: one
        # unconstrained draw per slot, in the table's suffix mix.
        acc = rej["accepted"] / rej["draws"]
        acc_se = math.sqrt(acc * (1 - acc) / rej["draws"])
        fp = rej["first_pass"] / args.n
        fp_se = math.sqrt(fp * (1 - fp) / args.n)
        out = {"seed": seed, "rejection": rej,
               "acceptance_harmonic": round(acc, 4), "acceptance_harmonic_se": round(acc_se, 4),
               "first_pass_rate": round(fp, 4), "first_pass_se": round(fp_se, 4),
               "c1_rate": c1.get(label),
               "h2_mean_w": cell["mean_w"], "tv": cell["tv"], "tv_ci95": cell["tv_ci95"],
               "tv_predicted": cell["tv_predicted"],
               "share_w_below_half": cell["share_w_below_half"],
               # The DRAWN suffixes are identical by construction (same seed, same
               # table.sample call). The PARSED ones can differ only where a
               # delimiter inside the host moved the parser's boundaries.
               "parsed_suffix_tv": round(parsed_tv(masked, exact), 4),
               "mean_lengths": {arm: {f: round(statistics.fmean(len(FIELDS[f](u))
                                                                for u in urls), 2)
                                      for f in ("LBL", "SUB", "PTH", "QRY")}
                                for arm, urls in (("masked", masked), ("exact", exact))},
               "fields": {}}
        print(f"  {label:<9} exact arm: first pass {100 * fp:.1f}% (C1 "
              f"{100 * c1.get(label, float('nan')):.1f}%, H2 mean w {100 * cell['mean_w']:.1f}%); "
              f"accepted/draws {rej['accepted']}/{rej['draws']} = {100 * acc:.1f}% (harmonic); "
              f"shortfall {rej['shortfall']}, "
              f"parsed-suffix TV {out['parsed_suffix_tv']}  ({time.time() - t0:.0f}s)",
              flush=True)
        real = pools(args.bundle, label)
        for field in ("QRY", "FULL"):
            pool = real[FIELD_POPULATION[field]]
            rows = []
            for d in range(args.splits):
                s = SPLIT_BASE + d
                a = discriminate_with_floor(pool, masked, field, n=1200, seed=s)
                gc.collect()        # as the gate does (2026-09-23): memory only
                b = discriminate_with_floor(pool, exact, field, n=1200, seed=s)
                gc.collect()
                ok = not (a.get("insufficient") or b.get("insufficient"))
                rows.append({"seed": s, "masked": a, "exact": b,
                             "delta": a["excess"] - b["excess"] if ok else None})
            deltas = [r["delta"] for r in rows if r["delta"] is not None]
            res = {"draws": rows, "usable": len(deltas)}
            if len(deltas) >= 2:
                m, sd = statistics.fmean(deltas), statistics.stdev(deltas)
                h = float(student_t.ppf(0.975, len(deltas) - 1)) * sd / math.sqrt(len(deltas))
                lo, hi = m - h, m + h
                res.update({
                    "delta": round(m, 4), "ci95": [round(lo, 4), round(hi, 4)],
                    "x_masked": round(statistics.fmean(r["masked"]["excess"] for r in rows
                                                       if r["delta"] is not None), 4),
                    "x_exact": round(statistics.fmean(r["exact"]["excess"] for r in rows
                                                      if r["delta"] is not None), 4),
                    "n_per_draw": rows[0]["masked"].get("n"),
                    "verdict": ("equivalent" if -TOL < lo and hi < TOL else
                                "distorts" if m >= TOL and lo > 0 else "unresolved")})
            else:
                res["verdict"] = "not computable"
            out["fields"][field] = res
            print(f"  {label:<9} {field:<5} delta {res.get('delta')}  CI {res.get('ci95')}  "
                  f"x_mask {res.get('x_masked')}  x_exact {res.get('x_exact')}  "
                  f"n {res.get('n_per_draw')}  -> {res['verdict']}", flush=True)
        report["classes"][label] = out
        (args.out / "h3_report.json").write_text(json.dumps(report, indent=1, default=float),
                                                 encoding="utf-8")


if __name__ == "__main__":
    main()
