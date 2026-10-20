"""HC1 -- H1 (the rules hold) and H2 (the mask is the model's own probability).

Registered in research/hc1/PREREG_HC1.md, section 4. For each of the 8 cells
(C1's order: each class x PTH, QRY x on, off), n URLs through the canonical
generation path -- `generate_with_suffixes`, reproduced step for step so each
URL keeps its trace -- with `enforce="mask"`:

    H1  every URL honours its control (the gate's FIELDS), none truncated
    H2  mean w = prod(1 - m_t) against C1's honouring rate: |z| < 3 in all 8

Also reported per cell: TV(masked, exact conditional) = 1/2 E|1 - w/E[w]| with
a bootstrap CI, and the share of rows with w < 0.5. Every URL and trace is
saved; the report is rewritten after each cell.

    python research/hc1/hc1_run.py --checkpoint data/best_e7.pt --bundle data/bundle2 \
        --c1 data/e7_eval/c1/c1_report.json --out data/hc1/h1
"""
import argparse
import collections
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

from uie.evaluation import FIELDS  # noqa: E402
from uie.model import generate  # noqa: E402
from uie.suffixes import SuffixTable  # noqa: E402
from uie.train import load_checkpoint  # noqa: E402

DELTA = {"benign": 0.22, "phishing": 0.30}          # adopted, as in the gate
DECODER, DECODER_KWARGS = "temperature", {"temperature": 1.0}
SEED = 20260925


def generate_traced(model, tok, *, label, n, table, seed, fields, device):
    """`generate_with_suffixes`, step for step, keeping each URL's trace beside
    it: same suffix draws, same groups and group seeds, same final shuffle."""
    rng = random.Random(seed)
    drawn = table.sample(label, n, rng)
    groups = sorted(collections.Counter(zip(["domain"] * n, drawn)).items())
    d = DELTA[label]
    fb = {"PTH": {tok.eos_id: -d}} if d else None
    urls, traces = [], []
    for i, ((frame, suffix), count) in enumerate(groups):
        t: list = []
        urls.extend(generate(model, tok, label=label, n=count, suffix=suffix,
                             decoder=DECODER, seed=seed + i, frame=frame,
                             field_bias=fb, batch_size=32, device=device,
                             decoder_kwargs=DECODER_KWARGS, fields=fields,
                             enforce="mask", trace=t))
        traces.extend(t)
    perm = list(range(n))
    rng.shuffle(perm)                  # the permutation generate_with_suffixes applies
    return [urls[j] for j in perm], [traces[j] for j in perm]


def tv_with_ci(w, resamples=2000, seed=SEED):
    """TV(q, p*) = 1/2 E_q|1 - w/E_q[w]|, and a percentile bootstrap CI."""
    def tv(ws):
        m = statistics.fmean(ws)
        return 0.5 * statistics.fmean(abs(1 - x / m) for x in ws) if m > 0 else 1.0
    rng = random.Random(seed)
    boot = sorted(tv([w[rng.randrange(len(w))] for _ in w]) for _ in range(resamples))
    return tv(w), [boot[int(0.025 * resamples)], boot[int(0.975 * resamples) - 1]]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--bundle", type=Path, required=True)
    ap.add_argument("--c1", type=Path, required=True, help="C1's c1_report.json")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--n", type=int, default=1200)
    args = ap.parse_args()

    device = torch.device("cpu")
    model, tok, _ = load_checkpoint(args.checkpoint, device=device)
    model.to(device).eval()
    table = SuffixTable.load(args.bundle / "suffix_table.json")
    c1 = {}
    for c in json.loads(args.c1.read_text(encoding="utf-8"))["cells"]:
        if c["kind"] == "single":
            (f, v), = c["per_field"].items()
            c1[(c["label"], f, bool(v["want"]))] = v
    args.out.mkdir(parents=True, exist_ok=True)

    cells = [(label, f, want) for label in tok.classes
             for f in ("PTH", "QRY") for want in (True, False)]
    report = {"test": "HC1 H1 + H2 (research/hc1/PREREG_HC1.md section 4)",
              "checkpoint": str(args.checkpoint),
              "checkpoint_sha256": hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
              "generation": {"path": "generate_with_suffixes, reproduced with traces",
                             "decoder": DECODER, "decoder_kwargs": DECODER_KWARGS,
                             "delta_pth_eos": DELTA, "frame": "domain",
                             "enforce": "mask", "n": args.n, "seed_base": SEED},
              "cells": []}
    started = time.time()
    for i, (label, f, want) in enumerate(cells):
        seed = SEED + 10_000 * i
        tag = f"{label}_{f[0]}{int(want)}"
        t0 = time.time()
        urls, traces = generate_traced(model, tok, label=label, n=args.n, table=table,
                                       seed=seed, fields={f: want}, device=device)
        (args.out / f"{tag}.json").write_text(json.dumps(urls), encoding="utf-8")
        (args.out / f"{tag}.trace.json").write_text(json.dumps(traces), encoding="utf-8")

        honoured = sum(bool(FIELDS[f](u)) == want for u in urls)
        truncated = sum(t["truncated"] for t in traces)
        w = [math.exp(t["log_keep"]) for t in traces]
        mean_w = statistics.fmean(w)
        se_w = statistics.stdev(w) / math.sqrt(len(w))
        ref = c1.get((label, f, want))
        z = None
        if ref:
            p = ref["rate"]
            se_c1 = math.sqrt(p * (1 - p) / ref["n"])
            z = (mean_w - p) / math.sqrt(se_w ** 2 + se_c1 ** 2)
        tv, tv_ci = tv_with_ci(w)
        cell = {"tag": tag, "label": label, "field": f, "want": int(want),
                "n": len(urls), "seed": seed, "honoured": honoured,
                "truncated": truncated,
                "unsatisfied_by_trace": sum(not t["satisfied"] for t in traces),
                "mean_w": round(mean_w, 5), "se_w": round(se_w, 5),
                "c1_rate": ref["rate"] if ref else None,
                "c1_n": ref["n"] if ref else None,
                "z": round(z, 2) if z is not None else None,
                "tv": round(tv, 4), "tv_ci95": [round(x, 4) for x in tv_ci],
                "tv_predicted": round(1 - ref["rate"], 4) if ref else None,
                "share_w_below_half": round(sum(x < 0.5 for x in w) / len(w), 4),
                "mean_masked_steps": round(statistics.fmean(t["masked_steps"]
                                                            for t in traces), 2),
                "seconds": round(time.time() - t0, 1)}
        report["cells"].append(cell)
        done = report["cells"]
        report["H1"] = ("PASS" if len(done) == 8 and all(
            c["honoured"] == c["n"] and c["truncated"] == 0 for c in done)
            else "FAIL" if any(c["honoured"] < c["n"] or c["truncated"] for c in done)
            else "INCOMPLETE")
        report["H2"] = ("PASS" if len(done) == 8 and all(
            c["z"] is not None and abs(c["z"]) < 3 for c in done)
            else "FAIL" if any(c["z"] is not None and abs(c["z"]) >= 3 for c in done)
            else "INCOMPLETE")
        report["seconds"] = round(time.time() - started, 1)
        (args.out / "h1_h2_report.json").write_text(json.dumps(report, indent=1),
                                                    encoding="utf-8")
        print(f"  {tag:<12} honoured {honoured}/{len(urls)}  truncated {truncated}  "
              f"mean w {mean_w:.4f} (se {se_w:.4f}) vs C1 "
              f"{'%.4f' % ref['rate'] if ref else '--'}  z {cell['z']}  "
              f"TV {tv:.4f} [{tv_ci[0]:.4f}, {tv_ci[1]:.4f}]  "
              f"({time.time() - t0:.0f}s)", flush=True)
    print(f"\nH1 {report['H1']}   H2 {report['H2']}")


if __name__ == "__main__":
    main()
