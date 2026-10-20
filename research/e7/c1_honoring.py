"""C1 -- controllability, the PRIMARY test registered in PREREG_E7.md section 4.

For each class and each control value (P:1, P:0, Q:1, Q:0, the other control
`?`), generate n = 4,000 through the canonical gate's OWN generation path --
`generate_with_suffixes`: suffix drawn from the class's marginal table, frame
"domain", temperature 1.0, the adopted delta on PTH EOS -- and measure the
field's presence with the gate's own definition (`uie.evaluation.FIELDS`).

    PASS  honouring >= 0.98 in all 8 cells (point estimates, as registered)

The four joint settings (P:1/Q:1, P:1/Q:0, P:0/Q:1, P:0/Q:0) are reported per
class at n = 1,000, descriptively. Wilson 95% intervals are printed beside every
rate; they inform, they do not change the rule. Every generated URL is saved,
and the result file is rewritten after each cell so a partial run survives.

    python research/e7/c1_honoring.py --checkpoint <best_e7.pt> --bundle <bundle2> --out <dir>
"""
import argparse
import hashlib
import json
import math
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import torch  # noqa: E402

import uie  # noqa: E402
from uie.evaluation import FIELDS  # noqa: E402
from uie.suffixes import SuffixTable, generate_with_suffixes  # noqa: E402
from uie.train import load_checkpoint  # noqa: E402

BAR = 0.98
DELTA = {"benign": 0.22, "phishing": 0.30}     # adopted, as in the gate
SEED = 20260924                                # C1's own block; recorded per cell


def wilson(k: int, n: int, z: float = 1.959964):
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return [round(c - h, 4), round(c + h, 4)]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--bundle", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--n", type=int, default=4000)
    ap.add_argument("--n-joint", type=int, default=1000)
    args = ap.parse_args()

    # Two copies of `uie` live on this machine; only the E7 one has controls.
    if not Path(uie.__file__).resolve().is_relative_to(ROOT):
        raise SystemExit(f"uie imported from {uie.__file__}, not {ROOT}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, tokenizer, _ = load_checkpoint(args.checkpoint, device=device)
    if not getattr(tokenizer, "include_fields", False):
        raise SystemExit("not an E7 checkpoint: its tokenizer has no presence controls")
    table = SuffixTable.load(args.bundle / "suffix_table.json")
    args.out.mkdir(parents=True, exist_ok=True)

    cells = [("single", label, {field: want}, args.n)
             for label in tokenizer.classes
             for field in ("PTH", "QRY") for want in (True, False)]
    cells += [("joint", label, {"PTH": p, "QRY": q}, args.n_joint)
              for label in tokenizer.classes
              for p in (True, False) for q in (True, False)]

    report = {"test": "C1 controllability (PREREG_E7.md section 4)",
              "checkpoint": str(args.checkpoint),
              "checkpoint_sha256": hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
              "generation": {"path": "uie.suffixes.generate_with_suffixes",
                             "decoder": "temperature", "temperature": 1.0,
                             "frame": "domain", "delta_pth_eos": DELTA,
                             "device": str(device)},
              "bar": BAR, "cells": []}
    started = time.time()
    for i, (kind, label, fields, n) in enumerate(cells):
        seed = SEED + 10_000 * i
        tag = label + "_" + "_".join(f"{f[0]}{int(v)}" for f, v in fields.items())
        d = DELTA[label]
        t0 = time.time()
        urls = generate_with_suffixes(
            model, tokenizer, label=label, n=n, table=table, decoder="temperature",
            seed=seed, device=device, field_bias={"PTH": {tokenizer.eos_id: -d}},
            decoder_kwargs={"temperature": 1.0}, fields=fields)
        (args.out / f"{tag}.json").write_text(json.dumps(urls), encoding="utf-8")

        per_field = {}
        for f, want in fields.items():
            k = sum(bool(FIELDS[f](u)) == want for u in urls)
            per_field[f] = {"want": int(want), "honoured": k, "n": len(urls),
                            "rate": round(k / len(urls), 4), "ci95": wilson(k, len(urls))}
        both = sum(all(bool(FIELDS[f](u)) == w for f, w in fields.items()) for u in urls)
        report["cells"].append({"kind": kind, "label": label, "tag": tag,
                                "fields": {f: int(v) for f, v in fields.items()},
                                "n": len(urls), "seed": seed, "delta": d,
                                "per_field": per_field,
                                "all_honoured": round(both / len(urls), 4),
                                "seconds": round(time.time() - t0, 1)})
        rates = "  ".join(f"{f}:{v['want']} {100 * v['rate']:5.1f}% "
                          f"[{100 * v['ci95'][0]:.1f}, {100 * v['ci95'][1]:.1f}]"
                          for f, v in per_field.items())
        print(f"  {kind:<6} {label:<9} {rates}   ({time.time() - t0:.0f}s)", flush=True)

        singles = [c for c in report["cells"] if c["kind"] == "single"]
        failing = [c["tag"] for c in singles
                   if next(iter(c["per_field"].values()))["rate"] < BAR]
        report["verdict"] = ("PASS" if len(singles) == 8 and not failing else
                             "FAIL" if failing else "INCOMPLETE")
        report["failing_cells"] = failing
        report["seconds"] = round(time.time() - started, 1)
        (args.out / "c1_report.json").write_text(json.dumps(report, indent=1),
                                                 encoding="utf-8")

    print(f"\nC1 verdict: {report['verdict']}"
          + (f"  failing: {report['failing_cells']}" if report["failing_cells"] else ""))


if __name__ == "__main__":
    main()
