"""E11 -- predictions from E10's readout (research/native/PREREG_E11.md section 3.1).
READ-ONLY; runs only after E10's VERIFY passed.

    python research/native/e11_predict.py

The supplied plan (D6) gives each free URL a state k drawn with the data's share
pi_k. What e11_plan then emits is, per field, sum_k pi_k * P(field present | k
requested) -- the plan arm's own behaviour when told the state. E10 measured that
behaviour: e10_plan's control files at every state (the plan supplied, SC1's state
tables). Their per-field rates, weighted by pi_k, predict e11_plan's free-mode path
and query rates, to set against the tolerance. Assumes e11_plan honours as e10_plan
did; the slash and the repair may change it, and the suffix mix within a state
differs (SC1's tables against the shipped table times the state's rate).
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE))

from e8_common import G, LABELS, STATES  # noqa: E402
from uie.evaluation import FIELDS  # noqa: E402

E10 = ROOT / "data" / "e10"


def main() -> None:
    v = json.loads((E10 / "VERIFY.json").read_text(encoding="utf-8"))
    if v.get("verdict") != "PASS":
        raise SystemExit(f"STOP: E10's verification has not passed: {v.get('failures')}")
    targets = json.loads((ROOT / "data" / "e11" / "targets.json").read_text(encoding="utf-8"))
    out = {}
    for label in LABELS:
        share = targets[label]["state_share_free"]
        rates = {}
        for arm in ("e10_base", "e10_plan"):
            for st in STATES:
                urls = [u for g in range(G) for u in json.loads(
                    (E10 / "gen" / f"ctrl_{arm}_{label}_{st}_g{g}.json").read_text(encoding="utf-8"))]
                p = sum(bool(FIELDS["PTH"](u)) for u in urls) / len(urls)
                q = sum(bool(FIELDS["QRY"](u)) for u in urls) / len(urls)
                rates.setdefault(arm, {})[st] = {"n": len(urls), "path": p, "query": q}
        pred = {}
        for arm in rates:
            for f, key in (("PTH", "path"), ("QRY", "query")):
                pred.setdefault(arm, {})[f] = sum(share[k] * rates[arm][k][key] for k in STATES)
        real = {"PTH": targets[label]["path_rate_free"], "QRY": targets[label]["query_rate_free"]}
        tol = {f: 2 * math.sqrt(p * (1 - p) / 8000) for f, p in real.items()}
        out[label] = {"state_share": share, "rates_when_told": rates, "predicted_free": pred,
                      "real": real, "T": tol}
        print(f"== {label}")
        for arm in rates:
            print(f"  {arm} told the state: " + "; ".join(
                f"{st} path {100 * r['path']:.2f}% query {100 * r['query']:.2f}%"
                for st, r in rates[arm].items()))
            print(f"    predicted free emission under the supplied plan: " + ", ".join(
                f"{f} {100 * pred[arm][f]:.2f}% (real {100 * real[f]:.2f}, "
                f"diff {100 * (pred[arm][f] - real[f]):+.2f}, T +-{100 * tol[f]:.2f})"
                for f in ("PTH", "QRY")))
    (ROOT / "data" / "e11" / "predict_from_e10.json").write_text(json.dumps(out, indent=1),
                                                                 encoding="utf-8")
    print("wrote data/e11/predict_from_e10.json")


if __name__ == "__main__":
    main()
