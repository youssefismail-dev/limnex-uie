"""Gate v2 verification V3 (PREREG_GATE_V2.md section 6).

Applies the IN-GATE section-26.1 function (`s261_verdict`, SE = sd/sqrt(G)) to
the stored per-draw excesses of the three published G=5 x D=3 runs, and checks
every per-generation cell against the README: the value x, and the verdict.
Draws are stored generation-major (the gate appends g-outer, d-inner), so each
consecutive triple is one generation. Pooled cells (convict-only) are listed
with the pooled rule, not the t-test. Read-only.

    python research/qry/v3_verdict_check.py
"""
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from uie.corpus.run_eval import pooled_verdict, s261_verdict  # noqa: E402

RUNS = {"typical d=0": "data/e5_gate_n8000_d0",
        "typical d=adopted": "data/e5_gate_n8000_dadopted",
        "temperature d=adopted (R4)": "data/r4_temp_dadopted"}
MARK = {"✅": "CLEARED", "❌": "CONVICTED", "⚠️": "UNDECIDED"}
# README, as published: (x, mark). R4 benign QRY carries no gate verdict there.
README = {
    "typical d=0": {
        "benign": {"LBL": (0.0301, "✅"), "SUF": (0.0029, "✅"), "SUB": (0.0391, "✅"),
                   "PTH": (0.0767, "❌"), "QRY": (0.1196, "❌")},
        "phishing": {"LBL": (0.0187, "✅"), "SUF": (0.0003, "✅"), "SUB": (0.0576, "❌"),
                     "PTH": (0.0787, "❌"), "QRY": (0.1485, "❌")}},
    "typical d=adopted": {
        "benign": {"LBL": (0.0260, "✅"), "SUF": (0.0035, "✅"), "SUB": (0.0400, "✅"),
                   "PTH": (0.0728, "❌"), "QRY": (0.1233, "❌")},
        "phishing": {"LBL": (0.0265, "✅"), "SUF": (-0.0032, "✅"), "SUB": (0.0651, "⚠️"),
                     "PTH": (0.0834, "❌"), "QRY": (0.1488, "❌")}},
    "temperature d=adopted (R4)": {
        "benign": {"LBL": (-0.0014, "✅"), "SUF": (0.0001, "✅"), "SUB": (0.0048, "✅"),
                   "PTH": (0.0384, "✅"), "QRY": (0.0300, None)},
        "phishing": {"LBL": (0.0144, "✅"), "SUF": (-0.0064, "✅"), "SUB": (0.0332, "✅"),
                     "PTH": (0.0597, "❌"), "QRY": (0.1431, "❌")}},
}
G, D = 5, 3

bad = 0
checked = 0
for run, rel in RUNS.items():
    rep = json.loads((ROOT / rel / "eval_report.json").read_text(encoding="utf-8"))
    print(f"\n=== {run}  ({rel})")
    print(f"  {'cell':<14}{'x':>9}{'published':>11}{'SE/rtG':>9}{'t/rtG':>8}"
          f"{'t/rt(G-1)':>11}  verdict      README")
    for cls in ("benign", "phishing"):
        fields = rep[cls]["fields"]
        pooled = rep[cls].get("pooled_fields") or {}
        for f in ("LBL", "SUF", "SUB", "PTH", "QRY"):
            px, pmark = README[run][cls][f]
            row = fields[f]
            if row.get("draws") == G * D == row.get("draws_attempted"):
                ex = row["excess_draws"]
                acc = s261_verdict([ex[g * D:(g + 1) * D] for g in range(G)])
                t_old = acc["t_vs_bar"] / math.sqrt(G / (G - 1))
                want = MARK.get(pmark)
                ok_x = abs(acc["x"] - px) <= 0.00015   # stored draws are 4-dp rounded
                ok_v = want is None or acc["verdict"] == want
                checked += 1
                bad += (not ok_x) + (not ok_v)
                print(f"  {cls[:5] + ' ' + f:<14}{acc['x']:>+9.4f}{px:>+11.4f}"
                      f"{acc['se']:>9.4f}{acc['t_vs_bar']:>8.2f}{t_old:>11.2f}  "
                      f"{acc['verdict']:<12} {want or '(no gate verdict published)'}"
                      f"{'' if ok_x and ok_v else '   <-- MISMATCH'}")
            else:
                p = pooled.get(f)
                ex = (p or {}).get("excess_draws") or []
                pv = pooled_verdict(ex) if ex else "n/a"
                print(f"  {cls[:5] + ' ' + f:<14}{(p or {}).get('excess', float('nan')):>+9.4f}"
                      f"{px:>+11.4f}{'':>28}  pooled: {pv:<9} {MARK.get(pmark)} "
                      f"(convict-only)")

print(f"\nV3: {checked} per-generation cells checked, {bad} mismatches")
sys.exit(1 if bad else 0)
