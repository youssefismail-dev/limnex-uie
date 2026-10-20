"""R-a' (research/native/DECISION_2026-10-05.md section 2.3), applied post-hoc at the
owner's direction to the decoder arms already read: E14's D1' (e12_d1p against
e12_final), E14's combination (e14_qd against e14_q), and E13's D2 (e12_dec against
e12_d1), for the record. Reads the cell contrasts each compare_report.json already
holds; computes nothing new. Writes data/e14/rerule_report.json.

    python research/e14/e14_rerule.py

R-a' : REGRESSION if (i) delta > 0 at Holm p < 0.05 over the mode's cells, or
       (ii) the verdict worsens and the cell's own 95% interval lies wholly above 0.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
for p in (ROOT / "research" / "sc1", ROOT / "research" / "native"):
    sys.path.insert(0, str(p))

from e8_compare import RANK  # noqa: E402
from sc1_compare import holm  # noqa: E402

ALPHA = 0.05


def r_a_prime(cells: dict) -> dict:
    regs, flips = [], []
    for mode, cs in cells.items():
        adj = holm({k: c["p"] for k, c in cs.items() if c})
        for k, c in cs.items():
            if not c:
                continue
            worse_v = (c["v_ref"] in RANK and c["v_arm"] in RANK
                       and RANK[c["v_arm"]] > RANK[c["v_ref"]])
            row = {"mode": mode, "cell": k, "delta": c["delta"], "lo": c["lo"], "hi": c["hi"],
                   "p": c["p"], "p_holm": adj[k], "v_ref": c["v_ref"], "v_arm": c["v_arm"]}
            clause_i = c["delta"] > 0 and adj[k] < ALPHA
            clause_ii = worse_v and c["delta"] > 0 and c["lo"] > 0
            if clause_i or clause_ii:
                regs.append({**row, "clause": "i" if clause_i else "ii"})
            elif worse_v:
                flips.append(row)            # recorded, not counted (section 2.3)
    return {"pass": not regs, "regressions": regs, "verdict_changes_not_counted": flips}


def main() -> None:
    e14 = json.loads((ROOT / "data" / "e14" / "compare_report.json").read_text(encoding="utf-8"))
    e13 = json.loads((ROOT / "data" / "e13" / "compare_report.json").read_text(encoding="utf-8"))
    out = {"rule": "R-a' (research/native/DECISION_2026-10-05.md section 2.3)",
           "status": "post-hoc: written after these data were read; applied at the owner's direction"}
    for name, node, others in (("E14 D1' (e12_d1p vs e12_final)", e14["D1'"], ("K14-4", "R-b", "R-c", "R-d", "R-e", "R-f")),
                               ("E14 combination (e14_qd vs e14_q)", e14["combination"], ("K14-4", "R-b", "R-c", "R-d", "R-e", "R-f")),
                               ("E13 D2 (e12_dec vs e12_d1)", e13["D2"], ("K13-4", "R-b", "R-c", "R-d", "R-e", "R-f"))):
        ra = r_a_prime(node["cells"])
        rest = {k: node[k]["pass"] for k in others}
        out[name] = {"R-a (as registered)": node["R-a"]["pass"], "R-a'": ra,
                     "other criteria (unchanged)": rest,
                     "all criteria pass under R-a'": ra["pass"] and all(rest.values())}
        print(f"{name}: R-a {'PASS' if node['R-a']['pass'] else 'FAIL'} -> R-a' "
              f"{'PASS' if ra['pass'] else 'FAIL'}; other criteria "
              + " ".join(f"{k} {'PASS' if v else 'FAIL'}" for k, v in rest.items())
              + f"; verdict changes recorded, not counted: "
              + "; ".join(f"{f['cell']} {f['v_ref']}->{f['v_arm']} delta {f['delta']:+.4f} "
                          f"[{f['lo']:+.4f}, {f['hi']:+.4f}]" for f in ra["verdict_changes_not_counted"]))
    dest = ROOT / "data" / "e14" / "rerule_report.json"
    dest.write_text(json.dumps(out, indent=1, default=float), encoding="utf-8")
    print("wrote", dest)


if __name__ == "__main__":
    main()
