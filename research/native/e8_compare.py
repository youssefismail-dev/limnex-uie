"""E8 (N1) -- the registered criteria, the attribution and the candidate rule
(PREREG_E8 section 6). Reads data/e8/score_report.json and VERIFY.json; writes
data/e8/compare_report.json.

    python research/native/e8_compare.py

Every number below is the registration's. Honouring is a separate criterion
from realism and is never read as evidence about it.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Dict, Optional

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "dc1"))

import e8_common as K  # noqa: E402

RETRAIN = 0.0141                       # E7 C2's attributable-regression allowance
HONOUR_BAR = 0.98                      # D2 (C1's bar)
Q_BAND = {"benign": (0.0596, 0.0728), "phishing": (0.0421, 0.0515)}   # 6.62% / 4.68% +-10%
P_BAND = {"benign": (0.7037, 0.7637), "phishing": (0.3651, 0.4251)}   # 73.37% / 39.51% +-3 pts
SLASH_TARGET = {"benign": {"bare": 0.979, "query_no_path": 0.996},
                "phishing": {"bare": 0.999, "query_no_path": 1.0}}
SLASH_TOL = 0.02
VBASE_HONOUR, VBASE_EMISSION = 0.01, 0.005
GATED = ("LBL", "SUF", "SUB", "PTH", "QRY")
RANK = {"CLEARED": 0, "UNDECIDED": 1, "CONVICTED": 2}
PLAN_ARMS = tuple(a for a, f in K.ARMS.items() if f["presence_plan"])
SLASH_ARMS = tuple(a for a, f in K.ARMS.items() if f["root_slash"])


def _acc(row):
    a = (row or {}).get("acceptance", {})
    x = a.get("x")
    return (None if x is None or x != x else float(x)), a.get("verdict")


def free_cells(rep: dict, who: str) -> Dict[tuple, tuple]:
    src = rep["e7_free"] if who == "e7" else rep.get("free", {}).get(who, {})
    out = {}
    for label, part in src.items():
        for f in GATED:
            x, v = _acc(part["fields"].get(f))
            if x is not None and v in RANK:
                out[(label, f)] = (x, v)
    return out


def control_cells(rep: dict, who: str) -> Dict[tuple, tuple]:
    out = {}
    for key, part in rep.get("control", {}).get(who, {}).items():
        for f in GATED:
            x, v = _acc(part["fields"].get(f))
            if x is not None and v in RANK:
                out[(part["label"], part["state"], f)] = (x, v)
    return out


def regressions(base: dict, arm: dict) -> list:
    """K1d: the verdict worsens AND x rises by more than the retraining allowance."""
    out = []
    for key in sorted(set(base) & set(arm)):
        (xb, vb), (xa, va) = base[key], arm[key]
        if RANK[va] > RANK[vb] and xa - xb > RETRAIN:
            out.append({"cell": list(key), "base": [xb, vb], "arm": [xa, va]})
    return out


def rate(rep: dict, block: str, who: str, *path) -> Optional[float]:
    node = rep.get(block, {}).get(who)
    for p in path:
        if node is None:
            return None
        node = node.get(p)
    return None if node is None else node.get("rate")


def v_base(rep: dict) -> dict:
    fails = []
    b, e = free_cells(rep, "e8_base"), free_cells(rep, "e7")
    for key in sorted(set(b) & set(e)):
        if b[key][1] != e[key][1] and abs(b[key][0] - e[key][0]) > RETRAIN:
            fails.append(f"free {key}: E7 {e[key]} vs e8_base {b[key]}")
    for label in K.LABELS:
        for st in K.STATES:
            rb, re = (rate(rep, "honouring", w, f"{label}_{st}") for w in ("e8_base", "e7"))
            if rb is None or re is None or abs(rb - re) > VBASE_HONOUR:
                fails.append(f"honouring {label} {st}: E7 {re} vs e8_base {rb}")
        for f in ("PTH", "QRY"):
            rb, re = (rate(rep, "emission", w, label, f) for w in ("e8_base", "e7"))
            if rb is None or re is None or abs(rb - re) > VBASE_EMISSION:
                fails.append(f"emission {label} {f}: E7 {re} vs e8_base {rb}")
    return {"pass": not fails, "failures": fails}


def k1a(rep, arm):
    fails = [f"{label} {st}: {rate(rep, 'honouring', arm, f'{label}_{st}')}"
             for label in K.LABELS for st in K.STATES
             if (rate(rep, "honouring", arm, f"{label}_{st}") or 0.0) < HONOUR_BAR]
    return {"pass": not fails, "failures": fails}


def k1b(rep, arm):
    fails = []
    for label in K.LABELS:
        for f, band in (("QRY", Q_BAND[label]), ("PTH", P_BAND[label])):
            r = rate(rep, "emission", arm, label, f)
            if r is None or not band[0] <= r <= band[1]:
                fails.append(f"{label} {f}: {r} outside {band}")
    return {"pass": not fails, "failures": fails}


def k1c(rep, arm):
    fails = []
    for label in K.LABELS:
        for kind, target in SLASH_TARGET[label].items():
            r = rate(rep, "root_slash", arm, label, kind)
            if r is None or abs(r - target) > SLASH_TOL:
                fails.append(f"{label} {kind}: {r} vs {target}")
    return {"pass": not fails, "failures": fails}


def k1d(rep, arm):
    regs = (regressions(free_cells(rep, "e8_base"), free_cells(rep, arm))
            + regressions(control_cells(rep, "e8_base"), control_cells(rep, arm)))
    return {"pass": not regs, "regressions": regs}


def k1e(rep, arm):
    from dc1_compare import prop_worsening
    out, fails = {}, []
    for mode in ("free", "control"):
        b, a = rep["novelty"]["e8_base"][mode], rep["novelty"][arm][mode]
        out[f"{mode}_novelty"] = prop_worsening(b["novel"], b["n"], a["novel"], a["n"], False)
        out[f"{mode}_copying"] = prop_worsening(b["copied"], max(1, b["q"]), a["copied"],
                                                max(1, a["q"]), True)
    fails = [k for k, v in out.items() if v["worsened"]]
    return {"pass": not fails, "failures": fails, "readings": out}


def attribution(rep: dict) -> dict:
    """Plan = e8_pf - e8_base; slash = e8_rs - e8_base and e8_pfrs - e8_pf;
    interaction = their difference. Credited only beyond |e8_base - E7|."""
    def metrics(who):
        m = {}
        for label in K.LABELS:
            for st in K.STATES:
                m[f"honour {label} {st}"] = rate(rep, "honouring", who, f"{label}_{st}")
            for f in ("PTH", "QRY"):
                m[f"emission {label} {f}"] = rate(rep, "emission", who, label, f)
            for kind in ("bare", "query_no_path"):
                m[f"root slash {label} {kind}"] = rate(rep, "root_slash", who, label, kind)
        for key, (x, _) in free_cells(rep, who).items():
            m[f"free {' '.join(key)}"] = x
        for key, (x, _) in control_cells(rep, who).items():
            m[f"control {' '.join(key)}"] = x
        return m

    M = {w: metrics(w) for w in ("e7",) + tuple(K.ARMS)}
    out = {}
    for name in sorted(set().union(*(set(v) for v in M.values()))):
        g = {w: M[w].get(name) for w in M}
        if None in (g["e8_base"], g["e8_pf"], g["e8_rs"], g["e8_pfrs"]):
            continue
        noise = abs(g["e8_base"] - g["e7"]) if g["e7"] is not None else None
        plan = g["e8_pf"] - g["e8_base"]
        s1, s2 = g["e8_rs"] - g["e8_base"], g["e8_pfrs"] - g["e8_pf"]

        def credited(d):
            return None if noise is None else abs(d) > noise

        out[name] = {"values": g, "noise": noise,
                     "plan": plan, "plan_credited": credited(plan),
                     "slash_on_e7": s1, "slash_on_e7_credited": credited(s1),
                     "slash_on_plan": s2, "slash_on_plan_credited": credited(s2),
                     "interaction": s2 - s1}
    return out


def candidate(res: dict, vb: dict) -> str:
    """The registered rule, read literally on the candidate (e8_pfrs). Every
    other arm's failures are reported with the outcome, never dropped."""
    if not vb["pass"]:
        return "V-base FAILED: e8_base does not reproduce E7; no arm is read (investigate)"
    keys = ("K1a", "K1b", "K1c", "K1d", "K1e")
    fails = {arm: [k for k in keys if not r[k]["pass"]] for arm, r in res.items()}
    others = "; ".join(f"{a} fails {', '.join(f)}" for a, f in fails.items()
                       if f and a != "e8_pfrs")
    note = f" (reported: {others})" if others else ""
    p = fails["e8_pfrs"]
    if not p:
        return "e8_pfrs is the candidate (passes K1a-K1e); the owner decides adoption" + note
    if p == ["K1a"]:
        return "only K1a fails: N1b (unlikelihood) is licensed, and nothing else" + note
    out = [f"e8_pfrs fails {', '.join(p)}"]
    if any(k in p for k in ("K1b", "K1d", "K1e")):
        alone = not any(k in fails["e8_rs"] for k in ("K1c", "K1d", "K1e"))
        out.append("the plan is not taken forward; "
                   + ("e8_rs stands alone (passes K1c-K1e)" if alone
                      else "e8_rs does not pass K1c-K1e either"))
    if "K1c" in p or "K1c" in fails["e8_rs"]:
        out.append("K1c fails: the slash token is not taken forward")
    return "; ".join(out) + note


def main() -> None:
    e8 = ROOT / "data" / "e8"
    rep = json.loads((e8 / "score_report.json").read_text(encoding="utf-8"))
    verify = json.loads((e8 / "VERIFY.json").read_text(encoding="utf-8"))
    vb = v_base(rep)
    res = {}
    for arm in K.ARMS:
        res[arm] = {"K1d": k1d(rep, arm), "K1e": k1e(rep, arm)}
        res[arm]["K1a"] = k1a(rep, arm) if arm in PLAN_ARMS else {"pass": True, "applies": False}
        res[arm]["K1b"] = k1b(rep, arm) if arm in PLAN_ARMS else {"pass": True, "applies": False}
        res[arm]["K1c"] = k1c(rep, arm) if arm in SLASH_ARMS else {"pass": True, "applies": False}
    out = {"registration": "research/native/PREREG_E8.md", "verify": verify["verdict"],
           "v_base": vb, "arms": res, "attribution": attribution(rep),
           "outcome": candidate(res, vb) if verify["verdict"] == "PASS"
           else "VERIFY failed: nothing is read"}
    (e8 / "compare_report.json").write_text(json.dumps(out, indent=1, default=float),
                                            encoding="utf-8")
    print("V-base:", "PASS" if vb["pass"] else vb["failures"])
    for arm, r in res.items():
        print(f"  {arm:<8} " + "  ".join(f"{k} {'ok' if v['pass'] else 'FAIL'}"
                                         + ("" if v.get("applies", True) else " (n/a)")
                                         for k, v in sorted(r.items())))
    print("OUTCOME:", out["outcome"])


if __name__ == "__main__":
    main()
