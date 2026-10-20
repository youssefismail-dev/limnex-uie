"""N2a -- verification, then the registered decision (PREREG_N2A: Amendment 2's
table with Amendment 4's clarifications). Reads files only. DIAGNOSTIC.

    python research/native/n2a_decide.py verify     # V and K, before anything is read
    python research/native/n2a_decide.py decide     # after the readings below

Layout: the Kaggle output extracted to data/n2a/{A1,A2}/ (best.pt, final.pt,
N2A_RECORD.json, gen_<arm>_<label>_g<g>.json). The readings go beside them:
    n2a_qry.py --label L --files data/n2a/A/gen_A_L_g*.json --out data/n2a/A/n2a_qry_L.json
    n2a_reference.py --label L --checkpoint data/n2a/A/final.pt \
        --queries data/n2a/A/gen_A_L_g*.json --out data/n2a/A/n2a_reference_L.json
E7's references were recorded before any arm: data/native/n2a_qry_e7_<label>.json
(Q-VAL, Q-TRAIN) and data/native/n2a_reference_<label>.json (M4).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
for p in (str(ROOT), str(HERE), str(HERE.parent / "dc1")):
    if p not in sys.path:
        sys.path.insert(0, p)

N2A = ROOT / "data" / "n2a"
NATIVE = ROOT / "data" / "native"
ARMS = ("A1", "A2")
LABELS = ("benign", "phishing")
T_CRIT = 2.776                          # df 4 (G = 5): the power rule's half-width
CLOSES, LITTLE = 0.5, 0.25
ORDER = ("ALLOCATION", "EXPOSURE", "SPARSITY", "CAPACITY-OR-REPRESENTATION")
#: Amendment 4: fixed from the counts before any arm.
EXPECTED_SCHEDULE = {"A1": {"steps": 1_643, "warmup": 41, "eval_every": 82},
                     "A2": {"steps": 6_572, "warmup": 164, "eval_every": 329}}


# ---- the registered quantities --------------------------------------------------------

def half_width(cell: dict) -> float:
    return T_CRIT * cell["se"]


def usable(e7: dict) -> bool:
    """The power rule: x(E7) at least twice the minimum detectable change,
    which is 2 x E7's 95% half-width."""
    return e7["x"] >= 2 * (2 * half_width(e7))


def closure(arm: dict, e7: dict) -> float:
    return 1.0 - arm["x"] / e7["x"]


def closes(arm: dict, e7: dict) -> bool:
    """Closure >= 0.5 AND E7's x lies outside the arm's 95% CI (x +- 2.776 SE)."""
    return closure(arm, e7) >= CLOSES and abs(arm["x"] - e7["x"]) > half_width(arm)


def little(arm: dict, e7: dict) -> bool:
    return closure(arm, e7) < LITTLE


def copying(arm_m4: dict, e7_m4: dict) -> dict:
    """DC1's prop_worsening: a rise of at least 0.5 points whose normal 95% CI
    excludes 0 (Amendment 4)."""
    from dc1_compare import prop_worsening
    return prop_worsening(e7_m4["copied"], e7_m4["queries"], arm_m4["copied"],
                          arm_m4["queries"], worse_if_higher=True)


def reading(e7: dict, arms: Dict[str, dict]) -> dict:
    """One class's reading. e7 and arms[a] hold 'q_val' and 'q_train' (the
    acceptance dicts: x, se) and 'm4' (queries, copied).

    Rows are read in order and the first that holds is the reading; every row
    that holds is reported (Amendment 4). A guarded arm (copying rose) has its
    validation closure voided: it can count only as evidence for SPARSITY.
    SPARSITY and CAPACITY need a usable Q-TRAIN reference; where it is void
    (phishing, recorded before any arm) both rows are unavailable, and if no
    available row holds the reading is UNAVAILABLE, not MIXED."""
    avail = {k: usable(e7[k]) for k in ("q_val", "q_train")}
    cop = {a: copying(arms[a]["m4"], e7["m4"]) for a in ARMS}
    rose = {a: cop[a]["worsened"] for a in ARMS}
    clos = {a: {k: closure(arms[a][k], e7[k]) for k in ("q_val", "q_train")} for a in ARMS}

    def val_closes(a):
        return avail["q_val"] and not rose[a] and closes(arms[a]["q_val"], e7["q_val"])

    a2 = arms["A2"]
    sparse_fit = (avail["q_train"] and avail["q_val"]
                  and closes(a2["q_train"], e7["q_train"]) and little(a2["q_val"], e7["q_val"]))
    fit_rows = avail["q_train"]
    holds = {
        "ALLOCATION": val_closes("A1"),
        "EXPOSURE": val_closes("A2"),
        "SPARSITY": fit_rows and (any(rose.values()) or sparse_fit),
        "CAPACITY-OR-REPRESENTATION": (fit_rows and little(a2["q_train"], e7["q_train"])
                                       and not any(rose.values())),
    }
    first = next((r for r in ORDER if holds[r]),
                 "MIXED" if all(avail.values()) else "UNAVAILABLE")
    return {"reading": first, "holds": [r for r in ORDER if holds[r]], "usable": avail,
            "closure": clos, "copying": cop, "guarded": rose,
            "closes": {a: {k: (avail[k] and closes(arms[a][k], e7[k])) for k in ("q_val", "q_train")}
                       for a in ARMS},
            "little": {a: {k: (avail[k] and little(arms[a][k], e7[k])) for k in ("q_val", "q_train")}
                       for a in ARMS}}


def decision(by_label: Dict[str, dict]) -> dict:
    """Benign decides; phishing is reported, and only agreement extends the
    reading to phishing (Amendment 4)."""
    b, p = by_label["benign"]["reading"], by_label["phishing"]["reading"]
    agreement = ("UNAVAILABLE" if p == "UNAVAILABLE" else
                 "AGREES" if p == b else "DISAGREES")
    licenses = {"ALLOCATION": "N2b: allocation inside the mixture",
                "EXPOSURE": "N2b: exposure (a query-content loss weight)",
                "SPARSITY": "no training fix inside the frozen corpus; the owner decides",
                "CAPACITY-OR-REPRESENTATION": "N2c (capacity or representation; N2a cannot "
                                              "tell which)",
                "MIXED": "no content repair licensed; reported"}
    return {"reading": b, "licenses": licenses.get(b, "nothing"),
            "phishing_reading": p, "phishing": agreement,
            "extends_to_phishing": agreement == "AGREES"}


# ---- verification (V, K) --------------------------------------------------------------

def verify(n2a: Path = N2A, e7_path: Path = ROOT / "data" / "best_e7.pt") -> dict:
    import torch
    import n2a_train as T
    problems: List[str] = []
    _, _, e7_train, e7_tok = T.load_e7(e7_path)
    for arm in ARMS:
        d = n2a / arm
        if not (d / "N2A_RECORD.json").is_file():
            problems.append(f"V {arm}: no N2A_RECORD.json in {d}")
            continue
        rec = json.loads((d / "N2A_RECORD.json").read_text(encoding="utf-8"))
        for name, key in (("final.pt", "final_sha256"), ("best.pt", "best_sha256")):
            if T.sha256(d / name) != rec[key]:
                problems.append(f"V {arm}: {name} differs from its record")
        blob = torch.load(d / "final.pt", map_location="cpu", weights_only=False)
        if blob["tokenizer"] != e7_tok:
            problems.append(f"V/K {arm}: the tokenizer is not E7's")
        tc = blob["train_config"]
        for k, v in e7_train.items():
            want = EXPECTED_SCHEDULE[arm][k] if k in T.CHANGED else v
            if tc.get(k) != want:
                problems.append(f"V {arm}: train_config {k} = {tc.get(k)}, registered {want}")
        if rec["kept_query_rows"] != T.EXPECTED_KEPT:
            problems.append(f"V {arm}: kept query rows {rec['kept_query_rows']}")
        if not rec["finite"]:
            problems.append(f"K {arm}: non-finite validation loss -- the arm is void")
        want_files = {f"gen_{arm}_{label}_g{g}.json": T.gen_seed(label, g)
                      for label in LABELS for g in range(T.G)}
        got = rec["generation"]["files"]
        if set(got) != set(want_files):
            problems.append(f"V {arm}: generated files {sorted(got)}")
        for name, seed in want_files.items():
            meta = got.get(name, {})
            if (meta.get("seed") != seed or meta.get("n") != T.N_GEN
                    or not (d / name).is_file() or T.sha256(d / name) != meta.get("sha256")):
                problems.append(f"V {arm}: {name} differs from its record")
    out = {"ok": not problems, "problems": problems}
    (n2a / "verify.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    return out


# ---- decide --------------------------------------------------------------------------

def load_inputs(n2a: Path = N2A, native: Path = NATIVE) -> Dict[str, dict]:
    out = {}
    for label in LABELS:
        ref = json.loads((native / f"n2a_qry_e7_{label}.json").read_text(encoding="utf-8"))
        m4 = json.loads((native / f"n2a_reference_{label}.json").read_text(encoding="utf-8"))["M4"]
        e7 = {"q_val": ref["e7_q_val"]["acceptance"], "q_train": ref["e7_q_train"]["acceptance"],
              "m4": m4}
        arms = {}
        for arm in ARMS:
            q = json.loads((n2a / arm / f"n2a_qry_{label}.json").read_text(encoding="utf-8"))
            r = json.loads((n2a / arm / f"n2a_reference_{label}.json").read_text(encoding="utf-8"))
            arms[arm] = {"q_val": q["q_val"]["acceptance"], "q_train": q["q_train"]["acceptance"],
                         "m4": r["M4"]}
        out[label] = {"e7": e7, "arms": arms}
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("step", choices=("verify", "decide"))
    args = ap.parse_args()
    if args.step == "verify":
        v = verify()
        print("VERIFIED" if v["ok"] else "PROBLEMS:\n  " + "\n  ".join(v["problems"]))
        return
    v = json.loads((N2A / "verify.json").read_text(encoding="utf-8"))
    if not v["ok"]:
        raise SystemExit("STOP: verification failed; nothing is read (section 6, V/K)")
    inputs = load_inputs()
    by_label = {label: reading(inputs[label]["e7"], inputs[label]["arms"]) for label in LABELS}
    out = {"registration": "research/native/PREREG_N2A.md (Amendments 2 and 4)",
           "status": "DIAGNOSTIC", "by_label": by_label, "decision": decision(by_label),
           "inputs": inputs}
    (N2A / "decision.json").write_text(json.dumps(out, indent=1, default=float),
                                       encoding="utf-8")
    for label in LABELS:
        r = by_label[label]
        print(f"{label}: reading {r['reading']}  (holds: {r['holds'] or 'none'}; usable "
              f"{r['usable']})")
        for arm in ARMS:
            a, e = inputs[label]["arms"][arm], inputs[label]["e7"]
            print(f"  {arm}  Q-VAL x {a['q_val']['x']:+.4f} (E7 {e['q_val']['x']:+.4f}) closure "
                  f"{r['closure'][arm]['q_val']:+.2f}  Q-TRAIN x {a['q_train']['x']:+.4f} (E7 "
                  f"{e['q_train']['x']:+.4f}) closure {r['closure'][arm]['q_train']:+.2f}  copying "
                  f"{r['copying'][arm]['dc1']:.4f} vs {r['copying'][arm]['base']:.4f}"
                  f"{'  GUARDED' if r['guarded'][arm] else ''}")
    d = out["decision"]
    print(f"\nDECISION: {d['reading']} -> {d['licenses']}; phishing {d['phishing_reading']} "
          f"({d['phishing']})")


if __name__ == "__main__":
    main()
