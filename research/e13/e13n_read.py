"""E13 N -- N2a's readings on the current recipe (research/e13/PREREG_E13.md section
2.5). Reads the N sets (the reference from E13_decode, A1 and A2 from E13_train)
after VERIFY_TRAIN and VERIFY_DECODE pass. DIAGNOSTIC.

    python research/e13/e13n_read.py --measure     # Q-VAL, Q-TRAIN and M4 per cell and model
    python research/e13/e13n_read.py               # N2a's table, as amended, and the aggregation

Per cell, every model is measured with N2a's own cell instrument
(n2a_qry.qry_cell: the gate's QRY cell, discriminate_with_floor over 5 generations x
D splits): Q-VAL against the cell's validation pool, Q-TRAIN against the cell's
training strings. M4 is QC1-1's copy rate against the cell's natural rate.
N2a's table (Amendments 2 and 4) with section 2.5's changes:
  - "copying rises" becomes "the copy rate is OVER the natural rate" (r > r_n + T);
  - CAPACITY-OR-REPRESENTATION in both benign cells, with QC1's registered
    no-subwords reading for queries, is R3's strong proof.
Benign decides (both benign cells must agree, else MIXED); free phishing QRY is
reported, and agreement extends the reading to it.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
for p in (ROOT, HERE, ROOT / "research" / "native", ROOT / "research" / "cr1",
          ROOT / "research" / "qc1", ROOT / "research" / "dc1"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import e13_common as E13  # noqa: E402

E13D = ROOT / "data" / "e13"
NDIR = {"e12_final": E13D / "decode" / "gen", "A1": E13D / "train" / "n_A1",
        "A2": E13D / "train" / "n_A2"}
T_CRIT, CLOSES, LITTLE = 2.776, 0.5, 0.25
ORDER = ("ALLOCATION", "EXPOSURE", "SPARSITY", "CAPACITY-OR-REPRESENTATION")
LICENSES = {"ALLOCATION": "N2b: allocation inside the mixture (E14)",
            "EXPOSURE": "N2b: a query-content loss weight (E14)",
            "SPARSITY": "nothing; the owner decides",
            "CAPACITY-OR-REPRESENTATION": "N2c; with QC1's no-subwords reading for queries, "
                                          "R3's strong proof: a capacity proposal goes to the owner",
            "MIXED": "nothing; reported", "UNAVAILABLE": "nothing; reported"}


def gens(model: str, cell: str):
    return [json.loads((NDIR[model] / f"n_{model}_{cell}_g{g}.json").read_text(encoding="utf-8"))
            for g in range(E13.G)]


def measure() -> dict:
    import e13d_common as D
    import qc1_common as C
    from cr1_score import pools
    from n2a_qry import qry_cell
    import e8_common as K8
    bundle = ROOT / "data" / "bundle2"
    out = {}
    for cell, (label, state, _) in E13.N_CELLS.items():
        val = pools(bundle, label, K8.STATES[state] if state else None)["name"]
        train = D._urls(label, "train", state)
        n_main = 3_000 if state else 8_000          # the gate's n_main for the cell's mode
        counts = C.train_counts(label)
        real_q = [C.query(u) for u in C.real_pool(label, state)]
        node = {"natural": C.copy_profile(real_q, counts)}
        for m in ("e12_final", "A1", "A2"):
            g = gens(m, cell)
            gq = [q for f in g for q in (C.query(u) for u in f) if q]
            node[m] = {"q_val": qry_cell(val, g, n_main), "q_train": qry_cell(train, g, n_main),
                       "m4": C.copy_profile(gq, counts), "per_generation": len(gq) / E13.G}
            print(f"  N {cell} {m}: Q-VAL {node[m]['q_val']['acceptance']['x']:+.4f}  Q-TRAIN "
                  f"{node[m]['q_train']['acceptance']['x']:+.4f}  copy {node[m]['m4']['rate']:.4f}",
                  flush=True)
        out[cell] = node
        del counts
    return out


def half_width(a: dict) -> float:
    return T_CRIT * a["se"]


def usable(ref: dict) -> bool:
    return ref["x"] >= 2 * (2 * half_width(ref))


def closure(arm: dict, ref: dict) -> float:
    return 1.0 - arm["x"] / ref["x"]


def closes(arm: dict, ref: dict) -> bool:
    return closure(arm, ref) >= CLOSES and abs(arm["x"] - ref["x"]) > half_width(arm)


def little(arm: dict, ref: dict) -> bool:
    return closure(arm, ref) < LITTLE


def reading(node: dict) -> dict:
    import qc1_common as C
    acc = {m: {k: node[m][k]["acceptance"] for k in ("q_val", "q_train")} for m in ("e12_final", "A1", "A2")}
    ref = acc["e12_final"]
    avail = {k: usable(ref[k]) for k in ("q_val", "q_train")}
    rn = node["natural"]["rate"]
    over = {}
    for a in ("A1", "A2"):
        r, _ = C.rate_reading(node[a]["m4"]["rate"], rn, node[a]["per_generation"])
        over[a] = r == "OVER"

    def val_closes(a):
        return avail["q_val"] and not over[a] and closes(acc[a]["q_val"], ref["q_val"])

    a2 = acc["A2"]
    sparse_fit = (avail["q_train"] and avail["q_val"] and closes(a2["q_train"], ref["q_train"])
                  and little(a2["q_val"], ref["q_val"]))
    fit_rows = avail["q_train"]
    holds = {"ALLOCATION": val_closes("A1"), "EXPOSURE": val_closes("A2"),
             "SPARSITY": fit_rows and (any(over.values()) or sparse_fit),
             "CAPACITY-OR-REPRESENTATION": fit_rows and little(a2["q_train"], ref["q_train"])
                                           and not any(over.values())}
    first = next((r for r in ORDER if holds[r]), "MIXED" if all(avail.values()) else "UNAVAILABLE")
    return {"reading": first, "holds": [r for r in ORDER if holds[r]], "usable": avail,
            "over_natural": over,
            "closure": {a: {k: closure(acc[a][k], ref[k]) for k in ("q_val", "q_train")}
                        for a in ("A1", "A2")}}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--measure", action="store_true")
    args = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")
    for step in ("TRAIN", "DECODE"):
        v = json.loads((E13D / f"VERIFY_{step}.json").read_text(encoding="utf-8"))
        if v.get("verdict") != "PASS":
            raise SystemExit(f"STOP: VERIFY_{step} failed; nothing is read")
    path = E13D / "n_measure.json"
    if args.measure:
        path.write_text(json.dumps(measure(), indent=1, default=float), encoding="utf-8")
        print("wrote", path)
        return
    m = json.loads(path.read_text(encoding="utf-8"))
    by_cell = {cell: reading(node) for cell, node in m.items()}
    b = [by_cell[c]["reading"] for c in E13.N_CELLS if E13.N_CELLS[c][0] == "benign"]
    benign = b[0] if len(set(b)) == 1 else "MIXED"
    ph = by_cell["free_phishing_QRY"]["reading"]
    out = {"registration": E13.REGISTRATION + " (section 2.5)", "status": "DIAGNOSTIC",
           "by_cell": by_cell, "reading": benign, "licenses": LICENSES.get(benign, "nothing"),
           "phishing": ph, "extends_to_phishing": ph == benign}
    (E13D / "n_readout.json").write_text(json.dumps(out, indent=1, default=float), encoding="utf-8")
    for cell, r in by_cell.items():
        print(f"{cell}: {r['reading']} (holds {r['holds'] or 'none'}; usable {r['usable']}; "
              f"over natural {r['over_natural']}; closure {r['closure']})")
    print(f"\nN: {benign} -> {out['licenses']}; phishing {ph}")


if __name__ == "__main__":
    main()
