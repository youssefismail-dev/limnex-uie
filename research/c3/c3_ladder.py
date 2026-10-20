"""C3-QRY, C3-B -- where does E7's query tell live, in each mode?

Q3.1's ladder, unchanged: `research.qry.q3_1_resolve` (run_ladder, load_real,
LADDER, D = 12, t_crit 2.201) over `q3_content.scored` (the floor recomputed
under every transform). Four ladders: {benign, phishing} x {Q, F}. Q3.1's
pre-specified interaction, `clip15_first`, runs in a ladder only if
`clip_values_15` and `first_param` are both "no change" there.

Readings, fixed before any C3 number (PREREG_C3_QRY section 5):

    per ladder   SHAPE        skeleton R_T >= 0.85 and skeleton does not REDUCE
                 VOCABULARY   skeleton R_T <= 0.40 and it REDUCES
                 INTERMEDIATE otherwise
    per class    IDENTIFIED   a property REDUCES in both modes, with R_T <= 0.50 in
                              both, monotone in its dose series where it has one
                 DISTRIBUTED  otherwise, two or more properties REDUCE in both modes
                 UNRESOLVED   neither

A PROPERTY is a destructive transform, except that the value-tail caps
clip_values_20/15/10 are one property (a dose series). skeleton and
structure_only are sufficiency tests, not properties.

Validity (K3): a row whose floor leaves [0.42, 0.55] or whose distinct fraction
falls under 0.50 is VOID (run_ladder marks it). A whole ladder is VOID if `null`
is not a byte-exact no-op (its draws must equal identity's) or if identity's
excess differs from the same arm's C3-A x by more than 0.05.

    python research/c3/c3_ladder.py --c3 data/c3 --sc1 data/sc1
"""
import argparse
import gc
import json
import os
import sys
from pathlib import Path
from typing import Dict, List

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE))

from c3_common import G, tag  # noqa: E402


def collecting(fn):
    """gc.collect() after every discriminator call -- memory only, as the gate
    does (uie/corpus/run_eval.py, 2026-09-23) and cr1_score after it. Q3.1's
    run_ladder predates that fix: under Python 3.14 each call left ~120 MB of
    cyclic garbage the incremental collector did not reclaim in time, and the
    C3 dry run was stopped by its memory guard after ~50 calls. Measured on 8
    calls: values bit-identical with and without it; RSS +1,014 MB without,
    +137 MB (flat) with."""
    def wrapped(*args, **kwargs):
        try:
            return fn(*args, **kwargs)
        finally:
            gc.collect()
    return wrapped

PROPERTIES = {
    "long-value tail": ["clip_values_20", "clip_values_15", "clip_values_10"],
    "parameter count": ["first_param"],
    "name length": ["fixlen_names"],
    "percent-encoding": ["mask_pct"],
    "separators": ["normalize_seps"],
    "punctuation": ["mask_punct"],
    "digit vs alpha": ["alnum_flat"],
    "name identity": ["mask_name_chars"],
    "value composition": ["mask_values"],
    "value length": ["fixlen_values"],
    "ordering": ["sort_params"],
    "tail + count": ["clip15_first"],
}
DOSE = {"long-value tail": ["clip_values_20", "clip_values_15", "clip_values_10"]}
SHAPE_HI, VOCAB_LO, IDENT_R = 0.85, 0.40, 0.50
IDENTITY_TOL = 0.05


def void_row(row) -> bool:
    return row is None or str(row.get("verdict", "")).startswith("VOID")


def ladder_validity(rows: Dict[str, dict], c3a_x: float) -> List[str]:
    bad = []
    if rows.get("null", {}).get("excess_draws") != rows.get("identity", {}).get("excess_draws"):
        bad.append("null is not a byte-exact no-op")
    ident = rows.get("identity", {}).get("excess")
    if ident is None or abs(ident - c3a_x) > IDENTITY_TOL:
        bad.append(f"identity excess {ident} is more than {IDENTITY_TOL} from C3-A x {c3a_x}")
    return bad


def shape_reading(rows: Dict[str, dict]) -> str:
    sk = rows.get("skeleton")
    if void_row(sk):
        return "VOID"
    reduces = sk["verdict"] == "REDUCES"
    if sk["R_T"] >= SHAPE_HI and not reduces:
        return "SHAPE"
    if sk["R_T"] <= VOCAB_LO and reduces:
        return "VOCABULARY"
    return "INTERMEDIATE"


def _reducing(rows, transforms):
    return [t for t in transforms if not void_row(rows.get(t)) and rows[t]["verdict"] == "REDUCES"]


def _monotone(rows, series) -> bool:
    r = [rows.get(t) for t in series]
    if any(void_row(x) for x in r):
        return False
    return r[0]["R_T"] >= r[1]["R_T"] >= r[2]["R_T"]


def component_reading(modes: Dict[str, Dict[str, dict]]) -> dict:
    """modes = {"Q": rows, "F": rows} of one class."""
    both, detail = [], {}
    for prop, transforms in PROPERTIES.items():
        red = {m: _reducing(rows, transforms) for m, rows in modes.items()}
        detail[prop] = {m: {"reducing": red[m],
                            "best_R_T": min((rows[t]["R_T"] for t in red[m]), default=None)}
                        for m, rows in modes.items()}
        if all(red[m] for m in modes):
            both.append(prop)
    identified = [p for p in both
                  if all(detail[p][m]["best_R_T"] <= IDENT_R for m in modes)
                  and (p not in DOSE or all(_monotone(rows, DOSE[p]) for rows in modes.values()))]
    if identified:
        reading = "IDENTIFIED"
    elif len(both) >= 2:
        reading = "DISTRIBUTED"
    else:
        reading = "UNRESOLVED"
    return {"reading": reading, "identified": identified, "reduce_in_both": both,
            "detail": detail}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--c3", type=Path, required=True)
    ap.add_argument("--sc1", type=Path, required=True)
    args = ap.parse_args()
    c3, sc1 = args.c3.resolve(), args.sc1.resolve()
    os.chdir(ROOT)                                   # load_real reads data/bundle2

    import research.qry.q3_1_resolve as q31
    from research.qry.q3_1_resolve import LADDER, TCRIT, D, load_real, run_ladder
    from uie.evaluation import FIELDS
    q31.scored = collecting(q31.scored)              # run_ladder's global; Q3.1's file untouched

    q_rep = json.loads((sc1 / "T" / "cr1_report.json").read_text(encoding="utf-8"))["states"]
    f_rep = json.loads((c3 / "F" / "report.json").read_text(encoding="utf-8"))["states"]
    out = {"registration": "research/c3/PREREG_C3_QRY.md", "D": D, "t_crit": round(TCRIT, 4),
           "ladders": {}, "validity": {}, "shape": {}, "components": {}, "e5_published": {}}
    for label in ("benign", "phishing"):
        t = tag(label)
        if "k1" in f_rep[t]:                          # PREREG section 8: not read
            for arm in ("Q", "F"):
                out["validity"][f"{label}/{arm}"] = "K1"
                out["shape"][f"{label}/{arm}"] = "NOT READ (K1)"
            out["components"][label] = {"reading": "NOT READ (K1)", "k1": f_rep[t]["k1"]}
            (c3 / "c3b_report.json").write_text(json.dumps(out, indent=1, default=str),
                                                encoding="utf-8")
            continue
        real_q = load_real(label)
        arms = {}
        for arm, d in (("Q", sc1 / "T" / "gen"), ("F", c3 / "F" / "gen")):
            urls = [u for g in range(G)
                    for u in json.loads((d / f"{t}_g{g}.json").read_text(encoding="utf-8"))]
            arms[arm] = [x for x in (FIELDS["QRY"](u) for u in urls) if x]
        x_c3a = {"Q": q_rep[t]["fields"]["QRY"]["acceptance"]["x"],
                 "F": f_rep[t]["fields"]["QRY"]["acceptance"]["x"]}
        modes = {}
        for arm, fake_q in arms.items():
            rows = run_ladder(real_q, fake_q, LADDER, f"C3-B  {label}  arm {arm}")
            if all(rows.get(n, {}).get("verdict") == "no change"
                   for n in ("clip_values_15", "first_param")):
                inter = run_ladder(real_q, fake_q, ["identity", "clip15_first"],
                                   f"C3-B  {label}  arm {arm}  interaction")
                rows.update({k: v for k, v in inter.items() if k != "identity"})
            key = f"{label}/{arm}"
            out["ladders"][key] = rows
            bad = ladder_validity(rows, x_c3a[arm])
            out["validity"][key] = bad or "valid"
            out["shape"][key] = "NOT READ (ladder void)" if bad else shape_reading(rows)
            modes[arm] = rows if not bad else None
        if all(m is not None for m in modes.values()):
            out["components"][label] = component_reading(modes)
        else:
            out["components"][label] = {"reading": "NOT READ (a ladder is void)"}
        pub = ROOT / "data" / "qry" / f"q3_1_{label}.json"
        if pub.is_file():
            s1 = json.loads(pub.read_text(encoding="utf-8"))["stages"].get("stage1_typical", {})
            out["e5_published"][label] = {k: {"R_T": v.get("R_T"), "verdict": v.get("verdict")}
                                          for k, v in s1.items()}
        (c3 / "c3b_report.json").write_text(json.dumps(out, indent=1, default=str),
                                            encoding="utf-8")

    print("\nC3-B readings")
    for key, s in out["shape"].items():
        print(f"  {key:<14} skeleton -> {s}   validity: {out['validity'][key]}")
    for label, c in out["components"].items():
        print(f"  {label:<9} components -> {c['reading']}  "
              f"{c.get('identified') or c.get('reduce_in_both') or ''}")


if __name__ == "__main__":
    main()
