"""QG2-D -- the reading (research/qgap/PREREG_QG2D.md sections 3-7): per-category
calibration by region, the host-end and path-end splits, the slice contrasts with
paired bootstrap CIs, S4's route split, VP1-VP3, the audit, and the registered
reading. Prints the tables for QG2D_RESULTS.md; writes data/qgap/qg2_report.json.

    python research/qgap/qg2_read.py > data/qgap/qg2_tables.md
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE))

import numpy as np  # noqa: E402

import qg_common as Q  # noqa: E402

SLICES = ("S0", "S1", "S2", "S3", "S4")
PAIRED = ("S0", "S2", "S3", "S4")               # the same rows
QG1 = {"benign": {"host": 0.539, "path": 0.806},   # QG1_RESULTS.md section 2
       "phishing": {"host": 0.592, "path": 0.780}}
VP1_TOL, VP2_TOL = 0.05, 1e-4
C = {c: i for i, c in enumerate(Q.CATS)}
HE = [C[c] for c in Q.HOST_END]
PE = [C[c] for c in Q.PATH_END]


def load(name, label):
    base = Q.DATA / f"qg2_{name}_{label}"
    meta = json.loads(base.with_suffix(".json").read_text(encoding="utf-8"))
    if meta.get("limit"):
        raise SystemExit(f"{base.name} is a dry run; not read")
    arr = dict(np.load(base.with_suffix(".npz")))
    return arr, meta


def stats(a, idx=None) -> dict:
    """Everything the reading needs from one slice (optionally a resample)."""
    if idx is not None:
        a = {k: v[idx] for k, v in a.items()}
    out = {}
    for region, p, y in (("host", "hp", "hy"), ("path", "pp", "py")):
        sp, sy = a[p].sum(0), a[y].sum(0)
        for c, i in C.items():
            out[f"{region}_{c}"] = sp[i] / sy[i] if sy[i] > 0 else np.nan
            out[f"{region}_{c}_events"] = sy[i]
    he = a["hep"][:, HE]
    split = he / he.sum(1, keepdims=True)
    data = np.array([(a["hey"] == C[c]).mean() for c in Q.HOST_END])
    for j, c in enumerate(Q.HOST_END):
        out[f"he_model_{c}"] = split[:, j].mean()
        out[f"he_data_{c}"] = data[j]
    out["he_host_continues"] = a["hep"][:, [C["content"], C["SUB"], C["PRT"]]].sum(1).mean()
    has = a["has_path"] == 1
    pe = a["pep"][has][:, PE]
    psplit = pe / pe.sum(1, keepdims=True)
    pdata = np.array([(a["pey"][has] == C[c]).mean() for c in Q.PATH_END])
    for j, c in enumerate(Q.PATH_END):
        out[f"pe_model_{c}"] = psplit[:, j].mean()
        out[f"pe_data_{c}"] = pdata[j]
    out["pe_path_continues"] = a["pep"][has][:, C["content"]].mean()
    return out


def s4_split(a, idx=None) -> dict:
    """S4, the query state told: where the model puts a query it was told exists."""
    if idx is not None:
        a = {k: v[idx] for k, v in a.items()}
    q = C["QRY"]
    told = (a["hy"][:, q] + a["py"][:, q]) > 0
    hq, pq = a["hp"][told, q].sum(), a["pp"][told, q].sum()
    dh, dp = a["hy"][told, q].sum(), a["py"][told, q].sum()
    no = ~told
    return {"model_host_share": hq / (hq + pq), "data_host_share": dh / (dh + dp),
            "hazard_per_told_row": (hq + pq) / told.sum(),
            "hazard_per_forbidden_row": (a["hp"][no, q].sum() + a["pp"][no, q].sum()) / no.sum()}


def ci(v):
    v = np.asarray(v, dtype=float)
    v = v[np.isfinite(v)]
    if not len(v):                          # a category with no events: no estimate
        return [float("nan"), float("nan")]
    return [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]


def per_class(label: str) -> dict:
    arrs, metas = {}, {}
    for s in SLICES:
        arrs[s], metas[s] = load(s, label)
    rows0 = arrs["S0"]["row"]
    if any(not np.array_equal(arrs[s]["row"], rows0) for s in PAIRED):
        raise SystemExit(f"{label}: the paired slices were not read on the same rows")
    pt = {s: stats(arrs[s]) for s in SLICES}
    pt["S4_split"] = s4_split(arrs["S4"])
    rng = np.random.default_rng(Q.BOOT_SEED)
    n0, n1 = len(rows0), len(arrs["S1"]["row"])
    boot = {s: [] for s in SLICES}
    boot["S4_split"], boot["d_S3"], boot["d_he_PTH"], boot["d_he_QRY_frames"] = [], [], [], []
    for _ in range(Q.BOOT):
        i0, i1 = rng.integers(0, n0, n0), rng.integers(0, n1, n1)
        b = {s: stats(arrs[s], i0) for s in PAIRED}
        b["S1"] = stats(arrs["S1"], i1)
        for s in SLICES:
            boot[s].append(b[s])
        boot["S4_split"].append(s4_split(arrs["S4"], i0))
        boot["d_S3"].append(b["S3"]["host_QRY"] - b["S0"]["host_QRY"])
        boot["d_he_PTH"].append(b["S0"]["he_model_PTH"] - b["S0"]["he_data_PTH"])
        boot["d_he_QRY_frames"].append(b["S1"]["he_data_QRY"] - b["S0"]["he_data_QRY"])
    cis = {s: {k: ci([d[k] for d in boot[s]]) for k in pt[s]} for s in SLICES}
    cis["S4_split"] = {k: ci([d[k] for d in boot["S4_split"]]) for k in pt["S4_split"]}
    for k in ("d_S3", "d_he_PTH", "d_he_QRY_frames"):
        cis[k] = ci(boot[k])
    r0, r1, r3 = pt["S0"]["host_QRY"], pt["S1"]["host_QRY"], pt["S3"]["host_QRY"]
    closure = (r3 - r0) / (1 - r0)
    pt.update({"closure_S3": closure, "d_S3": r3 - r0,
               "d_he_PTH": pt["S0"]["he_model_PTH"] - pt["S0"]["he_data_PTH"],
               "d_he_QRY_frames": pt["S1"]["he_data_QRY"] - pt["S0"]["he_data_QRY"]})
    validity = {
        "VP1": {r: {"S0": pt["S0"][f"{r}_QRY"], "QG1": QG1[label][r],
                    "pass": abs(pt["S0"][f"{r}_QRY"] - QG1[label][r]) <= VP1_TOL}
                for r in ("host", "path")},
        "VP2": {s: metas[s]["vp2_max_dev"] for s in SLICES},
        "VP3": {s: metas[s]["vp3_bad_controls"] for s in SLICES},
        "rows": {s: metas[s]["readable"] for s in SLICES}}
    validity["pass"] = (all(v["pass"] for v in validity["VP1"].values())
                        and max(validity["VP2"].values()) <= VP2_TOL
                        and not any(validity["VP3"].values()))
    return {"point": pt, "ci": cis, "validity": validity}


def reading(rep: dict, audit: dict) -> str:
    """PREREG_QG2D section 7: rows in their registered order, the first that holds."""
    if audit.get("findings"):
        return "AUDIT-FOUND"
    p, c = rep["point"], rep["ci"]
    if (c["S1"]["host_QRY"][0] > 1 and c["S0"]["host_QRY"][1] < 1
            and c["d_he_QRY_frames"][1] < 0):
        return "FRAME-POOLING"
    if (p["closure_S3"] >= 0.5 and not c["d_S3"][0] <= 0 <= c["d_S3"][1]
            and c["d_he_PTH"][0] > 0):
        return "PATH-COMPETITION"
    if p["closure_S3"] < 0.25 and c["S1"]["host_QRY"][1] < 1:
        return "QUERY-STATE"
    return "UNRESOLVED"


def f3(x):
    return "—" if x is None or not np.isfinite(x) else f"{x:.3f}"


def pc(x):
    return f"{100 * x:.2f}%"


def wc(x, c, f=f3):
    return f"{f(x)} [{f(c[0])}, {f(c[1])}]"


def render(rep: dict, audit: dict) -> None:
    print("### Validity\n")
    for label in Q.LABELS:
        v = rep[label]["validity"]
        vp1 = "; ".join(f"{r} {v['VP1'][r]['S0']:.3f} against QG1 {v['VP1'][r]['QG1']:.3f}"
                        for r in ("host", "path"))
        print(f"- {label}: VP1 {vp1}; VP2 max |sum − 1| {max(v['VP2'].values()):.1e}; "
              f"VP3 mis-encoded controls {sum(v['VP3'].values())}; rows "
              + ", ".join(f"{s} {n:,}" for s, n in v["rows"].items())
              + f" → **{'PASS' if v['pass'] else 'FAIL'}**")
    print(f"- Audit (A1–A5): findings {audit.get('findings') or 'none'}\n")
    print("### Probe 1 — the host-route query decision by slice (Σ p(<QRY>) / queries)\n")
    print("| class | S0 free mode | S1 multi-page sites | S2 frame swapped | S3 path told | S4 query told |")
    print("|---|---|---|---|---|---|")
    for label in Q.LABELS:
        p, c = rep[label]["point"], rep[label]["ci"]
        print(f"| {label} | " + " | ".join(wc(p[s]["host_QRY"], c[s]["host_QRY"]) for s in SLICES) + " |")
    print("\n| class | S3 closure of S0's host-route deficit | r_S3 − r_S0 (paired) |\n|---|---|---|")
    for label in Q.LABELS:
        p, c = rep[label]["point"], rep[label]["ci"]
        print(f"| {label} | {p['closure_S3']:.2f} | {wc(p['d_S3'], c['d_S3'])} |")
    print("\n**The path route, by slice (description only):**\n")
    print("| class | S0 | S1 | S2 | S3 | S4 |\n|---|---|---|---|---|---|")
    for label in Q.LABELS:
        p, c = rep[label]["point"], rep[label]["ci"]
        print(f"| {label} | " + " | ".join(wc(p[s]["path_QRY"], c[s]["path_QRY"]) for s in SLICES) + " |")
    print("\n### Probe 2 — at the end of a real host (S0): the model's split against the data's\n")
    print("| class | slice | " + " | ".join(f"model {t} / data {t}" for t in Q.HOST_END)
          + " | model: host continues |")
    print("|---|---|" + "---|" * len(Q.HOST_END) + "---|")
    for label in Q.LABELS:
        for s in ("S0", "S1", "S3"):
            p = rep[label]["point"][s]
            print(f"| {label} | {s} | " + " | ".join(
                f"{pc(p[f'he_model_{t}'])} / {pc(p[f'he_data_{t}'])}" for t in Q.HOST_END)
                + f" | {pc(p['he_host_continues'])} |")
    print("\n| class | S0: model PTH share − data PTH share | page − domain: data's host-end query share |")
    print("|---|---|---|")
    for label in Q.LABELS:
        p, c = rep[label]["point"], rep[label]["ci"]
        print(f"| {label} | {wc(p['d_he_PTH'], c['d_he_PTH'], pc)} | "
              f"{wc(p['d_he_QRY_frames'], c['d_he_QRY_frames'], pc)} |")
    print("\n**Per-category calibration in the host region (S0): Σ p / events**\n")
    cats = [c for c in Q.CATS if c not in ("other",)]
    print("| class | " + " | ".join(cats) + " |\n|---|" + "---|" * len(cats))
    for label in Q.LABELS:
        p, c = rep[label]["point"]["S0"], rep[label]["ci"]["S0"]
        print(f"| {label} | " + " | ".join(wc(p[f'host_{k}'], c[f'host_{k}']) if np.isfinite(p[f'host_{k}'])
                                           else "—" for k in cats) + " |")
    print("\n**At the end of a real path (S0): the model's split against the data's**\n")
    print("| class | " + " | ".join(f"model {t} / data {t}" for t in Q.PATH_END)
          + " | model: path continues |")
    print("|---|" + "---|" * len(Q.PATH_END) + "---|")
    for label in Q.LABELS:
        p = rep[label]["point"]["S0"]
        print(f"| {label} | " + " | ".join(f"{pc(p[f'pe_model_{t}'])} / {pc(p[f'pe_data_{t}'])}"
                                           for t in Q.PATH_END)
              + f" | {pc(p['pe_path_continues'])} |")
    print("\n### S4 — told a query exists: where the model puts it\n")
    print("| class | model host-route share | data host-route share | hazard per told row | hazard per row told none |")
    print("|---|---|---|---|---|")
    for label in Q.LABELS:
        p, c = rep[label]["point"]["S4_split"], rep[label]["ci"]["S4_split"]
        print(f"| {label} | {wc(p['model_host_share'], c['model_host_share'])} | "
              f"{f3(p['data_host_share'])} | {wc(p['hazard_per_told_row'], c['hazard_per_told_row'])} | "
              f"{wc(p['hazard_per_forbidden_row'], c['hazard_per_forbidden_row'])} |")
    print(f"\n### Reading\n\n- benign (decides): **{rep['benign']['reading']}**"
          f"\n- phishing: {rep['phishing']['reading']}")


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")   # a redirected Windows console is cp1252
    audit = json.loads((Q.DATA / "qg2_audit.json").read_text(encoding="utf-8"))
    rep = {"registration": "research/qgap/PREREG_QG2D.md"}
    for label in Q.LABELS:
        rep[label] = per_class(label)
        rep[label]["reading"] = reading(rep[label], audit)
    (Q.DATA / "qg2_report.json").write_text(json.dumps(rep, indent=1, default=float),
                                           encoding="utf-8")
    render(rep, audit)


if __name__ == "__main__":
    main()
