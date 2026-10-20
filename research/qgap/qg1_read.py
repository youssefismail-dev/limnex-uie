"""QG1 -- the reading (research/qgap/QG_PLAN.md sections 5.3-5.7).

D, T, S and G post-stratified by the shipped table's suffix groups; the three
factors and their log-shares of the gap; the route split; bootstrap CIs (1,000
draws, seed 20260806, resampling URLs within suffix groups -- the real rows are the
same rows for every model, so contrasts on real rows are paired); the descriptive
checks; VQ4; and the registered reading. Prints the tables for QG1_RESULTS.md and
writes data/qgap/qg1_report.json.

    python research/qgap/qg1_read.py > data/qgap/qg1_tables.md
"""
from __future__ import annotations

import argparse
import gzip
import json
import math
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE))

import numpy as np  # noqa: E402

import qg_common as Q  # noqa: E402
from uie.suffixes import SuffixTable  # noqa: E402

FINDING_D = {"benign": 0.0653, "phishing": 0.0451}     # FINDING_LENGTH_WEIGHTING.md 2.3
VQ4_TOL = 0.003
COLS = ("y_host", "y_path", "h_host", "h_path")
FACTORS = ("decision", "own_starts", "sampler")
MODELS = ("e9_ln", "e9_base", "e7")
#: models with generated files; --with-e10 adds `e10_base` to both (PREREG_E10 section 2)
GEN_MODELS = ("e9_ln", "e9_base")


ALLOW_DRY = False                  # --allow-dry: exercising the code only, never a reading


def load(d: Path, name: str, which: str, label: str):
    p = d / f"qg1_{name}_{which}_{label}.json.gz"
    if not p.is_file():
        return None
    with gzip.open(p, "rt", encoding="utf-8") as fh:
        payload = json.load(fh)
    if payload.get("limit") and not ALLOW_DRY:
        raise SystemExit(f"{p.name} is a dry run (limit {payload['limit']}); not read")
    return payload


def strata(payload, groups):
    """suffix group -> (n, 4) array (y_host, y_path, h_host, h_path) over readable
    URLs, in file order."""
    by = {}
    for suf, rec in zip(payload["suffix"], payload["records"]):
        if rec is not None:
            by.setdefault(Q.group_of(suf, groups), []).append([rec[c] for c in COLS])
    return {g: np.asarray(v, dtype=float) for g, v in by.items()}


def means(arrs, share, idx=None):
    """sum over present groups of share * group mean (renormalised), and the table
    mass of absent groups."""
    present = [g for g in share if g in arrs]
    tot = sum(share[g] for g in present)
    v = np.zeros(4)
    for g in present:
        a = arrs[g] if idx is None else arrs[g][idx[g]]
        v += share[g] / tot * a.mean(axis=0)
    return v, 1.0 - tot


def draw(arrs, rng):
    return {g: rng.integers(0, len(a), len(a)) for g, a in arrs.items()}


def quantities(real, gen) -> dict:
    """real = (D_host, D_path, T_host, T_path); gen = (G_host, G_path, S_host, S_path)."""
    q = {"D_host": real[0], "D_path": real[1], "T_host": real[2], "T_path": real[3],
         "G_host": gen[0], "G_path": gen[1], "S_host": gen[2], "S_path": gen[3]}
    for x in ("D", "T", "G", "S"):
        q[x] = q[f"{x}_host"] + q[f"{x}_path"]
    q["decision"], q["own_starts"], q["sampler"] = q["T"] / q["D"], q["S"] / q["T"], q["G"] / q["S"]
    q["G/D"] = q["G"] / q["D"]
    gap = math.log(q["G/D"])
    for f in FACTORS:
        q[f"share_{f}"] = math.log(q[f]) / gap
    for r in ("host", "path"):
        q[f"decision_{r}"] = q[f"T_{r}"] / q[f"D_{r}"]
        q[f"own_starts_{r}"] = q[f"S_{r}"] / q[f"T_{r}"]
        q[f"sampler_{r}"] = q[f"G_{r}"] / q[f"S_{r}"]
        q[f"gap_part_{r}"] = (q[f"D_{r}"] - q[f"G_{r}"]) / (q["D"] - q["G"])
    return q


def ci(values):
    a = np.sort(np.asarray(values, dtype=float))
    return [float(np.percentile(a, 2.5)), float(np.percentile(a, 97.5))]


def excludes_one(c):
    return not (c[0] <= 1.0 <= c[1])


def reading(pt: dict, c: dict) -> str:
    """QG_PLAN 5.6, rows in the order registered; the first that holds."""
    if excludes_one(c["sampler"]) and abs(pt["share_sampler"]) >= 0.2:
        return "SAMPLER"
    if pt["share_decision"] >= 0.5 and excludes_one(c["decision"]):
        return "DECISION-LED"
    if pt["share_own_starts"] >= 0.5 and excludes_one(c["own_starts"]):
        return "OWN-STARTS-LED"
    if (excludes_one(c["decision"]) and excludes_one(c["own_starts"])
            and pt["share_decision"] >= 0.25 and pt["share_own_starts"] >= 0.25):
        return "SPLIT"
    return "UNRESOLVED"


def per_class(d: Path, label: str, shipped) -> dict:
    groups, share = Q.suffix_groups(shipped[label])
    pay = {(m, w): load(d, m, w, label) for m in MODELS for w in ("train", "val", "gen")}
    real = {m: strata(pay[(m, "train")], groups) for m in MODELS if pay[(m, "train")]}
    val = {m: strata(pay[(m, "val")], groups) for m in MODELS if pay[(m, "val")]}
    gen = {m: strata(pay[(m, "gen")], groups) for m in GEN_MODELS if pay[(m, "gen")]}
    # real rows pair across models only if they are the same rows
    for which in ("train", "val"):
        urls = [pay[(m, which)]["urls"] for m in MODELS if pay[(m, which)]]
        if any(u != urls[0] for u in urls[1:]):
            raise SystemExit(f"{label} {which}: the models were not read on the same rows")

    def point(m):
        r, dropped_r = means(real[m], share)
        g, dropped_g = means(gen[m], share)
        return quantities(r, g), dropped_r, dropped_g

    out = {"groups": groups, "share": share}
    for m in gen:
        q, dr, dg = point(m)
        out[m] = {"point": q, "dropped_table_mass": {"real": dr, "gen": dg}}
    for m in real:
        r, _ = means(real[m], share)
        v, _ = means(val[m], share) if m in val else (None, None)
        out.setdefault(m, {})["decision_train"] = (r[2] + r[3]) / (r[0] + r[1])
        if v is not None:
            out[m]["decision_val"] = (v[2] + v[3]) / (v[0] + v[1])
            out[m]["D_val"] = v[0] + v[1]

    rng = np.random.default_rng(Q.BOOT_SEED)
    boots = {m: [] for m in gen}
    contrast, dec_tr, dec_va = [], {m: [] for m in real}, {m: [] for m in val}
    for _ in range(Q.BOOT):
        it, iv = draw(real["e9_ln"], rng), draw(val["e9_ln"], rng)
        qs = {}
        for m in gen:
            qs[m] = quantities(means(real[m], share, it)[0], means(gen[m], share, draw(gen[m], rng))[0])
            boots[m].append(qs[m])
        if "e9_base" in qs:
            contrast.append({f: qs["e9_ln"][f] / qs["e9_base"][f] for f in FACTORS})
        for m in real:
            r = means(real[m], share, it)[0]
            dec_tr[m].append((r[2] + r[3]) / (r[0] + r[1]))
        for m in val:
            v = means(val[m], share, iv)[0]
            dec_va[m].append((v[2] + v[3]) / (v[0] + v[1]))
    for m in gen:
        keys = out[m]["point"].keys()
        out[m]["ci"] = {k: ci([b[k] for b in boots[m]]) for k in keys}
        halfwidth = {f: (out[m]["ci"][f"share_{f}"][1] - out[m]["ci"][f"share_{f}"][0]) / 2
                     for f in FACTORS}
        out[m]["share_not_resolved"] = [f for f, h in halfwidth.items() if h > 0.25]
        out[m]["reading"] = reading(out[m]["point"], out[m]["ci"])
    if contrast:
        out["objective"] = {f: {"point": out["e9_ln"]["point"][f] / out["e9_base"]["point"][f],
                                "ci": ci([c[f] for c in contrast])} for f in FACTORS}
    for m in real:
        out[m]["decision_train_ci"] = ci(dec_tr[m])
        if m in val:
            out[m]["decision_val_ci"] = ci(dec_va[m])

    # descriptive: decoder and sampling, on the generated files
    for m in gen:
        p = pay[(m, "gen")]
        a = p["all"]
        g_by = [q / n for n, q in a["by_generation"]]
        pbar = sum(a["query"]) / a["n"]
        sub = p["records"]
        readable = [r is not None for r in sub]
        agree = sum(1 for r, qg in zip(sub, p["query_gate"])
                    if r is not None and (r["y_host"] + r["y_path"]) == qg)
        lost = [qg for r, qg in zip(sub, p["query_gate"]) if r is None]
        per_group = {}
        gq = {}
        for suf, qq in zip(a["suffix"], a["query"]):
            gq.setdefault(Q.group_of(suf, groups), []).append(qq)
        for g in groups + [Q.OTHER]:
            rr = real[m].get(g)
            if rr is None or g not in gq:
                continue
            ev = int(rr[:, 0].sum() + rr[:, 1].sum())
            per_group[g] = {"real_rows": len(rr), "real_events": ev, "D": float((rr[:, 0] + rr[:, 1]).mean()),
                            "gen_urls": len(gq[g]), "G": sum(gq[g]) / len(gq[g]),
                            "read": ev >= Q.MIN_EVENTS and sum(gq[g]) >= Q.MIN_EVENTS}
        out[m]["descriptive"] = {
            "G_all_plain": pbar, "G_by_generation": g_by,
            "G_by_generation_sd": float(np.std(g_by, ddof=1)),
            "binomial_sd": math.sqrt(pbar * (1 - pbar) / (a["n"] / len(g_by))),
            "empty_query_share": sum(a["empty_query"]) / a["n"],
            "subsample": len(sub), "unreadable": len(sub) - sum(readable),
            "unreadable_query_rate": (sum(lost) / len(lost)) if lost else None,
            "too_long": sum(p["too_long"]),
            "record_y_agrees_with_gate": agree / max(1, sum(readable)),
            "per_group": per_group}
    # VQ4: the post-stratified D against FINDING section 2.3
    d_strat = out["e9_ln"]["point"]["D"]
    out["vq4"] = {"D_stratified": d_strat, "finding": FINDING_D[label],
                  "pass": abs(d_strat - FINDING_D[label]) <= VQ4_TOL}
    return out


def pct(x):
    return f"{100 * x:.2f}%"


def r3(x):
    return f"{x:.3f}"


def with_ci(pt, c, f=r3):
    return f"{f(pt)} [{f(c[0])}, {f(c[1])}]"


def render(rep: dict) -> None:
    print("### Validity\n")
    for label in Q.LABELS:
        v = rep[label]["vq4"]
        print(f"- VQ4 {label}: post-stratified D {pct(v['D_stratified'])} against FINDING's "
              f"{pct(v['finding'])}: **{'PASS' if v['pass'] else 'FAIL'}**")
    print("- VQ1: every model and file was verified before measuring (qg1_measure.py); "
          "VQ2-VQ3: tests/test_qgap.py\n")
    print("### The decomposition (the baseline, `e9_ln` / raw)\n")
    print("| class | D | T | S | G | G / D |\n|---|---|---|---|---|---|")
    for label in Q.LABELS:
        p, c = rep[label]["e9_ln"]["point"], rep[label]["e9_ln"]["ci"]
        print(f"| {label} | " + " | ".join(with_ci(p[k], c[k], pct) for k in ("D", "T", "S", "G"))
              + f" | {with_ci(p['G/D'], c['G/D'])} |")
    print("\n| class | decision T / D | own starts S / T | sampler G / S | shares (decision / own starts / sampler) | reading |")
    print("|---|---|---|---|---|---|")
    for label in Q.LABELS:
        m = rep[label]["e9_ln"]
        p, c = m["point"], m["ci"]
        shares = " / ".join(with_ci(p[f"share_{f}"], c[f"share_{f}"], lambda x: f"{x:.2f}")
                            for f in FACTORS)
        nr = f" (not resolved: {', '.join(m['share_not_resolved'])})" if m["share_not_resolved"] else ""
        print(f"| {label} | " + " | ".join(with_ci(p[f], c[f]) for f in FACTORS)
              + f" | {shares}{nr} | **{m['reading']}** |")
    print("\n### By route (`e9_ln`)\n")
    print("| class | route | D | T | S | G | decision | own starts | sampler | part of the gap |")
    print("|---|---|---|---|---|---|---|---|---|---|")
    for label in Q.LABELS:
        p, c = rep[label]["e9_ln"]["point"], rep[label]["e9_ln"]["ci"]
        for r in ("host", "path"):
            print(f"| {label} | {r} | " + " | ".join(pct(p[f"{x}_{r}"]) for x in ("D", "T", "S", "G"))
                  + " | " + " | ".join(with_ci(p[f"{f}_{r}"], c[f"{f}_{r}"]) for f in FACTORS)
                  + f" | {with_ci(p[f'gap_part_{r}'], c[f'gap_part_{r}'], lambda x: f'{x:.2f}')} |")
    print("\n### The objective: `e9_base` (same start, old objective) and the ratio `e9_ln` / `e9_base`\n")
    print("| class | factor | e9_base | e9_ln | e9_ln / e9_base |\n|---|---|---|---|---|")
    for label in Q.LABELS:
        if "objective" not in rep[label]:
            continue
        for f in FACTORS:
            b, n = rep[label]["e9_base"], rep[label]["e9_ln"]
            o = rep[label]["objective"][f]
            print(f"| {label} | {f} | {with_ci(b['point'][f], b['ci'][f])} | "
                  f"{with_ci(n['point'][f], n['ci'][f])} | {with_ci(o['point'], o['ci'])} |")
    if any("e10_base" in rep[label] for label in Q.LABELS):
        print("\n### E10, reported: `e10_base` (the baseline recipe, another run, same start) "
              "against `e9_ln`\n")
        print("| class | factor | e9_ln | e10_base |\n|---|---|---|---|")
        for label in Q.LABELS:
            n, t = rep[label]["e9_ln"], rep[label]["e10_base"]
            for f in FACTORS + ("G/D",):
                print(f"| {label} | {f} | {with_ci(n['point'][f], n['ci'][f])} | "
                      f"{with_ci(t['point'][f], t['ci'][f])} |")
            for r in ("host", "path"):
                k = f"decision_{r}"
                print(f"| {label} | decision, {r} route | {with_ci(n['point'][k], n['ci'][k])} | "
                      f"{with_ci(t['point'][k], t['ci'][k])} |")
            print(f"| {label} | reading | {n['reading']} | {t['reading']} |")
    print("\n### Capacity: the decision on training and validation rows\n")
    print("| class | model | T / D, training rows | T / D, validation rows | D, validation |\n|---|---|---|---|---|")
    for label in Q.LABELS:
        for m in MODELS:
            x = rep[label].get(m, {})
            if "decision_train" not in x:
                continue
            print(f"| {label} | {m} | {with_ci(x['decision_train'], x['decision_train_ci'])} | "
                  f"{with_ci(x['decision_val'], x['decision_val_ci'])} | {pct(x['D_val'])} |")
    print("\n### Decoder and sampling (generated files)\n")
    print("| class | model | G, all 40,000 | per generation | SD (binomial SD) | empty `?` | "
          "unreadable in the subsample (their query rate) | re-encoded y = gate's |")
    print("|---|---|---|---|---|---|---|---|")
    for label in Q.LABELS:
        for m in GEN_MODELS:
            if "descriptive" not in rep[label].get(m, {}):
                continue
            x = rep[label][m]["descriptive"]
            ur = x["unreadable_query_rate"]
            print(f"| {label} | {m} | {pct(x['G_all_plain'])} | "
                  + ", ".join(pct(g) for g in x["G_by_generation"])
                  + f" | {pct(x['G_by_generation_sd'])} ({pct(x['binomial_sd'])}) | "
                  f"{pct(x['empty_query_share'])} | {x['unreadable']} of {x['subsample']:,} "
                  f"({'—' if ur is None else pct(ur)}) | {pct(x['record_y_agrees_with_gate'])} |")
    print("\n### G against D by suffix group (`e9_ln`; groups with at least 100 query events on each side)\n")
    print("| class | group | real rows | D | generated | G | G / D |\n|---|---|---|---|---|---|---|")
    for label in Q.LABELS:
        for g, x in rep[label]["e9_ln"]["descriptive"]["per_group"].items():
            if x["read"]:
                print(f"| {label} | {g} | {x['real_rows']:,} | {pct(x['D'])} | {x['gen_urls']:,} | "
                      f"{pct(x['G'])} | {x['G'] / x['D']:.2f} |")
    data = rep.get("data")
    if data:
        print("\n### The data side (whole pools; `qg1_data.py`)\n")
        print("| class | pool | rows | plain, all | plain, fit | shipped, all | shipped, fit |"
              "\n|---|---|---|---|---|---|---|")
        for label in Q.LABELS:
            for which in ("train", "val", "page"):
                x = data[label][which]
                print(f"| {label} | {x['stem']} | {x['rows']:,} | {pct(x['plain_all'])} | "
                      f"{pct(x['plain_fit'])} | {pct(x['shipped_all']['finding'])} | "
                      f"{pct(x['shipped_fit']['finding'])} |")
    print(f"\n### Reading\n\n- benign (decides): **{rep['benign']['e9_ln']['reading']}**"
          f"\n- phishing: {rep['phishing']['e9_ln']['reading']}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", type=Path, default=Q.DATA)
    ap.add_argument("--allow-dry", action="store_true",
                    help="exercise the code on dry-run files; the output is NOT a reading")
    ap.add_argument("--with-e10", action="store_true",
                    help="E10's reported reading: adds e10_base; writes qg1_report_e10.json")
    args = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")   # a redirected Windows console is cp1252
    global ALLOW_DRY, MODELS, GEN_MODELS
    ALLOW_DRY = args.allow_dry
    if args.with_e10:
        MODELS, GEN_MODELS = MODELS + ("e10_base",), GEN_MODELS + ("e10_base",)
    if ALLOW_DRY:
        print("**DRY RUN -- exercising the code; NOT a reading of QG1.**\n")
    shipped = SuffixTable.load(Q.BUNDLE / "suffix_table.json").counts
    rep = {"registration": "research/qgap/QG_PLAN.md (QG1)"}
    for label in Q.LABELS:
        rep[label] = per_class(args.dir, label, shipped)
    data = args.dir / "qg1_data.json"
    if data.is_file():
        rep["data"] = json.loads(data.read_text(encoding="utf-8"))
    out = "qg1_report_e10.json" if args.with_e10 else "qg1_report.json"
    (args.dir / out).write_text(json.dumps(rep, indent=1, default=float), encoding="utf-8")
    render(rep)


if __name__ == "__main__":
    main()
