"""E13 -- render the registered readout as tables (research/e13/PREREG_E13.md section 3).
Reads data/e13/compare_report.json, the merged score report, dx8_ctx.json,
n_measure.json, n_readout.json and data/e13p/readout.json; prints markdown.

    python research/e13/e13_tables.py > data/e13/tables.md
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
for p in (ROOT, HERE, ROOT / "research" / "native", ROOT / "research" / "cr1"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from e13_compare import merged  # noqa: E402

E13D = ROOT / "data" / "e13"
ARMS = ("e12_final", "e12_d1", "e12_dec", "e13_ctx")


def load(p):
    return json.loads(Path(p).read_text(encoding="utf-8"))


def acc(row):
    a = (row or {}).get("acceptance") or {}
    return a.get("x"), a.get("verdict")


def cell(x, v):
    return "—" if x is None else f"{x:+.4f} {v}"


def slash_attribution(arm, base):
    """Each phishing free bare host without the slash, by its supplied state
    (e12_score.state_positions: the seeds, never the output)."""
    import collections
    import e12_common as K
    from e11_score import slash_state
    from e12_score import state_positions
    from uie.suffixes import SuffixTable
    shipped = SuffixTable.load(ROOT / "data" / "bundle2" / "suffix_table.json")
    ps = load(ROOT / "data" / "e12" / "presence_slash_table.json")["table"]
    ft = K.supplied_tables(ps, shipped.counts, "phishing")
    miss, bare = collections.Counter(), 0
    for g in range(K.G):
        urls = load(base / f"free_{arm}_phishing_g{g}.json")
        for u, k in zip(urls, state_positions(ft, len(urls), K.seed_free(g))):
            st = slash_state(u)
            if st is not None and st[0] is False:
                bare += 1
                miss[k] += not st[1]
    return bare, {k: v for k, v in miss.items() if v}


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    rep, cmp_ = merged(), load(E13D / "compare_report.json")
    print("### Validity\n\nVERIFY_TRAIN, VERIFY_DECODE, VERIFY_GENERATE: "
          + ", ".join(load(E13D / f"VERIFY_{s}.json")["verdict"] for s in ("TRAIN", "DECODE", "GENERATE"))
          + f"; V4: max |x - published| {load(E13D / 'score' / 'v4.json')['max_abs_diff']}\n")

    print("### The criteria\n\n| lever | criterion | result |\n|---|---|---|")
    for lever, key in (("C (e13_ctx)", "C"), ("D1 (e12_d1)", "D1"), ("D2 (e12_dec)", "D2")):
        node = cmp_[key]
        for k, v in node.items():
            if isinstance(v, dict) and "pass" in v:
                print(f"| {lever} | {k} | {'PASS' if v['pass'] else '**FAIL**'} |")
        print(f"| {lever} | **adopted** | **{'yes' if node['adopted'] else 'no'}** |")
    c = cmp_["C"]
    k1, k2 = c["K13-1"], c["K13-2"]
    print(f"\nK13-1: free phishing QRY {k1['x_ref']:+.4f} -> {k1['x_arm']:+.4f} (fall {k1['fall']:.4f}; "
          f"needs > 0.0141 and not CONVICTED); verdict {k1['verdict']}.")
    print(f"K13-2: x_GV {k2['x_gv']:+.4f}, x_RC {k2['x_rc']:+.4f}, gap {k2['gap_arm']:.4f} (needs <= "
          f"{0.5 * k2['gap_ref']:.4f}; e12_final: x_GV +0.1237, x_RC +0.0463, gap {k2['gap_ref']}).")
    nov = c["G5_G6"]["novelty"]
    print("G5/G6 (DC1's novelty rule): " + "; ".join(
        f"{k} {v['base']:.4f} -> {v['dc1']:.4f}{' WORSENED' if v['worsened'] else ''}" for k, v in nov.items()))
    for name in ("D2",):
        t = cmp_[name]["K13-4"]["targets"]
        print("K13-4: " + "; ".join(f"{s} {r['cell']}: delta {r['delta']:+.4f} (p {r['p']:.2f}), "
                                    f"{r['v_ref']} -> {r['v_arm']}" for s, r in t.items()))
        print("D2 R-a: " + "; ".join(f"{g['cell']} {g['v_ref']} -> {g['v_arm']} (delta {g['delta']:+.4f}, "
                                     f"Holm p {g['p_holm']:.2f})" for g in cmp_[name]["R-a"]["regressions"]))

    print("\n### The slash rows (K12b; T as registered)\n\n| row | real | " + " | ".join(ARMS[:2]) +
          " | e13_ctx |\n|---|---|---|---|---|")
    rows_ref = cmp_["C"]["G3"]["k12b"]["rows"]
    d1 = cmp_["D1"]["K13-3"]["rows"]
    from e12_compare import k12b
    targets = load(ROOT / "data" / "e11" / "targets.json")
    base = k12b(rep, targets, "e12_final")["rows"]
    for k in d1:
        def f(r):
            return f"{100 * r['q']:.3f}%{'' if r['pass'] else ' **miss**'}"
        print(f"| {k} | {100 * d1[k]['p']:.3f}% (±{100 * d1[k]['T']:.3f}) | {f(base[k])} | {f(d1[k])} | "
              f"{f(rows_ref[k])} |")

    print("\n### Gated cells that are not CLEARED in some arm, and FULL\n\n| cell | " + " | ".join(ARMS) +
          " |\n|---|---|---|---|---|")
    for label in ("benign", "phishing"):
        for f in ("LBL", "SUF", "SUB", "PTH", "QRY", "FULL"):
            cs = [acc(rep["free"][a][label]["fields"].get(f)) for a in ARMS]
            if f == "FULL" or any(v not in ("CLEARED",) for _, v in cs):
                print(f"| free {label} {f} | " + " | ".join(cell(*x) for x in cs) + " |")
    for key in rep["control"]["e12_final"]:
        for f in ("LBL", "SUF", "SUB", "PTH", "QRY", "FULL"):
            cs = [acc(rep["control"][a][key]["fields"].get(f)) for a in ARMS]
            if cs[0][0] is None:
                continue
            if f == "FULL" or any(v not in ("CLEARED",) for _, v in cs):
                print(f"| control {key} {f} | " + " | ".join(cell(*x) for x in cs) + " |")

    print("\n### The slash misses, by supplied state (phishing free bare hosts; S from the seeds)\n")
    print("| arm | bare hosts | without the slash | by supplied state |\n|---|---|---|---|")
    for arm, base in (("e12_final", ROOT / "data" / "e12" / "gen"), ("e12_d1", E13D / "decode" / "gen")):
        bare, miss = slash_attribution(arm, base)
        print(f"| {arm} | {bare:,} | {sum(miss.values())} | {dict(sorted(miss.items()))} |")
    print("\n### S honouring (control P0 states, and free mode with every state)\n")
    print("| population | S | e12_final | e12_d1 | e13_ctx |\n|---|---|---|---|---|")
    for key in rep["s_honouring"]["e12_final"]:
        for sv in rep["s_honouring"]["e12_final"][key]:
            vals = [rep["s_honouring"][a].get(key, {}).get(sv) for a in ("e12_final", "e12_d1", "e13_ctx")]
            print(f"| {key} | {sv} | " + " | ".join(f"{100 * v['rate']:.3f}% of {v['n']:,}" if v else "—"
                                                    for v in vals) + " |")
    d = load(E13D / "dx8_ctx.json")
    gaps = [x["gen"] - x["rc"] for x in d["draws"]]
    print(f"\nK13-2's per-seed gaps (x_GV - x_RC): {min(gaps):+.3f} to {max(gaps):+.3f}; void {d['void']}; "
          f"{d['sampled_valid']:,} of {d['urls']:,} real-context samples valid")
    vt = load(E13D / "VERIFY_TRAIN.json")
    cs = vt["context_swap"]
    print(f"VERIFY_TRAIN: ContextSwap matched {cs['matched_rows']:,} of {cs['listed_rows']:,}; swapped "
          f"{cs['swapped']:,} + {cs['too_long']} too long of {cs['eligible_draws']:,} eligible draws: "
          f"{vt['swap_share']:.4f} (band +- {vt['band']:.4f})")
    for step in ("DECODE", "GENERATE"):
        ident = load(E13D / f"VERIFY_{step}.json")["identity"]
        print(f"VERIFY_{step} byte identity: " + "; ".join(f"{k} {v['identical']}/{v['compared']}"
                                                          for k, v in ident.items()))

    print("\n### N — N2a on the current recipe (registered reading, then the measurements)\n")
    nr = load(E13D / "n_readout.json")
    print(f"Reading: **{nr['reading']}** -> {nr['licenses']}; free phishing QRY: {nr['phishing']}\n")
    m = load(E13D / "n_measure.json")
    print("| cell | model | Q-VAL x | Q-TRAIN x | copy rate (natural) |\n|---|---|---|---|---|")
    for cellname, node in m.items():
        for k in ("e12_final", "A1", "A2"):
            a = node[k]
            print(f"| {cellname} | {k} | {cell(*acc(a['q_val']))} | {cell(*acc(a['q_train']))} | "
                  f"{a['m4']['rate']:.4f} ({node['natural']['rate']:.4f}) |")
    print("\n| cell | reading | reference Q-VAL x / power floor | reference Q-TRAIN x / power floor | "
          "A2 closure (Q-VAL, Q-TRAIN) |\n|---|---|---|---|---|")
    for cellname, r in nr["by_cell"].items():
        cl = r["closure"]["A2"]
        ref = m[cellname]["e12_final"]

        def fl(k):
            a = ref[k]["acceptance"]
            floor = 4 * 2.776 * a["se"]           # N2a's power rule: x >= 2 x (2 x half-width)
            return f"{a['x']:.4f} / {floor:.4f} ({'usable' if a['x'] >= floor else 'VOID'})"
        print(f"| {cellname} | {r['reading']} | {fl('q_val')} | {fl('q_train')} | "
              f"{cl['q_val']:+.2f}, {cl['q_train']:+.2f} |")


if __name__ == "__main__":
    main()
