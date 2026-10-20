"""E14 -- render the registered readout as tables (research/e14/PREREG_E14.md section 3).
Reads data/e14/compare_report.json, the merged score report, k14_2.json, k14_3.json,
the n2a_* files and the VERIFY files; prints markdown.

    python research/e14/e14_tables.py > data/e14/tables.md
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
for p in (ROOT, HERE, ROOT / "research" / "e13", ROOT / "research" / "native",
          ROOT / "research" / "cr1"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from e13_tables import acc, cell, slash_attribution  # noqa: E402
from e14_compare import merged  # noqa: E402

E14D = ROOT / "data" / "e14"
ARMS = ("e12_final", "e14_q", "e12_d1p", "e14_qd")
GEN = {"e12_final": ROOT / "data" / "e12" / "gen", "e14_q": E14D / "generate" / "gen",
       "e12_d1p": E14D / "decode" / "gen", "e14_qd": E14D / "generate" / "gen"}


def load(p):
    return json.loads(Path(p).read_text(encoding="utf-8"))


def verdicts(node):
    return " ".join(f"{k} {'PASS' if v['pass'] else '**FAIL**'}" for k, v in node.items()
                    if isinstance(v, dict) and "pass" in v)


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    rep, cmp_ = merged(), load(E14D / "compare_report.json")
    print("### Validity\n\nVERIFY_TRAIN, VERIFY_DECODE, VERIFY_GENERATE: "
          + ", ".join(load(E14D / f"VERIFY_{s}.json")["verdict"] for s in ("TRAIN", "DECODE", "GENERATE"))
          + f"; V4: max |x - published| {load(E14D / 'score' / 'v4.json')['max_abs_diff']}")
    vt, vd = load(E14D / "VERIFY_TRAIN.json"), load(E14D / "VERIFY_DECODE.json")
    q = vt["query_weight"]
    print(f"\nV5: band steps on the w = 1 mass {vt['band_steps']}; query weight {q['w']}, query share "
          f"of the scored positions drawn {q['query_share']:.4f}")
    v7 = vd["v7"]
    print(f"V7: P0-plan URLs identical to e12_d1's {v7['p0'][0]:,} of {v7['p0'][1]:,}; P1-plan URLs "
          f"re-drawn by D1' {v7['p1_changed'][0]:,} of {v7['p1_changed'][1]:,} (descriptive)")
    for step in ("DECODE", "GENERATE"):
        ident = load(E14D / f"VERIFY_{step}.json")["identity"]
        print(f"VERIFY_{step} byte identity: " + "; ".join(f"{k} {v['identical']}/{v['compared']}"
                                                          for k, v in ident.items()))

    print("\n### The criteria\n\n| lever | criteria | adopted |\n|---|---|---|")
    for name, key in (("Q (e14_q)", "Q"), ("D1' (e12_d1p)", "D1'"), ("combination (e14_qd)", "combination")):
        node = cmp_[key]
        print(f"| {name} | {verdicts(node)} | **{'yes' if node['adopted'] else 'no'}** |")
    print(f"\nAdopted: {cmp_['adopted']}")

    print("\n### K14-1 -- the target query cells (e14_q against e12_final)\n")
    print("| cell | e12_final | e14_q | fall | moves (> 0.0141) |\n|---|---|---|---|---|")
    for name, c in cmp_["Q"]["K14-1"]["cells"].items():
        print(f"| {name} | {c['x_ref']:+.4f} {c['v_ref']} | {c['x_arm']:+.4f} {c['v_arm']} | "
              f"{c['fall']:+.4f} | {'yes' if c['moves'] else 'no'} |")

    k2 = load(E14D / "k14_2.json")
    print("\n### K14-2 -- held-out query fit (N2a's M3, nats per query character)\n")
    print("| class | split | rows | e12_final | e14_q | difference [95% CI] | falls |\n"
          "|---|---|---|---|---|---|---|")
    for label in ("benign", "phishing"):
        for split in ("validation", "train"):
            n = k2[label][split]
            print(f"| {label} | {split} | {n['rows']:,} | {n['nll_per_char']['e12_final']:.4f} | "
                  f"{n['nll_per_char']['e14_q']:.4f} | {n['diff']:+.4f} [{n['ci'][0]:+.4f}, "
                  f"{n['ci'][1]:+.4f}] | {'yes' if n['falls'] else 'no'} |")
        g = k2[label]["gap"]
        print(f"| {label} | gap (val - train) | | {g['e12_final']:+.4f} | {g['e14_q']:+.4f} | | |")

    k3 = load(E14D / "k14_3.json")
    print("\n### K14-3 -- the band profile (N2a's M2: rare-band share of generated names)\n")
    print("| class | e12_final | e14_q [95% CI] | null | overshoot |\n|---|---|---|---|---|")
    for label in ("benign", "phishing"):
        base = load(E14D / f"n2a_e12_final_{label}.json")["M2"]
        n = k3[label]
        print(f"| {label} | {base['observed_rare']:.4f} | {n['observed_rare']:.4f} "
              f"[{n['observed_ci'][0]:.4f}, {n['observed_ci'][1]:.4f}] | {n['null_rare']:.4f} | "
              f"{'**yes**' if n['overshoot'] else 'no'} |")
    print("\nN2a's M1 (value-termination hazard, positions 25-100, 1.0 ideal) and M4 (query copying):\n")
    print("| class | model | M1 train | M1 validation | M4 |\n|---|---|---|---|---|")
    for label in ("benign", "phishing"):
        for m in ("e12_final", "e14_q"):
            n = load(E14D / f"n2a_{m}_{label}.json")
            t, v = n["train"]["M1_long"], n["validation"]["M1_long"]
            print(f"| {label} | {m} | {t['ratio']:.3f} [{t['lo']:.3f}, {t['hi']:.3f}] | "
                  f"{v['ratio']:.3f} [{v['lo']:.3f}, {v['hi']:.3f}] | {n['M4']['rate']:.4f} |")

    print("\n### The slash rows (K12b; T as registered)\n\n| row | real | " + " | ".join(ARMS)
          + " |\n|---|---|" + "---|" * len(ARMS))
    from e12_compare import k12b
    targets = load(ROOT / "data" / "e11" / "targets.json")
    rows = {a: k12b(rep, targets, a)["rows"] for a in ARMS}
    for k, r0 in rows["e12_final"].items():
        def f(r):
            return f"{100 * r['q']:.3f}%{'' if r['pass'] else ' **miss**'}"
        print(f"| {k} | {100 * r0['p']:.3f}% (±{100 * r0['T']:.3f}) | "
              + " | ".join(f(rows[a][k]) for a in ARMS) + " |")

    print("\n### The slash misses, by supplied state (phishing free bare hosts; S from the seeds)\n")
    print("| arm | bare hosts | without the slash | by supplied state |\n|---|---|---|---|")
    for arm in ARMS:
        bare, miss = slash_attribution(arm, GEN[arm])
        print(f"| {arm} | {bare:,} | {sum(miss.values())} | {dict(sorted(miss.items()))} |")

    print("\n### Gated cells that are not CLEARED in some arm, and FULL\n\n| cell | "
          + " | ".join(ARMS) + " |\n|---|" + "---|" * len(ARMS))
    for label in ("benign", "phishing"):
        for f in ("LBL", "SUF", "SUB", "PTH", "QRY", "FULL"):
            cs = [acc(rep["free"][a][label]["fields"].get(f)) for a in ARMS]
            if f == "FULL" or any(v != "CLEARED" for _, v in cs):
                print(f"| free {label} {f} | " + " | ".join(cell(*x) for x in cs) + " |")
    for key in rep["control"]["e12_final"]:
        for f in ("LBL", "SUF", "SUB", "PTH", "QRY", "FULL"):
            cs = [acc(rep["control"][a][key]["fields"].get(f)) for a in ARMS]
            if cs[0][0] is None:
                continue
            if f == "FULL" or any(v != "CLEARED" for _, v in cs):
                print(f"| control {key} {f} | " + " | ".join(cell(*x) for x in cs) + " |")

    print("\n### Copying against the natural rate (QC1-1's cells)\n")
    print("| cell | natural | " + " | ".join(ARMS) + " |\n|---|---|" + "---|" * len(ARMS))
    for c, node in rep["copy"].items():
        vals = [f"{node[a]['generated']['rate']:.4f}" if a in node else "—" for a in ARMS]
        print(f"| {c} | {node['real']['rate']:.4f} | " + " | ".join(vals) + " |")

    print("\n### Honouring (control states, P and Q) and S honouring\n")
    print("| population | " + " | ".join(ARMS) + " |\n|---|" + "---|" * len(ARMS))
    for key in rep["honouring"]["e12_final"]:
        vals = [rep["honouring"].get(a, {}).get(key) for a in ARMS]
        print(f"| PQ {key} | " + " | ".join(f"{100 * v['rate']:.2f}%" if v else "—" for v in vals) + " |")
    for key in rep["s_honouring"]["e12_final"]:
        for sv in rep["s_honouring"]["e12_final"][key]:
            vals = [((rep["s_honouring"].get(a) or {}).get(key) or {}).get(sv) for a in ARMS]
            print(f"| S {key} {sv} | " + " | ".join(f"{100 * v['rate']:.3f}% of {v['n']:,}" if v else "—"
                                                   for v in vals) + " |")

    print("\n### E3 (section 3.4)\n\n| arm | levers | E3 | cells not CLEARED | slash rows missed |\n"
          "|---|---|---|---|---|")
    for arm, e in cmp_["E3"].items():
        cells_ = "; ".join(f"{c['cell']} {cell(c['x'], c['verdict'])}" for c in e["cells_not_cleared"])
        print(f"| {arm} | {', '.join(e['levers'])} | **{e['E3']}** | {cells_ or '—'} | "
              f"{', '.join(e['slash_rows_missed']) or '—'} |")


if __name__ == "__main__":
    main()
