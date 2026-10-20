"""QC1 -- the registered readout (research/qc1/PREREG_QC1.md sections 3-5). Checks
validity first and stops if it fails; then applies each reading's rule, the
replication rule and the licence table; writes data/qc1/readout.json and prints
the tables QC1_RESULTS.md carries verbatim.

    python research/qc1/qc1_read.py > data/qc1/tables.md

Refuses dry-run files (data/qc1/dry is never read).
"""
from __future__ import annotations

import json
import statistics as st
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import qc1_common as C  # noqa: E402

V2_TOL = 0.05                 # Q3.1's kill criterion
NAME_TRANSFORM = "mask_name_chars"


ALLOW_DRY = False             # --allow-dry: a pipeline test on data/qc1/dry, stamped


def load(name: str) -> dict:
    d = json.loads((C.OUT / name).read_text(encoding="utf-8"))
    if d.get("dry") and not ALLOW_DRY:
        raise SystemExit(f"STOP: {name} is a dry run; it is never read")
    return d


def cell_items(d: dict) -> list:
    """The registered cells of a part's output, in registration order; the file's
    metadata keys (registration, part, dry, seconds) are not cells."""
    return [(name, d[name]) for name in (C.cell_name(*c) for c in C.CELLS) if name in d]


def valid_draws(draws: list) -> list:
    """V4: a seed whose floor is outside Q3's bounds is VOID for every condition."""
    return [d for d in draws if C.FLOOR_LOW <= d["floor"] <= C.FLOOR_HIGH]


def validity() -> list:
    from qc1_measure import sha256
    v = load("validity.json")
    fails = []
    if v["v1"]["verify"] != "PASS":
        fails.append("V1: data/e11_final/VERIFY.json is not PASS")
    for rel, h in v["v1"]["hashes"].items():
        if sha256(C.ROOT / rel) != h:
            fails.append(f"V1: {rel} changed since the run")
    for cell, r in v["v2"].items():
        d = st.mean(r["harness_x"]) - r["gate_x"]
        if abs(d) > V2_TOL:
            fails.append(f"V2: {cell} harness {st.mean(r['harness_x']):+.4f} against the gate "
                         f"{r['gate_x']:+.4f} (difference {d:+.4f})")
    # amendment 1: V5 (the probe's rows reproduce the data) and V2 extended (QC1-8)
    sl = load("slash.json")
    target = 1 - json.loads((C.ROOT / "data" / "e11" / "targets.json").read_text(
        encoding="utf-8"))["benign"]["slash_P0Q1"]["control"]
    for arm, node in sl["arms"].items():
        if sum(node["phishing_P0Q0"]["data"]):
            fails.append(f"V5: {arm} phishing bare-host rows include a real no-slash row")
        d = node["benign_P0Q1"]["data"]
        lo, hi = C.wilson(sum(d), len(d))
        if not lo <= target <= hi:
            fails.append(f"V5: {arm} benign P0Q1 rows' no-slash share {sum(d) / len(d):.5f} "
                         f"[{lo:.5f}, {hi:.5f}] excludes the target {target:.5f}")
    for field, r in load("state.json")["arms"][C.PRIMARY].items():
        d = st.mean(r["v2"]["harness_x"]) - r["v2"]["gate_x"]
        if abs(d) > V2_TOL:
            fails.append(f"V2: benign P1Q1 {field} harness against the gate differs by {d:+.4f}")
    sizing = json.loads(C.SIZING.read_text(encoding="utf-8"))["classes"]
    for label, r in v["v3"].items():
        want = sizing[label]["natural_copy_rate"]
        if abs(r["natural"] - want) > 1e-12:
            fails.append(f"V3: {label} natural {r['natural']} against sizing {want}")
        if "domain_disjoint" in r and abs(r["domain_disjoint"] -
                                          sizing[label]["domain_disjoint_copy_rate"]) > 1e-12:
            fails.append(f"V3: {label} domain-disjoint rate differs from sizing")
    return fails


def pct(x, d=2):
    return "—" if x is None else f"{100 * x:.{d}f}%"


def main() -> None:
    import argparse
    global ALLOW_DRY
    ap = argparse.ArgumentParser()
    ap.add_argument("--allow-dry", action="store_true",
                    help="test the reader on data/qc1/dry; the output is stamped and never a result")
    ALLOW_DRY = ap.parse_args().allow_dry
    if ALLOW_DRY:
        C.OUT = C.OUT / "dry"
    sys.stdout.reconfigure(encoding="utf-8")
    if ALLOW_DRY:
        print("**DRY RUN: a pipeline test at toy size. NOT A RESULT; never read.**\n")
    fails = validity()
    print(f"### Validity\n\nVerdict: **{'PASS' if not fails else 'FAIL'}**; failures: {fails or 'none'}\n")
    v = load("validity.json")
    print("| cell | the gate's x | the harness at the gate's n (mean of 12) | difference |\n|---|---|---|---|")
    for cell, r in v["v2"].items():
        print(f"| {cell} | {r['gate_x']:+.4f} | {st.mean(r['harness_x']):+.4f} | "
              f"{st.mean(r['harness_x']) - r['gate_x']:+.4f} |")
    if fails and not ALLOW_DRY:
        (C.OUT / "readout.json").write_text(json.dumps({"validity": fails}, indent=1), encoding="utf-8")
        raise SystemExit("STOP: validity failed; nothing is read")
    out = {"validity": "PASS" if not fails else fails, "dry": ALLOW_DRY, "readings": {}, "licences": {}}

    # QC1-1
    cp = load("copy.json")
    print("\n### QC1-1 — copying against the natural rate\n\n"
          "| cell | arm | generated | natural (T) | reading | strings seen once: generated / natural | reading |\n"
          "|---|---|---|---|---|---|---|")
    q1 = {}
    for cell, node in cell_items(cp):
        refs = [("host", node["real"])] + ([("domain-disjoint", node["real_domain_disjoint"])]
                                           if "real_domain_disjoint" in node else [])
        for arm in C.ARMS:
            g, n = node[arm]["generated"], node[arm]["per_generation"]
            for ref, real in refs:
                rr, t = C.rate_reading(g["rate"], real["rate"], n)
                sr, _ = C.rate_reading(g["singleton"], real["singleton"], n)
                q1[f"{cell} {arm} {ref}"] = {"rate": rr, "singleton": sr}
                print(f"| {cell}{'' if ref == 'host' else ' (domain-disjoint)'} | {arm} | {pct(g['rate'])} | "
                      f"{pct(real['rate'])} (±{100 * t:.2f}) | {rr} | {pct(g['singleton'])} / "
                      f"{pct(real['singleton'])} | {sr} |")
    out["readings"]["QC1-1"] = q1
    print("\nThe band profile (share of queries matching training strings seen 1 / 2–4 / 5–19 / 20+ times):\n")
    for cell, node in cell_items(cp):
        rb = node["real"]["bands"]
        print(f"- {cell}: real " + " / ".join(pct(rb[b], 1) for b in C.BANDS) + "; " + "; ".join(
            f"{arm} " + " / ".join(pct(node[arm]['generated']['bands'][b], 1) for b in C.BANDS)
            for arm in C.ARMS))

    # QC1-2
    print("\n### QC1-2 — the mix, or the content\n\n"
          "| cell | arm | n | x plain | x at the natural share | rho | t | reading | replicated |\n"
          "|---|---|---|---|---|---|---|---|---|")
    q2, mixled = {}, []
    for (_, label, state) in C.CELLS:
        cell = C.cell_name(_, label, state)
        rows = {}
        for arm in C.ARMS:
            d = load(f"mix_{arm}_{cell}.json")
            draws = valid_draws(d["draws"])
            r = C.mix_reading([x["plain"] for x in draws], [x["mix"] for x in draws])
            r.update({"n": d["n"], "void": len(d["draws"]) - len(draws),
                      "x_plain": st.mean(x["plain"] for x in draws),
                      "x_mix": st.mean(x["mix"] for x in draws)})
            if "mix_domain_disjoint" in d["draws"][0]:
                r["x_mix_domain_disjoint"] = st.mean(x["mix_domain_disjoint"] for x in draws)
            for s in ("never_seen", "found"):
                sd = d.get(s, {})
                r[f"x_{s}"] = (None if sd.get("insufficient") else
                               st.mean(x["x"] for x in valid_draws(sd["draws"])))
            rows[arm] = r
        rep = rows[C.PRIMARY]["reading"] == rows[C.REPLICATION]["reading"]
        q2[cell] = {"arms": rows, "replicated": rep}
        if rep and rows[C.PRIMARY]["reading"] == "MIX-LED":
            mixled.append(cell)
        for arm, r in rows.items():
            print(f"| {cell} | {arm} | {r['n']} | {r['x_plain']:+.4f} | {r['x_mix']:+.4f} | "
                  f"{r['rho']:.2f} | {r['t']:+.2f} | {r['reading']} | {'yes' if rep else '**no**'} |")
    out["readings"]["QC1-2"] = q2
    print("\nStrata (x within never-seen and within found strings), and phishing at the "
          "domain-disjoint share:\n")
    for cell, node in q2.items():
        for arm, r in node["arms"].items():
            extra = (f"; x at the domain-disjoint share {r['x_mix_domain_disjoint']:+.4f}"
                     if "x_mix_domain_disjoint" in r else "")
            ns = "insufficient" if r["x_never_seen"] is None else f"{r['x_never_seen']:+.4f}"
            fd = "insufficient" if r["x_found"] is None else f"{r['x_found']:+.4f}"
            print(f"- {cell} {arm}: never-seen {ns}; found {fd}{extra}")

    # QC1-3
    print("\n### QC1-3 — the ladder (Q3.1's, verbatim)\n")
    q3, name_identified = {}, False
    for label in ("benign", "phishing"):
        d = load(f"ladder_{label}.json")
        s1, s2 = d["stages"]["stage1"], d["stages"].get("stage2", {})
        rep = {n: (s1[n], s2[n]) for n in d["hits"] if n in s2
               and s1[n]["verdict"] == s2[n]["verdict"] and s1[n]["verdict"] in ("REDUCES", "raises")}
        reduces = {n: r for n, r in rep.items() if r[0]["verdict"] == "REDUCES"}
        identified = [n for n, (a, b) in reduces.items() if a["R_T"] <= 0.5 and b["R_T"] <= 0.5]
        reading = ("IDENTIFIED" if identified else "DISTRIBUTED" if len(reduces) >= 2 else
                   "NEITHER")
        sk = s1.get("skeleton", {})
        shape = sk.get("R_T") is not None and sk["R_T"] >= 0.85 and not sk["verdict"].startswith("VOID")
        q3[label] = {"reading": reading, "identified": identified, "replicated": sorted(rep),
                     "shape_sufficient": shape, "skeleton_R_T": sk.get("R_T")}
        name_identified |= NAME_TRANSFORM in identified
        print(f"**{label}:** {reading}; replicated: {sorted(rep) or 'none'}; skeleton R_T "
              f"{sk.get('R_T')} ({'SHAPE-SUFFICIENT' if shape else 'not shape-sufficient'})\n")
        print("| transform | stage 1 R_T | t | verdict | stage 2 R_T | t | verdict |\n|---|---|---|---|---|---|---|")
        for n, r in s1.items():
            r2 = s2.get(n, {})
            print(f"| {n} | {r.get('R_T')} | {r.get('t')} | {r['verdict']} | {r2.get('R_T', '—')} | "
                  f"{r2.get('t', '—')} | {r2.get('verdict', '—')} |")
        print()
    out["readings"]["QC1-3"] = q3

    # QC1-4
    nr = load("near.json")
    print("### QC1-4 — near misses (one edit from a training string seen 5+ times)\n\n"
          "| cell | arm | generated never-seen, near | real never-seen, near | difference [95% CI] | reading | replicated |\n"
          "|---|---|---|---|---|---|---|")
    q4, blending = {}, []
    for cell, node in cell_items(nr):
        rows = {arm: C.blending_reading(node[arm]["near"], node[arm]["n"], node["real"]["near"],
                                        node["real"]["n"]) for arm in C.ARMS}
        rep = all(r["reading"] == "BLENDING" for r in rows.values())
        q4[cell] = {"arms": rows, "replicated": rep}
        if rep:
            blending.append(cell)
        for arm, r in rows.items():
            print(f"| {cell} | {arm} | {pct(r['m_g'])} of {node[arm]['n']:,} | {pct(r['m_r'])} of "
                  f"{node['real']['n']:,} | {100 * r['diff']:+.2f} [{100 * r['lo']:+.2f}, {100 * r['hi']:+.2f}] | "
                  f"{r['reading']} | {'yes' if rep else 'no'} |")
    out["readings"]["QC1-4"] = q4

    # QC1-5
    print("\n### QC1-5 — phishing pulled toward benign\n\n"
          "| arm | n | x, real phishing against benign | x, generated phishing against benign | gap (t) | relative | reading |\n"
          "|---|---|---|---|---|---|---|")
    q5 = {}
    for arm in C.ARMS:
        d = load(f"leak_{arm}.json")
        draws = valid_draws(d["draws"])
        r = C.leak_reading([x["real"] for x in draws], [x["gen"] for x in draws])
        r.update({"x_real": st.mean(x["real"] for x in draws), "x_gen": st.mean(x["gen"] for x in draws),
                  "benign_only": d["benign_only"], "n": d["n"]})
        q5[arm] = r
        print(f"| {arm} | {d['n']} | {r['x_real']:+.4f} | {r['x_gen']:+.4f} | {r['gap']:+.4f} ({r['t']:+.2f}) | "
              f"{r['relative']:.2f} | {r['reading']} |")
    leak = all(r["reading"] == "LEAK" for r in q5.values())
    out["readings"]["QC1-5"] = {"arms": q5, "replicated": leak}
    print("\nBenign-only training strings among phishing queries: " + "; ".join(
        f"{arm} generated {pct(r['benign_only']['generated'])}, real {pct(r['benign_only']['real'])}"
        for arm, r in q5.items()))

    # QC1-6
    print("\n### QC1-6 — the length limit\n\n| class | real pool / fits | n | x all | x fits | S | t | reading |\n"
          "|---|---|---|---|---|---|---|---|")
    q6, budget = {}, []
    for label in ("benign", "phishing"):
        d = load(f"budget_{label}.json")
        draws = [x for x in d["draws"] if all(C.FLOOR_LOW <= x[k]["floor"] <= C.FLOOR_HIGH
                                               for k in ("all", "fit"))]
        r = C.budget_reading([x["all"]["x"] for x in draws], [x["fit"]["x"] for x in draws])
        q6[label] = r
        if r["reading"] == "BUDGET":
            budget.append(label)
        print(f"| {label} | {d['real_all']:,} / {d['real_fit']:,} | {d['n']} | "
              f"{st.mean(x['all']['x'] for x in draws):+.4f} | {st.mean(x['fit']['x'] for x in draws):+.4f} | "
              f"{r['S']:.2f} | {r['t']:+.2f} | {r['reading']} |")
    out["readings"]["QC1-6"] = q6

    # QC1-7 (amendment 1)
    sl = load("slash.json")
    print("\n### QC1-7 — the slash at the host end (teacher-forced; own class and swapped)\n\n"
          "| arm | rows | n | data no-slash | model, own class | model, other class | reading |\n"
          "|---|---|---|---|---|---|---|")
    q7 = {}
    for arm, node in sl["arms"].items():
        a = node["phishing_P0Q0"]
        ra = C.slash_leak_reading(a["own"], a["swap"])
        b = node["benign_P0Q1"]
        rb = C.suppressed_reading(b["own"], b["data"], b["swap"])
        c = node["benign_P0Q0"]
        q7[arm] = {"phishing_P0Q0": ra, "benign_P0Q1": rb,
                   "benign_P0Q0": {"data": st.mean(c["data"]), "own": st.mean(c["own"]),
                                   "swap": st.mean(c["swap"])}}
        print(f"| {arm} | phishing bare hosts | {a['n']:,} | {pct(st.mean(a['data']), 3)} | "
              f"{pct(ra['own'], 3)} | {pct(ra['swap'], 3)} | {ra['reading']} (rho {ra['rho']:.2f} "
              f"[{ra['lo']:.2f}, {ra['hi']:.2f}], lambda {ra['lambda']:.2f}) |")
        print(f"| {arm} | benign host-route queries | {b['n']:,} | {pct(rb['data'], 3)} | "
              f"{pct(rb['own'], 3)} | {pct(rb['swap'], 3)} | {rb['reading']} (ratio {rb['ratio']:.2f} "
              f"[{rb['ratio_lo']:.2f}, {rb['ratio_hi']:.2f}]) |")
        print(f"| {arm} | benign bare hosts (reported) | {c['n']:,} | {pct(st.mean(c['data']), 3)} | "
              f"{pct(st.mean(c['own']), 3)} | {pct(st.mean(c['swap']), 3)} | — |")
    slash_leak = all(q7[a]["phishing_P0Q0"]["reading"] == "SLASH-LEAK" for a in C.ARMS)
    sup_leak = all(q7[a]["benign_P0Q1"]["reading"] == "SUPPRESSED, LEAK-CONSISTENT" for a in C.ARMS)
    sup = all(q7[a]["benign_P0Q1"]["reading"].startswith("SUPPRESSED") for a in C.ARMS)
    out["readings"]["QC1-7"] = {"arms": q7, "slash_leak_replicated": slash_leak,
                                "suppressed_replicated": sup, "leak_consistent_replicated": sup_leak}

    # QC1-8 (amendment 1)
    stt = load("state.json")
    print("\n### QC1-8 — benign P1Q1 paths and subdomains: state blindness\n\n"
          "| field | arm | n | x own state | x other state (P1Q0) | x between real states | Welch t | reading |\n"
          "|---|---|---|---|---|---|---|---|")
    q8 = {}
    for field in ("PTH", "SUB"):
        rows = {}
        for arm, node in stt["arms"].items():
            r = node[field]
            draws = [d for d in r["draws"] if all(C.FLOOR_LOW <= d[k]["floor"] <= C.FLOOR_HIGH
                                                  for k in ("own", "cross", "states"))]
            rr = C.state_blind_reading([d["own"]["x"] for d in draws], [d["cross"]["x"] for d in draws],
                                       [d["states"]["x"] for d in draws])
            rr.update({"n": r["n"], "void": len(r["draws"]) - len(draws)})
            rows[arm] = rr
            print(f"| {field} | {arm} | {r['n']} | {rr['x_own']:+.4f} | {rr['x_cross']:+.4f} | "
                  f"{rr['x_states']:+.4f} | {rr['t']:+.2f} | {rr['reading']} |")
        q8[field] = {"arms": rows, "replicated": all(r["reading"] == "STATE-BLIND" for r in rows.values())}
    out["readings"]["QC1-8"] = q8
    state_blind = [f for f, r in q8.items() if r["replicated"]]

    # QC1-9 (amendment 2)
    frame_leak = []
    if (C.OUT / "frame.json").exists():
        fr = load("frame.json")
        print("\n### QC1-9 — domain-frame queries pulled toward the page frame\n\n"
              "| cell | arm | n | x, real domain against page | x, generated against page | gap (t) | "
              "relative | reading | page-repeated strings: page / real / generated |\n"
              "|---|---|---|---|---|---|---|---|---|")
        q9 = {}
        for cell in (C.cell_name(*c) for c in C.CELLS):
            rows = {}
            for arm, node in fr["arms"].items():
                r = node[cell]
                sh = r["page_repeated"]
                shs = " / ".join(pct(sh[k], 1) for k in ("page", "real", "generated"))
                if r.get("insufficient"):
                    rows[arm] = {"reading": "INSUFFICIENT", "n": r["n"]}
                    print(f"| {cell} | {arm} | {r['n']} | — | — | — | — | INSUFFICIENT | {shs} |")
                    continue
                draws = valid_draws(r["draws"])
                rr = C.frame_leak_reading([d["real"] for d in draws], [d["gen"] for d in draws])
                rr.update({"n": r["n"], "x_real": st.mean(d["real"] for d in draws),
                           "x_gen": st.mean(d["gen"] for d in draws), "page_repeated": sh})
                rows[arm] = rr
                print(f"| {cell} | {arm} | {r['n']} | {rr['x_real']:+.4f} | {rr['x_gen']:+.4f} | "
                      f"{rr['gap']:+.4f} ({rr['t']:+.2f}) | {rr['relative']:.2f} | {rr['reading']} | {shs} |")
            rep = all(r["reading"] == "FRAME-LEAK" for r in rows.values())
            q9[cell] = {"arms": rows, "replicated": rep}
            if rep:
                frame_leak.append(cell)
        out["readings"]["QC1-9"] = q9

    # Licences (section 4, and amendment 1)
    lic = {"query subwords": {"licensed": bool(mixled or blending or name_identified),
                              "by": {"MIX-LED": mixled, "BLENDING": blending,
                                     "IDENTIFIED on parameter names": name_identified}},
           "the class at every position": {"licensed": leak or slash_leak or sup_leak,
                                           "by": {"LEAK": leak, "SLASH-LEAK": slash_leak,
                                                  "SUPPRESSED and LEAK-CONSISTENT": sup_leak}},
           "the plan at every position": {"licensed": bool(state_blind),
                                          "by": {"STATE-BLIND": state_blind}},
           "the frame at every position": {"licensed": bool(frame_leak),
                                           "by": {"FRAME-LEAK": frame_leak}},
           "R1 (max_len 384)": {"licensed": bool(budget), "by": {"BUDGET": budget}}}
    out["licences"] = lic
    print("\n### Licences for E12 (section 4)\n\n| lever family | licensed | by |\n|---|---|---|")
    for k, r in lic.items():
        print(f"| {k} | **{'yes' if r['licensed'] else 'no'}** | {r['by']} |")
    (C.OUT / "readout.json").write_text(json.dumps(out, indent=1, default=float), encoding="utf-8")


if __name__ == "__main__":
    main()
