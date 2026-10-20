"""E17 -- the pre-training gates P1-P8 on stream 2 (research/e17/PREREG_E17.md section 3.1).
Laptop only; nothing is trained. E15's gates (research/e15/e15_gates.py, PREREG_E15.md
section 9.5) with one change: the reference is the old corpus PLUS stream 1, which
e16_cap trained on.

    python research/e17/e17_gates.py        # writes data/e17/gates.json

Stream 2 serves the same cells as stream 1: benign P0Q1 -> control_benign_P0Q1_QRY,
benign P1Q1 -> control_benign_P1Q1_QRY.
"""
from __future__ import annotations

import collections
import json
import math
import random
import statistics as st
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
for p in (ROOT, HERE, ROOT / "research" / "e15", ROOT / "research" / "e13", ROOT / "research" / "native",
          ROOT / "research" / "cr1", ROOT / "research" / "qc1", ROOT / "research" / "dc1"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import e13d_common as D  # noqa: E402
import e15_acquire as A  # noqa: E402
from e15_gates import (BATCH, C_MASS, CLASS_ROWS, DIVERSITY_DRAWS, DIVERSITY_SEED, E13D,  # noqa: E402
                       E14_RESULT, LABEL_COUNTS, old_rows, qry, suffix_of, tv)

C = D.C
S1 = ROOT / "data" / "e15" / "qry_extra"
S2 = ROOT / "data" / "e17" / "qry_extra2"
OUT_JSON = ROOT / "data" / "e17" / "gates.json"
STRATA = ("P0Q1", "P1Q1")
SERVES = {"benign_P0Q1": "control_benign_P0Q1_QRY", "benign_P1Q1": "control_benign_P1Q1_QRY"}
#: stream 1's registered files (PREREG_E15 Amendment 3) and stream 2's (ACQUISITION_E17A)
S1_SHA = {"benign_P0Q1": "4ef1b466f1190e2705b07cfcd691dd48649bb9069e796be2a99987d78c30b746",
          "benign_P1Q1": "ce66a7db77b4fabb079a0500889c099e0daabf7f18c484461b7e70c7e77ee8cd"}
S2_SHA = {"benign_P0Q1": "23c64419096a04d4a35042094320ee57ae9392d547e6a3a1fd3168f0e069b698",
          "benign_P1Q1": "a153e75eefd348b107d3a89e32a1292df12674aaa73510bba4a90c9bf7a6b1de"}
RATE = 2.0                           # section 3.2: two new rows per old stratum draw


def rows_of(folder: Path, key: str, want: str) -> list:
    path = folder / f"{key}.jsonl"
    if A.sha256(path) != want:
        raise SystemExit(f"STOP: {path} is not the registered file")
    return [json.loads(line) for line in open(path, encoding="utf-8")]


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    from uie.splits import group_key, group_key_host, load_split
    from dc1_compare import training_queries
    targets = json.loads(A.TARGETS.read_text(encoding="utf-8"))["strata"]
    s1 = {k: rows_of(S1, k, S1_SHA[k]) for k in SERVES}
    s2 = {k: rows_of(S2, k, S2_SHA[k]) for k in SERVES}
    res = {"registration": "research/e17/PREREG_E17.md section 3.1", "strata": {}, "cells": {}}

    # --- P3 and P7: leakage, stream-1 overlap and contamination --------------------------
    split_b = load_split(A.SPLIT_BENIGN)
    feed = json.loads((S1 / "feed_index.json").read_text(encoding="utf-8"))
    feed_hosts, feed_domains = set(feed["hosts"]), set(feed["domains"])
    s1_domains = {group_key(r["url"]) for rows in s1.values() for r in rows}
    all_new = {r["url"] for rows in s2.values() for r in rows}
    exact = set()
    for stem in A.TRAIN_VAL:
        for row in A.rows_of(A.BUNDLE / f"{stem}.jsonl.gz"):
            if row["url"] in all_new:
                exact.add(row["url"])
    for key, rows in s2.items():
        leak = sum(split_b.partition_of(r["url"]) is not None for r in rows)
        in_s1 = sum(group_key(r["url"]) in s1_domains for r in rows)
        contam = sum(group_key(r["url"]) in feed_domains or group_key_host(r["url"]) in feed_hosts for r in rows)
        units = len({group_key(r["url"]) for r in rows})
        ex = sum(r["url"] in exact for r in rows)
        res["strata"][key] = {
            "N_new2": len(rows), "N_new1": len(s1[key]), "N_old": targets[key]["N_old"],
            "L_old": targets[key]["L_old"],
            "P3": {"split_key_overlap": leak, "exact_url_in_train_or_val": ex, "stream1_domain_overlap": in_s1,
                   "one_per_unit": units == len(rows),
                   "pass": leak == 0 and ex == 0 and in_s1 == 0 and units == len(rows)},
            "P7": {"in_phishing_feed": contam, "pass": contam == 0}}
        print(f"{key}: N_new2 {len(rows):,}; P3 leak {leak} exact {ex} stream-1 {in_s1} one-per-unit "
              f"{units == len(rows)}; P7 {contam}", flush=True)

    # --- P4: suffix and crawl mix against the stratum's own train-vs-validation spread ---
    for key, rows in s2.items():
        label, state = key.split("_")
        tr, va = old_rows(label, "train", state), old_rows(label, "val", state)
        sfx = {n: collections.Counter(suffix_of(r["url"]) for r in rs) for n, rs in
               (("new", rows), ("train", tr), ("val", va))}
        cr = {n: collections.Counter(r.get("crawl") for r in rs) for n, rs in
              (("new", rows), ("train", tr), ("val", va))}
        node = {"suffix_tv_new_train": tv(sfx["new"], sfx["train"]),
                "suffix_tv_ref_train_val": tv(sfx["train"], sfx["val"]),
                "crawl_tv_new_train": tv(cr["new"], cr["train"]),
                "crawl_tv_ref_train_val": tv(cr["train"], cr["val"])}
        node["pass"] = bool(node["suffix_tv_new_train"] <= node["suffix_tv_ref_train_val"]
                            and node["crawl_tv_new_train"] <= node["crawl_tv_ref_train_val"])
        res["strata"][key]["P4"] = node
        print(f"{key}: P4 {node}", flush=True)

    # --- P1, P2, P5, P6: per gate cell, against old + stream 1 -------------------------
    known = training_queries(A.BUNDLE, "benign") | {q for rows in s1.values() for q in (qry(r["url"]) for r in rows) if q}
    for key, cell_name in SERVES.items():
        cell = D.BY_NAME[cell_name]
        new_q = [qry(r["url"]) for r in s2[key]]
        s1_q = [qry(r["url"]) for r in s1[key]]
        val, old = D.pool(cell, "val"), D.pool(cell, "train")
        node = {"N_new_strings": len(new_q), "val": len(val), "old_twin": len(old), "stream1": len(s1_q)}
        decomp = json.loads((E13D / f"decomp_{cell_name}.json").read_text(encoding="utf-8"))
        x_gv = [d[D.PRIMARY] for d in decomp["against_val"]]
        n = min(C.N_REQ, len(val) // 3, len(new_q) // 3)
        x_nv = [C.harness(val, {"new": random.Random(s + 303).sample(new_q, n)}, s, n)["new"] for s in D.SEEDS]
        r1 = D.shift_reading(x_nv, x_gv)
        node["P1"] = {**r1, "n": n, "pass": r1["reading"] == "NO SHIFT"}
        vt = [D.template(q) for q in val]
        have_ref = {D.template(q) for q in old} | {D.template(q) for q in s1_q}
        have_all = have_ref | {D.template(q) for q in new_q}
        cov_ref = sum(t in have_ref for t in vt) / len(vt)
        cov_all = sum(t in have_all for t in vt) / len(vt)
        quant = json.loads((E13D / f"quantity_{cell_name}.json").read_text(encoding="utf-8"))
        g = quant["coverage"]["1.0"] - quant["coverage"]["0.5"]
        r = len(new_q) / (targets[key]["N_old"] + len(s1_q))
        thr = g * math.log2(1 + r) - 0.01
        node["P2"] = {"cov_old_plus_stream1": cov_ref, "cov_with_stream2": cov_all, "gain": cov_all - cov_ref,
                      "g_dx3": g, "r": r, "threshold": thr, "pass": (cov_all - cov_ref) >= thr}
        k = min(len(new_q), len(old))
        ratios = []
        for d in range(DIVERSITY_DRAWS):
            rng = random.Random(DIVERSITY_SEED + d)
            ratios.append(len({D.template(x) for x in rng.sample(new_q, k)}) /
                          len({D.template(x) for x in rng.sample(old, k)}))
        top = {nm: sum(c for _, c in collections.Counter(D.template(x) for x in xs).most_common(10)) / len(xs)
               for nm, xs in (("new", new_q), ("old", old))}
        node["P5"] = {"k": k, "distinct_ratio_median": st.median(ratios), "top10_new": top["new"],
                      "top10_old": top["old"], "pass": st.median(ratios) >= 0.9 and top["new"] <= top["old"] + 0.05}
        never_new = sum(x not in known for x in new_q) / len(new_q)
        never_val = sum(x not in known for x in val) / len(val)
        node["P6"] = {"never_seen_new": never_new, "never_seen_val": never_val,
                      "new_template_share": sum(D.template(x) not in have_ref for x in new_q) / len(new_q),
                      "pass": never_new >= never_val - 0.05}
        res["cells"][cell_name] = node
        print(f"{cell_name}: P1 {r1['reading']} (x_NV {r1['x_tv']:+.4f}, n {n}); P2 gain {cov_all - cov_ref:+.4f} "
              f"vs {thr:+.4f}; P5 {node['P5']['distinct_ratio_median']:.3f} top10 {top['new']:.3f}/{top['old']:.3f}; "
              f"P6 {never_new:.3f} vs {never_val:.3f}", flush=True)

    # --- admission and P8 ---------------------------------------------------------------
    e14 = json.loads(E14_RESULT.read_text(encoding="utf-8"))["query_weight"]
    q_batch = e14["query_share"] * C_MASS
    p_benign = LABEL_COUNTS["benign"] / sum(LABEL_COUNTS.values())
    p8 = {"batch": BATCH, "C": C_MASS, "query_positions_per_batch_e12_final": q_batch, "rate": RATE, "strata": {}}
    appended = added = added1 = 0.0
    tok = A.tokenizer()
    for key in SERVES:
        s = res["strata"][key]
        cell = res["cells"][SERVES[key]]
        gates = [s["P3"]["pass"], s["P4"]["pass"], s["P7"]["pass"], cell["P1"]["pass"], cell["P2"]["pass"],
                 cell["P5"]["pass"], cell["P6"]["pass"]]
        admitted = all(g is True for g in gates) and s["N_new2"] >= 0.5 * s["N_old"]
        s["admitted"] = admitted
        l1 = sum(A.query_steps(tok, A.encoded(r["url"], "benign")) for r in s1[key]) / len(s1[key])
        l2 = sum(A.query_steps(tok, A.encoded(r["url"], "benign")) for r in s2[key]) / len(s2[key])
        n_s = BATCH * p_benign * s["N_old"] / CLASS_ROWS["benign"]
        rate = (s["N_new1"] + s["N_new2"]) / s["N_old"] if admitted else s["N_new1"] / s["N_old"]
        add_s = n_s * (l1 + (l2 if admitted else 0.0)) * (1.0 if rate <= 2 else rate / 2)
        p8["strata"][key] = {"admitted": admitted, "N_old": s["N_old"], "N_new1": s["N_new1"], "N_new2": s["N_new2"],
                             "rate": rate, "old_rows_per_batch": n_s, "appended_per_batch": rate * n_s,
                             "L_old": s["L_old"], "L_new1": l1, "L_new2": l2,
                             "added_query_positions_per_batch": add_s,
                             "added_by_stream1_per_batch": n_s * l1}
        appended += rate * n_s
        added += add_s
        added1 += n_s * l1
    p8.update({"appended_rows_per_batch": appended, "added_query_positions_per_batch": added,
               "query_mass_increase_vs_e12_final": added / q_batch,
               "query_mass_increase_vs_e16_cap": (q_batch + added) / (q_batch + added1) - 1,
               "scored_mass_increase": added / C_MASS,
               "stream_files": {"stream1": S1_SHA, "stream2": S2_SHA},
               "seeds": {"inject_draw": 20261007, "inject_fraction": 20261008}})
    res["P8"] = p8
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(res, indent=1, default=float) + "\n", encoding="utf-8")
    print(json.dumps({k: {"admitted": v["admitted"], "rate": round(v["rate"], 4),
                          "appended": round(v["appended_per_batch"], 3)} for k, v in p8["strata"].items()}))
    print(f"appended rows/batch {appended:.3f}; added query positions/batch {added:.1f} "
          f"(query mass +{100 * added / q_batch:.1f}% vs e12_final, "
          f"+{100 * p8['query_mass_increase_vs_e16_cap']:.1f}% vs e16_cap)")
    print("wrote", OUT_JSON)


if __name__ == "__main__":
    main()
