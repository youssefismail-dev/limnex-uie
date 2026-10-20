"""E15-B -- the pre-training gates P1-P8 (research/e15/PREREG_E15.md sections 9.5, 10.4, 11).
Laptop only: reads the acquired stream (data/e15/qry_extra/), the bundle, the splits, E13-D's
stored DX1/DX3 records and the targets; writes data/e15/gates.json. Nothing is trained.

    python research/e15/e15_gates.py

The cells each stratum serves (the gate's own, E13-D's definitions):
    benign P0Q1  -> control_benign_P0Q1_QRY        benign P1Q1 -> control_benign_P1Q1_QRY
    phishing (both strata) -> free_phishing_QRY
P1 is DX1's rule with the new rows as the twin (the same harness, seeds and n rule); P2 uses
DX3's template and coverage; P6 uses the project's training-query set (dc1_compare).
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
for p in (ROOT, HERE, ROOT / "research" / "e13", ROOT / "research" / "native",
          ROOT / "research" / "cr1", ROOT / "research" / "qc1", ROOT / "research" / "dc1"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import e13d_common as D  # noqa: E402
import e15_acquire as A  # noqa: E402

C = D.C
OUT_JSON = ROOT / "data" / "e15" / "gates.json"
E13D = ROOT / "data" / "e13d"
DIVERSITY_DRAWS, DIVERSITY_SEED = 20, 20261010
#: section 10.4: e12_final's scored mass per batch (C) and its query share (E14's record)
C_MASS = 14397.885481960007
E14_RESULT = ROOT / "data" / "e14" / "train" / "e14_q" / "train_result.json"
BATCH, LABEL_COUNTS = 256, {"benign": 906_214, "phishing": 192_652}
CLASS_ROWS = {"benign": 2_551_259, "phishing": 238_707}          # as encoded (E14's dry run)
#: stratum -> the gate cell it serves
SERVES = {"benign_P0Q1": "control_benign_P0Q1_QRY", "benign_P1Q1": "control_benign_P1Q1_QRY",
          "phishing_P0Q1": "free_phishing_QRY", "phishing_P1Q1": "free_phishing_QRY"}


def stream(name: str):
    path = A.OUT / f"{name}.jsonl"
    return [json.loads(line) for line in open(path, encoding="utf-8")] if path.is_file() else []


def qry(url: str) -> str:
    from uie.evaluation import FIELDS
    return FIELDS["QRY"](url)


def tv(a: collections.Counter, b: collections.Counter) -> float:
    na, nb = sum(a.values()), sum(b.values())
    return 0.5 * sum(abs(a[k] / na - b[k] / nb) for k in set(a) | set(b)) if na and nb else float("nan")


def suffix_of(url: str) -> str:
    from uie.factorization import parse
    return parse(url).public_suffix or "<none>"


def old_rows(label: str, split: str, state: str):
    pre = "" if label == "benign" else "phishing_"
    out = []
    for r in A.rows_of(A.BUNDLE / f"{pre}name_{split}.jsonl.gz"):
        if A.state_of(r["url"]) == state:
            out.append(r)
    return out


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    from uie.splits import group_key, group_key_host, load_split
    from dc1_compare import training_queries
    targets = json.loads(A.TARGETS.read_text(encoding="utf-8"))["strata"]
    man = json.loads((A.OUT / "manifest.json").read_text(encoding="utf-8"))
    new = {f"{c}_{s}": stream(f"{c}_{s}") for c in ("benign", "phishing") for s in A.STRATA}
    res = {"registration": "PREREG_E15 sections 9.5, 10.4, 11", "strata": {}, "cells": {}}

    # --- P3 and P7: leakage and contamination, re-checked independently of the acquisition ---
    split_b, split_p = load_split(A.SPLIT_BENIGN), load_split(A.SPLIT_PHISH)
    feed = json.loads((A.OUT / "feed_index.json").read_text(encoding="utf-8"))
    feed_hosts, feed_domains = set(feed["hosts"]), set(feed["domains"])
    all_new = {r["url"] for rows in new.values() for r in rows}
    exact = set()
    for stem in A.TRAIN_VAL:
        for row in A.rows_of(A.BUNDLE / f"{stem}.jsonl.gz"):
            if row["url"] in all_new:
                exact.add(row["url"])
    for key, rows in new.items():
        label = key.split("_")[0]
        if label == "benign":
            leak = sum(split_b.partition_of(r["url"]) is not None for r in rows)
            contam = sum(group_key(r["url"]) in feed_domains or group_key_host(r["url"]) in feed_hosts
                         for r in rows)
            units = len({group_key(r["url"]) for r in rows})
        else:
            leak = sum(split_p.partition_of(r["url"]) is not None for r in rows)
            contam = 0                      # section 11.2.2: the host key, and exact URLs below
            units = len({group_key_host(r["url"]) for r in rows})
        ex = sum(r["url"] in exact for r in rows)
        res["strata"][key] = {
            "N_new": len(rows), "N_old": targets[key]["N_old"], "L_old": targets[key]["L_old"],
            "P3": {"split_key_overlap": leak, "exact_url_in_train_or_val": ex,
                   "one_per_unit": units == len(rows), "pass": leak == 0 and ex == 0 and units == len(rows)},
            "P7": {"in_phishing_feed": contam, "pass": contam == 0}}
        print(f"{key}: N_new {len(rows):,} (N_old {targets[key]['N_old']:,}); P3 leak {leak} exact {ex} "
              f"one-per-unit {units == len(rows)}; P7 {contam}", flush=True)

    # --- P4: suffix (and crawl) mix against the stratum's own train-vs-validation spread ---------
    for key, rows in new.items():
        label, state = key.split("_")
        tr, va = old_rows(label, "train", state), old_rows(label, "val", state)
        sfx = {n: collections.Counter(suffix_of(r["url"]) for r in rs) for n, rs in
               (("new", rows), ("train", tr), ("val", va))}
        node = {"suffix_tv_new_train": tv(sfx["new"], sfx["train"]),
                "suffix_tv_ref_train_val": tv(sfx["train"], sfx["val"])}
        ok = node["suffix_tv_new_train"] <= node["suffix_tv_ref_train_val"]
        if label == "benign":
            cr = {n: collections.Counter(r.get("crawl") for r in rs) for n, rs in
                  (("new", rows), ("train", tr), ("val", va))}
            node.update({"crawl_tv_new_train": tv(cr["new"], cr["train"]),
                         "crawl_tv_ref_train_val": tv(cr["train"], cr["val"])})
            ok = ok and node["crawl_tv_new_train"] <= node["crawl_tv_ref_train_val"]
        node["pass"] = bool(ok) if rows else None
        res["strata"][key]["P4"] = node
        print(f"{key}: P4 {node}", flush=True)

    # --- P1, P2, P5, P6: per gate cell ----------------------------------------------------------
    by_cell = collections.defaultdict(list)
    for key, rows in new.items():
        by_cell[SERVES[key]].extend(rows)
    for cell_name, rows in by_cell.items():
        if not rows:
            print(f"{cell_name}: no new rows yet; skipped", flush=True)
            continue
        cell = D.BY_NAME[cell_name]
        label = cell[2]
        new_q = [qry(r["url"]) for r in rows]
        val, old = D.pool(cell, "val"), D.pool(cell, "train")
        node = {"N_new_strings": len(new_q), "val": len(val), "old_twin": len(old)}
        # P1: DX1 with the new rows as the twin
        decomp = json.loads((E13D / f"decomp_{cell_name}.json").read_text(encoding="utf-8"))
        x_gv = [d[D.PRIMARY] for d in decomp["against_val"]]
        n = min(C.N_REQ, len(val) // 3, len(new_q) // 3)
        x_nv = []
        for s in D.SEEDS:
            x_nv.append(C.harness(val, {"new": random.Random(s + 303).sample(new_q, n)}, s, n)["new"])
        r1 = D.shift_reading(x_nv, x_gv)
        node["P1"] = {**r1, "n": n, "n_dx1": decomp["n"], "x_tv_old_twin": st.mean(d["train"] for d in decomp["against_val"]),
                      "pass": r1["reading"] == "NO SHIFT"}
        # P2: coverage of the validation templates, old against old + new
        vt = [D.template(q) for q in val]
        have_old = {D.template(q) for q in old}
        have_both = have_old | {D.template(q) for q in new_q}
        cov_old = sum(t in have_old for t in vt) / len(vt)
        cov_both = sum(t in have_both for t in vt) / len(vt)
        q = json.loads((E13D / f"quantity_{cell_name}.json").read_text(encoding="utf-8"))
        g = q["coverage"]["1.0"] - q["coverage"]["0.5"]
        n_old = sum(targets[k]["N_old"] for k in SERVES if SERVES[k] == cell_name)
        r = len(rows) / n_old
        thr = g * math.log2(1 + r) - 0.01
        node["P2"] = {"cov_old": cov_old, "cov_old_plus_new": cov_old + (cov_both - cov_old),
                      "gain": cov_both - cov_old, "g_dx3": g, "r": r, "threshold": thr,
                      "binding": label == "benign",
                      "pass": (cov_both - cov_old) >= thr if label == "benign" else None}
        # P5: diversity at equal size
        k = min(len(new_q), len(old))
        ratios = []
        for d in range(DIVERSITY_DRAWS):
            rng = random.Random(DIVERSITY_SEED + d)
            dn = len({D.template(x) for x in rng.sample(new_q, k)})
            do = len({D.template(x) for x in rng.sample(old, k)})
            ratios.append(dn / do)
        top = {n_: sum(c for _, c in collections.Counter(D.template(x) for x in xs).most_common(10)) / len(xs)
               for n_, xs in (("new", new_q), ("old", old))}
        node["P5"] = {"k": k, "distinct_ratio_median": st.median(ratios), "top10_new": top["new"],
                      "top10_old": top["old"],
                      "pass": st.median(ratios) >= 0.9 and top["new"] <= top["old"] + 0.05}
        # P6: genuinely new strings, against the validation pool's own never-seen share
        known = training_queries(A.BUNDLE, label)
        never_new = sum(x not in known for x in new_q) / len(new_q)
        never_val = sum(x not in known for x in val) / len(val)
        new_tpl = sum(D.template(x) not in have_old for x in new_q) / len(new_q)
        node["P6"] = {"never_seen_new": never_new, "never_seen_val": never_val,
                      "new_template_share": new_tpl, "pass": never_new >= never_val - 0.05}
        res["cells"][cell_name] = node
        print(f"{cell_name}: P1 {node['P1']['reading']} (x_NV {node['P1']['x_tv']:+.4f}, n {n}); "
              f"P2 gain {node['P2']['gain']:+.4f} vs {thr:+.4f}; P5 {node['P5']['distinct_ratio_median']:.3f} "
              f"top10 {top['new']:.3f}/{top['old']:.3f}; P6 {never_new:.3f} vs {never_val:.3f}", flush=True)

    # --- admission and P8: the frozen numbers --------------------------------------------------
    e14 = json.loads(E14_RESULT.read_text(encoding="utf-8"))["query_weight"]
    q_batch = e14["query_share"] * C_MASS
    p_class = {c: LABEL_COUNTS[c] / sum(LABEL_COUNTS.values()) for c in LABEL_COUNTS}
    p8 = {"batch": BATCH, "C": C_MASS, "query_positions_per_batch_e12_final": q_batch, "strata": {}}
    appended = added = 0.0
    for key, rows in new.items():
        label, state = key.split("_")
        s = res["strata"][key]
        cell = res["cells"].get(SERVES[key], {})
        gates = [s["P3"]["pass"], s["P4"]["pass"], s["P7"]["pass"], cell.get("P1", {}).get("pass"),
                 cell.get("P5", {}).get("pass"), cell.get("P6", {}).get("pass")]
        if label == "benign":
            gates.append(cell.get("P2", {}).get("pass"))
            size_ok = s["N_new"] >= 0.5 * s["N_old"]
        else:
            size_ok = s["N_new"] >= 1                           # section 11.2.1
        admitted = bool(rows) and size_ok and all(g is True for g in gates)
        s["admitted"] = admitted
        l_new = (sum(A.query_steps(A.tokenizer(), A.encoded(r["url"], label)) for r in rows) / len(rows)
                 if rows else 0.0)
        n_s = BATCH * p_class[label] * s["N_old"] / CLASS_ROWS[label]
        p_s = min(1.0, s["N_new"] / s["N_old"]) if admitted else 0.0
        p8["strata"][key] = {"admitted": admitted, "N_old": s["N_old"], "N_new": s["N_new"],
                             "p_inject": p_s, "old_rows_per_batch": n_s, "appended_per_batch": p_s * n_s,
                             "L_old": s["L_old"], "L_new": l_new,
                             "added_query_positions_per_batch": p_s * n_s * l_new,
                             "stratum_query_mass_ratio": 1 + p_s * l_new / s["L_old"] if s["L_old"] else None}
        appended += p_s * n_s
        added += p_s * n_s * l_new
    p8.update({"appended_rows_per_batch": appended, "added_query_positions_per_batch": added,
               "query_mass_increase": added / q_batch, "scored_mass_increase": added / C_MASS,
               "e14_added_per_batch": 3 * q_batch, "dose_vs_e14": added / (3 * q_batch),
               "stream_files": {**{k: v for k, v in man.get("benign", {}).get("files", {}).items()},
                                **{f"phishing_{k}": v for k, v in man.get("phishing", {}).get("files", {}).items()}},
               "seeds": {"blocks": "20261005 + 7919*i + 1000003*(r-1)", "name": A.SEED_NAME,
                         "phishing_host_pick": A.SEED_PHISH, "inject_draw": 20261007,
                         "inject_fraction": 20261008}})
    res["P8"] = p8
    OUT_JSON.write_text(json.dumps(res, indent=1, default=float) + "\n", encoding="utf-8")
    print(json.dumps({k: {"admitted": v["admitted"], "p": round(v["p_inject"], 4),
                          "appended": round(v["appended_per_batch"], 3)} for k, v in p8["strata"].items()}))
    print(f"appended rows/batch {appended:.3f}; added query positions/batch {added:.1f} "
          f"(query mass +{100 * added / q_batch:.1f}%, scored +{100 * added / C_MASS:.2f}%, "
          f"{added / (3 * q_batch):.3f} of E14's added mass)")
    print("wrote", OUT_JSON)


if __name__ == "__main__":
    main()
