"""E15-B -- K15-2, held-out query fit (research/e15/PREREG_E15.md sections 9.8, 13.3).
Laptop, CPU; needs only the two checkpoints.

    python research/e15/e15_fit.py --part m3

E14's K14-2 instrument, unchanged (research/e14/e14_fit.py `per_row_m3`, `paired`):
N2a's M3, teacher-forced NLL per query character on every validation query row of
the class (the domain frame), PER ROW for e12_final and e15_new on the same rows;
"falls" = the pooled difference (e15_new - e12_final) is below 0 and its paired
row-bootstrap 95% interval (1,000 draws, seed 20260806) lies below 0. The per-row
sums are first checked against N2a's own pooled M3 on e12_final (stops if they differ).

The licensed class is benign (no phishing rows were admitted, section 13.1); K15-2 is
read there. Reported beside it: per stratum (the two admitted strata, P0Q1 and P1Q1,
by the injector's own stratum reader), phishing (untouched), and M3 on N2a's training
sample (the train/validation gap).

Writes data/e15/k15_2.json.

    python research/e15/e15_fit.py --part m2       # K15-3: N2a's M2 on e15_new's free files
    python research/e15/e15_fit.py --part report   # section 9.9's coverage row (reported)

K15-3. N2a's `band_profile`, unchanged, on e15_new's five free files per class. Bands
and the faithful-sampler null come from the model's own training queries: the old
rows plus the stream's (section 13.2). OVERSHOOT if the observed rare-band share's
95% interval lies wholly above the null (PREREG_E14 section 3.1). Self-check first:
the same function on e12_final's free files with the old reference must give E14's
recorded observed share and parameter count exactly (data/e14/n2a_e12_final_*.json).
The old-only reading is reported beside it.

Report. On the licensed cells (and free benign, DX3's reference), e15_new against
e12_d1p: coverage of the validation templates (E13-D's `template`), distinct templates
per 1,000 queries at equal size, and the never-seen share against each arm's own
training queries. Writes data/e15/k15_3.json and data/e15/coverage_report.json.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
for p in (ROOT, HERE, ROOT / "research" / "e14", ROOT / "research" / "native", ROOT / "research" / "dc1"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import e15_common as E15  # noqa: E402
from e14_fit import SELF_CHECK_ROWS, paired, per_row_m3  # noqa: E402

E15D = ROOT / "data" / "e15"
CHECKPOINTS = {"e12_final": ROOT / "data" / "e12" / "train" / "e12_final" / "final.pt",
               E15.ARM: E15D / "train" / E15.ARM / "final.pt"}
LICENSED = "benign"


def strata_of(tok, urls, label):
    """Each row's stratum as the injector reads it from its domain-frame tokens."""
    from n2a_reference import encode
    from uie.train import QueryInjector
    probe = QueryInjector(tok, {}, {}, 0, 0)
    return [probe.stratum_of(encode(tok, u, label)) for u in urls]


def part_m3(args) -> None:
    import torch
    from n2a_reference import SAMPLE_SEED, STREAMS, TRAIN_ROWS, encode, rows, teacher_forced
    from uie.evaluation import FIELDS
    from uie.train import load_checkpoint
    if json.loads((E15D / "VERIFY_TRAIN.json").read_text(encoding="utf-8")).get("verdict") != "PASS":
        raise SystemExit("STOP: VERIFY_TRAIN has not passed; nothing is read")
    models = {}
    for name, path in CHECKPOINTS.items():
        m, t, _ = load_checkpoint(path, device=torch.device("cpu"))
        m.eval()
        models[name] = (m, t)
    if models["e12_final"][1].state_dict() != models[E15.ARM][1].state_dict():
        raise SystemExit("STOP: e15_new's tokenizer is not e12_final's; nothing is read")
    out = {"registration": E15.REGISTRATION + " sections 9.8, 13.3 (K15-2)",
           "instrument": "research/e14/e14_fit.py per_row_m3 + paired (N2a's M3, per row)",
           "licensed_class": LICENSED,
           "checkpoints": {k: str(v) for k, v in CHECKPOINTS.items()}}
    for label in ("benign", "phishing"):
        train_stem, val_stem, _ = STREAMS[label]
        val_q = [u for u in rows(val_stem) if FIELDS["QRY"](u)]
        train_q = [u for u in rows(train_stem) if FIELDS["QRY"](u)]
        train_q = random.Random(SAMPLE_SEED).sample(train_q, min(TRAIN_ROWS, len(train_q)))
        m, t = models["e12_final"]
        probe = val_q[:SELF_CHECK_ROWS]
        mine = per_row_m3(m, t, probe, label)
        _, _, theirs = teacher_forced(m, t, probe, label, {})
        a = sum(x[0] for x in mine) / sum(x[1] for x in mine)
        b = sum(v[0] for v in theirs.values()) / sum(v[1] for v in theirs.values())
        if abs(a - b) > 1e-9 * max(1.0, abs(b)) or sum(x[1] for x in mine) != sum(
                v[1] for v in theirs.values()):
            raise SystemExit(f"STOP: per-row M3 {a!r} is not N2a's M3 {b!r} ({label}); nothing read")
        node = {"self_check": {"rows": len(probe), "per_row": a, "n2a": b}}
        for split, urls in (("validation", val_q), ("train", train_q)):
            kept = [u for u in urls if len(encode(t, u, label)) <= m.cfg.max_len]   # per_row_m3's own filter, applied first so strata align
            per = {name: per_row_m3(m_, t_, kept, label) for name, (m_, t_) in models.items()}
            res = paired(per["e12_final"], per[E15.ARM])
            pooled = {name: sum(x[0] for x in v) / sum(x[1] for x in v) for name, v in per.items()}
            node[split] = {**res, "nll_per_char": pooled}
            print(f"  {label:<9} {split:<10} e12_final {pooled['e12_final']:.4f}  e15_new "
                  f"{pooled[E15.ARM]:.4f}  diff {res['diff']:+.4f} [{res['ci'][0]:+.4f}, "
                  f"{res['ci'][1]:+.4f}]  rows {res['rows']:,}", flush=True)
            if split == "validation":
                strata = strata_of(t, kept, label)
                by = {}
                for s in sorted({x for x in strata if x}) + ["other"]:
                    idx = [i for i, x in enumerate(strata) if (x or "other") == s]
                    if not idx:
                        continue
                    r_ = paired([per["e12_final"][i] for i in idx], [per[E15.ARM][i] for i in idx])
                    by[s] = r_
                    print(f"      {s:<16} diff {r_['diff']:+.4f} [{r_['ci'][0]:+.4f}, {r_['ci'][1]:+.4f}]"
                          f"  rows {r_['rows']:,}", flush=True)
                node["validation_by_stratum"] = by
        node["gap"] = {name: node["validation"]["nll_per_char"][name] - node["train"]["nll_per_char"][name]
                       for name in CHECKPOINTS}
        out[label] = node
    out["pass"] = out[LICENSED]["validation"]["falls"]
    E15D.mkdir(parents=True, exist_ok=True)
    (E15D / "k15_2.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    print("K15-2:", "PASS" if out["pass"] else "FAIL", "->", E15D / "k15_2.json")


GEN = {E15.ARM: E15D / "generate" / "gen", "e12_d1p": ROOT / "data" / "e14" / "decode" / "gen",
       "e12_final": ROOT / "data" / "e12" / "gen"}
STREAM = E15D / "qry_extra" / E15.STREAM_FILE
DIVERSITY_DRAWS, DIVERSITY_SEED = 20, 20261010


def stream_urls(label: str) -> list:
    rows_ = (json.loads(x) for x in STREAM.read_text(encoding="utf-8").splitlines() if x.strip())
    return [r["url"] for r in rows_ if r["label"] == label]


def part_m2(args) -> None:
    import collections
    from n2a_reference import BOOT_SEED, STREAMS, band_profile, names_of, rows
    if json.loads((E15D / "VERIFY_GENERATE.json").read_text(encoding="utf-8")).get("verdict") != "PASS":
        raise SystemExit("STOP: VERIFY_GENERATE has not passed; nothing is read")
    out = {"registration": E15.REGISTRATION + " sections 9.8, 13.2 (K15-3)",
           "instrument": "research/native/n2a_reference.py band_profile (M2), unchanged",
           "reference": "bands and null: old training queries + the stream's (section 13.2)"}
    for label in ("benign", "phishing"):
        train_stem, _, struct_stem = STREAMS[label]
        freq_old = collections.Counter(n for stem in (train_stem, struct_stem)
                                       for u in rows(stem) for n in names_of(u))
        null_old = [n for u in rows(train_stem) for n in names_of(u)]
        extra = [n for u in stream_urls(label) for n in names_of(u)]
        freq_new, null_new = freq_old + collections.Counter(extra), null_old + extra

        def m2(arm, freq, null):
            gen = [u for g in range(E15.G)
                   for u in json.loads((GEN[arm] / f"free_{arm}_{label}_g{g}.json").read_text(encoding="utf-8"))]
            r = band_profile(gen, freq, null, random.Random(BOOT_SEED))
            return {**r, "overshoot": r["observed_ci"][0] > r["null_rare"]}

        rec = json.loads((ROOT / "data" / "e14" / f"n2a_e12_final_{label}.json").read_text(encoding="utf-8"))["M2"]
        chk = m2("e12_final", freq_old, null_old)
        if chk["observed_rare"] != rec["observed_rare"] or chk["params"] != rec["params"]:
            raise SystemExit(f"STOP: M2 on e12_final {chk['observed_rare']!r}/{chk['params']} is not "
                             f"E14's {rec['observed_rare']!r}/{rec['params']} ({label}); nothing read")
        node = {"self_check": {"observed_rare": chk["observed_rare"], "e14_record": rec["observed_rare"]},
                "stream_names": len(extra),
                E15.ARM: m2(E15.ARM, freq_new, null_new),
                f"{E15.ARM}_old_ref": m2(E15.ARM, freq_old, null_old),
                "e12_d1p": m2("e12_d1p", freq_old, null_old)}
        out[label] = node
        for k in (E15.ARM, f"{E15.ARM}_old_ref", "e12_d1p"):
            v = node[k]
            print(f"  {label:<9} {k:<16} rare {v['observed_rare']:.4f} [{v['observed_ci'][0]:.4f}, "
                  f"{v['observed_ci'][1]:.4f}] vs null {v['null_rare']:.4f}  overshoot={v['overshoot']}", flush=True)
    out["pass"] = not any(out[label][E15.ARM]["overshoot"] for label in ("benign", "phishing"))
    (E15D / "k15_3.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    print("K15-3:", "PASS" if out["pass"] else "FAIL", "->", E15D / "k15_3.json")


def part_report(args) -> None:
    import statistics as st
    sys.path.insert(0, str(ROOT / "research" / "e13"))
    import e13d_common as D
    from dc1_compare import training_queries
    from uie.evaluation import FIELDS
    if json.loads((E15D / "VERIFY_GENERATE.json").read_text(encoding="utf-8")).get("verdict") != "PASS":
        raise SystemExit("STOP: VERIFY_GENERATE has not passed; nothing is read")
    cells = [D.BY_NAME[n] for n in ("control_benign_P0Q1_QRY", "control_benign_P1Q1_QRY", "free_benign_QRY")]
    known_old = training_queries(ROOT / "data" / "bundle2", "benign")
    known_new = known_old | {q for q in (FIELDS["QRY"](u) for u in stream_urls("benign")) if q}
    out = {"registration": E15.REGISTRATION + " sections 9.9, 10.6 (reported, not a criterion)"}
    for cell in cells:
        val = D.pool(cell, "val")
        vt = [D.template(q) for q in val]
        gens = {arm: D.generated(cell, arm, GEN[arm]) for arm in (E15.ARM, "e12_d1p")}
        k = min(len(g) for g in gens.values())
        node = {"val_queries": len(val), "equal_size": k}
        for arm, g in gens.items():
            have = {D.template(q) for q in g}
            per_k = [len({D.template(x) for x in random.Random(DIVERSITY_SEED + d).sample(g, k)}) / k * 1000
                     for d in range(DIVERSITY_DRAWS)]
            known = known_new if arm == E15.ARM else known_old
            node[arm] = {"queries": len(g),
                         "coverage_of_val_templates": sum(t in have for t in vt) / len(vt),
                         "distinct_templates_per_1000": st.median(per_k),
                         "never_seen": sum(q not in known for q in g) / len(g)}
            if arm == E15.ARM:
                node[arm]["never_seen_old_ref"] = sum(q not in known_old for q in g) / len(g)
        node["coverage_gain"] = (node[E15.ARM]["coverage_of_val_templates"]
                                 - node["e12_d1p"]["coverage_of_val_templates"])
        out[cell[0]] = node
        a, b = node[E15.ARM], node["e12_d1p"]
        print(f"  {cell[0]:<26} coverage {a['coverage_of_val_templates']:.4f} vs {b['coverage_of_val_templates']:.4f}"
              f"  templates/1000 {a['distinct_templates_per_1000']:.1f} vs {b['distinct_templates_per_1000']:.1f}"
              f"  never-seen {a['never_seen']:.4f} vs {b['never_seen']:.4f}", flush=True)
    (E15D / "coverage_report.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    print("wrote", E15D / "coverage_report.json")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--part", required=True, choices=("m3", "m2", "report"))
    args = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")
    {"m3": part_m3, "m2": part_m2, "report": part_report}[args.part](args)


if __name__ == "__main__":
    main()
