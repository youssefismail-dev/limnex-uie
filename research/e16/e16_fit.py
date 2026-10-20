"""E16 -- K16-2 (held-out query fit), K16-3 (the band profile) and the coverage report
(research/e16/PREREG_E16.md sections 4, 5, 7). Laptop, CPU.

    python research/e16/e16_fit.py --part m3       # K16-2: e16_cap against e15_new (checkpoints only)
    python research/e16/e16_fit.py --part m2       # K16-3: on e16_cap's free files
    python research/e16/e16_fit.py --part report   # coverage, diversity, never-seen (reported)

K16-2. E14's instrument, unchanged (research/e14/e14_fit.py `per_row_m3`, `paired`): the
pooled per-character difference (e16_cap - e15_new) on every held-out query row, with a
paired row-bootstrap 95% interval. PASS iff the BENIGN difference falls with its interval
below 0, both pooled and on the P1Q1 stratum's rows (the injector's stratum reader).
Phishing, the P0Q1 stratum and N2a's training sample are reported. The per-row sums are
first checked against N2a's own M3 on e15_new.

K16-3. N2a's `band_profile`, unchanged, on e16_cap's five free files per class; bands and
null from the old training queries plus the stream's (both models trained on both). Self-
check first: the same function on e15_new's files must give data/e15/k15_3.json's
recorded share and parameter count exactly.

Writes data/e16/k16_2.json, data/e16/k16_3.json, data/e16/coverage_report.json.
"""
from __future__ import annotations

import argparse
import collections
import json
import random
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
for p in (ROOT, HERE, ROOT / "research" / "e15", ROOT / "research" / "e14", ROOT / "research" / "native",
          ROOT / "research" / "dc1"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import e16_common as E16  # noqa: E402
from e14_fit import SELF_CHECK_ROWS, paired, per_row_m3  # noqa: E402
from e15_fit import stream_urls, strata_of  # noqa: E402

E16D = ROOT / "data" / "e16"
CHECKPOINTS = {E16.BASELINE: ROOT / "data" / "e15" / "train" / "e15_new" / "final.pt",
               E16.ARM: E16D / "train" / E16.ARM / "final.pt"}
GEN = {E16.ARM: E16D / "generate" / "gen", E16.BASELINE: ROOT / "data" / "e15" / "generate" / "gen"}
DIVERSITY_DRAWS, DIVERSITY_SEED = 20, 20261010          # E15's report, unchanged


def verified(step: str) -> None:
    v = json.loads((E16D / f"VERIFY_{step}.json").read_text(encoding="utf-8"))
    if v.get("verdict") != "PASS":
        raise SystemExit(f"STOP: VERIFY_{step} has not passed; nothing is read")


def part_m3(args) -> None:
    import torch
    from n2a_reference import SAMPLE_SEED, STREAMS, TRAIN_ROWS, encode, rows, teacher_forced
    from uie.evaluation import FIELDS
    from uie.train import load_checkpoint
    verified("TRAIN")
    models = {}
    for name, path in CHECKPOINTS.items():
        m, t, _ = load_checkpoint(path, device=torch.device("cpu"))
        m.eval()
        models[name] = (m, t)
    if models[E16.BASELINE][1].state_dict() != models[E16.ARM][1].state_dict():
        raise SystemExit("STOP: e16_cap's tokenizer is not e15_new's; nothing is read")
    out = {"registration": E16.REGISTRATION + " section 5 (K16-2)",
           "instrument": "research/e14/e14_fit.py per_row_m3 + paired (N2a's M3, per row)",
           "checkpoints": {k: str(v) for k, v in CHECKPOINTS.items()}}
    base, arm = E16.BASELINE, E16.ARM
    for label in ("benign", "phishing"):
        train_stem, val_stem, _ = STREAMS[label]
        val_q = [u for u in rows(val_stem) if FIELDS["QRY"](u)]
        train_q = [u for u in rows(train_stem) if FIELDS["QRY"](u)]
        train_q = random.Random(SAMPLE_SEED).sample(train_q, min(TRAIN_ROWS, len(train_q)))
        m, t = models[base]
        probe = val_q[:SELF_CHECK_ROWS]
        mine = per_row_m3(m, t, probe, label)
        _, _, theirs = teacher_forced(m, t, probe, label, {})
        a = sum(x[0] for x in mine) / sum(x[1] for x in mine)
        b = sum(v[0] for v in theirs.values()) / sum(v[1] for v in theirs.values())
        if abs(a - b) > 1e-9 * max(1.0, abs(b)):
            raise SystemExit(f"STOP: per-row M3 {a!r} is not N2a's M3 {b!r} ({label}); nothing read")
        node = {"self_check": {"rows": len(probe), "per_row": a, "n2a": b}}
        for split, urls in (("validation", val_q), ("train", train_q)):
            kept = [u for u in urls if len(encode(t, u, label)) <= m.cfg.max_len]
            per = {name: per_row_m3(m_, t_, kept, label) for name, (m_, t_) in models.items()}
            res = paired(per[base], per[arm])
            pooled = {name: sum(x[0] for x in v) / sum(x[1] for x in v) for name, v in per.items()}
            node[split] = {**res, "nll_per_char": pooled}
            print(f"  {label:<9} {split:<10} e15_new {pooled[base]:.4f}  e16_cap {pooled[arm]:.4f}  "
                  f"diff {res['diff']:+.4f} [{res['ci'][0]:+.4f}, {res['ci'][1]:+.4f}]  rows {res['rows']:,}",
                  flush=True)
            if split == "validation":
                strata = strata_of(t, kept, label)
                by = {}
                for s in sorted({x for x in strata if x}) + ["other"]:
                    idx = [i for i, x in enumerate(strata) if (x or "other") == s]
                    if idx:
                        by[s] = paired([per[base][i] for i in idx], [per[arm][i] for i in idx])
                        print(f"      {s:<16} diff {by[s]['diff']:+.4f} [{by[s]['ci'][0]:+.4f}, "
                              f"{by[s]['ci'][1]:+.4f}]  rows {by[s]['rows']:,}", flush=True)
                node["validation_by_stratum"] = by
        node["gap"] = {name: node["validation"]["nll_per_char"][name] - node["train"]["nll_per_char"][name]
                       for name in CHECKPOINTS}
        out[label] = node
    p1 = out["benign"]["validation_by_stratum"].get("benign_P1Q1", {})
    out["pass"] = bool(out["benign"]["validation"]["falls"] and p1.get("falls"))
    E16D.mkdir(parents=True, exist_ok=True)
    (E16D / "k16_2.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    print("K16-2:", "PASS" if out["pass"] else "FAIL", "->", E16D / "k16_2.json")


def part_m2(args) -> None:
    from n2a_reference import BOOT_SEED, STREAMS, band_profile, names_of, rows
    verified("GENERATE")
    rec = json.loads((ROOT / "data" / "e15" / "k15_3.json").read_text(encoding="utf-8"))
    out = {"registration": E16.REGISTRATION + " section 5 (K16-3)",
           "instrument": "research/native/n2a_reference.py band_profile (M2), unchanged",
           "reference": "bands and null: old training queries + the stream's"}
    for label in ("benign", "phishing"):
        train_stem, _, struct_stem = STREAMS[label]
        freq = collections.Counter(n for stem in (train_stem, struct_stem) for u in rows(stem) for n in names_of(u))
        null = [n for u in rows(train_stem) for n in names_of(u)]
        extra = [n for u in stream_urls(label) for n in names_of(u)]
        freq, null = freq + collections.Counter(extra), null + extra

        def m2(arm):
            gen = [u for g in range(E16.G)
                   for u in json.loads((GEN[arm] / f"free_{arm}_{label}_g{g}.json").read_text(encoding="utf-8"))]
            r = band_profile(gen, freq, null, random.Random(BOOT_SEED))
            return {**r, "overshoot": r["observed_ci"][0] > r["null_rare"]}

        chk, want = m2(E16.BASELINE), rec[label]["e15_new"]
        if chk["observed_rare"] != want["observed_rare"] or chk["params"] != want["params"]:
            raise SystemExit(f"STOP: M2 on e15_new is not k15_3.json's ({label}); nothing read")
        node = {"self_check": {"e15_new": chk["observed_rare"], "recorded": want["observed_rare"]},
                E16.ARM: m2(E16.ARM), E16.BASELINE: chk}
        out[label] = node
        for k in (E16.ARM, E16.BASELINE):
            v = node[k]
            print(f"  {label:<9} {k:<8} rare {v['observed_rare']:.4f} [{v['observed_ci'][0]:.4f}, "
                  f"{v['observed_ci'][1]:.4f}] vs null {v['null_rare']:.4f}  overshoot={v['overshoot']}",
                  flush=True)
    out["pass"] = not any(out[label][E16.ARM]["overshoot"] for label in ("benign", "phishing"))
    (E16D / "k16_3.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    print("K16-3:", "PASS" if out["pass"] else "FAIL", "->", E16D / "k16_3.json")


def part_report(args) -> None:
    import statistics as st
    sys.path.insert(0, str(ROOT / "research" / "e13"))
    import e13d_common as D
    from dc1_compare import training_queries
    from uie.evaluation import FIELDS
    verified("GENERATE")
    cells = [D.BY_NAME[n] for n in ("control_benign_P0Q1_QRY", "control_benign_P1Q1_QRY", "free_benign_QRY")]
    known = training_queries(ROOT / "data" / "bundle2", "benign") | {
        q for q in (FIELDS["QRY"](u) for u in stream_urls("benign")) if q}
    out = {"registration": E16.REGISTRATION + " section 7 (reported, not a criterion)"}
    for cell in cells:
        val = D.pool(cell, "val")
        vt = [D.template(q) for q in val]
        gens = {arm: D.generated(cell, arm, GEN[arm]) for arm in (E16.ARM, E16.BASELINE)}
        k = min(len(g) for g in gens.values())
        node = {"val_queries": len(val), "equal_size": k}
        for arm, g in gens.items():
            have = {D.template(q) for q in g}
            per_k = [len({D.template(x) for x in random.Random(DIVERSITY_SEED + d).sample(g, k)}) / k * 1000
                     for d in range(DIVERSITY_DRAWS)]
            node[arm] = {"queries": len(g), "coverage_of_val_templates": sum(t in have for t in vt) / len(vt),
                         "distinct_templates_per_1000": st.median(per_k),
                         "never_seen": sum(q not in known for q in g) / len(g),
                         "mean_chars": sum(len(q) for q in g) / len(g),
                         "over_128_chars": sum(len(q) > 128 for q in g) / len(g)}
        out[cell[0]] = node
        a, b = node[E16.ARM], node[E16.BASELINE]
        print(f"  {cell[0]:<26} coverage {a['coverage_of_val_templates']:.4f} vs {b['coverage_of_val_templates']:.4f}"
              f"  templates/1000 {a['distinct_templates_per_1000']:.1f} vs {b['distinct_templates_per_1000']:.1f}"
              f"  never-seen {a['never_seen']:.4f} vs {b['never_seen']:.4f}"
              f"  >128 chars {a['over_128_chars']:.4f} vs {b['over_128_chars']:.4f}", flush=True)
    (E16D / "coverage_report.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    print("wrote", E16D / "coverage_report.json")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--part", required=True, choices=("m3", "m2", "report"))
    args = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")
    {"m3": part_m3, "m2": part_m2, "report": part_report}[args.part](args)


if __name__ == "__main__":
    main()
