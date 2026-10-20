"""E17 -- K17-2 (held-out query fit), K17-3 (the band profile) and the coverage report
(research/e17/PREREG_E17.md sections 3.3, 4). Laptop, CPU.

    python research/e17/e17_fit.py --part m3       # K17-2: e17_a against e16_cap (checkpoints only;
                                                   #   after VERIFY_TRAIN, before E17_generate)
    python research/e17/e17_fit.py --part m2       # K17-3: on e17_a's free files
    python research/e17/e17_fit.py --part report   # coverage, diversity, never-seen (reported)

K17-2. E14's instrument, unchanged (research/e14/e14_fit.py `per_row_m3`, `paired`): the
pooled per-character difference (e17_a - e16_cap) on every held-out query row, with a
paired row-bootstrap 95% interval. PASS iff the BENIGN difference falls with its interval
below 0, pooled (section 3.3). Per stratum (the injector's stratum reader) and phishing
are reported. The per-row sums are first checked against N2a's own M3 on e16_cap.

K17-3. N2a's `band_profile`, unchanged, on e17_a's five free files per class; bands and
null from the old training queries plus both streams' (section 3.3). Self-check first:
the same function on e16_cap's files, with old + stream 1, must give data/e16/k16_3.json's
recorded share and parameter count exactly.

Writes data/e17/k17_2.json, data/e17/k17_3.json, data/e17/coverage_report.json.
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
for p in (ROOT, HERE, ROOT / "research" / "e16", ROOT / "research" / "e15", ROOT / "research" / "e14",
          ROOT / "research" / "native", ROOT / "research" / "dc1"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import e17_common as E17  # noqa: E402
from e14_fit import SELF_CHECK_ROWS, paired, per_row_m3  # noqa: E402
from e15_fit import stream_urls, strata_of  # noqa: E402

E17D = ROOT / "data" / "e17"
BASE, ARM = E17.BASELINE, E17.TRAIN_ARM
CHECKPOINTS = {BASE: ROOT / "data" / "e16" / "train" / BASE / "final.pt",
               ARM: E17D / "train" / ARM / "final.pt"}
GEN = {ARM: E17D / "generate" / "gen", BASE: ROOT / "data" / "e16" / "generate" / "gen",
       E17.DECODE_ARM: E17D / "decode" / "gen"}
POOLED = E17.STREAM2_DIR / E17.STREAM_FILE
DIVERSITY_DRAWS, DIVERSITY_SEED = 20, 20261010          # E15's report, unchanged


def verified(step: str) -> None:
    v = json.loads((E17D / f"VERIFY_{step}.json").read_text(encoding="utf-8"))
    if v.get("verdict") != "PASS":
        raise SystemExit(f"STOP: VERIFY_{step} has not passed; nothing is read")


def pooled_urls(label: str) -> list:
    rows_ = (json.loads(x) for x in POOLED.read_text(encoding="utf-8").splitlines() if x.strip())
    return [r["url"] for r in rows_ if r["label"] == label]


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
    if models[BASE][1].state_dict() != models[ARM][1].state_dict():
        raise SystemExit("STOP: e17_a's tokenizer is not e16_cap's; nothing is read")
    out = {"registration": E17.REGISTRATION + " section 3.3 (K17-2)",
           "instrument": "research/e14/e14_fit.py per_row_m3 + paired (N2a's M3, per row)",
           "checkpoints": {k: str(v) for k, v in CHECKPOINTS.items()}}
    for label in ("benign", "phishing"):
        train_stem, val_stem, _ = STREAMS[label]
        val_q = [u for u in rows(val_stem) if FIELDS["QRY"](u)]
        train_q = [u for u in rows(train_stem) if FIELDS["QRY"](u)]
        train_q = random.Random(SAMPLE_SEED).sample(train_q, min(TRAIN_ROWS, len(train_q)))
        m, t = models[BASE]
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
            res = paired(per[BASE], per[ARM])
            pooled = {name: sum(x[0] for x in v) / sum(x[1] for x in v) for name, v in per.items()}
            node[split] = {**res, "nll_per_char": pooled}
            print(f"  {label:<9} {split:<10} e16_cap {pooled[BASE]:.4f}  e17_a {pooled[ARM]:.4f}  "
                  f"diff {res['diff']:+.4f} [{res['ci'][0]:+.4f}, {res['ci'][1]:+.4f}]  rows {res['rows']:,}",
                  flush=True)
            if split == "validation":
                strata = strata_of(t, kept, label)
                by = {}
                for s in sorted({x for x in strata if x}) + ["other"]:
                    idx = [i for i, x in enumerate(strata) if (x or "other") == s]
                    if idx:
                        by[s] = paired([per[BASE][i] for i in idx], [per[ARM][i] for i in idx])
                        print(f"      {s:<16} diff {by[s]['diff']:+.4f} [{by[s]['ci'][0]:+.4f}, "
                              f"{by[s]['ci'][1]:+.4f}]  rows {by[s]['rows']:,}", flush=True)
                node["validation_by_stratum"] = by
        node["gap"] = {name: node["validation"]["nll_per_char"][name] - node["train"]["nll_per_char"][name]
                       for name in CHECKPOINTS}
        out[label] = node
    out["pass"] = bool(out["benign"]["validation"]["falls"])
    E17D.mkdir(parents=True, exist_ok=True)
    (E17D / "k17_2.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    print("K17-2:", "PASS" if out["pass"] else "FAIL", "->", E17D / "k17_2.json")


def part_m2(args) -> None:
    from n2a_reference import BOOT_SEED, STREAMS, band_profile, names_of, rows
    verified("GENERATE")
    rec = json.loads((ROOT / "data" / "e16" / "k16_3.json").read_text(encoding="utf-8"))
    out = {"registration": E17.REGISTRATION + " section 3.3 (K17-3)",
           "instrument": "research/native/n2a_reference.py band_profile (M2), unchanged",
           "reference": "bands and null: old training queries + both streams' (e17_a); "
                        "the self-check on e16_cap uses old + stream 1, as E16 recorded"}
    for label in ("benign", "phishing"):
        train_stem, _, struct_stem = STREAMS[label]
        freq_old = collections.Counter(n for stem in (train_stem, struct_stem) for u in rows(stem) for n in names_of(u))
        null_old = [n for u in rows(train_stem) for n in names_of(u)]
        ex1 = [n for u in stream_urls(label) for n in names_of(u)]
        ex12 = [n for u in pooled_urls(label) for n in names_of(u)]

        def m2(arm, extra):
            gen = [u for g in range(E17.G)
                   for u in json.loads((GEN[arm] / f"free_{arm}_{label}_g{g}.json").read_text(encoding="utf-8"))]
            r = band_profile(gen, freq_old + collections.Counter(extra), null_old + extra,
                             random.Random(BOOT_SEED))
            return {**r, "overshoot": r["observed_ci"][0] > r["null_rare"]}

        chk, want = m2(BASE, ex1), rec[label][BASE]
        if chk["observed_rare"] != want["observed_rare"] or chk["params"] != want["params"]:
            raise SystemExit(f"STOP: M2 on e16_cap is not k16_3.json's ({label}); nothing read")
        node = {"self_check": {BASE: chk["observed_rare"], "recorded": want["observed_rare"]},
                ARM: m2(ARM, ex12), f"{BASE}_both_streams": m2(BASE, ex12)}
        out[label] = node
        for k in (ARM, f"{BASE}_both_streams"):
            v = node[k]
            print(f"  {label:<9} {k:<22} rare {v['observed_rare']:.4f} [{v['observed_ci'][0]:.4f}, "
                  f"{v['observed_ci'][1]:.4f}] vs null {v['null_rare']:.4f}  overshoot={v['overshoot']}",
                  flush=True)
    out["pass"] = not any(out[label][ARM]["overshoot"] for label in ("benign", "phishing"))
    (E17D / "k17_3.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    print("K17-3:", "PASS" if out["pass"] else "FAIL", "->", E17D / "k17_3.json")


def part_report(args) -> None:
    """Coverage, diversity and never-seen on the benign query cells (reported, not a
    criterion): e17_a, its reference decode on the same weights family (e16_b5 if it was
    generated) and e16_cap. One known set for all arms: old training queries + both streams'."""
    import statistics as st
    sys.path.insert(0, str(ROOT / "research" / "e13"))
    import e13d_common as D
    from dc1_compare import training_queries
    from uie.evaluation import FIELDS
    verified("GENERATE")
    cells = [D.BY_NAME[n] for n in ("control_benign_P0Q1_QRY", "control_benign_P1Q1_QRY", "free_benign_QRY")]
    known = training_queries(ROOT / "data" / "bundle2", "benign") | {
        q for q in (FIELDS["QRY"](u) for u in pooled_urls("benign")) if q}
    arms = [ARM, BASE] + ([E17.DECODE_ARM] if (GEN[E17.DECODE_ARM] / "PROVENANCE.json").is_file() else [])
    out = {"registration": E17.REGISTRATION + " section 4 (reported, not a criterion)",
           "known": "old training queries + streams 1 and 2 (benign), for every arm"}
    for cell in cells:
        val = D.pool(cell, "val")
        vt = [D.template(q) for q in val]
        gens = {arm: D.generated(cell, arm, GEN[arm]) for arm in arms}
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
        print(f"  {cell[0]:<26} " + "  ".join(
            f"{arm}: coverage {node[arm]['coverage_of_val_templates']:.4f}, templates/1000 "
            f"{node[arm]['distinct_templates_per_1000']:.1f}, never-seen {node[arm]['never_seen']:.4f}"
            for arm in arms), flush=True)
    (E17D / "coverage_report.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    print("wrote", E17D / "coverage_report.json")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--part", required=True, choices=("m3", "m2", "report"))
    args = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")
    {"m3": part_m3, "m2": part_m2, "report": part_report}[args.part](args)


if __name__ == "__main__":
    main()
