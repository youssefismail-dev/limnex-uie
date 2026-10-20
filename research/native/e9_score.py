"""E9 -- every registered reading, per condition (PREREG_E9 section 4, R2-R6).

    python research/native/e9_score.py --part free       # R3: gate v2, FULL FREE, canonical
    python research/native/e9_score.py --part control    # R3: CR1's instrument, native output
    python research/native/e9_score.py --part misc       # R2 emission, R4 honouring, R6 guardrails
    python research/native/e9_score.py --part qry        # R5: Q-VAL / Q-TRAIN (section 7's rule)
    python research/native/e9_score.py --merge

A condition is `<arm>_<decode>` (for example `e9_ln_raw`). Each part writes
data/e9/score_<part>[_<who...>].json, so parts run in parallel; --merge joins them
into data/e9/score_report.json, with E7's canonical free record from V4.

Instruments are E8's, unchanged: `cr1_score` (pools, score) for the gate cells,
E8's presence and honouring definitions, DC1's novelty and copying, N2a's
QRY-only cell (`n2a_qry.qry_cell`) for Q-VAL / Q-TRAIN.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "cr1"))
sys.path.insert(0, str(HERE.parent / "dc1"))

import e9_common as K  # noqa: E402
from e8_score import _cell, presence, wilson  # noqa: E402

FREE = tuple(f"{a}_{d}" for a, d in K.FREE_CONDITIONS)
CTRL = tuple(f"{a}_{d}" for a, d in K.CTRL_CONDITIONS)
QRY_WHO = ("e9_base_resting", "e9_base_raw", "e9_ln_raw")


def load(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def free_files(e9: Path, who: str, label: str) -> list:
    return [load(e9 / "gen" / f"free_{who}_{label}_g{g}.json") for g in range(K.G)]


def ctrl_files(e9: Path, who: str, label: str, state: str) -> list:
    return [load(e9 / "gen" / f"ctrl_{who}_{label}_{state}_g{g}.json") for g in range(K.G)]


def part_free(args, whos) -> dict:
    from cr1_score import pools, score
    out = {}
    for who in whos:
        for label in K.LABELS:
            t0 = time.time()
            fields, pooled = score(pools(args.bundle, label), free_files(args.e9, who, label),
                                   K.N_FREE)
            out.setdefault(who, {})[label] = {"fields": fields, "pooled_fields": pooled}
            print(f"  free {who:<16} {label:<9} " + "  ".join(
                _cell(f, r) for f, r in fields.items()) + f"  ({time.time() - t0:.0f}s)", flush=True)
    return {"free": out}


def part_control(args, whos) -> dict:
    from cr1_score import pools, score
    out = {}
    for who in whos:
        for label in K.LABELS:
            for st in K.STATES:
                if (label, st) in K.UNMEASURABLE:
                    continue
                t0 = time.time()
                pop = pools(args.bundle, label, K.STATES[st])
                fields, pooled = score(pop, ctrl_files(args.e9, who, label, st), K.N_CTRL)
                out.setdefault(who, {})[f"{label}_{st}"] = {
                    "label": label, "state": st, "real_name_pool": len(pop["name"]),
                    "fields": fields, "pooled_fields": pooled}
                print(f"  ctrl {who:<16} {label}_{st:<5} " + "  ".join(
                    _cell(f, r) for f, r in fields.items()) + f"  ({time.time() - t0:.0f}s)",
                    flush=True)
    return {"control": out}


def part_misc(args, whos) -> dict:
    from dc1_compare import training_queries
    from uie.bloom import BloomFilter
    from uie.evaluation import FIELDS
    from uie.gate import normalize_for_dedup

    bloom = BloomFilter.load(args.bundle / "known_urls.bloom.gz")
    out = {"honouring": {}, "emission": {}, "novelty": {}}
    for who in whos:
        if who in CTRL:
            for label in K.LABELS:
                for st, want in K.STATES.items():
                    per_g, k_all, n_all = [], 0, 0
                    for urls in ctrl_files(args.e9, who, label, st):
                        k = sum(presence(u) == (want["PTH"], want["QRY"]) for u in urls)
                        per_g.append(k / len(urls))
                        k_all, n_all = k_all + k, n_all + len(urls)
                    out["honouring"].setdefault(who, {})[f"{label}_{st}"] = {
                        **wilson(k_all, n_all), "per_generation": per_g}
        if who in FREE:
            for label in K.LABELS:
                urls = [u for f in free_files(args.e9, who, label) for u in f]
                p = sum(presence(u)[0] for u in urls)
                q = sum(presence(u)[1] for u in urls)
                out["emission"].setdefault(who, {})[label] = {"PTH": wilson(p, len(urls)),
                                                              "QRY": wilson(q, len(urls))}
    for label in K.LABELS:
        known_q = training_queries(args.bundle, label)
        for who in whos:
            modes = []
            if who in FREE:
                modes.append(("free", free_files(args.e9, who, label)))
            if who in CTRL:
                modes.append(("control", [f for st in K.STATES
                                          for f in ctrl_files(args.e9, who, label, st)]))
            for mode, files in modes:
                c = out["novelty"].setdefault(who, {}).setdefault(
                    mode, {"n": 0, "novel": 0, "q": 0, "copied": 0})
                for u in (u for f in files for u in f):
                    c["n"] += 1
                    c["novel"] += normalize_for_dedup(u) not in bloom
                    qs = FIELDS["QRY"](u)
                    if qs:
                        c["q"] += 1
                        c["copied"] += qs in known_q
        del known_q
    for who, parts in out["honouring"].items():
        print(f"  honouring {who:<16} " + "  ".join(f"{k} {v['rate']:.4f}" for k, v in parts.items()))
    for who, parts in out["emission"].items():
        print(f"  emission  {who:<16} " + "  ".join(
            f"{l} P {v['PTH']['rate']:.4f} Q {v['QRY']['rate']:.4f}" for l, v in parts.items()))
    return out


def part_qry(args, whos) -> dict:
    """R5: the gate's QRY cell against validation (Q-VAL) and domain-frame training
    (Q-TRAIN) queries, N2a's exact instrument and n."""
    from cr1_score import pools
    from n2a_qry import N_MAIN, qry_cell, train_pool
    out = {}
    for label in K.LABELS:
        val, trn = pools(args.bundle, label)["name"], train_pool(label)
        for who in whos:
            gens = free_files(args.e9, who, label)
            out.setdefault(who, {})[label] = {"q_val": qry_cell(val, gens, N_MAIN[label]),
                                              "q_train": qry_cell(trn, gens, N_MAIN[label])}
            r = out[who][label]
            print(f"  qry {who:<16} {label:<9} Q-VAL {_cell('', r['q_val'])}  "
                  f"Q-TRAIN {_cell('', r['q_train'])}", flush=True)
    return {"qry": out}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--e9", type=Path, default=ROOT / "data" / "e9")
    ap.add_argument("--bundle", type=Path, default=ROOT / "data" / "bundle2")
    ap.add_argument("--part", choices=("free", "control", "misc", "qry"))
    ap.add_argument("--who", nargs="*")
    ap.add_argument("--merge", action="store_true")
    args = ap.parse_args()
    if args.merge:
        merged = {"registration": "research/native/PREREG_E9.md",
                  "scorer": "research/cr1/cr1_score.py (pools, score), unchanged",
                  "e7_free": load(args.e9 / "e7_free_fields.json")}
        for p in sorted(args.e9.glob("score_*.json")):
            if p.name == "score_report.json":
                continue
            for key, val in load(p).items():
                if isinstance(val, dict):
                    for who, v in val.items():
                        merged.setdefault(key, {}).setdefault(who, {}).update(v)
        (args.e9 / "score_report.json").write_text(json.dumps(merged, indent=1, default=float),
                                                    encoding="utf-8")
        print("merged", args.e9 / "score_report.json")
        return
    if not args.part:
        ap.error("--part or --merge")
    default = {"free": FREE, "control": CTRL, "misc": tuple(dict.fromkeys(FREE + CTRL)),
               "qry": QRY_WHO}[args.part]
    whos = tuple(args.who) if args.who else default
    report = {"free": part_free, "control": part_control, "misc": part_misc,
              "qry": part_qry}[args.part](args, whos)
    tag = args.part + ("_" + "_".join(args.who) if args.who else "")
    (args.e9 / f"score_{tag}.json").write_text(json.dumps(report, indent=1, default=float),
                                               encoding="utf-8")
    print("wrote", args.e9 / f"score_{tag}.json")


if __name__ == "__main__":
    main()
