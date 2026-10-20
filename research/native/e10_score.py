"""E10 -- every registered reading, per arm (research/native/PREREG_E10.md D7).

    python research/native/e10_score.py --part plan      # K10a: the plan's calibration (checkpoint only)
    python research/native/e10_score.py --part free      # gate v2, FULL FREE, canonical
    python research/native/e10_score.py --part control   # CR1's instrument, native output
    python research/native/e10_score.py --part misc      # emission, honouring, novelty, copying, plan_trace
    python research/native/e10_score.py --merge

Each part writes data/e10/score_<part>[_<arm...>].json, so parts run in parallel;
--merge joins them into data/e10/score_report.json with E7's canonical free record
from V4. The instruments are E9's, unchanged; the plan's calibration is
e8_plan_calibration.plan_by_suffix, weighted by the shipped table (FINDING 2.1).
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

import e10_common as K  # noqa: E402
from e8_score import _cell, presence, wilson  # noqa: E402

ARMS = tuple(K.ARMS)


def load(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def free_files(e10: Path, arm: str, label: str) -> list:
    return [load(e10 / "gen" / f"free_{arm}_{label}_g{g}.json") for g in range(K.G)]


def ctrl_files(e10: Path, arm: str, label: str, state: str) -> list:
    return [load(e10 / "gen" / f"ctrl_{arm}_{label}_{state}_g{g}.json") for g in range(K.G)]


def part_plan(args, arms) -> dict:
    """K10a: the plan arm's own P(path) and P(query) per shipped suffix, weighted
    by the shipped table, as FINDING 2.1 computed it."""
    import torch
    from e8_plan_calibration import plan_by_suffix
    from uie.suffixes import SuffixTable
    from uie.train import load_checkpoint
    model, tok, _ = load_checkpoint(args.e10 / "train" / "e10_plan" / "best.pt",
                                    device=torch.device("cpu"))
    model.eval()
    shipped = SuffixTable.load(args.bundle / "suffix_table.json").counts
    out = {}
    for label in K.LABELS:
        weights = shipped[label]
        wsum = sum(weights.values())
        per = plan_by_suffix(model, tok, label, sorted(weights))
        P = sum(c / wsum * per[s]["p"] for s, c in weights.items())
        Qr = sum(c / wsum * per[s]["q"] for s, c in weights.items())
        legal = sum(c / wsum * per[s]["legal_mass_P"] for s, c in weights.items())
        top = sorted(weights, key=lambda s: -weights[s])[:10]
        out[label] = {"P": P, "Q": Qr, "legal_mass_P": legal, "suffixes": len(weights),
                      "top": {s: {**per[s], "share": weights[s] / wsum} for s in top}}
        print(f"  plan {label:<9} P {P:.4f}  Q {Qr:.4f}  (legal mass {legal:.4f}, "
              f"{len(weights)} suffixes)", flush=True)
    return {"plan": out}


def part_free(args, arms) -> dict:
    from cr1_score import pools, score
    out = {}
    for arm in arms:
        for label in (args.label or K.LABELS):
            t0 = time.time()
            fields, pooled = score(pools(args.bundle, label), free_files(args.e10, arm, label),
                                   K.N_FREE)
            out.setdefault(arm, {})[label] = {"fields": fields, "pooled_fields": pooled}
            print(f"  free {arm:<9} {label:<9} " + "  ".join(
                _cell(f, r) for f, r in fields.items()) + f"  ({time.time() - t0:.0f}s)", flush=True)
    return {"free": out}


def part_control(args, arms) -> dict:
    from cr1_score import pools, score
    out = {}
    for arm in arms:
        for label in (args.label or K.LABELS):
            for st in K.STATES:
                if (label, st) in K.UNMEASURABLE:
                    continue
                t0 = time.time()
                pop = pools(args.bundle, label, K.STATES[st])
                fields, pooled = score(pop, ctrl_files(args.e10, arm, label, st), K.n_ctrl(label, st))
                out.setdefault(arm, {})[f"{label}_{st}"] = {
                    "label": label, "state": st, "real_name_pool": len(pop["name"]),
                    "fields": fields, "pooled_fields": pooled}
                print(f"  ctrl {arm:<9} {label}_{st:<5} " + "  ".join(
                    _cell(f, r) for f, r in fields.items()) + f"  ({time.time() - t0:.0f}s)",
                    flush=True)
    return {"control": out}


def part_misc(args, arms) -> dict:
    from dc1_compare import training_queries
    from uie.bloom import BloomFilter
    from uie.evaluation import FIELDS
    from uie.gate import normalize_for_dedup

    bloom = BloomFilter.load(args.bundle / "known_urls.bloom.gz")
    out = {"honouring": {}, "emission": {}, "novelty": {}, "plan_trace": {}}
    gprov = load(args.e10 / "gen" / "PROVENANCE.json")["files"]
    for arm in arms:
        for label in K.LABELS:
            for st, want in K.STATES.items():
                per_g, k_all, n_all = [], 0, 0
                for urls in ctrl_files(args.e10, arm, label, st):
                    k = sum(presence(u) == (want["PTH"], want["QRY"]) for u in urls)
                    per_g.append(k / len(urls))
                    k_all, n_all = k_all + k, n_all + len(urls)
                out["honouring"].setdefault(arm, {})[f"{label}_{st}"] = {
                    **wilson(k_all, n_all), "per_generation": per_g}
            urls = [u for f in free_files(args.e10, arm, label) for u in f]
            p = sum(presence(u)[0] for u in urls)
            q = sum(presence(u)[1] for u in urls)
            out["emission"].setdefault(arm, {})[label] = {"PTH": wilson(p, len(urls)),
                                                          "QRY": wilson(q, len(urls))}
        for mode in ("free", "control"):
            vals = [r.get("plan_trace_mean") for r in gprov.values()
                    if r.get("arm") == arm and r.get("mode") == mode
                    and r.get("plan_trace_mean") is not None]
            out["plan_trace"].setdefault(arm, {})[mode] = (sum(vals) / len(vals)) if vals else None
    for label in K.LABELS:
        known_q = training_queries(args.bundle, label)
        for arm in arms:
            for mode, files in (("free", free_files(args.e10, arm, label)),
                                ("control", [f for st in K.STATES
                                             for f in ctrl_files(args.e10, arm, label, st)])):
                c = out["novelty"].setdefault(arm, {}).setdefault(
                    mode, {"n": 0, "novel": 0, "q": 0, "copied": 0})
                for u in (u for f in files for u in f):
                    c["n"] += 1
                    c["novel"] += normalize_for_dedup(u) not in bloom
                    qs = FIELDS["QRY"](u)
                    if qs:
                        c["q"] += 1
                        c["copied"] += qs in known_q
        del known_q
    for arm, parts in out["emission"].items():
        print(f"  emission  {arm:<9} " + "  ".join(
            f"{l} P {v['PTH']['rate']:.4f} Q {v['QRY']['rate']:.4f}" for l, v in parts.items()))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--e10", type=Path, default=ROOT / "data" / "e10")
    ap.add_argument("--bundle", type=Path, default=ROOT / "data" / "bundle2")
    ap.add_argument("--part", choices=("plan", "free", "control", "misc"))
    ap.add_argument("--arm", nargs="*")
    ap.add_argument("--label", nargs="*", help="free / control only: run these classes "
                    "(parallel processes; the output file is tagged with them)")
    ap.add_argument("--merge", action="store_true")
    args = ap.parse_args()
    if args.merge:
        merged = {"registration": "research/native/PREREG_E10.md",
                  "scorer": "research/cr1/cr1_score.py (pools, score), unchanged",
                  "e7_free": load(args.e10 / "e7_free_fields.json")}
        for p in sorted(args.e10.glob("score_*.json")):
            if p.name == "score_report.json":
                continue
            for key, val in load(p).items():
                if isinstance(val, dict):
                    for who, v in val.items():
                        merged.setdefault(key, {}).setdefault(who, {}).update(v)
        (args.e10 / "score_report.json").write_text(json.dumps(merged, indent=1, default=float),
                                                     encoding="utf-8")
        print("merged", args.e10 / "score_report.json")
        return
    if not args.part:
        ap.error("--part or --merge")
    arms = tuple(args.arm) if args.arm else ARMS
    report = {"plan": part_plan, "free": part_free, "control": part_control,
              "misc": part_misc}[args.part](args, arms)
    tag = (args.part + ("_" + "_".join(args.arm) if args.arm else "")
           + ("_" + "_".join(args.label) if args.label else ""))
    (args.e10 / f"score_{tag}.json").write_text(json.dumps(report, indent=1, default=float),
                                                encoding="utf-8")
    print("wrote", args.e10 / f"score_{tag}.json")


if __name__ == "__main__":
    main()
