"""E12 -- every registered reading, per arm (research/native/PREREG_E12.md D8,
section 2).

    python research/native/e12_score.py --part free      # gate v2, FULL FREE, canonical
    python research/native/e12_score.py --part control   # CR1's instrument, native output
    python research/native/e12_score.py --part misc      # emission, slash, honouring (P, Q and S),
                                                         # novelty, copying
    python research/native/e12_score.py --part copy      # K12d: QC1-1's instrument per cell
    python research/native/e12_score.py --merge

--arm and --label run a subset (parallel processes; each output file is tagged with
them); --merge joins data/e12/score_*.json into score_report.json with E7's
canonical record from V4. The instruments are E11's, unchanged; S honouring is new:
each URL's supplied S is reconstructed from the seeds, exactly as e12_verify locates
it, and never read from the output.
"""
from __future__ import annotations

import argparse
import collections
import json
import random
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "cr1"))
sys.path.insert(0, str(HERE.parent / "dc1"))
sys.path.insert(0, str(HERE.parent / "qc1"))

import e12_common as K  # noqa: E402
from e8_score import _cell, presence, wilson  # noqa: E402
from e11_score import slash_state  # noqa: E402

ARMS = tuple(K.ARMS)
PS_PATH = ROOT / "data" / "e12" / "presence_slash_table.json"


def load(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def free_files(d: Path, arm: str, label: str) -> list:
    return [load(d / "gen" / f"free_{arm}_{label}_g{g}.json") for g in range(K.G)]


def ctrl_files(d: Path, arm: str, label: str, state: str) -> list:
    return [load(d / "gen" / f"ctrl_{arm}_{label}_{state}_g{g}.json") for g in range(K.G)]


def state_positions(tables, n: int, seed: int) -> list:
    """Each position's supplied state in a supplied-plan file: the state draw and the
    outer shuffle from the seed (e12_common._draw_and_generate), no regeneration."""
    states = list(tables)
    rng = random.Random(seed)
    counts = collections.Counter(rng.choices(states, weights=[sum(tables[k].values())
                                                              for k in states], k=n))
    perm = list(range(n))
    rng.shuffle(perm)
    outer = {orig: pos for pos, orig in enumerate(perm)}
    at = [None] * n
    block = 0
    for k in states:
        for orig in range(block, block + counts[k]):
            at[outer[orig]] = k
        block += counts[k]
    return at


def part_free(args, arms) -> dict:
    from cr1_score import pools, score
    out = {}
    for arm in arms:
        for label in (args.label or K.LABELS):
            t0 = time.time()
            fields, pooled = score(pools(args.bundle, label), free_files(args.e12, arm, label), K.N_FREE)
            out.setdefault(arm, {})[label] = {"fields": fields, "pooled_fields": pooled}
            print(f"  free {arm:<10} {label:<9} " + "  ".join(
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
                fields, pooled = score(pop, ctrl_files(args.e12, arm, label, st), K.n_ctrl(label, st))
                out.setdefault(arm, {})[f"{label}_{st}"] = {
                    "label": label, "state": st, "real_name_pool": len(pop["name"]),
                    "fields": fields, "pooled_fields": pooled}
                print(f"  ctrl {arm:<10} {label}_{st:<5} " + "  ".join(
                    _cell(f, r) for f, r in fields.items()) + f"  ({time.time() - t0:.0f}s)", flush=True)
    return {"control": out}


def part_misc(args, arms) -> dict:
    from dc1_compare import training_queries
    from uie.bloom import BloomFilter
    from uie.controlled import StateSuffixTables
    from uie.evaluation import FIELDS
    from uie.factorization import parse
    from uie.gate import normalize_for_dedup
    from uie.suffixes import SuffixTable

    bloom = BloomFilter.load(args.bundle / "known_urls.bloom.gz")
    shipped = SuffixTable.load(args.bundle / "suffix_table.json")
    tables = StateSuffixTables.load(ROOT / "data" / "state_suffix_tables.json")
    ps = load(PS_PATH)["table"]
    out = {"honouring": {}, "emission": {}, "slash": {}, "novelty": {}, "plan_trace": {},
           "s_honouring": {}}
    gprov = load(args.e12 / "gen" / "PROVENANCE.json")["files"]
    for arm in arms:
        slash_arm = arm in K.SLASH_ARMS
        for label in K.LABELS:
            for st, want in K.STATES.items():
                per_g, k_all, n_all = [], 0, 0
                files = ctrl_files(args.e12, arm, label, st)
                for urls in files:
                    k = sum(presence(u) == (want["PTH"], want["QRY"]) for u in urls)
                    per_g.append(k / len(urls))
                    k_all, n_all = k_all + k, n_all + len(urls)
                out["honouring"].setdefault(arm, {})[f"{label}_{st}"] = {
                    **wilson(k_all, n_all), "per_generation": per_g}
                if st in ("P0Q0", "P0Q1"):                 # K11b's rows: honoured URLs of the state
                    s = [slash_state(u) for f in files for u in f]
                    hit = [x[1] for x in s if x is not None and x[0] == want["QRY"]]
                    out["slash"].setdefault(arm, {})[f"control_{label}_{st}"] = wilson(sum(hit), len(hit))
                if slash_arm:                               # S honouring, by the supplied S
                    ct = K.control_tables(ps, tables, label, st)
                    c = collections.Counter()
                    for g, urls in enumerate(files):
                        at = state_positions(ct, len(urls), K.seed_ctrl(label, st, g))
                        for u, k in zip(urls, at):
                            c[(k[-2:], bool(parse(u).path) == (k[-1] == "1"))] += 1
                    out["s_honouring"].setdefault(arm, {})[f"control_{label}_{st}"] = {
                        sv: wilson(c[(sv, True)], c[(sv, True)] + c[(sv, False)])
                        for sv in ("S0", "S1") if c[(sv, True)] + c[(sv, False)]}
            ffiles = free_files(args.e12, arm, label)
            urls = [u for f in ffiles for u in f]
            p = sum(presence(u)[0] for u in urls)
            q = sum(presence(u)[1] for u in urls)
            out["emission"].setdefault(arm, {})[label] = {"PTH": wilson(p, len(urls)),
                                                          "QRY": wilson(q, len(urls))}
            s = [slash_state(u) for u in urls]
            for key, has_q in (("free_bare", False), ("free_host_query", True)):
                hit = [x[1] for x in s if x is not None and x[0] == has_q]
                out["slash"].setdefault(arm, {})[f"{key}_{label}"] = wilson(sum(hit), len(hit))
            if slash_arm:
                ft = K.supplied_tables(ps, shipped.counts, label)
                c = collections.Counter()
                for g, f in enumerate(ffiles):
                    at = state_positions(ft, len(f), K.seed_free(g))
                    for u, k in zip(f, at):
                        c[(k[-2:], bool(parse(u).path) == (k[-1] == "1"))] += 1
                out["s_honouring"].setdefault(arm, {})[f"free_{label}"] = {
                    sv: wilson(c[(sv, True)], c[(sv, True)] + c[(sv, False)])
                    for sv in ("S0", "S1") if c[(sv, True)] + c[(sv, False)]}
        for mode in ("free", "control"):
            vals = [r.get("plan_trace_mean") for r in gprov.values()
                    if r.get("arm") == arm and r.get("mode") == mode and r.get("plan_trace_mean") is not None]
            out["plan_trace"].setdefault(arm, {})[mode] = (sum(vals) / len(vals)) if vals else None
    for label in K.LABELS:
        known_q = training_queries(args.bundle, label)
        for arm in arms:
            for mode, files in (("free", free_files(args.e12, arm, label)),
                                ("control", [f for st in K.STATES for f in ctrl_files(args.e12, arm, label, st)])):
                c = out["novelty"].setdefault(arm, {}).setdefault(mode, {"n": 0, "novel": 0, "q": 0, "copied": 0})
                for u in (u for f in files for u in f):
                    c["n"] += 1
                    c["novel"] += normalize_for_dedup(u) not in bloom
                    qs = FIELDS["QRY"](u)
                    if qs:
                        c["q"] += 1
                        c["copied"] += qs in known_q
        del known_q
    for arm, parts in out["emission"].items():
        print(f"  emission  {arm:<10} " + "  ".join(
            f"{lab} P {v['PTH']['rate']:.4f} Q {v['QRY']['rate']:.4f}" for lab, v in parts.items()))
    return out


def part_copy(args, arms) -> dict:
    """K12d's memorisation guard: QC1-1's instrument (copy rate, strings seen once,
    the band profile) per arm, on QC1's cells, against the same real pools."""
    import qc1_common as C
    from uie.evaluation import FIELDS
    out = {}
    for label in K.LABELS:
        counts = C.train_counts(label)
        doms = C.train_domains(label) if label == "phishing" else None
        for cell in [c for c in C.CELLS if c[1] == label]:
            mode, _, state = cell
            urls = C.real_pool(label, state)
            node = {"real": C.copy_profile([C.query(u) for u in urls], counts)}
            if doms is not None:
                node["real_domain_disjoint"] = C.copy_profile(
                    [C.query(u) for u in urls if C.registrable(u) not in doms], counts)
            for arm in arms:
                files = (free_files(args.e12, arm, label) if state is None
                         else ctrl_files(args.e12, arm, label, state))
                gq = [q for f in files for q in (FIELDS["QRY"](u) for u in f) if q]
                node[arm] = {"generated": C.copy_profile(gq, counts), "per_generation": len(gq) / K.G}
            out[C.cell_name(*cell)] = node
            print(f"  copy {C.cell_name(*cell)}", flush=True)
        del counts
    return {"copy": out}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--e12", type=Path, default=ROOT / "data" / "e12")
    ap.add_argument("--bundle", type=Path, default=ROOT / "data" / "bundle2")
    ap.add_argument("--part", choices=("free", "control", "misc", "copy"))
    ap.add_argument("--arm", nargs="*")
    ap.add_argument("--label", nargs="*", help="free / control only")
    ap.add_argument("--merge", action="store_true")
    args = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")
    if args.merge:
        merged = {"registration": "research/native/PREREG_E12.md",
                  "scorer": "research/cr1/cr1_score.py (pools, score), unchanged",
                  "e7_free": load(args.e12 / "e7_free_fields.json")}
        for p in sorted(args.e12.glob("score_*.json")):
            if p.name == "score_report.json":
                continue
            for key, val in load(p).items():
                if isinstance(val, dict):
                    for who, v in val.items():
                        merged.setdefault(key, {}).setdefault(who, {}).update(v)
        (args.e12 / "score_report.json").write_text(json.dumps(merged, indent=1, default=float),
                                                     encoding="utf-8")
        print("merged", args.e12 / "score_report.json")
        return
    if not args.part:
        ap.error("--part or --merge")
    arms = tuple(args.arm) if args.arm else ARMS
    report = {"free": part_free, "control": part_control, "misc": part_misc,
              "copy": part_copy}[args.part](args, arms)
    tag = (args.part + ("_" + "_".join(args.arm) if args.arm else "")
           + ("_" + "_".join(args.label) if args.label else ""))
    (args.e12 / f"score_{tag}.json").write_text(json.dumps(report, indent=1, default=float),
                                                encoding="utf-8")
    print("wrote", args.e12 / f"score_{tag}.json")


if __name__ == "__main__":
    main()
