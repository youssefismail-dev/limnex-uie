"""E8 (N1) -- every registered reading, per checkpoint (PREREG_E8 section 5).

    python research/native/e8_score.py --part free     # gate v2, FULL FREE, canonical
    python research/native/e8_score.py --part control  # CR1's instrument, native output
    python research/native/e8_score.py --part misc     # honouring, emission, root slash,
                                                       # off-grammar mass, novelty, copying
    python research/native/e8_score.py --merge

Each part writes data/e8/score_<part>.json, so the three can run in parallel
(`--who` narrows a part to some checkpoints, writing score_<part>_<who...>.json);
--merge joins every part file into data/e8/score_report.json.

Gate cells use `research/cr1/cr1_score.py` unchanged: FULL FREE on the
unrestricted pools at the canonical n_main 8,000 (E7's own record comes from
e8_verify's V4, which reproduced HC1 C2 exactly); FULL CONTROL on the state's
pools at n_main 3,000, as delivered -- violating URLs included (section 5).
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "cr1"))
sys.path.insert(0, str(HERE.parent / "dc1"))

import e8_common as K  # noqa: E402

WHO_FREE = tuple(K.ARMS) + ("e7dc1",)          # E7's free record comes from V4
WHO_CTRL = tuple(K.ARMS) + ("e7",)
WHO_ALL = ("e7",) + tuple(K.ARMS) + ("e7dc1",)


def wilson(k: int, n: int) -> dict:
    if not n:
        return {"k": k, "n": n, "rate": None, "lo": None, "hi": None}
    p, z = k / n, 1.959964
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return {"k": k, "n": n, "rate": p, "lo": c - h, "hi": c + h}


def load(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def free_files(e8: Path, who: str, label: str) -> list:
    if who == "e7":
        cache = ROOT / "data" / "gen_cache_e7_gpu"
        return [load(p) for p in sorted(cache.glob(f"*_{label}_n8000_*.json"))]
    return [load(e8 / "gen" / f"free_{who}_{label}_g{g}.json") for g in range(K.G)]


def ctrl_files(e8: Path, who: str, label: str, state: str) -> list:
    return [load(e8 / "gen" / f"ctrl_{who}_{label}_{state}_g{g}.json") for g in range(K.G)]


def presence(u: str):
    from uie.evaluation import FIELDS
    return bool(FIELDS["PTH"](u)), bool(FIELDS["QRY"](u))


def root_slash(u: str):
    """None when the URL has a path body; else (has_query, written_with_root_slash)."""
    from uie.evaluation import _parse, _path_body
    p = _parse(u)
    if _path_body(p.path or ""):
        return None
    cut = min([i for i in (u.find("?"), u.find("#")) if i >= 0] or [len(u)])
    return bool(p.query), u[:cut].endswith("/")


def _cell(field: str, row: dict) -> str:
    """One cell for a progress line. A starved cell has no per-generation x (phishing
    QRY at the canonical size, for instance) -- it prints its verdict, never crashes."""
    a = (row or {}).get("acceptance", {})
    x = a.get("x")
    if x is None or x != x:
        return f"{field} {a.get('verdict', '--')}"
    return f"{field} {x:+.4f} {a.get('verdict')}"


def part_free(args, whos) -> dict:
    from cr1_score import pools, score
    out = {}
    for who in whos:
        for label in K.LABELS:
            t0 = time.time()
            fields, pooled = score(pools(args.bundle, label), free_files(args.e8, who, label), K.N_FREE)
            out.setdefault(who, {})[label] = {"fields": fields, "pooled_fields": pooled}
            print(f"  free {who:<8} {label:<9} " + "  ".join(
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
                fields, pooled = score(pop, ctrl_files(args.e8, who, label, st), K.N_CTRL)
                out.setdefault(who, {})[f"{label}_{st}"] = {
                    "label": label, "state": st, "real_name_pool": len(pop["name"]),
                    "fields": fields, "pooled_fields": pooled}
                print(f"  ctrl {who:<8} {label}_{st:<5} " + "  ".join(
                    _cell(f, r) for f, r in fields.items()) + f"  ({time.time() - t0:.0f}s)",
                    flush=True)
    return {"control": out}


def part_misc(args, whos) -> dict:
    from dc1_compare import training_queries
    from uie.bloom import BloomFilter
    from uie.evaluation import FIELDS
    from uie.gate import normalize_for_dedup

    gprov = {r["file"]: r for r in load(args.e8 / "gen" / "PROVENANCE.json")["files"]}
    bloom = BloomFilter.load(args.bundle / "known_urls.bloom.gz")
    out = {"honouring": {}, "emission": {}, "root_slash": {}, "off_grammar": {}, "novelty": {}}
    for who in whos:
        # honouring -- every joint state, per generation and pooled
        if who in WHO_CTRL:
            for label in K.LABELS:
                for st, want in K.STATES.items():
                    per_g, k_all, n_all = [], 0, 0
                    for urls in ctrl_files(args.e8, who, label, st):
                        k = sum(presence(u) == (want["PTH"], want["QRY"]) for u in urls)
                        per_g.append(k / len(urls))
                        k_all, n_all = k_all + k, n_all + len(urls)
                    out["honouring"].setdefault(who, {})[f"{label}_{st}"] = {
                        **wilson(k_all, n_all), "per_generation": per_g}
        # emission and root slash -- FULL FREE (E7 from HC1's canonical files)
        for label in K.LABELS:
            urls = [u for f in free_files(args.e8, who, label) for u in f]
            p = sum(presence(u)[0] for u in urls)
            q = sum(presence(u)[1] for u in urls)
            out["emission"].setdefault(who, {})[label] = {"PTH": wilson(p, len(urls)),
                                                          "QRY": wilson(q, len(urls))}
            no_body = [b for b in (root_slash(u) for u in urls) if b is not None]
            bare = [slash for has_q, slash in no_body if not has_q]
            qnp = [slash for has_q, slash in no_body if has_q]
            out["root_slash"].setdefault(who, {})[label] = {
                "bare": wilson(sum(bare), len(bare)),
                "query_no_path": wilson(sum(qnp), len(qnp))}
        # off-grammar mass -- plan arms, from the generation records
        if K.ARMS.get(who, {}).get("presence_plan"):
            recs = [r for r in gprov.values() if r["who"] == who and r.get("plan_off_mass")]
            for mode in ("free", "control"):
                rs = [r["plan_off_mass"] for r in recs if r["mode"] == mode]
                if rs:
                    out["off_grammar"].setdefault(who, {})[mode] = {
                        "mean": sum(r["mean"] for r in rs) / len(rs), "max": max(r["max"] for r in rs)}
    # novelty and query copying -- per checkpoint and mode, both classes together
    for label in K.LABELS:
        known_q = training_queries(args.bundle, label)
        for who in whos:
            modes = [("free", free_files(args.e8, who, label))]
            if who in WHO_CTRL:
                modes.append(("control", [f for st in K.STATES for f in ctrl_files(args.e8, who, label, st)]))
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
        print(f"  honouring {who:<8} " + "  ".join(f"{k} {v['rate']:.4f}" for k, v in parts.items()))
    for who, parts in out["emission"].items():
        print(f"  emission  {who:<8} " + "  ".join(
            f"{l} P {v['PTH']['rate']:.4f} Q {v['QRY']['rate']:.4f}" for l, v in parts.items()))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--e8", type=Path, default=ROOT / "data" / "e8")
    ap.add_argument("--bundle", type=Path, default=ROOT / "data" / "bundle2")
    ap.add_argument("--part", choices=("free", "control", "misc"))
    ap.add_argument("--who", nargs="*")
    ap.add_argument("--merge", action="store_true")
    args = ap.parse_args()
    if args.merge:
        merged = {"registration": "research/native/PREREG_E8.md",
                  "scorer": "research/cr1/cr1_score.py (pools, score), unchanged",
                  "e7_free": load(args.e8 / "e7_free_fields.json")}
        for p in sorted(args.e8.glob("score_*.json")):
            if p.name == "score_report.json":
                continue
            for key, val in load(p).items():
                if isinstance(val, dict):
                    for who, v in val.items():
                        merged.setdefault(key, {}).setdefault(who, {}).update(v)
        (args.e8 / "score_report.json").write_text(json.dumps(merged, indent=1, default=float),
                                                    encoding="utf-8")
        print("merged", args.e8 / "score_report.json")
        return
    if not args.part:
        ap.error("--part or --merge")
    whos = tuple(args.who) if args.who else {"free": WHO_FREE, "control": WHO_CTRL,
                                             "misc": WHO_ALL}[args.part]
    report = {"free": part_free, "control": part_control, "misc": part_misc}[args.part](args, whos)
    tag = args.part + ("_" + "_".join(args.who) if args.who else "")
    (args.e8 / f"score_{tag}.json").write_text(json.dumps(report, indent=1, default=float),
                                               encoding="utf-8")
    print("wrote", args.e8 / f"score_{tag}.json")


if __name__ == "__main__":
    main()
