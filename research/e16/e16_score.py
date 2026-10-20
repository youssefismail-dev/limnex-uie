"""E16 -- every registered reading for e16_cap (research/e16/PREREG_E16.md sections 4-5),
with E12's instruments unchanged (research/native/e12_score.py), pointed at E16's files.

    python research/e16/e16_score.py --v4                 # V4, before any reading
    python research/e16/e16_score.py --part free          # gate v2, FULL FREE
    python research/e16/e16_score.py --part control       # CR1's instrument
    python research/e16/e16_score.py --part misc          # emission, slash, honouring, novelty
    python research/e16/e16_score.py --part copy          # QC1-1's instrument
    python research/e16/e16_score.py --merge

Both e16_cap and e15_new trained on the old corpus plus E15's stream, so "the model's
training data" (PREREG_E15.md section 13.2) is that for both: novelty's known URLs and
copying's counts include the stream, exactly as e15_new's recorded readings did
(data/e15/score_report.json). Reported beside copying: the share of generated queries
that are stream-only strings, for e16_cap and e15_new (descriptive).
"""
from __future__ import annotations

import argparse
import collections
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
for p in (ROOT, HERE, ROOT / "research" / "e15", ROOT / "research" / "e13", ROOT / "research" / "native",
          ROOT / "research" / "cr1", ROOT / "research" / "dc1", ROOT / "research" / "qc1"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import e12_score as S12  # noqa: E402
import e16_common as E16  # noqa: E402
from e13_score import v4  # noqa: E402
from e15_score import _Known, stream_queries, stream_rows  # noqa: E402

E16D = ROOT / "data" / "e16"
SCORE = E16D / "score"
BASE = E16D / "generate"                                   # <base>/gen, E12's layout
BASELINE_GEN = ROOT / "data" / "e15" / "generate"          # e15_new's files
S12.K.SLASH_ARMS = frozenset(set(S12.K.SLASH_ARMS) | {E16.ARM})


def part_misc(ns) -> dict:
    """S12.part_misc with the old corpus + stream as the training reference."""
    import dc1_compare
    import uie.bloom
    from uie.gate import normalize_for_dedup
    sq = stream_queries()
    surls = {normalize_for_dedup(r["url"]) for r in stream_rows()}
    orig_tq, orig_load = dc1_compare.training_queries, uie.bloom.BloomFilter.load
    try:
        dc1_compare.training_queries = lambda bundle, label: orig_tq(bundle, label) | set(sq.get(label, {}))
        uie.bloom.BloomFilter.load = staticmethod(lambda path, *a, **k: _Known(orig_load(path, *a, **k), surls))
        out = S12.part_misc(ns, (E16.ARM,))
    finally:
        dc1_compare.training_queries, uie.bloom.BloomFilter.load = orig_tq, orig_load
    out["novelty_reference"] = {E16.ARM: "old corpus + stream (PREREG_E15 section 13.2)"}
    return out


def part_copy(ns) -> dict:
    """QC1-1's instrument for e16_cap against old + stream counts, with the natural rate
    on the same counts; stream-only reuse for both models, reported."""
    import qc1_common as C
    from uie.evaluation import FIELDS
    sq = stream_queries()
    out, reuse = {}, {}
    for label in S12.K.LABELS:
        old = C.train_counts(label)
        counts = old + sq.get(label, collections.Counter())
        stream_only = {q for q in sq.get(label, {}) if q not in old}
        doms = C.train_domains(label) if label == "phishing" else None
        for cell in [c for c in C.CELLS if c[1] == label]:
            mode, _, state = cell
            urls = C.real_pool(label, state)
            node = {"real_stream": C.copy_profile([C.query(u) for u in urls], counts)}
            if doms is not None:
                node["real_domain_disjoint_stream"] = C.copy_profile(
                    [C.query(u) for u in urls if C.registrable(u) not in doms], counts)
            gens = {}
            for arm, base in ((E16.ARM, BASE), (E16.BASELINE, BASELINE_GEN)):
                files = (S12.free_files(base, arm, label) if state is None
                         else S12.ctrl_files(base, arm, label, state))
                gens[arm] = [q for f in files for q in (FIELDS["QRY"](u) for u in f) if q]
            node[E16.ARM] = {"generated": C.copy_profile(gens[E16.ARM], counts),
                             "per_generation": len(gens[E16.ARM]) / S12.K.G}
            name = C.cell_name(*cell)
            out[name] = node
            reuse[name] = {arm: {"n": len(qs),
                                 "stream_only": sum(q in stream_only for q in qs) / len(qs) if qs else None,
                                 "old_training": sum(q in old for q in qs) / len(qs) if qs else None}
                           for arm, qs in gens.items()}
            print(f"  copy {name}: e16_cap {node[E16.ARM]['generated']['rate']:.4f} "
                  f"(natural {node['real_stream']['rate']:.4f})", flush=True)
        del old, counts
    return {"copy": out, "stream_reuse": reuse,
            "copy_reference": {E16.ARM: "old training queries + stream (PREREG_E15 section 13.2)"}}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bundle", type=Path, default=ROOT / "data" / "bundle2")
    ap.add_argument("--part", choices=("free", "control", "misc", "copy"))
    ap.add_argument("--label", nargs="*", help="free / control only")
    ap.add_argument("--v4", action="store_true")
    ap.add_argument("--merge", action="store_true")
    args = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")
    SCORE.mkdir(parents=True, exist_ok=True)
    if args.v4:
        r = v4(args.bundle)
        (SCORE / "v4.json").write_text(json.dumps(r, indent=1, default=float), encoding="utf-8")
        print(f"V4: max |x - published| {r['max_abs_diff']:.2e} -> {'PASS' if r['pass'] else 'FAIL'}")
        return
    if args.merge:
        merged = {"registration": E16.REGISTRATION,
                  "scorer": "research/native/e12_score.py (E12's instruments), unchanged",
                  "baseline": "data/e15/score_report.json (e15_new, the official baseline)"}
        for p in sorted(SCORE.glob("score_*.json")):
            for key, val in json.loads(p.read_text(encoding="utf-8")).items():
                if isinstance(val, dict):
                    for who, v in val.items():
                        if isinstance(v, dict):
                            merged.setdefault(key, {}).setdefault(who, {}).update(v)
                        else:
                            merged.setdefault(key, {})[who] = v
        (E16D / "score_report.json").write_text(json.dumps(merged, indent=1, default=float), encoding="utf-8")
        print("merged", E16D / "score_report.json")
        return
    if not args.part:
        ap.error("--part, --v4 or --merge")
    ns = argparse.Namespace(e12=BASE, bundle=args.bundle, label=args.label)
    if args.part in ("free", "control"):
        report = {"free": S12.part_free, "control": S12.part_control}[args.part](ns, (E16.ARM,))
    else:
        report = {"misc": part_misc, "copy": part_copy}[args.part](ns)
    tag = args.part + ("_" + "_".join(args.label) if args.label else "")
    (SCORE / f"score_{tag}.json").write_text(json.dumps(report, indent=1, default=float), encoding="utf-8")
    print("wrote", SCORE / f"score_{tag}.json")


if __name__ == "__main__":
    main()
