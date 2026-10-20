"""E17 -- every registered reading for e16_b5 and e17_a (research/e17/PREREG_E17.md
sections 2-4), with E12's instruments unchanged (research/native/e12_score.py), pointed at
E17's files.

    python research/e17/e17_score.py --v4                         # V4, before any reading
    python research/e17/e17_score.py --arm e16_b5 --part free      # gate v2, FULL FREE
    python research/e17/e17_score.py --arm e16_b5 --part control   # CR1's instrument
    python research/e17/e17_score.py --arm e16_b5 --part misc      # emission, slash, honouring, novelty
    python research/e17/e17_score.py --arm e16_b5 --part copy      # QC1-1's instrument
    python research/e17/e17_score.py --arm e16_b5 --merge          # -> data/e17/score_report_e16_b5.json
    (the same with --arm e17_a)

"The model's training data" (PREREG_E15.md section 13.2) is each model's own:
- e16_b5 is e16_cap's weights, trained on the old corpus + stream 1: novelty's known URLs
  and copying's counts include stream 1, exactly as E16 read e16_cap (e15_score's stream
  file), so e16_b5 and e16_cap share one natural rate.
- e17_a trained on the old corpus + streams 1 and 2 (the pooled file).
Reported beside e17_a's copying: uptake -- the share of generated queries that are
stream-2-only strings (in stream 2, not in the old corpus, not in stream 1), and stream-1-
only strings, for e17_a and for e16_b5 and e16_cap, which never saw stream 2 (the chance
reference).
"""
from __future__ import annotations

import argparse
import collections
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
for p in (ROOT, HERE, ROOT / "research" / "e16", ROOT / "research" / "e15", ROOT / "research" / "e13",
          ROOT / "research" / "native", ROOT / "research" / "cr1", ROOT / "research" / "dc1",
          ROOT / "research" / "qc1"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import e12_score as S12  # noqa: E402
import e17_common as E17  # noqa: E402
from e13_score import v4  # noqa: E402
from e15_score import _Known, stream_queries, stream_rows  # noqa: E402

E17D = ROOT / "data" / "e17"
SCORE = E17D / "score"
BASES = {E17.DECODE_ARM: E17D / "decode", E17.TRAIN_ARM: E17D / "generate"}   # <base>/gen
E16_BASE = ROOT / "data" / "e16" / "generate"                                 # e16_cap's files
POOLED = E17.STREAM2_DIR / E17.STREAM_FILE
S12.K.SLASH_ARMS = frozenset(set(S12.K.SLASH_ARMS) | set(BASES))


def pooled_rows() -> list:
    return [json.loads(line) for line in POOLED.read_text(encoding="utf-8").splitlines() if line.strip()]


def queries(rows) -> dict:
    """label -> Counter of query strings (FIELDS['QRY'], as every instrument)."""
    from uie.evaluation import FIELDS
    out = collections.defaultdict(collections.Counter)
    for r in rows:
        q = FIELDS["QRY"](r["url"])
        if q:
            out[r["label"]][q] += 1
    return out


def reference(arm: str):
    """The arm's added training rows: stream 1 (E15's file) for e16_b5, the pooled file
    (streams 1 and 2) for e17_a."""
    if arm == E17.DECODE_ARM:
        return stream_rows(), "old corpus + stream 1 (e16_cap's weights; PREREG_E15 section 13.2)"
    return pooled_rows(), "old corpus + streams 1 and 2 (PREREG_E17 section 3.3)"


def check_stream_1() -> None:
    """The pooled file's stream-1 rows are E15's stream file, URL for URL."""
    a = sorted((r["label"], r["url"]) for r in stream_rows())
    b = sorted((r["label"], r["url"]) for r in pooled_rows() if r["stream"] == 1)
    if a != b:
        raise SystemExit("STOP: the pooled file's stream 1 is not E15's stream file")


def part_misc(ns, arm: str) -> dict:
    """S12.part_misc with the arm's own training reference."""
    import dc1_compare
    import uie.bloom
    from uie.gate import normalize_for_dedup
    rows, ref = reference(arm)
    sq = queries(rows)
    surls = {normalize_for_dedup(r["url"]) for r in rows}
    orig_tq, orig_load = dc1_compare.training_queries, uie.bloom.BloomFilter.load
    try:
        dc1_compare.training_queries = lambda bundle, label: orig_tq(bundle, label) | set(sq.get(label, {}))
        uie.bloom.BloomFilter.load = staticmethod(lambda path, *a, **k: _Known(orig_load(path, *a, **k), surls))
        out = S12.part_misc(ns, (arm,))
    finally:
        dc1_compare.training_queries, uie.bloom.BloomFilter.load = orig_tq, orig_load
    out["novelty_reference"] = {arm: ref}
    return out


def generated_queries(arm: str, label: str, state) -> list:
    from uie.evaluation import FIELDS
    base = BASES.get(arm, E16_BASE)
    files = S12.free_files(base, arm, label) if state is None else S12.ctrl_files(base, arm, label, state)
    return [q for f in files for q in (FIELDS["QRY"](u) for u in f) if q]


def part_copy(ns, arm: str) -> dict:
    """QC1-1's instrument for the arm against its own training counts, with the natural
    rate on the same counts. e16_b5: stream-only reuse as E16 reported it, for e16_b5 and
    e16_cap. e17_a: uptake of stream-1-only and stream-2-only strings for e17_a, e16_b5
    (if generated) and e16_cap."""
    import qc1_common as C
    rows, ref = reference(arm)
    sq = queries(rows)
    by_stream = {s: queries([r for r in rows if r.get("stream", 1) == s]) for s in (1, 2)}
    out, reuse = {}, {}
    for label in S12.K.LABELS:
        old = C.train_counts(label)
        counts = old + sq.get(label, collections.Counter())
        s1, s2 = by_stream[1].get(label, {}), by_stream[2].get(label, {})
        only1 = {q for q in s1 if q not in old and q not in s2}
        only2 = {q for q in s2 if q not in old and q not in s1}
        doms = C.train_domains(label) if label == "phishing" else None
        for cell in [c for c in C.CELLS if c[1] == label]:
            mode, _, state = cell
            urls = C.real_pool(label, state)
            node = {"real_stream": C.copy_profile([C.query(u) for u in urls], counts)}
            if doms is not None:
                node["real_domain_disjoint_stream"] = C.copy_profile(
                    [C.query(u) for u in urls if C.registrable(u) not in doms], counts)
            gq = generated_queries(arm, label, state)
            node[arm] = {"generated": C.copy_profile(gq, counts), "per_generation": len(gq) / S12.K.G}
            name = C.cell_name(*cell)
            out[name] = node
            who = [arm, "e16_cap"] + ([E17.DECODE_ARM] if arm == E17.TRAIN_ARM
                                      and (BASES[E17.DECODE_ARM] / "gen" / "PROVENANCE.json").is_file() else [])
            gens = {a: (gq if a == arm else generated_queries(a, label, state)) for a in who}
            if arm == E17.DECODE_ARM:
                reuse[name] = {a: {"n": len(qs),
                                   "stream_only": sum(q in only1 for q in qs) / len(qs) if qs else None,
                                   "old_training": sum(q in old for q in qs) / len(qs) if qs else None}
                               for a, qs in gens.items()}
            else:
                reuse[name] = {a: {"n": len(qs),
                                   "stream_1_only": sum(q in only1 for q in qs) / len(qs) if qs else None,
                                   "stream_2_only": sum(q in only2 for q in qs) / len(qs) if qs else None,
                                   "distinct_stream_2_only": len({q for q in qs if q in only2}),
                                   "old_training": sum(q in old for q in qs) / len(qs) if qs else None}
                               for a, qs in gens.items()}
            u = reuse[name]
            extra = ("" if arm == E17.DECODE_ARM else
                     "; stream-2-only " + ", ".join(f"{a} {v['stream_2_only']:.4f}" for a, v in u.items()
                                                    if v["stream_2_only"] is not None))
            print(f"  copy {name}: {arm} {node[arm]['generated']['rate']:.4f} "
                  f"(natural {node['real_stream']['rate']:.4f}){extra}", flush=True)
        del old, counts
    key = "stream_reuse" if arm == E17.DECODE_ARM else "uptake"
    return {"copy": out, key: reuse, "copy_reference": {arm: ref}}


def merge(arm: str) -> Path:
    merged = {"registration": E17.REGISTRATION, "arm": arm,
              "scorer": "research/native/e12_score.py (E12's instruments), unchanged",
              "baseline": "data/e16/score_report.json (e16_cap, the official baseline)"}
    for p in sorted((SCORE / arm).glob("score_*.json")):
        for key, val in json.loads(p.read_text(encoding="utf-8")).items():
            if isinstance(val, dict):
                for who, v in val.items():
                    if isinstance(v, dict):
                        merged.setdefault(key, {}).setdefault(who, {}).update(v)
                    else:
                        merged.setdefault(key, {})[who] = v
    dest = E17D / f"score_report_{arm}.json"
    dest.write_text(json.dumps(merged, indent=1, default=float), encoding="utf-8")
    return dest


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bundle", type=Path, default=ROOT / "data" / "bundle2")
    ap.add_argument("--arm", choices=tuple(BASES))
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
    if not args.arm:
        ap.error("--arm (with --part or --merge), or --v4")
    step = "DECODE" if args.arm == E17.DECODE_ARM else "GENERATE"
    v = json.loads((E17D / f"VERIFY_{step}.json").read_text(encoding="utf-8"))
    if v.get("verdict") != "PASS":
        raise SystemExit(f"STOP: VERIFY_{step} has not passed; nothing is read")
    if args.merge:
        print("merged", merge(args.arm))
        return
    if not args.part:
        ap.error("--part or --merge")
    if args.arm == E17.TRAIN_ARM:
        check_stream_1()
    ns = argparse.Namespace(e12=BASES[args.arm], bundle=args.bundle, label=args.label)
    if args.part in ("free", "control"):
        report = {"free": S12.part_free, "control": S12.part_control}[args.part](ns, (args.arm,))
    else:
        report = {"misc": part_misc, "copy": part_copy}[args.part](ns, args.arm)
    tag = args.part + ("_" + "_".join(args.label) if args.label else "")
    (SCORE / args.arm).mkdir(parents=True, exist_ok=True)
    (SCORE / args.arm / f"score_{tag}.json").write_text(json.dumps(report, indent=1, default=float),
                                                        encoding="utf-8")
    print("wrote", SCORE / args.arm / f"score_{tag}.json")


if __name__ == "__main__":
    main()
