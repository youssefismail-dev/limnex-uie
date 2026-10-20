"""E15-B -- every registered reading for e15_new (research/e15/PREREG_E15.md sections 9.8,
10.6, 13.2), with E12's instruments unchanged (research/native/e12_score.py), pointed at
E15's files. e15_new carries the slash plan, so E12's S-honouring reading applies.

    python research/e15/e15_score.py --v4                 # V4 (section 9.10), before any reading
    python research/e15/e15_score.py --part free          # gate v2, FULL FREE
    python research/e15/e15_score.py --part control       # CR1's instrument
    python research/e15/e15_score.py --part misc          # emission, slash, honouring, novelty
    python research/e15/e15_score.py --part copy          # QC1-1's instrument + uptake
    python research/e15/e15_score.py --merge

Section 13.2: for e15_new, "the model's training data" is the old corpus PLUS the
57,292 injected rows. So
- novelty (G8): the known URLs are the bundle's known-URL set plus the stream's URLs,
  and a copied query is one in the old training queries plus the stream's;
- copying (G7): the counts are the old training queries plus the stream's, and the
  natural rate is recomputed against those same counts (`real_stream`).
The old-corpus-only reading of both is reported beside it. e12_d1p keeps its recorded
numbers (data/e14/score_report.json).

Uptake (section 10.6, reported): the share of generated queries that exactly match a
stream query ABSENT from old training, beside the share matching old training queries
-- for e15_new and, as the chance reference, for e12_d1p (which never saw the stream).
"""
from __future__ import annotations

import argparse
import collections
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
for p in (ROOT, HERE, ROOT / "research" / "e13", ROOT / "research" / "native",
          ROOT / "research" / "cr1", ROOT / "research" / "dc1", ROOT / "research" / "qc1"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import e12_score as S12  # noqa: E402
import e15_common as E15  # noqa: E402
from e13_score import v4  # noqa: E402

E15D = ROOT / "data" / "e15"
SCORE = E15D / "score"
BASE = E15D / "generate"                         # <base>/gen, E12's layout
BASELINE_GEN = ROOT / "data" / "e14" / "decode"  # e12_d1p's files (E14_decode)
STREAM = E15D / "qry_extra" / E15.STREAM_FILE
S12.K.SLASH_ARMS = frozenset(set(S12.K.SLASH_ARMS) | {E15.ARM})


def stream_rows() -> list:
    return [json.loads(line) for line in STREAM.read_text(encoding="utf-8").splitlines() if line.strip()]


def stream_queries() -> dict:
    """label -> Counter of the stream's query strings (FIELDS['QRY'], as every instrument)."""
    from uie.evaluation import FIELDS
    out = collections.defaultdict(collections.Counter)
    for r in stream_rows():
        q = FIELDS["QRY"](r["url"])
        if q:
            out[r["label"]][q] += 1
    return out


class _Known:
    """The bundle's known-URL bloom filter plus the stream's URLs (normalised the same way)."""

    def __init__(self, bloom, extra: set):
        self.bloom, self.extra = bloom, extra

    def __contains__(self, x) -> bool:
        return x in self.extra or x in self.bloom


def part_misc(ns) -> dict:
    """S12.part_misc twice: with section 13.2's reference (the registered reading), then
    unchanged (old corpus only; its novelty is kept as `novelty_old_ref`)."""
    import dc1_compare
    import uie.bloom
    from uie.gate import normalize_for_dedup
    sq = stream_queries()
    surls = {normalize_for_dedup(r["url"]) for r in stream_rows()}
    orig_tq, orig_load = dc1_compare.training_queries, uie.bloom.BloomFilter.load
    try:
        dc1_compare.training_queries = lambda bundle, label: orig_tq(bundle, label) | set(sq.get(label, {}))
        uie.bloom.BloomFilter.load = staticmethod(lambda path, *a, **k: _Known(orig_load(path, *a, **k), surls))
        out = S12.part_misc(ns, (E15.ARM,))
    finally:
        dc1_compare.training_queries, uie.bloom.BloomFilter.load = orig_tq, orig_load
    old = S12.part_misc(ns, (E15.ARM,))
    out["novelty_old_ref"] = old["novelty"]
    out["novelty_reference"] = {E15.ARM: "old corpus + stream (PREREG_E15 section 13.2)"}
    return out


def part_copy(ns) -> dict:
    """QC1-1's instrument (S12.part_copy's loop) for e15_new against old + stream counts,
    with its own natural rate; the old-only reading beside it; uptake for e15_new and
    e12_d1p on every QC1 cell."""
    import qc1_common as C
    from uie.evaluation import FIELDS
    sq = stream_queries()
    out, uptake = {}, {}
    for label in S12.K.LABELS:
        old = C.train_counts(label)
        new = old + sq.get(label, collections.Counter())
        new_only = {q for q in sq.get(label, {}) if q not in old}
        doms = C.train_domains(label) if label == "phishing" else None
        for cell in [c for c in C.CELLS if c[1] == label]:
            mode, _, state = cell
            urls = C.real_pool(label, state)
            real_q = [C.query(u) for u in urls]
            node = {"real_stream": C.copy_profile(real_q, new), "real_old": C.copy_profile(real_q, old)}
            if doms is not None:
                dq = [C.query(u) for u in urls if C.registrable(u) not in doms]
                node["real_domain_disjoint_stream"] = C.copy_profile(dq, new)
            gens = {}
            for arm, base in ((E15.ARM, BASE), ("e12_d1p", BASELINE_GEN)):
                files = (S12.free_files(base, arm, label) if state is None
                         else S12.ctrl_files(base, arm, label, state))
                gens[arm] = [q for f in files for q in (FIELDS["QRY"](u) for u in f) if q]
            gq = gens[E15.ARM]
            node[E15.ARM] = {"generated": C.copy_profile(gq, new), "per_generation": len(gq) / S12.K.G}
            node[f"{E15.ARM}_old_ref"] = {"generated": C.copy_profile(gq, old),
                                          "per_generation": len(gq) / S12.K.G}
            name = C.cell_name(*cell)
            out[name] = node
            uptake[name] = {arm: {"n": len(qs),
                                  "new_stream_only": sum(q in new_only for q in qs) / len(qs) if qs else None,
                                  "old_training": sum(q in old for q in qs) / len(qs) if qs else None,
                                  "distinct_new_stream_only": len({q for q in qs if q in new_only})}
                            for arm, qs in gens.items()}
            u = uptake[name]
            print(f"  copy {name}: e15_new {node[E15.ARM]['generated']['rate']:.4f} "
                  f"(natural {node['real_stream']['rate']:.4f}); uptake e15_new "
                  f"{u[E15.ARM]['new_stream_only']:.4f} vs e12_d1p {u['e12_d1p']['new_stream_only']:.4f}",
                  flush=True)
        del old, new
    return {"copy": out, "uptake": uptake,
            "copy_reference": {E15.ARM: "old training queries + stream (PREREG_E15 section 13.2)"}}


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
        merged = {"registration": E15.REGISTRATION,
                  "scorer": "research/native/e12_score.py (E12's instruments), unchanged",
                  "baseline": "data/e14/score_report.json (e12_d1p, E14's record)"}
        for p in sorted(SCORE.glob("score_*.json")):
            for key, val in json.loads(p.read_text(encoding="utf-8")).items():
                if isinstance(val, dict):
                    for who, v in val.items():
                        if isinstance(v, dict):
                            merged.setdefault(key, {}).setdefault(who, {}).update(v)
                        else:
                            merged.setdefault(key, {})[who] = v
        (E15D / "score_report.json").write_text(json.dumps(merged, indent=1, default=float),
                                                encoding="utf-8")
        print("merged", E15D / "score_report.json")
        return
    if not args.part:
        ap.error("--part, --v4 or --merge")
    ns = argparse.Namespace(e12=BASE, bundle=args.bundle, label=args.label)
    if args.part in ("free", "control"):
        report = {"free": S12.part_free, "control": S12.part_control}[args.part](ns, (E15.ARM,))
    else:
        report = {"misc": part_misc, "copy": part_copy}[args.part](ns)
    tag = args.part + ("_" + "_".join(args.label) if args.label else "")
    (SCORE / f"score_{tag}.json").write_text(json.dumps(report, indent=1, default=float),
                                             encoding="utf-8")
    print("wrote", SCORE / f"score_{tag}.json")


if __name__ == "__main__":
    main()
