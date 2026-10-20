"""
research/v2/v2_p1_score.py

V2-P1 -- the gate and structure (research/v2/PREREG_V2_P1_MEMORY.md section 4.4): E12's
instruments, unchanged, through research/e17/e17_score.py, pointed at p1_mem's 50 files.

    python research/v2/v2_p1_score.py --v4                     # V4, before any reading
    python research/v2/v2_p1_score.py --part free              # gate v2, FULL FREE
    python research/v2/v2_p1_score.py --part control           # CR1's instrument
    python research/v2/v2_p1_score.py --part misc              # emission, slash, honouring, novelty
    python research/v2/v2_p1_score.py --part copy              # QC1-1's instrument (queries)
    python research/v2/v2_p1_score.py --merge                  # -> data/v2/p1/score_report_p1_mem.json

p1_mem trained on e16_cap's data -- the old corpus plus stream 1 -- so novelty's known URLs
and copying's counts are exactly e16_b5's (e17_score.reference for e16_b5), and the two
engines share one natural rate. The reference is e16_b5's published report
(data/e17/score_report_e16_b5.json); nothing of e16_b5's is rescored.
"""
from __future__ import annotations

import argparse
import collections
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
for p in (ROOT, HERE, ROOT / "research" / "e17", ROOT / "research" / "e16", ROOT / "research" / "e15",
          ROOT / "research" / "e13", ROOT / "research" / "native", ROOT / "research" / "cr1",
          ROOT / "research" / "dc1", ROOT / "research" / "qc1"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import e17_score as S17  # noqa: E402
from v2_arm import P1  # noqa: E402  (V2_ARM: p1 or p1b)

from v2_arm import DATA as P1D  # noqa: E402
SCORE = P1D / "score"
S17.BASES[P1.ARM] = P1D / "generate"                      # <base>/gen
S17.S12.K.SLASH_ARMS = frozenset(set(S17.S12.K.SLASH_ARMS) | {P1.ARM})
REFERENCE = "old corpus + stream 1: e16_cap's training data, which p1_mem trained on (section 4.4)"


def reference(arm: str):
    if arm != P1.ARM:
        raise ValueError(arm)
    return S17.stream_rows(), REFERENCE


S17.reference = reference                                # e17_score.part_misc reads it


def part_copy(ns, arm: str) -> dict:
    """QC1-1's instrument (K12d) for p1_mem's queries against its training counts, with the
    natural rate on the same counts (e17_score.part_copy's computation, without its
    stream-reuse report)."""
    import qc1_common as C
    rows, ref = reference(arm)
    sq = S17.queries(rows)
    out = {}
    for label in S17.S12.K.LABELS:
        old = C.train_counts(label)
        counts = old + sq.get(label, collections.Counter())
        doms = C.train_domains(label) if label == "phishing" else None
        for cell in [c for c in C.CELLS if c[1] == label]:
            mode, _, state = cell
            urls = C.real_pool(label, state)
            node = {"real_stream": C.copy_profile([C.query(u) for u in urls], counts)}
            if doms is not None:
                node["real_domain_disjoint_stream"] = C.copy_profile(
                    [C.query(u) for u in urls if C.registrable(u) not in doms], counts)
            gq = S17.generated_queries(arm, label, state)
            node[arm] = {"generated": C.copy_profile(gq, counts), "per_generation": len(gq) / S17.S12.K.G}
            out[C.cell_name(*cell)] = node
            print(f"  copy {C.cell_name(*cell)}: {arm} {node[arm]['generated']['rate']:.4f} "
                  f"(natural {node['real_stream']['rate']:.4f})", flush=True)
        del old, counts
    return {"copy": out, "copy_reference": {arm: ref}}


def merge() -> Path:
    merged = {"registration": P1.REGISTRATION, "arm": P1.ARM,
              "scorer": "research/native/e12_score.py (E12's instruments) through research/e17/e17_score.py",
              "baseline": "data/e17/score_report_e16_b5.json (e16_b5, the frozen baseline)"}
    for p in sorted((SCORE / P1.ARM).glob("score_*.json")):
        for key, val in json.loads(p.read_text(encoding="utf-8")).items():
            if isinstance(val, dict):
                for who, v in val.items():
                    if isinstance(v, dict):
                        merged.setdefault(key, {}).setdefault(who, {}).update(v)
                    else:
                        merged.setdefault(key, {})[who] = v
    dest = P1D / f"score_report_{P1.ARM}.json"
    dest.write_text(json.dumps(merged, indent=1, default=float), encoding="utf-8")
    return dest


def main() -> None:
    from e13_score import v4
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
    v = json.loads((P1D / "VERIFY_GENERATE.json").read_text(encoding="utf-8"))
    if v.get("verdict") != "PASS":
        raise SystemExit("STOP: VERIFY_GENERATE has not passed; nothing is read")
    if args.merge:
        print("merged", merge())
        return
    if not args.part:
        ap.error("--part, --merge or --v4")
    ns = argparse.Namespace(e12=S17.BASES[P1.ARM], bundle=args.bundle, label=args.label)
    if args.part in ("free", "control"):
        report = {"free": S17.S12.part_free, "control": S17.S12.part_control}[args.part](ns, (P1.ARM,))
    else:
        report = {"misc": S17.part_misc, "copy": part_copy}[args.part](ns, P1.ARM)
    tag = args.part + ("_" + "_".join(args.label) if args.label else "")
    (SCORE / P1.ARM).mkdir(parents=True, exist_ok=True)
    (SCORE / P1.ARM / f"score_{tag}.json").write_text(json.dumps(report, indent=1, default=float),
                                                      encoding="utf-8")
    print("wrote", SCORE / P1.ARM / f"score_{tag}.json")


if __name__ == "__main__":
    main()
