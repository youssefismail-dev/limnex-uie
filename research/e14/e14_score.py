"""E14 -- every registered reading, per arm (research/e14/PREREG_E14.md section 3), with
E12's instruments unchanged (research/native/e12_score.py), pointed at E14's files.
Every E14 arm carries the slash plan, so E12's S-honouring reading applies to all of
them (S reconstructed from the seeds, never read from the output).

    python research/e14/e14_score.py --part free --arm e14_q          # gate v2, FULL FREE
    python research/e14/e14_score.py --part control --arm e14_q       # CR1's instrument
    python research/e14/e14_score.py --part misc --arm e14_q          # emission, slash, honouring, novelty
    python research/e14/e14_score.py --part copy --arm e14_q          # QC1-1's instrument
    python research/e14/e14_score.py --v4                             # V4 (section 5), before any reading
    python research/e14/e14_score.py --merge

Layout: data/e14/decode/gen (e12_d1p) and data/e14/generate/gen (e14_q, e14_qd); the
baseline is e12_final's published E12 record (data/e12/score_report.json), whose
scorer V4 proves (E13's V4, re-run).
"""
from __future__ import annotations

import argparse
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
import e14_common as E14  # noqa: E402
from e13_score import v4  # noqa: E402

E14D = ROOT / "data" / "e14"
SCORE = E14D / "score"
BASES = {"e12_d1p": E14D / "decode", "e14_q": E14D / "generate", "e14_qd": E14D / "generate"}
# Every E14 arm is a slash-plan arm (it reads S from the same table).
S12.K.SLASH_ARMS = frozenset(set(S12.K.SLASH_ARMS) | set(E14.ARMS))
PARTS = {"free": S12.part_free, "control": S12.part_control, "misc": S12.part_misc,
         "copy": S12.part_copy}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bundle", type=Path, default=ROOT / "data" / "bundle2")
    ap.add_argument("--part", choices=tuple(PARTS))
    ap.add_argument("--arm", choices=tuple(E14.ARMS))
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
        merged = {"registration": E14.REGISTRATION,
                  "scorer": "research/native/e12_score.py (E12's instruments), unchanged",
                  "baseline": "data/e12/score_report.json (e12_final, E12's published record)"}
        for p in sorted(SCORE.glob("score_*.json")):
            for key, val in json.loads(p.read_text(encoding="utf-8")).items():
                if isinstance(val, dict):
                    for who, v in val.items():
                        merged.setdefault(key, {}).setdefault(who, {}).update(v)
        (E14D / "score_report.json").write_text(json.dumps(merged, indent=1, default=float),
                                                encoding="utf-8")
        print("merged", E14D / "score_report.json")
        return
    if not (args.part and args.arm):
        ap.error("--part and --arm, or --v4, or --merge")
    ns = argparse.Namespace(e12=BASES[args.arm], bundle=args.bundle, label=args.label)
    report = PARTS[args.part](ns, (args.arm,))
    tag = f"{args.part}_{args.arm}" + ("_" + "_".join(args.label) if args.label else "")
    (SCORE / f"score_{tag}.json").write_text(json.dumps(report, indent=1, default=float),
                                             encoding="utf-8")
    print("wrote", SCORE / f"score_{tag}.json")


if __name__ == "__main__":
    main()
