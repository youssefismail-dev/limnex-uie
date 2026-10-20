"""E13 -- every registered reading, per arm (research/e13/PREREG_E13.md sections 3-4),
with E12's instruments unchanged (research/native/e12_score.py), pointed at E13's
files. Every E13 arm carries the slash plan, so E12's S-honouring reading applies
to all of them (S reconstructed from the seeds, never read from the output).

    python research/e13/e13_score.py --part free --arm e12_d1          # gate v2, FULL FREE
    python research/e13/e13_score.py --part control --arm e12_d1       # CR1's instrument
    python research/e13/e13_score.py --part misc --arm e12_d1          # emission, slash, honouring, novelty
    python research/e13/e13_score.py --part copy --arm e12_d1          # QC1-1's instrument
    python research/e13/e13_score.py --v4                              # V4 (section 5), before any reading
    python research/e13/e13_score.py --merge

Layout: E13's Kaggle outputs under data/e13/decode/gen (e12_d1, e12_dec) and
data/e13/ctx/gen (e13_ctx); the baseline is e12_final's published E12 record
(data/e12/score_report.json, the owner's decision 4), whose scorer V4 proves.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
for p in (ROOT, HERE, ROOT / "research" / "native", ROOT / "research" / "cr1",
          ROOT / "research" / "dc1", ROOT / "research" / "qc1"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import e12_score as S12  # noqa: E402
import e13_common as E13  # noqa: E402

E13D = ROOT / "data" / "e13"
SCORE = E13D / "score"
BASES = {"e12_d1": E13D / "decode", "e12_dec": E13D / "decode", "e13_ctx": E13D / "ctx",
         "e12_final": ROOT / "data" / "e12"}
# Every E13 arm is a slash-plan arm (it reads S from the same table): E12's S
# honouring and supplied-plan placement apply to it unchanged.
S12.K.SLASH_ARMS = frozenset(set(S12.K.SLASH_ARMS) | set(E13.ARMS))
PARTS = {"free": S12.part_free, "control": S12.part_control, "misc": S12.part_misc,
         "copy": S12.part_copy}


def x_of(row):
    x = ((row or {}).get("acceptance") or {}).get("x")
    return None if x is None or x != x else float(x)


def v4(bundle: Path) -> dict:
    """V4: e12_final's free cells re-scored from E12's own files, against E12's
    published record; every x must be identical."""
    args = argparse.Namespace(e12=BASES["e12_final"], bundle=bundle, label=None)
    got = S12.part_free(args, ("e12_final",))["free"]["e12_final"]
    pub = json.loads((ROOT / "data" / "e12" / "score_report.json").read_text(encoding="utf-8"))
    pub = pub["free"]["e12_final"]
    diffs = {}
    for label in E13.LABELS:
        for f, row in pub[label]["fields"].items():
            a, b = x_of(got[label]["fields"].get(f)), x_of(row)
            diffs[f"{label} {f}"] = (0.0 if a is None and b is None else
                                     float("inf") if a is None or b is None else abs(a - b))
    worst = max(diffs.values())
    return {"max_abs_diff": worst, "diffs": diffs, "pass": worst <= 1e-9}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bundle", type=Path, default=ROOT / "data" / "bundle2")
    ap.add_argument("--part", choices=tuple(PARTS))
    ap.add_argument("--arm", choices=tuple(E13.ARMS))
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
        merged = {"registration": E13.REGISTRATION,
                  "scorer": "research/native/e12_score.py (E12's instruments), unchanged",
                  "baseline": "data/e12/score_report.json (e12_final, E12's published record)"}
        for p in sorted(SCORE.glob("score_*.json")):
            for key, val in json.loads(p.read_text(encoding="utf-8")).items():
                if isinstance(val, dict):
                    for who, v in val.items():
                        merged.setdefault(key, {}).setdefault(who, {}).update(v)
        (E13D / "score_report.json").write_text(json.dumps(merged, indent=1, default=float),
                                                encoding="utf-8")
        print("merged", E13D / "score_report.json")
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
