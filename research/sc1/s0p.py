"""SC1 V6 (S0'): the laptop must still produce CR1's numbers (PREREG_SC1 section 6).

T - CR1 pairs on CR1's real samples and floors, so the environment that scores
SC1 must reproduce them. CR1's scorer, called on CR1's own generations for two
states, must return `data/cr1/cr1_report.json`'s records for them exactly: every
field's draws and acceptance block, and the pooled rows (NaN equal to NaN).

    python research/sc1/s0p.py --cr1 data/cr1 --bundle data/bundle2 --out data/sc1
"""
import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "cr1"))

from s0_compare import same  # noqa: E402
from sc1_score import score_states  # noqa: E402

S0P_STATES = ("benign_P0", "phishing_P1")


def differences(fresh: dict, reference: dict) -> list:
    """Every (state, part, field) whose record is not identical."""
    fresh = json.loads(json.dumps(fresh, default=float))   # as cr1_score writes it
    bad = []
    for tag, rec in fresh.items():
        ref = reference[tag]
        for part in ("fields", "pooled_fields"):
            if set(rec[part]) != set(ref[part]):
                bad.append((tag, part, "field set"))
            for field in rec[part]:
                if not same(rec[part][field], ref[part].get(field)):
                    bad.append((tag, part, field))
    return bad


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cr1", type=Path, required=True)
    ap.add_argument("--bundle", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    reference = json.loads((args.cr1 / "cr1_report.json").read_text(encoding="utf-8"))["states"]
    fresh = score_states(args.cr1 / "gen", args.bundle, S0P_STATES)
    bad = differences(fresh, reference)
    verdict = "PASS" if not bad else "FAIL"
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "s0p.json").write_text(json.dumps(
        {"registration": "research/sc1/PREREG_SC1.md (V6)", "states": list(S0P_STATES),
         "differences": bad, "verdict": verdict}, indent=1), encoding="utf-8")
    print(f"S0': {len(S0P_STATES)} CR1 states re-scored; differences: {bad or 'none'}")
    print("S0':", verdict)
    sys.exit(0 if not bad else 1)


if __name__ == "__main__":
    main()
