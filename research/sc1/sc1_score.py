"""SC1 -- score named states with CR1's scorer, unchanged.

`cr1_score.py`'s command line scores all 14 states and refuses a missing file:
right for an arm, wrong for S0' (two CR1 states, PREREG_SC1 V6) and for a
replication (the flagged states only, section 7.4). This calls the same
functions -- `cr1_score.pools` and `cr1_score.score` -- for the named states and
records them exactly as `cr1_score.py` records a state.

    python research/sc1/sc1_score.py --gens data/sc1/T_rep/gen --bundle data/bundle2 \
        --states phishing_P1 --out data/sc1/T_rep
"""
import argparse
import json
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE.parent / "cr1"))

import cr1_score  # noqa: E402


def score_states(gens: Path, bundle: Path, tags) -> dict:
    """{tag: the record cr1_score.py writes for that state}."""
    by_tag = {cr1_score.tag_of(label, state): (label, state)
              for label, state in cr1_score.STATES}
    out = {}
    for tag in tags:
        label, state = by_tag[tag]
        t0 = time.time()
        gens_ = [json.loads((gens / f"{tag}_g{g}.json").read_text(encoding="utf-8"))
                 for g in range(cr1_score.G)]
        assert all(len(x) == cr1_score.N_MAIN for x in gens_), tag
        assert all(cr1_score.in_state(u, state) for x in gens_ for u in x), \
            f"{tag}: non-compliant URL"
        pop = cr1_score.pools(bundle, label, state)
        fields, pooled = cr1_score.score(pop, gens_, cr1_score.N_MAIN)
        out[tag] = {"label": label, "state": state, "real_name_pool": len(pop["name"]),
                    "fields": fields, "pooled_fields": pooled,
                    "seconds": round(time.time() - t0, 1)}
        print(f"  {tag:<16} scored ({out[tag]['seconds']}s)", flush=True)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gens", type=Path, required=True)
    ap.add_argument("--bundle", type=Path, required=True)
    ap.add_argument("--states", required=True, help="comma-separated tags")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    states = score_states(args.gens, args.bundle, args.states.split(","))
    (args.out / "report.json").write_text(json.dumps(
        {"registration": "research/sc1/PREREG_SC1.md",
         "scorer": "research/cr1/cr1_score.py (pools, score), unchanged",
         "G": cr1_score.G, "D": cr1_score.D, "n_main": cr1_score.N_MAIN,
         "seed": cr1_score.SEED, "states": states}, indent=1, default=float),
        encoding="utf-8")


if __name__ == "__main__":
    main()
