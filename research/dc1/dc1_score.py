"""DC1 -- score every generated file with gate v2's own functions (PREREG_DC1 s4).

FULL FREE: the canonical gate's pools (unrestricted), n requested min(n_main,
1,200) -- `cr1_score.score`, whose canonical mode reproduces a published gate v2
record from its own generations (dc1_verify V4). FULL CONTROL: pools restricted
to the state and n_main 3,000 -- exactly as SC1 arm T was scored.

    python research/dc1/dc1_score.py --dc1 data/dc1 --bundle data/bundle2

`--part free` and `--part control` score the two modes in parallel processes
(each writes score_report_<part>.json); `--merge` joins them into score_report.json.
"""
import argparse
import json
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "cr1"))

from dc1_common import G, MEASURABLE, N_CTRL, N_FREE, STATES  # noqa: E402


def summary(fields: dict) -> str:
    return "  ".join(
        f"{f} {r['acceptance']['x']:+.4f} {r['acceptance']['verdict']}"
        if r.get("acceptance", {}).get("x") is not None
        else f"{f} {r.get('acceptance', {}).get('verdict', '--')}" for f, r in fields.items())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dc1", type=Path, required=True)
    ap.add_argument("--bundle", type=Path, required=False)
    ap.add_argument("--part", choices=("all", "free", "control"), default="all")
    ap.add_argument("--merge", action="store_true")
    args = ap.parse_args()
    if args.merge:
        parts = [json.loads((args.dc1 / f"score_report_{p}.json").read_text(encoding="utf-8"))
                 for p in ("free", "control")]
        merged = dict(parts[0], control=parts[1]["control"])
        (args.dc1 / "score_report.json").write_text(json.dumps(merged, indent=1, default=float),
                                                    encoding="utf-8")
        print("merged", args.dc1 / "score_report.json")
        return
    from cr1_score import pools, score

    gens = args.dc1 / "gen"
    out_path = args.dc1 / ("score_report.json" if args.part == "all"
                           else f"score_report_{args.part}.json")
    report = {"registration": "research/dc1/PREREG_DC1.md",
              "scorer": "research/cr1/cr1_score.py (pools, score), unchanged",
              "free": {}, "control": {}}

    def load(name):
        return json.loads((gens / name).read_text(encoding="utf-8"))

    for arm in (("base", "dc1") if args.part in ("all", "free") else ()):
        for label in ("benign", "phishing"):
            names = [f"free_{arm}_{label}_g{g}.json" for g in range(G)]
            if not all((gens / n).is_file() for n in names):
                continue
            t0 = time.time()
            fields, pooled = score(pools(args.bundle, label), [load(n) for n in names],
                                   N_FREE[label])
            report["free"].setdefault(arm, {})[label] = {"fields": fields, "pooled_fields": pooled}
            out_path.write_text(json.dumps(report, indent=1, default=float), encoding="utf-8")
            print(f"  free {arm:<4} {label:<9} {summary(fields)}  ({time.time() - t0:.0f}s)",
                  flush=True)

    for label, states in (MEASURABLE.items() if args.part in ("all", "control") else ()):
        for st in states:
            names = [f"ctrl_dc1_{label}_{st}_g{g}.json" for g in range(G)]
            if not all((gens / n).is_file() for n in names):
                continue
            t0 = time.time()
            pop = pools(args.bundle, label, STATES[st])
            fields, pooled = score(pop, [load(n) for n in names], N_CTRL)
            report["control"][f"{label}_{st}"] = {"label": label, "state": st,
                                                  "real_name_pool": len(pop["name"]),
                                                  "fields": fields, "pooled_fields": pooled}
            out_path.write_text(json.dumps(report, indent=1, default=float), encoding="utf-8")
            print(f"  ctrl dc1  {label}_{st:<5} {summary(fields)}  ({time.time() - t0:.0f}s)",
                  flush=True)
    print("wrote", out_path)


if __name__ == "__main__":
    main()
