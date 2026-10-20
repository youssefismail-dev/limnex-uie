"""CR1, descriptive: how often the FULL string carries a root slash.

`host/` and `host/?q` have a path of exactly "/", whose body is empty, so the
gate counts them as having NO path (PTH absent) -- yet FULL keeps the slash
(`strip_scheme` deliberately leaves trailing slashes in). The tokenizer
normalises an empty path away (E3_REPORT section 7.3), so generated URLs can
never carry it. Pure counts over the same-state real pool the FULL scorer uses
(name + structure, one URL per host with the gate's seed) and over all 15,000
generated URLs per state; nothing is scored.

For a single binary feature with rates p_real and p_gen, the AUC it gives on its
own is 0.5 + (p_real - p_gen) / 2, so (p_real - p_gen) / 2 is the excess that
feature alone supplies (the floor is ~0.5).

    python research/cr1/cr1_slash.py --gens data/cr1/gen --bundle data/bundle2 \
        --report data/cr1/cr1_report.json --out data/cr1/slash.json
"""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from uie.evaluation import _one_per_host  # noqa: E402
from uie.factorization import parse, strip_scheme  # noqa: E402
from cr1_score import G, NOT_MEASURABLE, SEED, STATES, pools, tag_of  # noqa: E402


def root_slash(u: str) -> bool:
    return (parse(u).path or "") == "/"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gens", type=Path, required=True)
    ap.add_argument("--bundle", type=Path, required=True)
    ap.add_argument("--report", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    rep = json.loads(args.report.read_text(encoding="utf-8"))["states"]
    out = {}
    for label, state in STATES:
        tag = tag_of(label, state)
        if (label, tuple(sorted(state.items()))) in NOT_MEASURABLE:
            continue
        gens = [u for g in range(G) for u in
                json.loads((args.gens / f"{tag}_g{g}.json").read_text(encoding="utf-8"))]
        mixed = pools(args.bundle, label, state)["mixed"]
        one = _one_per_host(mixed, SEED)
        p_real = sum(map(root_slash, one)) / len(one)
        p_gen = sum(map(root_slash, gens)) / len(gens)
        full = rep.get(tag, {}).get("fields", {}).get("FULL", {}).get("acceptance", {})
        out[tag] = {"real_mixed": len(mixed), "real_hosts": len(one),
                    "root_real_raw": sum(map(root_slash, mixed)) / len(mixed),
                    "root_real_1ph": p_real, "root_gen": p_gen,
                    "any_slash_gen": sum("/" in strip_scheme(u) for u in gens) / len(gens),
                    "bound": (p_real - p_gen) / 2, "full_x": full.get("x")}
        print(f"  {tag:<16} real {p_real:.3f}  generated {p_gen:.3f}  "
              f"slash alone {(p_real - p_gen) / 2:+.3f}  FULL x {full.get('x')}")
    args.out.write_text(json.dumps(out, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
