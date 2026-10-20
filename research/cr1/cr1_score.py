"""CR1 -- the absolute realism of controlled output, scored by gate v2 itself.

Registered in research/cr1/PREREG_CR1.md. For each control state: G = 5
generations of compliant URLs, each scored per field by the canonical gate's
own functions -- `field_report` (the gate's per-draw scorer), `split_seed` (v2:
an independent floor per generation and split), `_aggregate`, `s261_verdict`
and the pooled fallback -- with ONE difference: the real pools are restricted to
the state, so a control is judged against real URLs that look like its request.

    --canonical  unrestricted pools and the gate's n_main: check S0, which must
                 reproduce a gate v2 report exactly from its own generations

    python research/cr1/cr1_score.py --gens data/cr1/gen --bundle data/bundle2 --out data/cr1
    python research/cr1/cr1_score.py --canonical --gens data/gen_cache_e7_gpu \
        --label phishing --bundle data/bundle2 --out data/cr1/s0
"""
import argparse
import gc
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from uie.corpus.run_eval import (_aggregate, _find, _rows, pooled_verdict,  # noqa: E402
                                 s261_verdict, split_seed)
from uie.evaluation import (FIELD_POPULATION, FIELDS, GENERATOR_FRAME,  # noqa: E402
                            discriminate_with_floor, field_report)

SEED = 20260806                 # the canonical gate's seed
G, D = 5, 10                    # gate v2's canonical G and D
N_MAIN = 3000                   # compliant URLs per generation (PREREG_CR1)

#: The 16 control states, in registration order; state index s fixes the seeds.
STATES = ([(label, {f: want}) for label in ("benign", "phishing")
           for f in ("PTH", "QRY") for want in (True, False)]
          + [(label, {"PTH": p, "QRY": q}) for label in ("benign", "phishing")
             for p in (True, False) for q in (True, False)])
#: Registered as not measurable: too few real URLs in the state (< 3 x MIN_N hosts).
NOT_MEASURABLE = {("phishing", (("PTH", True), ("QRY", True))),
                  ("phishing", (("PTH", False), ("QRY", True)))}


def tag_of(label, state):
    return label + "_" + "_".join(f"{f[0]}{int(v)}" for f, v in state.items())


def gen_seed(s: int, g: int) -> int:
    """Disjoint blocks of 1e9: a request's redraw rounds (+1,000,003*k, k <= 100)
    and suffix groups (+i) never reach the next block."""
    return 20261001 + (100 * s + g) * 1_000_000_000


def in_state(url, state) -> bool:
    return all(bool(FIELDS[f](url)) == want for f, want in state.items())


def score(populations, generations, n_main):
    """The canonical gate's scoring loop (uie/corpus/run_eval.py, estimator v2),
    for one class: G x D field_report draws, _aggregate, the section-26.1 verdict
    per field, and the pooled fallback for starved fields."""
    draws, every = {}, []
    for g, fake_g in enumerate(generations):
        every.extend(fake_g)
        for dsi in range(D):
            r = field_report(populations, fake_g, n=min(n_main, 1200),
                             seed=split_seed(SEED, g, dsi, "v2"))
            gc.collect()                          # memory only, as in the gate
            for name, row in r.items():
                draws.setdefault(name, []).append(row)
    fields = _aggregate(draws)
    for name, rows in draws.items():
        by_gen = [[r["excess"] for r in rows[g * D:(g + 1) * D] if not r.get("insufficient")]
                  for g in range(len(generations))]
        acc = s261_verdict(by_gen)
        if name == "FULL":
            acc["verdict"] = "AUXILIARY"
        fields[name]["acceptance"] = acc
    starved = sorted(k for k, v in fields.items() if v.get("insufficient"))
    pooled = {}
    if starved and len(every) > n_main:
        pdraws = {}
        for dsi in range(D):
            for name in starved:
                pop = FIELD_POPULATION[name]
                if not populations.get(pop):
                    continue
                row = discriminate_with_floor(populations[pop], every, name,
                                              n=min(n_main, 1200), seed=SEED + dsi)
                gc.collect()
                row["population"] = pop
                row["frame"] = GENERATOR_FRAME
                pdraws.setdefault(name, []).append(row)
        pooled = _aggregate(pdraws)
        for name, rows in pdraws.items():
            if name in pooled:
                pooled[name]["d3_verdict"] = pooled_verdict(
                    [r["excess"] for r in rows[:3] if not r.get("insufficient")])
                pooled[name]["verdict"] = (
                    pooled_verdict(pooled[name]["excess_draws"])
                    if not pooled[name].get("insufficient") else "NOT COMPUTABLE")
    return fields, pooled


def pools(bundle: Path, label: str, state=None) -> dict:
    prefix = "" if label == "benign" else f"{label}_"
    name = _rows(_find(bundle, f"{prefix}name_val"))
    structure = _rows(_find(bundle, f"{prefix}structure_val"))
    if state:
        name = [u for u in name if in_state(u, state)]
        structure = [u for u in structure if in_state(u, state)]
    return {"name": name, "structure": structure, "mixed": name + structure}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gens", type=Path, required=True)
    ap.add_argument("--bundle", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--canonical", action="store_true")
    ap.add_argument("--label", default="phishing", help="--canonical only")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    if args.canonical:                    # S0: the gate's own inputs, unrestricted
        files = sorted(args.gens.glob(f"*_{args.label}_n8000_*.json"))
        assert len(files) == G, files
        gens = [json.loads(p.read_text(encoding="utf-8")) for p in files]
        fields, pooled = score(pools(args.bundle, args.label), gens, 8000)
        (args.out / "s0_report.json").write_text(json.dumps(
            {"label": args.label, "fields": fields, "pooled_fields": pooled},
            indent=1, default=float), encoding="utf-8")
        for name, r in fields.items():
            print(f"  {name:<5} {r.get('acceptance')}")
        return

    report = {"registration": "research/cr1/PREREG_CR1.md", "G": G, "D": D,
              "n_main": N_MAIN, "seed": SEED, "states": {}}
    for s, (label, state) in enumerate(STATES):
        tag = tag_of(label, state)
        if (label, tuple(sorted(state.items()))) in NOT_MEASURABLE:
            report["states"][tag] = {"status": "NOT MEASURABLE (registered)"}
            continue
        t0 = time.time()
        gens = [json.loads((args.gens / f"{tag}_g{g}.json").read_text(encoding="utf-8"))
                for g in range(G)]
        assert all(len(x) == N_MAIN for x in gens), tag
        assert all(in_state(u, state) for x in gens for u in x), f"{tag}: non-compliant URL"
        pop = pools(args.bundle, label, state)
        fields, pooled = score(pop, gens, N_MAIN)
        report["states"][tag] = {
            "label": label, "state": state, "real_name_pool": len(pop["name"]),
            "fields": fields, "pooled_fields": pooled,
            "seconds": round(time.time() - t0, 1)}
        summary = "  ".join(
            f"{name} {r['acceptance'].get('x', float('nan')):+.4f} "
            f"{r['acceptance']['verdict']}" if r.get("acceptance", {}).get("x") is not None
            else f"{name} {r.get('acceptance', {}).get('verdict', '--')}"
            for name, r in fields.items())
        print(f"  {tag:<15} {summary}  ({time.time() - t0:.0f}s)", flush=True)
        (args.out / "cr1_report.json").write_text(json.dumps(report, indent=1, default=float),
                                                  encoding="utf-8")


if __name__ == "__main__":
    main()
