"""Q3.1 -- resolve WHICH shape property carries the query-content tell.

Pre-registered in research/qry/PREREG_Q3_1.md before running. Continues the Q3
ladder at higher resolution; it is not a new methodology. Same `scored`, same
paired-by-seed design, same floor recomputation, same kill criteria.

The only change is D: 3 split seeds -> 12. The decision rule is unchanged --
two-sided t at p<0.05 -- so the threshold moves from 4.303 to 2.201 purely
because df moves from 2 to 11.

Stage 1 runs the ladder on the production arm. Stage 2 re-runs whatever cleared
the threshold on the second arm; a finding is promoted only if it REPLICATES.
"""
from __future__ import annotations

import argparse
import gzip
import json
import statistics as st
import sys
import time
from pathlib import Path
from typing import Dict, List, Sequence

from scipy import stats as sps

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from research.qry.q3_content import (DEGENERATE, FLOOR_HIGH,  # noqa: E402
                                     FLOOR_LOW, scored)
from research.qry.qtransforms import COMPONENT, TRANSFORMS   # noqa: E402
from uie.evaluation import FIELDS, _one_per_host             # noqa: E402

D = 12
SEEDS = tuple(20260806 + i for i in range(D))
TCRIT = float(sps.t.ppf(0.975, D - 1))

#: `count_only` is excluded: Q3 found it VOID degenerate in both arms
#: (distinct 0.013 / 0.012). Power does not repair a degenerate discriminator.
LADDER = ["identity", "null",
          "clip_values_20", "clip_values_15", "clip_values_10",
          "first_param", "fixlen_names", "mask_pct", "normalize_seps",
          "mask_punct", "alnum_flat",
          "mask_name_chars", "mask_values", "fixlen_values", "sort_params",
          "skeleton", "structure_only"]


def load_real(cls: str) -> List[str]:
    pre = "" if cls == "benign" else f"{cls}_"
    with gzip.open(Path("data/bundle2") / f"{pre}name_val.jsonl.gz", "rt",
                   encoding="utf-8") as fh:
        urls = [json.loads(l)["url"] for l in fh]
    return [x for x in (FIELDS["QRY"](u) for u in _one_per_host(urls, SEEDS[0]))
            if x]


def run_ladder(real_q: Sequence[str], fake_q: Sequence[str],
               names: Sequence[str], tag: str) -> Dict[str, dict]:
    print(f"\n=== {tag} ===  real {len(real_q):,}  fake {len(fake_q)}  "
          f"D={D}  df={D-1}  t_crit={TCRIT:.3f}", flush=True)
    print(f"  {'transform':<17}{'component':<38}{'excess':>9}{'floor':>8}"
          f"{'R_T':>6}{'diff':>9}{'SE':>8}{'t':>8}{'dist':>7}  verdict")
    base: List[float] = []
    rows: Dict[str, dict] = {}
    for name in names:
        t0 = time.time()
        draws = [scored(real_q, fake_q, TRANSFORMS[name], s) for s in SEEDS]
        ok = [d for d in draws if not d.get("insufficient")]
        if not ok:
            print(f"  {name:<17}{COMPONENT[name][0]:<38}   INSUFFICIENT")
            continue
        ex = [d["excess"] for d in ok]
        floor = st.mean([d["floor"] for d in ok])
        dist = st.mean([d["distinct_frac"] for d in ok])
        if name == "identity":
            base = ex
        diff = [a - b for a, b in zip(ex, base)]
        m = st.mean(diff)
        sd = st.stdev(diff) if len(diff) > 1 and any(diff) else 0.0
        se = sd / len(diff) ** 0.5 if sd else 0.0
        t = m / se if se else float("nan")

        if not (FLOOR_LOW <= floor <= FLOOR_HIGH):
            verdict = "VOID floor"
        elif dist < DEGENERATE:
            verdict = "VOID degenerate"
        elif name == "identity":
            verdict = "baseline"
        elif not se:
            verdict = "no change"
        elif t < -TCRIT:
            verdict = "REDUCES"
        elif t > TCRIT:
            verdict = "raises"
        else:
            verdict = "no change"

        rows[name] = {"excess": round(st.mean(ex), 4), "floor": round(floor, 4),
                      "R_T": round(st.mean(ex) / st.mean(base), 3) if base else None,
                      "diff": round(m, 4), "se": round(se, 4),
                      "t": round(t, 2) if se else None, "df": D - 1,
                      "t_crit": round(TCRIT, 3), "distinct_frac": round(dist, 3),
                      "verdict": verdict, "n": ok[0]["n"], "draws": len(ok),
                      "component": COMPONENT[name][0],
                      "excess_draws": [round(x, 4) for x in ex],
                      "seconds": round(time.time() - t0, 1)}
        r = rows[name]
        print(f"  {name:<17}{r['component']:<38}{r['excess']:>+9.4f}"
              f"{r['floor']:>8.4f}{(r['R_T'] or 0):>6.2f}{r['diff']:>+9.4f}"
              f"{r['se']:>8.4f}"
              f"{(f'{t:+.2f}' if se else '--'):>8}{r['distinct_frac']:>7.3f}"
              f"  {verdict}", flush=True)
    return rows


def main() -> None:
    # Was hard-coded to "benign" while the scope was benign-only. Parameterised
    # 2026-09-23 for the phishing phase (PREREG_PHISHING_QRY.md, PQ1). The
    # default keeps every published benign run reproducible with no arguments.
    ap = argparse.ArgumentParser()
    ap.add_argument("--class", dest="cls", default="benign",
                    choices=["benign", "phishing"])
    cls = ap.parse_args().cls
    real_q = load_real(cls)
    arms = {}
    for arm in ("typical", "temperature"):
        urls = json.loads(Path(f"data/qry/q4_{cls}_{arm}_queries.json").read_text())
        arms[arm] = [x for x in (FIELDS["QRY"](u) for u in urls) if x]

    out: Dict[str, object] = {"class": cls, "D": D, "df": D - 1,
                              "t_crit": round(TCRIT, 4),
                              "seeds": list(SEEDS), "stages": {}}

    # ---- stage 1: full ladder, production arm ----
    s1 = run_ladder(real_q, arms["typical"], LADDER, "STAGE 1  typical")
    out["stages"]["stage1_typical"] = s1

    # ---- pre-specified interaction, only if both parts are individually null ----
    cond = [n for n in ("clip_values_15", "first_param")
            if s1.get(n, {}).get("verdict") == "no change"]
    if len(cond) == 2:
        print("\n  both tail and count individually null -> running the "
              "pre-specified interaction", flush=True)
        s1i = run_ladder(real_q, arms["typical"], ["identity", "clip15_first"],
                         "STAGE 1b  typical, interaction")
        s1.update({k: v for k, v in s1i.items() if k != "identity"})
        out["stages"]["stage1_interaction"] = s1i
    else:
        print(f"\n  interaction NOT run: {cond} was/were not null", flush=True)

    # ---- stage 2: replication of whatever cleared threshold ----
    hits = [n for n, r in s1.items()
            if r["verdict"] in ("REDUCES", "raises")]
    print(f"\n  stage 1 hits: {hits or 'none'}", flush=True)
    if hits:
        s2 = run_ladder(real_q, arms["temperature"],
                        ["identity", "null"] + hits, "STAGE 2  temperature")
        out["stages"]["stage2_temperature"] = s2

    p = Path(f"data/qry/q3_1_{cls}.json")
    p.write_text(json.dumps(out, indent=1, default=str) + "\n", encoding="utf-8")
    print(f"\nwrote {p}")


if __name__ == "__main__":
    main()
