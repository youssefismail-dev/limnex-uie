"""DC1 -- the registered readings (PREREG_DC1.md section 5), per mode.

Delta = x(DC1) - x(baseline), SC1's contrast (paired by generation where n is
equal in every generation, else unpaired). Baselines: FULL FREE, today's E7 from
the same notebook and seeds; FULL CONTROL, SC1 arm T's files (read only).

    FIXED       a target cell's DC1 verdict is CLEARED
    IMPROVED    a target's Delta < 0, Holm p < 0.05 over that mode's targets
    REGRESSION  any cell's Delta > 0, Holm p < 0.05 over all the mode's cells;
                or a CLEARED cell becomes CONVICTED
    novelty     whole-URL novelty (the gate's normalisation and training Bloom
                filter) and the share of query strings copied from training:
                each worsens by less than 0.5 points (upper 95% bound)
    cost        next-token predictions per delivered URL <= 2x the baseline
    ADOPT       >= 1 IMPROVED, 0 REGRESSIONS, novelty held, cost held

    python research/dc1/dc1_compare.py --dc1 data/dc1 --sc1 data/sc1 --bundle data/bundle2
"""
import argparse
import gzip
import json
import math
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "sc1"))
sys.path.insert(0, str(HERE.parent / "cr1"))

from dc1_common import G, MEASURABLE, STATES  # noqa: E402
from sc1_compare import cap_of, contrast, holm, pairable  # noqa: E402

CELL_FIELDS = ("LBL", "SUF", "SUB", "PTH", "QRY")
TARGETS = {"free": [("benign", None, "PTH"), ("benign", None, "QRY"),
                    ("phishing", None, "PTH"), ("phishing", None, "QRY")],
           "control": [("benign", "P1Q1", "SUB"), ("benign", "P1Q1", "PTH"),
                       ("benign", "P1Q1", "QRY"), ("benign", "P0Q1", "QRY"),
                       ("phishing", "P1Q0", "LBL"), ("phishing", "P1Q0", "SUB"),
                       ("phishing", "P1Q0", "PTH")]}
ALPHA, NOVELTY_MARGIN, COST_MAX = 0.05, 0.005, 2.0


def sc1_tag(label, state):
    p, q = STATES[state]["PTH"], STATES[state]["QRY"]
    return f"{label}_P{int(p)}_Q{int(q)}"


def verdict(row):
    return (row or {}).get("acceptance", {}).get("verdict")


def read_mode(cells: dict, targets: list) -> dict:
    """cells[key] = {"delta", "p", ..., "v_base", "v_dc1"} or None."""
    tkeys = [k for k in targets if cells.get(k)]
    t_adj = holm({k: cells[k]["p"] for k in tkeys})
    a_adj = holm({k: c["p"] for k, c in cells.items() if c})
    out = {"targets": {}, "regressions": [], "fixed": [], "improved": []}
    for k in targets:
        c = cells.get(k)
        row = {"verdict_base": None, "verdict_dc1": None, "reading": "NOT COMPUTABLE"}
        if c:
            row = {**c, "p_holm": t_adj[k],
                   "improved": c["delta"] < 0 and t_adj[k] < ALPHA,
                   "fixed": c["v_dc1"] == "CLEARED"}
            if row["improved"]:
                out["improved"].append(k)
            if row["fixed"]:
                out["fixed"].append(k)
        out["targets"][str(k)] = row
    out["cells"] = {}
    for k, c in cells.items():
        if not c:
            out["cells"][str(k)] = None
            continue
        out["cells"][str(k)] = {**c, "p_holm_all": a_adj[k]}
        if (c["delta"] > 0 and a_adj[k] < ALPHA) or (c["v_base"] == "CLEARED" and c["v_dc1"] == "CONVICTED"):
            out["regressions"].append({"cell": str(k), "delta": c["delta"], "p_holm": a_adj[k],
                                       "v_base": c["v_base"], "v_dc1": c["v_dc1"]})
    return out


def prop_worsening(k_base, n_base, k_dc1, n_dc1, worse_if_higher):
    """Worsening in proportion points, with a normal 95% CI. As SC1's K3: it
    counts only if it is at least 0.5 points AND its CI excludes 0 -- an
    upper-bound reading would fail on noise alone (0.86 points at zero change
    for ~9,400 query URLs per arm)."""
    pb, pd = k_base / n_base, k_dc1 / n_dc1
    d = (pd - pb) if worse_if_higher else (pb - pd)
    se = math.sqrt(pb * (1 - pb) / n_base + pd * (1 - pd) / n_dc1)
    worsened = d >= NOVELTY_MARGIN and d - 1.96 * se > 0
    return {"base": pb, "dc1": pd, "worsening": d, "lo": d - 1.96 * se, "hi": d + 1.96 * se,
            "worsened": worsened, "ok": not worsened}


def training_queries(bundle: Path, label: str) -> set:
    from uie.evaluation import FIELDS
    pre = "" if label == "benign" else f"{label}_"
    out = set()
    for kind in ("name", "structure"):
        with gzip.open(bundle / f"{pre}{kind}_train.jsonl.gz", "rt", encoding="utf-8") as fh:
            for line in fh:
                q = FIELDS["QRY"](json.loads(line)["url"])
                if q:
                    out.add(q)
    return out


def novelty(url_sets: dict, bundle: Path, bloom) -> dict:
    """url_sets[arm][label] = list of URLs; arms 'base' and 'dc1'."""
    from uie.evaluation import FIELDS
    from uie.gate import normalize_for_dedup
    counts = {arm: {"n": 0, "novel": 0, "q": 0, "copied": 0} for arm in url_sets}
    for label in ("benign", "phishing"):
        known_q = training_queries(bundle, label)
        for arm, by_label in url_sets.items():
            for u in by_label.get(label, []):
                c = counts[arm]
                c["n"] += 1
                c["novel"] += normalize_for_dedup(u) not in bloom
                q = FIELDS["QRY"](u)
                if q:
                    c["q"] += 1
                    c["copied"] += q in known_q
        del known_q
    b, d = counts["base"], counts["dc1"]
    return {"counts": counts,
            "whole_url_novelty": prop_worsening(b["novel"], b["n"], d["novel"], d["n"], False),
            "query_copying": prop_worsening(b["copied"], max(1, b["q"]), d["copied"], max(1, d["q"]), True)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dc1", type=Path, required=True)
    ap.add_argument("--sc1", type=Path, required=True)
    ap.add_argument("--bundle", type=Path, required=True)
    args = ap.parse_args()

    from cr1_score import pools
    from uie.bloom import BloomFilter
    from uie.evaluation import FIELDS

    gens = args.dc1 / "gen"
    rep = json.loads((args.dc1 / "score_report.json").read_text(encoding="utf-8"))
    prov = {r["file"]: r for r in json.loads((gens / "PROVENANCE.json")
                                             .read_text(encoding="utf-8"))["files"]}
    q_rep = json.loads((args.sc1 / "T" / "cr1_report.json").read_text(encoding="utf-8"))["states"]
    bloom = BloomFilter.load(args.bundle / "known_urls.bloom.gz")

    def load(path):
        return json.loads(path.read_text(encoding="utf-8"))

    out = {"registration": "research/dc1/PREREG_DC1.md", "modes": {}}

    # ---- FULL FREE: DC1 against today's E7, same notebook, same seeds ----
    if "dc1" in rep["free"] and "base" in rep["free"]:
        cells, urls = {}, {"base": {}, "dc1": {}}
        for label in ("benign", "phishing"):
            real = pools(args.bundle, label)["name"]
            arms = {arm: [load(gens / f"free_{arm}_{label}_g{g}.json") for g in range(G)]
                    for arm in ("base", "dc1")}
            for arm in arms:
                urls[arm][label] = [u for x in arms[arm] for u in x]
            for f in CELL_FIELDS:
                cap = cap_of(sum(1 for u in real if FIELDS[f](u)))
                fk = {arm: [sum(1 for u in x if FIELDS[f](u)) for x in xs] for arm, xs in arms.items()}
                rb = rep["free"]["base"][label]["fields"].get(f)
                rd = rep["free"]["dc1"][label]["fields"].get(f)
                c = contrast(rd, rb, pairable(cap, fk["dc1"], fk["base"]))
                cells[(label, None, f)] = c and {**c, "v_base": verdict(rb), "v_dc1": verdict(rd)}
        mode = read_mode(cells, TARGETS["free"])
        rows = {arm: sum(prov[f"free_{arm}_{lab}_g{g}.json"]["forward_rows"]
                         for lab in ("benign", "phishing") for g in range(G)) for arm in ("base", "dc1")}
        n = {arm: sum(len(v) for v in urls[arm].values()) for arm in urls}
        mode["cost"] = {"base": rows["base"] / n["base"], "dc1": rows["dc1"] / n["dc1"],
                        "ratio": (rows["dc1"] / n["dc1"]) / (rows["base"] / n["base"])}
        mode["novelty"] = novelty(urls, args.bundle, bloom)
        out["modes"]["free"] = mode

    # ---- FULL CONTROL: DC1 against SC1 arm T (read only) ----
    if rep["control"]:
        cells, urls, cost = {}, {"base": {}, "dc1": {}}, {}
        for label, states in MEASURABLE.items():
            for st in states:
                key = f"{label}_{st}"
                if key not in rep["control"]:
                    continue
                tag = sc1_tag(label, st)
                real = pools(args.bundle, label, STATES[st])["name"]
                arms = {"dc1": [load(gens / f"ctrl_dc1_{key}_g{g}.json") for g in range(G)],
                        "base": [load(args.sc1 / "T" / "gen" / f"{tag}_g{g}.json") for g in range(G)]}
                for arm in arms:
                    urls[arm].setdefault(label, []).extend(u for x in arms[arm] for u in x)
                for f in CELL_FIELDS:
                    rb = q_rep[tag]["fields"].get(f)
                    rd = rep["control"][key]["fields"].get(f)
                    if not (rb and rd):
                        continue
                    cap = cap_of(sum(1 for u in real if FIELDS[f](u)))
                    fk = {arm: [sum(1 for u in x if FIELDS[f](u)) for x in xs] for arm, xs in arms.items()}
                    c = contrast(rd, rb, pairable(cap, fk["dc1"], fk["base"]))
                    cells[(label, st, f)] = c and {**c, "v_base": verdict(rb), "v_dc1": verdict(rd)}
                pb, pd = prov[f"ctrl_base_{key}_g0.json"], prov[f"ctrl_dc1_{key}_g0.json"]
                cost[key] = {"base": pb["forward_rows"] / pb["n"], "dc1": pd["forward_rows"] / pd["n"]}
        mode = read_mode(cells, TARGETS["control"])
        tb = sum(c["base"] for c in cost.values())
        td = sum(c["dc1"] for c in cost.values())
        mode["cost"] = {"per_state": cost, "ratio": td / tb}
        mode["novelty"] = novelty(urls, args.bundle, bloom)
        out["modes"]["control"] = mode

    for name, m in out["modes"].items():
        nov_ok = m["novelty"]["whole_url_novelty"]["ok"] and m["novelty"]["query_copying"]["ok"]
        cost_ok = m["cost"]["ratio"] <= COST_MAX
        m["adopt"] = bool(m["improved"]) and not m["regressions"] and nov_ok and cost_ok
        m["criteria"] = {"improved": len(m["improved"]), "regressions": len(m["regressions"]),
                         "novelty_held": nov_ok, "cost_held": cost_ok}
    (args.dc1 / "compare_report.json").write_text(json.dumps(out, indent=1, default=str),
                                                  encoding="utf-8")
    for name, m in out["modes"].items():
        print(f"\n== FULL {name.upper()}  ->  {'ADOPT' if m['adopt'] else 'NOT ADOPTED'}  {m['criteria']}")
        for k, r in m["targets"].items():
            if "delta" in r:
                print(f"   {k:<34} {r['v_base']:>10} -> {r['v_dc1']:<10} delta {r['delta']:+.4f} "
                      f"[{r['lo']:+.4f}, {r['hi']:+.4f}] Holm {r['p_holm']:.3g}"
                      + ("  IMPROVED" if r["improved"] else "") + ("  FIXED" if r["fixed"] else ""))
            else:
                print(f"   {k:<34} NOT COMPUTABLE")
        for r in m["regressions"]:
            print(f"   REGRESSION {r}")
        nv = m["novelty"]
        print(f"   novelty {nv['whole_url_novelty']['base']:.4f} -> {nv['whole_url_novelty']['dc1']:.4f}; "
              f"query copying {nv['query_copying']['base']:.4f} -> {nv['query_copying']['dc1']:.4f}; "
              f"cost ratio {m['cost']['ratio']:.2f}")


if __name__ == "__main__":
    main()
