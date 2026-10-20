"""SC1 -- the registered contrasts, readings and flags (PREREG_SC1 sections 5, 7, 8).

Reads the three arms' scorer reports (CR1's, and C's and T's from the unchanged
`cr1_score.py`) and their generation files. Computes, per (state, field):

    paired     when n is equal in every generation (both arms then see the same
               real samples and floors):  delta_g = mean_d excess_A - mean_d excess_B,
               Delta = mean delta_g, SE = sd(delta_g)/sqrt(5), t with df 4
    unpaired   otherwise:  Delta = x_A - x_B, SE = sqrt(SE_A^2 + SE_B^2), df 4

then Holm within the registered families, the H_SUF and H_DOWN readings, the
K2-K5 flags, the predictions, the descriptive table and the adoption gate.

    python research/sc1/sc1_compare.py --root data/sc1 --cr1 data/cr1 \
        --bundle data/bundle2 --design data/sc1/design_review/design_review.json
"""
import argparse
import json
import math
import statistics
import sys
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE))

from scipy.stats import t as tdist  # noqa: E402

from sc1_common import G, MEASURABLE, tag_of  # noqa: E402

D = 10                                   # CR1's split seeds per generation
FIELDS5 = ("LBL", "SUF", "SUB", "PTH", "QRY")
REQUESTED_N, MIN_N = 1200, 250           # the gate's per-draw n and floor
PRIMARY = ("phishing_P1", "phishing_P1_Q0", "phishing_Q1")
SECONDARY = ("benign_P0", "benign_P0_Q0", "benign_P1_Q1")
HDOWN = tuple((tag, f) for tag in PRIMARY for f in ("LBL", "SUB"))
MARGIN = 0.02                            # K2's margin and H_INDEP's band
NOVELTY_DROP, RUNAWAY_RISE, LONG = 0.005, 0.002, 240
ALPHA = 0.05


# --------------------------------------------------------------------------- #
# statistics
# --------------------------------------------------------------------------- #

def t_crit(df: int) -> float:
    return float(tdist.ppf(0.975, df))


def gen_means(draws: List[float], g: int = G, d: int = D) -> List[float]:
    if len(draws) != g * d:
        raise ValueError(f"{len(draws)} draws, expected {g * d}")
    return [statistics.mean(draws[i * d:(i + 1) * d]) for i in range(g)]


def t_stats(delta: float, se: float, df: int) -> dict:
    if se == 0.0:
        t = math.copysign(math.inf, delta) if delta else 0.0
    else:
        t = delta / se
    tc = t_crit(df)
    p = 1.0 if t == 0.0 else float(2 * tdist.sf(abs(t), df))
    return {"delta": delta, "se": se, "t": t, "df": df,
            "lo": delta - tc * se, "hi": delta + tc * se, "p": p}


def paired(draws_a: List[float], draws_b: List[float]) -> dict:
    d = [a - b for a, b in zip(gen_means(draws_a), gen_means(draws_b))]
    return {**t_stats(statistics.mean(d), statistics.stdev(d) / math.sqrt(len(d)),
                      len(d) - 1), "paired": True}


def unpaired(acc_a: dict, acc_b: dict) -> dict:
    return {**t_stats(acc_a["x"] - acc_b["x"], math.hypot(acc_a["se"], acc_b["se"]),
                      G - 1), "paired": False}


def holm(pvals: Dict[str, float]) -> Dict[str, float]:
    """Holm-Bonferroni step-down adjusted p-values."""
    out, running, m = {}, 0.0, len(pvals)
    for i, (key, p) in enumerate(sorted(pvals.items(), key=lambda kv: kv[1])):
        running = max(running, min(1.0, (m - i) * p))
        out[key] = running
    return out


def prop_diff(k1: int, n1: int, k2: int, n2: int) -> dict:
    """p1 - p2 with a normal-approximation 95% interval."""
    p1, p2 = k1 / n1, k2 / n2
    se = math.sqrt(p1 * (1 - p1) / n1 + p2 * (1 - p2) / n2)
    d = p1 - p2
    return {"p1": p1, "p2": p2, "delta": d, "se": se, "lo": d - 1.96 * se, "hi": d + 1.96 * se}


# --------------------------------------------------------------------------- #
# pairing: n exactly as discriminate_with_floor computes it
# --------------------------------------------------------------------------- #

def cap_of(real_nonempty: int) -> int:
    return min(REQUESTED_N, real_nonempty // 3)


def n_per_gen(cap: int, fake_counts: List[int]) -> List[int]:
    return [min(cap, c) for c in fake_counts]


def pairable(cap: int, fakes_a: List[int], fakes_b: List[int]) -> bool:
    return n_per_gen(cap, fakes_a) == n_per_gen(cap, fakes_b)


def computable(field_row: Optional[dict]) -> bool:
    return bool(field_row) and field_row.get("acceptance", {}).get("x") is not None


def contrast(row_a: dict, row_b: dict, can_pair: bool) -> Optional[dict]:
    """The registered contrast between two arms' records of one (state, field)."""
    if not (computable(row_a) and computable(row_b)):
        return None
    if can_pair and len(row_a["excess_draws"]) == len(row_b["excess_draws"]) == G * D:
        return paired(row_a["excess_draws"], row_b["excess_draws"])
    return unpaired(row_a["acceptance"], row_b["acceptance"])


# --------------------------------------------------------------------------- #
# readings
# --------------------------------------------------------------------------- #

def read_h_suf(rows: Dict[str, dict], v4_ok: bool) -> dict:
    """rows[tag] = contrast + p_holm + t_verdict. Section 7.1."""
    per = {}
    for tag, r in rows.items():
        per[tag] = bool(r and r["delta"] < 0 and r["p_holm"] < ALPHA
                        and r["t_verdict"] == "CLEARED")
    n = sum(per.values())
    if n == len(rows):
        overall = "SUPPORTED"
    elif n:
        overall = "PARTIAL"
    else:
        overall = "REFUTED" if v4_ok else "NOT READ (V4 failed)"
    return {"per_state": per, "overall": overall}


def read_h_down(cells: Dict[str, dict]) -> dict:
    """cells[key] = contrast + p_holm. Section 7.2: a cell may support either
    reading, both, or neither."""
    per = {}
    for key, r in cells.items():
        down = bool(r and r["delta"] < 0 and r["p_holm"] < ALPHA)
        indep = bool(r and r["lo"] > -MARGIN and r["hi"] < MARGIN)
        per[key] = {"H_DOWN": down, "H_INDEP": indep}
    n_down = sum(v["H_DOWN"] for v in per.values())
    n_indep = sum(v["H_INDEP"] for v in per.values())
    if n_down >= 4 and n_indep == 0:
        overall = "H_DOWN"
    elif n_indep >= 4:
        overall = "H_INDEP"
    else:
        overall = "MIXED"
    return {"per_cell": per, "n_down": n_down, "n_indep": n_indep, "overall": overall}


def k2_flags(t_rep: dict, cr1_rep: dict, contrasts: Dict[str, dict]) -> List[dict]:
    """Section 7.3, K2 -- T against CR1."""
    flags = []
    for key, c in contrasts.items():
        if c and c["delta"] >= MARGIN and c["lo"] > 0:
            flags.append({"cell": key, "rule": "margin", "delta": c["delta"], "lo": c["lo"]})
    for tag, t_state in t_rep.items():
        if "fields" not in t_state:        # a registered not-measurable state
            continue
        c_state = cr1_rep[tag]
        for f in FIELDS5:
            a_t = t_state["fields"].get(f, {}).get("acceptance", {})
            a_c = c_state["fields"].get(f, {}).get("acceptance", {})
            if a_c.get("verdict") == "CLEARED" and a_t.get("verdict") == "CONVICTED":
                flags.append({"cell": f"{tag}/{f}", "rule": "CLEARED->CONVICTED"})
        for f, p_c in (c_state.get("pooled_fields") or {}).items():
            p_t = (t_state.get("pooled_fields") or {}).get(f)
            if p_t and p_c.get("verdict") == "UNDECIDED" and p_t.get("verdict") == "TELL":
                flags.append({"cell": f"{tag}/{f}", "rule": "pooled UNDECIDED->TELL"})
    return flags


def novelty_harm(d: dict) -> bool:
    """K3: novelty(T) - novelty(CR1) <= -0.5 points, CI excluding 0."""
    return d["delta"] <= -NOVELTY_DROP and d["hi"] < 0


def runaway_harm(d: dict) -> bool:
    """K4: long-URL share(T) - share(CR1) >= +0.2 points, CI excluding 0."""
    return d["delta"] >= RUNAWAY_RISE and d["lo"] > 0


def adoption(h_suf: str, t_suf_convicted: List[str], surviving_k2: Optional[List[dict]],
             k3: List[str], k4: List[str], k5: List[str], valid: bool) -> dict:
    """Section 7.5. `surviving_k2` is None while a flag still awaits replication."""
    reasons = []
    if h_suf != "SUPPORTED":
        reasons.append(f"H_SUF {h_suf}")
    if t_suf_convicted:
        reasons.append(f"T SUF CONVICTED in {t_suf_convicted}")
    if surviving_k2:
        reasons.append(f"K2 surviving replication: {[f['cell'] for f in surviving_k2]}")
    for name, hits in (("K3", k3), ("K4", k4), ("K5", k5)):
        if hits:
            reasons.append(f"{name}: {hits}")
    if not valid:
        reasons.append("validity V1-V6 not all passed")
    if surviving_k2 is None and not reasons:
        return {"gate": "PENDING -- K2 flags await replication (7.4)", "reasons": []}
    return {"gate": "MET" if not reasons else "NOT MET", "reasons": reasons}


# --------------------------------------------------------------------------- #
# the run
# --------------------------------------------------------------------------- #

def load(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, required=True, help="data/sc1")
    ap.add_argument("--cr1", type=Path, required=True, help="data/cr1")
    ap.add_argument("--bundle", type=Path, required=True)
    ap.add_argument("--design", type=Path, required=True)
    ap.add_argument("--replication", type=Path, default=None,
                    help="a T replication directory: report.json + gen/ (section 7.4)")
    args = ap.parse_args()

    sys.path.insert(0, str(HERE.parent / "cr1"))
    from cr1_score import pools
    from uie.bloom import BloomFilter
    from uie.evaluation import FIELDS
    from uie.factorization import parse
    from uie.gate import Gate
    from sc1_verify import agreement
    from sc1_common import table_counts
    from uie.suffixes import SuffixTable
    from build_tables import read_rows

    reps = {"CR1": load(args.cr1 / "cr1_report.json")["states"],
            "C": load(args.root / "C" / "cr1_report.json")["states"],
            "T": load(args.root / "T" / "cr1_report.json")["states"]}
    gen_dirs = {"CR1": args.cr1 / "gen", "C": args.root / "C" / "gen", "T": args.root / "T" / "gen"}
    verify = load(args.root / "VERIFY.json")
    s0p = load(args.root / "s0p.json")
    valid = verify["verdict"] == "PASS" and s0p["verdict"] == "PASS"
    tables = load(args.root / "tables.json")
    design = load(args.design)["C2_suf_ceiling"]
    bloom = next(p for p in (args.bundle / "known_urls.bloom.gz", args.bundle / "known_urls.bloom")
                 if p.exists())
    gate = Gate(known_membership=BloomFilter.load(bloom))
    provs = {arm: load(gen_dirs[arm] / "PROVENANCE.json") for arm in gen_dirs}
    unmet = [f"{arm}/{r['file']}" for arm in ("C", "T") for r in provs[arm]["files"]
             if r.get("error")]
    if unmet:
        sys.exit(f"K5 fired -- PresenceUnmet in {unmet}. The unchanged scorer cannot score an "
                 "arm with a missing file; section 7.3 applies (that state is not adoptable).")

    print("loading generations and real pools", flush=True)
    urls: Dict[str, Dict[str, List[List[str]]]] = {arm: {} for arm in reps}
    fakes: Dict[str, Dict[str, Dict[str, List[int]]]] = {arm: {} for arm in reps}
    caps: Dict[str, Dict[str, int]] = {}
    for s, label, state in MEASURABLE:
        tag = tag_of(label, state)
        real = pools(args.bundle, label, state)["name"]
        hosts = {FIELDS["SUB"](u) + "|" + FIELDS["LBL"](u) + "|" + FIELDS["SUF"](u) for u in real}
        assert len(hosts) == len(real), f"{tag}: duplicate hosts -- n would depend on the seed"
        caps[tag] = {f: cap_of(sum(1 for u in real if FIELDS[f](u))) for f in FIELDS5}
        for arm, d in gen_dirs.items():
            gens = [load(d / f"{tag}_g{g}.json") for g in range(G)]
            urls[arm][tag] = gens
            fakes[arm][tag] = {f: [sum(1 for u in x if FIELDS[f](u)) for x in gens] for f in FIELDS5}

    def contrasts(a: str, b: str) -> Dict[str, dict]:
        out = {}
        for s, label, state in MEASURABLE:
            tag = tag_of(label, state)
            for f in FIELDS5:
                if state.get(f) is False:
                    continue
                can = pairable(caps[tag][f], fakes[a][tag][f], fakes[b][tag][f])
                out[f"{tag}/{f}"] = contrast(reps[a][tag]["fields"].get(f),
                                             reps[b][tag]["fields"].get(f), can)
        return out

    TC, TCR1, CCR1 = contrasts("T", "C"), contrasts("T", "CR1"), contrasts("C", "CR1")

    def family(keys):
        rows = {k: TC.get(k) for k in keys}
        adj = holm({k: r["p"] for k, r in rows.items() if r})
        return {k: ({**r, "p_holm": adj[k]} if r else None) for k, r in rows.items()}

    prim = family([f"{t}/SUF" for t in PRIMARY])
    sec = family([f"{t}/SUF" for t in SECONDARY])
    hd = family([f"{t}/{f}" for t, f in HDOWN])
    for fam in (prim, sec):
        for k, r in fam.items():
            if r:
                r["t_verdict"] = reps["T"][k.split("/")[0]]["fields"]["SUF"]["acceptance"]["verdict"]
    v4_ok = not any("agreement" in f for f in verify["failures"])
    h_suf = read_h_suf({k.split("/")[0]: r for k, r in prim.items()}, v4_ok)
    h_suf_secondary = read_h_suf({k.split("/")[0]: r for k, r in sec.items()}, v4_ok)
    h_down = read_h_down(hd)

    # K2 with its replication, K3-K5
    flags = k2_flags(reps["T"], reps["CR1"], TCR1)
    surviving: Optional[List[dict]] = [] if not flags else None
    replication = {}
    if flags and args.replication:
        rep_r = load(args.replication / "report.json")["states"]
        surviving = []
        for fl in flags:
            tag, f = fl["cell"].split("/")
            if tag not in rep_r:
                replication[fl["cell"]] = "not replicated yet"
                surviving = None
                continue
            gens = [load(args.replication / "gen" / f"{tag}_g{g}.json") for g in range(G)]
            fk = [sum(1 for u in x if FIELDS[f](u)) for x in gens]
            c = contrast(rep_r[tag]["fields"].get(f), reps["CR1"][tag]["fields"].get(f),
                         pairable(caps[tag][f], fk, fakes["CR1"][tag][f]))
            again = k2_flags({tag: rep_r[tag]}, {tag: reps["CR1"][tag]}, {fl["cell"]: c})
            hit = any(x["cell"] == fl["cell"] and x["rule"] == fl["rule"] for x in again)
            replication[fl["cell"]] = "REPLICATED" if hit else "not replicated"
            if hit and surviving is not None:
                surviving.append(fl)

    k3, k4, k5, desc = [], [], [], {}
    print("novelty, length and copy rate", flush=True)
    seen_domains = set()
    for stem in ("name_train", "structure_train", "phishing_name_train", "phishing_structure_train"):
        for row in read_rows(args.bundle / f"{stem}.jsonl.gz"):
            parts = parse(row["url"])
            seen_domains.add(parts.registrable_label + "." + parts.public_suffix)
    for s, label, state in MEASURABLE:
        tag = tag_of(label, state)
        desc[tag] = {}
        for arm in ("CR1", "C", "T"):
            flat = [u for x in urls[arm][tag] for u in x]
            _, rep = gate.filter(flat)
            mem = round((1 - rep.novelty_rate) * rep.n_total)
            prov_rows = [r for r in provs[arm]["files"] if r["file"].startswith(tag + "_g")]
            first = sum(r["stats"]["first_pass"] for r in prov_rows)
            draws = sum(r["stats"]["draws"] for r in prov_rows)
            row = {"novelty": rep.novelty_rate, "memorized": mem, "n": rep.n_total,
                   "long": sum(len(u) >= LONG for u in flat), "mean_len": statistics.mean(map(len, flat)),
                   "distinct": len(set(flat)) / len(flat),
                   "copy_rate": sum((FIELDS["LBL"](u) + "." + FIELDS["SUF"](u)) in seen_domains
                                    for u in flat) / len(flat),
                   "first_pass": first / (G * 3000), "draws_per_url": draws / (G * 3000),
                   "rounds": max(r["stats"]["rounds"] for r in prov_rows)}
            if arm in ("C", "T"):
                agr = [agreement(urls[arm][tag][g],
                                 SuffixTable({label: table_counts(tables, arm, label, state)}),
                                 label, next(r["seed"] for r in prov_rows if r["generation"] == g))
                       for g in range(G)]
                row["suffix_agreement_min"] = min(agr)
            desc[tag][arm] = row
        nov = prop_diff(desc[tag]["T"]["n"] - desc[tag]["T"]["memorized"], desc[tag]["T"]["n"],
                        desc[tag]["CR1"]["n"] - desc[tag]["CR1"]["memorized"], desc[tag]["CR1"]["n"])
        lng = prop_diff(desc[tag]["T"]["long"], desc[tag]["T"]["n"],
                        desc[tag]["CR1"]["long"], desc[tag]["CR1"]["n"])
        desc[tag]["novelty_T_minus_CR1"], desc[tag]["long_T_minus_CR1"] = nov, lng
        if novelty_harm(nov):
            k3.append(tag)
        if runaway_harm(lng):
            k4.append(tag)

    # predictions (section 4) -- reported, never criteria
    preds = {}
    for s, label, state in MEASURABLE:
        tag = tag_of(label, state)
        ceil = design[tag]
        x = {arm: reps[arm][tag]["fields"]["SUF"]["acceptance"].get("x") for arm in reps}
        preds[tag] = {
            "P1_T": {"x": x["T"], "ceiling": ceil["T"].get("x"),
                     "within": abs(x["T"] - ceil["T"]["x"]) <= MARGIN},
            "P1_C": {"x": x["C"], "ceiling": ceil["C"].get("x"),
                     "within": abs(x["C"] - ceil["C"]["x"]) <= MARGIN},
            "P2_delta_T_minus_C": {"observed": (TC.get(f"{tag}/SUF") or {}).get("delta"),
                                   "predicted": ceil["T"]["x"] - ceil["C"]["x"]},
            "P3_C_minus_CR1": {"observed": (CCR1.get(f"{tag}/SUF") or {}).get("delta"),
                               "predicted": ceil["C"]["x"] - x["CR1"]}}

    t_suf_convicted = [tag_of(l, st) for _, l, st in MEASURABLE
                       if reps["T"][tag_of(l, st)]["fields"]["SUF"]["acceptance"]["verdict"] == "CONVICTED"]
    gate_result = adoption(h_suf["overall"], t_suf_convicted, surviving, k3, k4, k5, valid)

    out = {"registration": "research/sc1/PREREG_SC1.md",
           "validity": {"verify": verify["verdict"], "s0p": s0p["verdict"]},
           "primary": prim, "h_suf": h_suf, "secondary": sec, "h_suf_secondary": h_suf_secondary,
           "h_down_cells": hd, "h_down": h_down,
           "k2_flags": flags, "k2_replication": replication, "k3": k3, "k4": k4, "k5": k5,
           "adoption": gate_result, "predictions": preds, "descriptive": desc,
           "contrasts": {"T-C": TC, "T-CR1": TCR1, "C-CR1": CCR1}}
    (args.root / "sc1_report.json").write_text(json.dumps(out, indent=1, default=float),
                                               encoding="utf-8")

    def show(name, fam):
        print(f"\n{name}")
        for k, r in fam.items():
            if r:
                print(f"  {k:<22} delta {r['delta']:+.4f} [{r['lo']:+.4f}, {r['hi']:+.4f}] "
                      f"p {r['p']:.4g} holm {r['p_holm']:.4g} {'paired' if r['paired'] else 'UNPAIRED'}"
                      + (f"  T verdict {r['t_verdict']}" if "t_verdict" in r else ""))
            else:
                print(f"  {k:<22} not computable in an arm")
    show("PRIMARY -- SUF, T - C", prim)
    print("  H_SUF:", h_suf["overall"])
    show("SECONDARY -- SUF, T - C", sec)
    print("  secondary:", h_suf_secondary["overall"])
    show("H_DOWN -- LBL and SUB, T - C", hd)
    print(f"  reading: {h_down['overall']} ({h_down['n_down']} cells H_DOWN, "
          f"{h_down['n_indep']} H_INDEP)")
    print(f"\nK2 flags: {flags or 'none'}  replication: {replication or '--'}")
    print(f"K3: {k3 or 'none'}  K4: {k4 or 'none'}  K5: {k5 or 'none'}")
    print(f"validity: VERIFY {verify['verdict']}, S0' {s0p['verdict']}")
    print(f"ADOPTION GATE: {gate_result['gate']}  {gate_result['reasons'] or ''}")


if __name__ == "__main__":
    main()
