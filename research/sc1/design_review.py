"""SC1 design review -- read-only measurements made BEFORE PREREG_SC1 is registered.

No model is loaded and no URL is generated. Everything measured is a property of
the data (and of CR1/HC2 outputs that already exist), used to fix the
registration's predictions and rules before any SC1 generation:

  A   frame composition of each state's training rows (name vs structure)
  A2  reconstruction: the shipped suffix_table.json from all *_train rows, exactly
  B   coverage of each candidate table -- distinct suffixes, singletons, Good-Turing
      unseen mass -- and the share of held-out hosts whose suffix it lacks
  C1  premise H7: train-state vs held-out-state TLD mix (top-10 + other), against a
      two-sample multinomial null
  C2  the SUF CEILING of each table: training URLs drawn i.i.d. from the table's own
      rows, scored as the generated side by the gate's SUF instrument exactly as
      cr1_score does (discriminate_with_floor, split seeds v2, G=5 x D=10,
      n=min(3000,1200), s261_verdict). Tables: T (name_train, state),
      C (name_train, class), S (all *_train rows, class = the shipped table).
  D   parse stability: HC2's held-mode outputs (suffix drawn from the shipped table
      and held), parsed-suffix mix against the table vs a multinomial null
  E   n-matching: which (state, field) cells were fake-limited in CR1
  F   CR1 first-pass rates (held-mode cost)

    python research/sc1/design_review.py --repo . --out data/sc1/design_review
"""
import argparse
import gc
import gzip
import json
import random
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--repo", type=Path, required=True)
ap.add_argument("--out", type=Path, required=True)
ap.add_argument("--skip-c2", action="store_true")
args = ap.parse_args()
REPO = args.repo
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "research/cr1"))

from uie.corpus.run_eval import s261_verdict, split_seed  # noqa: E402
from uie.evaluation import FIELDS, _one_per_host, discriminate_with_floor  # noqa: E402
from uie.factorization import parse  # noqa: E402
from uie.ngram import path_body  # noqa: E402
from cr1_score import D, G, NOT_MEASURABLE, SEED, STATES, pools, tag_of  # noqa: E402

B = REPO / "data/bundle2"
OUT = args.out
OUT.mkdir(parents=True, exist_ok=True)
REPORT = OUT / "design_review.json"
report = {"purpose": "SC1 design review; read-only; no model, no generation"}
DR_SEED = 20260925                      # design-review sampling seed block


def save():
    REPORT.write_text(json.dumps(report, indent=1, default=float), encoding="utf-8")


def log(msg):
    print(f"{time.strftime('%H:%M:%S')} {msg}", flush=True)


def tv(a, b):
    return 0.5 * sum(abs(a.get(k, 0) - b.get(k, 0)) for k in set(a) | set(b))


def norm(c):
    n = sum(c.values()) or 1
    return {k: v / n for k, v in c.items()}


# ---------------------------------------------------------------- training rows
FILES = {"benign": {"name": "name_train", "structure": "structure_train"},
         "phishing": {"name": "phishing_name_train", "structure": "phishing_structure_train"}}


def load_rows(stem):
    """(url, suffix, P, Q, label) per row, parsing once. Presence exactly as
    FIELDS defines it (checked against FIELDS on a sample below)."""
    out = []
    with gzip.open(B / f"{stem}.jsonl.gz", "rt", encoding="utf-8") as fh:
        for line in fh:
            row = json.loads(line)
            u = row["url"]
            p = parse(u)
            out.append((u, p.public_suffix, bool(path_body(p.path or "")), bool(p.query or ""),
                        row.get("label", "benign")))
    return out


log("loading training rows")
ROWS = {cls: {fr: load_rows(stem) for fr, stem in f.items()} for cls, f in FILES.items()}
chk = 0
for cls in ROWS:
    for fr in ROWS[cls]:
        for (u, s, p, q, lab) in ROWS[cls][fr][:5000]:
            assert p == bool(FIELDS["PTH"](u)) and q == bool(FIELDS["QRY"](u)), u
            assert s == FIELDS["SUF"](u), u
            chk += 1
        assert all(r[4] == cls for r in ROWS[cls][fr]), f"label mismatch in {cls}/{fr}"
log(f"presence/suffix parse-once equals FIELDS on {chk} rows; labels consistent")

IN_STATE = lambda st: (lambda p, q: all((p if f == "PTH" else q) == w for f, w in st.items()))
MEASURABLE = [(s, label, st) for s, (label, st) in enumerate(STATES)
              if (label, tuple(sorted(st.items()))) not in NOT_MEASURABLE]


def table_rows(kind, cls, st):
    """The rows a table is built from. T: name_train in the state; C: name_train;
    S: every *_train row (the shipped construction)."""
    if kind == "T":
        f = IN_STATE(st)
        return [r for r in ROWS[cls]["name"] if f(r[2], r[3]) and r[1]]
    if kind == "C":
        return [r for r in ROWS[cls]["name"] if r[1]]
    return [r for r in ROWS[cls]["name"] + ROWS[cls]["structure"] if r[1]]


# ---------------------------------------------------------------- A2 reconstruction
shipped = json.loads((B / "suffix_table.json").read_text(encoding="utf-8"))
rebuilt = {cls: dict(Counter(r[1] for r in table_rows("S", cls, None))) for cls in ROWS}
same = all(rebuilt[c] == shipped[c] for c in shipped) and set(rebuilt) == set(shipped)
report["A2_reconstruction"] = {"shipped_equals_rebuilt": same,
                               "rows": {c: sum(rebuilt[c].values()) for c in rebuilt},
                               "suffixes": {c: len(rebuilt[c]) for c in rebuilt}}
log(f"A2 shipped table rebuilt from all *_train rows: identical = {same}")
save()

# ---------------------------------------------------------------- A frame composition
TOP = {cls: [k for k, _ in Counter(r[1] for r in ROWS[cls]["name"] if r[1]).most_common(10)]
       for cls in ROWS}
report["top10"] = TOP


def coll(cls, suffixes):
    return norm(Counter(s if s in TOP[cls] else "other" for s in suffixes))


A = {}
for s, label, st in MEASURABLE:
    tag = tag_of(label, st)
    f = IN_STATE(st)
    nm = [r[1] for r in ROWS[label]["name"] if f(r[2], r[3]) and r[1]]
    sr = [r[1] for r in ROWS[label]["structure"] if f(r[2], r[3]) and r[1]]
    A[tag] = {"name_rows": len(nm), "structure_rows": len(sr),
              "structure_share": len(sr) / (len(nm) + len(sr)),
              "tv_name_vs_structure_top10": tv(coll(label, nm), coll(label, sr)),
              "tv_all_vs_name_top10": tv(coll(label, nm + sr), coll(label, nm)),
              "tv_all_vs_name_full": tv(norm(Counter(nm + sr)), norm(Counter(nm)))}
report["A_frame"] = A
log("A frame composition done")
save()

# ---------------------------------------------------------------- B coverage + C1 premise
Bc, C1 = {}, {}
nrng = np.random.default_rng(DR_SEED)
for s, label, st in MEASURABLE:
    tag = tag_of(label, st)
    real = _one_per_host(pools(B, label, st)["name"], SEED)
    real_suf = [FIELDS["SUF"](u) for u in real]
    row = {"heldout_hosts": len(real)}
    for kind in ("T", "C", "S"):
        cnt = Counter(r[1] for r in table_rows(kind, label, st))
        n_rows = sum(cnt.values())
        single = sum(1 for v in cnt.values() if v == 1)
        row[kind] = {"rows": n_rows, "distinct": len(cnt), "singletons": single,
                     "good_turing_unseen": single / n_rows,
                     "heldout_hosts_suffix_absent": sum(1 for x in real_suf if x not in cnt) / len(real_suf)}
    Bc[tag] = row
    # C1: train-state vs held-out-state, top-10 + other, two-sample multinomial null
    tr = [r[1] for r in table_rows("T", label, st)]
    p_tr, p_ho = coll(label, tr), coll(label, real_suf)
    obs = tv(p_tr, p_ho)
    keys = sorted(set(p_tr) | set(p_ho))
    pooled = norm(Counter({k: p_tr.get(k, 0) * len(tr) + p_ho.get(k, 0) * len(real_suf) for k in keys}))
    pv = np.array([pooled.get(k, 0) for k in keys])
    null = []
    for _ in range(1000):
        a = nrng.multinomial(len(tr), pv) / len(tr)
        b = nrng.multinomial(len(real_suf), pv) / len(real_suf)
        null.append(0.5 * np.abs(a - b).sum())
    null = np.array(null)
    C1[tag] = {"train_rows": len(tr), "heldout_hosts": len(real_suf), "tv": obs,
               "null_mean": float(null.mean()), "null_p95": float(np.quantile(null, 0.95)),
               "null_p99": float(np.quantile(null, 0.99)),
               "p_value": float((null >= obs).mean()), "premise_holds": bool(obs <= np.quantile(null, 0.99))}
    log(f"B/C1 {tag:<16} T rows {row['T']['rows']:>7} distinct {row['T']['distinct']:>4} "
        f"GT {row['T']['good_turing_unseen']:.4f} absent T {row['T']['heldout_hosts_suffix_absent']:.4f} "
        f"C {row['C']['heldout_hosts_suffix_absent']:.4f} S {row['S']['heldout_hosts_suffix_absent']:.4f} | "
        f"premise TV {obs:.4f} null99 {C1[tag]['null_p99']:.4f} p {C1[tag]['p_value']:.3f}")
report["B_coverage"] = Bc
report["C1_premise"] = C1
save()

# ---------------------------------------------------------------- D parse stability
Dd = {}
for cell in ("benign_P1", "benign_Q1", "phishing_P1", "phishing_Q1"):
    cls = cell.split("_")[0]
    urls = json.loads((REPO / f"data/hc2/held/{cell}.json").read_text(encoding="utf-8"))
    suf = [FIELDS["SUF"](u) for u in urls]
    tbl = shipped[cls]
    tot = sum(tbl.values())
    p_tab = {}
    for k, v in tbl.items():
        kk = k if k in TOP[cls] else "other"
        p_tab[kk] = p_tab.get(kk, 0) + v / tot
    obs = tv(coll(cls, suf), p_tab)
    keys = sorted(p_tab)
    pv = np.array([p_tab[k] for k in keys])
    null = np.array([0.5 * np.abs(nrng.multinomial(len(suf), pv) / len(suf) - pv).sum() for _ in range(1000)])
    Dd[cell] = {"n": len(suf), "tv_parsed_vs_table_top10": obs, "null_p99": float(np.quantile(null, 0.99)),
                "p_value": float((null >= obs).mean()),
                "parsed_suffix_not_in_table": sum(1 for x in suf if x not in tbl) / len(suf)}
    log(f"D {cell:<12} TV {obs:.4f} null99 {Dd[cell]['null_p99']:.4f} p {Dd[cell]['p_value']:.3f} "
        f"not-in-table {Dd[cell]['parsed_suffix_not_in_table']:.4f}")
report["D_parse"] = Dd
save()

# ---------------------------------------------------------------- E n-matching (CR1 files)
E = {}
for s, label, st in MEASURABLE:
    tag = tag_of(label, st)
    real = _one_per_host(pools(B, label, st)["name"], SEED)
    gens = [json.loads((REPO / f"data/cr1/gen/{tag}_g{g}.json").read_text(encoding="utf-8")) for g in range(G)]
    E[tag] = {}
    for f in ("LBL", "SUF", "SUB", "PTH", "QRY"):
        if st.get(f) is False:
            continue
        rc = sum(1 for u in real if FIELDS[f](u))
        cap = min(1200, rc // 3)
        fc = [sum(1 for u in x if FIELDS[f](u)) for x in gens]
        E[tag][f] = {"real_nonempty": rc, "cap": cap, "fake_nonempty_per_gen": fc,
                     "fake_limited": min(fc) < cap, "insufficient": min(cap, min(fc)) < 250}
    lim = [f for f, v in E[tag].items() if v["fake_limited"] and not v["insufficient"]]
    log(f"E {tag:<16} fake-limited (CR1): {lim or 'none'}")
report["E_n_matching"] = E
save()

# ---------------------------------------------------------------- F cost
prov = json.loads((REPO / "data/cr1/gen/PROVENANCE.json").read_text(encoding="utf-8"))
F = {}
for r in prov["files"]:
    tag = r["file"].rsplit("_g", 1)[0]
    a = F.setdefault(tag, {"first_pass": 0, "n": 0, "draws": 0, "seconds": 0.0})
    a["first_pass"] += r["stats"]["first_pass"]; a["n"] += r["n"]
    a["draws"] += r["stats"]["draws"]; a["seconds"] += r["seconds"]
for tag, a in F.items():
    a["first_pass_rate"] = a["first_pass"] / a["n"]
    a["bound_tilt"] = (1 - a["first_pass_rate"]) / a["first_pass_rate"]
report["F_cost_cr1"] = F
report["F_cr1_total_minutes"] = prov["total_minutes"]
save()
log("F done")

# ---------------------------------------------------------------- C2 SUF ceilings
if not args.skip_c2:
    C2 = report.setdefault("C2_suf_ceiling", {})
    for s, label, st in MEASURABLE:
        tag = tag_of(label, st)
        pop = pools(B, label, st)
        for ti, kind in enumerate(("T", "C", "S")):
            rows = [r[0] for r in table_rows(kind, label, st)]
            by_gen = []
            t0 = time.time()
            for g in range(G):
                rng = random.Random(DR_SEED + 1000 * s + 10 * ti + g)
                fake = rng.choices(rows, k=3000)
                ex = []
                for d in range(D):
                    r = discriminate_with_floor(pop["name"], fake, "SUF", n=min(3000, 1200),
                                                seed=split_seed(SEED, g, d, "v2"))
                    gc.collect()
                    if not r.get("insufficient"):
                        ex.append(r["excess"])
                by_gen.append(ex)
            acc = s261_verdict(by_gen)
            C2.setdefault(tag, {})[kind] = acc
            save()
            log(f"C2 {tag:<16} {kind}: x {acc.get('x')} se {acc.get('se')} {acc['verdict']} "
                f"({time.time() - t0:.0f}s)")
log("DONE")
