"""
research/v2/v2_r0.py

V2-P1, step R0 (research/v2/PREREG_V2_P1.md sections 14.10 and 15.3): the frozen engines'
word tell in the path, query and fragment, read before anything is built.

    python research/v2/v2_r0.py --part data     # N_R and the samples (written once)
    python research/v2/v2_r0.py --part run      # V-D1, D1-R and D1-H (resumable)
    python research/v2/v2_r0.py --part read     # the reading -> data/v2/r0/r0_result.json

The detector is E18's D1, imported unchanged from research/e18/strong_disc.py: gradient
boosting on global and vocabulary features, 5-fold out-of-fold ROC AUC, with the class's
vocabulary built from its real training pools.

  D1-R  the URL with its host replaced by `h.h`, over URLs that carry a path body, a query or
        a fragment; N_R per side (section 6.1's rule)
  D1-H  the host alone, over every URL: E18's own cached samples, N = 6,000
  V-D1  E18's whole-URL D1, rerun on E18's cached samples: must equal E18's record exactly

Each reading is a gate cell's: excess = AUC(real C against the source) - AUC(real A against
real B), the mean over the 5 generations, t = (mean - 0.05) / SE against 2.776.
"""
from __future__ import annotations

import argparse
import gc
import json
import random
import statistics as st
import sys
import time
from pathlib import Path
from typing import Dict, List

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
for p in (ROOT, ROOT / "research" / "e18"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import strong_disc as SD  # noqa: E402  (it puts research/cr1 and research/native on the path)

OUT = ROOT / "data" / "v2" / "r0"
SAMPLES = OUT / "samples"
LABELS, G = SD.LABELS, SD.G
SEED = 20261022                                  # section 15.3
N_CAP = 6_000
SOURCES = {"e16_b5": SD.SOURCES["e16_b5"],
           "e15_new": ROOT / "data" / "e15" / "generate" / "gen" / "free_e15_new_{label}_g{g}.json"}
SOURCE_INDEX = {"e16_b5": 0, "e15_new": 1}
#: Fold seeds follow E18's rule, seed_of(class, g) * 10 + the task's index. Floor and e16_b5 keep
#: E18's indices, so V-D1 splits exactly as E18 did.
TASK_INDEX = {"floor": 0, "e16_b5": 1, "e15_new": 4}
TASKS = {"W": ("floor", "e16_b5"), "R": ("floor", "e16_b5", "e15_new"),
         "H": ("floor", "e16_b5", "e15_new")}
NAMES = {"W": "V-D1 (whole URL)", "R": "D1-R (path, query, fragment)", "H": "D1-H (host)"}
MASK = "h.h"                                     # a host with no word of three letters


# --------------------------------------------------------------------------- #
# the transforms and the registered rules
# --------------------------------------------------------------------------- #

def rest_bearing(s: str) -> bool:
    """A scheme-stripped URL with a non-empty path body, query or fragment -- the URLs the
    gate would score a PTH, QRY or FRG field on."""
    _, path, query, frag = SD.split_url(s)
    return path not in ("", "/") or bool(query) or bool(frag)


def mask_host(s: str) -> str:
    """The URL with its host replaced by a constant, so only the path, query and fragment
    carry information."""
    _, path, query, frag = SD.split_url(s)
    return SD.join_url(MASK, path, query, frag)


def host_only(s: str) -> str:
    return SD.split_url(s)[0]


def n_r(real_rest: List[int], e16_rest: List[int]) -> int:
    """Section 6.1: the largest multiple of 100 not above 6,000, a quarter of the real pool's
    rest-bearing URLs, or 0.9 x the fewest rest-bearing URLs in any e16_b5 file."""
    cap = min(N_CAP, min(real_rest) // 4, int(0.9 * min(e16_rest)))
    return (cap // 100) * 100


def seed_real(label: str, g: int) -> int:
    return SEED + 100 * LABELS.index(label) + g


def seed_source(src: str, label: str, g: int, kind: str = "R") -> int:
    """D1-R's draw; D1-H's adds 1,000 (section 15.3)."""
    return (SEED + 10_000 * (1 + SOURCE_INDEX[src]) + (1_000 if kind == "H" else 0)
            + 100 * LABELS.index(label) + g)


def _full(path) -> List[str]:
    from uie.evaluation import FIELDS
    urls = json.loads(Path(path).read_text(encoding="utf-8"))
    return [x for x in (FIELDS["FULL"](u) for u in urls) if x]


def _write(path: Path, obj) -> None:
    path.write_text(json.dumps(obj), encoding="utf-8")


# --------------------------------------------------------------------------- #
# part data
# --------------------------------------------------------------------------- #

def part_data() -> None:
    from cr1_score import pools
    from uie.evaluation import FIELDS, _one_per_host
    SAMPLES.mkdir(parents=True, exist_ok=True)
    meta: Dict[str, dict] = {}
    for label in LABELS:
        t0 = time.time()
        pop = pools(SD.BUNDLE, label)["mixed"]
        real = {}
        for g in range(G):
            full = [x for x in (FIELDS["FULL"](u) for u in _one_per_host(pop, SD.seed_of(label, g)))
                    if x]
            real[g] = [s for s in full if rest_bearing(s)]
        files = {src: {g: _full(str(p).format(label=label, g=g)) for g in range(G)}
                 for src, p in SOURCES.items()}
        rest = {src: {g: [s for s in files[src][g] if rest_bearing(s)] for g in range(G)}
                for src in SOURCES}
        nr = n_r([len(real[g]) for g in range(G)], [len(rest["e16_b5"][g]) for g in range(G)])
        meta[label] = {"N_R": nr, "real_rest": [len(real[g]) for g in range(G)],
                       **{f"{src}_rest": [len(rest[src][g]) for g in range(G)] for src in SOURCES}}
        for g in range(G):
            pool = list(real[g])
            random.Random(seed_real(label, g)).shuffle(pool)
            node = {k: [mask_host(s) for s in pool[i * nr:(i + 1) * nr]] for i, k in enumerate("ABC")}
            for src in SOURCES:
                if len(rest[src][g]) < nr:
                    raise SystemExit(f"STOP: {src} {label} g{g} holds {len(rest[src][g])} "
                                     f"rest-bearing URLs, fewer than N_R {nr}")
                pick = random.Random(seed_source(src, label, g)).sample(rest[src][g], nr)
                node[src] = [mask_host(s) for s in pick]
            node["meta"] = {"N": nr, "seed_real": seed_real(label, g)}
            _write(SAMPLES / f"R_{label}_g{g}.json", node)
            # D1-H: E18's cached samples, hosts only; e15_new drawn as E18 drew e16_b5
            e18 = json.loads((SD.SAMPLES / f"{label}_g{g}.json").read_text(encoding="utf-8"))
            hnode = {k: [host_only(s) for s in e18[k]] for k in ("A", "B", "C", "e16_b5")}
            pick = random.Random(seed_source("e15_new", label, g, "H")).sample(files["e15_new"][g], SD.N)
            hnode["e15_new"] = [host_only(s) for s in pick]
            hnode["meta"] = {"N": SD.N, "from": "data/e18/strong_disc/samples"}
            _write(SAMPLES / f"H_{label}_g{g}.json", hnode)
        print(f"  {label}: N_R {nr}; real rest-bearing {meta[label]['real_rest']}; "
              f"e16_b5 {meta[label]['e16_b5_rest']}; e15_new {meta[label]['e15_new_rest']} "
              f"({time.time() - t0:.0f}s)", flush=True)
    (OUT / "meta.json").write_text(json.dumps(meta, indent=1), encoding="utf-8")


# --------------------------------------------------------------------------- #
# part run
# --------------------------------------------------------------------------- #

def _node(kind: str, label: str, g: int) -> dict:
    path = (SD.SAMPLES / f"{label}_g{g}.json" if kind == "W"
            else SAMPLES / f"{kind}_{label}_g{g}.json")
    return json.loads(path.read_text(encoding="utf-8"))


def part_run() -> None:
    t_all = time.time()
    for kind in ("W", "R", "H"):          # V-D1 first: a drifted pipeline invalidates the rest
        dest = OUT / f"D1{kind}.json"
        res = json.loads(dest.read_text(encoding="utf-8")) if dest.exists() else {}
        for label in LABELS:
            fp = SD.make_d1(label)
            for g in range(G):
                node = _node(kind, label, g)
                for task in TASKS[kind]:
                    key = f"{label}|{g}|{task}"
                    if key in res:
                        continue
                    t0 = time.time()
                    texts, y = SD.task_data(node, task)
                    res[key] = SD.oof_auc(fp, texts, y, SD.seed_of(label, g) * 10 + TASK_INDEX[task])
                    dest.write_text(json.dumps(res, indent=1), encoding="utf-8")
                    print(f"  D1-{kind} {label:<9} g{g} {task:<8} done  ({time.time() - t0:.0f}s; "
                          f"{(time.time() - t_all) / 60:.0f} min so far)", flush=True)
                    gc.collect()
            del fp
            gc.collect()
    print("run done", flush=True)


# --------------------------------------------------------------------------- #
# part read
# --------------------------------------------------------------------------- #

def _words(label: str) -> dict:
    """Descriptive: E18's word statistics on D1-R's samples (path, query and fragment only,
    per URL that has one), real C against each source, averaged over the 5 generations."""
    vocab = SD.Vocab(label)
    acc: Dict[str, Dict[str, list]] = {}
    for g in range(G):
        node = _node("R", label, g)
        for src in ("C",) + tuple(SOURCES):
            known = total = unseen = 0
            for s in node[src]:
                toks = SD.TOKEN.findall(s.lower())
                cnt = vocab.counts(toks)
                total += len(toks)
                known += int((cnt > 0).sum())
                unseen += int((cnt == 0).sum())
            row = acc.setdefault("real" if src == "C" else src, {"known_share": [], "unseen_per_url": []})
            row["known_share"].append(known / total if total else 1.0)
            row["unseen_per_url"].append(unseen / len(node[src]))
    return {src: {k: st.mean(v) for k, v in row.items()} for src, row in acc.items()}


def part_read() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    meta = json.loads((OUT / "meta.json").read_text(encoding="utf-8"))
    out: dict = {"registration": "research/v2/PREREG_V2_P1.md sections 14.10 and 15.3",
                 "meta": meta, "readings": {}}
    # V-D1: E18's record, reproduced exactly
    e18 = json.loads((SD.OUT / "D1.json").read_text(encoding="utf-8"))
    w = json.loads((OUT / "D1W.json").read_text(encoding="utf-8"))
    diffs = {k: abs(w[k] - e18[k]) for k in w}
    out["V-D1"] = {"tasks": len(w), "max_abs_diff": max(diffs.values()),
                   "pass": len(w) == 2 * G * len(TASKS["W"]) and max(diffs.values()) == 0.0}
    for kind in ("R", "H"):
        r = json.loads((OUT / f"D1{kind}.json").read_text(encoding="utf-8"))
        node: dict = {}
        for label in LABELS:
            fl = [r[f"{label}|{g}|floor"] for g in range(G)]
            node[f"{label} floor"] = {"mean_auc": st.mean(fl), "per_g": fl,
                                      "flag": abs(st.mean(fl) - 0.5) > 0.03}
            xs = {src: [r[f"{label}|{g}|{src}"] - r[f"{label}|{g}|floor"] for g in range(G)]
                  for src in SOURCES}
            for src in SOURCES:
                v = SD.verdict(xs[src])
                v["clear_line"] = SD.BAR - SD.TCRIT * v["se"]
                node[f"{label} {src}"] = v
            node[f"{label} e15_new - e16_b5"] = SD.paired_t(xs["e15_new"], xs["e16_b5"])
        out["readings"][NAMES[kind]] = node
    rr = out["readings"][NAMES["R"]]
    # section 14.4: a class can be read only if the instrument convicts e16_b5 there
    out["sensitivity"] = {label: rr[f"{label} e16_b5"]["verdict"] == "CONVICTED"
                          and not rr[f"{label} floor"]["flag"] for label in LABELS}
    out["early_stop"] = not any(out["sensitivity"].values())
    out["words"] = {label: _words(label) for label in LABELS}
    (OUT / "r0_result.json").write_text(json.dumps(out, indent=1, default=float), encoding="utf-8")

    print(f"V-D1: {out['V-D1']['tasks']} tasks, max |diff| from E18 {out['V-D1']['max_abs_diff']:.2e} "
          f"-> {'PASS' if out['V-D1']['pass'] else 'FAIL'}")
    for kind in ("R", "H"):
        node = out["readings"][NAMES[kind]]
        print(f"\n{NAMES[kind]}")
        print(f"{'class':<9} {'source':<8} {'excess':>8} {'SE':>7} {'t':>8}  {'verdict':<10} clear line")
        for label in LABELS:
            print(f"{label:<9} {'floor':<8} AUC {node[f'{label} floor']['mean_auc']:.4f}"
                  f"{'  FLAG' if node[f'{label} floor']['flag'] else ''}")
            for src in SOURCES:
                v = node[f"{label} {src}"]
                print(f"{label:<9} {src:<8} {v['mean']:>+8.4f} {v['se']:>7.4f} {v['t']:>8.2f}  "
                      f"{v['verdict']:<10} {v['clear_line']:.4f}")
            d = node[f"{label} e15_new - e16_b5"]
            print(f"{label:<9} e15_new - e16_b5 {d['mean_diff']:+.4f} (t {d['t']:.2f})")
    print("\nwords in the path, query and fragment (per URL that has one; mean of 5 generations)")
    for label in LABELS:
        for src, v in out["words"][label].items():
            print(f"  {label:<9} {src:<8} known {v['known_share']:.1%}  unseen per URL {v['unseen_per_url']:.2f}")
    print("\nN_R: " + ", ".join(f"{lab} {meta[lab]['N_R']}" for lab in LABELS))
    print(f"sensitivity (class readable): {out['sensitivity']}; early stop: {out['early_stop']}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--part", required=True, choices=("data", "run", "read"))
    args = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")
    OUT.mkdir(parents=True, exist_ok=True)
    {"data": part_data, "run": part_run, "read": part_read}[args.part]()


if __name__ == "__main__":
    main()
