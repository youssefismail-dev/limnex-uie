"""
research/v2/v2_p1_words.py

V2-P1 -- the word readings, the blind spot, and copying and creativity
(research/v2/PREREG_V2_P1_MEMORY.md sections 4.1-4.3 and 4.5). Laptop CPU, after
VERIFY_GENERATE.

    python research/v2/v2_p1_words.py --part samples   # p1_mem's samples beside R2 Part A's
    python research/v2/v2_p1_words.py --part run       # D1b (R, H, W), D1 (R, W), D0 (W), D4 (R),
                                                       #   the copier control (resumable)
    python research/v2/v2_p1_words.py --part metrics   # W1, word bands, R3's copying and creativity

The samples. Real A, B and C, `e16_b5`'s `base` and the copier control O4 are R2 Part A's
own, read from data/v2/r2/samples (R0's rest and host samples; E18's whole-URL samples).
p1_mem's are drawn from its free files by R0's rule as source index 2: seed 20261022 +
10,000 x 3 + 100 x class + g, plus 1,000 for the host view and 2,000 for the whole URL.
If a p1_mem file holds fewer rest-bearing URLs than N_R, every source's rest samples in
that class are cut to the largest multiple of 100 that fits, each keeping its first n
(recorded as a deviation).

The fits use E18's fold rule, seed_of(class, g) x 10 + the task's index: floor 0, base 1
and O4 9 are R2's own, so e16_b5's readings and the copier control must reproduce R2's
exactly (V-D1b); p1_mem is index 12.
"""
from __future__ import annotations

import argparse
import gc
import random
import statistics as st
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))

from v2_arm import P1  # noqa: E402  (V2_ARM: p1 or p1b)
import v2_r0 as R0  # noqa: E402
import v2_r1 as R1  # noqa: E402
import v2_r2 as R2  # noqa: E402

SD = R2.SD
from v2_arm import DATA as P1D  # noqa: E402
OUT = P1D / "words"
SAMPLES = OUT / "samples"
GEN = P1D / "generate" / "gen"
LABELS, G = SD.LABELS, SD.G
SOURCE_INDEX = 2
TASK_INDEX = {"floor": 0, "base": 1, "O4": 9, "p1": 12}
VIEWS = {"R": "rest", "H": "host", "W": "whole"}
#: which detectors read which view (section 4.1, 4.3); D1x is amendment 1's blind-spot judge
DETECTORS = {"R": ("D1b", "D1", "D4", "D1x"), "H": ("D1b",), "W": ("D1b", "D1", "D0")}
M_PER_G = R2.M_PER_G                                  # R3's 1,600 per file


def p1_file(label: str, g: int) -> Path:
    return GEN / f"free_{P1.ARM}_{label}_g{g}.json"


def p1_seed(label: str, g: int, view: str) -> int:
    return (R0.SEED + 10_000 * (1 + SOURCE_INDEX) + {"R": 0, "H": 1_000, "W": 2_000}[view]
            + 100 * LABELS.index(label) + g)


def part_samples() -> None:
    SAMPLES.mkdir(parents=True, exist_ok=True)
    meta: dict = {}
    for label in LABELS:
        full = {g: R0._full(p1_file(label, g)) for g in range(G)}
        rest = {g: [s for s in full[g] if R0.rest_bearing(s)] for g in range(G)}
        nr = R1._read(R0.OUT / "meta.json")[label]["N_R"]
        cut = min(nr, (min(len(v) for v in rest.values()) // 100) * 100)
        meta[label] = {"N_R": nr, "N_R_used": cut, "p1_rest": [len(rest[g]) for g in range(G)],
                       "deviation": cut != nr}
        for g in range(G):
            for view in VIEWS:
                src = R1._read(R2.SAMPLES / f"A_{VIEWS[view]}_{label}_g{g}.json")
                node = {k: src[k] for k in ("A", "B", "C", "base")}
                if view == "R":
                    node["O4"] = src["O4"]
                    pick = random.Random(p1_seed(label, g, view)).sample(rest[g], cut)
                    node["p1"] = [R0.mask_host(s) for s in pick]
                    if cut != nr:
                        node = {k: v[:cut] for k, v in node.items()}
                else:
                    pick = random.Random(p1_seed(label, g, view)).sample(full[g], SD.N)
                    node["p1"] = [R0.host_only(s) for s in pick] if view == "H" else pick
                R1._write(SAMPLES / f"{view}_{label}_g{g}.json", node)
        print(f"  {label}: N_R {nr} (used {cut}); p1_mem rest-bearing {meta[label]['p1_rest']}", flush=True)
    R1._write(OUT / "samples_meta.json", meta, indent=1)


def cross_half_vocab(label: str) -> dict:
    """Amendment 1's D1b-x vocabulary: Vocab_B's counts for the words also in Vocab_A, so a
    site word from half B looks as unseen as one from half A."""
    a = R2.memory_vocab(label)
    return {w: c for w, c in R2.judge_vocab(label).items() if w in a}


def make_d1x(label: str):
    """D1b-x (PREREG_V2_P1_MEMORY.md section 12.3): R2's D1b, unchanged, with the
    cross-half vocabulary."""
    vocab = R2.DictVocab(cross_half_vocab(label))
    cache: dict = {}

    def fit_predict(xtr, ytr, xte, seed):
        from sklearn.ensemble import HistGradientBoostingClassifier
        for s in set(xtr) | set(xte):
            if s not in cache:
                cache[s] = SD.d1_features(s, vocab)
        m = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.1, random_state=seed)
        m.fit(np.array([cache[s] for s in xtr], dtype=float), ytr)
        return m.predict_proba(np.array([cache[s] for s in xte], dtype=float))[:, 1]
    return fit_predict


def part_run() -> None:
    t_all = time.time()
    dest = OUT / "fits.json"
    res = R1._resume(dest)
    for label in LABELS:
        fps = {"D1b": R2.make_d1b(label), "D1": SD.make_d1(label), "D0": SD.d0_fit_predict,
               "D4": R1.d4_fit_predict, "D1x": make_d1x(label)}
        for view, dets in DETECTORS.items():
            for g in range(G):
                node = R1._read(SAMPLES / f"{view}_{label}_g{g}.json")
                seed = SD.seed_of(label, g) * 10
                for det in dets:
                    key = f"{det}|{view}|{label}|{g}"
                    R1._task(res, dest, f"{key}|floor", fps[det], node["A"], node["B"], seed, t_all)
                    for task in ("base", "p1") + (("O4",) if (det, view) == ("D1b", "R") else ()):
                        R1._task(res, dest, f"{key}|{task}", fps[det], node["C"], node[task],
                                 seed + TASK_INDEX[task], t_all)
        del fps
        gc.collect()
    print("run done", flush=True)


# --------------------------------------------------------------------------- #
# metrics
# --------------------------------------------------------------------------- #

def _words(strings):
    return [[w for w in R2.WORD.findall(s.lower())] for s in strings]


def blind_spot(label: str) -> dict:
    """W1 (section 4.2): per view (R, H), the share of URLs with at least one half-B-only
    word (in Vocab_B, not in Vocab_A), pooled over the 5 generations, for real C, base and
    p1; the half-A-only share beside it."""
    a, b = R2.memory_vocab(label), R2.judge_vocab(label)
    out = {}
    for view in ("R", "H"):
        node = {}
        for who in ("C", "base", "p1"):
            urls = [s for g in range(G) for s in R1._read(SAMPLES / f"{view}_{label}_g{g}.json")[who]]
            ws = _words(urls)
            b_only = sum(any(w in b and w not in a for w in x) for x in ws)
            a_only = sum(any(w in a and w not in b for w in x) for x in ws)
            node[who] = {"n": len(urls), "b_only": b_only / len(urls), "a_only": a_only / len(urls)}
        out[view] = node
    return out


def bands(label: str) -> dict:
    """R2's diagnosis: shares of words by the judge's count (never, 1-9, 10-99, 100-999,
    1,000+), on the R and H samples."""
    judge = R2.judge_vocab(label)
    edges = ((0, 0), (1, 9), (10, 99), (100, 999), (1000, None))
    out = {}
    for view in ("R", "H"):
        node = {}
        for who in ("C", "base", "p1"):
            urls = [s for g in range(G) for s in R1._read(SAMPLES / f"{view}_{label}_g{g}.json")[who]]
            counts = [judge.get(w, 0) for x in _words(urls) for w in x]
            node[who] = {f"{lo}-{hi if hi is not None else 'up'}": float(np.mean([(c >= lo) and (hi is None or c <= hi)
                                                                                for c in counts]))
                         for lo, hi in edges} if counts else {}
            node[who]["words"] = len(counts)
        out[view] = node
    return out


def part_metrics() -> None:
    """R3's instrument (v2_r2.part_metrics's measures) on the first 1,600 URLs of each free
    file, real against e16_b5 against p1_mem; then W1 and the word bands."""
    from cr1_score import pools
    from qc1_common import NearIndex
    from uie.bloom import BloomFilter
    from uie.factorization import parse
    from uie.gate import Gate
    bloom = BloomFilter.load(R1.BLOOM)
    out: dict = {}
    for label in LABELS:
        t0 = time.time()
        judge = R2.DictVocab(R2.judge_vocab(label))
        real = R1._real_pool(pools, label)
        srcs = {"real": [s for g in range(G) for s in real[g][:M_PER_G]],
                "base": [s for g in range(G) for s in R1._full(str(SD.SOURCES["e16_b5"]).format(label=label, g=g))[:M_PER_G]],
                "p1": [s for g in range(G) for s in R1._full(p1_file(label, g))[:M_PER_G]]}
        tables = R1._training_tables(label, parse)
        frequent = R1._frequent(label, parse, tables)
        pairs = np.load(R2.OUT / f"train_pairs_{label}.npy")
        node: dict = {"n": {k: len(v) for k, v in srcs.items()}}
        vals = {k: [R1._field_values(u, parse) for u in v] for k, v in srcs.items()}
        for f in ("LBL", "SUB", "PTH", "QRY", "RD"):
            index = NearIndex(frequent[f]) if f in frequent else None
            node[f] = {k: R1._copy_profile([x[f] for x in vals[k] if x[f]], tables[f], index) for k in srcs}
            del index
            gc.collect()
        gate = Gate(known_membership=bloom)
        node["gate"] = {}
        for k, urls in srcs.items():
            _, rep = gate.filter(urls)
            node["gate"][k] = {"n": rep.n_total, **{str(r): c / rep.n_total for r, c in rep.by_reason.items()}}
        node["pair_novelty"] = {k: R2._pair_novelty([x["PTH"] for x in vals[k] if x["PTH"]], pairs) for k in srcs}
        for grp, fields in (("host", ("LBL", "SUB")), ("rest", ("PTH", "QRY"))):
            words = {k: [w for x in vals[k] for f in fields for w in R2.WORD.findall(x[f].lower())] for k in srcs}
            n = min(20_000, min(len(w) for w in words.values()) // 2)
            seed = R2.R2_SEED + 70_000 + 100 * LABELS.index(label)
            node[f"distinct_{grp}"] = {"n": n, **{k: R1._distinct_per_1000(w, n, seed) for k, w in words.items()}}
            node[f"unseen_judge_{grp}"] = {k: (float(np.mean(judge.counts(w) == 0)) if w else None, len(w))
                                           for k, w in words.items()}
        node["structure"] = {k: {"path": st.mean(bool(x["PTH"]) for x in vals[k]),
                                 "query": st.mean(bool(x["QRY"]) for x in vals[k]),
                                 "path_chars": st.mean(len(x["PTH"]) for x in vals[k] if x["PTH"]),
                                 "path_segments": st.mean(x["PTH"].count("/") + 1 for x in vals[k] if x["PTH"])}
                             for k in srcs}
        node["blind_spot"] = blind_spot(label)
        node["bands"] = bands(label)
        node["examples"] = {k: _unseen_examples(srcs[k], judge, label) for k in ("real", "base", "p1")}
        out[label] = node
        print(f"  {label}: metrics done ({time.time() - t0:.0f}s)", flush=True)
        del tables, frequent, vals, judge
        gc.collect()
    R1._write(OUT / "metrics.json", out, indent=1)


def _unseen_examples(urls, judge, label: str, k: int = 16) -> list:
    """Reported: k random words the judge has never seen, from paths and queries (seeded)."""
    from uie.evaluation import FIELDS
    pool = [w for u in urls for f in ("PTH", "QRY") for w in R2.WORD.findall((FIELDS[f](u) or "").lower())
            if judge.counts([w])[0] == 0]
    rng = random.Random(P1.P1_SEED + 7 + LABELS.index(label))
    return rng.sample(pool, min(k, len(pool)))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--part", required=True, choices=("samples", "run", "metrics"))
    args = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")
    v = R1._read(P1D / "VERIFY_GENERATE.json")
    if v.get("verdict") != "PASS":
        raise SystemExit("STOP: VERIFY_GENERATE has not passed; nothing is read")
    OUT.mkdir(parents=True, exist_ok=True)
    {"samples": part_samples, "run": part_run, "metrics": part_metrics}[args.part]()


if __name__ == "__main__":
    main()
