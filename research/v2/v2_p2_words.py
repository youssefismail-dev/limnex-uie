"""
research/v2/v2_p2_words.py

V2-P2 -- the word readings, the blind spot, copying, creativity and the host guards
(research/v2/PREREG_V2_P2.md sections 7 and 9). Laptop CPU, after both arms' VERIFY_GENERATE.

    python research/v2/v2_p2_words.py --part samples   # both arms' samples beside R2 Part A's
    python research/v2/v2_p2_words.py --part run       # the judge fits (resumable)
    python research/v2/v2_p2_words.py --part metrics   # R3's measures, W1, the bands, the host guards

**The samples** (section 7). Real A, B and C, `e16_b5`'s `base` and the copier control O4 are R2
Part A's own (data/v2/r2/samples). Each arm is drawn from its free files by R0's rule, as
source 3 (`p2_a`) and 4 (`p2_b`): seed 20261022 + 10,000 x (source + 1) + 100 x class + g,
plus 1,000 for the host view and 2,000 for the whole URL. If an arm's file holds fewer
rest-bearing URLs than N_R, every source's rest samples in that class are cut to the largest
multiple of 100 that fits both arms, each keeping its first n (a recorded deviation).

**The fits** use E18's fold rule, seed_of(class, g) x 10 + the task's index: floor 0, base 1
and O4 9 are R2's own, so `e16_b5`'s readings and the copier control reproduce R2's exactly
(V-D1b); `p2_a` is 13 and `p2_b` 14. Per view (section 7):
    R  D1b, D1, D4, D1b-x        H  D1b, D1b-x, D4        W  D1b, D1, D0, D1b-x

**The metrics** (section 9), on the first 1,600 URLs of each free file (8,000 per class), real
held-out URLs beside them: R3's measures (v2_p1_words.part_metrics's), W1 and the word bands,
tokens per URL, examples, and the host guards of section 9.6:
- H-RD: the registrable domain is a training one (G-COPY's domain row, the class's training
  rows);
- H-HOST: the full host equals a training URL's host (every training row of both classes and
  stream 1, the rows the tables learned from);
- H-LIVE: the full host equals a host in the offline threat snapshot (its hash recorded);
- H-LBL1 / H-SUB1: the registrable label is, or a subdomain label is, a label found in only one
  training source (section 4.3's unit: the registrable domain under the full suffix list,
  near-duplicate sites of one platform as one);
- H-LBL: the registrable label is any training label;
- H-TYPO: host words that are lookalikes: never seen, and one edit from a word seen 10+ times
  (R1-3's rule, the class's training counts);
- H-DUP: URLs whose full host repeats within its file;
- H-VAL: URLs the sample gate rejects for the host (a label over 63 characters, a host too
  long, no public suffix, unparseable).
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
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

import v2_p1_words as W1M  # noqa: E402  (P1's helpers, unchanged)
import v2_p2_common as C  # noqa: E402
import v2_r0 as R0  # noqa: E402
import v2_r1 as R1  # noqa: E402
import v2_r2 as R2  # noqa: E402

SD = R2.SD
ARMS = ("p2_a", "p2_b")
DATA = Path(os.environ["V2_DATA"]) if os.environ.get("V2_DATA") else ROOT / "data" / "v2"
OUT = DATA / "p2" / "words"
SAMPLES = OUT / "samples"
LABELS, G = SD.LABELS, SD.G
SOURCE_INDEX = {"p2_a": 3, "p2_b": 4}
TASK_INDEX = {"floor": 0, "base": 1, "O4": 9, "p2_a": 13, "p2_b": 14}
VIEWS = {"R": "rest", "H": "host", "W": "whole"}
DETECTORS = {"R": ("D1b", "D1", "D4", "D1x"), "H": ("D1b", "D1x", "D4"), "W": ("D1b", "D1", "D0", "D1x")}
M_PER_G = R2.M_PER_G                                  # R3's 1,600 per file
FEED = ROOT / "data" / "feeds" / "openphish" / "feed.txt"
HOST_REASONS = ("label_too_long", "host_too_long", "no_public_suffix", "unparseable_host")
TABLES_DIR = ROOT / "data" / "v2" / "p2" / "tables"
WHO = ("real", "base") + ARMS


def gen_dir(arm: str) -> Path:
    return DATA / arm / "generate" / "gen"


def arm_file(arm: str, label: str, g: int) -> Path:
    return gen_dir(arm) / f"free_{arm}_{label}_g{g}.json"


def arm_seed(arm: str, label: str, g: int, view: str) -> int:
    return (R0.SEED + 10_000 * (1 + SOURCE_INDEX[arm]) + {"R": 0, "H": 1_000, "W": 2_000}[view]
            + 100 * LABELS.index(label) + g)


# --------------------------------------------------------------------------- #
# samples and fits
# --------------------------------------------------------------------------- #

def part_samples() -> None:
    SAMPLES.mkdir(parents=True, exist_ok=True)
    meta: dict = {}
    for label in LABELS:
        full = {(arm, g): R0._full(arm_file(arm, label, g)) for arm in ARMS for g in range(G)}
        rest = {k: [s for s in v if R0.rest_bearing(s)] for k, v in full.items()}
        nr = R1._read(R0.OUT / "meta.json")[label]["N_R"]
        cut = min(nr, (min(len(v) for v in rest.values()) // 100) * 100)
        meta[label] = {"N_R": nr, "N_R_used": cut, "deviation": cut != nr,
                       "rest_bearing": {arm: [len(rest[(arm, g)]) for g in range(G)] for arm in ARMS}}
        for g in range(G):
            for view in VIEWS:
                src = R1._read(R2.SAMPLES / f"A_{VIEWS[view]}_{label}_g{g}.json")
                node = {k: src[k] for k in ("A", "B", "C", "base")}
                if view == "R":
                    node["O4"] = src["O4"]
                    for arm in ARMS:
                        pick = random.Random(arm_seed(arm, label, g, view)).sample(rest[(arm, g)], cut)
                        node[arm] = [R0.mask_host(s) for s in pick]
                    if cut != nr:
                        node = {k: v[:cut] for k, v in node.items()}
                else:
                    for arm in ARMS:
                        pick = random.Random(arm_seed(arm, label, g, view)).sample(full[(arm, g)], SD.N)
                        node[arm] = [R0.host_only(s) for s in pick] if view == "H" else pick
                R1._write(SAMPLES / f"{view}_{label}_g{g}.json", node)
        print(f"  {label}: N_R {nr} (used {cut}); rest-bearing {meta[label]['rest_bearing']}", flush=True)
    R1._write(OUT / "samples_meta.json", meta, indent=1)


def part_run() -> None:
    t_all = time.time()
    dest = OUT / "fits.json"
    res = R1._resume(dest)
    for label in LABELS:
        fps = {"D1b": R2.make_d1b(label), "D1": SD.make_d1(label), "D0": SD.d0_fit_predict,
               "D4": R1.d4_fit_predict, "D1x": W1M.make_d1x(label)}
        for view, dets in DETECTORS.items():
            for g in range(G):
                node = R1._read(SAMPLES / f"{view}_{label}_g{g}.json")
                seed = SD.seed_of(label, g) * 10
                for det in dets:
                    key = f"{det}|{view}|{label}|{g}"
                    R1._task(res, dest, f"{key}|floor", fps[det], node["A"], node["B"], seed, t_all)
                    for task in ("base",) + ARMS + (("O4",) if (det, view) == ("D1b", "R") else ()):
                        R1._task(res, dest, f"{key}|{task}", fps[det], node["C"], node[task],
                                 seed + TASK_INDEX[task], t_all)
        del fps
        gc.collect()
    print("run done", flush=True)


# --------------------------------------------------------------------------- #
# metrics
# --------------------------------------------------------------------------- #

def blind_spot(label: str) -> dict:
    """W1 (section 9.4): per view (R, H), the share of URLs with at least one half-B-only word,
    pooled over the 5 generations, for real C, base and both arms; the half-A-only share beside."""
    a, b = R2.memory_vocab(label), R2.judge_vocab(label)
    out = {}
    for view in ("R", "H"):
        node = {}
        for who in ("C", "base") + ARMS:
            urls = [s for g in range(G) for s in R1._read(SAMPLES / f"{view}_{label}_g{g}.json")[who]]
            ws = W1M._words(urls)
            node[who] = {"n": len(urls), "b_only": sum(any(w in b and w not in a for w in x) for x in ws) / len(urls),
                         "a_only": sum(any(w in a and w not in b for w in x) for x in ws) / len(urls)}
        out[view] = node
    return out


def bands(label: str) -> dict:
    """R2's diagnosis (reported): shares of words by the judge's count, on the R and H samples."""
    judge = R2.judge_vocab(label)
    edges = ((0, 0), (1, 9), (10, 99), (100, 999), (1000, None))
    out = {}
    for view in ("R", "H"):
        node = {}
        for who in ("C", "base") + ARMS:
            urls = [s for g in range(G) for s in R1._read(SAMPLES / f"{view}_{label}_g{g}.json")[who]]
            counts = [judge.get(w, 0) for x in W1M._words(urls) for w in x]
            node[who] = {f"{lo}-{hi if hi is not None else 'up'}": float(np.mean([(c >= lo) and (hi is None or c <= hi)
                                                                                for c in counts]))
                         for lo, hi in edges} if counts else {}
            node[who]["words"] = len(counts)
        out[view] = node
    return out


def full_host(parts) -> str:
    return (parts.host_raw or "").lower().rstrip(".")


class HostIndex:
    """What the host guards look up, from the rows the piece tables learned from (every kept
    training row of both classes and stream 1; v2_p2_tables.rows): the full hosts, and for
    each registrable and subdomain label the source it is found in -- or MANY."""
    MANY = "\x00many"

    def __init__(self) -> None:
        import v2_p2_tables as T
        rep = dict(json.loads((TABLES_DIR / "clusters.json").read_text(encoding="utf-8")))
        src = T.Sources()
        self.hosts: set = set()
        self.label_src: dict = {}
        self.sub_src: dict = {}
        t0 = time.time()
        n = 0
        for _cls, _url, _stream, parts in T.rows():
            key = src(parts.host_raw)[0]
            key = rep.get(key, key)
            self.hosts.add(full_host(parts))
            lab = (parts.registrable_label or "").lower()
            if lab:
                self._see(self.label_src, lab, key)
            for s in parts.subdomains:
                if s:
                    self._see(self.sub_src, s.lower(), key)
            n += 1
        print(f"  host index: {n:,} rows, {len(self.hosts):,} hosts, {len(self.label_src):,} labels, "
              f"{len(self.sub_src):,} subdomain labels ({time.time() - t0:.0f}s)", flush=True)

    def _see(self, table: dict, word: str, key: str) -> None:
        old = table.get(word)
        if old is None:
            table[word] = key
        elif old != key and old != self.MANY:
            table[word] = self.MANY

    def single(self, table: dict, word: str) -> bool:
        v = table.get(word)
        return v is not None and v != self.MANY


def feed_hosts() -> tuple:
    from uie.factorization import parse
    lines = [x.strip() for x in FEED.read_text(encoding="utf-8").splitlines() if x.strip()]
    hosts = {full_host(parse(u)) for u in lines}
    return hosts, hashlib.sha256(FEED.read_bytes()).hexdigest(), len(lines)


def host_measures(label: str, srcs: dict, hidx: HostIndex, feed: set, vocab, rd_copy: dict) -> dict:
    """Section 9.6's measures for each source's 8,000 URLs (5 files of 1,600, in file order).
    `vocab` holds the class's word counts over the whole training data (E18's, as R1-3 read
    them): H-TYPO's "seen" and "seen 10+ times" are counts in training, not the judge's half."""
    from uie.factorization import parse
    from uie.gate import Gate
    counts_of = vocab.counts
    out = {}
    for who, urls in srcs.items():
        parts = [parse(u) for u in urls]
        hosts = [full_host(p) for p in parts]
        labs = [(p.registrable_label or "").lower() for p in parts]
        words = [w for p in parts for f in ([p.registrable_label or ""] + list(p.subdomains))
                 for w in R2.WORD.findall(f.lower())]
        seen = counts_of(words) if words else np.zeros(0)
        unseen = sorted({w for w, c in zip(words, seen) if int(c) == 0})
        near = {w: R1.anatomy(w, counts_of) == "near" for w in unseen}
        typo = sum(1 for w, c in zip(words, seen) if int(c) == 0 and near[w])
        dup = 0
        for g in range(G):                                  # within each file's 1,600
            block = hosts[g * M_PER_G:(g + 1) * M_PER_G]
            c = Counter(block)
            dup += sum(1 for h in block if c[h] > 1)
        gate = Gate()
        bad_host = sum(1 for u in urls if any(r in HOST_REASONS for r in gate.check(u).reasons))
        n = len(urls)
        out[who] = {"n": n,
                    "H-RD": rd_copy[who],
                    "H-HOST": sum(h in hidx.hosts for h in hosts) / n,
                    "H-LIVE": sum(h in feed for h in hosts) / n,
                    "H-LBL1": sum(hidx.single(hidx.label_src, x) for x in labs) / n,
                    "H-SUB1": sum(any(hidx.single(hidx.sub_src, s.lower()) for s in p.subdomains) for p in parts) / n,
                    "H-LBL": sum(x in hidx.label_src for x in labs) / n,
                    "H-TYPO": typo / len(words) if words else None, "host_words": len(words),
                    "H-DUP": dup / n,
                    "H-VAL": bad_host / n}
    return out


def tokens_per_url(label: str, srcs: dict) -> dict:
    """Reported: each arm's tokens per URL (canonical, its own tokenizer) against characters."""
    rec = json.loads((ROOT / "data" / "v2" / "p1b" / C.E16_CAP_RECORD).read_text(encoding="utf-8"))
    out = {}
    for arm in ARMS:
        tok = C.tokenizer(rec["tokenizer"], arm, TABLES_DIR)
        urls = srcs[arm]
        n_tok = [len(tok.encode(u, label)) for u in urls]
        n_chr = [len(tok.encode_chars(u, label)) for u in urls]
        out[arm] = {"tokens": st.mean(n_tok), "character_encoding": st.mean(n_chr),
                    "ratio": sum(n_tok) / sum(n_chr), "over_384_characters": sum(x > 384 for x in n_chr)}
    return out


def unseen_examples(urls, judge, label: str, k: int = 16) -> list:
    """Reported: k random path and query words the judge has never seen (v2_p1_words's, seeded
    from e16_cap's seed so it reads no P1 constant)."""
    from uie.evaluation import FIELDS
    pool = [w for u in urls for f in ("PTH", "QRY") for w in R2.WORD.findall((FIELDS[f](u) or "").lower())
            if judge.counts([w])[0] == 0]
    rng = random.Random(C.E16.SEED + 7 + LABELS.index(label))
    return rng.sample(pool, min(k, len(pool)))


def part_metrics() -> None:
    """R3's instrument (v2_r2.part_metrics's measures, as v2_p1_words) on the first 1,600 URLs
    of each free file, real against e16_b5 against both arms; W1, the bands, the host guards."""
    from cr1_score import pools
    from qc1_common import NearIndex
    from uie.bloom import BloomFilter
    from uie.factorization import parse
    from uie.gate import Gate
    bloom = BloomFilter.load(R1.BLOOM)
    hidx = HostIndex()
    feed, feed_sha, feed_n = feed_hosts()
    out: dict = {"threat_snapshot": {"file": str(FEED.relative_to(ROOT)), "sha256": feed_sha, "urls": feed_n,
                                     "hosts": len(feed)}}
    for label in LABELS:
        t0 = time.time()
        judge = R2.DictVocab(R2.judge_vocab(label))
        real = R1._real_pool(pools, label)
        srcs = {"real": [s for g in range(G) for s in real[g][:M_PER_G]],
                "base": [s for g in range(G)
                         for s in R1._full(str(SD.SOURCES["e16_b5"]).format(label=label, g=g))[:M_PER_G]]}
        for arm in ARMS:
            srcs[arm] = [s for g in range(G) for s in R1._full(arm_file(arm, label, g))[:M_PER_G]]
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
        rd_copy = {k: node["RD"][k].get("copy", 0.0) for k in srcs}
        full_vocab = SD.Vocab(label)
        node["host"] = host_measures(label, srcs, hidx, feed, full_vocab, rd_copy)
        del full_vocab
        node["blind_spot"] = blind_spot(label)
        node["bands"] = bands(label)
        node["tokens_per_url"] = tokens_per_url(label, srcs)
        node["examples"] = {k: unseen_examples(srcs[k], judge, label) for k in WHO}
        node["host_examples"] = {k: random.Random(C.E16.SEED + LABELS.index(label)).sample(
            [R0.host_only(u) for u in srcs[k]], 16) for k in WHO}
        out[label] = node
        print(f"  {label}: metrics done ({time.time() - t0:.0f}s)", flush=True)
        del tables, frequent, vals, judge
        gc.collect()
    R1._write(OUT / "metrics.json", out, indent=1)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--part", required=True, choices=("samples", "run", "metrics"))
    args = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")
    for arm in ARMS:
        v = R1._read(DATA / arm / "VERIFY_GENERATE.json")
        if v.get("verdict") != "PASS":
            raise SystemExit(f"STOP: {arm}'s VERIFY_GENERATE has not passed; nothing is read")
    OUT.mkdir(parents=True, exist_ok=True)
    {"samples": part_samples, "run": part_run, "metrics": part_metrics}[args.part]()


if __name__ == "__main__":
    main()
