"""
research/v2/v2_r1.py

V2-R1, the lexical diagnosis (research/v2/PREREG_R1.md). Laptop only, on existing files: no
model is trained, no GPU is used, nothing is generated.

    python research/v2/v2_r1.py --part data      # the samples, the piece-learning sample, input hashes
    python research/v2/v2_r1.py --part bpe       # the rest and host piece tables, to 32,000 merges
    python research/v2/v2_r1.py --part run       # V-D1', D1 per field, D4 and its controls
    python research/v2/v2_r1.py --part words     # R1-2 the frequency profile, R1-3 the anatomy
    python research/v2/v2_r1.py --part cover     # R1-4 coverage
    python research/v2/v2_r1.py --part base      # R1-6 the reference values for P1
    python research/v2/v2_r1.py --part verify    # every validity check (section 4)
    python research/v2/v2_r1.py --part read      # the reading and the decision rules (section 5)

Every part writes under data/v2/r1/ and can be rerun; `run` resumes where it stopped.
"""
from __future__ import annotations

import argparse
import gc
import gzip
import hashlib
import json
import random
import re
import statistics as st
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
for p in (ROOT, ROOT / "research" / "e18", ROOT / "research" / "qc1", HERE):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import strong_disc as SD  # noqa: E402  (E18's detectors, unchanged; adds research/cr1, research/native)
import v2_bpe as BPE      # noqa: E402
import v2_r0 as R0        # noqa: E402

OUT = ROOT / "data" / "v2" / "r1"
SAMPLES = OUT / "samples"
LABELS, G = SD.LABELS, SD.G
R1_SEED = 20261030                                   # section 3
FIELDS_R1 = ("LBL", "SUB", "PTH", "QRY")
FIELD_INDEX = {f: i for i, f in enumerate(FIELDS_R1)}
GROUP = {"LBL": "host", "SUB": "host", "PTH": "rest", "QRY": "rest"}
SOURCES = {"e16_b5": SD.SOURCES["e16_b5"], "e15_new": R0.SOURCES["e15_new"]}
SOURCE_INDEX = {"e16_b5": 0, "e15_new": 1}
#: E18's fold rule, seed_of(class, g) * 10 + the task's index; floor, e16_b5, e9_ln and chimera keep
#: E18's indices, so those tasks split exactly as E18 split them.
TASK_INDEX = {"floor": 0, "e16_b5": 1, "e9_ln": 2, "chimera": 3, "e15_new": 4, "shuffle": 5}
D4_TASKS = ("floor", "e16_b5", "e9_ln", "chimera", "e15_new", "shuffle")
N_CAP, N_MIN = 6_000, 300
BAND_EDGES = (1_000, 8_000, 32_000)
BANDS = ("top1k", "1k-8k", "8k-32k", "beyond", "unseen")
TABLE_SIZES = (2_000, 4_000, 8_000, 16_000, 32_000)
BPE_ROWS, BPE_MERGES, BPE_SEED = 600_000, 32_000, R1_SEED + 7
BPE_HOURS = 3.0                                      # section 3: over this, the sample halves
LABEL_COUNTS = {"benign": 906_214, "phishing": 192_652}
WORD = SD.TOKEN                                      # D1's word: [a-z]{3,}
NEAR_MIN, JOIN_MIN, FIELD_NEAR_MIN = 10, 10, 5
DIV_WORDS = 20_000
FEED = ROOT / "data" / "feeds" / "openphish" / "feed.txt"
BLOOM = ROOT / "data" / "bundle2" / "known_urls.bloom.gz"
E9_LN = SD.SOURCES["e9_ln"]
_AZ = "abcdefghijklmnopqrstuvwxyz"


# --------------------------------------------------------------------------- #
# the registered rules and transforms
# --------------------------------------------------------------------------- #

def seed_real(field: str, label: str, g: int) -> int:
    return R1_SEED + 1_000 * FIELD_INDEX[field] + 100 * LABELS.index(label) + g


def seed_source(src: str, field: str, label: str, g: int) -> int:
    return (R1_SEED + 100_000 * (1 + SOURCE_INDEX[src]) + 1_000 * FIELD_INDEX[field]
            + 100 * LABELS.index(label) + g)


def seed_shuffle(label: str, g: int) -> int:
    return R1_SEED + 50_000 + 100 * LABELS.index(label) + g


def n_f(real_counts: Sequence[int], e16_counts: Sequence[int]) -> int:
    """Section 3, R1-1: the largest multiple of 100 not above 6,000, a quarter of the real pool's
    field-bearing URLs, or 0.9 x the fewest field-bearing URLs in an e16_b5 file."""
    cap = min(N_CAP, min(real_counts) // 4, int(0.9 * min(e16_counts)))
    return max(0, (cap // 100) * 100)


def d1_string(field: str, value: str) -> str:
    """The string D1 reads for one field: LBL and SUB alone, as a host; PTH and QRY behind a
    constant host that holds no word."""
    if field in ("LBL", "SUB"):
        return value
    if field == "PTH":
        return "h.h/" + value
    if field == "QRY":
        return "h.h?" + value
    raise KeyError(field)


def _permute(text: str, rng: random.Random) -> str:
    low = text.lower()
    if len(low) != len(text):
        return text                                  # lower-casing moved a span: leave it whole
    spans = [(m.start(), m.end()) for m in WORD.finditer(low)]
    if len(spans) < 2:
        return text
    words = [text[a:b] for a, b in spans]
    order = list(range(len(words)))
    rng.shuffle(order)
    out, last = [], 0
    for (a, b), k in zip(spans, order):
        out.append(text[last:a])
        out.append(words[k])
        last = b
    out.append(text[last:])
    return "".join(out)


def shuffle_words(s: str, rng: random.Random) -> str:
    """Section 3, R1-5: the URL's words permuted among their own positions -- the host's
    among the host's, the path, query and fragment's among theirs. Same words, same
    separators, different order."""
    host, path, query, frag = SD.split_url(s)
    return _permute(host, rng) + _permute(SD.join_url("", path, query, frag), rng)


def d4_tokens(s: str) -> List[str]:
    """D4's tokens: letter runs, digit runs (each `0`), single other characters."""
    return ["0" if t[0].isdigit() else t for t in re.findall(r"[a-z]+|[0-9]+|[^a-z0-9]", s.lower())]


def d4_features(s: str) -> List[str]:
    toks = d4_tokens(s)
    return ["\x01".join(toks[i:i + n]) for n in (1, 2, 3) for i in range(len(toks) - n + 1)]


def d4_fit_predict(xtr, ytr, xte, seed):
    """D4: TF-IDF over 1-3-grams of D4's tokens, logistic regression (section 3, R1-5)."""
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    m = make_pipeline(TfidfVectorizer(analyzer=d4_features, sublinear_tf=True, max_features=50_000),
                      LogisticRegression(max_iter=2000, C=1.0))
    m.fit(xtr, ytr)
    return m.decision_function(xte)


def edit1(word: str) -> set:
    """Every a-z string one insertion, deletion or substitution away (QC1's edit rule)."""
    out = set()
    for i in range(len(word)):
        out.add(word[:i] + word[i + 1:])
        for c in _AZ:
            if c != word[i]:
                out.add(word[:i] + c + word[i + 1:])
    for i in range(len(word) + 1):
        for c in _AZ:
            out.add(word[:i] + c + word[i:])
    out.discard(word)
    return out


def anatomy(word: str, counts_of) -> str:
    """Section 3, R1-3, for a never-seen word, in order: near miss, join, other.
    `counts_of(list_of_words)` returns their training counts."""
    variants = sorted(edit1(word))
    if variants and int(np.max(counts_of(variants))) >= NEAR_MIN:
        return "near"
    for i in range(3, len(word) - 2):
        left, right = word[:i], word[i:]
        c = counts_of([left, right])
        if int(c[0]) >= JOIN_MIN and int(c[1]) >= JOIN_MIN:
            return "join"
    return "other"


def band_of(count: int, thresholds: Sequence[int]) -> str:
    if count <= 0:
        return "unseen"
    t1, t8, t32 = thresholds
    return "top1k" if count >= t1 else "1k-8k" if count >= t8 else "8k-32k" if count >= t32 else "beyond"


def shortfall(real: Dict[str, float], gen: Dict[str, float]) -> dict:
    """Section 3, R1-2: e, the split of the shortfall across the known bands, S_common, the reading."""
    e = gen["unseen"] - real["unseen"]
    d = {b: real[b] - gen[b] for b in BANDS[:-1]}
    if e < 0.01:
        return {"e": e, "d": d, "S_common": None, "reading": "NO SHORTFALL"}
    s = (d["top1k"] + d["1k-8k"]) / e
    return {"e": e, "d": d, "S_common": s,
            "reading": "COMMON-WORD SHORTFALL" if s >= 0.5 else "RARE-WORD SHORTFALL"}


def file_sha(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _full(path) -> List[str]:
    from uie.evaluation import FIELDS
    urls = json.loads(Path(path).read_text(encoding="utf-8"))
    return [x for x in (FIELDS["FULL"](u) for u in urls) if x]


def _write(path: Path, obj, indent=None) -> None:
    path.write_text(json.dumps(obj, indent=indent, ensure_ascii=False), encoding="utf-8")


def _read(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def _real_pool(pools_fn, label: str) -> Dict[int, List[str]]:
    """E18's own real selection per g: the gate's mixed validation pool, one URL per host,
    scheme stripped -- BEFORE E18's shuffle."""
    from uie.evaluation import FIELDS, _one_per_host
    pop = pools_fn(SD.BUNDLE, label)["mixed"]
    return {g: [x for x in (FIELDS["FULL"](u) for u in _one_per_host(pop, SD.seed_of(label, g))) if x]
            for g in range(G)}


def _train_rows(label: str):
    for stem in SD.TRAIN_STEMS[label]:
        with gzip.open(SD.BUNDLE / f"{stem}.jsonl.gz", "rt", encoding="utf-8") as fh:
            for line in fh:
                yield json.loads(line)["url"]


# --------------------------------------------------------------------------- #
# part data
# --------------------------------------------------------------------------- #

def _bpe_sample() -> dict:
    """Section 3, R1-4: 600,000 training rows drawn as training draws them -- the class by its
    label count (alpha = 1), the row uniform within the class -- with replacement. Returns the
    rest and host sequences with their weights, for all rows and for the first half."""
    from uie.evaluation import FIELDS
    from uie.factorization import parse
    sizes = {lab: sum(1 for _ in _train_rows(lab)) for lab in LABELS}
    rng = random.Random(BPE_SEED)
    p_benign = LABEL_COUNTS["benign"] / sum(LABEL_COUNTS.values())
    picks = {lab: Counter() for lab in LABELS}
    half = {lab: Counter() for lab in LABELS}
    for i in range(BPE_ROWS):
        lab = "benign" if rng.random() < p_benign else "phishing"
        j = rng.randrange(sizes[lab])
        picks[lab][j] += 1
        if i < BPE_ROWS // 2:
            half[lab][j] += 1
    out = {k: {"rest": Counter(), "host": Counter()} for k in ("full", "half")}
    for lab in LABELS:
        for j, url in enumerate(_train_rows(lab)):
            w_full, w_half = picks[lab].get(j, 0), half[lab].get(j, 0)
            if not w_full:
                continue
            parts = parse(url)
            rest = [FIELDS["PTH"](url), FIELDS["QRY"](url),
                    parts.fragment if parts.has_fragment else ""]
            host = [parts.registrable_label] + list(parts.subdomains)
            for key, w in (("full", w_full), ("half", w_half)):
                if not w:
                    continue
                for s in rest:
                    if s:
                        out[key]["rest"][s] += w
                for s in host:
                    if s:
                        out[key]["host"][s] += w
    return {"sizes": sizes, "p_benign": p_benign,
            "rows": {lab: sum(picks[lab].values()) for lab in LABELS},
            "full": {g: dict(c) for g, c in out["full"].items()},
            "half": {g: dict(c) for g, c in out["half"].items()}}


def part_data() -> None:
    from cr1_score import pools
    from uie.evaluation import FIELDS
    SAMPLES.mkdir(parents=True, exist_ok=True)
    meta: dict = {"N": {}, "counts": {}, "measurable": {}, "pool_check": {}, "shuffle_changed": {}}
    inputs = {}
    for label in LABELS:
        t0 = time.time()
        real = _real_pool(pools, label)
        files = {src: {g: _full(str(p).format(label=label, g=g)) for g in range(G)}
                 for src, p in SOURCES.items()}
        for src, p in list(SOURCES.items()) + [("e9_ln", E9_LN)]:
            for g in range(G):
                path = Path(str(p).format(label=label, g=g))
                inputs[str(path.relative_to(ROOT))] = file_sha(path)
        # R1-1/R1-2/R1-3/R1-4 samples, per field
        for field in FIELDS_R1:
            real_v = {g: [v for v in (FIELDS[field](s) for s in real[g]) if v] for g in range(G)}
            gen_v = {src: {g: [v for v in (FIELDS[field](s) for s in files[src][g]) if v]
                           for g in range(G)} for src in SOURCES}
            n = n_f([len(real_v[g]) for g in range(G)], [len(gen_v["e16_b5"][g]) for g in range(G)])
            key = f"{field}|{label}"
            meta["N"][key] = n
            meta["measurable"][key] = n >= N_MIN
            meta["counts"][key] = {"real": [len(real_v[g]) for g in range(G)],
                                   **{src: [len(gen_v[src][g]) for g in range(G)] for src in SOURCES}}
            if n < N_MIN:
                continue
            for g in range(G):
                pool = list(real_v[g])
                random.Random(seed_real(field, label, g)).shuffle(pool)
                node = {k: pool[i * n:(i + 1) * n] for i, k in enumerate("ABC")}
                for src in SOURCES:
                    node[src] = random.Random(seed_source(src, field, label, g)).sample(gen_v[src][g], n)
                node["meta"] = {"N": n, "field": field, "seed_real": seed_real(field, label, g)}
                _write(SAMPLES / f"F_{field}_{label}_g{g}.json", node)
        # R1-5: the shuffle control from E18's D block, and e15_new as R0 drew it
        for g in range(G):
            shuffled_pool = list(real[g])
            random.Random(SD.seed_of(label, g)).shuffle(shuffled_pool)
            e18 = _read(SD.SAMPLES / f"{label}_g{g}.json")
            meta["pool_check"][f"{label}|{g}"] = all(
                shuffled_pool[i * SD.N:(i + 1) * SD.N] == e18[k] for i, k in enumerate("ABC"))
            block_d = shuffled_pool[3 * SD.N:4 * SD.N]
            rng = random.Random(seed_shuffle(label, g))
            shuffled = [shuffle_words(s, rng) for s in block_d]
            meta["shuffle_changed"][f"{label}|{g}"] = sum(a != b for a, b in zip(block_d, shuffled)) / len(block_d)
            e15 = random.Random(R0.seed_source("e15_new", label, g, "H")).sample(files["e15_new"][g], SD.N)
            _write(SAMPLES / f"D4_{label}_g{g}.json", {"shuffle": shuffled, "e15_new": e15})
        print(f"  {label}: samples written ({time.time() - t0:.0f}s); N "
              + ", ".join(f"{f} {meta['N'][f'{f}|{label}']}" for f in FIELDS_R1), flush=True)
    for name in (BLOOM, FEED, SD.OUT / "vocab_benign.npz", SD.OUT / "vocab_phishing.npz"):
        inputs[str(Path(name).relative_to(ROOT))] = file_sha(name)
    for stem in sorted({s for v in SD.TRAIN_STEMS.values() for s in v}):
        path = SD.BUNDLE / f"{stem}.jsonl.gz"
        inputs[str(path.relative_to(ROOT))] = file_sha(path)
    meta["inputs"] = inputs
    t0 = time.time()
    sample = _bpe_sample()
    _write(OUT / "bpe_sample.json", sample)
    meta["bpe_sample"] = {"sizes": sample["sizes"], "rows": sample["rows"], "p_benign": sample["p_benign"],
                          "sequences": {k: {g: len(v) for g, v in sample[k].items()} for k in ("full", "half")}}
    print(f"  piece-learning sample: {sample['rows']} rows; "
          f"{meta['bpe_sample']['sequences']['full']} distinct sequences ({time.time() - t0:.0f}s)", flush=True)
    _write(OUT / "meta.json", meta, indent=1)


# --------------------------------------------------------------------------- #
# part bpe
# --------------------------------------------------------------------------- #

def part_bpe() -> None:
    sample = _read(OUT / "bpe_sample.json")
    record = {}
    for name in ("rest", "host"):
        t0 = time.time()
        print(f"  learning the {name} table: {len(sample['full'][name]):,} sequences", flush=True)
        table = BPE.learn(sample["full"][name], BPE_MERGES, log=lambda m: print(m, flush=True))
        used = "full"
        if table.meta["seconds"] > BPE_HOURS * 3600:     # section 3: over 3 h, the sample halves
            print(f"  over {BPE_HOURS} h: relearning on the first half of the sample", flush=True)
            table = BPE.learn(sample["half"][name], BPE_MERGES, log=lambda m: print(m, flush=True))
            used = "half"
        table.meta.update({"sample": used, "table": name})
        sha = table.save(OUT / f"bpe_{name}.json")
        record[name] = {"sha256": sha, "merges": len(table), "alphabet": len(table.alphabet),
                        "sample": used, "seconds": table.meta["seconds"]}
        print(f"  {name}: {len(table):,} merges, alphabet {len(table.alphabet)}, sha {sha[:12]}..., "
              f"{time.time() - t0:.0f}s", flush=True)
        del table
        gc.collect()
    _write(OUT / "bpe.json", record, indent=1)


# --------------------------------------------------------------------------- #
# part run: V-D1', D1 per field, D4 and its controls
# --------------------------------------------------------------------------- #

def _resume(path: Path) -> dict:
    return _read(path) if path.exists() else {}


def _task(res: dict, dest: Path, key: str, fp, a: list, b: list, fold_seed: int, t_all: float) -> None:
    if key in res:
        return
    t0 = time.time()
    texts, y = a + b, np.array([0] * len(a) + [1] * len(b))
    res[key] = SD.oof_auc(fp, texts, y, fold_seed)
    dest.write_text(json.dumps(res, indent=1), encoding="utf-8")
    print(f"  {dest.stem:<4} {key:<34} done  ({time.time() - t0:.0f}s; "
          f"{(time.time() - t_all) / 60:.0f} min so far)", flush=True)
    gc.collect()


def part_run() -> None:
    t_all = time.time()
    meta = _read(OUT / "meta.json")
    # V-D1': E18's D1 on its cached benign g0 floor and e16_b5 tasks
    dest = OUT / "VD1.json"
    res = _resume(dest)
    fp = SD.make_d1("benign")
    e18 = _read(SD.SAMPLES / "benign_g0.json")
    _task(res, dest, "benign|0|floor", fp, e18["A"], e18["B"], SD.seed_of("benign", 0) * 10 + 0, t_all)
    _task(res, dest, "benign|0|e16_b5", fp, e18["C"], e18["e16_b5"], SD.seed_of("benign", 0) * 10 + 1, t_all)
    del fp
    # D1 per field
    dest = OUT / "D1F.json"
    res = _resume(dest)
    for field in FIELDS_R1:
        for label in LABELS:
            if not meta["measurable"][f"{field}|{label}"]:
                continue
            fp = SD.make_d1(label)
            for g in range(G):
                node = _read(SAMPLES / f"F_{field}_{label}_g{g}.json")
                s = {k: [d1_string(field, v) for v in node[k]] for k in ("A", "B", "C") + tuple(SOURCES)}
                base = SD.seed_of(label, g) * 10
                _task(res, dest, f"{field}|{label}|{g}|floor", fp, s["A"], s["B"], base + TASK_INDEX["floor"], t_all)
                for src in SOURCES:
                    _task(res, dest, f"{field}|{label}|{g}|{src}", fp, s["C"], s[src],
                          base + TASK_INDEX[src], t_all)
            del fp
            gc.collect()
    # D4 with its controls, and D1 on the shuffled URLs
    d4, d1s = OUT / "D4.json", OUT / "D1S.json"
    res4, res1 = _resume(d4), _resume(d1s)
    for label in LABELS:
        fp1 = SD.make_d1(label)
        for g in range(G):
            node = {**_read(SD.SAMPLES / f"{label}_g{g}.json"), **_read(SAMPLES / f"D4_{label}_g{g}.json")}
            base = SD.seed_of(label, g) * 10
            for task in D4_TASKS:
                a, b = (node["A"], node["B"]) if task == "floor" else (node["C"], node[task])
                _task(res4, d4, f"{label}|{g}|{task}", d4_fit_predict, a, b, base + TASK_INDEX[task], t_all)
            _task(res1, d1s, f"{label}|{g}|shuffle", fp1, node["C"], node["shuffle"],
                  base + TASK_INDEX["shuffle"], t_all)
        del fp1
        gc.collect()
    print("run done", flush=True)


# --------------------------------------------------------------------------- #
# part words: R1-2 and R1-3
# --------------------------------------------------------------------------- #

def _thresholds(vocab) -> Tuple[int, int, int]:
    c = np.sort(vocab.c)[::-1]
    return tuple(int(c[min(r, len(c)) - 1]) for r in BAND_EDGES)


def part_words() -> None:
    meta = _read(OUT / "meta.json")
    out: dict = {"thresholds": {}, "fields": {}, "groups": {}}
    for label in LABELS:
        vocab = SD.Vocab(label)
        thr = _thresholds(vocab)
        out["thresholds"][label] = dict(zip(BAND_EDGES, thr))
        kinds: Dict[str, str] = {}

        def counts_of(words):
            return vocab.counts(list(words))

        group_bands: Dict[str, Dict[str, List[Counter]]] = defaultdict(lambda: defaultdict(list))
        for field in FIELDS_R1:
            if not meta["measurable"][f"{field}|{label}"]:
                continue
            per_g = {src: {"bands": [], "anat": [], "n_urls": []} for src in ("C",) + tuple(SOURCES)}
            for g in range(G):
                node = _read(SAMPLES / f"F_{field}_{label}_g{g}.json")
                for src in per_g:
                    words = [w for v in node[src] for w in WORD.findall(v.lower())]
                    cnt = counts_of(words) if words else np.zeros(0, dtype=np.int64)
                    bands = Counter(band_of(int(c), thr) for c in cnt)
                    anat = Counter()
                    for w, c in zip(words, cnt):
                        if int(c) == 0:
                            if w not in kinds:
                                kinds[w] = anatomy(w, counts_of)
                            anat[kinds[w]] += 1
                    per_g[src]["bands"].append(bands)
                    per_g[src]["anat"].append(anat)
                    per_g[src]["n_urls"].append(len(node[src]))
                    group_bands[GROUP[field]][src].append(bands)
            node_out = {}
            for src, d in per_g.items():
                shares = [{b: bc[b] / max(sum(bc.values()), 1) for b in BANDS} for bc in d["bands"]]
                unseen = [sum(a.values()) for a in d["anat"]]
                node_out["real" if src == "C" else src] = {
                    "bands": {b: st.mean(s[b] for s in shares) for b in BANDS},
                    "words_per_url": st.mean(sum(bc.values()) / n for bc, n in zip(d["bands"], d["n_urls"])),
                    "anatomy_share": {k: st.mean(a[k] / max(u, 1) for a, u in zip(d["anat"], unseen))
                                      for k in ("near", "join", "other")},
                    "anatomy_per_url": {k: st.mean(a[k] / n for a, n in zip(d["anat"], d["n_urls"]))
                                        for k in ("near", "join", "other")},
                }
            for src in SOURCES:
                node_out[f"{src} shortfall"] = shortfall(node_out["real"]["bands"], node_out[src]["bands"])
                node_out[f"{src} excess_per_url"] = {
                    k: node_out[src]["anatomy_per_url"][k] - node_out["real"]["anatomy_per_url"][k]
                    for k in ("near", "join", "other")}
            out["fields"][f"{field}|{label}"] = node_out
        for grp, d in group_bands.items():
            agg = {}
            for src, lst in d.items():
                tot = Counter()
                for bc in lst:
                    tot.update(bc)
                agg["real" if src == "C" else src] = {b: tot[b] / max(sum(tot.values()), 1) for b in BANDS}
            out["groups"][f"{grp}|{label}"] = {"bands": agg, **{f"{src} shortfall": shortfall(agg["real"], agg[src])
                                                               for src in SOURCES}}
        print(f"  {label}: words done ({len(kinds):,} distinct never-seen words classed)", flush=True)
        del vocab
        gc.collect()
    _write(OUT / "words.json", out, indent=1)


# --------------------------------------------------------------------------- #
# part cover: R1-4
# --------------------------------------------------------------------------- #

def part_cover() -> None:
    meta = _read(OUT / "meta.json")
    tables = {name: BPE.BPETable.load(OUT / f"bpe_{name}.json") for name in ("rest", "host")}
    out: dict = {"tables": {n: t.sha256() for n, t in tables.items()}, "fields": {}, "groups": {},
                 "lossless_failures": 0}
    for label in LABELS:
        vocab = SD.Vocab(label)
        thr = _thresholds(vocab)
        group_acc = defaultdict(lambda: {"cov": Counter(), "words": 0, "band_cov": defaultdict(Counter),
                                         "band_words": Counter(), "chars": 0, "pieces": Counter()})
        for field in FIELDS_R1:
            if not meta["measurable"][f"{field}|{label}"]:
                continue
            table = tables[GROUP[field]]
            acc = {"cov": Counter(), "words": 0, "band_cov": defaultdict(Counter), "band_words": Counter(),
                   "chars": 0, "pieces": Counter()}
            for g in range(G):
                for value in _read(SAMPLES / f"F_{field}_{label}_g{g}.json")["C"]:
                    for seg in (value.split(".") if field == "SUB" else [value]):
                        if not seg:
                            continue
                        low = seg.lower()
                        if len(low) != len(seg):
                            continue
                        spans = [(m.start(), m.end()) for m in WORD.finditer(low)]
                        words = [low[a:b] for a, b in spans]
                        bands = [band_of(int(c), thr) for c in vocab.counts(words)] if words else []
                        enc = table.encode_at(seg, TABLE_SIZES)
                        acc["chars"] += len(seg)
                        for k, pieces in enc.items():
                            if "".join(pieces) != seg:
                                out["lossless_failures"] += 1
                            acc["pieces"][k] += len(pieces)
                            if spans:
                                cov = BPE.covered_words(seg, pieces, spans)
                                acc["cov"][k] += sum(cov)
                                for bnd, ok in zip(bands, cov):
                                    acc["band_cov"][bnd][k] += int(ok)
                        acc["words"] += len(spans)
                        acc["band_words"].update(bands)
            out["fields"][f"{field}|{label}"] = _cover_summary(acc)
            ga = group_acc[GROUP[field]]
            ga["cov"].update(acc["cov"])
            ga["words"] += acc["words"]
            ga["chars"] += acc["chars"]
            ga["pieces"].update(acc["pieces"])
            ga["band_words"].update(acc["band_words"])
            for bnd, c in acc["band_cov"].items():
                ga["band_cov"][bnd].update(c)
        for grp, ga in group_acc.items():
            out["groups"][f"{grp}|{label}"] = _cover_summary(ga)
        print(f"  {label}: coverage done", flush=True)
    _write(OUT / "cover.json", out, indent=1)


def _cover_summary(acc: dict) -> dict:
    return {"words": acc["words"],
            "coverage": {k: acc["cov"][k] / max(acc["words"], 1) for k in TABLE_SIZES},
            "chars_per_piece": {k: acc["chars"] / max(acc["pieces"][k], 1) for k in TABLE_SIZES},
            "by_band": {b: {"words": acc["band_words"][b],
                            "coverage": {k: acc["band_cov"][b][k] / max(acc["band_words"][b], 1)
                                         for k in TABLE_SIZES}} for b in BANDS}}


# --------------------------------------------------------------------------- #
# part base: R1-6
# --------------------------------------------------------------------------- #

def _h64(s: str) -> int:
    return SD.h64(s)


def _field_values(url: str, parse) -> Dict[str, str]:
    from uie.evaluation import FIELDS
    parts = parse(url)
    return {"LBL": FIELDS["LBL"](url), "SUB": FIELDS["SUB"](url), "PTH": FIELDS["PTH"](url),
            "QRY": FIELDS["QRY"](url), "RD": (parts.registrable_domain or "").lower()}


def _training_tables(label: str, parse) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
    """Per field, the class's training values as sorted 64-bit hashes with their counts,
    merged in chunks (E18's `_merge`) so memory stays flat."""
    acc = {f: (np.empty(0, np.uint64), np.empty(0, np.int64)) for f in ("LBL", "SUB", "PTH", "QRY", "RD")}
    buf: Dict[str, List[int]] = defaultdict(list)
    for url in _train_rows(label):
        for f, v in _field_values(url, parse).items():
            if v:
                buf[f].append(_h64(v))
                if len(buf[f]) >= 1_000_000:
                    acc[f] = SD._merge(*acc[f], buf[f])
                    buf[f] = []
    for f, lst in buf.items():
        if lst:
            acc[f] = SD._merge(*acc[f], lst)
    return acc


def _lookup(table: Tuple[np.ndarray, np.ndarray], values: Sequence[str]) -> np.ndarray:
    u, c = table
    if not values:
        return np.zeros(0, dtype=np.int64)
    q = np.array([_h64(v) for v in values], dtype=np.uint64)
    i = np.clip(np.searchsorted(u, q), 0, len(u) - 1)
    return np.where(u[i] == q, c[i], 0)


def _frequent(label: str, parse, tables) -> Dict[str, List[str]]:
    """Per field, the training values seen at least 5 times (QC1-4's frequent set)."""
    hot = {f: set(tables[f][0][tables[f][1] >= FIELD_NEAR_MIN].tolist())
           for f in ("LBL", "SUB", "PTH", "QRY")}
    keep: Dict[str, set] = defaultdict(set)
    for url in _train_rows(label):
        for f, v in _field_values(url, parse).items():
            if v and f in hot and v not in keep[f] and _h64(v) in hot[f]:
                keep[f].add(v)
    return {f: sorted(s) for f, s in keep.items()}


def _copy_profile(values: List[str], table, index) -> dict:
    n = len(values)
    if not n:
        return {"n": 0}
    cnt = _lookup(table, values)
    unseen = [v for v, c in zip(values, cnt) if int(c) == 0]
    out = {"n": n, "copy": float(np.mean(cnt > 0)), "copy_once": float(np.mean(cnt == 1)),
           "duplicate": 1 - len(set(values)) / n}
    if index is not None:
        out["near_of_unseen"] = (sum(index.near(v) for v in unseen) / len(unseen)) if unseen else 0.0
    return out


def _distinct_per_1000(words: List[str], n: int, seed: int) -> float:
    pick = random.Random(seed).sample(words, n)
    return 1000 * len(set(pick)) / n


def part_base() -> None:
    from cr1_score import pools
    from qc1_common import NearIndex
    from uie.bloom import BloomFilter
    from uie.factorization import parse
    from uie.gate import Gate
    bloom = BloomFilter.load(BLOOM)
    feed = [u.strip() for u in FEED.read_text(encoding="utf-8").splitlines() if u.strip()]
    blocked = {(parse(u).registrable_domain or "").lower() for u in feed} - {""}
    out: dict = {"blocked_domains": len(blocked), "bloom_n_added": bloom.n_added, "classes": {}}
    for label in LABELS:
        t0 = time.time()
        tables = _training_tables(label, parse)
        frequent = _frequent(label, parse, tables)
        print(f"  {label}: training tables ({time.time() - t0:.0f}s); frequent "
              + ", ".join(f"{f} {len(v):,}" for f, v in frequent.items()), flush=True)
        real = _real_pool(pools, label)
        cls: dict = {"fields": {}, "gate": {}, "diversity": {}}
        sources = {"real": {g: real[g][:8_000] for g in range(G)}}
        sources.update({src: {g: _full(str(p).format(label=label, g=g)) for g in range(G)}
                        for src, p in SOURCES.items()})
        vals = {src: {g: [_field_values(u, parse) for u in urls] for g, urls in d.items()}
                for src, d in sources.items()}
        for f in ("LBL", "SUB", "PTH", "QRY", "RD"):
            index = NearIndex(frequent[f]) if f in frequent else None
            node = {}
            for src in sources:
                prof = [_copy_profile([v[f] for v in vals[src][g] if v[f]], tables[f], index) for g in range(G)]
                node[src] = {k: st.mean(p[k] for p in prof) for k in prof[0] if k != "n"}
                node[src]["n"] = st.mean(p["n"] for p in prof)
            cls["fields"][f] = node
            del index
            gc.collect()
        gate = Gate(known_membership=bloom, blocked_domains=blocked)
        for src, d in sources.items():
            by = Counter()
            n = 0
            for g in range(G):
                _, rep = gate.filter(d[g])
                by.update(rep.by_reason)
                n += rep.n_total
            cls["gate"][src] = {"n": n, **{str(k): v / n for k, v in sorted(by.items())}}
        for grp, fields in (("host", ("LBL", "SUB")), ("rest", ("PTH", "QRY"))):
            words = {src: [w for g in range(G) for v in vals[src][g] for f in fields
                           for w in WORD.findall(v[f].lower())] for src in sources}
            n = min(DIV_WORDS, min(len(w) for w in words.values()) // 2)
            seed = R1_SEED + 70_000 + 100 * LABELS.index(label)
            node = {"n_words": n}
            for src, w in words.items():
                node[src] = _distinct_per_1000(w, n, seed)
            pick = random.Random(seed + 1).sample(words["real"], 2 * n)
            node["real_vs_real"] = [1000 * len(set(pick[:n])) / n, 1000 * len(set(pick[n:])) / n]
            cls["diversity"][grp] = node
        out["classes"][label] = cls
        print(f"  {label}: base done ({time.time() - t0:.0f}s)", flush=True)
        del tables, frequent, vals
        gc.collect()
    _write(OUT / "base.json", out, indent=1)


# --------------------------------------------------------------------------- #
# verify and read
# --------------------------------------------------------------------------- #

def _verdicts(res: dict, prefix: str, tasks: Sequence[str]) -> dict:
    fl = [res[f"{prefix}|{g}|floor"] for g in range(G)]
    node = {"floor": {"mean_auc": st.mean(fl), "flag": abs(st.mean(fl) - 0.5) > 0.03}}
    xs = {}
    for t in tasks:
        xs[t] = [res[f"{prefix}|{g}|{t}"] - res[f"{prefix}|{g}|floor"] for g in range(G)]
        v = SD.verdict(xs[t])
        v["clear_line"] = SD.BAR - SD.TCRIT * v["se"]
        node[t] = v
    if "e15_new" in xs and "e16_b5" in xs:
        node["e15_new - e16_b5"] = SD.paired_t(xs["e15_new"], xs["e16_b5"])
    return node


def part_verify() -> None:
    meta = _read(OUT / "meta.json")
    e18 = _read(SD.OUT / "D1.json")
    vd1 = _read(OUT / "VD1.json")
    checks = {"V-D1'": all(vd1[k] == e18[k] for k in vd1) and len(vd1) == 2,
              "V-pool (E18's real selection rebuilt)": all(meta["pool_check"].values()),
              "V-inputs (hashed)": bool(meta.get("inputs"))}
    d1f, d4 = _read(OUT / "D1F.json"), _read(OUT / "D4.json")
    floors = {}
    for field in FIELDS_R1:
        for label in LABELS:
            if meta["measurable"][f"{field}|{label}"]:
                floors[f"D1 {field} {label}"] = _verdicts(d1f, f"{field}|{label}", SOURCES)["floor"]
    for label in LABELS:
        floors[f"D4 {label}"] = _verdicts(d4, label, D4_TASKS[1:])["floor"]
    checks["V-floor"] = {k: not v["flag"] for k, v in floors.items()}
    d4v = {label: _verdicts(d4, label, D4_TASKS[1:]) for label in LABELS}
    checks["V-D4 (convicts e9_ln in both classes)"] = all(d4v[l]["e9_ln"]["verdict"] == "CONVICTED" for l in LABELS)
    cover = _read(OUT / "cover.json")
    bpe = _read(OUT / "bpe.json")
    checks["V-BPE (lossless; tables as recorded)"] = (cover["lossless_failures"] == 0 and all(
        cover["tables"][n] == bpe[n]["sha256"] for n in ("rest", "host")))
    passed = all(v if isinstance(v, bool) else all(v.values()) for v in checks.values())
    _write(OUT / "verify.json", {"checks": checks, "floors": floors, "pass": passed}, indent=1)
    for k, v in checks.items():
        print(f"  {k}: {v if isinstance(v, bool) else ('all ok' if all(v.values()) else v)}")
    print(f"VERIFY: {'PASS' if passed else 'FAIL'}")


def part_read() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    ver = _read(OUT / "verify.json")
    if not ver["pass"]:
        raise SystemExit("STOP: R1's validity checks did not pass (section 4); nothing is read")
    meta = _read(OUT / "meta.json")
    d1f, d4, d1s = _read(OUT / "D1F.json"), _read(OUT / "D4.json"), _read(OUT / "D1S.json")
    words, cover, base = _read(OUT / "words.json"), _read(OUT / "cover.json"), _read(OUT / "base.json")
    e18 = _read(SD.OUT / "D1.json")
    out: dict = {"registration": "research/v2/PREREG_R1.md", "N": meta["N"], "measurable": meta["measurable"],
                 "R1-1": {}, "R1-5": {}, "decisions": {}}
    for field in FIELDS_R1:
        for label in LABELS:
            key = f"{field}|{label}"
            out["R1-1"][key] = _verdicts(d1f, key, SOURCES) if meta["measurable"][key] else "NOT MEASURABLE"
    for label in LABELS:
        node = _verdicts(d4, label, D4_TASKS[1:])
        xs = [d1s[f"{label}|{g}|shuffle"] - e18[f"{label}|{g}|floor"] for g in range(G)]
        node["D1 on shuffle"] = SD.verdict(xs)
        node["shuffle_changed"] = st.mean(meta["shuffle_changed"][f"{label}|{g}"] for g in range(G))
        out["R1-5"][label] = node
    # DR1: a host field is excluded only when it is measurable in both classes and convicted in neither
    dr1 = {}
    for field in ("LBL", "SUB"):
        rows = [out["R1-1"][f"{field}|{label}"] for label in LABELS]
        shown_clean = all(r != "NOT MEASURABLE" for r in rows) and not any(
            r["e16_b5"]["verdict"] == "CONVICTED" for r in rows if r != "NOT MEASURABLE")
        dr1[field] = not shown_clean
    out["decisions"]["DR1"] = {"arm A": ["PTH", "QRY", "FRG"], "arm B adds": [f for f, ok in dr1.items() if ok],
                               "arm B kept": any(dr1.values())}
    # DR2 and DR3, per table (field group)
    for grp in ("rest", "host"):
        cov = {label: cover["groups"][f"{grp}|{label}"]["coverage"] for label in LABELS
               if f"{grp}|{label}" in cover["groups"]}
        chosen = None
        for k in (8_000, 16_000, 32_000):
            if all(c[str(32_000)] - c[str(k)] <= 0.05 for c in cov.values()):
                chosen = k
                break
        rare = all(words["groups"].get(f"{grp}|{label}", {}).get("e16_b5 shortfall", {}).get("reading")
                   == "RARE-WORD SHORTFALL" for label in LABELS)
        low = all(c[str(32_000)] < 0.5 for c in cov.values())
        out["decisions"][f"DR2 {grp}"] = {"merges": chosen, "parameters": chosen * 512 if chosen else None,
                                          "coverage": cov}
        out["decisions"][f"DR3 {grp}"] = "NO-GO" if (rare and low) else "GO"
    v4 = out["R1-5"]
    out["decisions"]["DR4"] = {
        "valid": all(v4[l]["e9_ln"]["verdict"] == "CONVICTED" and not v4[l]["floor"]["flag"] for l in LABELS),
        "order_sensitive": {l: v4[l]["shuffle"]["verdict"] == "CONVICTED" for l in LABELS}}
    out["words"], out["cover"], out["base"] = words, cover, base
    _write(OUT / "r1_result.json", out, indent=1)
    _print(out, meta)


def _print(out: dict, meta: dict) -> None:
    print("R1-1  D1 per field (excess over the floor; mean of 5 generations)")
    print(f"{'field':<5} {'class':<9} {'N':>5} {'floor':>7} {'e16_b5':>8} {'SE':>7} {'t':>7} {'verdict':<10} "
          f"{'e15_new':>8} {'e15-e16 (t)':>16}")
    for key, r in out["R1-1"].items():
        field, label = key.split("|")
        if r == "NOT MEASURABLE":
            print(f"{field:<5} {label:<9} {meta['N'][key]:>5}  NOT MEASURABLE")
            continue
        v, w, d = r["e16_b5"], r["e15_new"], r["e15_new - e16_b5"]
        print(f"{field:<5} {label:<9} {meta['N'][key]:>5} {r['floor']['mean_auc']:>7.4f} {v['mean']:>+8.4f} "
              f"{v['se']:>7.4f} {v['t']:>7.2f} {v['verdict']:<10} {w['mean']:>+8.4f} "
              f"{d['mean_diff']:>+8.4f} ({d['t']:.2f})")
    print("\nR1-5  D4 (word sequences) and D1 on shuffled real URLs")
    for label, r in out["R1-5"].items():
        print(f"  {label}: floor {r['floor']['mean_auc']:.4f}; shuffled share changed {r['shuffle_changed']:.1%}")
        for t in D4_TASKS[1:]:
            print(f"    D4 {t:<8} {r[t]['mean']:>+8.4f} (SE {r[t]['se']:.4f}, t {r[t]['t']:.2f}) {r[t]['verdict']}")
        s = r["D1 on shuffle"]
        print(f"    D1 shuffle   {s['mean']:>+8.4f} (SE {s['se']:.4f}, t {s['t']:.2f}) {s['verdict']}")
    print("\nR1-2  words by training-frequency band (token shares), and the shortfall")
    for key, node in out["words"]["fields"].items():
        print(f"  {key}: " + "; ".join(
            f"{src} " + " ".join(f"{b} {node[src]['bands'][b]:.3f}" for b in BANDS)
            for src in ("real", "e16_b5")))
        sf = node["e16_b5 shortfall"]
        print(f"    e16_b5: e {sf['e']:+.4f}; S_common {sf['S_common'] if sf['S_common'] is None else round(sf['S_common'], 3)}"
              f" -> {sf['reading']}")
    for key, node in out["words"]["groups"].items():
        sf = node["e16_b5 shortfall"]
        print(f"  GROUP {key}: e {sf['e']:+.4f}; S_common "
              f"{sf['S_common'] if sf['S_common'] is None else round(sf['S_common'], 3)} -> {sf['reading']}")
    print("\nR1-3  never-seen words: shares (near / join / other) and the generated excess per URL")
    for key, node in out["words"]["fields"].items():
        r, gsh, ex = node["real"]["anatomy_share"], node["e16_b5"]["anatomy_share"], node["e16_b5 excess_per_url"]
        print(f"  {key}: real {r['near']:.2f}/{r['join']:.2f}/{r['other']:.2f}  e16_b5 "
              f"{gsh['near']:.2f}/{gsh['join']:.2f}/{gsh['other']:.2f}  excess per URL "
              f"near {ex['near']:+.3f} join {ex['join']:+.3f} other {ex['other']:+.3f}")
    print("\nR1-4  coverage of real words by single pieces (2k / 4k / 8k / 16k / 32k merges)")
    for key, node in list(out["cover"]["groups"].items()) + list(out["cover"]["fields"].items()):
        print(f"  {key:<14} " + " ".join(f"{node['coverage'][str(k)]:.3f}" for k in TABLE_SIZES)
              + f"   chars/piece at 8k {node['chars_per_piece'][str(8_000)]:.2f}")
    print("\nR1-6  reference values (mean over the 5 files)")
    for label, cls in out["base"]["classes"].items():
        for f, node in cls["fields"].items():
            print(f"  {label:<9} {f:<4} " + "  ".join(
                f"{src}: copy {v.get('copy', 0):.3f} once {v.get('copy_once', 0):.3f} dup {v.get('duplicate', 0):.3f}"
                + (f" near {v['near_of_unseen']:.3f}" if "near_of_unseen" in v else "")
                for src, v in node.items()))
        for src, v in cls["gate"].items():
            print(f"  {label:<9} gate {src}: " + ", ".join(f"{k} {x:.4f}" for k, x in v.items() if k != "n"))
        for grp, v in cls["diversity"].items():
            print(f"  {label:<9} distinct words per 1,000 ({grp}, n {v['n_words']}): "
                  + ", ".join(f"{s} {v[s]:.1f}" for s in ("real", "e16_b5", "e15_new"))
                  + f"; real vs real {v['real_vs_real'][0]:.1f} / {v['real_vs_real'][1]:.1f}")
    print("\nDECISIONS:", json.dumps(out["decisions"], indent=1, default=str))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--part", required=True,
                    choices=("data", "bpe", "run", "words", "cover", "base", "verify", "read"))
    args = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")
    OUT.mkdir(parents=True, exist_ok=True)
    {"data": part_data, "bpe": part_bpe, "run": part_run, "words": part_words, "cover": part_cover,
     "base": part_base, "verify": part_verify, "read": part_read}[args.part]()


if __name__ == "__main__":
    main()
