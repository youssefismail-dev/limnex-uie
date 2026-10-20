"""
research/v2/v2_r2.py

V2-R2: can a vocabulary memory beat the word tell without copying? (research/v2/PREREG_R2.md)
Laptop only: no training, no GPU. e16_b5's weights are read, never changed.

    python research/v2/v2_r2.py --part vocab              # the A/B split, Vocab_A, Vocab_B, r_f
    python research/v2/v2_r2.py --part oracle             # Part A's samples (O1, O2, O2e, O4)
    python research/v2/v2_r2.py --part runA               # Part A's fits (D1b, D4) and copying
    python research/v2/v2_r2.py --part readA              # Part A's reading and the rule for Part B
    python research/v2/v2_r2.py --part wrapcheck          # V-wrap: the wrapper off == e16_b5
    python research/v2/v2_r2.py --part gen --label benign # Part B: the memory on the frozen model
    python research/v2/v2_r2.py --part runB               # Part B's fits
    python research/v2/v2_r2.py --part metrics            # Part B's copying, creativity, structure
    python research/v2/v2_r2.py --part verify
    python research/v2/v2_r2.py --part read

THE HONESTY RULE (section 2): nothing that builds an oracle or the wrapper may read Vocab_B.
Those code paths receive Vocab_A only; Vocab_B is opened by `judge_vocab` alone.
"""
from __future__ import annotations

import argparse
import bisect
import gc
import hashlib
import json
import math
import random
import statistics as st
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
for p in (ROOT, ROOT / "research" / "e18", ROOT / "research" / "qc1", ROOT / "research" / "native",
          ROOT / "research" / "e13", ROOT / "research" / "e15", ROOT / "research" / "e16",
          ROOT / "research" / "e17", HERE):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import strong_disc as SD  # noqa: E402
import v2_r0 as R0        # noqa: E402
import v2_r1 as R1        # noqa: E402

OUT = ROOT / "data" / "v2" / "r2"
SAMPLES = OUT / "samples"
GEN = OUT / "gen"
LABELS, G = SD.LABELS, SD.G
R2_SEED = 20261040
WORD = SD.TOKEN
WORD_FIELDS = ("LBL", "SUB", "PTH", "QRY", "FRG")
ORACLE_INDEX = {"O1": 1, "O2": 2, "O2e": 3, "O4": 4}
TASK_INDEX = {"floor": 0, "base": 1, "O1": 6, "O2": 7, "O2e": 8, "O4": 9, "lx": 10}
RARE_RANK = 8_000
NEAR_MIN = 10
PART_A = {"rest": ("base", "O1", "O2", "O2e", "O4"), "host": ("base", "O1", "O2", "O2e"),
          "whole": ("base", "O1", "O2", "O2e")}
M_PER_G = 1_600                                     # Part B: URLs per generation and class
PENALTY = -1.0e4                                    # section 4: finite, so D1/B5 stay finite
E16_CAP = ROOT / "data" / "e16" / "train" / "e16_cap" / "final.pt"


# --------------------------------------------------------------------------- #
# the split and the vocabularies
# --------------------------------------------------------------------------- #

def half_of(domain: str) -> str:
    """Section 2: a registrable domain's half -- blake2b of the lower-cased domain, even -> A."""
    h = int.from_bytes(hashlib.blake2b(domain.lower().encode("utf-8"), digest_size=8).digest(), "little")
    return "A" if h % 2 == 0 else "B"


def field_texts(url: str, parts) -> Dict[str, str]:
    from uie.evaluation import FIELDS
    return {"LBL": parts.registrable_label or "", "SUB": ".".join(parts.subdomains),
            "PTH": FIELDS["PTH"](url), "QRY": FIELDS["QRY"](url),
            "FRG": parts.fragment if parts.has_fragment else ""}


def _vocab_path(label: str, half: str) -> Path:
    return OUT / f"vocab_{label}_{half}.json"


def memory_vocab(label: str) -> Dict[str, int]:
    """Vocab_A: the ONLY vocabulary an oracle or the wrapper may read."""
    return json.loads(_vocab_path(label, "A").read_text(encoding="utf-8"))


def judge_vocab(label: str) -> Dict[str, int]:
    """Vocab_B: the judge's vocabulary. Read by D1b alone."""
    return json.loads(_vocab_path(label, "B").read_text(encoding="utf-8"))


def part_vocab() -> None:
    from uie.evaluation import _strip_scheme
    from uie.factorization import parse
    OUT.mkdir(parents=True, exist_ok=True)
    meta: dict = {}
    for label in LABELS:
        t0 = time.time()
        vocab = {"A": Counter(), "B": Counter()}
        fields_b = {f: Counter() for f in WORD_FIELDS}
        rows = Counter()
        rest_a: List[str] = []
        rest_hashes: List[int] = []
        pair_hashes: List[int] = []
        for url in R1._train_rows(label):
            parts = parse(url)
            half = half_of(parts.registrable_domain or parts.host_raw or "")
            rows[half] += 1
            vocab[half].update(WORD.findall(_strip_scheme(url).lower()))
            texts = field_texts(url, parts)
            if half == "B":
                for f, text in texts.items():
                    if text:
                        fields_b[f].update(WORD.findall(text.lower()))
            full = _strip_scheme(url)
            _, path, query, frag = SD.split_url(full)
            rest = SD.join_url("", path, query, frag)
            if R0.rest_bearing(full):
                rest_hashes.append(SD.h64(rest))
                if half == "A" and len(rest_a) < 200_000:
                    rest_a.append(SD.join_url(R0.MASK, path, query, frag))
            pw = WORD.findall(texts["PTH"].lower())
            pair_hashes.extend(SD.h64(a + " " + b) for a, b in zip(pw, pw[1:]))
        for half in ("A", "B"):
            _vocab_path(label, half).write_text(json.dumps(dict(vocab[half])), encoding="utf-8")
        np.save(OUT / f"train_rest_{label}.npy", np.unique(np.array(rest_hashes, dtype=np.uint64)))
        np.save(OUT / f"train_pairs_{label}.npy", np.unique(np.array(pair_hashes, dtype=np.uint64)))
        (OUT / f"rest_a_{label}.json").write_text(json.dumps(rest_a), encoding="utf-8")
        a = vocab["A"]
        r = {f: (sum(c for w, c in cnt.items() if w not in a) / max(sum(cnt.values()), 1))
             for f, cnt in fields_b.items()}
        meta[label] = {"rows": dict(rows), "types": {h: len(v) for h, v in vocab.items()},
                       "tokens": {h: sum(v.values()) for h, v in vocab.items()}, "r": r,
                       "rest_a_pool": len(rest_a), "train_rest_distinct": int(len(set(rest_hashes))),
                       "seconds": round(time.time() - t0)}
        print(f"  {label}: rows {dict(rows)}; types A {len(a):,} B {len(vocab['B']):,}; r "
              + ", ".join(f"{f} {v:.3f}" for f, v in r.items()) + f" ({time.time() - t0:.0f}s)", flush=True)
        del vocab, fields_b, rest_hashes, pair_hashes
        gc.collect()
    (OUT / "vocab.json").write_text(json.dumps(meta, indent=1), encoding="utf-8")


class DictVocab:
    """E18's `Vocab` interface over a word -> count dict."""

    def __init__(self, counts: Dict[str, int]) -> None:
        self.d = counts

    def counts(self, toks: Sequence[str]) -> np.ndarray:
        return np.array([self.d.get(t, 0) for t in toks], dtype=np.int64)


def make_d1b(label: str):
    """D1b: E18's D1, unchanged, with the judge's vocabulary (Vocab_B)."""
    vocab = DictVocab(judge_vocab(label))
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


# --------------------------------------------------------------------------- #
# the memory (Vocab_A only)
# --------------------------------------------------------------------------- #

class Memory:
    """What an oracle or the wrapper may know: Vocab_A, its frequent words, its rare words."""

    def __init__(self, vocab_a: Dict[str, int]) -> None:
        self.a = vocab_a
        self.frequent = {w for w, c in vocab_a.items() if c >= NEAR_MIN}
        ranked = sorted(vocab_a.items(), key=lambda kv: (-kv[1], kv[0]))
        rare = ranked[RARE_RANK:]
        self.rare_words = [w for w, _ in rare]
        self.rare_cum = list(np.cumsum([c for _, c in rare], dtype=np.float64))
        self.sorted_words = sorted(vocab_a)

    def known(self, w: str) -> bool:
        return w in self.a

    def nearest_frequent(self, w: str) -> Optional[str]:
        best, best_c = None, -1
        for v in R1.edit1(w):
            if v in self.frequent:
                c = self.a[v]
                if c > best_c or (c == best_c and v < best):
                    best, best_c = v, c
        return best

    def draw_rare(self, rng: random.Random) -> str:
        x = rng.random() * self.rare_cum[-1]
        return self.rare_words[min(bisect.bisect_right(self.rare_cum, x), len(self.rare_words) - 1)]

    def is_prefix(self, p: str) -> bool:
        i = bisect.bisect_left(self.sorted_words, p)
        return i < len(self.sorted_words) and self.sorted_words[i].startswith(p)


def recase(original: str, new: str) -> str:
    if original.isupper():
        return new.upper()
    if original[:1].isupper() and original[1:].islower():
        return new.capitalize()
    return new


def _host_fields(host: str) -> List[Tuple[int, int, str]]:
    """(start, end, field) for each dot-label of a host: SUB, LBL, or SUF."""
    from uie.factorization import parse
    parts = parse(host)
    labels = host.split(".")
    n_sub = len(parts.subdomains)
    out, pos = [], 0
    for i, lab in enumerate(labels):
        f = "SUB" if i < n_sub else "LBL" if i == n_sub else "SUF"
        out.append((pos, pos + len(lab), f))
        pos += len(lab) + 1
    return out


def _rest_fields(rest: str) -> List[Tuple[int, int, str]]:
    """(start, end, field) for the path, query and fragment of a rest string ("/p?q#f")."""
    out, pos = [], 0
    q, h = rest.find("?"), rest.find("#")
    if h >= 0 and (q < 0 or h < q):
        q = -1
    ends = [x for x in (q, h) if x >= 0] + [len(rest)]
    first = min(ends)
    out.append((0, first, "PTH"))
    if q >= 0:
        qe = h if h > q else len(rest)
        out.append((q, qe, "QRY"))
    if h >= 0:
        out.append((h, len(rest), "FRG"))
    return out


def transform(text: str, spans_fields: List[Tuple[int, int, str]], kind: str, mem: Memory,
              keep: Dict[str, float], rng: random.Random, stats: Counter) -> str:
    """Apply an oracle to the words of `text`. A fake is a word not in Vocab_A."""
    low = text.lower()
    if len(low) != len(text):
        return text
    out, last = [], 0
    for m in WORD.finditer(low):
        w = m.group(0)
        f = next((fld for a, b, fld in spans_fields if a <= m.start() < b), "PTH")
        new = None
        if f != "SUF" and not mem.known(w):
            stats["fakes"] += 1
            if kind == "O1":
                new = mem.nearest_frequent(w)
            elif kind == "O2":
                new = mem.draw_rare(rng)
            elif kind == "O2e":
                if rng.random() < keep.get(f, 1.0):
                    stats["kept"] += 1
                else:
                    new = mem.draw_rare(rng)
            if new is not None:
                stats["replaced"] += 1
        out.append(text[last:m.start()])
        out.append(recase(text[m.start():m.end()], new) if new else text[m.start():m.end()])
        last = m.end()
    out.append(text[last:])
    return "".join(out)


def transform_url(s: str, part: str, kind: str, mem: Memory, keep: Dict[str, float],
                  rng: random.Random, stats: Counter) -> str:
    """An oracle on one sample string: `rest` (h.h + rest), `host`, or `whole`."""
    host, path, query, frag = SD.split_url(s)
    rest = SD.join_url("", path, query, frag)
    if part == "host":
        return transform(s, _host_fields(s), kind, mem, keep, rng, stats)
    new_rest = transform(rest, _rest_fields(rest), kind, mem, keep, rng, stats)
    if part == "rest":
        return host + new_rest
    return transform(host, _host_fields(host), kind, mem, keep, rng, stats) + new_rest


def field_unseen(strings: Sequence[str], part: str, mem: Memory) -> Dict[str, Tuple[int, int]]:
    """Per field: (words not in Vocab_A, words) over sample strings -- u_f's counts."""
    acc = defaultdict(lambda: [0, 0])
    for s in strings:
        host, path, query, frag = SD.split_url(s)
        rest = SD.join_url("", path, query, frag)
        pieces = []
        if part in ("host", "whole"):
            pieces.append((host if part == "whole" else s, _host_fields(host if part == "whole" else s)))
        if part in ("rest", "whole"):
            pieces.append((rest, _rest_fields(rest)))
        for text, spans in pieces:
            low = text.lower()
            for m in WORD.finditer(low):
                f = next((fld for a, b, fld in spans if a <= m.start() < b), "PTH")
                if f == "SUF":
                    continue
                acc[f][1] += 1
                acc[f][0] += int(not mem.known(m.group(0)))
    return {f: (v[0], v[1]) for f, v in acc.items()}


def keep_rates(label: str, mem: Memory, e16_strings: Dict[str, List[str]]) -> Tuple[dict, dict, dict]:
    """e_f = min(1, r_f / u_f): r_f from training (vocab.json), u_f from e16_b5's own words."""
    r = json.loads((OUT / "vocab.json").read_text(encoding="utf-8"))[label]["r"]
    tot = defaultdict(lambda: [0, 0])
    for part, strings in e16_strings.items():
        for f, (k, n) in field_unseen(strings, part, mem).items():
            tot[f][0] += k
            tot[f][1] += n
    u = {f: v[0] / v[1] for f, v in tot.items() if v[1]}
    e = {f: min(1.0, r[f] / u[f]) if u.get(f) else 1.0 for f in WORD_FIELDS}
    return e, u, r


# --------------------------------------------------------------------------- #
# Part A
# --------------------------------------------------------------------------- #

def _source_samples(part: str, label: str, g: int) -> dict:
    if part == "rest":
        return R1._read(R0.SAMPLES / f"R_{label}_g{g}.json")
    if part == "host":
        return R1._read(R0.SAMPLES / f"H_{label}_g{g}.json")
    return R1._read(SD.SAMPLES / f"{label}_g{g}.json")


def part_oracle() -> None:
    """Part A's samples. Reads Vocab_A only (the honesty rule)."""
    SAMPLES.mkdir(parents=True, exist_ok=True)
    record: dict = {}
    for label in LABELS:
        mem = Memory(memory_vocab(label))
        e16 = {part: [s for g in range(G) for s in _source_samples(part, label, g)["e16_b5"]]
               for part in ("rest", "host")}
        keep, u, r = keep_rates(label, mem, e16)
        rest_a = R1._read(OUT / f"rest_a_{label}.json")
        record[label] = {"keep": keep, "u": u, "r": r, "stats": {}}
        for part in PART_A:
            for g in range(G):
                src = _source_samples(part, label, g)
                node = {k: src[k] for k in ("A", "B", "C")}
                node["base"] = src["e16_b5"]
                for kind in PART_A[part][1:]:
                    rng = random.Random(R2_SEED + 10_000 * ORACLE_INDEX[kind] + 100 * LABELS.index(label) + g)
                    stats = Counter()
                    if kind == "O4":
                        node[kind] = [rng.choice(rest_a) for _ in node["base"]]
                    else:
                        node[kind] = [transform_url(s, part, kind, mem, keep, rng, stats) for s in node["base"]]
                    record[label]["stats"][f"{part}|{g}|{kind}"] = dict(stats)
                R1._write(SAMPLES / f"A_{part}_{label}_g{g}.json", node)
        print(f"  {label}: oracles built; keep " + ", ".join(f"{f} {v:.3f}" for f, v in keep.items()), flush=True)
        del mem
        gc.collect()
    R1._write(OUT / "oracle.json", record, indent=1)


def _fit(res: dict, dest: Path, key: str, fp, a, b, fold_seed: int, t_all: float) -> None:
    R1._task(res, dest, key, fp, a, b, fold_seed, t_all)


def part_runA() -> None:
    t_all = time.time()
    dest, d4dest = OUT / "A_D1b.json", OUT / "A_D4.json"
    res, res4 = R1._resume(dest), R1._resume(d4dest)
    for label in LABELS:
        fp = make_d1b(label)
        for part, sources in PART_A.items():
            for g in range(G):
                node = R1._read(SAMPLES / f"A_{part}_{label}_g{g}.json")
                base = SD.seed_of(label, g) * 10
                _fit(res, dest, f"{part}|{label}|{g}|floor", fp, node["A"], node["B"], base, t_all)
                for s in sources:
                    _fit(res, dest, f"{part}|{label}|{g}|{s}", fp, node["C"], node[s], base + TASK_INDEX[s], t_all)
                if part == "rest":
                    _fit(res4, d4dest, f"rest|{label}|{g}|floor", R1.d4_fit_predict, node["A"], node["B"], base, t_all)
                    for s in ("base", "O2e", "O4"):
                        _fit(res4, d4dest, f"rest|{label}|{g}|{s}", R1.d4_fit_predict, node["C"], node[s],
                             base + TASK_INDEX[s], t_all)
        del fp
        gc.collect()
    # copying on the path part: exact training rest strings
    copy = {}
    for label in LABELS:
        train = np.load(OUT / f"train_rest_{label}.npy")
        for s in ("base", "O2e", "O4"):
            vals = []
            for g in range(G):
                node = R1._read(SAMPLES / f"A_rest_{label}_g{g}.json")
                rests = [SD.join_url("", *SD.split_url(x)[1:]) for x in node[s]]
                q = np.array([SD.h64(x) for x in rests], dtype=np.uint64)
                i = np.clip(np.searchsorted(train, q), 0, len(train) - 1)
                vals.append(float(np.mean(train[i] == q)))
            copy[f"{label}|{s}"] = st.mean(vals)
        reals = []
        for g in range(G):
            node = R1._read(SAMPLES / f"A_rest_{label}_g{g}.json")
            q = np.array([SD.h64(SD.join_url("", *SD.split_url(x)[1:])) for x in node["C"]], dtype=np.uint64)
            i = np.clip(np.searchsorted(train, q), 0, len(train) - 1)
            reals.append(float(np.mean(train[i] == q)))
        copy[f"{label}|real"] = st.mean(reals)
    R1._write(OUT / "A_copy.json", copy, indent=1)
    print("runA done", flush=True)


def part_readA() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    res, res4 = R1._read(OUT / "A_D1b.json"), R1._read(OUT / "A_D4.json")
    copy = R1._read(OUT / "A_copy.json")
    out: dict = {"D1b": {}, "D4": {}, "copy": copy}
    for part, sources in PART_A.items():
        for label in LABELS:
            node = R1._verdicts(res, f"{part}|{label}", sources)
            xb = node["base"]["mean"]
            for s in sources[1:]:
                node[s]["closed"] = (xb - node[s]["mean"]) / xb if xb > 0 else None
            out["D1b"][f"{part}|{label}"] = node
    for label in LABELS:
        out["D4"][label] = R1._verdicts(res4, f"rest|{label}", ("base", "O2e", "O4"))
    flagged = [k for k, v in list(out["D1b"].items()) + list(out["D4"].items()) if v["floor"]["flag"]]
    if flagged:
        raise SystemExit(f"STOP: V-floor fails for {flagged}; Part A is not read")
    runs = any((out["D1b"][f"{part}|{label}"]["O2e"]["closed"] or 0) >= 0.5
               for part in ("rest", "host") for label in LABELS)
    out["part_B_runs"] = runs
    R1._write(OUT / "A_result.json", out, indent=1)
    for key, node in out["D1b"].items():
        print(f"{key:<15} floor {node['floor']['mean_auc']:.4f}" + ("  FLAG" if node["floor"]["flag"] else ""))
        for s in [k for k in PART_A[key.split('|')[0]]]:
            v = node[s]
            c = f"  closed {v['closed']:+.0%}" if v.get("closed") is not None else ""
            print(f"    {s:<5} {v['mean']:>+8.4f} (SE {v['se']:.4f}, t {v['t']:>7.2f}) {v['verdict']:<10}{c}")
    for label, node in out["D4"].items():
        print(f"D4 rest {label}: floor {node['floor']['mean_auc']:.4f}; " + "; ".join(
            f"{s} {node[s]['mean']:+.4f} {node[s]['verdict']}" for s in ("base", "O2e", "O4")))
    print("copy (exact training rest strings):", json.dumps({k: round(v, 4) for k, v in copy.items()}))
    print("PART B RUNS:", runs)


# --------------------------------------------------------------------------- #
# Part B: the lexicon wrapper
# --------------------------------------------------------------------------- #

class LexiconWrapper:
    """e16_b5's frozen model with a word memory (PREREG_R2 section 4). Inside LBL, SUB, PTH,
    QRY and FRG, a word starts free with probability e_f (a hash of the row so far); any other
    word must stay inside Vocab_A and may end only as a complete Vocab_A word (or under 3
    letters). The model's own probabilities choose among what is allowed."""

    def __init__(self, model, tok, mem: Memory, keep: Dict[str, float]) -> None:
        import torch
        self.inner, self.cfg, self.mem, self.keep, self.tok = model, model.cfg, mem, keep, tok
        self.letter = {}
        for c in "abcdefghijklmnopqrstuvwxyz":
            ids = [tok.stoi[x] for x in (c, c.upper()) if x in tok.stoi]
            if ids:
                self.letter[c] = ids
        self.letter_ids = {i: c for c, ids in self.letter.items() for i in ids}
        self.nonletter = torch.tensor(sorted(set(range(tok.vocab_size)) - set(self.letter_ids)), dtype=torch.long)
        from uie.tokenizer import _MARKERS
        self.marker_of = {tok.marker_id(m): m for m in _MARKERS}
        self.rs = tok.stoi.get("<RS>")
        self.stats = Counter()
        self._torch = torch

    # torch.nn.Module surface that uie.model.generate uses
    def parameters(self):
        return self.inner.parameters()

    def eval(self):
        self.inner.eval()
        return self

    def __call__(self, ids):
        logits = self.inner(ids)
        last = logits[:, -1]
        for b, row in enumerate(ids.tolist()):
            bad = self.disallowed(row)
            if bad is not None:
                last[b, bad] = PENALTY
        return logits

    def _field(self, row: List[int]) -> Optional[str]:
        for t in reversed(row):
            if t == self.rs:
                return None
            if t in self.marker_of:
                return self.marker_of[t]
        return None

    def _word(self, row: List[int]) -> Tuple[str, int]:
        i = len(row)
        while i > 0 and row[i - 1] in self.letter_ids:
            i -= 1
        return "".join(self.letter_ids[t] for t in row[i:]), i

    def free(self, row: List[int], start: int, field: str) -> bool:
        h = hashlib.blake2b(np.array(row[:start], dtype=np.int32).tobytes(), digest_size=8).digest()
        return int.from_bytes(h, "little") / 2 ** 64 < self.keep.get(field, 1.0)

    def check_last(self, row: List[int]) -> None:
        """V-trie: the token just sampled kept a constrained word inside the memory."""
        if len(row) < 2:
            return
        field = self._field(row[:-1])
        if field not in WORD_FIELDS:
            return
        prev, start = self._word(row[:-1])
        if not prev or self.free(row, start, field):
            return
        t = row[-1]
        if t in self.letter_ids:
            if not self.mem.is_prefix(prev + self.letter_ids[t]):
                self.stats["violations"] += 1
        elif len(prev) >= 3 and not self.mem.known(prev):
            self.stats["violations"] += 1

    def disallowed(self, row: List[int]):
        self.check_last(row)
        field = self._field(row)
        if field not in WORD_FIELDS:
            return None
        prefix, start = self._word(row)
        if not prefix:
            return None
        free = self.free(row, start, field)
        if len(prefix) == 1:
            self.stats[f"words_{field}"] += 1
            self.stats[f"free_{field}"] += int(free)
        if free:
            return None
        bad = [i for c, ids in self.letter.items() if not self.mem.is_prefix(prefix + c) for i in ids]
        self.stats["constrained_steps"] += 1
        if len(prefix) >= 3 and not self.mem.known(prefix):
            return self._torch.cat([self._torch.tensor(bad, dtype=self._torch.long), self.nonletter]) if bad \
                else self.nonletter
        return self._torch.tensor(bad, dtype=self._torch.long) if bad else None


def _engine():
    import torch
    from uie.suffixes import SuffixTable
    from uie.train import load_checkpoint
    from e8_verify import TABLES
    from e13_verify import PS
    model, tok, _ = load_checkpoint(E16_CAP, device=torch.device("cpu"))
    model.eval()
    shipped = SuffixTable.load(ROOT / TABLES["suffix_table.json"][0])
    ps = json.loads((ROOT / PS[0]).read_text(encoding="utf-8"))["table"]
    return model, tok, shipped, ps


def part_wrapcheck() -> None:
    """V-wrap: with every e_f = 1 the wrapper never constrains, so generation is identical."""
    import e17_common as E17
    model, tok, shipped, ps = _engine()
    out = {}
    for label in LABELS:
        mem = Memory(memory_vocab(label))
        plain = E17.free_urls(model, tok, label, 0, shipped, ps, b5=True, n=48)
        wrapped = E17.free_urls(LexiconWrapper(model, tok, mem, {f: 1.0 for f in WORD_FIELDS}),
                                tok, label, 0, shipped, ps, b5=True, n=48)
        out[label] = {"identical": plain == wrapped, "n": len(plain)}
        print(f"  V-wrap {label}: identical {plain == wrapped} ({len(plain)} URLs)", flush=True)
        del mem
        gc.collect()
    R1._write(OUT / "wrapcheck.json", out, indent=1)


def part_gen(label: str) -> None:
    import torch
    import e17_common as E17
    torch.set_num_threads(5)
    GEN.mkdir(parents=True, exist_ok=True)
    keep = R1._read(OUT / "oracle.json")[label]["keep"]
    model, tok, shipped, ps = _engine()
    mem = Memory(memory_vocab(label))
    for g in range(G):
        dest = GEN / f"free_lx_{label}_g{g}.json"
        if dest.exists():
            continue
        t0 = time.time()
        wrap = LexiconWrapper(model, tok, mem, keep)
        urls = E17.free_urls(wrap, tok, label, g, shipped, ps, b5=True, n=M_PER_G)
        dest.write_text(json.dumps(urls), encoding="utf-8")
        R1._write(GEN / f"trace_lx_{label}_g{g}.json", {"stats": dict(wrap.stats), "keep": keep,
                                                         "seconds": round(time.time() - t0)}, indent=1)
        print(f"  {label} g{g}: {len(urls)} URLs in {time.time() - t0:.0f}s; violations "
              f"{wrap.stats['violations']}; e.g. {urls[:2]}", flush=True)


# --------------------------------------------------------------------------- #
# Part B readings
# --------------------------------------------------------------------------- #

def _lx(label: str, g: int) -> List[str]:
    return R1._full(GEN / f"free_lx_{label}_g{g}.json")


def _n_rule(cands: Sequence[int], real: Sequence[int]) -> int:
    return max(0, (min(6_000, min(real) // 4, int(0.9 * min(cands))) // 100) * 100)


def part_runB() -> None:
    from cr1_score import pools
    from uie.evaluation import FIELDS
    t_all = time.time()
    dest = OUT / "B_fits.json"
    res = R1._resume(dest)
    meta = {}
    for label in LABELS:
        real = R1._real_pool(pools, label)
        lx = {g: _lx(label, g) for g in range(G)}
        base = {g: R1._full(str(SD.SOURCES["e16_b5"]).format(label=label, g=g)) for g in range(G)}
        views = {
            "whole": lambda s: s,
            "rest": lambda s: R0.mask_host(s) if R0.rest_bearing(s) else None,
            "host": lambda s: R0.host_only(s),
            "path": lambda s: FIELDS["PTH"](s) or None,
        }
        detectors = {"whole": ("D1b", "D1", "D0", "D4"), "rest": ("D1b",), "host": ("D1b",), "path": ("D0",)}
        fps = {"D1b": make_d1b(label), "D1": SD.make_d1(label), "D0": SD.d0_fit_predict, "D4": R1.d4_fit_predict}
        for view, fn in views.items():
            r_v = {g: [x for x in (fn(s) for s in real[g]) if x] for g in range(G)}
            l_v = {g: [x for x in (fn(s) for s in lx[g]) if x] for g in range(G)}
            b_v = {g: [x for x in (fn(s) for s in base[g]) if x] for g in range(G)}
            n = _n_rule([len(v) for v in l_v.values()], [len(v) for v in r_v.values()])
            meta[f"{view}|{label}"] = n
            for g in range(G):
                pool = list(r_v[g])
                random.Random(R2_SEED + 7 + 100 * LABELS.index(label) + g).shuffle(pool)
                A, B, C = (pool[i * n:(i + 1) * n] for i in range(3))
                lxs = random.Random(R2_SEED + 8 + 100 * LABELS.index(label) + g).sample(l_v[g], n)
                bs = random.Random(R2_SEED + 9 + 100 * LABELS.index(label) + g).sample(b_v[g], n)
                seed = SD.seed_of(label, g) * 10
                for det in detectors[view]:
                    key = f"{det}|{view}|{label}|{g}"
                    R1._task(res, dest, f"{key}|floor", fps[det], A, B, seed, t_all)
                    R1._task(res, dest, f"{key}|base", fps[det], C, bs, seed + TASK_INDEX["base"], t_all)
                    R1._task(res, dest, f"{key}|lx", fps[det], C, lxs, seed + TASK_INDEX["lx"], t_all)
        del fps
        gc.collect()
    R1._write(OUT / "B_n.json", meta, indent=1)
    print("runB done", flush=True)


def _pair_novelty(paths: Sequence[str], train_pairs: np.ndarray) -> float:
    pairs = []
    for p in paths:
        w = WORD.findall(p.lower())
        pairs.extend(SD.h64(a + " " + b) for a, b in zip(w, w[1:]))
    if not pairs:
        return float("nan")
    q = np.array(pairs, dtype=np.uint64)
    i = np.clip(np.searchsorted(train_pairs, q), 0, len(train_pairs) - 1)
    return float(np.mean(train_pairs[i] != q))


def part_metrics() -> None:
    from cr1_score import pools
    from qc1_common import NearIndex
    from uie.bloom import BloomFilter
    from uie.evaluation import FIELDS
    from uie.factorization import parse
    from uie.gate import Gate
    bloom = BloomFilter.load(R1.BLOOM)
    out: dict = {}
    for label in LABELS:
        t0 = time.time()
        judge = DictVocab(judge_vocab(label))
        real = R1._real_pool(pools, label)
        srcs = {"real": [s for g in range(G) for s in real[g][:M_PER_G]],
                "base": [s for g in range(G) for s in R1._full(str(SD.SOURCES["e16_b5"]).format(label=label, g=g))[:M_PER_G]],
                "lx": [s for g in range(G) for s in _lx(label, g)]}
        tables = R1._training_tables(label, parse)
        frequent = R1._frequent(label, parse, tables)
        pairs = np.load(OUT / f"train_pairs_{label}.npy")
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
        node["pair_novelty"] = {k: _pair_novelty([x["PTH"] for x in vals[k] if x["PTH"]], pairs) for k in srcs}
        for grp, fields in (("host", ("LBL", "SUB")), ("rest", ("PTH", "QRY"))):
            words = {k: [w for x in vals[k] for f in fields for w in WORD.findall(x[f].lower())] for k in srcs}
            n = min(20_000, min(len(w) for w in words.values()) // 2)
            seed = R2_SEED + 70_000 + 100 * LABELS.index(label)
            node[f"distinct_{grp}"] = {"n": n, **{k: R1._distinct_per_1000(w, n, seed) for k, w in words.items()}}
            node[f"unseen_judge_{grp}"] = {k: (float(np.mean(judge.counts(w) == 0)) if w else None, len(w))
                                           for k, w in words.items()}
        node["structure"] = {k: {"path": st.mean(bool(x["PTH"]) for x in vals[k]),
                                 "query": st.mean(bool(x["QRY"]) for x in vals[k]),
                                 "path_chars": st.mean(len(x["PTH"]) for x in vals[k] if x["PTH"]),
                                 "path_segments": st.mean(x["PTH"].count("/") + 1 for x in vals[k] if x["PTH"])}
                             for k in srcs}
        traces = [R1._read(GEN / f"trace_lx_{label}_g{g}.json")["stats"] for g in range(G)]
        tot = Counter()
        for t in traces:
            tot.update(t)
        node["wrapper"] = {"violations": tot["violations"], "constrained_steps": tot["constrained_steps"],
                           "free_share": {f: tot[f"free_{f}"] / tot[f"words_{f}"] for f in WORD_FIELDS
                                          if tot[f"words_{f}"]}}
        out[label] = node
        print(f"  {label}: metrics done ({time.time() - t0:.0f}s)", flush=True)
        del tables, frequent, vals, judge
        gc.collect()
    R1._write(OUT / "B_metrics.json", out, indent=1)


# --------------------------------------------------------------------------- #
# verify and read
# --------------------------------------------------------------------------- #

def _tol(p: float, n: int) -> float:
    return 2 * math.sqrt(max(p * (1 - p), 1e-12) / max(n, 1))


def part_verify() -> None:
    vocab = R1._read(OUT / "vocab.json")
    checks = {"V-split": all(sum(v["rows"].values()) == R1._read(R1.OUT / "meta.json")["bpe_sample"]["sizes"][l]
                             for l, v in vocab.items())}
    import inspect
    src = inspect.getsource(part_oracle) + inspect.getsource(LexiconWrapper) + inspect.getsource(part_gen) \
        + inspect.getsource(Memory) + inspect.getsource(transform) + inspect.getsource(keep_rates)
    checks["V-honest"] = "judge_vocab" not in src and '"B")' not in src
    wc = R1._read(OUT / "wrapcheck.json") if (OUT / "wrapcheck.json").exists() else {}
    checks["V-wrap"] = bool(wc) and all(v["identical"] for v in wc.values())
    floors = {}
    for name, path in (("A", OUT / "A_D1b.json"), ("A4", OUT / "A_D4.json"), ("B", OUT / "B_fits.json")):
        if not path.exists():
            continue
        res = R1._read(path)
        prefixes = sorted({k.rsplit("|", 2)[0] for k in res if k.endswith("|floor")})
        for p in prefixes:
            fl = [res[f"{p}|{g}|floor"] for g in range(G) if f"{p}|{g}|floor" in res]
            floors[f"{name} {p}"] = abs(st.mean(fl) - 0.5) <= 0.03
    checks["V-floor"] = all(floors.values())
    if (OUT / "B_metrics.json").exists():
        mtr = R1._read(OUT / "B_metrics.json")
        checks["V-trie"] = all(v["wrapper"]["violations"] == 0 for v in mtr.values())
    passed = all(checks.values())
    R1._write(OUT / "verify.json", {"checks": checks, "floors": floors, "pass": passed}, indent=1)
    for k, v in checks.items():
        print(f"  {k}: {v}")
    if not checks["V-floor"]:
        print("  floors off:", [k for k, v in floors.items() if not v])
    print(f"VERIFY: {'PASS' if passed else 'FAIL'}")


def part_read() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    ver = R1._read(OUT / "verify.json")
    if not ver["pass"]:
        raise SystemExit("STOP: R2's validity checks did not pass; nothing is read")
    fits, mtr, ns = R1._read(OUT / "B_fits.json"), R1._read(OUT / "B_metrics.json"), R1._read(OUT / "B_n.json")
    out: dict = {"fits": {}, "metrics": mtr, "n": ns, "reading": {}}
    for det, view in (("D1b", "whole"), ("D1b", "rest"), ("D1b", "host"), ("D1", "whole"), ("D0", "whole"),
                      ("D0", "path"), ("D4", "whole")):
        for label in LABELS:
            node = R1._verdicts(fits, f"{det}|{view}|{label}", ("base", "lx"))
            xb = node["base"]["mean"]
            node["lx"]["closed"] = (xb - node["lx"]["mean"]) / xb if xb > 0 else None
            node["paired"] = SD.paired_t([fits[f"{det}|{view}|{label}|{g}|lx"] for g in range(G)],
                                         [fits[f"{det}|{view}|{label}|{g}|base"] for g in range(G)])
            out["fits"][f"{det}|{view}|{label}"] = node
    for label in LABELS:
        m = mtr[label]
        whole = out["fits"][f"D1b|whole|{label}"]
        realism = (whole["lx"]["verdict"] == "CLEARED" or (whole["lx"]["closed"] or 0) >= 0.7) and \
            out["fits"][f"D0|path|{label}"]["lx"]["mean"] - out["fits"][f"D0|path|{label}"]["base"]["mean"] <= 0.0141
        g = {}
        for f in ("PTH", "RD"):
            p_lx, p_nat, p_base = m[f]["lx"]["copy"], m[f]["real"]["copy"], m[f]["base"]["copy"]
            ref = max(p_nat, p_base)
            g[f"copy {f}"] = p_lx <= ref + _tol(ref, m["n"]["lx"])
        mem_lx = m["gate"]["lx"].get("memorized", 0.0)
        mem_base = m["gate"]["base"].get("memorized", 0.0)
        g["novelty"] = mem_lx <= mem_base + _tol(mem_base, m["n"]["lx"])
        pn = m["pair_novelty"]
        g["pair novelty"] = pn["lx"] >= pn["real"] - _tol(pn["real"], 20_000)
        crea = {}
        for grp in ("host", "rest"):
            d = m[f"distinct_{grp}"]
            crea[f"distinct {grp}"] = abs(d["lx"] - d["real"]) <= 0.05 * d["real"]
            ur, nr = m[f"unseen_judge_{grp}"]["real"]
            ul, nl = m[f"unseen_judge_{grp}"]["lx"]
            crea[f"invents {grp}"] = ul >= ur - _tol(ur, min(nr, nl))
        s_lx, s_base = m["structure"]["lx"], m["structure"]["base"]
        struct = {"path presence": abs(s_lx["path"] - s_base["path"]) <= _tol(s_base["path"], m["n"]["lx"]),
                  "query presence": abs(s_lx["query"] - s_base["query"]) <= _tol(s_base["query"], m["n"]["lx"])}
        inv = lambda gk: sum(v for k, v in m["gate"][gk].items() if k not in ("n", "memorized", "duplicate_in_batch"))
        struct["validity"] = inv("lx") <= inv("base") + _tol(max(inv("base"), 1e-4), m["n"]["lx"])
        guards = {**g, **crea, **struct}
        improved = (whole["lx"]["closed"] or 0) >= 0.3
        verdict = ("MEETS" if realism and all(guards.values()) else
                   "COSTLY" if improved and not all(guards.values()) else
                   "PARTIAL" if improved else "NO EFFECT")
        out["reading"][label] = {"realism": realism, "guards": guards, "verdict": verdict}
    R1._write(OUT / "r2_result.json", out, indent=1)
    for key, node in out["fits"].items():
        b, l = node["base"], node["lx"]
        c = f"closed {l['closed']:+.0%}" if l.get("closed") is not None else ""
        print(f"{key:<20} floor {node['floor']['mean_auc']:.4f}  base {b['mean']:+.4f} {b['verdict']:<10} "
              f"lx {l['mean']:+.4f} {l['verdict']:<10} {c}")
    for label, r in out["reading"].items():
        print(f"\n{label}: {r['verdict']}  realism {r['realism']}  guards {json.dumps(r['guards'])}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--part", required=True, choices=("vocab", "oracle", "runA", "readA", "wrapcheck", "gen",
                                                      "runB", "metrics", "verify", "read"))
    ap.add_argument("--label", choices=LABELS)
    args = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")
    OUT.mkdir(parents=True, exist_ok=True)
    if args.part == "gen":
        if not args.label:
            ap.error("--label")
        part_gen(args.label)
        return
    {"vocab": part_vocab, "oracle": part_oracle, "runA": part_runA, "readA": part_readA,
     "wrapcheck": part_wrapcheck, "runB": part_runB, "metrics": part_metrics, "verify": part_verify,
     "read": part_read}[args.part]()


if __name__ == "__main__":
    main()
