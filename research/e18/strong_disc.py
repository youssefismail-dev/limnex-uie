"""The stronger-discriminator test (research/e18/STRONG_DISC_PREREG.md, committed at 83821c8
BEFORE this code was written). Informational only: E3 stays closed, nothing is tuned.

    python research/e18/strong_disc.py --part data                 # samples + vocabulary (cached)
    python research/e18/strong_disc.py --part run --detector D0    # then D1, D2, D3
    python research/e18/strong_disc.py --part read                 # the registered reading

Per class and generation g = 0..4: four disjoint real samples of N (A, B: the floor; C: the
real side of every main task; D: chimera hosts and donors), and N URLs from each source
(e16_b5, e9_ln raw, chimeras). Every task is 5-fold stratified, out-of-fold ROC AUC; the
folds depend only on (class, g, task), so every detector sees identical splits.
excess = AUC(C vs source) - AUC(A vs B). Read like a gate cell (t vs 0.05 at 2.776).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import re
import statistics as st
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
for p in (ROOT, ROOT / "research" / "cr1", ROOT / "research" / "native"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

OUT = ROOT / "data" / "e18" / "strong_disc"
SAMPLES = OUT / "samples"
BUNDLE = ROOT / "data" / "bundle2"
LABELS = ("benign", "phishing")
G = 5
N = 6_000
FOLDS = 5
SEED = 20261008
TCRIT = 2.7764                       # t(0.975, 4), the gate's
BAR = 0.05
SOURCES = {"e16_b5": ROOT / "data" / "e17" / "decode" / "gen" / "free_e16_b5_{label}_g{g}.json",
           "e9_ln": ROOT / "data" / "e9" / "gen" / "free_e9_ln_raw_{label}_g{g}.json"}
TASKS = ("floor", "e16_b5", "e9_ln", "chimera")
TRAIN_STEMS = {"benign": ("name_train", "structure_train"),
               "phishing": ("phishing_name_train", "phishing_structure_train")}
TOKEN = re.compile(r"[a-z]{3,}")
MAX_LEN = 160


def seed_of(label: str, g: int) -> int:
    return SEED + 100 * LABELS.index(label) + g


# --------------------------------------------------------------------------- #
# part data
# --------------------------------------------------------------------------- #

def split_url(s: str):
    """A scheme-stripped URL as (host, path, query, fragment); path keeps its '/'s,
    query and fragment exclude their markers (None when the marker is absent)."""
    cut = min([i for i in (s.find("/"), s.find("?"), s.find("#")) if i >= 0] or [len(s)])
    host, rest = s[:cut], s[cut:]
    frag = None
    if "#" in rest:
        rest, frag = rest.split("#", 1)
    query = None
    if "?" in rest:
        rest, query = rest.split("?", 1)
    return host, rest, query, frag


def join_url(host, path, query, frag) -> str:
    return host + path + ("?" + query if query is not None else "") + ("#" + frag if frag is not None else "")


def chimeras(donors: list, rng: random.Random) -> tuple:
    """Each URL keeps its own host and presence pattern; its path body and query are
    replaced by those of OTHER real URLs of the same class (registration section 3)."""
    parts = [split_url(s) for s in donors]
    paths = [p for _, p, _, _ in parts if p not in ("", "/")]
    queries = [q for _, _, q, _ in parts if q]
    out, changed = [], 0
    for host, path, query, frag in parts:
        new_path = rng.choice(paths) if path not in ("", "/") else path
        new_query = rng.choice(queries) if query else query
        changed += (new_path != path) or (new_query != query)
        out.append(join_url(host, new_path, new_query, frag))
    return out, changed / len(out)


def part_data() -> None:
    from cr1_score import pools
    from uie.evaluation import FIELDS, _one_per_host
    SAMPLES.mkdir(parents=True, exist_ok=True)
    for label in LABELS:
        pop = pools(BUNDLE, label)["mixed"]
        for g in range(G):
            seed = seed_of(label, g)
            rng = random.Random(seed)
            real = [x for x in (FIELDS["FULL"](u) for u in _one_per_host(pop, seed)) if x]
            rng.shuffle(real)
            if len(real) < 4 * N:
                raise SystemExit(f"STOP: {label} pool {len(real)} < 4N")
            A, B, C, D = (real[i * N:(i + 1) * N] for i in range(4))
            node = {"A": A, "B": B, "C": C}
            for src, pattern in SOURCES.items():
                urls = json.loads(Path(str(pattern).format(label=label, g=g)).read_text(encoding="utf-8"))
                full = [x for x in (FIELDS["FULL"](u) for u in urls) if x]
                node[src] = rng.sample(full, N)
            node["chimera"], share = chimeras(D, rng)
            node["meta"] = {"seed": seed, "N": N, "pool": len(real), "chimera_changed_share": share}
            (SAMPLES / f"{label}_g{g}.json").write_text(json.dumps(node), encoding="utf-8")
            print(f"  {label} g{g}: pool {len(real):,}; chimeras altered {share:.1%}", flush=True)
        build_vocab(label)


def h64(t: str) -> int:
    return int.from_bytes(hashlib.blake2b(t.encode(), digest_size=8).digest(), "little")


def _merge(u, c, batch):
    nu, nc = np.unique(np.array(batch, dtype=np.uint64), return_counts=True)
    allu, allc = np.concatenate([u, nu]), np.concatenate([c, nc])
    order = np.argsort(allu, kind="stable")
    allu, allc = allu[order], allc[order]
    keep = np.r_[True, allu[1:] != allu[:-1]]
    idx = np.cumsum(keep) - 1
    return allu[keep], np.bincount(idx, weights=allc).astype(np.int64)


def build_vocab(label: str) -> None:
    """Token counts over the class's real TRAINING pools (never the held-out URLs), as
    sorted 64-bit hashes: exact enough, and memory-light."""
    from n2a_reference import rows
    from uie.evaluation import _strip_scheme
    t0 = time.time()
    u, c, buf, n = np.empty(0, np.uint64), np.empty(0, np.int64), [], 0
    for stem in TRAIN_STEMS[label]:
        for url in rows(stem):
            n += 1
            buf.extend(h64(t) for t in TOKEN.findall(_strip_scheme(url).lower()))
            if len(buf) >= 2_000_000:
                u, c = _merge(u, c, buf)
                buf = []
    if buf:
        u, c = _merge(u, c, buf)
    np.savez(OUT / f"vocab_{label}.npz", hashes=u, counts=c, urls=n)
    print(f"  vocabulary {label}: {n:,} training URLs, {len(u):,} distinct tokens "
          f"({time.time() - t0:.0f}s)", flush=True)


# --------------------------------------------------------------------------- #
# the detectors
# --------------------------------------------------------------------------- #

def oof_auc(fit_predict, texts: list, y: np.ndarray, seed: int) -> float:
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import StratifiedKFold
    scores = np.zeros(len(texts), dtype=float)
    for k, (tr, te) in enumerate(StratifiedKFold(FOLDS, shuffle=True, random_state=seed).split(texts, y)):
        scores[te] = fit_predict([texts[i] for i in tr], y[tr], [texts[i] for i in te], seed + k)
    return float(roc_auc_score(y, scores))


def d0_fit_predict(xtr, ytr, xte, seed):
    """The gate's own model (uie/evaluation.py real_vs_fake_auc), unchanged."""
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    m = make_pipeline(TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), max_features=50_000,
                                      sublinear_tf=True),
                      LogisticRegression(max_iter=2000, C=1.0))
    m.fit(xtr, ytr)
    return m.decision_function(xte)


class Vocab:
    def __init__(self, label: str):
        z = np.load(OUT / f"vocab_{label}.npz")
        self.h, self.c = z["hashes"], z["counts"]

    def counts(self, toks: list) -> np.ndarray:
        if not toks:
            return np.zeros(0, dtype=np.int64)
        q = np.array([h64(t) for t in toks], dtype=np.uint64)
        i = np.clip(np.searchsorted(self.h, q), 0, len(self.h) - 1)
        return np.where(self.h[i] == q, self.c[i], 0)


def _entropy(s: str) -> float:
    if not s:
        return 0.0
    n = len(s)
    return -sum(k / n * math.log2(k / n) for k in (s.count(ch) for ch in set(s)))


def d1_features(s: str, vocab: Vocab) -> list:
    host, path, query, frag = split_url(s)
    labels = host.split(".")
    segs = [x for x in path.split("/") if x]
    low = s.lower()
    toks = TOKEN.findall(low)
    cnt = vocab.counts(toks)
    seen = cnt > 0
    lc = np.log1p(cnt) if len(cnt) else np.zeros(1)
    host_toks = TOKEN.findall(host.lower())
    hc = vocab.counts(host_toks)
    rest_toks = TOKEN.findall((path + "?" + (query or "")).lower())
    rc = vocab.counts(rest_toks)
    n = max(len(s), 1)
    return [
        len(s), len(host), len(path), len(query or ""), len(frag or ""),
        len(labels), max((len(x) for x in labels), default=0), len(labels[-1]) if labels else 0,
        len(segs), (sum(len(x) for x in segs) / len(segs)) if segs else 0.0,
        max((len(x) for x in segs), default=0),
        path.endswith("/"), query is not None, (query or "").count("&") + (1 if query else 0),
        sum(ch.isdigit() for ch in s) / n, sum(ch.isupper() for ch in s) / n,
        sum(not ch.isalnum() for ch in s) / n, s.count("-"), s.count("_"), s.count("."),
        "%" in s, "@" in s, ":" in host, _entropy(s),
        max((len(m) for m in re.findall(r"\d+", s)), default=0),
        len(toks), (sum(len(t) for t in toks) / len(toks)) if toks else 0.0,
        max((len(t) for t in toks), default=0),
        # vocabulary: are the words real words? (counts from the real TRAINING corpus)
        float(seen.mean()) if len(cnt) else 1.0, float(lc.mean()), float(lc.min()),
        int((~seen).sum()) if len(cnt) else 0,
        max((len(t) for t, ok in zip(toks, seen) if not ok), default=0),
        float((hc > 0).mean()) if len(hc) else 1.0, float((rc > 0).mean()) if len(rc) else 1.0,
    ]


def make_d1(label: str):
    vocab = Vocab(label)
    cache: dict = {}

    def fit_predict(xtr, ytr, xte, seed):
        from sklearn.ensemble import HistGradientBoostingClassifier
        for s in set(xtr) | set(xte):
            if s not in cache:
                cache[s] = d1_features(s, vocab)
        m = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.1, random_state=seed)
        m.fit(np.array([cache[s] for s in xtr], dtype=float), ytr)
        return m.predict_proba(np.array([cache[s] for s in xte], dtype=float))[:, 1]
    return fit_predict


# --- D2 / D3: character models (fixed architecture, fixed epochs; no tuning) ---------------

def encode(texts: list):
    """Printable ASCII -> 2..96, anything else -> 1 (unknown), 0 = pad. A fixed map, so
    nothing about the test data leaks into the encoding."""
    out = []
    for s in texts:
        ids = [(ord(ch) - 30) if 32 <= ord(ch) <= 126 else 1 for ch in s[:MAX_LEN]]
        out.append(ids or [1])
    return out


def batches(seqs, ys, batch: int, rng: random.Random, shuffle: bool):
    """Length-bucketed batches (sorted by length, chunked, batch order shuffled), each
    padded only to its own longest sequence."""
    import torch
    order = sorted(range(len(seqs)), key=lambda i: len(seqs[i]))
    chunks = [order[i:i + batch] for i in range(0, len(order), batch)]
    if shuffle:
        rng.shuffle(chunks)
    for ch in chunks:
        L = max(len(seqs[i]) for i in ch)
        x = torch.zeros((len(ch), L), dtype=torch.long)
        for r, i in enumerate(ch):
            x[r, :len(seqs[i])] = torch.tensor(seqs[i])
        yield ch, x, (torch.tensor([ys[i] for i in ch], dtype=torch.float32) if ys is not None else None)


def make_torch_detector(kind: str):
    import torch
    import torch.nn as nn

    class CNN(nn.Module):
        def __init__(self):
            super().__init__()
            self.emb = nn.Embedding(98, 32, padding_idx=0)
            self.convs = nn.ModuleList([nn.Conv1d(32, 64, k, padding=k // 2) for k in (2, 3, 4, 5, 6)])
            self.head = nn.Sequential(nn.Dropout(0.3), nn.Linear(320, 64), nn.ReLU(), nn.Linear(64, 1))

        def forward(self, x):
            e = self.emb(x).transpose(1, 2)
            mask = (x == 0).unsqueeze(1)
            h = [c(e)[..., :x.shape[1]].masked_fill(mask, -1e4).max(dim=2).values.relu() for c in self.convs]
            return self.head(torch.cat(h, dim=1)).squeeze(1)

    class Transformer(nn.Module):
        def __init__(self):
            super().__init__()
            self.emb = nn.Embedding(98, 64, padding_idx=0)
            self.pos = nn.Embedding(MAX_LEN, 64)
            layer = nn.TransformerEncoderLayer(64, 4, 128, dropout=0.1, batch_first=True)
            self.enc = nn.TransformerEncoder(layer, 2)
            self.head = nn.Linear(64, 1)

        def forward(self, x):
            pad = x == 0
            h = self.emb(x) + self.pos(torch.arange(x.shape[1]).unsqueeze(0))
            h = self.enc(h, src_key_padding_mask=pad)
            keep = (~pad).unsqueeze(2).float()
            return self.head((h * keep).sum(1) / keep.sum(1).clamp(min=1)).squeeze(1)

    # fixed before any real result; the transformer went 8 -> 5 epochs on SYNTHETIC timing
    # alone (about 12 s per epoch on this CPU), recorded in STRONG_DISC_RESULT.md
    epochs, lr = {"cnn": (6, 2e-3), "transformer": (5, 1e-3)}[kind]

    def fit_predict(xtr, ytr, xte, seed):
        torch.manual_seed(seed)
        rng = random.Random(seed)
        model = CNN() if kind == "cnn" else Transformer()
        opt = torch.optim.Adam(model.parameters(), lr=lr)
        lossf = nn.BCEWithLogitsLoss()
        str_, ste = encode(xtr), encode(xte)
        model.train()
        for _ in range(epochs):
            for _, x, y in batches(str_, list(ytr), 128, rng, True):
                opt.zero_grad()
                lossf(model(x), y).backward()
                opt.step()
        model.eval()
        out = np.zeros(len(ste), dtype=float)
        with torch.no_grad():
            for ch, x, _ in batches(ste, None, 256, rng, False):
                out[ch] = model(x).numpy()
        return out
    return fit_predict


# --------------------------------------------------------------------------- #
# part run / part read
# --------------------------------------------------------------------------- #

def detector(name: str, label: str):
    if name == "D0":
        return d0_fit_predict
    if name == "D1":
        return make_d1(label)
    return make_torch_detector({"D2": "cnn", "D3": "transformer"}[name])


def task_data(node: dict, task: str):
    if task == "floor":
        a, b = node["A"], node["B"]
    else:
        a, b = node["C"], node[task]
    return a + b, np.array([0] * len(a) + [1] * len(b))


def part_run(name: str) -> None:
    dest = OUT / f"{name}.json"
    res = json.loads(dest.read_text(encoding="utf-8")) if dest.exists() else {}
    t_all = time.time()
    for label in LABELS:
        fp = detector(name, label)
        for g in range(G):
            node = json.loads((SAMPLES / f"{label}_g{g}.json").read_text(encoding="utf-8"))
            for ti, task in enumerate(TASKS):
                key = f"{label}|{g}|{task}"
                if key in res:
                    continue
                t0 = time.time()
                texts, y = task_data(node, task)
                auc = oof_auc(fp, texts, y, seed_of(label, g) * 10 + ti)
                res[key] = auc
                dest.write_text(json.dumps(res, indent=1), encoding="utf-8")
                print(f"  {name} {label:<9} g{g} {task:<8} done  ({time.time() - t0:.0f}s; "
                      f"{(time.time() - t_all) / 60:.0f} min so far)", flush=True)
    print(f"{name} done: {len(res)} tasks")


def verdict(xs):
    m = st.mean(xs)
    se = st.stdev(xs) / math.sqrt(len(xs))
    t = (m - BAR) / se if se else float("nan")
    v = "CLEARED" if t < -TCRIT else "CONVICTED" if t > TCRIT else "UNDECIDED"
    return {"mean": m, "se": se, "t": t, "verdict": v, "per_g": xs}


def paired_t(a, b):
    d = [x - y for x, y in zip(a, b)]
    m = st.mean(d)
    se = st.stdev(d) / math.sqrt(len(d))
    return {"mean_diff": m, "se": se, "t": m / se if se else float("nan")}


def part_read() -> None:
    res = {n: json.loads((OUT / f"{n}.json").read_text(encoding="utf-8"))
           for n in ("D0", "D1", "D2", "D3") if (OUT / f"{n}.json").exists()}
    out = {"registration": "research/e18/STRONG_DISC_PREREG.md (83821c8)", "N": N, "folds": FOLDS,
           "detectors": {}}
    excess = {}
    for name, r in res.items():
        node = {}
        for label in LABELS:
            fl = [r[f"{label}|{g}|floor"] for g in range(G)]
            node[f"{label} floor"] = {"mean_auc": st.mean(fl), "per_g": fl,
                                      "flag": abs(st.mean(fl) - 0.5) > 0.03}
            for task in TASKS[1:]:
                xs = [r[f"{label}|{g}|{task}"] - r[f"{label}|{g}|floor"] for g in range(G)]
                excess[(name, label, task)] = xs
                node[f"{label} {task}"] = verdict(xs)
        out["detectors"][name] = node
    # validity and the reading (registration section 5)
    valid, reading = {}, {}
    for name in res:
        p1 = all(out["detectors"][name][f"{lab} e9_ln"]["verdict"] == "CONVICTED" for lab in LABELS)
        floor_ok = not any(out["detectors"][name][f"{lab} floor"]["flag"] for lab in LABELS)
        p2 = {lab: (paired_t(excess[(name, lab, "chimera")], excess[("D0", lab, "chimera")])
                    if name != "D0" and "D0" in res else None) for lab in LABELS}
        valid[name] = {"V-P1 convicts e9_ln": p1, "V-floor ok": floor_ok,
                       "V-P2 coherence over D0": {lab: (v["t"] > TCRIT if v else None) for lab, v in p2.items()},
                       "chimera_vs_D0": p2}
    out["validity"] = valid
    deeper, all_cleared = [], True
    for name in res:
        if not valid[name]["V-P1 convicts e9_ln"]:
            continue
        for lab in LABELS:
            v = out["detectors"][name][f"{lab} e16_b5"]
            if v["verdict"] != "CLEARED":
                all_cleared = False
            if v["verdict"] == "CONVICTED":
                deeper.append(f"{name} {lab}: CONVICTED")
            if name != "D0" and "D0" in res:
                pt = paired_t(excess[(name, lab, "e16_b5")], excess[("D0", lab, "e16_b5")])
                reading[f"{name} {lab} vs D0"] = pt
                if pt["mean_diff"] > 0.05 and pt["t"] > TCRIT:
                    deeper.append(f"{name} {lab}: +{pt['mean_diff']:.4f} over D0 (t {pt['t']:.2f})")
    out["e16_b5_vs_D0"] = reading
    out["outcome"] = "DEEPER TELL" if deeper else ("HOLDS" if all_cleared else "MIXED")
    out["deeper"] = deeper
    (OUT / "result.json").write_text(json.dumps(out, indent=1, default=float), encoding="utf-8")
    print(f"{'detector':<4} {'class':<9} {'source':<8} {'excess':>8} {'SE':>7} {'t':>8}  verdict")
    for name in res:
        for lab in LABELS:
            print(f"{name:<4} {lab:<9} {'floor':<8} AUC {out['detectors'][name][f'{lab} floor']['mean_auc']:.4f}")
            for task in TASKS[1:]:
                v = out["detectors"][name][f"{lab} {task}"]
                print(f"{name:<4} {lab:<9} {task:<8} {v['mean']:>+8.4f} {v['se']:>7.4f} {v['t']:>8.2f}  {v['verdict']}")
    print("validity:", json.dumps({k: {kk: vv for kk, vv in v.items() if kk != 'chimera_vs_D0'} for k, v in valid.items()}))
    print("OUTCOME:", out["outcome"], deeper)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--part", required=True, choices=("data", "run", "read"))
    ap.add_argument("--detector", choices=("D0", "D1", "D2", "D3"))
    args = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")
    OUT.mkdir(parents=True, exist_ok=True)
    if args.part == "data":
        part_data()
    elif args.part == "run":
        if not args.detector:
            ap.error("--detector")
        part_run(args.detector)
    else:
        part_read()


if __name__ == "__main__":
    main()
