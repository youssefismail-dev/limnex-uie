"""QG -- shared definitions for the query-gap investigation (research/qgap/QG_PLAN.md).

The query hazard of a sequence (QG_PLAN section 2): from the `<LBL>` marker to the
last position before the first `<QRY>`, `<EOS>` or `<FRG>` after it -- or, for a row
that reached max_len without stopping, to the last position whose prediction was
sampled -- sum p(`<QRY>`). y is 1 when that first stop is `<QRY>`. A position is on
the HOST route before the `<PTH>` marker and on the PATH route from it on.

Diagnostics only: nothing here is imported by `uie/`, and nothing changes the
baseline (uie/baseline.py).
"""
from __future__ import annotations

import gzip
import json
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Sequence

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data" / "qgap"
BUNDLE = ROOT / "data" / "bundle2"

SEED = 20260928                  # QG_PLAN 5.1: every row and URL sample
BOOT_SEED, BOOT = 20260806, 1000
N_TRAIN, N_VAL, N_GEN = 40_000, 10_000, 20_000
TOP = 20                         # suffix groups: the class's 20 largest, plus "other"
MIN_EVENTS = 100                 # a cell with fewer query events is not read alone
BATCH = 48
LABELS = ("benign", "phishing")
STREAMS = {"benign": ("name_train", "name_val"),
           "phishing": ("phishing_name_train", "phishing_name_val")}
PAGE = {"benign": "structure_train", "phishing": "phishing_structure_train"}
OTHER = "(other)"


def rows(stem: str) -> Iterator[str]:
    with gzip.open(BUNDLE / f"{stem}.jsonl.gz", "rt", encoding="utf-8") as fh:
        for line in fh:
            yield json.loads(line)["url"]


def encode_free(tok, url: str, label: str) -> Optional[List[int]]:
    """Free mode's exact conditioning: domain frame, both presence controls `?`."""
    try:
        return tok.encode(url, label, frame="domain", hide_fields=("PTH", "QRY"))
    except Exception:                   # a malformed generated string
        return None


def decision(seq: Sequence[int], tok) -> Optional[dict]:
    """Where the query decision lives in one sequence; None if it has no `<LBL>`.

    `positions` are the indices i whose prediction (of token i + 1) could have been
    `<QRY>`; `pth` is the `<PTH>` marker's index or None; `y` whether the first stop
    is `<QRY>`; `stopped` whether any stop was reached (False: cut at max_len).
    """
    lbl, qry, pth = tok.marker_id("LBL"), tok.marker_id("QRY"), tok.marker_id("PTH")
    stops = {qry, tok.eos_id, tok.marker_id("FRG")}
    try:
        k = list(seq).index(lbl)
    except ValueError:
        return None
    j = next((i for i in range(k + 1, len(seq)) if seq[i] in stops), None)
    end = j if j is not None else len(seq) - 1          # exclusive
    p = next((i for i in range(k + 1, end) if seq[i] == pth), None)
    return {"k": k, "end": end, "pth": p, "y": j is not None and seq[j] == qry,
            "stopped": j is not None,
            "path_len": (end - p - 1) if p is not None else 0}


def _probs(model, seqs: List[List[int]], pad_id: int) -> torch.Tensor:
    width = max(len(s) for s in seqs)
    ids = torch.tensor([s + [pad_id] * (width - len(s)) for s in seqs])
    with torch.no_grad():
        return F.softmax(model(ids).float(), dim=-1)


def hazards(model, tok, seqs: Sequence[Optional[List[int]]], *, max_len: int,
            batch: int = BATCH) -> List[Optional[dict]]:
    """Per sequence: y and h on each route, the path, the length -- or None when
    the sequence is missing, longer than max_len, or has no `<LBL>`. Sequences are
    forwarded in length order (less padding; causal, so padding changes nothing)."""
    qry = tok.marker_id("QRY")
    out: List[Optional[dict]] = [None] * len(seqs)
    todo = [(i, s, d) for i, s in enumerate(seqs)
            if s is not None and len(s) <= max_len and (d := decision(s, tok)) is not None]
    todo.sort(key=lambda t: len(t[1]))
    for b in range(0, len(todo), batch):
        chunk = todo[b:b + batch]
        probs = _probs(model, [s for _, s, _ in chunk], tok.pad_id)
        for r, (i, s, d) in enumerate(chunk):
            pq = probs[r, d["k"]:d["end"], qry].double()
            cut = (d["pth"] - d["k"]) if d["pth"] is not None else len(pq)
            h_host, h_path = float(pq[:cut].sum()), float(pq[cut:].sum())
            on_path = d["pth"] is not None
            out[i] = {"y_host": int(d["y"] and not on_path), "y_path": int(d["y"] and on_path),
                      "h_host": h_host, "h_path": h_path, "has_path": int(on_path),
                      "path_len": d["path_len"], "n": len(s), "stopped": int(d["stopped"])}
    return out


#: QG2-D (PREREG_QG2D section 2): the categories a decision position's probability
#: is split into. `content` is any character; `other` is everything else.
CATS = ("QRY", "PTH", "EOS", "FRG", "PRT", "SUB", "content", "other")
HOST_END = ("PTH", "QRY", "EOS", "FRG")          # what can follow a finished host
PATH_END = ("QRY", "EOS", "FRG")                 # what can follow a finished path


def category_map(tok) -> torch.Tensor:
    """A (vocab, len(CATS)) one-hot matrix: token id -> its category."""
    cat = [CATS.index("other")] * tok.vocab_size
    for c in ("QRY", "PTH", "FRG", "PRT", "SUB"):
        cat[tok.marker_id(c)] = CATS.index(c)
    cat[tok.eos_id] = CATS.index("EOS")
    for ch in tok._chars:
        cat[tok.stoi[ch]] = CATS.index("content")
    return F.one_hot(torch.tensor(cat), len(CATS)).to(torch.float64)


def profile(model, tok, seqs: Sequence[Optional[List[int]]], *, max_len: int,
            batch: int = BATCH) -> List[Optional[dict]]:
    """Per sequence (PREREG_QG2D section 2): for the host region and the path
    region, the summed probability of each category (`hp`, `pp`) and the count of
    true next tokens in it (`hy`, `py`); the host-end and path-end snapshots
    (`hep`/`hey`, `pep`/`pey`: the category probabilities at that position and the
    true outcome's category index); and `dev`, the largest |sum of categories - 1|.
    None where `hazards` would give None, or where the row never stopped."""
    onehot = category_map(tok)
    cat_of = onehot.argmax(dim=1).tolist()
    out: List[Optional[dict]] = [None] * len(seqs)
    todo = [(i, s, d) for i, s in enumerate(seqs)
            if s is not None and len(s) <= max_len and (d := decision(s, tok)) is not None
            and d["stopped"]]
    todo.sort(key=lambda t: len(t[1]))
    for b in range(0, len(todo), batch):
        chunk = todo[b:b + batch]
        probs = _probs(model, [s for _, s, _ in chunk], tok.pad_id).double()
        cats = probs @ onehot                                  # (B, T, len(CATS))
        for r, (i, s, d) in enumerate(chunk):
            k, end, p = d["k"], d["end"], d["pth"]
            cut = p if p is not None else end
            c = cats[r, k:end]
            nxt = torch.tensor([cat_of[t] for t in s[k + 1:end + 1]])
            y = F.one_hot(nxt, len(CATS)).to(torch.float64)
            h = cut - k                                        # host positions
            rec = {"hp": c[:h].sum(0).tolist(), "hy": y[:h].sum(0).tolist(),
                   "pp": c[h:].sum(0).tolist(), "py": y[h:].sum(0).tolist(),
                   "hep": c[h - 1].tolist(), "hey": int(nxt[h - 1]),
                   "pep": c[-1].tolist() if p is not None else None,
                   "pey": int(nxt[-1]) if p is not None else None,
                   "has_path": int(p is not None),
                   "dev": float((c.sum(1) - 1.0).abs().max())}
            out[i] = rec
    return out


#: QG3 (PREREG_QG3 section 2): where a position sits; it predicts the next token.
REGIONS = ("host", "path", "query", "fragment")


def regions(seq: Sequence[int], tok) -> Optional[Dict[str, tuple]]:
    """(start, end) position ranges per region of a stopped sequence, or None.
    host: <LBL> to the path or the first stop; path: <PTH> to the first stop;
    query: <QRY> to <FRG> or <EOS>; fragment: <FRG> to <EOS>."""
    d = decision(seq, tok)
    if d is None or not d["stopped"]:
        return None
    qry, frg, eos = tok.marker_id("QRY"), tok.marker_id("FRG"), tok.eos_id
    k, j1, p = d["k"], d["end"], d["pth"]
    out = {"host": (k, p if p is not None else j1), "path": (p, j1) if p is not None else None,
           "query": None, "fragment": None}
    j2 = j1
    if seq[j1] == qry:
        j2 = next((i for i in range(j1 + 1, len(seq)) if seq[i] in (frg, eos)), None)
        if j2 is None:
            return None
        out["query"] = (j1, j2)
    if seq[j2] == frg:
        j3 = next((i for i in range(j2 + 1, len(seq)) if seq[i] == eos), None)
        if j3 is None:
            return None
        out["fragment"] = (j2, j3)
    return out


def ledger(model, tok, seqs: Sequence[Optional[List[int]]], *, max_len: int,
           block: int = 100, batch: int = BATCH):
    """Per block of `block` consecutive rows (in the given order): for each region,
    the summed probability of every token over the region's positions (P), the
    count of each token as the true next token there (N), and the number of
    positions (POS). Returns (P, N, POS, rows_read) as numpy arrays:
    P, N: (blocks, len(REGIONS), vocab); POS: (blocks, len(REGIONS))."""
    import numpy as np
    V, R = tok.vocab_size, len(REGIONS)
    nb = -(-len(seqs) // block)
    P = np.zeros((nb, R, V))
    N = np.zeros((nb, R, V))
    POS = np.zeros((nb, R))
    todo = [(i, s, g) for i, s in enumerate(seqs)
            if s is not None and len(s) <= max_len and (g := regions(s, tok)) is not None]
    read = len(todo)
    todo.sort(key=lambda t: len(t[1]))
    for b in range(0, len(todo), batch):
        chunk = todo[b:b + batch]
        probs = _probs(model, [s for _, s, _ in chunk], tok.pad_id).double().numpy()
        for r, (i, s, g) in enumerate(chunk):
            blk = i // block
            for ri, name in enumerate(REGIONS):
                span = g[name]
                if span is None or span[1] <= span[0]:
                    continue
                lo, hi = span
                P[blk, ri] += probs[r, lo:hi].sum(0)
                N[blk, ri] += np.bincount(s[lo + 1:hi + 1], minlength=V)
                POS[blk, ri] += hi - lo
    return P, N, POS, read


def suffix_groups(shipped: Dict[str, int], top: int = TOP):
    """(the class's `top` largest suffixes in the shipped table, each group's share,
    with the rest pooled as OTHER)."""
    total = sum(shipped.values())
    ranked = sorted(shipped.items(), key=lambda kv: (-kv[1], kv[0]))
    groups = [s for s, _ in ranked[:top]]
    share = {s: shipped[s] / total for s in groups}
    share[OTHER] = 1.0 - sum(share.values())
    return groups, share


def group_of(suffix: str, groups: Sequence[str]) -> str:
    return suffix if suffix in groups else OTHER
