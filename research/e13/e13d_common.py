"""E13-D -- the cells, the pools, and the registered rules (research/e13/PREREG_E13D.md).

The detector machinery is QC1's (qc1_common: the gate's detector, the three-draw
harness, the seeds, t_crit, the void bounds), unchanged. New here:
- each cell's training twin;
- query templates;
- the n-gram learner;
- the DX1-DX5 rules.
"""
from __future__ import annotations

import collections
import json
import math
import random
import re
import statistics as st
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
for p in (ROOT, ROOT / "research" / "native", ROOT / "research" / "cr1", ROOT / "research" / "qc1",
          ROOT / "research" / "dc1"):
    sys.path.insert(0, str(p))

import qc1_common as C  # noqa: E402

E12 = ROOT / "data" / "e12"
OUT = ROOT / "data" / "e13d"
PRIMARY, REPLICATION = "e12_final", "e12_base"
SEEDS, TCRIT, N_REQ = C.SEEDS, C.TCRIT, C.N_REQ
FRACTIONS = (0.125, 0.25, 0.5, 1.0)
NGRAM_ORDER = 6
PROBE_MAX = 3000
PROBE_SEED = 20261003

#: (name, mode, label, state, field). The last is DX3's CLEARED reference.
CELLS = (
    ("free_phishing_QRY", "free", "phishing", None, "QRY"),
    ("control_benign_P0Q1_QRY", "control", "benign", "P0Q1", "QRY"),
    ("control_benign_P1Q1_QRY", "control", "benign", "P1Q1", "QRY"),
    ("control_benign_P1Q1_PTH", "control", "benign", "P1Q1", "PTH"),
    ("control_benign_P1Q1_SUB", "control", "benign", "P1Q1", "SUB"),
    ("control_phishing_P1Q0_PTH", "control", "phishing", "P1Q0", "PTH"),
)
REFERENCE = ("free_benign_QRY", "free", "benign", None, "QRY")
QUERY_CELLS = tuple(c for c in CELLS if c[4] == "QRY") + (REFERENCE,)
BY_NAME = {c[0]: c for c in CELLS + (REFERENCE,)}

#: DX5's generated populations (amendment 1): (label, mode, state, S).
GEN_PROBES = {
    "gen_phishing_P0Q0_S1": ("phishing", "control", "P0Q0", 1),
    "gen_phishing_P0Q1_S1": ("phishing", "control", "P0Q1", 1),
    "gen_benign_P0Q1_S0": ("benign", "control", "P0Q1", 0),
    "gen_benign_P0Q1_S1": ("benign", "control", "P0Q1", 1),
    "gen_benign_P0Q0_S0": ("benign", "control", "P0Q0", 0),
    "gen_benign_P0Q0_S1": ("benign", "control", "P0Q0", 1),
    "gen_free_phishing_P0Q0_S1": ("phishing", "free", "P0Q0", 1),
}

#: DX5's real populations (as registered; reported beside after amendment 1).
PROBES = {
    "phishing_P0Q0_S1": ("phishing", "P0Q0", 1),
    "phishing_P0Q1_S1": ("phishing", "P0Q1", 1),
    "benign_P0Q1_S0": ("benign", "P0Q1", 0),
    "benign_P0Q1_S1": ("benign", "P0Q1", 1),
    "benign_P0Q0_S0": ("benign", "P0Q0", 0),
    "benign_P0Q0_S1": ("benign", "P0Q0", 1),
}


# --------------------------------------------------------------------------- #
# Pools: the gate's real validation pool and its training twin
# --------------------------------------------------------------------------- #

def field_of(field: str):
    from uie.evaluation import FIELDS
    return FIELDS[field]


def _urls(label: str, split: str, state: Optional[str]) -> List[str]:
    """`name_<split>` of the class, filtered to the state with CR1's own test."""
    pre = "" if label == "benign" else f"{label}_"
    urls = C._jsonl(f"{pre}name_{split}")
    if state is not None:
        import e11_common as K
        from cr1_score import in_state
        urls = [u for u in urls if in_state(u, K.STATES[state])]
    return urls


def pool(cell, split: str = "val") -> List[str]:
    """The cell's field strings from one URL per host (the gate's seed), the field
    non-empty: the gate's own pool for `val`; its training twin for `train`."""
    from uie.evaluation import _one_per_host
    _, _, label, state, field = cell
    f = field_of(field)
    out = []
    for u in _one_per_host(_urls(label, split, state), C.POOL_SEED):
        s = f(u)
        if s:
            out.append(s)
    return out


def generated(cell, arm: str, gen: Path = E12 / "gen") -> List[str]:
    """The cell's generated field strings, the five generations pooled."""
    import e12_common as K
    name, mode, label, state, field = cell
    f = field_of(field)
    out = []
    for g in range(K.G):
        fn = f"free_{arm}_{label}_g{g}.json" if mode == "free" else f"ctrl_{arm}_{label}_{state}_g{g}.json"
        for u in json.loads((gen / fn).read_text(encoding="utf-8")):
            s = f(u)
            if s:
                out.append(s)
    return out


# --------------------------------------------------------------------------- #
# Templates and the n-gram learner (DX3)
# --------------------------------------------------------------------------- #

_UUID = re.compile(r"^[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}$")
_HEX = re.compile(r"^[0-9a-fA-F]{8,}$")


def value_class(v: Optional[str]) -> str:
    if v is None:
        return "-"                       # a key with no '='
    if v == "":
        return "e"
    if _UUID.match(v):
        return "U"
    if "@" in v:
        return "E"
    if v.isdigit():
        return "D"
    if _HEX.match(v) and any(c.isdigit() for c in v) and any(c.isalpha() for c in v):
        return "H"
    if v.isalpha():
        return "A"
    return "X"


def template(q: str) -> str:
    """The ordered parameter keys, each value replaced by its character class."""
    out = []
    for part in q.split("&"):
        k, sep, v = part.partition("=")
        out.append(f"{k}={value_class(v if sep else None)}")
    return "&".join(out)


class CharNGram:
    """Order-k character model with stupid back-off sampling: each step samples the
    next character from the longest context seen in training."""

    START, END = "\x02", "\x03"

    def __init__(self, order: int = NGRAM_ORDER, max_len: int = 400):
        self.order, self.max_len = order, max_len
        self.next: List[Dict[str, str]] = []

    def fit(self, strings: Sequence[str]) -> "CharNGram":
        build: List[Dict[str, list]] = [collections.defaultdict(list) for _ in range(self.order)]
        pad = self.START * (self.order - 1)
        for s in strings:
            t = pad + s + self.END
            for i in range(self.order - 1, len(t)):
                for k in range(self.order):              # context length k
                    build[k][t[i - k:i]].append(t[i])
        self.next = [{ctx: "".join(cs) for ctx, cs in b.items()} for b in build]
        return self

    def sample(self, n: int, rng: random.Random) -> List[str]:
        out = []
        for _ in range(n):
            ctx, chars = self.START * (self.order - 1), []
            while len(chars) < self.max_len:
                for k in range(self.order - 1, -1, -1):
                    nxt = self.next[k].get(ctx[len(ctx) - k:] if k else "")
                    if nxt:
                        break
                c = nxt[rng.randrange(len(nxt))]
                if c == self.END:
                    break
                chars.append(c)
                ctx = ctx + c
            out.append("".join(chars))
        return out


class WittenBell:
    """Interpolated Witten-Bell character model (DX7, amendment 2): order k, a
    uniform base over the training alphabet plus END and one unknown symbol."""

    START, END = "\x02", "\x03"

    def __init__(self, order: int = NGRAM_ORDER):
        self.order = order

    def fit(self, strings: Sequence[str]) -> "WittenBell":
        self.counts: List[Dict[str, Dict[str, int]]] = [collections.defaultdict(dict) for _ in range(self.order)]
        alphabet = set()
        pad = self.START * (self.order - 1)
        for s in strings:
            t = pad + s + self.END
            alphabet.update(s)
            for i in range(self.order - 1, len(t)):
                c = t[i]
                for k in range(self.order):
                    d = self.counts[k][t[i - k:i]]
                    d[c] = d.get(c, 0) + 1
        self.tot = [{h: sum(d.values()) for h, d in ck.items()} for ck in self.counts]
        self.alphabet = sorted(alphabet)
        self.vocab = len(alphabet) + 2               # + END + one unknown symbol
        return self

    def prob(self, c: str, history: str) -> float:
        p = 1.0 / self.vocab
        for k in range(self.order):                  # back-off chain, short to long
            h = history[len(history) - k:] if k else ""
            d = self.counts[k].get(h)
            if not d:
                break
            tot, typ = self.tot[k][h], len(d)
            p = (d.get(c, 0) + typ * p) / (tot + typ)
        return p

    def nll(self, s: str) -> float:
        """Nats per symbol over the string's characters and END."""
        t = self.START * (self.order - 1) + s + self.END
        total = 0.0
        for i in range(self.order - 1, len(t)):
            total -= math.log(self.prob(t[i], t[i - self.order + 1:i]))
        return total / (len(s) + 1)


def fit_reading(deltas: Sequence[float], seed: int = 20261004, resamples: int = 2000) -> dict:
    """DX7: UNDER-FIT if mean delta > 0 with the bootstrap 95% CI above 0; FITS if
    below 0 with the CI below 0; UNRESOLVED otherwise."""
    import numpy as np
    x = np.asarray(deltas, dtype=float)
    rng = np.random.default_rng(seed)
    means = x[rng.integers(0, len(x), size=(resamples, len(x)))].mean(axis=1)
    lo, hi = np.quantile(means, [0.025, 0.975])
    m = float(x.mean())
    reading = "UNDER-FIT" if lo > 0 else "FITS" if hi < 0 else "UNRESOLVED"
    return {"n": len(x), "delta": m, "ci": [float(lo), float(hi)], "reading": reading}


def pool_urls(cell, split: str = "val") -> List[str]:
    """The URLs behind `pool(cell, split)`, in the same order."""
    from uie.evaluation import _one_per_host
    f = field_of(cell[4])
    return [u for u in _one_per_host(_urls(cell[2], split, cell[3]), C.POOL_SEED) if f(u)]


def subsample(xs: Sequence[str], frac: float, seed: int) -> List[str]:
    if frac >= 1:
        return list(xs)
    return random.Random(seed).sample(list(xs), max(1, round(len(xs) * frac)))


# --------------------------------------------------------------------------- #
# The registered rules
# --------------------------------------------------------------------------- #

def one_sample_t(xs: Sequence[float]):
    m = st.mean(xs)
    sd = st.stdev(xs) if len(xs) > 1 else 0.0
    se = sd / math.sqrt(len(xs)) if sd else 0.0
    return m, se, (m / se if se else float("nan"))


def shift_reading(x_tv: Sequence[float], x_gv: Sequence[float]) -> dict:
    """DX1: SHIFT-DOMINANT if mean x_TV >= 0.5 mean x_GV and t > t_crit;
    SHIFT-PRESENT if significant and >= 0.25 x_GV; NO SHIFT otherwise."""
    m, se, t = one_sample_t(x_tv)
    gv = st.mean(x_gv)
    sig = t == t and t > TCRIT
    reading = ("SHIFT-DOMINANT" if sig and m >= 0.5 * gv else
               "SHIFT-PRESENT" if sig and m >= 0.25 * gv else "NO SHIFT")
    return {"x_tv": m, "se": se, "t": t, "x_gv": gv, "share": m / gv if gv else float("nan"),
            "reading": reading}


def gap_reading(x_gt: Sequence[float], x_gv: Sequence[float]) -> dict:
    """DX2: LEARNING-GAP if mean x_GT >= 0.5 mean x_GV and t > t_crit."""
    m, se, t = one_sample_t(x_gt)
    gv = st.mean(x_gv)
    gap = t == t and t > TCRIT and m >= 0.5 * gv
    return {"x_gt": m, "se": se, "t": t, "x_gv": gv, "share": m / gv if gv else float("nan"),
            "reading": "LEARNING-GAP" if gap else "NO GAP"}


def replicated(primary: str, replication: str) -> str:
    return primary if primary == replication else "NOT REPLICATED"


def attribution(shift: str, gap: str) -> str:
    """The registered table: DATA, LEARNING, BOTH, or UNRESOLVED."""
    if shift == "SHIFT-DOMINANT" and gap == "NO GAP":
        return "DATA (population)"
    if shift == "NO SHIFT" and gap == "LEARNING-GAP":
        return "LEARNING"
    if shift in ("SHIFT-DOMINANT", "SHIFT-PRESENT") and gap == "LEARNING-GAP":
        return "BOTH"
    return "UNRESOLVED"


def quantity_reading(g: float, x_half: Sequence[float], x_full: Sequence[float]) -> dict:
    """DX3: DATA-LIMITED only if the last doubling's template-coverage gain
    g >= 0.02 AND the learner's last-doubling improvement s >= 0.01 with paired
    t > t_crit."""
    diffs = [a - b for a, b in zip(x_half, x_full)]
    s, se, t = C.paired_t(diffs)
    limited = g >= 0.02 and s >= 0.01 and t == t and t > TCRIT
    return {"g": g, "s": s, "se": se, "t": t,
            "reading": "DATA-LIMITED" if limited else "NOT DATA-LIMITED"}


def obedience_reading(viol: Sequence[float], support: Sequence[float]) -> dict:
    """DX5: c = the top decile's share of the violation mass. UNIFORM if c < 0.30;
    CONCENTRATED if c >= 0.50 and the top decile's median training support is below
    a tenth of the population's; MIXED otherwise."""
    order = sorted(range(len(viol)), key=lambda i: viol[i], reverse=True)
    k = max(1, len(viol) // 10)
    total = sum(viol)
    c = sum(viol[i] for i in order[:k]) / total if total else float("nan")
    top_sup = st.median(support[i] for i in order[:k])
    all_sup = st.median(support)
    reading = ("UNIFORM" if c < 0.30 else
               "CONCENTRATED" if c >= 0.50 and top_sup < 0.1 * all_sup else "MIXED")
    return {"n": len(viol), "m": st.mean(viol), "c": c, "top_support": top_sup,
            "support": all_sup, "reading": reading}


def clear_probability(mean_x: float, sd_run: float, se: float, t_gate: float = 2.7764) -> dict:
    """DX4: P(CLEARED) for a new run at mean_x with between-run SD sd_run and the
    gate's rule at standard error se; and the mean x needed for P >= 0.8."""
    line = 0.05 - t_gate * se
    sd = max(sd_run, 1e-9)
    p = 0.5 * (1 + math.erf((line - mean_x) / (sd * math.sqrt(2))))
    return {"line": line, "p_clear": p, "x_for_0.8": line - 0.8416 * sd}
