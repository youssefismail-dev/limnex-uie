"""QC1 -- shared definitions (research/qc1/PREREG_QC1.md). Loaders, the paired
harness, the near-miss index and the registered rules, as pure functions where
they can be, so the tests pin them.

Every instrument is an existing one: the gate's detector (`real_vs_fake_auc`)
inside the Q3 harness's three-way real split with its floor, and Q3.1's ladder.
The only new code arranges draws (paired conditions, a stratified re-draw) and
counts strings.
"""
from __future__ import annotations

import collections
import gc
import gzip
import json
import math
import random
import statistics as st
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
for p in (ROOT, ROOT / "research" / "native", ROOT / "research" / "cr1",
          ROOT / "research" / "dc1", ROOT / "research" / "qry"):
    sys.path.insert(0, str(p))

BUNDLE = ROOT / "data" / "bundle2"
GEN = ROOT / "data" / "e11_final" / "gen"
TRAIN = ROOT / "data" / "e11" / "train"
OUT = ROOT / "data" / "qc1"
SIZING = ROOT / "data" / "e12" / "sizing.json"

PRIMARY, REPLICATION = "e11_plan", "e11_ref"
ARMS = (PRIMARY, REPLICATION)
POOL_SEED = 20260806                         # the gate's seed: one URL per host
D = 12                                       # Q3.1's split seeds
SEEDS = tuple(20260806 + i for i in range(D))


def _tcrit(df: int) -> float:
    from scipy import stats
    return float(stats.t.ppf(0.975, df))


TCRIT = _tcrit(D - 1)                        # 2.2010, as Q3.1 computes it
N_REQ = 1800                                 # the Q3 harness default
FREQUENT = 5                                 # QC1-4: a frequent training string
BANDS = ("1", "2-4", "5-19", "20+")
FLOOR_LOW, FLOOR_HIGH = 0.42, 0.55           # Q3's floor bounds (V4)

#: The E3 query cells (mode, label, state). Phishing's control query states are
#: not measurable by the gate (e11_common.UNMEASURABLE).
CELLS = (("free", "benign", None), ("free", "phishing", None),
         ("control", "benign", "P0Q1"), ("control", "benign", "P1Q1"))


def cell_name(mode: str, label: str, state: Optional[str]) -> str:
    return f"{mode}_{label}" + (f"_{state}" if state else "")


# --------------------------------------------------------------------------- #
# Data
# --------------------------------------------------------------------------- #

def _jsonl(name: str) -> List[str]:
    with gzip.open(BUNDLE / f"{name}.jsonl.gz", "rt", encoding="utf-8") as fh:
        return [json.loads(line)["url"] for line in fh]


def query(url: str) -> str:
    from uie.evaluation import FIELDS
    return FIELDS["QRY"](url)


def registrable(url: str) -> str:
    from uie.factorization import parse
    p = parse(url)
    return f"{p.registrable_label}.{p.public_suffix}" if p.registrable_label else ""


def real_pool(label: str, state: Optional[str] = None) -> List[str]:
    """The gate's own real pool for the cell, one URL per host (the gate's seed),
    URLs with a query only. Free: `name_val`; control: CR1's state pool."""
    from uie.evaluation import _one_per_host
    if state is None:
        urls = _jsonl(("" if label == "benign" else f"{label}_") + "name_val")
    else:
        import e11_common as K
        from cr1_score import pools
        urls = pools(BUNDLE, label, K.STATES[state])["name"]
    return [u for u in _one_per_host(urls, POOL_SEED) if query(u)]


def generated(arm: str, label: str, state: Optional[str] = None, gen: Path = GEN) -> List[str]:
    """The cell's generated URLs with a query, the five generations pooled."""
    import e11_common as K
    out = []
    for g in range(K.G):
        name = (f"free_{arm}_{label}_g{g}.json" if state is None
                else f"ctrl_{arm}_{label}_{state}_g{g}.json")
        out += [u for u in json.loads((gen / name).read_text(encoding="utf-8")) if query(u)]
    return out


def train_counts(label: str) -> collections.Counter:
    """Every training query of the class with its count: the K11d set
    (`dc1_compare.training_queries`: name_train + structure_train)."""
    pre = "" if label == "benign" else f"{label}_"
    c = collections.Counter()
    for kind in ("name", "structure"):
        for u in _jsonl(f"{pre}{kind}_train"):
            q = query(u)
            if q:
                c[q] += 1
    return c


def train_domains(label: str) -> set:
    pre = "" if label == "benign" else f"{label}_"
    return {registrable(u) for kind in ("name", "structure") for u in _jsonl(f"{pre}{kind}_train")}


# --------------------------------------------------------------------------- #
# Rates and rules
# --------------------------------------------------------------------------- #

def tolerance(p: float, n: float) -> float:
    """The owner's standard with E11's edge rule: T = 2 sqrt(max(p(1-p), 1/n) / n)."""
    return 2 * math.sqrt(max(p * (1 - p), 1 / n) / n)


def band(count: int) -> str:
    return "1" if count == 1 else "2-4" if count < 5 else "5-19" if count < 20 else "20+"


def rate_reading(r_g: float, r_n: float, n: float) -> Tuple[str, float]:
    """QC1-1: UNDER / WITHIN / OVER against the natural rate, and T."""
    t = tolerance(r_n, n)
    return ("UNDER" if r_g < r_n - t else "OVER" if r_g > r_n + t else "WITHIN"), t


def copy_profile(queries: Sequence[str], counts: collections.Counter) -> dict:
    """Copy rate, singleton rate and the band profile, as shares of `queries`."""
    n = len(queries)
    found = [counts[q] for q in queries if q in counts]
    bands = collections.Counter(band(c) for c in found)
    return {"n": n, "rate": len(found) / n if n else None,
            "singleton": bands["1"] / n if n else None,
            "bands": {b: bands[b] / n if n else None for b in BANDS}}


def paired_t(diffs: Sequence[float]) -> Tuple[float, float, float]:
    """Mean, SE and t of paired differences (df = len - 1)."""
    m = st.mean(diffs)
    sd = st.stdev(diffs) if len(diffs) > 1 else 0.0
    se = sd / math.sqrt(len(diffs)) if sd else 0.0
    return m, se, (m / se if se else float("nan"))


def mix_reading(x_plain: Sequence[float], x_mix: Sequence[float]) -> dict:
    """QC1-2 (registered): rho = mean x(ii) / mean x(i). MIX-LED if rho <= 0.5 and
    the paired difference x(i) - x(ii) has t > t_crit; CONTENT-LED if rho >= 0.8
    or the difference is not significant; SPLIT otherwise."""
    rho = st.mean(x_mix) / st.mean(x_plain) if st.mean(x_plain) else float("nan")
    m, se, t = paired_t([a - b for a, b in zip(x_plain, x_mix)])
    sig = t == t and t > TCRIT
    reading = ("MIX-LED" if rho <= 0.5 and sig else
               "CONTENT-LED" if rho >= 0.8 or not sig else "SPLIT")
    return {"rho": rho, "diff": m, "se": se, "t": t, "reading": reading}


def leak_reading(x_real: Sequence[float], x_gen: Sequence[float]) -> dict:
    """QC1-5 (registered): LEAK if x_real - x_gen > t_crit * SE (paired) and the
    relative gap (x_real - x_gen) / x_real >= 0.2."""
    m, se, t = paired_t([a - b for a, b in zip(x_real, x_gen)])
    rel = m / st.mean(x_real) if st.mean(x_real) else float("nan")
    return {"gap": m, "se": se, "t": t, "relative": rel,
            "reading": "LEAK" if (t == t and t > TCRIT and rel >= 0.2) else "NO LEAK"}


def frame_leak_reading(x_real: Sequence[float], x_gen: Sequence[float]) -> dict:
    """QC1-9 (amendment 2): FRAME-LEAK if x_real >= 0.05 (the frames differ) and
    QC1-5's rule holds with the page pool as the floor: x_real - x_gen > t_crit *
    SE (paired) and the relative gap >= 0.2 (the replication is read by the reader)."""
    r = leak_reading(x_real, x_gen)
    informative = st.mean(x_real) >= 0.05
    r["reading"] = ("FRAME-LEAK" if informative and r["reading"] == "LEAK" else
                    "NO FRAME-LEAK" if informative else "UNINFORMATIVE (frames alike)")
    return r


def budget_reading(x_all: Sequence[float], x_fit: Sequence[float]) -> dict:
    """QC1-6 (registered): S = 1 - mean x(fits) / mean x(all); BUDGET if S >= 0.25
    with the paired difference significant."""
    s = 1 - st.mean(x_fit) / st.mean(x_all) if st.mean(x_all) else float("nan")
    m, se, t = paired_t([a - b for a, b in zip(x_all, x_fit)])
    return {"S": s, "diff": m, "se": se, "t": t,
            "reading": "BUDGET" if (s >= 0.25 and t == t and t > TCRIT) else "NO BUDGET"}


def blending_reading(k_g: int, n_g: int, k_r: int, n_r: int) -> dict:
    """QC1-4 (registered): BLENDING if m_g - m_r >= 0.05 with a two-proportion
    95% CI excluding 0 (the replication is read by the reader)."""
    m_g, m_r = k_g / n_g, k_r / n_r
    se = math.sqrt(m_g * (1 - m_g) / n_g + m_r * (1 - m_r) / n_r)
    d = m_g - m_r
    lo, hi = d - 1.96 * se, d + 1.96 * se
    return {"m_g": m_g, "m_r": m_r, "diff": d, "lo": lo, "hi": hi,
            "reading": "BLENDING" if d >= 0.05 and lo > 0 else "NO BLENDING"}


# --------------------------------------------------------------------------- #
# Amendment 1: QC1-7 (the slash) and QC1-8 (state blindness)
# --------------------------------------------------------------------------- #

BOOT = 1000                                  # bootstrap resamples over rows
BOOT_SEED = 20260930


def _spearman(x: Sequence[float], y: Sequence[float]) -> float:
    from scipy import stats
    r = stats.spearmanr(x, y).correlation
    return float(r) if r == r else 0.0


def _boot_indices(n: int):
    """BOOT resamples of n rows, seeded, in chunks (numpy; memory-bounded)."""
    import numpy as np
    rng = np.random.default_rng(BOOT_SEED)
    for _ in range(BOOT // 100):
        yield rng.integers(0, n, size=(100, n))


def _ci(values) -> Tuple[float, float]:
    import numpy as np
    v = np.sort(np.asarray(values, dtype=float))
    return float(v[int(0.025 * len(v))]), float(v[int(0.975 * len(v)) - 1])


def slash_leak_reading(own: Sequence[float], swap: Sequence[float]) -> dict:
    """QC1-7a (registered): SLASH-LEAK if Spearman rho(own, swap) >= 0.5 with the
    bootstrap 95% CI's lower end above 0.3 (the replication is read by the reader).
    lambda = mean own / mean swap is reported."""
    import numpy as np
    o, s = np.asarray(own, dtype=float), np.asarray(swap, dtype=float)
    rho = _spearman(o, s)
    boots = [_spearman(o[i], s[i]) for block in _boot_indices(len(o)) for i in block]
    lo, hi = _ci(boots)
    lam = float(o.mean() / s.mean()) if s.mean() else float("nan")
    return {"rho": rho, "lo": lo, "hi": hi, "lambda": lam, "own": float(o.mean()),
            "swap": float(s.mean()),
            "reading": "SLASH-LEAK" if rho >= 0.5 and lo > 0.3 else "NO SLASH-LEAK"}


def suppressed_reading(own: Sequence[float], data: Sequence[int], swap: Sequence[float]) -> dict:
    """QC1-7b (registered): SUPPRESSED if mean own / data rate <= 0.5 with the
    bootstrap CI of the ratio below 1; LEAK-CONSISTENT if SUPPRESSED and the
    swapped-class mass is lower still (bootstrap CI of mean(swap - own) below 0)."""
    import numpy as np
    o, d, s = (np.asarray(x, dtype=float) for x in (own, data, swap))
    ratio = float(o.mean() / d.mean()) if d.mean() else float("nan")
    rs, ds = [], []
    for block in _boot_indices(len(o)):
        dd, oo, ss = d[block].mean(axis=1), o[block].mean(axis=1), s[block].mean(axis=1)
        rs.extend(np.where(dd > 0, oo / np.where(dd > 0, dd, 1.0), np.inf))
        ds.extend(ss - oo)
    r_lo, r_hi = _ci(rs)
    _, d_hi = _ci(ds)
    sup = ratio <= 0.5 and r_hi < 1
    return {"own": float(o.mean()), "data": float(d.mean()), "swap": float(s.mean()),
            "ratio": ratio, "ratio_lo": r_lo, "ratio_hi": r_hi, "swap_minus_own_hi": d_hi,
            "reading": ("SUPPRESSED, LEAK-CONSISTENT" if sup and d_hi < 0 else
                        "SUPPRESSED" if sup else "NOT SUPPRESSED")}


def welch_t(a: Sequence[float], b: Sequence[float]) -> float:
    """NaN (never significant) when either side has fewer than two valid draws."""
    if len(a) < 2 or len(b) < 2:
        return float("nan")
    va, vb = st.variance(a) / len(a), st.variance(b) / len(b)
    return (st.mean(a) - st.mean(b)) / math.sqrt(va + vb) if va + vb else float("nan")


def state_blind_reading(x_own: Sequence[float], x_cross: Sequence[float],
                        x_states: Sequence[float]) -> dict:
    """QC1-8 (registered): STATE-BLIND if mean x_states >= 0.02 and x_cross <
    x_own with a Welch t over the seeds above t_crit (replication by the reader)."""
    if not (x_own and x_cross and x_states):
        return {"x_own": float("nan"), "x_cross": float("nan"), "x_states": float("nan"),
                "t": float("nan"), "reading": "INSUFFICIENT (every draw VOID)"}
    t = welch_t(x_own, x_cross)
    blind = st.mean(x_states) >= 0.02 and t == t and t > TCRIT
    return {"x_own": st.mean(x_own), "x_cross": st.mean(x_cross), "x_states": st.mean(x_states),
            "t": t, "reading": "STATE-BLIND" if blind else "NOT STATE-BLIND"}


def wilson(k: int, n: int, z: float = 1.96) -> Tuple[float, float]:
    if not n:
        return 0.0, 1.0
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return c - h, c + h


# --------------------------------------------------------------------------- #
# Near misses (QC1-4): one edit from a frequent training string
# --------------------------------------------------------------------------- #

def _deletes(s: str) -> set:
    return {s[:i] + s[i + 1:] for i in range(len(s))}


def levenshtein_le1(a: str, b: str) -> bool:
    if a == b:
        return True
    la, lb = len(a), len(b)
    if abs(la - lb) > 1:
        return False
    if la == lb:
        return sum(x != y for x, y in zip(a, b)) == 1
    if la > lb:
        a, b = b, a
    # b is one longer: b minus one character equals a
    return any(b[:i] + b[i + 1:] == a for i in range(len(b)))


class NearIndex:
    """Symmetric-delete index over the frequent strings: candidates share a
    one-deletion variant (or equal one); each candidate is then checked exactly."""

    def __init__(self, frequent: Iterable[str]) -> None:
        self.keys: Dict[str, List[str]] = collections.defaultdict(list)
        for s in frequent:
            for k in _deletes(s) | {s}:
                self.keys[k].append(s)

    def near(self, q: str) -> bool:
        for k in _deletes(q) | {q}:
            for s in self.keys.get(k, ()):
                if levenshtein_le1(q, s):
                    return True
        return False


# --------------------------------------------------------------------------- #
# The paired harness
# --------------------------------------------------------------------------- #

def auc(real: Sequence[str], fake: Sequence[str], seed: int) -> float:
    from uie.evaluation import real_vs_fake_auc
    a = real_vs_fake_auc(real, fake, seed=seed)["auc_mean"]
    gc.collect()                         # the lesson recorded for loops over the detector
    return a


def real_split(real_q: Sequence[str], n: int, seed: int) -> Tuple[list, list, list]:
    """The Q3 harness's three disjoint real draws for this seed."""
    picked = random.Random(seed).sample(list(real_q), 3 * n)
    return picked[:n], picked[n:2 * n], picked[2 * n:]


def stratified(found: Sequence[str], novel: Sequence[str], share: float, n: int,
               seed: int) -> List[str]:
    """QC1-2 (ii): exactly round(share * n) drawn from `found`, the rest from
    `novel`, shuffled. A diagnostic re-draw of the model's own output."""
    rng = random.Random(seed)
    k = round(share * n)
    out = rng.sample(list(found), k) + rng.sample(list(novel), n - k)
    rng.shuffle(out)
    return out


def feasible_n(n: int, found: int, novel: int, share: float) -> int:
    """The largest n <= `n` whose stratified draw fits the generated strata."""
    while n > 0 and (round(share * n) > found or n - round(share * n) > novel):
        n -= 1
    return n


def harness(real_q: Sequence[str], fakes: Dict[str, List[str]], seed: int, n: int) -> dict:
    """One seed: the floor on real draws a, b, and the excess of every fake set
    against real draw c. Every fake set must already hold n queries."""
    a, b, c = real_split(real_q, n, seed)
    floor = auc(a, b, seed)
    out = {"floor": floor, "n": n}
    for name, f in fakes.items():
        assert len(f) == n, (name, len(f), n)
        out[name] = auc(c, f, seed) - floor
    return out
