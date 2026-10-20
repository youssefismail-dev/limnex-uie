"""
uie/engine.py

The adopted engine's free-mode decode, in the library (D2 step 4, research/d2/D2_PLAN.md).

`e16_b5` (uie.baseline.E16_B5) writes every URL to a SUPPLIED presence-slash plan: before a URL
is generated, its state -- whether it has a path, a query, and a root slash -- is drawn, and the
URL is generated under that state with its suffix drawn from the state's own table. Until D2 this
procedure lived only in the experiment layer (research/native/e12_common.py `supplied_tables`,
research/e13/e13_common.py `draw_and_generate`, called by research/e17/e17_common.py
`free_urls` with B5 on). It is carried here unchanged so the public interface needs nothing from
research/; tests/test_api.py proves this copy returns exactly what the experiment layer returns.

The two tables it reads are statistics of the training corpus, shipped in uie/assets/ and pinned
by sha256: `suffix_table.json` (suffix counts per class) and `presence_slash_table.json` (per
class and suffix, how often each of the six presence-slash states occurs). They hold no URL.
"""
from __future__ import annotations

import collections
import random
from typing import Dict, List, Mapping, Sequence

from uie.baseline import E16_B5
from uie.suffixes import SuffixTable, generate_with_suffixes

__all__ = ["PS_STATES", "STATE_SEED", "fields_of", "plan_tables", "generate_free"]

#: The six presence-slash states (E12 D2): P = path, Q = query, S = root slash.
PS_STATES = ("P0Q0S0", "P0Q0S1", "P0Q1S0", "P0Q1S1", "P1Q0S1", "P1Q1S1")
#: Each state's generator is seeded at seed + STATE_SEED * (its index + 1) (E11 D6).
STATE_SEED = 10_000_000


def fields_of(state: str) -> Dict[str, bool]:
    """A presence-slash state as a supplied plan, e.g. P0Q1S0 -> PTH off, QRY on, SLASH off."""
    return {"PTH": state[1] == "1", "QRY": state[3] == "1", "SLASH": state[5] == "1"}


def plan_tables(presence_slash: Mapping[str, Mapping[str, Sequence[int]]],
                shipped: Mapping[str, Mapping[str, float]], label: str) -> Dict[str, Dict[str, float]]:
    """Per state k, the suffix weights w_k(s) = shipped(s) * n(s, k) / n(s) (E12 D2, free mode)."""
    n = presence_slash[label]
    out: Dict[str, Dict[str, float]] = {k: {} for k in PS_STATES}
    for s, c in shipped[label].items():
        row = n.get(s)
        if not row:
            raise KeyError(f"{label} suffix {s!r} is in the shipped table but has no row")
        tot = sum(row)
        for i, k in enumerate(PS_STATES):
            if row[i]:
                out[k][s] = c * row[i] / tot
    return out


def generate_free(model, tokenizer, *, label: str, n: int, seed: int,
                  presence_slash: Mapping[str, Mapping[str, Sequence[int]]],
                  shipped: Mapping[str, Mapping[str, float]], device=None) -> List[str]:
    """`n` free-mode URLs of class `label` from e16_b5, deterministic for a seed: the state counts
    from one multinomial draw at `seed`; each state generated with its suffix drawn from its own
    table at seed + STATE_SEED * (i + 1); the URLs shuffled by the same generator. Decoded with
    E16_B5's rules (temperature 1.0, D1, D1', B5)."""
    tables = plan_tables(presence_slash, shipped, label)
    states = [k for k in tables]
    rng = random.Random(seed)
    counts = collections.Counter(rng.choices(states, weights=[sum(tables[k].values()) for k in states], k=n))
    urls: List[str] = []
    for i, k in enumerate(states):
        if not counts[k]:
            continue
        urls.extend(generate_with_suffixes(
            model, tokenizer, label=label, n=counts[k], table=SuffixTable({label: tables[k]}),
            seed=seed + STATE_SEED * (i + 1), device=device, fields=fields_of(k),
            **E16_B5.decode_kwargs()))
    rng.shuffle(urls)
    return urls
