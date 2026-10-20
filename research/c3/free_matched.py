"""C3-QRY arm F -- matched free generation (research/c3/PREREG_C3_QRY.md, section 3).

The model writes with every control token `?` -- nothing asks for a query -- while
the suffix is drawn from the Q:1 state table and HELD, exactly as arm Q's was. A
URL is kept only if it carries a query, as the gate reads it (FIELDS["QRY"]).
Keeping the first `count` query-bearing draws of a suffix group, in draw order,
is an exact sample from p(url | suffix, <Q:?>, query present), whatever the batch
sizes: they set only the cost.

    round r of a group draws  m * 2**min(r, doubling_cap) * (URLs still needed)
    with seed + group index + stride * r;  after max_rounds rounds -> FreeUnmet

The structure mirrors uie.presence's held mode step for step -- suffixes drawn
first, grouped and sorted, each group resampled, the result shuffled with the
same generator -- so the verifier can find every URL's place (V3).
"""
import collections
import random
from typing import List, Optional

from uie.evaluation import FIELDS
from uie.model import generate

__all__ = ["FreeUnmet", "generate_free_matched"]


class FreeUnmet(RuntimeError):
    """A suffix group found too few query-bearing draws in `max_rounds` rounds
    (K1). Raised, never papered over with a short result."""


def generate_free_matched(model, tokenizer, *, label: str, n: int, table, seed: int,
                          m: int, max_rounds: int = 100, doubling_cap: int = 10,
                          stride: int = 1_000_003, stats: Optional[dict] = None,
                          frame="domain", batch_size: int = 32, **kwargs) -> List[str]:
    """`n` free (`<Q:?>`) URLs of class `label` that carry a query, the suffix
    of each drawn from `table` and held. Takes `generate`'s decoding arguments
    (decoder, decoder_kwargs, field_bias, device, ...); a presence control or
    masking is refused -- arm F sets no control."""
    for bad in ("fields", "enforce", "suffix", "seed", "n"):
        if bad in kwargs:
            raise ValueError(f"arm F does not take {bad!r}")
    rng = random.Random(seed)
    drawn = table.sample(label, n, rng)
    if isinstance(frame, str):
        frames = [frame] * n
    else:
        names = sorted(frame)
        frames = rng.choices(names, weights=[frame[f] for f in names], k=n)
    groups = collections.Counter(zip(frames, drawn))

    out: List[str] = []
    tally = {"draws": 0, "with_query": 0, "round0_draws": 0, "round0_with_query": 0,
             "rounds": 0, "by_suffix": {}}
    for i, ((fr, suffix), count) in enumerate(sorted(groups.items())):
        kept: List[str] = []
        r = 0
        while len(kept) < count:
            if r >= max_rounds:
                raise FreeUnmet(f"suffix {suffix!r}: {count - len(kept)} of {count} {label} "
                                f"URLs without a query after {max_rounds} rounds")
            k = m * 2 ** min(r, doubling_cap) * (count - len(kept))
            urls = generate(model, tokenizer, label=label, n=k, suffix=suffix,
                            seed=seed + i + stride * r, frame=fr, batch_size=batch_size,
                            fields=None, **kwargs)
            hits = [u for u in urls if FIELDS["QRY"](u)]
            tally["draws"] += len(urls)
            tally["with_query"] += len(hits)
            if r == 0:
                tally["round0_draws"] += len(urls)
                tally["round0_with_query"] += len(hits)
                by = tally["by_suffix"].setdefault(suffix, [0, 0])
                by[0] += len(urls)
                by[1] += len(hits)
            kept.extend(hits[:count - len(kept)])
            r += 1
        tally["rounds"] = max(tally["rounds"], r)
        out.extend(kept)
    rng.shuffle(out)                    # the order held mode returns
    if stats is not None:
        stats.update(tally)
    return out
