"""
uie/presence.py

HC2 (research/hc2/PREREG_HC2.md): presence controls, enforced by RESAMPLING.

E7's presence controls are learned, and sampling can still end a URL before a
required field (E7 C1: 76.7% on phishing <Q:1>). Masking the forbidden tokens
holds the rules, but can force a URL the model wants to end into a runaway
(HC1 H1). Resampling forces nothing: a URL that fails its control is drawn
again, so every output is a draw from the model's own conditional. That is
rejection sampling -- exact by construction, at 1/P(control holds) draws per
kept URL.

Two modes, differing only in what a redraw keeps:

    joint (default)  redraw the suffix AND the URL, the suffix again from the
                     table: exact for p(suffix, url | control holds). Presence
                     shifts the real TLD mix (phishing with a query: TV 0.35
                     from the class marginal), and only this mode can follow it.
    held suffix      redraw the URL with the slot's own suffix: exact for
                     p(url | suffix, control holds) -- right when the caller
                     has fixed the suffix.

Presence is the gate's own definition: `uie.evaluation.FIELDS` on the rendered
URL. With no control set, this IS `generate_with_suffixes`, byte for byte.
Nothing here is imported by the gate.
"""
import collections
import random
from typing import Dict, List, Optional

from uie.evaluation import FIELDS
from uie.suffixes import SuffixTable, generate_with_suffixes

__all__ = ["PresenceUnmet", "honours", "generate_with_presence", "ROUND_STRIDE"]

#: Seed offset between redraw rounds -- the stride HC1 H3's exact arm used.
ROUND_STRIDE = 1_000_003


class PresenceUnmet(RuntimeError):
    """No honouring draw within `max_rounds`. Raised, never papered over: a
    silent violating URL is the one outcome a presence control must not have."""


def honours(url: str, fields: Dict[str, Optional[bool]]) -> bool:
    """Does `url` honour every control that is set, as the gate reads it?"""
    return all(bool(FIELDS[f](url)) == want
               for f, want in fields.items() if want is not None)


def generate_with_presence(model, tokenizer, *, label: str, n: int,
                           table: SuffixTable,
                           fields: Optional[Dict[str, Optional[bool]]] = None,
                           seed: int = 20260806, hold_suffix: bool = False,
                           max_rounds: int = 100, stats: Optional[dict] = None,
                           frame="domain", batch_size: int = 32,
                           **kwargs) -> List[str]:
    """`n` URLs of class `label` that honour `fields`, drawn by rejection.

    Takes the arguments of `generate_with_suffixes` (decoder, decoder_kwargs,
    field_bias, device, ...). Round k of redraws uses seed + ROUND_STRIDE*k, so
    a request is reproducible. `stats`, if given, receives the draw count, the
    first pass's honouring count and the rounds used. Raises PresenceUnmet
    rather than return a URL that breaks its control.
    """
    if "enforce" in kwargs:
        raise ValueError("resampling does not combine with masking (`enforce`)")
    wanted = {f: bool(v) for f, v in (fields or {}).items() if v is not None}
    if not wanted:                      # nothing to enforce: the gate's own path
        return generate_with_suffixes(model, tokenizer, label=label, n=n, table=table,
                                      seed=seed, frame=frame, batch_size=batch_size,
                                      fields=fields, **kwargs)
    if hold_suffix:
        out, tally = _held(model, tokenizer, label, n, table, fields, wanted, seed,
                           max_rounds, frame, batch_size, kwargs)
    else:
        out, tally = _joint(model, tokenizer, label, n, table, fields, wanted, seed,
                            max_rounds, frame, batch_size, kwargs)
    if stats is not None:
        stats.update(tally, mode="held" if hold_suffix else "joint")
    return out


def _joint(model, tokenizer, label, n, table, fields, wanted, seed, max_rounds,
           frame, batch_size, kwargs):
    """Every failing slot is replaced by a fresh (suffix, url) draw -- the whole
    generator again -- until it honours."""
    def draw(k: int, count: int) -> List[str]:
        return generate_with_suffixes(model, tokenizer, label=label, n=count,
                                      table=table, seed=seed + ROUND_STRIDE * k,
                                      frame=frame, batch_size=batch_size,
                                      fields=fields, **kwargs)

    out = draw(0, n)
    todo = [i for i, u in enumerate(out) if not honours(u, wanted)]
    tally = {"draws": n, "first_pass": n - len(todo), "rounds": 0}
    k = 0
    while todo and k < max_rounds:
        k += 1
        still = []
        for pos, url in zip(todo, draw(k, len(todo))):
            if honours(url, wanted):
                out[pos] = url
            else:
                still.append(pos)
        tally["draws"] += len(todo)
        tally["rounds"] = k
        todo = still
    if todo:
        raise PresenceUnmet(f"{len(todo)} of {n} {label} URLs found no draw honouring "
                            f"{wanted} in {max_rounds} rounds")
    return out, tally


def _held(model, tokenizer, label, n, table, fields, wanted, seed, max_rounds,
          frame, batch_size, kwargs):
    """`generate_with_suffixes`, step for step, except that each suffix group is
    redrawn -- same suffix, seed + i + ROUND_STRIDE*k -- until it honours: the
    algorithm of HC1 H3's exact arm."""
    from uie.model import generate

    rng = random.Random(seed)
    drawn = table.sample(label, n, rng)
    if isinstance(frame, str):
        frames = [frame] * n
    else:
        names = sorted(frame)
        frames = rng.choices(names, weights=[frame[f] for f in names], k=n)
    groups = collections.Counter(zip(frames, drawn))

    out: List[str] = []
    tally = {"draws": 0, "first_pass": 0, "rounds": 0}
    for i, ((fr, suffix), count) in enumerate(sorted(groups.items())):
        kept: List[str] = []
        k = 0
        while len(kept) < count and k <= max_rounds:
            urls = generate(model, tokenizer, label=label, n=count - len(kept),
                            suffix=suffix, seed=seed + i + ROUND_STRIDE * k,
                            frame=fr, batch_size=batch_size, fields=fields, **kwargs)
            ok = [u for u in urls if honours(u, wanted)]
            tally["draws"] += len(urls)
            if k == 0:
                tally["first_pass"] += len(ok)
            kept.extend(ok)
            k += 1
        tally["rounds"] = max(tally["rounds"], k - 1)
        if len(kept) < count:
            raise PresenceUnmet(f"suffix {suffix!r}: {count - len(kept)} of {count} "
                                f"{label} URLs found no draw honouring {wanted} "
                                f"in {max_rounds} rounds")
        out.extend(kept[:count])
    rng.shuffle(out)                    # the order generate_with_suffixes returns
    return out, tally
