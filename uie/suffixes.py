"""
uie/suffixes.py

Per-class public-suffix frequency tables, and generation that draws from them.

Why the suffix is supplied rather than generated
------------------------------------------------
DESIGN.md 4.5 has always said the suffix is sampled from an EXTERNAL,
updateable table: that is what lets a TLD registered after training be used
without retraining, since the model reads the suffix as characters and can
condition on its spelling. The tokenizer emits `[SUF]` before `[LBL]` precisely
so the label is invented knowing the TLD.

The corpus audit turned that from a nicety into a requirement. Benign block
sampling is suffix-stratified at alpha = 0.5, which deliberately flattens the
suffix marginal — `.com` from ~42% of the index down to ~10% — to maximise the
data available per suffix conditional. That is the right trade for LEARNING
P(label | suffix), and it makes the model's own marginal over suffixes wrong on
purpose. Letting the model sample its own suffix would ship that distortion into
the output and, worse, hand a discriminator a tell that is an artifact of a
sampling decision rather than a flaw in generation.

Why the table is per class
--------------------------
Phishing and benign do not name on the same TLDs, and the difference is real
rather than an artifact:

    benign  top suffixes   com, de, org, nl, net, fr, ru, co.uk, pl, it
    phishing top suffixes  com, io, app, dev, top, cc, xyz, vip, com.au, shop

Total variation between the two distributions is 0.579 on the E0 corpus. A
single shared table would erase a genuine property of phishing — cheap and
novelty gTLDs — and replace it with a benign-shaped prior.

Which distribution the default table carries
--------------------------------------------
The TRAINING CORPUS's, per class. Not "the web's": the benign corpus is
stratified and its suffix marginal is known to be flattened. Using the corpus's
own distribution keeps generation matched to the held-out data it is scored
against, which is what makes the evaluation interpretable. A caller who wants
web-realistic output supplies their own table — which is exactly the external,
updateable table 4.5 describes.
"""
from __future__ import annotations

import collections
import gzip
import json
import random
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence

from uie.factorization import parse

__all__ = ["SuffixTable", "generate_with_suffixes"]


class SuffixTable:
    """Per-class public-suffix frequencies, sampled at generation time."""

    def __init__(self, counts: Dict[str, Dict[str, int]]) -> None:
        self.counts = {cls: dict(c) for cls, c in counts.items()}

    # --- construction -----------------------------------------------------

    @classmethod
    def from_urls(cls, rows: Iterable[dict]) -> "SuffixTable":
        counts: Dict[str, collections.Counter] = collections.defaultdict(
            collections.Counter)
        for row in rows:
            suffix = parse(row["url"]).public_suffix
            if suffix:
                counts[row.get("label", "benign")][suffix] += 1
        return cls({k: dict(v) for k, v in counts.items()})

    @classmethod
    def from_files(cls, paths: Sequence[Path]) -> "SuffixTable":
        def rows():
            for path in paths:
                if not Path(path).is_file():
                    continue
                opener = gzip.open if str(path).endswith(".gz") else open
                with opener(path, "rt", encoding="utf-8") as fh:
                    for line in fh:
                        yield json.loads(line)
        return cls.from_urls(rows())

    # --- sampling ---------------------------------------------------------

    def classes(self) -> List[str]:
        return sorted(self.counts)

    def sample(self, label: str, n: int, rng: random.Random) -> List[str]:
        table = self.counts.get(label)
        if not table:
            raise KeyError(f"no suffixes for class {label!r}; "
                           f"known: {self.classes()}")
        keys = sorted(table)
        weights = [table[k] for k in keys]
        return rng.choices(keys, weights=weights, k=n)

    # --- persistence ------------------------------------------------------

    def save(self, path: Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {cls: dict(sorted(c.items(), key=lambda kv: (-kv[1], kv[0])))
                   for cls, c in self.counts.items()}
        path.write_text(json.dumps(payload, indent=1) + "\n", encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> "SuffixTable":
        return cls(json.loads(Path(path).read_text(encoding="utf-8")))

    def summary(self, label: str, top: int = 10) -> str:
        table = self.counts.get(label, {})
        total = sum(table.values()) or 1
        head = sorted(table.items(), key=lambda kv: -kv[1])[:top]
        return (f"{label}: {len(table):,} suffixes; " +
                ", ".join(f"{s} {100*n/total:.1f}%" for s, n in head))


#: Share of generated rows drawn in each frame, when a mixture is requested.
#: Defaults to the DOMAIN frame alone because that is what the generator
#: structurally samples -- one URL per invented host, never revisited -- and it
#: is the frame `run_eval` scores in (`evaluation.GENERATOR_FRAME`). Ask for
#: `page` explicitly to generate a URL as it would appear on a multi-page site.
DEFAULT_FRAME_MIX = {"domain": 1.0}


def generate_with_suffixes(model, tokenizer, *, label: str, n: int,
                           table: SuffixTable, seed: int = 20260806,
                           # ADOPTED 2026-09-23, was "typical" -- see
                           # research/qry/R4_RESULTS.md
                           decoder: str = "temperature",
                           frame="domain",
                           logit_bias=None, field_bias=None,
                           batch_size: int = 32, device=None,
                           **kwargs) -> List[str]:
    """Generate `n` URLs, each with a suffix drawn from `table`.

    `generate` forces ONE suffix per call, so samples are grouped by the suffix
    drawn for them and each group generated together. Grouping also keeps the
    batched path intact — a per-row suffix would grow rows at different rates
    and break the rectangular tensor, which is the bug that produced AUC 1.0000
    when the suffix was prepended instead of injected.
    """
    from uie.model import generate

    rng = random.Random(seed)
    drawn = table.sample(label, n, rng)

    # A frame may be a single name or a mixture {frame: share}. Grouping by
    # (frame, suffix) keeps each `generate` call homogeneous in both supplied
    # controls, which is what the batched path requires.
    if isinstance(frame, str):
        frames = [frame] * n
    else:
        names = sorted(frame)
        weights = [frame[f] for f in names]
        frames = rng.choices(names, weights=weights, k=n)

    groups: Dict[tuple, int] = collections.Counter(zip(frames, drawn))

    out: List[str] = []
    for i, ((fr, suffix), count) in enumerate(sorted(groups.items())):
        out.extend(generate(model, tokenizer, label=label, n=count,
                            suffix=suffix, decoder=decoder, seed=seed + i,
                            frame=fr, logit_bias=logit_bias,
                            field_bias=field_bias,
                            batch_size=batch_size, device=device, **kwargs))
    rng.shuffle(out)
    return out
