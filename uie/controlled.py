"""
uie/controlled.py

Controlled generation as adopted from SC1 (research/sc1/PREREG_SC1.md,
research/sc1/SC1_RESULTS.md; adopted by the owner 2026-09-25).

Since 2026-09-26 the resampling step is a reference instrument, not the production
mechanism (research/native/NATIVE_REPAIR_PLAN.md section 0, the native principle).
E9's `e9_ln` control mode (research/native/DECISION_2026-09-28.md) keeps SC1's
state tables and draws natively: `uie.baseline.generate_control`. The engine
adopted on 2026-10-08, `e16_b5`, instead writes to a supplied presence-slash
plan drawn per state (research/e17/e17_common.py `control_urls`).

A presence request -- a path required or forbidden, a query required or
forbidden -- is served in two steps:

  1. the suffix is drawn from the REQUESTED STATE's table: public-suffix counts
     over the class's domain-frame training rows (name_train) that are in that
     state, counted exactly as `SuffixTable.from_urls` counts them;
  2. that suffix is HELD while presence is enforced by resampling
     (`uie.presence`, held mode): the model redraws the URL, never the suffix.

Why. The shipped table is the class marginal over every *_train row, both
frames, and it is drawn before and without the control; resampling can tilt it
only as far as the control fails. CR1 measured the consequence -- SUF tells in
every state whose real TLD mix departs from the class's -- and SC1 the repair:
SUF cleared in all 14 measurable states, and the phishing name tells fell with
it.

What does not change. With no presence control nothing here applies: the gate's
path is `uie.suffixes.generate_with_suffixes` with the shipped table, and
`uie.presence.generate_with_presence` keeps its own defaults, so HC2's and CR1's
runs stay reproducible. `generate_controlled` refuses a request with no control
rather than guess which path was meant.

    python -m uie.controlled --bundle data/bundle2 --out data/state_suffix_tables.json
"""
from __future__ import annotations

import argparse
import collections
import gzip
import hashlib
import json
from pathlib import Path
from typing import Dict, Iterable, Iterator, Mapping, Optional

from uie.evaluation import FIELDS
from uie.presence import generate_with_presence
from uie.suffixes import SuffixTable

__all__ = ["SOURCES", "StateSuffixTables", "generate_controlled", "state_code"]

#: The files a state table is counted from: the domain frame, the frame the
#: generator samples and the population SUF is scored against (SC1 section 3.1).
SOURCES = {"benign": "name_train", "phishing": "phishing_name_train"}

#: The fields a presence control can name, in the order a state code lists them.
CONTROL_FIELDS = ("PTH", "QRY")


def state_code(fields: Optional[Mapping[str, Optional[bool]]]) -> str:
    """{"PTH": True} -> "P1"; {"PTH": True, "QRY": False} -> "P1_Q0".

    A None value leaves that field free. At least one control must be set.
    """
    wanted = {f: bool(v) for f, v in (fields or {}).items() if v is not None}
    unknown = sorted(set(wanted) - set(CONTROL_FIELDS))
    if unknown:
        raise ValueError(f"no presence control for {unknown}; known: {list(CONTROL_FIELDS)}")
    if not wanted:
        raise ValueError("no presence control is set")
    return "_".join(f"{f[0]}{int(wanted[f])}" for f in CONTROL_FIELDS if f in wanted)


def _read_rows(path: Path) -> Iterator[dict]:
    with gzip.open(path, "rt", encoding="utf-8") as fh:
        for line in fh:
            yield json.loads(line)


class StateSuffixTables:
    """Per-class, per-state public-suffix counts: counts[label][code][suffix]."""

    def __init__(self, counts: Mapping[str, Mapping[str, Mapping[str, int]]]) -> None:
        self.counts = {label: {code: dict(c) for code, c in by_code.items()}
                       for label, by_code in counts.items()}

    # --- construction -----------------------------------------------------

    @classmethod
    def from_rows(cls, rows: Iterable[dict]) -> "StateSuffixTables":
        """Count every row under the three codes it satisfies: its path state,
        its query state and both together. A row counts as `SuffixTable.from_urls`
        counts it: its parsed public suffix, empty suffixes skipped, under its
        `label`. Presence is the gate's own reading, FIELDS on the URL string."""
        counts: Dict[str, Dict[str, collections.Counter]] = collections.defaultdict(
            lambda: collections.defaultdict(collections.Counter))
        for row in rows:
            url = row["url"]
            suffix = FIELDS["SUF"](url)
            if not suffix:
                continue
            p, q = int(bool(FIELDS["PTH"](url))), int(bool(FIELDS["QRY"](url)))
            by_code = counts[row.get("label", "benign")]
            for code in (f"P{p}", f"Q{q}", f"P{p}_Q{q}"):
                by_code[code][suffix] += 1
        return cls({label: {code: dict(c) for code, c in by_code.items()}
                    for label, by_code in counts.items()})

    @classmethod
    def build(cls, bundle: Path, sources: Mapping[str, str] = SOURCES) -> "StateSuffixTables":
        """The adopted tables, from a bundle's domain-frame training files. A row
        whose label is not its file's class is refused, not recounted."""
        def rows():
            for label, stem in sources.items():
                for row in _read_rows(Path(bundle) / f"{stem}.jsonl.gz"):
                    if row.get("label", "benign") != label:
                        raise ValueError(f"{stem}: a row labelled {row.get('label')!r}")
                    yield row
        return cls.from_rows(rows())

    # --- use --------------------------------------------------------------

    def for_state(self, label: str, fields: Mapping[str, Optional[bool]]) -> SuffixTable:
        """The table a request for (label, fields) draws its suffix from."""
        code = state_code(fields)
        counts = self.counts.get(label, {}).get(code)
        if counts is None:
            raise KeyError(f"no table for {label} {code}; known: "
                           f"{ {k: sorted(v) for k, v in self.counts.items()} }")
        if not counts:
            raise ValueError(f"the table for {label} {code} is empty")
        return SuffixTable({label: counts})

    def states(self, label: str):
        return sorted(self.counts.get(label, {}))

    # --- persistence ------------------------------------------------------

    def to_bytes(self) -> bytes:
        """One byte-exact serialisation, so a sha256 names the tables."""
        return (json.dumps(self.counts, sort_keys=True, indent=1) + "\n").encode("utf-8")

    def sha256(self) -> str:
        return hashlib.sha256(self.to_bytes()).hexdigest()

    def save(self, path: Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(self.to_bytes())

    @classmethod
    def load(cls, path: Path) -> "StateSuffixTables":
        return cls(json.loads(Path(path).read_text(encoding="utf-8")))


def generate_controlled(model, tokenizer, *, label: str, n: int,
                        state_tables: StateSuffixTables,
                        fields: Mapping[str, Optional[bool]], seed: int = 20260806,
                        max_rounds: int = 100, stats: Optional[dict] = None,
                        frame="domain", batch_size: int = 32, **kwargs):
    """`n` URLs of class `label` honouring `fields`: the suffix drawn from the
    requested state's table and held while presence is enforced by resampling.

    Exactly SC1 arm T's call. Takes `generate_with_suffixes`'s generation
    arguments (decoder, decoder_kwargs, field_bias, device, ...). Deterministic
    for a seed; raises `uie.presence.PresenceUnmet` rather than return a URL
    that breaks its control, and ValueError when no control is set.
    """
    table = state_tables.for_state(label, fields)
    return generate_with_presence(model, tokenizer, label=label, n=n, table=table,
                                  fields=dict(fields), seed=seed, hold_suffix=True,
                                  max_rounds=max_rounds, stats=stats, frame=frame,
                                  batch_size=batch_size, **kwargs)


def main() -> None:
    ap = argparse.ArgumentParser(description="Build the adopted state suffix tables.")
    ap.add_argument("--bundle", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    tables = StateSuffixTables.build(args.bundle)
    for label in sorted(tables.counts):
        for code in tables.states(label):
            counts = tables.counts[label][code]
            print(f"  {label:<9} {code:<6} rows {sum(counts.values()):>9,}  "
                  f"suffixes {len(counts):>5}")
    tables.save(args.out)
    print(f"wrote {args.out}  sha256 {tables.sha256()}")


if __name__ == "__main__":
    main()
