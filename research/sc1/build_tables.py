"""SC1 -- build the suffix tables (research/sc1/PREREG_SC1.md, section 3.1).

    T  one table per measurable state: the class's name_train rows in that state
    C  one table per class: all the class's name_train rows

A row counts exactly as `SuffixTable.from_urls` counts it -- its parsed public
suffix, empty suffixes skipped, under its `label` -- the construction the design
review showed rebuilds the shipped table from all *_train rows (section 2.1). A
row's state is the gate's own reading: FIELDS on the URL string. Raw counts: no
smoothing, no minimum count, no pruning. The row totals must equal the ones the
design review registered, or nothing is written.

    python research/sc1/build_tables.py --bundle data/bundle2 --out data/sc1/tables.json
"""
import argparse
import collections
import gzip
import hashlib
import json
import sys
from pathlib import Path
from typing import Callable, Dict, Iterable, Iterator, Optional

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE))

from uie.evaluation import FIELDS  # noqa: E402
from sc1_common import (MEASURABLE, REGISTERED_C_ROWS, REGISTERED_T_ROWS,  # noqa: E402
                        tag_of)

#: The only files a table is built from (section 3.1): the domain frame.
SOURCES = {"benign": "name_train", "phishing": "phishing_name_train"}


def read_rows(path: Path) -> Iterator[dict]:
    with gzip.open(path, "rt", encoding="utf-8") as fh:
        for line in fh:
            yield json.loads(line)


def count_suffixes(rows: Iterable[dict],
                   keep: Optional[Callable[[str], bool]] = None) -> Dict[str, Dict[str, int]]:
    """{label: {suffix: count}}: `SuffixTable.from_urls`'s rule, plus a row filter."""
    counts: Dict[str, collections.Counter] = collections.defaultdict(collections.Counter)
    for row in rows:
        url = row["url"]
        suffix = FIELDS["SUF"](url)
        if suffix and (keep is None or keep(url)):
            counts[row.get("label", "benign")][suffix] += 1
    return {label: dict(c) for label, c in counts.items()}


def build(bundle: Path) -> dict:
    """{"C": {label: counts}, "T": {tag: {label: counts}}}, one pass per file."""
    tables: dict = {"C": {}, "T": {}}
    for label, stem in SOURCES.items():
        states = [(tag_of(lab, st), st) for _, lab, st in MEASURABLE if lab == label]
        marginal = collections.Counter()
        per_state = {tag: collections.Counter() for tag, _ in states}
        for row in read_rows(bundle / f"{stem}.jsonl.gz"):
            if row.get("label", "benign") != label:
                raise ValueError(f"{stem}: a row labelled {row.get('label')!r}")
            url = row["url"]
            suffix = FIELDS["SUF"](url)
            if not suffix:
                continue
            marginal[suffix] += 1
            present = {"PTH": bool(FIELDS["PTH"](url)), "QRY": bool(FIELDS["QRY"](url))}
            for tag, state in states:
                if all(present[f] == want for f, want in state.items()):
                    per_state[tag][suffix] += 1
        tables["C"][label] = dict(marginal)
        for tag, counts in per_state.items():
            tables["T"][tag] = {label: dict(counts)}
    return tables


def canonical(tables: dict) -> bytes:
    """One byte-exact serialisation, so a sha256 identifies the tables."""
    return (json.dumps(tables, sort_keys=True, indent=1) + "\n").encode("utf-8")


def totals_failures(tables: dict) -> list:
    """Row totals against the ones the design review registered."""
    bad = []
    for label, want in REGISTERED_C_ROWS.items():
        got = sum(tables["C"].get(label, {}).values())
        if got != want:
            bad.append(f"C {label}: {got} rows, registered {want}")
    for tag, want in REGISTERED_T_ROWS.items():
        label = tag.split("_")[0]
        got = sum(tables["T"].get(tag, {}).get(label, {}).values())
        if got != want:
            bad.append(f"T {tag}: {got} rows, registered {want}")
    if set(tables["T"]) != set(REGISTERED_T_ROWS):
        bad.append(f"T states {sorted(tables['T'])} are not the registered 14")
    return bad


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bundle", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    tables = build(args.bundle)
    bad = totals_failures(tables)
    for label, counts in sorted(tables["C"].items()):
        print(f"  C {label:<16} rows {sum(counts.values()):>9,}  suffixes {len(counts):>5}")
    for tag, by_label in tables["T"].items():
        counts = next(iter(by_label.values()))
        print(f"  T {tag:<16} rows {sum(counts.values()):>9,}  suffixes {len(counts):>5}")
    if bad:
        sys.exit("REFUSED -- row totals differ from the registration:\n  " + "\n  ".join(bad))
    blob = canonical(tables)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_bytes(blob)
    print(f"wrote {args.out}  sha256 {hashlib.sha256(blob).hexdigest()}")


if __name__ == "__main__":
    main()
