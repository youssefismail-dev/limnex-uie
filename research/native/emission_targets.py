"""Achievable free-mode emission targets (NATIVE_REPAIR_PLAN section 3, K1) -- READ-ONLY.

Free mode draws its suffix from the SHIPPED table (every *_train row, both
frames). A model whose per-suffix presence rates equal the domain-frame training
data's would therefore emit sum_s p_shipped(s) * P_train(field | s), which is not
the plain domain-frame rate. That sum is the target E8's K1b is registered
against. Rates are measured on name_train with the gate's own field extractors.

    python research/native/emission_targets.py
"""
import collections
import gzip
import json
import sys
from pathlib import Path

R = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(R))
from uie.evaluation import FIELDS  # noqa: E402
from uie.factorization import parse  # noqa: E402
from uie.suffixes import SuffixTable  # noqa: E402


def rows(path):
    with gzip.open(path, "rt", encoding="utf-8") as fh:
        for line in fh:
            yield json.loads(line)["url"]


def main() -> None:
    shipped = SuffixTable.load(R / "data" / "bundle2" / "suffix_table.json").counts
    out = {}
    for label, prefix in (("benign", ""), ("phishing", "phishing_")):
        n = collections.Counter()
        has = {"PTH": collections.Counter(), "QRY": collections.Counter()}
        for u in rows(R / "data" / "bundle2" / f"{prefix}name_train.jsonl.gz"):
            s = parse(u).public_suffix
            n[s] += 1
            for f in has:
                has[f][s] += bool(FIELDS[f](u))
        table = shipped[label]
        tot = sum(table.values())
        total = sum(n.values())
        row = {"name_train_rows": total,
               "shipped_mass_unseen_in_name_train":
                   sum(c for s, c in table.items() if not n[s]) / tot}
        for f in has:
            base = sum(has[f].values()) / total
            row[f"{f}_domain_frame_train"] = base
            row[f"{f}_achievable_under_shipped"] = sum(
                c / tot * (has[f][s] / n[s] if n[s] else base) for s, c in table.items())
        out[label] = row
        print(label, json.dumps(row, indent=1))
    (R / "data" / "native").mkdir(parents=True, exist_ok=True)
    (R / "data" / "native" / "emission_targets.json").write_text(json.dumps(out, indent=1),
                                                                 encoding="utf-8")


if __name__ == "__main__":
    main()
