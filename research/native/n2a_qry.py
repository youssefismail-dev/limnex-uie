"""N2a's primary readout (PREREG_N2A Amendment 2): the gate's own QRY cell,
against VALIDATION queries (Q-VAL, generalisation) and against the domain frame's
TRAINING queries (Q-TRAIN, "fits what it saw").

    python research/native/n2a_qry.py --label benign --check-e7    # validity + E7's Q-TRAIN
    python research/native/n2a_qry.py --label benign --files <generated...> --out <json>

The cell is computed exactly as `cr1_score.score` computes QRY: for every
generation g and split d, `discriminate_with_floor(name_pool, fake, "QRY",
n=min(n_main, 1200), seed=split_seed(SEED, g, d, "v2"))` -- the call
`field_report` makes for QRY -- then `_aggregate` and the section-26.1 verdict.
`--check-e7` first proves it: on E7's raw free files (DC1's base arm) against the
validation pool it must reproduce DC1's recorded QRY cell exactly.
"""
from __future__ import annotations

import argparse
import gc
import gzip
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE.parent / "cr1"))

from cr1_score import D, SEED, pools  # noqa: E402
from uie.corpus.run_eval import _aggregate, s261_verdict, split_seed  # noqa: E402
from uie.evaluation import FIELDS, discriminate_with_floor  # noqa: E402

#: QRY n per class: the validation pool caps phishing at 356 (1,068 hosts // 3),
#: so the training reading uses the same n -- the discriminator's power depends on n.
N_MAIN = {"benign": 1_200, "phishing": 356}
TRAIN_STEM = {"benign": "name_train", "phishing": "phishing_name_train"}


def qry_cell(real_pool, gens, n_main):
    draws = []
    for g, fake in enumerate(gens):
        for d in range(D):
            draws.append(discriminate_with_floor(real_pool, fake, "QRY", n=min(n_main, 1200),
                                                 seed=split_seed(SEED, g, d, "v2")))
            gc.collect()                                   # memory only, as in the gate
    agg = _aggregate({"QRY": draws})["QRY"]
    by_gen = [[x["excess"] for x in draws[g * D:(g + 1) * D] if not x.get("insufficient")]
              for g in range(len(gens))]
    return {**agg, "acceptance": s261_verdict(by_gen)}


def train_pool(label):
    """Query-bearing domain-frame training rows. The domain frame holds one URL
    per host, so keeping only query rows leaves the QRY pool exactly what the
    gate's one-per-host draw would build from the full stream."""
    with gzip.open(ROOT / "data" / "bundle2" / f"{TRAIN_STEM[label]}.jsonl.gz", "rt",
                   encoding="utf-8") as fh:
        return [u for u in (json.loads(line)["url"] for line in fh) if FIELDS["QRY"](u)]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--label", choices=("benign", "phishing"), required=True)
    ap.add_argument("--files", nargs="*", type=Path)
    ap.add_argument("--check-e7", action="store_true")
    ap.add_argument("--out", type=Path)
    args = ap.parse_args()
    label = args.label

    def load(paths):
        return [json.loads(Path(p).read_text(encoding="utf-8")) for p in paths]

    out = {"registration": "research/native/PREREG_N2A.md (Amendment 2)", "label": label,
           "n_main": N_MAIN[label]}
    if args.check_e7:
        files = sorted((ROOT / "data" / "dc1" / "gen").glob(f"free_base_{label}_g*.json"))
        gens = load(files)
        val = qry_cell(pools(ROOT / "data" / "bundle2", label)["name"], gens, N_MAIN[label])
        rec = json.loads((ROOT / "data" / "dc1" / "score_report.json").read_text(encoding="utf-8"))
        want = rec["free"]["base"][label]["fields"]["QRY"]["acceptance"]["x"]
        got = val["acceptance"]["x"]
        out["validity"] = {"recorded": want, "reproduced": got, "exact": got == want}
        print(f"  validity: DC1 recorded {want:+.4f}, reproduced {got:+.4f} -> "
              f"{'EXACT' if got == want else 'DIFFERS'}", flush=True)
        out["e7_q_val"] = val
        out["e7_q_train"] = qry_cell(train_pool(label), gens, N_MAIN[label])
        a = out["e7_q_train"]["acceptance"]
        print(f"  E7 Q-TRAIN {a['x']:+.4f} {a['verdict']} (se {a.get('se')})", flush=True)
    if args.files:
        gens = load(args.files)
        out["q_val"] = qry_cell(pools(ROOT / "data" / "bundle2", label)["name"], gens, N_MAIN[label])
        out["q_train"] = qry_cell(train_pool(label), gens, N_MAIN[label])
    dest = args.out or ROOT / "data" / "native" / f"n2a_qry_e7_{label}.json"
    dest.write_text(json.dumps(out, indent=1, default=float), encoding="utf-8")
    print("wrote", dest)


if __name__ == "__main__":
    main()
