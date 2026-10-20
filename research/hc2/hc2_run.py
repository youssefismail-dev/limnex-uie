"""HC2 -- presence by resampling: R1, R2, R3 and the descriptive joint settings.

Registered in research/hc2/PREREG_HC2.md, section 3. Canonical generation path
(marginal suffix table, frame "domain", temperature 1.0, adopted delta); HC1's
seeds, cell i (C1's order) = 20260925 + 10,000*i, n = 1,200, so every cell is
paired with HC1's masked arm.

    R1  joint resampling, 8 cells: every URL honours; no request exhausts
    R2  held-suffix resampling on the <Q:1> cells reproduces HC1 H3's exact
        arms exactly (as multisets)
    R3  on the 4 field-on cells, SUF-field excess against real URLs WITH the
        field, joint vs held, D = 10 shared split seeds (20260925 + 900 + d):
        joint stays the default unless Delta >= 0.02 with the CI excluding 0

Writes only under --out. Reads C1's and HC1's outputs, never writes them.

    python research/hc2/hc2_run.py --checkpoint data/best_e7.pt --bundle data/bundle2 \
        --c1 data/e7_eval/c1 --hc1 data/hc1 --out data/hc2
"""
import argparse
import collections
import gc
import hashlib
import json
import math
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import torch  # noqa: E402
from scipy.stats import t as student_t  # noqa: E402

from uie.corpus.run_eval import _find, _rows  # noqa: E402
from uie.evaluation import FIELDS, discriminate_with_floor  # noqa: E402
from uie.presence import PresenceUnmet, generate_with_presence  # noqa: E402
from uie.suffixes import SuffixTable  # noqa: E402
from uie.train import load_checkpoint  # noqa: E402

DELTA = {"benign": 0.22, "phishing": 0.30}
SEED = 20260925
SPLIT_BASE = SEED + 900
TOL = 0.02
LONG = 240                      # characters: HC1's truncation proxy


def cells(classes):
    single = [(label, {f: want}) for label in classes
              for f in ("PTH", "QRY") for want in (True, False)]
    joint = [(label, {"PTH": p, "QRY": q}) for label in classes
             for p in (True, False) for q in (True, False)]
    return single, joint


def tag_of(label, fields):
    return label + "_" + "_".join(f"{f[0]}{int(v)}" for f, v in fields.items())


def share(urls, pred):
    return round(sum(map(pred, urls)) / len(urls), 4) if urls else None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--bundle", type=Path, required=True)
    ap.add_argument("--c1", type=Path, required=True, help="C1's output directory")
    ap.add_argument("--hc1", type=Path, required=True, help="HC1's output directory")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--n", type=int, default=1200)
    ap.add_argument("--n-joint", type=int, default=1000)
    ap.add_argument("--splits", type=int, default=10)
    args = ap.parse_args()
    for d in ("joint", "held", "settings"):
        (args.out / d).mkdir(parents=True, exist_ok=True)

    device = torch.device("cpu")
    model, tok, _ = load_checkpoint(args.checkpoint, device=device)
    model.to(device).eval()
    table = SuffixTable.load(args.bundle / "suffix_table.json")
    single, joint_settings = cells(tok.classes)
    report = {"test": "HC2 (research/hc2/PREREG_HC2.md section 3)",
              "checkpoint_sha256": hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
              "n": args.n, "seed_base": SEED, "split_seeds": [SPLIT_BASE + d
                                                             for d in range(args.splits)],
              "R1": {}, "R2": {}, "R3": {}, "settings": {}}
    save = lambda: (args.out / "hc2_report.json").write_text(  # noqa: E731
        json.dumps(report, indent=1, default=float), encoding="utf-8")

    def run(label, fields, seed, n, hold, folder):
        stats, t0 = {}, time.time()
        try:
            urls = generate_with_presence(
                model, tok, label=label, n=n, table=table, fields=fields, seed=seed,
                hold_suffix=hold, stats=stats, decoder="temperature",
                decoder_kwargs={"temperature": 1.0},
                field_bias={"PTH": {tok.eos_id: -DELTA[label]}}, device=device)
            error = None
        except PresenceUnmet as exc:
            urls, error = [], str(exc)
        (args.out / folder / f"{tag_of(label, fields)}.json").write_text(
            json.dumps(urls), encoding="utf-8")
        return urls, stats, error, round(time.time() - t0, 1)

    # --- R1: joint resampling, the 8 single cells -------------------------------------
    for i, (label, fields) in enumerate(single):
        tag = tag_of(label, fields)
        urls, st, err, secs = run(label, fields, SEED + 10_000 * i, args.n, False, "joint")
        (f, want), = fields.items()
        masked = json.loads((args.hc1 / "h1" / f"{tag}.json").read_text(encoding="utf-8"))
        c1 = json.loads((args.c1 / f"{tag}.json").read_text(encoding="utf-8"))
        report["R1"][tag] = {
            "seed": SEED + 10_000 * i, "error": err, "n": len(urls),
            "honoured": sum(bool(FIELDS[f](u)) == want for u in urls),
            "draws": st.get("draws"), "first_pass": st.get("first_pass"),
            "rounds": st.get("rounds"),
            "draws_per_kept": round(st["draws"] / args.n, 4) if st else None,
            "inverse_first_pass": round(args.n / st["first_pass"], 4) if st.get("first_pass") else None,
            "long_share": share(urls, lambda u: len(u) >= LONG),
            "long_share_c1_unconstrained": share(c1, lambda u: len(u) >= LONG),
            "path_share": {"joint": share(urls, lambda u: bool(FIELDS["PTH"](u))),
                           "hc1_masked": share(masked, lambda u: bool(FIELDS["PTH"](u)))},
            "query_share": {"joint": share(urls, lambda u: bool(FIELDS["QRY"](u))),
                            "hc1_masked": share(masked, lambda u: bool(FIELDS["QRY"](u)))},
            "seconds": secs}
        r = report["R1"][tag]
        print(f"  R1 {tag:<12} honoured {r['honoured']}/{args.n}  draws {r['draws']} "
              f"(1st pass {r['first_pass']}, rounds {r['rounds']})  long "
              f"{r['long_share']} vs C1 {r['long_share_c1_unconstrained']}  "
              f"err {err}  ({secs:.0f}s)", flush=True)
        save()
    report["R1"]["verdict"] = ("PASS" if all(
        v["error"] is None and v["honoured"] == args.n
        for k, v in report["R1"].items() if k != "verdict") else "FAIL")
    save()

    # --- R2 + the held arms of the 4 field-on cells --------------------------------------
    on = [(i, label, fields) for i, (label, fields) in enumerate(single)
          if next(iter(fields.values()))]
    held = {}
    for i, label, fields in on:
        tag = tag_of(label, fields)
        urls, st, err, secs = run(label, fields, SEED + 10_000 * i, args.n, True, "held")
        held[tag] = urls
        entry = {"seed": SEED + 10_000 * i, "error": err, "draws": st.get("draws"),
                 "first_pass": st.get("first_pass"), "rounds": st.get("rounds"),
                 "seconds": secs}
        if "_Q1" in tag:
            h3 = json.loads((args.hc1 / "h3" / f"{label}_Q1_exact.json")
                            .read_text(encoding="utf-8"))
            entry["identical_to_hc1_h3_exact_arm"] = (collections.Counter(urls)
                                                      == collections.Counter(h3))
            entry["joint_first_pass"] = report["R1"][tag]["first_pass"]
        report["R2"][tag] = entry
        print(f"  held {tag:<12} draws {entry['draws']} (1st pass {entry['first_pass']})  "
              f"== HC1 H3: {entry.get('identical_to_hc1_h3_exact_arm', '--')}  "
              f"err {err}  ({secs:.0f}s)", flush=True)
        save()
    report["R2"]["verdict"] = ("PASS" if all(
        report["R2"][t].get("identical_to_hc1_h3_exact_arm") for t in report["R2"]
        if "_Q1" in t) else "FAIL")
    save()

    # --- R3: joint against held on SUF, against real URLs with the field --------------
    for i, label, fields in on:
        tag = tag_of(label, fields)
        (f, _), = fields.items()
        prefix = "" if label == "benign" else f"{label}_"
        pool = [u for u in _rows(_find(args.bundle, f"{prefix}name_val")) if FIELDS[f](u)]
        joint = json.loads((args.out / "joint" / f"{tag}.json").read_text(encoding="utf-8"))
        rows = []
        for d in range(args.splits):
            s = SPLIT_BASE + d
            a = discriminate_with_floor(pool, joint, "SUF", n=1200, seed=s)
            gc.collect()
            b = discriminate_with_floor(pool, held[tag], "SUF", n=1200, seed=s)
            gc.collect()
            ok = not (a.get("insufficient") or b.get("insufficient"))
            rows.append({"seed": s, "joint": a, "held": b,
                         "delta": a["excess"] - b["excess"] if ok else None})
        deltas = [r["delta"] for r in rows if r["delta"] is not None]
        mix = lambda urls: collections.Counter(FIELDS["SUF"](u) for u in urls)  # noqa: E731
        real = mix(pool)
        tv = lambda c: round(0.5 * sum(abs(c[k] / sum(c.values()) - real[k] / sum(real.values()))  # noqa: E731
                                       for k in set(c) | set(real)), 4)
        res = {"real_pool": len(pool), "draws": rows, "usable": len(deltas),
               "suffix_tv_to_real": {"joint": tv(mix(joint)), "held": tv(mix(held[tag]))}}
        if len(deltas) >= 2:
            m, sd = statistics.fmean(deltas), statistics.stdev(deltas)
            h = float(student_t.ppf(0.975, len(deltas) - 1)) * sd / math.sqrt(len(deltas))
            res.update({"delta": round(m, 4), "ci95": [round(m - h, 4), round(m + h, 4)],
                        "x_joint": round(statistics.fmean(r["joint"]["excess"] for r in rows
                                                          if r["delta"] is not None), 4),
                        "x_held": round(statistics.fmean(r["held"]["excess"] for r in rows
                                                         if r["delta"] is not None), 4),
                        "n_per_draw": rows[0]["joint"].get("n"),
                        "held_preferred": m >= TOL and m - h > 0})
        report["R3"][tag] = res
        print(f"  R3 {tag:<12} SUF delta {res.get('delta')} CI {res.get('ci95')}  "
              f"x_joint {res.get('x_joint')} x_held {res.get('x_held')}  "
              f"TV to real {res['suffix_tv_to_real']}  n {res.get('n_per_draw')}", flush=True)
        save()
    report["R3"]["default"] = ("held" if any(v.get("held_preferred") for k, v in
                                             report["R3"].items() if k != "default")
                               else "joint")
    save()

    # --- descriptive: the 8 joint settings, joint resampling --------------------------
    for j, (label, fields) in enumerate(joint_settings):
        i = len(single) + j
        tag = tag_of(label, fields)
        urls, st, err, secs = run(label, fields, SEED + 10_000 * i, args.n_joint, False,
                                  "settings")
        report["settings"][tag] = {
            "seed": SEED + 10_000 * i, "error": err, "n": len(urls),
            "honoured": sum(all(bool(FIELDS[f](u)) == w for f, w in fields.items())
                            for u in urls),
            "draws": st.get("draws"), "first_pass": st.get("first_pass"),
            "rounds": st.get("rounds"), "seconds": secs}
        r = report["settings"][tag]
        print(f"  set {tag:<15} honoured {r['honoured']}/{args.n_joint}  draws {r['draws']} "
              f"(1st pass {r['first_pass']})  err {err}  ({secs:.0f}s)", flush=True)
        save()
    print(f"\nR1 {report['R1']['verdict']}   R2 {report['R2']['verdict']}   "
          f"R3 default: {report['R3']['default']}")


if __name__ == "__main__":
    main()
