"""Q7 -- the termination hazard: did the model LEARN the tail it saw?

READ-ONLY. Teacher forcing only; no sampling anywhere, so the decoder and prefix
drift are both excluded by construction. Nothing under uie/ is modified, no
weight is touched, no training runs.

Pre-registered in PREREG_Q7.md (H1, H2, H4, H5).

Q6 left a 46% middle term: the loss saw value p99 = 87 and the model emits 56.
That gap has two very different explanations and sampling-based measurements
cannot separate them:

    the model never learned the tail   -> conditioned on a REAL long-value
                                          prefix it still wants to stop
    the model learned it but drifts    -> conditioned on a real prefix it
                                          continues correctly, and the loss
                                          happens somewhere downstream

The TERMINATION HAZARD separates them exactly. Inside a query value the next
token either continues the value or ends it (`&`, <FRG>, <eos>):

    h(i) = P(value ends at position i | value reached position i)

Empirical h is counted from the corpus. Model h is read off the teacher-forced
distribution on those same rows. If model h tracks empirical h at large i, the
conditional is right and the deficit is downstream. If model h runs above it,
the model is too eager to stop and the tail was genuinely not learned.

Reuses Q2's machinery rather than inventing a method: same encode, same
teacher-forced forward pass, same convention that the target at t is scored from
slot t-1.
"""
from __future__ import annotations

import argparse
import gzip
import json
import random
import statistics as st
import sys
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from uie.evaluation import FIELDS                    # noqa: E402
from uie.train import CORPUS_FRAME, load_checkpoint  # noqa: E402

BATCH = 48
BOOT = 1000
MIN_ROWS = 200                      # below this a bucket is UNMEASURED
#: Within-value position buckets. Chosen to straddle the p90 (24-26) and p99
#: (87-88) the loss actually sees, so the tail region is resolved rather than
#: lumped into one "long" bin.
BUCKETS = [(0, 4), (4, 10), (10, 25), (25, 50), (50, 100), (100, 10 ** 9)]
#: Absolute-position strata for H4, in tokens from sequence start.
ABS_STRATA = [(0, 60), (60, 120), (120, 256)]


def bucket_of(i: int, table=BUCKETS) -> Tuple[int, int]:
    for lo, hi in table:
        if lo <= i < hi:
            return (lo, hi)
    return table[-1]


def label(b: Tuple[int, int]) -> str:
    lo, hi = b
    return f"{lo}-{'inf' if hi > 10 ** 8 else hi}"


def boot_ci(pairs: Sequence[Tuple[int, int]], rng: random.Random):
    """Bootstrap CI for a ratio of two summed counts over ROWS, not events.

    Resampling rows rather than positions is the point: positions inside one
    value are not independent, and resampling them would give an interval far
    too tight -- the same mistake that made a 3-seed SE look authoritative in Q3.
    """
    if not pairs:
        return (float("nan"),) * 3
    n = len(pairs)
    tot_e = sum(p[0] for p in pairs)
    tot_m = sum(p[1] for p in pairs)
    point = (tot_m / tot_e) if tot_e else float("nan")
    draws = []
    for _ in range(BOOT):
        s_e = s_m = 0.0
        for _ in range(n):
            e, m = pairs[rng.randrange(n)]
            s_e += e
            s_m += m
        if s_e:
            draws.append(s_m / s_e)
    draws.sort()
    if not draws:
        return (point, float("nan"), float("nan"))
    return (point, draws[int(0.025 * len(draws))], draws[int(0.975 * len(draws))])


def field_spans(seq: List[int], tok, field: str) -> List[Tuple[int, int]]:
    """(start, end) token indices of each FIELD BODY in an encoded sequence.

    For QRY the body is everything after the <QRY> marker up to the next marker
    or the sequence end; values inside it are split on '&' by the caller. For
    PTH likewise. Markers are located by id, never by re-parsing the string, so
    this cannot drift from what the model actually consumed.
    """
    mid = tok.marker_id(field)
    markers = {tok.marker_id(m) for m in
               ("SCH", "USR", "SUF", "LBL", "SUB", "PRT", "PTH", "QRY", "FRG")}
    spans, i = [], 0
    while i < len(seq):
        if seq[i] == mid:
            j = i + 1
            while j < len(seq) and seq[j] not in markers and seq[j] != tok.eos_id:
                j += 1
            spans.append((i + 1, j))
            i = j
        else:
            i += 1
    return spans


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--field", default="QRY", choices=["QRY", "PTH"])
    ap.add_argument("--split", default="name_val")
    # `--corpus` and `--label` are DERIVED from the split by default and are
    # cross-checked when given. Before 2026-09-23 the class label was hard-coded
    # to "benign" here: pointing this script at phishing_name_val scored phishing
    # URLs under BENIGN conditioning and printed a clean, plausible, wrong hazard
    # table with no error at all. Deriving both from the one argument that names
    # the data makes that mismatch unrepresentable rather than merely unlikely.
    ap.add_argument("--corpus", default=None,
                    help="default: derived from --split")
    ap.add_argument("--label", default=None, choices=["benign", "phishing"],
                    help="default: derived from --split")
    ap.add_argument("--rows", type=int, default=120_000)
    ap.add_argument("--max-queries", type=int, default=6000)
    # Defaults to the frozen baseline, so every published Q7 run stays
    # bit-identical. A non-default checkpoint TAGS its artifact rather than
    # overwriting the baseline's.
    ap.add_argument("--checkpoint", default="data/best_e5.pt")
    args = ap.parse_args()

    # split is "[phishing_]<corpus>_<train|val>"
    stem_split = args.split
    derived_label = "benign"
    if stem_split.startswith("phishing_"):
        derived_label = "phishing"
        stem_split = stem_split[len("phishing_"):]
    derived_corpus = stem_split.split("_")[0]

    for nm, given, derived in (("--label", args.label, derived_label),
                               ("--corpus", args.corpus, derived_corpus)):
        if given is not None and given != derived:
            raise SystemExit(
                f"{nm}={given!r} contradicts --split={args.split!r}, which "
                f"implies {derived!r}. Refusing to run: a class/corpus mismatch "
                f"here produces a plausible WRONG table, not an error.")
    cls_label = args.label or derived_label
    corpus = args.corpus or derived_corpus

    dev = torch.device("cpu")
    model, tok, _ = load_checkpoint(Path(args.checkpoint), device=dev)
    model.to(dev).eval()
    frame = CORPUS_FRAME.get(corpus, "domain")
    max_len = model.cfg.max_len

    with gzip.open(Path("data/bundle2") / f"{args.split}.jsonl.gz", "rt",
                   encoding="utf-8") as fh:
        urls = [json.loads(l)["url"] for i, l in enumerate(fh) if i < args.rows]
    urls = [u for u in urls if FIELDS[args.field](u)][:args.max_queries]

    amp = tok.stoi.get("&")
    stoppers = {tok.eos_id, tok.marker_id("FRG"), tok.marker_id("QRY")}
    if args.field == "QRY" and amp is not None:
        stoppers.add(amp)
    slash = tok.stoi.get("/")
    if args.field == "PTH":
        # '/' MUST be a stopper here. The empirical event is "this segment
        # ends", and a segment ends at '/' just as a query value ends at '&'.
        # Counting the '/' event empirically while omitting its probability
        # from the model mass would compare two different quantities and make
        # the model look systematically too reluctant to stop -- an artifact,
        # not a finding. QRY already includes its cut token ('&') for exactly
        # this reason; PTH must match.
        stoppers = {tok.eos_id, tok.marker_id("QRY"), tok.marker_id("FRG")}
        if slash is not None:
            stoppers.add(slash)

    stem = Path(args.checkpoint).stem
    tag = "" if stem == "best_e5" else f"_{stem}"
    print(f"=== Q7 hazard  field={args.field}  split={args.split}  "
          f"label={cls_label}  corpus={corpus}  frame={frame}  ckpt={stem} ===",
          flush=True)

    # per row: bucket -> [events_empirical, mass_model]
    rows_by_bucket: Dict[Tuple[int, int], List[Tuple[int, int]]] = {}
    strat: Dict[Tuple, List[Tuple[int, int]]] = {}
    kept = skipped = 0

    for start in range(0, len(urls), BATCH):
        chunk = urls[start:start + BATCH]
        enc = [tok.encode(u, cls_label, "und", frame=frame) for u in chunk]
        enc = [e for e in enc if len(e) <= max_len]
        skipped += len(chunk) - len(enc)
        if not enc:
            continue
        width = max(len(e) for e in enc)
        ids = torch.tensor([e + [tok.pad_id] * (width - len(e)) for e in enc],
                           dtype=torch.long, device=dev)
        with torch.no_grad():
            logits = model(ids)
        probs = F.softmax(logits, dim=-1)

        for b, seq in enumerate(enc):
            for s, e in field_spans(seq, tok, args.field):
                kept += 1
                # split the field body into VALUES at '&' (QRY) or '/' (PTH)
                cut = amp if args.field == "QRY" else slash
                pos = s
                while pos < e:
                    end = pos
                    while end < e and seq[end] != cut:
                        end += 1
                    # positions pos..end-1 are one value; at each, does it stop?
                    per_row: Dict[Tuple[int, int], List[int]] = {}
                    for i in range(pos, end):
                        within = i - pos
                        bk = bucket_of(within)
                        # empirical: does the value end right after this token?
                        stops = 1 if (i + 1 >= end) else 0
                        # model: mass on any stopper, read from slot i
                        # (target at i+1 is predicted from slot i)
                        if i >= probs.shape[1]:
                            continue
                        pm = float(sum(probs[b, i, t].item() for t in stoppers
                                       if t is not None))
                        per_row.setdefault(bk, [0, 0.0])
                        per_row[bk][0] += stops
                        per_row[bk][1] += pm
                        ab = bucket_of(i, ABS_STRATA)
                        strat.setdefault((ab, bk), []).append((stops, pm))
                    for bk, (ev, ms) in per_row.items():
                        rows_by_bucket.setdefault(bk, []).append((ev, ms))
                    pos = end + 1
        if start % (BATCH * 20) == 0:
            print(f"  {start + len(chunk):>6}/{len(urls)}", flush=True)

    rng = random.Random(20260806)
    print(f"\n  rows used {kept:,}   over-max_len skipped {skipped:,}")
    print(f"\n  {'within-value':<14}{'rows':>8}{'emp h':>9}{'model h':>10}"
          f"{'ratio':>8}{'95% CI':>18}   reading")
    out: Dict[str, object] = {"field": args.field, "split": args.split,
                              "label": cls_label, "corpus": corpus,
                              "rows": kept, "skipped": skipped, "buckets": {}}
    for bk in BUCKETS:
        pairs = rows_by_bucket.get(bk, [])
        if len(pairs) < MIN_ROWS:
            print(f"  {label(bk):<14}{len(pairs):>8}{'--':>9}{'--':>10}"
                  f"{'--':>8}{'--':>18}   UNMEASURED (<{MIN_ROWS} rows)")
            out["buckets"][label(bk)] = {"rows": len(pairs), "status": "UNMEASURED"}
            continue
        tot_e = sum(p[0] for p in pairs)
        tot_m = sum(p[1] for p in pairs)
        npos = sum(1 for p in pairs for _ in range(1))
        point, lo, hi = boot_ci(pairs, rng)
        reading = ("model STOPS TOO EAGERLY" if lo > 1.0 else
                   ("model stops too rarely" if hi < 1.0 else "tracks empirical"))
        print(f"  {label(bk):<14}{len(pairs):>8}{tot_e:>9}{tot_m:>10.1f}"
              f"{point:>8.3f}{f'[{lo:.3f}, {hi:.3f}]':>18}   {reading}")
        out["buckets"][label(bk)] = {
            "rows": len(pairs), "empirical_events": tot_e,
            "model_mass": round(tot_m, 2), "ratio": round(point, 4),
            "ci": [round(lo, 4), round(hi, 4)], "reading": reading}

    # ---- H4: absolute position, holding within-value position fixed ----
    print(f"\n  H4 -- hazard ratio by ABSOLUTE token position, "
          f"within-value bucket held fixed")
    print(f"  {'within-value':<14}" + "".join(f"{label(a):>22}" for a in ABS_STRATA))
    h4: Dict[str, Dict[str, object]] = {}
    for bk in BUCKETS:
        line = f"  {label(bk):<14}"
        row: Dict[str, object] = {}
        for ab in ABS_STRATA:
            pairs = strat.get((ab, bk), [])
            if len(pairs) < MIN_ROWS:
                line += f"{'--':>22}"
                row[label(ab)] = "UNMEASURED"
                continue
            point, lo, hi = boot_ci(pairs, rng)
            line += f"{f'{point:.2f} [{lo:.2f},{hi:.2f}]':>22}"
            row[label(ab)] = {"ratio": round(point, 3), "ci": [round(lo, 3), round(hi, 3)]}
        print(line)
        h4[label(bk)] = row
    out["h4_absolute_position"] = h4

    p = Path(f"data/qry/q7_hazard_{args.field}_{args.split}{tag}.json")
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(out, indent=1) + "\n", encoding="utf-8")
    print(f"\nwrote {p}")


if __name__ == "__main__":
    main()
