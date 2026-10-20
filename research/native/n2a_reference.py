"""N2a-0 -- E7's reference for the query-content diagnostic, and the power check
(research/native/PREREG_N2A.md sections 3-4 and Amendment 1). READ-ONLY: nothing
is trained, generated or modified.

    python research/native/n2a_reference.py --label benign      # then phishing

Writes data/native/n2a_reference_<label>.json. The same functions read A1 and A2
later (`--checkpoint`, `--queries`), so every model is measured identically.

M1 LENGTH   Q7's termination hazard inside query values: per value-position bucket,
            model stop mass / corpus stop events, summed over ROWS, with Q7's
            row bootstrap. Controls hidden (`?`): E7's free mode. Domain frame.
M2 FREQUENCY  generated parameter names by corpus-frequency band (every
            query-bearing training row of the class), against faithful draws
            from the class's domain-frame training queries (Amendment 1).
M3 FIT      teacher-forced negative log-likelihood per query character, by the
            band of the parameter it belongs to; TRAIN against VALIDATION.
M4 COPYING  share of generated query strings found verbatim in training queries.
Power       each metric's minimum detectable change (2 x its 95% half-width);
            a gap under twice it is void for the decision (section 4).
"""
from __future__ import annotations

import argparse
import collections
import gzip
import json
import math
import random
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE.parent / "dc1"))

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

from research.qry.q7_hazard import BATCH, BUCKETS, boot_ci, bucket_of, field_spans  # noqa: E402
from research.qry.qtransforms import split_params  # noqa: E402
from uie.evaluation import FIELDS  # noqa: E402
from uie.train import load_checkpoint  # noqa: E402

TRAIN_ROWS, SAMPLE_SEED, BOOT_SEED = 20_000, 20261001, 20260806
BANDS = [(1, 1), (2, 4), (5, 19), (20, 99), (100, 999), (1000, 10 ** 9)]
RARE = [(1, 1), (2, 4), (5, 19)]
LONG = (25, 100)                                   # the LENGTH sub-deficit (section 5)
STREAMS = {"benign": ("name_train", "name_val", "structure_train"),
           "phishing": ("phishing_name_train", "phishing_name_val", "phishing_structure_train")}


def band_of(c: int):
    for lo, hi in BANDS:
        if lo <= c <= hi:
            return (lo, hi)
    return BANDS[-1]


def rows(stem: str):
    with gzip.open(ROOT / "data" / "bundle2" / f"{stem}.jsonl.gz", "rt", encoding="utf-8") as fh:
        for line in fh:
            yield json.loads(line)["url"]


def names_of(url: str):
    q = FIELDS["QRY"](url)
    return [k for k, _, _ in split_params(q)] if q else []


def encode(tok, url, label):
    """Free mode: E7-type controls hidden; a plan tokenizer scores its own plan."""
    hide = ("PTH", "QRY") if getattr(tok, "include_fields", False) and not \
        getattr(tok, "presence_plan", False) else ()
    return tok.encode(url, label, "und", frame="domain", hide_fields=hide)


def teacher_forced(model, tok, urls, label, freq):
    """M1 per-row (events, mass) per bucket and pooled over LONG; M3 per-band
    (nll_sum, chars). Q7's definitions: a value ends at '&', <FRG>, <QRY> or the
    end token; the target at i is predicted from slot i-1."""
    amp = tok.stoi.get("&")
    stop_ids = torch.tensor(sorted({tok.eos_id, tok.marker_id("FRG"), tok.marker_id("QRY")}
                                   | ({amp} if amp is not None else set())))
    per_bucket = collections.defaultdict(list)
    pooled_long = []
    nll = collections.defaultdict(lambda: [0.0, 0])
    for start in range(0, len(urls), BATCH):
        enc = [encode(tok, u, label) for u in urls[start:start + BATCH]]
        enc = [e for e in enc if len(e) <= model.cfg.max_len]
        if not enc:
            continue
        width = max(len(e) for e in enc)
        ids = torch.tensor([e + [tok.pad_id] * (width - len(e)) for e in enc], dtype=torch.long)
        with torch.no_grad():
            logp = F.log_softmax(model(ids).float(), dim=-1)
        stop_mass = logp[:, :, stop_ids].exp().sum(-1)
        for b, seq in enumerate(enc):
            for s, e in field_spans(seq, tok, "QRY"):
                # M3: attribute every query character to its parameter's name band
                text = "".join(tok.itos[t] if len(tok.itos[t]) == 1 else "\x00" for t in seq[s:e])
                offset = 0
                for part in text.split("&"):
                    name = part.split("=", 1)[0]
                    bk = band_of(freq.get(name, 0)) if name in freq else "OOV"
                    for k in range(len(part) + 1):             # the part and its '&' / end
                        i = s + offset + k
                        if i >= e or i >= len(seq):
                            break
                        nll[bk][0] -= float(logp[b, i - 1, seq[i]])
                        nll[bk][1] += 1
                    offset += len(part) + 1
                # M1: Q7's hazard, value by value
                pos = s
                while pos < e:
                    end = pos
                    while end < e and seq[end] != amp:
                        end += 1
                    row = collections.defaultdict(lambda: [0, 0.0])
                    long_pair = [0, 0.0]
                    for i in range(pos, end):
                        within = i - pos
                        stops = 1 if i + 1 >= end else 0
                        pm = float(stop_mass[b, i])
                        bk = bucket_of(within)
                        row[bk][0] += stops
                        row[bk][1] += pm
                        if LONG[0] <= within < LONG[1]:
                            long_pair[0] += stops
                            long_pair[1] += pm
                    for bk, pair in row.items():
                        per_bucket[bk].append(tuple(pair))
                    if long_pair != [0, 0.0]:
                        pooled_long.append(tuple(long_pair))
                    pos = end + 1
    return per_bucket, pooled_long, {("OOV" if k == "OOV" else f"{k[0]}-{k[1]}"): v
                                     for k, v in nll.items()}


def ratio(pairs, rng):
    point, lo, hi = boot_ci(pairs, rng)
    return {"rows": len(pairs), "ratio": point, "lo": lo, "hi": hi}


def band_profile(gen_urls, freq, null_names, rng, boot=1000):
    """M2: observed shares of generated names by band (bootstrap over QUERIES),
    and the faithful-sampler null of the same size."""
    per_query = [[band_of(freq[n]) if n in freq else "OOV" for n in names_of(u)]
                 for u in gen_urls if FIELDS["QRY"](u)]
    per_query = [q for q in per_query if q]
    n_params = sum(len(q) for q in per_query)

    def rare_share(qs):
        tot = sum(len(q) for q in qs)
        return sum(1 for q in qs for b in q if b in RARE) / tot if tot else float("nan")

    obs = rare_share(per_query)
    boots = sorted(rare_share([per_query[rng.randrange(len(per_query))] for _ in per_query])
                   for _ in range(boot))
    nulls = sorted(sum(1 for _ in range(n_params)
                       if band_of(freq[null_names[rng.randrange(len(null_names))]]) in RARE) / n_params
                   for _ in range(200))
    return {"queries": len(per_query), "params": n_params,
            "observed_rare": obs, "observed_ci": [boots[25], boots[974]],
            "null_rare": sum(nulls) / len(nulls), "null_ci": [nulls[5], nulls[194]],
            "by_band": {f"{b[0]}-{b[1]}": sum(1 for q in per_query for x in q if x == b) / n_params
                        for b in BANDS}}


def power(gap, lo, hi):
    half = (hi - lo) / 2
    mdc = 2 * half
    return {"gap": gap, "mdc": mdc, "void": not (abs(gap) >= 2 * mdc)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--label", choices=("benign", "phishing"), required=True)
    ap.add_argument("--checkpoint", type=Path, default=ROOT / "data" / "best_e7.pt")
    ap.add_argument("--queries", nargs="*", type=Path,
                    help="generated free-mode files; default: E7's raw free files")
    ap.add_argument("--out", type=Path)
    args = ap.parse_args()
    label = args.label
    train_stem, val_stem, struct_stem = STREAMS[label]

    # bands: every query-bearing training row of the class (both frames)
    freq = collections.Counter(n for stem in (train_stem, struct_stem)
                               for u in rows(stem) for n in names_of(u))
    null_names = [n for u in rows(train_stem) for n in names_of(u)]
    train_q = [u for u in rows(train_stem) if FIELDS["QRY"](u)]
    train_q = random.Random(SAMPLE_SEED).sample(train_q, min(TRAIN_ROWS, len(train_q)))
    val_q = [u for u in rows(val_stem) if FIELDS["QRY"](u)]
    print(f"{label}: {len(freq):,} distinct names; train sample {len(train_q):,}; "
          f"validation {len(val_q):,} query rows", flush=True)

    model, tok, _ = load_checkpoint(args.checkpoint, device=torch.device("cpu"))
    model.eval()
    rng = random.Random(BOOT_SEED)
    out = {"registration": "research/native/PREREG_N2A.md (Amendment 1)",
           "checkpoint": str(args.checkpoint), "label": label}
    for split, urls in (("train", train_q), ("validation", val_q)):
        per_bucket, long_pairs, nll = teacher_forced(model, tok, urls, label, freq)
        m1 = {f"{lo}-{hi}": ratio(per_bucket.get((lo, hi), []), rng) for lo, hi in BUCKETS}
        m1_long = ratio(long_pairs, rng)
        out[split] = {"M1_buckets": m1, "M1_long": m1_long,
                      "M1_long_power": power(m1_long["ratio"] - 1.0, m1_long["lo"], m1_long["hi"]),
                      "M3_nll_by_band": {k: {"nll_per_char": v[0] / v[1], "chars": v[1]}
                                         for k, v in sorted(nll.items()) if v[1]}}
        print(f"  {split:<10} M1 long ({LONG[0]}-{LONG[1]}) ratio {m1_long['ratio']:.3f} "
              f"[{m1_long['lo']:.3f}, {m1_long['hi']:.3f}] rows {m1_long['rows']:,}  "
              f"void={out[split]['M1_long_power']['void']}", flush=True)

    src = args.queries or (sorted((ROOT / "data" / "dc1" / "gen").glob(f"free_base_{label}_g*.json"))
                           + sorted((ROOT / "data" / "gen_cache_e7_gpu").glob(f"*_{label}_n8000_*.json")))
    gen = [u for p in src for u in json.loads(Path(p).read_text(encoding="utf-8"))]
    m2 = band_profile(gen, freq, null_names, rng)
    lo, hi = m2["observed_ci"]
    m2["power"] = power(m2["null_rare"] - m2["observed_rare"], lo, hi)
    out["M2"] = {**m2, "sources": [str(p) for p in src]}
    print(f"  M2 rare-band share {m2['observed_rare']:.4f} [{lo:.4f}, {hi:.4f}] vs null "
          f"{m2['null_rare']:.4f} on {m2['params']:,} params ({m2['queries']:,} queries)  "
          f"void={m2['power']['void']}", flush=True)

    from dc1_compare import training_queries
    known = training_queries(ROOT / "data" / "bundle2", label)
    qs = [FIELDS["QRY"](u) for u in gen if FIELDS["QRY"](u)]
    copied = sum(q in known for q in qs)
    out["M4"] = {"queries": len(qs), "copied": copied, "rate": copied / len(qs) if qs else None}
    print(f"  M4 copying {out['M4']['rate']:.4f} of {len(qs):,} generated queries", flush=True)

    dest = args.out or ROOT / "data" / "native" / f"n2a_reference_{label}.json"
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(out, indent=1, default=str), encoding="utf-8")
    print("wrote", dest)


if __name__ == "__main__":
    main()
