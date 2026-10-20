"""
research/v2/v2_p2_prepare.py

V2-P2 (research/v2/PREREG_V2_P2.md), on the laptop, on the final tables, before any bundle:

    python research/v2/v2_p2_prepare.py --arm p2_a     # -> data/v2/p2/prepare_p2_a.json
    python research/v2/v2_p2_prepare.py --arm p2_b     # -> data/v2/p2/prepare_p2_b.json

One pass over the data the run trains and validates on (the training and validation
streams of `data/bundle2`, read by `uie.kaggle_train.load_streams` as the run reads them,
and stream 1), with the arm's tokenizer:

- **V-T1 (section 12):** losslessness of every training, validation and stream URL.
  - Canonical, and under dropout splits (4 per training and stream row, 1 per validation
    row): each split spells out exactly the character encoding (`chars_of`).
  - The canonical split and the first dropout split decode exactly as `e16_cap`'s
    tokenizer decodes the characters.
- **C (section 18.2):** batch size x the expected scored mass of one drawn row, old rows
  only, under the arm's encoding, the name scope and BPE-dropout. It is `uie.train.
  expected_scored_tokens`'s arithmetic, with each kept row's mass averaged over 4 dropout
  draws. Each row's draws come from its own generator, seeded from (20261072, the row's
  index).
- **The checks on the script itself:**
  - the kept and dropped rows must be `e16_cap`'s;
  - `--validate` (on a subset, after the same subset's main pass, `--limit`): the script's C in
    characters, and in canonical pieces with the name scope, must equal what
    `uie.train.expected_scored_tokens` gives on the same rows encoded by the run's own
    `encode_corpus` (the characters' and the P2 seam's), bit for bit.
- **Reported, not a check:** C in characters against `e16_cap`'s pinned C. That constant
  (14397.885481960007 = 256 x 56.24174016390628, PREREG_E9 D3) was computed for E7's sampler
  and tokenizer and carried unchanged through E10-E16, so E9's rule on `e16_cap`'s own data
  need not give it. (A first version of this script made it a check; `--recheck` re-reads a
  result with the corrected checks, recomputing nothing.)
- **Reported:** tokens per row (canonical and under dropout) against characters per row, by
  class.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
import time
from array import array
from pathlib import Path
from typing import Dict, List

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
for _p in (ROOT, HERE):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import v2_p2_common as C  # noqa: E402

DATA = ROOT / "data" / "bundle2"
STREAM = ROOT / "data" / "e15" / "qry_extra" / C.STREAM_FILE
TABLES = ROOT / "data" / "v2" / "p2" / "tables"
OUT = ROOT / "data" / "v2" / "p2"
C_SEED = 20261072
DRAWS = 4
EXAMPLES = 10


def row_rng(index: int) -> random.Random:
    h = hashlib.blake2b(f"{C_SEED}|{index}".encode(), digest_size=8).digest()
    return random.Random(int.from_bytes(h, "little"))


def peak_rss_gb():
    try:
        import psutil
        return round(psutil.Process().memory_info().rss / 2 ** 30, 2)
    except Exception:
        return None


def corrected_checks(lossless: bool, rows_ok) -> dict:
    return {"V-T1: every split spells the characters and decodes as e16_cap's": lossless,
            "rows kept and dropped are e16_cap's": rows_ok}


def recheck(arm: str) -> None:
    """Re-read prepare_<arm>.json with the corrected checks (module docstring): nothing is
    recomputed; the old checks are kept beside the new ones, marked superseded."""
    path = OUT / f"prepare_{arm}.json"
    out = json.loads(path.read_text(encoding="utf-8"))
    old = out.get("checks", {})
    new = corrected_checks(old["V-T1: every split spells the characters and decodes as e16_cap's"],
                           old["rows kept and dropped are e16_cap's"])
    out.update({"checks": new, "pass": all(v is not False for v in new.values()),
                "checks_superseded": old,
                "rechecked": "the pinned C is E9's constant for E7's sampler (PREREG_E9 D3), carried "
                             "since; E9's rule on e16_cap's data need not equal it -- reported, not a "
                             "check. The script's arithmetic is validated against uie's own function "
                             "instead (--validate)."})
    path.write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(f"{arm}: {new} -> {'PASS' if out['pass'] else 'FAIL'} (rechecked, nothing recomputed)")


def validate(arm: str, limit: int) -> None:
    """The script's arithmetic against uie's own (module docstring), on the subset of the
    main pass run with the same --limit: C in characters, and in canonical pieces with the
    name scope, from `uie.train.expected_scored_tokens` over `encode_corpus`'s rows."""
    import uie.kaggle_train as KT
    import uie.train as UT
    import v2_p2_train as P2T
    smoke = json.loads((OUT / f"prepare_{arm}_smoke.json").read_text(encoding="utf-8"))
    if smoke.get("limit") != limit:
        raise SystemExit(f"run the main pass with --limit {limit} first")
    record = json.loads((ROOT / "data" / "v2" / "p1b" / C.E16_CAP_RECORD).read_text(encoding="utf-8"))
    tables = C.load_tables(arm, TABLES)
    tok = C.tokenizer(record["tokenizer"], arm, TABLES)
    base = tok.base
    max_len = C.E16.recipe_cap(STREAM)["max_len"]
    urls, labels, langs, corpora, label_counts = KT.load_streams(DATA, KT.STREAMS)
    seen, keep = {}, []
    for i, key in enumerate(zip(corpora, labels)):
        seen[key] = seen.get(key, 0) + 1
        if seen[key] <= limit:
            keep.append(i)
    known = set(base.languages)
    urls, labels, corpora = ([x[i] for i in keep] for x in (urls, labels, corpora))
    langs = [langs[i] if langs[i] in known else "und" for i in keep]
    cfg = UT.TrainConfig(batch_size=C.E16.recipe_cap(STREAM)["batch_size"], class_alpha=1.0)
    char_enc, _ = UT.encode_corpus(urls, labels, base, max_len, langs, corpora)
    c_char = cfg.batch_size * UT.expected_scored_tokens(UT.URLDataset(char_enc, label_counts=label_counts), cfg)
    P2T.patch(rest=tables["rest"], host=tables["host"], dropout=0.0, scope=C.NAME_SCOPE)
    try:
        piece_enc, _ = P2T._encode_corpus(urls, labels, tok, max_len, langs, corpora)
    finally:
        P2T.restore()
    piece_enc = {c: [(ids, mask) for ids, mask, _extra in rows] for c, rows in piece_enc.items()}
    c_piece = cfg.batch_size * UT.expected_scored_tokens(UT.URLDataset(piece_enc, label_counts=label_counts), cfg)
    lc = smoke["loss_constant"]
    mine_char = lc["check: characters, no scope (e16_cap's)"]["loss_constant"]
    mine_piece = lc["reported: canonical split, name scope, no dropout"]["loss_constant"]
    out = {"arm": arm, "limit": limit, "batch_size": cfg.batch_size,
           "characters": {"script": mine_char, "uie": c_char, "equal": mine_char == c_char},
           "canonical pieces, name scope": {"script": mine_piece, "uie": c_piece, "equal": mine_piece == c_piece}}
    out["pass"] = out["characters"]["equal"] and out["canonical pieces, name scope"]["equal"]
    (OUT / f"prepare_{arm}_validate.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(json.dumps(out, indent=1))


def over_length(arm: str) -> None:
    """V-T1's remainder: the training and validation rows over max_len (characters), which
    neither the run nor the main pass encodes, checked as the others are (canonical and 4
    dropout splits spell out the characters; canonical and the first split decode as
    e16_cap's tokenizer decodes them). -> data/v2/p2/prepare_<arm>_over_length.json"""
    import uie.kaggle_train as KT
    import uie.train as UT
    t0 = time.time()
    record = json.loads((ROOT / "data" / "v2" / "p1b" / C.E16_CAP_RECORD).read_text(encoding="utf-8"))
    tok = C.tokenizer(record["tokenizer"], arm, TABLES)
    base = tok.base
    max_len = C.E16.recipe_cap(STREAM)["max_len"]
    known = set(base.languages)
    fails, examples, n = {}, [], {"train": 0, "val": 0}
    for part, streams, offset in (("train", KT.STREAMS, 30_000_000), ("val", KT.VAL_STREAMS, 40_000_000)):
        urls, labels, langs, corpora, _ = KT.load_streams(DATA, streams)
        for i, (url, label, lang, corpus) in enumerate(zip(urls, labels, langs, corpora)):
            lang = lang if lang in known else "und"
            chars = tok.encode_chars(url, label, lang, frame=UT.CORPUS_FRAME.get(corpus, "domain"))
            if len(chars) <= max_len:
                continue
            n[part] += 1
            rng = row_rng(offset + i)
            canon = tok.pieces_from_chars(chars)
            splits = [tok.pieces_from_chars(chars, rng, tok.dropout) for _ in range(DRAWS)]
            bad = [k for k in ("spelling",) if any(tok.chars_of(x) != chars for x in [canon] + splits)]
            want = base.decode(chars)
            bad += [k for k, x in (("decode (canonical)", canon), ("decode (dropout)", splits[0]))
                    if tok.decode(x) != want]
            for k in bad:
                fails[f"{part}: {k}"] = fails.get(f"{part}: {k}", 0) + 1
                if len(examples) < EXAMPLES:
                    examples.append({"part": part, "kind": k, "url": url})
    out = {"registration": C.REGISTRATION, "arm": arm, "rows_over_max_len": n, "failures": fails,
           "failure_examples": examples, "pass": not fails, "seconds": round(time.time() - t0, 1)}
    (OUT / f"prepare_{arm}_over_length.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(f"{arm}: rows over max_len {n}; failures {fails or 'none'} -> {'PASS' if not fails else 'FAIL'}")


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True, choices=sorted(C.ARMS))
    ap.add_argument("--limit", type=int, default=None, help="smoke tests only: rows per stream")
    ap.add_argument("--over-length", action="store_true", help="V-T1 on the rows over max_len only")
    ap.add_argument("--validate", action="store_true", help="the arithmetic against uie's, on --limit's subset")
    ap.add_argument("--recheck", action="store_true", help="re-read a result with the corrected checks")
    args = ap.parse_args()
    if args.over_length:
        over_length(args.arm)
        return
    if args.recheck:
        recheck(args.arm)
        return
    if args.validate:
        if args.limit is None:
            raise SystemExit("--validate needs --limit (the subset of a smoke main pass)")
        validate(args.arm, args.limit)
        return
    import uie.kaggle_train as KT
    import uie.train as UT
    import v2_p2_train as P2T
    from uie.tokenizer import URLTokenizer

    t0 = time.time()
    record = json.loads((ROOT / "data" / "v2" / "p1b" / C.E16_CAP_RECORD).read_text(encoding="utf-8"))
    tables = C.load_tables(args.arm, TABLES)
    tok = C.tokenizer(record["tokenizer"], args.arm, TABLES)
    base: URLTokenizer = tok.base
    P2T._STATE.clear()
    P2T._STATE.update({"scope": C.NAME_SCOPE})
    langs_known = set(base.languages)
    recipe = C.E16.recipe_cap(STREAM)
    batch, max_len = recipe["batch_size"], recipe["max_len"]
    print(f"{args.arm}: vocab {tok.vocab_size:,} ({tok.n_base} base + "
          f"{ {k: len(tok.ids_of_table(k)) for k in ('rest', 'host')} }), dropout {tok.dropout}, "
          f"batch {batch}", flush=True)

    def stream_rows(streams, limit):
        urls, labels, langs, corpora, label_counts = KT.load_streams(DATA, streams)
        if limit is not None:
            seen: Dict[tuple, int] = {}
            keep = []
            for i, key in enumerate(zip(corpora, labels)):
                seen[key] = seen.get(key, 0) + 1
                if seen[key] <= limit:
                    keep.append(i)
            urls, labels, langs, corpora = ([x[i] for i in keep] for x in (urls, labels, langs, corpora))
        return urls, labels, [g if g in langs_known else "und" for g in langs], corpora, label_counts

    fails: Dict[str, Dict[str, int]] = {}
    examples: List[dict] = []

    def fail(part: str, kind: str, url: str) -> None:
        fails.setdefault(part, {}).setdefault(kind, 0)
        fails[part][kind] += 1
        if len(examples) < EXAMPLES:
            examples.append({"part": part, "kind": kind, "url": url})

    def lossless(part: str, url: str, chars: List[int], rng: random.Random, draws: int) -> List[List[int]]:
        canon = tok.pieces_from_chars(chars)
        splits = [tok.pieces_from_chars(chars, rng, tok.dropout) for _ in range(draws)]
        for ids in [canon] + splits:
            if tok.chars_of(ids) != chars:
                fail(part, "spelling", url)
        want = base.decode(chars)
        if tok.decode(canon) != want:
            fail(part, "decode (canonical)", url)
        if splits and tok.decode(splits[0]) != want:
            fail(part, "decode (dropout)", url)
        return [canon] + splits

    # --- the training rows: V-T1, C, and the character check ------------------------------
    urls, labels, langs, corpora, label_counts = stream_rows(KT.STREAMS, args.limit)
    mass = {"char": {}, "canonical": {}, "dropout": {}}
    tokens = {"char": {}, "canonical": {}, "dropout": {}}
    kept, dropped = {}, 0
    for i, (url, label, lang, corpus) in enumerate(zip(urls, labels, langs, corpora)):
        frame = UT.CORPUS_FRAME.get(corpus, "domain")
        chars = tok.encode_chars(url, label, lang, frame=frame)
        if len(chars) > max_len:
            dropped += 1
            continue
        kept[label] = kept.get(label, 0) + 1
        canon, *splits = lossless("train", url, chars, row_rng(i), DRAWS)
        for key, ids_list, scoped in (("char", [chars], False), ("canonical", [canon], True),
                                      ("dropout", splits, True)):
            if scoped:
                m = sum(sum(P2T.row_mask(ids, tok, corpus)) for ids in ids_list) / len(ids_list)
            else:
                m = float(sum(UT.field_mask(chars, base, corpus)))
            mass[key].setdefault(label, array("d")).append(m)
            tokens[key].setdefault(label, array("d")).append(sum(len(x) for x in ids_list) / len(ids_list))
        if (i + 1) % 250_000 == 0:
            print(f"  train {i + 1:,} rows ({time.time() - t0:.0f}s; rss {peak_rss_gb()} GB)", flush=True)
    classes = sorted(mass["char"])
    w = {c: max(label_counts.get(c, 1), 1) ** 1.0 for c in classes}
    total_w = sum(w.values())

    def constant(key: str) -> dict:
        out, per_class = 0.0, {}
        for c in classes:
            rows = mass[key][c]
            # expected_scored_tokens: sum(float(sum(mask))) over the class's rows / their count
            m = sum(float(x) for x in rows) / len(rows)
            per_class[c] = m
            out += w[c] / total_w * m
        return {"loss_constant": float(batch * out), "expected_mass_per_row": out, "per_class_mean": per_class}

    c_char, c_canon, c_drop = constant("char"), constant("canonical"), constant("dropout")
    print(f"  train done: kept {kept}, dropped {dropped} ({time.time() - t0:.0f}s)", flush=True)

    # --- the validation rows and stream 1: V-T1 -------------------------------------------
    v_urls, v_labels, v_langs, v_corpora, _ = stream_rows(KT.VAL_STREAMS, args.limit)
    v_kept = 0
    for i, (url, label, lang, corpus) in enumerate(zip(v_urls, v_labels, v_langs, v_corpora)):
        chars = tok.encode_chars(url, label, lang, frame=UT.CORPUS_FRAME.get(corpus, "domain"))
        if len(chars) > max_len:
            continue
        v_kept += 1
        lossless("val", url, chars, row_rng(10_000_000 + i), 1)
    s_rows = 0
    for i, line in enumerate(STREAM.read_text(encoding="utf-8").splitlines()):
        if not line.strip() or (args.limit is not None and s_rows >= args.limit):
            continue
        r = json.loads(line)
        chars = tok.encode_chars(r["url"], r["label"], r.get("lang", "und"), frame="domain")
        lossless("stream", r["url"], chars, row_rng(20_000_000 + i), DRAWS)
        s_rows += 1
    print(f"  validation {v_kept:,} and stream {s_rows:,} rows checked ({time.time() - t0:.0f}s)", flush=True)

    def per_class_mean(key: str) -> Dict[str, float]:
        return {c: sum(tokens[key][c]) / len(tokens[key][c]) for c in classes}

    full = args.limit is None
    checks = corrected_checks(not fails, (kept == C.COUNTS and dropped == C.DROPPED) if full else None)
    out = {
        "registration": C.REGISTRATION, "arm": args.arm, "limit": args.limit,
        "tables": {k: (C.table_sha256(v) if v is not None else None) for k, v in tables.items()},
        "rules_sha256": C.rules_sha256(), "vocab_size": tok.vocab_size,
        "pieces": {k: len(tok.ids_of_table(k)) for k in ("rest", "host")},
        "rows": {"train_kept": kept, "train_dropped": dropped, "val_checked": v_kept, "stream_checked": s_rows},
        "label_counts": label_counts,
        "loss_constant": {"seed": C_SEED, "draws": DRAWS, "batch_size": batch,
                          "arm (pieces, name scope, dropout)": c_drop,
                          "reported: canonical split, name scope, no dropout": c_canon,
                          "check: characters, no scope (e16_cap's)": c_char,
                          "e16_cap's pinned": C.LOSS_CONSTANT},
        "tokens_per_row": {"characters": per_class_mean("char"), "canonical": per_class_mean("canonical"),
                           "dropout": per_class_mean("dropout")},
        "lossless_failures": fails, "failure_examples": examples,
        "checks": checks, "pass": all(v is not False for v in checks.values()),
        "seconds": round(time.time() - t0, 1), "rss_gb": peak_rss_gb(),
    }
    OUT.mkdir(parents=True, exist_ok=True)
    name = f"prepare_{args.arm}.json" if full else f"prepare_{args.arm}_smoke.json"
    (OUT / name).write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(json.dumps(checks))
    print(f"C ({args.arm}) = {c_drop['loss_constant']!r}  (canonical {c_canon['loss_constant']:.3f}; "
          f"characters {c_char['loss_constant']!r}, e16_cap {C.LOSS_CONSTANT!r})")
    print(f"tokens per row: {out['tokens_per_row']}")
    print("PASS" if out["pass"] else "FAIL", f"-> {OUT / name}")


if __name__ == "__main__":
    main()
