"""E14 -- the laptop pre-flight on the real data, before the notebooks are built
(research/e14/PREREG_E14.md section 6, step 2). CPU; nothing here is data.

    python research/e14/e14_smoke.py --part dryrun     # Q's dry run: training's rows, counts, query share
    python research/e14/e14_smoke.py --part train_q    # train() with the weight on real rows, a few steps
    python research/e14/e14_smoke.py --part decode     # the arms generate; D1' slashes every P1 host
    python research/e14/e14_smoke.py --part fit        # K14-2's per-row M3 IS N2a's M3
    python research/e14/e14_smoke.py --part compare    # the criteria code on E13's (already read) files

Each part prints PASS or raises. `dryrun` writes data/e14/dryrun.json (Status).
"""
from __future__ import annotations

import argparse
import collections
import gzip
import json
import sys
import tempfile
import time
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
for p in (ROOT, HERE, ROOT / "research" / "e13", ROOT / "research" / "native",
          ROOT / "research" / "cr1", ROOT / "research" / "dc1", ROOT / "research" / "sc1",
          ROOT / "research" / "qc1"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import e12_common as E12  # noqa: E402
import e14_common as E14  # noqa: E402

DATA = ROOT / "data" / "bundle2"
E12_FINAL = ROOT / "data" / "e12" / "train" / "e12_final" / "final.pt"
E14D = ROOT / "data" / "e14"


def query_steps(tok, ids):
    """Section 2.1's positions, located by id: targets from the first character after
    each <QRY> through the target that ends the query (<FRG> or the end token)."""
    q_id, ends = tok.marker_id("QRY"), {tok.marker_id("FRG"), tok.eos_id}
    out = set()
    for q in (i for i, x in enumerate(ids) if x == q_id):
        e = next((i for i in range(q + 1, len(ids)) if ids[i] in ends), None)
        if e is not None:
            out.update(range(q + 1, e + 1))
    return out


def part_dryrun() -> None:
    """Every training row, encoded exactly as `kaggle_train._run` encodes it for
    e14_q (the rebuilt training tokenizer, its frame per corpus, max_len 384, the
    w = 4 mask), streamed: the counts must be e12_final's, and every weighted
    position must be a query step."""
    from e13_smoke import training_tokenizer
    from uie.kaggle_train import STREAMS, load_streams
    from uie.train import CORPUS_FRAME, field_mask, load_checkpoint
    _, e12_tok, cfg = load_checkpoint(E12_FINAL)
    tok = training_tokenizer()
    assert tok.state_dict() == e12_tok.state_dict(), "training's tokenizer is not e12_final's"
    urls, labels, langs, corp, _ = load_streams(DATA, STREAMS)
    counts, dropped = collections.Counter(), 0
    scored, query, rows_q = collections.Counter(), collections.Counter(), collections.Counter()
    bad = 0
    t0 = time.time()
    for i, (u, label) in enumerate(zip(urls, labels)):
        ids = tok.encode(u, label, langs[i], frame=CORPUS_FRAME.get(corp[i], "domain"))
        if len(ids) > cfg.max_len:
            dropped += 1
            continue
        counts[label] += 1
        m = field_mask(ids, tok, corp[i], 1.0, E14.QUERY_WEIGHT)
        heavy = {pos + 1 for pos, w in enumerate(m) if w == E14.QUERY_WEIGHT}
        if heavy:
            rows_q[label] += 1
            bad += heavy != query_steps(tok, ids)
        scored[label] += sum(1 for w in m if w > 0)
        query[label] += len(heavy)
        if i % 500_000 == 0:
            print(f"  {i:,} rows ({time.time() - t0:.0f}s)", flush=True)
    out = {"registration": E14.REGISTRATION + " section 6 step 2 (Q's dry run)",
           "counts": dict(counts), "dropped_over_max_len": dropped,
           "rows_with_a_query": dict(rows_q),
           "query_share_of_scored_positions": {c: query[c] / scored[c] for c in counts},
           "rows_where_weighted_positions_are_not_the_query_steps": bad}
    E14D.mkdir(parents=True, exist_ok=True)
    (E14D / "dryrun.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(json.dumps(out, indent=1))
    assert dict(counts) == E14.E12_FINAL_COUNTS, (dict(counts), E14.E12_FINAL_COUNTS)
    assert dropped == E14.E12_FINAL_DROPPED, dropped
    assert bad == 0, bad
    print("PASS dryrun")


def part_train_q() -> None:
    from uie.evaluation import FIELDS
    from uie.kaggle_train import new_model_e12
    from uie.train import TrainConfig, URLDataset, encode_corpus, load_checkpoint, train
    _, tok, cfg = load_checkpoint(E12_FINAL)
    e12_train = dict(torch.load(E12_FINAL, map_location="cpu", weights_only=False)["train_config"])
    with gzip.open(DATA / "phishing_name_train.jsonl.gz", "rt", encoding="utf-8") as fh:
        ph = [u for u, _ in zip((u for u in (json.loads(line)["url"] for line in fh)
                                 if FIELDS["QRY"](u)), range(300))]
    with gzip.open(DATA / "name_train.jsonl.gz", "rt", encoding="utf-8") as fh:
        bn = [json.loads(line)["url"] for _, line in zip(range(600), fh)]
    urls, labels = ph + bn, ["phishing"] * len(ph) + ["benign"] * len(bn)
    res = {}
    for w in (1.0, E14.QUERY_WEIGHT):
        enc, _ = encode_corpus(urls, labels, tok, cfg.max_len, corpora=["name"] * len(urls),
                               query_weight=w)
        if w != 1.0:                                   # the weight sits on the query steps only
            for rows in enc.values():
                for ids, m in rows:
                    assert {p + 1 for p, x in enumerate(m) if x == w} == query_steps(tok, ids)
        model, _ = new_model_e12(cfg, tok, e12_train["seed"])
        tc = TrainConfig(**{**e12_train, "steps": 4, "batch_size": 16, "eval_every": 4,
                            "eval_batches": 1, "warmup": 1, "query_weight": w})
        with tempfile.TemporaryDirectory() as d:
            res[w] = train(model, URLDataset(enc), URLDataset(enc), tok, tc,
                           device=torch.device("cpu"), out_dir=Path(d), log_every=10)
    steps = {w: {k: v["steps"] for k, v in r["grad_diagnostics"].items()} for w, r in res.items()}
    assert steps[1.0] == steps[E14.QUERY_WEIGHT], steps
    q = res[E14.QUERY_WEIGHT]["query_weight"]
    assert res[E14.QUERY_WEIGHT]["loss_constant"] == E14.E12_FINAL_LOSS_CONSTANT
    print(f"train(): 4 steps at w = 1 and w = 4 on real rows: same batches {steps[1.0]}; {q}")
    print("PASS train_q")


def _tables():
    from uie.controlled import StateSuffixTables
    from uie.suffixes import SuffixTable
    shipped = SuffixTable.load(DATA / "suffix_table.json")
    st = StateSuffixTables.load(ROOT / "data" / "state_suffix_tables.json")
    ps = json.loads((ROOT / "data" / "e12" / "presence_slash_table.json").read_text())["table"]
    return shipped, st, ps


def part_decode() -> None:
    import e13_common as E13
    from uie.factorization import parse
    from uie.suffixes import SuffixTable, generate_with_suffixes
    from uie.train import load_checkpoint
    model, tok, _ = load_checkpoint(E12_FINAL)
    model.eval()
    shipped, st, ps = _tables()
    for arm in E14.ARMS:
        for label in ("benign", "phishing"):
            trace = []
            urls = E14.free_urls(model, tok, arm, label, 0, shipped, ps, n=24,
                                 slash_trace=trace if E14.ARMS[arm]["slash_rule"] else None)
            assert len(urls) == 24
        urls = E14.control_urls(model, tok, arm, "benign", "P1Q1", 0, st, ps, n=24)
        assert len(urls) == 24
        print(f"{arm}: free and control generate")
    # D1' on P1 plans: every host is slashed (a path, or host/)
    for label, state in (("phishing", "P1Q0S1"), ("phishing", "P1Q1S1"), ("benign", "P1Q0S1")):
        table = SuffixTable({label: E12.supplied_tables(ps, shipped.counts, label)[state]})
        urls = generate_with_suffixes(model, tok, label=label, n=60, table=table, seed=7,
                                      fields=E12.fields_of(state), **E14.state_kwargs("e12_d1p"))
        assert all(bool(parse(u).path) for u in urls), [u for u in urls if not parse(u).path]
        print(f"D1': {label} {state}: all 60 hosts slashed")
    # D1' equals D1 on P0 plans: e12_d1p's P0 URLs are e12_d1's
    for state in ("P0Q0S1", "P0Q1S0"):
        table = SuffixTable({"benign": E12.supplied_tables(ps, shipped.counts, "benign")[state]})
        kw = dict(label="benign", n=30, table=table, seed=9, fields=E12.fields_of(state))
        a = generate_with_suffixes(model, tok, **kw, **E13.state_kwargs("e12_d1", "benign", state))
        b = generate_with_suffixes(model, tok, **kw, **E14.state_kwargs("e12_d1p"))
        assert a == b
    print("D1': P0 plans byte-identical to E13's D1")
    print("PASS decode")


def part_fit() -> None:
    from e14_fit import paired, per_row_m3
    from n2a_reference import STREAMS, rows, teacher_forced
    from uie.evaluation import FIELDS
    from uie.train import load_checkpoint
    model, tok, _ = load_checkpoint(E12_FINAL)
    model.eval()
    for label in E14.LABELS:
        urls = [u for u, _ in zip((u for u in rows(STREAMS[label][1]) if FIELDS["QRY"](u)), range(150))]
        mine = per_row_m3(model, tok, urls, label)
        _, _, theirs = teacher_forced(model, tok, urls, label, {})
        a = sum(x[0] for x in mine) / sum(x[1] for x in mine)
        b = sum(v[0] for v in theirs.values()) / sum(v[1] for v in theirs.values())
        assert abs(a - b) <= 1e-9 * abs(b), (a, b)
        same = paired(mine, mine)
        assert same["diff"] == 0.0 and not same["falls"]
        print(f"M3 per row = N2a's M3 ({label}): {a:.6f} vs {b:.6f} on {len(mine)} rows")
    print("PASS fit")


def part_compare() -> None:
    """The criteria code on E13's files, with E13's arms standing in. D1's own reading
    (E13 section 3.2) must come out exactly as E13 recorded it."""
    import e13_compare as C13
    import e14_compare as C14
    from e12_compare import finish_line, k12b
    rep = C13.merged()
    targets = json.loads((ROOT / "data" / "e11" / "targets.json").read_text(encoding="utf-8"))
    rec = json.loads((ROOT / "data" / "e13" / "compare_report.json").read_text(encoding="utf-8"))["D1"]
    d = C14.criteria_decoder(rep, targets, "e12_d1", "e12_final")
    for k in ("K14-4", "R-a", "R-b", "R-c", "R-d"):
        want = rec["K13-3" if k == "K14-4" else k]["pass"]
        assert d[k]["pass"] == want, (k, d[k]["pass"], want)
    assert d["adopted"] == rec["adopted"]
    print("D1' rules on E13's D1: " + " ".join(f"{k} {'PASS' if d[k]['pass'] else 'FAIL'}"
                                               for k in ("K14-4", "R-a", "R-b", "R-c", "R-d"))
          + f"; adopted {d['adopted']} (E13 recorded {rec['adopted']})")
    k1 = C14.k14_1(rep, "e13_ctx")
    fall = k1["cells"]["free phishing QRY"]["fall"]
    assert abs(fall - 0.0137) < 5e-5 and not k1["cells"]["free phishing QRY"]["moves"], k1
    for f in (C14.honouring_pq, C14.copying_q, C14.novelty_q):
        r = f(rep, "e13_ctx", "e12_final")
        print(f"  {f.__name__} on e13_ctx: {'PASS' if r['pass'] else 'FAIL'}")
    e3 = finish_line(rep, "e12_d1", k12b(rep, targets, "e12_d1"))
    assert not e3["met"] and e3["slash_rows_missed"] == ["phishing free_bare"], e3["slash_rows_missed"]
    print(f"K14-1 on e13_ctx: fall {fall:.4f} (E13: 0.0137); E3 on e12_d1: NOT MET, "
          f"{len(e3['cells_not_cleared'])} cells, slash {e3['slash_rows_missed']}")
    print("PASS compare")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--part", required=True,
                    choices=("dryrun", "train_q", "decode", "fit", "compare"))
    args = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")
    t0 = time.time()
    {"dryrun": part_dryrun, "train_q": part_train_q, "decode": part_decode, "fit": part_fit,
     "compare": part_compare}[args.part]()
    print(f"({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
