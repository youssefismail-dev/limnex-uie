"""E13 -- local smoke tests on the real data, before the notebooks are built
(research/e13/PREREG_E13.md section 6, step 4). CPU; nothing here is data.

    python research/e13/e13_smoke.py --part tokenizer   # training's tokenizer IS e12_final's; every artifact row matches
    python research/e13/e13_smoke.py --part train_ctx   # train() with ContextSwap on the real artifact, 3 steps
    python research/e13/e13_smoke.py --part decode      # D1 / D2 arms and N's sets generate; D1 obeys S
    python research/e13/e13_smoke.py --part train_n     # N's path (data -> 1 step -> query sets), tiny

Each part prints PASS or raises.
"""
from __future__ import annotations

import argparse
import gzip
import json
import sys
import tempfile
import time
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
for p in (ROOT, HERE, ROOT / "research" / "native"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import e13_common as E13  # noqa: E402

DATA = ROOT / "data" / "bundle2"
E12_FINAL = ROOT / "data" / "e12" / "train" / "e12_final" / "final.pt"
ARTIFACT = ROOT / "data" / "e13" / E13.PREFIX_ARTIFACT
SETTINGS = ROOT / "data" / "e13" / E13.SETTINGS_FILE


def training_tokenizer():
    """The tokenizer `uie.kaggle_train._run` builds for e12_final's recipe, rebuilt
    the same way from the same data."""
    from uie.kaggle_train import STREAMS, TOP_LANGUAGES, load_streams
    from uie.tokenizer import URLTokenizer
    tr_urls, _, tr_langs, _, _ = load_streams(DATA, STREAMS)
    counts = {}
    for lang in tr_langs:
        counts[lang] = counts.get(lang, 0) + 1
    keep = {lang for lang, _ in sorted(counts.items(), key=lambda kv: -kv[1])[:TOP_LANGUAGES]}
    keep.add("und")
    return URLTokenizer(languages=sorted(keep), include_frame=True, include_fields=True,
                        root_slash=True, presence_plan=True, slash_plan=True).fit(tr_urls, min_count=1)


def part_tokenizer() -> None:
    from uie.evaluation import FIELDS
    from uie.train import encode_corpus, load_checkpoint
    _, e12_tok, _ = load_checkpoint(E12_FINAL)
    t = training_tokenizer()
    assert t.state_dict() == e12_tok.state_dict(), "training's tokenizer differs from e12_final's"
    print("tokenizer: training rebuilds e12_final's tokenizer exactly")
    with gzip.open(DATA / "phishing_name_train.jsonl.gz", "rt", encoding="utf-8") as fh:
        urls = [u for u in (json.loads(line)["url"] for line in fh) if FIELDS["QRY"](u)]
    enc, dropped = encode_corpus(urls, ["phishing"] * len(urls), t, 384, corpora=["name"] * len(urls))
    keys = {tuple(r[0]) for r in enc["phishing"]}
    with gzip.open(ARTIFACT, "rt", encoding="utf-8") as fh:
        blob = json.load(fh)
    listed = {tuple(r["ids"]) for r in blob["rows"]}
    assert listed <= keys, f"{len(listed - keys)} artifact rows match no training row"
    assert len(listed) == len(keys) == blob["stats"]["rows"], (len(listed), len(keys))
    print(f"artifact: all {len(listed):,} rows match training rows exactly "
          f"({dropped} over 384 tokens, as the artifact dropped)")
    print("PASS tokenizer")


def part_train_ctx() -> None:
    from uie.evaluation import FIELDS
    from uie.train import TrainConfig, URLDataset, encode_corpus, load_checkpoint, train
    from uie.kaggle_train import new_model_e12
    _, tok, cfg = load_checkpoint(E12_FINAL)
    e12_train = dict(torch.load(E12_FINAL, map_location="cpu", weights_only=False)["train_config"])
    with gzip.open(DATA / "phishing_name_train.jsonl.gz", "rt", encoding="utf-8") as fh:
        ph = [u for u in (json.loads(line)["url"] for line in fh) if FIELDS["QRY"](u)]
    with gzip.open(DATA / "name_train.jsonl.gz", "rt", encoding="utf-8") as fh:
        bn = [json.loads(line)["url"] for _, line in zip(range(2000), fh)]
    urls, labels = ph + bn, ["phishing"] * len(ph) + ["benign"] * len(bn)
    enc, _ = encode_corpus(urls, labels, tok, cfg.max_len, corpora=["name"] * len(urls))
    data = URLDataset(enc)
    model, _ = new_model_e12(cfg, tok, e12_train["seed"])
    tc = TrainConfig(**{**e12_train, "steps": 3, "batch_size": 16, "eval_every": 3,
                        "eval_batches": 1, "warmup": 1, "context_prefixes": str(ARTIFACT),
                        "context_p": E13.CONTEXT_P})
    with tempfile.TemporaryDirectory() as d:
        res = train(model, data, data, tok, tc, device=torch.device("cpu"), out_dir=Path(d),
                    log_every=10)
    cs = res["context_swap"]
    assert cs["matched_rows"] == cs["listed_rows"], cs
    print(f"train(): 3 steps with ContextSwap on the real artifact: {cs}")
    print("PASS train_ctx")


def _tables():
    from uie.controlled import StateSuffixTables
    from uie.suffixes import SuffixTable
    shipped = SuffixTable.load(DATA / "suffix_table.json")
    st = StateSuffixTables.load(ROOT / "data" / "state_suffix_tables.json")
    ps = json.loads((ROOT / "data" / "e12" / "presence_slash_table.json").read_text())["table"]
    return shipped, st, ps


def part_decode() -> None:
    import e12_common as E12
    from uie.suffixes import SuffixTable, generate_with_suffixes
    from uie.train import load_checkpoint
    from uie.factorization import parse
    model, tok, _ = load_checkpoint(E12_FINAL)
    model.eval()
    shipped, st, ps = _tables()
    settings = E13.load_settings(SETTINGS) if SETTINGS.is_file() else {
        "scopes": {"D2-Q": {"applied": True, "T": 1.05, "delta": 0.2},
                   "D2-P": {"applied": True, "T": 1.04, "delta": -0.2}}}
    for arm in ("e12_d1", "e12_dec"):
        for label in ("benign", "phishing"):
            trace = []
            urls = E13.free_urls(model, tok, arm, label, 0, shipped, ps, settings=settings, n=24,
                                 slash_trace=trace)
            assert len(urls) == 24
        urls = E13.control_urls(model, tok, arm, "benign", "P0Q1", 0, st, ps, settings=settings, n=24)
        assert len(urls) == 24
        print(f"{arm}: free and control generate")
    # D1's obedience, one supplied state at a time (the shuffled files cannot show it)
    for label, state in (("phishing", "P0Q0S1"), ("benign", "P0Q1S0"), ("benign", "P0Q0S0")):
        tables = (E12.supplied_tables(ps, shipped.counts, label))[state]
        urls = generate_with_suffixes(model, tok, label=label, n=40, table=SuffixTable({label: tables}),
                                      seed=7, fields=E12.fields_of(state), slash_rule=True,
                                      **E12.decode())
        want = state.endswith("S1")
        got = [bool(parse(u).path) for u in urls]
        assert all(g == want for g in got), (state, sum(got), len(got))
        print(f"D1: {label} {state}: all 40 hosts obey S = {int(want)}")
    # D1 must switch off once the host has ended: after <RS>, the query still comes
    # (were the rule still on, <QRY> right after the slash would be forbidden).
    from uie.evaluation import FIELDS
    tables = E12.supplied_tables(ps, shipped.counts, "benign")["P0Q1S1"]
    urls = generate_with_suffixes(model, tok, label="benign", n=60, table=SuffixTable({"benign": tables}),
                                  seed=11, fields=E12.fields_of("P0Q1S1"), slash_rule=True, **E12.decode())
    with_q = sum(bool(FIELDS["QRY"](u)) and bool(parse(u).path) for u in urls)
    assert with_q >= 57, with_q
    print(f"D1: benign P0Q1S1: {with_q}/60 URLs are host/?query -- the rule stops at the host end")
    for cell in E13.N_CELLS:
        urls = E13.n_urls(model, tok, cell, 0, shipped, ps, st, n=12)
        assert len(urls) == 12
    print("N: the three cells' query sets generate")
    print("PASS decode")


def part_train_n() -> None:
    import e13n_train as N
    with tempfile.TemporaryDirectory() as d:
        rec = N.run(DATA, E12_FINAL, "A1", Path(d) / "A1", ROOT / "data" / "_e13_tables",
                    device=torch.device("cpu"), steps=1, n_gen=6)
    assert rec["finite"] and len(rec["generation"]["files"]) == 15, rec["generation"]
    print("PASS train_n")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--part", required=True, choices=("tokenizer", "train_ctx", "decode", "train_n"))
    args = ap.parse_args()
    t0 = time.time()
    if args.part == "train_n":
        tables = ROOT / "data" / "_e13_tables"
        tables.mkdir(exist_ok=True)
        import shutil
        shutil.copy(DATA / "suffix_table.json", tables / "suffix_table.json")
        shutil.copy(ROOT / "data" / "state_suffix_tables.json", tables / "state_suffix_tables.json")
        shutil.copy(ROOT / "data" / "e12" / "presence_slash_table.json",
                    tables / "presence_slash_table.json")
    {"tokenizer": part_tokenizer, "train_ctx": part_train_ctx, "decode": part_decode,
     "train_n": part_train_n}[args.part]()
    print(f"({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
