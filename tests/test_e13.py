"""E13's library changes (research/e13/PREREG_E13.md sections 2.1-2.2, V5, V6): the
slash rule (D1) in generation and ContextSwap (C) in training. Both are off by
default, and off leaves every earlier path exactly as it was (the E7-E12 suites
run unchanged).
"""
import gzip
import json
import random

import pytest
import torch
import torch.nn.functional as F

from test_e7_fields import FIT
from uie.model import (ModelConfig, URLTransformer, generate, slash_ends,
                       slash_redistribute)
from uie.tokenizer import URLTokenizer
from uie.train import (ContextSwap, TrainConfig, URLDataset, encode_corpus,
                       load_checkpoint, save_checkpoint, train)


def tok():
    return URLTokenizer(include_frame=True, include_fields=True, root_slash=True,
                        presence_plan=True, slash_plan=True).fit(FIT)


def model_for(t, max_len=128, seed=0):
    torch.manual_seed(seed)
    return URLTransformer(ModelConfig(vocab_size=t.vocab_size, max_len=max_len, d_model=16,
                                      n_layers=1, n_heads=2, d_ff=32))


ENDS = ("PTH", "RS", "QRY", "FRG", "EOS")


def end_ids(t):
    return {"PTH": t.marker_id("PTH"), "RS": t.rs_id, "QRY": t.marker_id("QRY"),
            "FRG": t.marker_id("FRG"), "EOS": t.eos_id}


# ---- D1: the redistribution, as a pure function ------------------------------------

@pytest.mark.parametrize("temperature", [1.0, 0.7, 1.6])
def test_d1_moves_only_the_forbidden_end_mass_to_the_allowed_ends(temperature):
    g = torch.Generator().manual_seed(3)
    logits = torch.randn(40, generator=g) * 2
    good, bad = torch.tensor([5, 9]), torch.tensor([1, 2, 30])
    before = F.softmax(logits / temperature, dim=-1)
    after_logits, moved = slash_redistribute(logits.clone(), good, bad, temperature)
    after = F.softmax(after_logits / temperature, dim=-1)
    ends = torch.cat([good, bad])
    others = torch.tensor([i for i in range(40) if i not in set(ends.tolist())])
    assert torch.allclose(after[others], before[others], atol=1e-6)        # untouched
    assert float(after[bad].sum()) == 0.0                                  # forbidden
    assert abs(float(after[good].sum()) - float(before[ends].sum())) < 1e-6   # host-end mass kept
    ratio = after[good] / before[good]                                     # proportional
    assert torch.allclose(ratio, ratio[0].expand_as(ratio), atol=1e-5)
    assert abs(moved - float(before[bad].sum())) < 1e-6


def test_d1_survives_an_allowed_end_with_tiny_mass():
    logits = torch.zeros(10)
    logits[3] = -60.0                      # the only allowed end, almost impossible
    logits[4] = 8.0                        # a forbidden end the model prefers
    out, moved = slash_redistribute(logits.clone(), torch.tensor([3]), torch.tensor([4]))
    p = F.softmax(out, dim=-1)
    assert torch.isfinite(out[3]) and float(p[4]) == 0.0
    assert abs(float(p[3]) - float(F.softmax(logits, dim=-1)[[3, 4]].sum())) < 1e-5
    assert moved > 0.99


def test_d1_forbids_by_s_and_leaves_p1_plans_alone():
    t = tok()
    e = end_ids(t)
    assert slash_ends(t, {"PTH": True, "QRY": True, "SLASH": True}) is None
    s1 = slash_ends(t, {"PTH": False, "QRY": True, "SLASH": True})
    assert set(s1["good"].tolist()) == {e["PTH"], e["RS"]}
    assert set(s1["bad"].tolist()) == {e["QRY"], e["FRG"], e["EOS"]}
    s0 = slash_ends(t, {"PTH": False, "QRY": False, "SLASH": False})
    assert set(s0["good"].tolist()) == {e["QRY"], e["FRG"], e["EOS"]}
    assert set(s0["bad"].tolist()) == {e["PTH"], e["RS"]}


# ---- D1 in generation ---------------------------------------------------------------

def _host_end(t, ids):
    """The first host-end token after <LBL>, or None if the row has none."""
    lbl = t.marker_id("LBL")
    ends = set(end_ids(t).values())
    if lbl not in ids:                 # a random model can end before the suffix
        return None
    k = ids.index(lbl)
    return next((x for x in ids[k + 1:] if x in ends), None)


@pytest.mark.parametrize("plan", [{"PTH": False, "QRY": False, "SLASH": True},
                                  {"PTH": False, "QRY": True, "SLASH": True},
                                  {"PTH": False, "QRY": True, "SLASH": False},
                                  {"PTH": False, "QRY": False, "SLASH": False}])
def test_d1_every_host_ends_as_the_supplied_s_says(plan):
    t, m = tok(), model_for(tok())
    ids, trace = [], []
    generate(m, t, label="benign", n=24, suffix="com", seed=11, fields=plan, ids_out=ids,
             slash_rule=True, slash_trace=trace)
    good = set(slash_ends(t, plan)["good"].tolist())
    seen = [e for e in (_host_end(t, row) for row in ids) if e is not None]
    assert len(seen) >= 5                          # the test is not vacuous
    assert all(e in good for e in seen)
    assert len(trace) == 24 and all(x >= 0.0 for x in trace)
    raw = []                                       # and the rule is what did it
    generate(m, t, label="benign", n=24, suffix="com", seed=11, fields=plan, ids_out=raw)
    assert any(e not in good for e in (_host_end(t, row) for row in raw) if e is not None)


def test_d1_off_is_bit_identical_and_p1_plans_are_untouched():
    t, m = tok(), model_for(tok())
    p0 = {"PTH": False, "QRY": True, "SLASH": True}
    a = generate(m, t, label="benign", n=16, suffix="com", seed=5, fields=p0)
    b = generate(m, t, label="benign", n=16, suffix="com", seed=5, fields=p0, slash_rule=False)
    assert a == b
    p1 = {"PTH": True, "QRY": True, "SLASH": True}
    c = generate(m, t, label="phishing", n=16, suffix="com", seed=5, fields=p1)
    d = generate(m, t, label="phishing", n=16, suffix="com", seed=5, fields=p1, slash_rule=True)
    assert c == d


def test_d1_refuses_what_it_was_not_registered_for():
    t, m = tok(), model_for(tok())
    with pytest.raises(ValueError):            # no supplied plan
        generate(m, t, label="benign", n=2, suffix="com", seed=1, slash_rule=True)
    with pytest.raises(ValueError):            # not the raw decode
        generate(m, t, label="benign", n=2, suffix="com", seed=1, decoder="top_k",
                 fields={"PTH": False, "QRY": False, "SLASH": True}, slash_rule=True)
    with pytest.raises(ValueError):            # a trace without the rule
        generate(m, t, label="benign", n=2, suffix="com", seed=1, slash_trace=[],
                 fields={"PTH": False, "QRY": False, "SLASH": True})


# ---- C: ContextSwap -------------------------------------------------------------------

def _row(t, url, label="phishing"):
    enc, _ = encode_corpus([url], [label], t, 256, corpora=["name"])
    return enc[label][0]


def _prefix(t, url, label="phishing"):
    """A stand-in for a model-written prefix: another URL's tokens up to its <QRY>."""
    ids = t.encode(url, label)
    q = ids.index(t.marker_id("QRY"))
    return ids[:q + 1]


def test_c_splices_the_prefix_and_zeroes_every_model_written_target():
    t = tok()
    real = _row(t, "https://shop.example.com/login.php?id=12&x=y")   # characters in FIT
    pre = _prefix(t, "https://other.example.net/a/b/c?z=1")
    swap = ContextSwap({tuple(real[0]): [pre]}, t.marker_id("QRY"), p=1.0, seed=1, max_len=256)
    ids, mask = swap(real)
    q = real[0].index(t.marker_id("QRY"))
    assert ids == pre + real[0][q + 1:]
    assert len(mask) == len(ids) - 1
    assert mask[:len(pre) - 1] == [0.0] * (len(pre) - 1)       # every target in the prefix
    assert mask[len(pre) - 1:] == list(real[1][q:])           # the real query's own weights
    assert t.decode(ids).endswith("?id=12&x=y")
    assert swap.counts == {"eligible_draws": 1, "swapped": 1, "too_long": 0}


def test_c_leaves_unlisted_rows_and_respects_p_and_the_length_limit():
    t = tok()
    real = _row(t, "https://shop.example.com/login.php?id=123")
    other = _row(t, "https://bank.example.org/?q=1")
    pre = _prefix(t, "https://x.example.net/p?z=1")
    never = ContextSwap({tuple(real[0]): [pre]}, t.marker_id("QRY"), p=0.0, seed=1, max_len=256)
    assert never(real) is real and never(other) is other
    assert never.counts["eligible_draws"] == 1 and never.counts["swapped"] == 0
    half = ContextSwap({tuple(real[0]): [pre]}, t.marker_id("QRY"), p=0.5, seed=7, max_len=256)
    for _ in range(4000):
        half(real)
    share = half.counts["swapped"] / half.counts["eligible_draws"]
    assert abs(share - 0.5) < 3 * (0.25 / 4000) ** 0.5               # V5's band
    short = ContextSwap({tuple(real[0]): [pre]}, t.marker_id("QRY"), p=1.0, seed=1,
                        max_len=len(real[0]) - 1)
    long_pre = pre[:-1] + [t.stoi["a"]] * 40 + [t.marker_id("QRY")]
    short.table[tuple(real[0])] = [long_pre]
    assert short(real) is real and short.counts["too_long"] == 1


def test_c_refuses_a_prefix_that_does_not_end_at_the_query():
    t = tok()
    real = _row(t, "https://shop.example.com/a?id=1")
    with pytest.raises(ValueError):
        ContextSwap({tuple(real[0]): [[t.bos_id, t.stoi["a"]]]}, t.marker_id("QRY"), p=0.5,
                    seed=1, max_len=256)


def _artifact(tmp_path, t, rows):
    path = tmp_path / "prefixes.json.gz"
    with gzip.open(path, "wt", encoding="utf-8") as fh:
        json.dump({"tokenizer": t.state_dict(),
                   "rows": [{"ids": list(r[0]), "prefixes": [p]} for r, p in rows]}, fh)
    return path


def _tiny_training(t, tmp_path, **cfg):
    urls = ["https://a.example.com/x?id=1", "https://b.example.com/?q=2",
            "https://c.example.com/login", "https://d.example.com"] * 6
    labels = ["phishing", "phishing", "benign", "benign"] * 6
    enc, _ = encode_corpus(urls, labels, t, 128, corpora=["name"] * len(urls))
    data = URLDataset(enc)
    m = model_for(t, max_len=128)
    out = tmp_path / ("on" if cfg.get("context_prefixes") else "off")
    res = train(m, data, data, t, TrainConfig(steps=3, batch_size=4, eval_every=3,
                                              eval_batches=1, warmup=1, **cfg),
                device=torch.device("cpu"), out_dir=out, log_every=10)
    return m, res, data


def test_c_in_training_is_off_by_default_and_counts_when_on(tmp_path):
    t = tok()
    torch.manual_seed(0)
    _, off, data = _tiny_training(t, tmp_path)
    assert "context_swap" not in off
    real = _row(t, "https://a.example.com/x?id=1")
    path = _artifact(tmp_path, t, [(real, _prefix(t, "https://z.example.net/q/r?w=1"))])
    m, on, _ = _tiny_training(t, tmp_path, context_prefixes=str(path), context_p=0.5)
    cs = on["context_swap"]
    assert cs["listed_rows"] == 1 and cs["matched_rows"] == 1 and cs["p"] == 0.5
    assert cs["eligible_draws"] >= cs["swapped"] >= 0
    save_checkpoint(m, t, TrainConfig(context_prefixes=str(path), context_p=0.5),
                    tmp_path / "on.pt")
    save_checkpoint(m, t, TrainConfig(), tmp_path / "off.pt")
    on_cfg = torch.load(tmp_path / "on.pt", weights_only=False)["train_config"]
    off_cfg = torch.load(tmp_path / "off.pt", weights_only=False)["train_config"]
    assert on_cfg["context_p"] == 0.5 and "context_prefixes" not in off_cfg
    load_checkpoint(tmp_path / "off.pt")                  # earlier layout still loads


def test_c_refuses_an_artifact_built_for_another_tokenizer(tmp_path):
    t = tok()
    other = URLTokenizer(include_frame=True, include_fields=True, root_slash=True,
                         presence_plan=True, slash_plan=True).fit(FIT + ["https://zz.qq/é"])
    real = _row(t, "https://a.example.com/x?id=1")
    path = _artifact(tmp_path, t, [(real, _prefix(t, "https://z.example.net/q?w=1"))])
    with pytest.raises(ValueError):
        ContextSwap.load(path, other, 0.5, 1, 128)


# ---- C through uie.kaggle_train.run, end to end on a tiny bundle ------------------------

TINY_TRAIN = ["https://a.example.com/x?id=1", "https://b.example.org/?q=2", "https://c.example.net/login",
              "https://d.example.com", "https://e.example.com/p/q?r=1", "https://f.example.org/"]


def _write_bundle(tmp_path):
    def w(name, urls, label):
        with gzip.open(tmp_path / f"{name}.jsonl.gz", "wt", encoding="utf-8") as fh:
            for u in urls:
                fh.write(json.dumps({"url": u, "label": label}) + "\n")
    w("name_train", TINY_TRAIN * 3, "benign")
    w("structure_train", TINY_TRAIN[:2], "benign")
    w("phishing_name_train", TINY_TRAIN * 3, "phishing")
    w("phishing_structure_train", TINY_TRAIN[:2], "phishing")
    for v in ("name_val", "structure_val", "phishing_name_val", "phishing_structure_val"):
        w(v, TINY_TRAIN[:4], "phishing" if v.startswith("phishing") else "benign")
    return tmp_path


def test_c_through_kaggle_run_end_to_end(tmp_path):
    from uie.kaggle_train import STREAMS, load_streams, run
    (tmp_path / "data").mkdir()
    data = _write_bundle(tmp_path / "data")
    tr_urls = load_streams(data, STREAMS)[0]
    t = URLTokenizer(languages=["und"], include_frame=True, include_fields=True, root_slash=True,
                     presence_plan=True, slash_plan=True).fit(tr_urls, min_count=1)
    real = _row(t, "https://a.example.com/x?id=1")
    path = _artifact(tmp_path, t, [(real, _prefix(t, "https://e.example.com/p/q?r=1"))])
    out = tmp_path / "out"
    res = run(data, out, steps=2, batch_size=4, d_model=16, n_layers=1, n_heads=2, eval_every=2,
              frame=True, fields=True, presence_plan=True, root_slash=True, slash_plan=True,
              seed_init=True, loss_norm="fixed", loss_constant=100.0, max_len=128,
              context_prefixes=str(path), context_p=0.5)
    cs = res["context_swap"]
    assert cs["listed_rows"] == 1 and cs["matched_rows"] == 1 and cs["p"] == 0.5
    tc = torch.load(out / "final.pt", weights_only=False)["train_config"]
    assert tc["context_prefixes"] == str(path) and tc["context_p"] == 0.5


def test_c_loads_the_artifact_gzipped_or_plain(tmp_path):
    """Kaggle decompresses .gz files inside an uploaded dataset, so the bundle carries
    the artifact plain; the laptop keeps it gzipped. Both must load identically."""
    t = tok()
    real = _row(t, "https://a.example.com/x?id=1")
    gz = _artifact(tmp_path, t, [(real, _prefix(t, "https://z.example.net/q?w=1"))])
    plain = tmp_path / "prefixes.json"
    plain.write_bytes(gzip.decompress(gz.read_bytes()))
    a = ContextSwap.load(gz, t, 0.5, 1, 128)
    b = ContextSwap.load(plain, t, 0.5, 1, 128)
    assert a.table == b.table and len(a.table) == 1
