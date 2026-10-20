"""E15-B's library change (research/e15/PREREG_E15.md section 10.4, V-batch, V-mask,
V-mass): new query-bearing rows injected alongside the old ones. Off by default, and
off leaves every earlier path exactly as it was.
"""
import gzip
import hashlib
import json
import random

import pytest
import torch

from test_e7_fields import FIT
from uie.model import ModelConfig, URLTransformer
from uie.tokenizer import URLTokenizer
from uie.train import (QueryInjector, TrainConfig, URLDataset, batch_digest, batches,
                       encode_corpus, field_mask, load_checkpoint, save_checkpoint, train)


def tok():
    return URLTokenizer(include_frame=True, include_fields=True, root_slash=True,
                        presence_plan=True, slash_plan=True).fit(FIT)


def model_for(t, max_len=128, seed=0):
    torch.manual_seed(seed)
    return URLTransformer(ModelConfig(vocab_size=t.vocab_size, max_len=max_len, d_model=16,
                                      n_layers=1, n_heads=2, d_ff=32))


OLD = (["https://a.example.com/x?id=1", "https://b.example.com/?q=2",
        "https://c.example.com/login", "https://d.example.com"] * 6)
OLD_LABELS = ["phishing", "benign", "benign", "benign"] * 6
NEW = [{"url": "https://n1.example.org/?q=12", "label": "benign", "stratum": "P0Q1"},
       {"url": "https://n2.example.net/a?x=1&y=2", "label": "benign", "stratum": "P1Q1"},
       {"url": "https://n3.example.org/b?z=3", "label": "benign", "stratum": "P1Q1"}]


def dataset(t):
    enc, _ = encode_corpus(OLD, OLD_LABELS, t, 128, corpora=["name"] * len(OLD))
    return URLDataset(enc)


def stream_file(tmp_path, rows=NEW):
    p = tmp_path / "qry_extra.jsonl"
    p.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    return p


# ---- the mask and the strata ----------------------------------------------------------

@pytest.mark.parametrize("url", ["https://a.example.com/x?id=12#top", "https://b.example.com?x=1",
                                 "https://a.example.com/?id=1&b=2"])
def test_new_rows_supervise_exactly_the_query_steps(url):
    t = tok()
    ids = t.encode(url, "benign", "und", frame="domain")
    m = QueryInjector.query_only_mask(ids, t)
    w = field_mask(ids, t, "name", 1.0, 4.0)
    assert [x == 1.0 for x in m] == [x == 4.0 for x in w]          # E14 section 2.1's positions
    assert set(m) == {0.0, 1.0} and sum(m) >= 3
    q = ids.index(t.marker_id("QRY"))
    assert m[q - 1] == 0.0                                          # not the <QRY> step


def test_strata_are_read_from_the_rows_own_tokens():
    t = tok()
    inj = QueryInjector(t, {}, {}, 1, 2)
    enc = lambda u, frame="domain", label="benign": t.encode(u, label, "und", frame=frame)  # noqa: E731
    assert inj.stratum_of(enc("https://a.example.com/x?id=1")) == "benign_P1Q1"
    assert inj.stratum_of(enc("https://b.example.com/?q=2")) == "benign_P0Q1"
    assert inj.stratum_of(enc("https://a.example.com/x?id=1", label="phishing")) == "phishing_P1Q1"
    assert inj.stratum_of(enc("https://c.example.com/login")) is None          # no query
    assert inj.stratum_of(enc("https://a.example.com/x?id=1", frame="page")) is None


# ---- injection ---------------------------------------------------------------------------

def test_injection_appends_one_new_row_per_old_stratum_row_and_leaves_old_rows_alone(tmp_path):
    t = tok()
    data = dataset(t)
    with pytest.raises(ValueError):               # a stratum with new rows but no old: p undefined
        QueryInjector.load(stream_file(tmp_path), t, data, 128, 7, 8)
    inj = QueryInjector.load(stream_file(tmp_path, NEW[:1]), t, data, 128, 7, 8)
    assert inj.old_counts == {"benign_P0Q1": 6, "phishing_P1Q1": 6}
    assert set(inj.p) == {"benign_P0Q1"} and inj.p["benign_P0Q1"] == 1 / 6
    # The full-doubling case, built honestly: six new rows for six old (rate 1). Before
    # E17 this check overwrote `inj.p`; the injector now decides from `inj.rate`
    # (research/e17/PREREG_E17.md section 3.2), so the doubling is constructed instead.
    inj = QueryInjector.load(stream_file(tmp_path, NEW[:1] * 6), t, data, 128, 7, 8)
    assert inj.rate["benign_P0Q1"] == inj.p["benign_P0Q1"] == 1.0
    chunk = list(data.by_class["benign"][:4])
    before = [(list(i), list(m)) for i, m in chunk]
    out = inj(chunk)
    assert [(list(i), list(m)) for i, m in out[:len(chunk)]] == before      # untouched, in order
    n_old = sum(inj.stratum_of(i) == "benign_P0Q1" for i, _ in chunk)
    assert len(out) == len(chunk) + n_old                                   # p = 1 at a doubling
    for ids, mask in out[len(chunk):]:
        assert mask == QueryInjector.query_only_mask(ids, t)


def test_a_short_stratum_is_injected_at_n_new_over_n_old(tmp_path):
    t = tok()
    urls = [f"https://s{i}.example.com/?q={i}" for i in range(40)]
    enc, _ = encode_corpus(urls, ["benign"] * 40, t, 128, corpora=["name"] * 40)
    data = URLDataset(enc)
    inj = QueryInjector.load(stream_file(tmp_path, NEW[:1]), t, data, 128, 7, 8)
    assert inj.p["benign_P0Q1"] == pytest.approx(1 / 40)
    for _ in range(400):
        inj(list(data.by_class["benign"]))
    seen = inj.counts["old_rows_seen"]["benign_P0Q1"]
    got = inj.counts["appended_rows"]["benign_P0Q1"]
    assert seen == 16_000 and abs(got / seen - 1 / 40) < 0.006


def test_old_batches_are_the_ones_the_seed_draws_without_injection(tmp_path):
    """V-batch's basis: injection draws from its own generators, so the old rows'
    batches -- and their digest -- equal the no-injection sampler's."""
    t = tok()
    data = dataset(t)
    cfg = TrainConfig(batch_size=4, bucket_window=2, seed=11)
    ref = batch_digest(data, cfg, t.pad_id, 30)
    inj = QueryInjector.load(stream_file(tmp_path, NEW[:1]), t, data, 128, 7, 8)
    d = hashlib.sha256()
    s = batches(data, cfg, t.pad_id, random.Random(cfg.seed), None, inj, d)
    grew = 0
    for _ in range(30):
        ids, w = next(s)
        grew += ids.shape[0] > 4
    assert d.hexdigest() == ref and grew > 0


# ---- training: off by default, recorded when on --------------------------------------------

def test_injection_in_training_is_off_by_default_and_recorded_when_on(tmp_path):
    t = tok()
    data = dataset(t)
    kw = dict(steps=6, batch_size=4, eval_every=6, eval_batches=1, warmup=1,
              loss_norm="fixed", loss_constant=60.0)
    off = train(model_for(t), data, data, t, TrainConfig(**kw), device=torch.device("cpu"),
                out_dir=tmp_path / "off", log_every=100)
    assert "qry_extra" not in off
    path = stream_file(tmp_path, NEW[:1] * 6)
    on = train(model_for(t), data, data, t, TrainConfig(**kw, qry_extra=str(path)),
               device=torch.device("cpu"), out_dir=tmp_path / "on", log_every=100)
    q = on["qry_extra"]
    st = q["strata"]["benign_P0Q1"]
    assert st["appended_rows"] == st["old_rows_seen"] > 0 and st["p"] == 1.0
    assert st["appended_query_positions"] > 0
    assert q["old_batch_digest"] == batch_digest(data, TrainConfig(**kw), t.pad_id, 6)
    assert q["source_sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()
    save_checkpoint(model_for(t), t, TrainConfig(qry_extra=str(path)), tmp_path / "on.pt")
    save_checkpoint(model_for(t), t, TrainConfig(), tmp_path / "off.pt")
    assert torch.load(tmp_path / "on.pt", weights_only=False)["train_config"]["qry_extra"] == str(path)
    assert "qry_extra" not in torch.load(tmp_path / "off.pt", weights_only=False)["train_config"]
    load_checkpoint(tmp_path / "off.pt")


TINY_TRAIN = ["https://a.example.com/x?id=1", "https://b.example.org/?q=2",
              "https://c.example.net/login", "https://d.example.com",
              "https://e.example.com/p/q?r=1", "https://f.example.org/"]


def test_injection_through_kaggle_run_end_to_end(tmp_path):
    from uie.kaggle_train import run
    d = tmp_path / "data"
    d.mkdir()

    def w(name, urls, label):
        with gzip.open(d / f"{name}.jsonl.gz", "wt", encoding="utf-8") as fh:
            for u in urls:
                fh.write(json.dumps({"url": u, "label": label}) + "\n")
    w("name_train", TINY_TRAIN * 3, "benign")
    w("structure_train", TINY_TRAIN[:2], "benign")
    w("phishing_name_train", TINY_TRAIN * 3, "phishing")
    w("phishing_structure_train", TINY_TRAIN[:2], "phishing")
    for v in ("name_val", "structure_val", "phishing_name_val", "phishing_structure_val"):
        w(v, TINY_TRAIN[:4], "phishing" if v.startswith("phishing") else "benign")
    path = stream_file(tmp_path, [{"url": "https://zz.example.org/?q=77", "label": "benign",
                                   "stratum": "P0Q1"}])
    kw = dict(steps=4, batch_size=4, d_model=16, n_layers=1, n_heads=2, eval_every=4,
              frame=True, fields=True, presence_plan=True, root_slash=True, slash_plan=True,
              seed_init=True, loss_norm="fixed", loss_constant=100.0, max_len=128)
    res = run(d, tmp_path / "on", qry_extra=str(path), **kw)
    assert res["qry_extra"]["strata"]["benign_P0Q1"]["N_new"] == 1
    off = run(d, tmp_path / "off", **kw)
    assert off["qry_extra"] is None
