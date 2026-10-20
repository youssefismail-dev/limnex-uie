"""E7 -- field-presence controls (research/e7/PREREG_E7.md).

The first test is the one that matters most: with the controls OFF, the
tokenizer and generate() must be bit-identical to the code before E7, checked
against a fixture captured from that code (tests/fixtures_e5_path.json).
"""
import json
import random
from pathlib import Path

import pytest
import torch

from uie.evaluation import FIELDS
from uie.model import ModelConfig, URLTransformer, generate
from uie.tokenizer import URLTokenizer
from uie.train import (FieldDropout, TrainConfig, URLDataset, batches,
                       encode_corpus, field_mask, train)

FIXTURE = json.loads((Path(__file__).parent / "fixtures_e5_path.json")
                     .read_text(encoding="utf-8"))
FIT = FIXTURE["fit"]


def e7_tok():
    return URLTokenizer(include_frame=True, include_fields=True).fit(FIT)


def tiny_model(tok, seed=0):
    torch.manual_seed(seed)
    m = URLTransformer(ModelConfig(vocab_size=tok.vocab_size, max_len=64,
                                   d_model=32, n_layers=1, n_heads=2, d_ff=64))
    m.eval()
    return m


# ---- 1. the E5 path did not move -----------------------------------------------

def test_e5_path_is_bit_identical_to_the_pre_e7_code():
    tok = URLTokenizer(include_frame=True).fit(FIT)
    assert tok.state_dict() == FIXTURE["state"]
    assert tok.vocab_size == FIXTURE["vocab"]
    for key, want in FIXTURE["encodings"].items():
        url, label, frame = key.split("|")
        assert tok.encode(url, label, frame=frame) == want, key
    model = tiny_model(tok)
    for dec, want in FIXTURE["generate"].items():
        got = generate(model, tok, label="phishing", n=6, frame="domain",
                       decoder=dec, seed=11, batch_size=3)
        assert got == want, dec


def test_a_pre_e7_state_dict_loads_with_controls_off():
    state = dict(FIXTURE["state"])
    assert "include_fields" not in state
    assert URLTokenizer.from_state_dict(state).include_fields is False


# ---- 2. vocabulary and persistence ---------------------------------------------------

def test_controls_add_exactly_six_tokens():
    assert e7_tok().vocab_size == FIXTURE["vocab"] + 6
    toks = {t for t in e7_tok().itos if t[:3] in ("<P:", "<Q:")}
    assert toks == {"<P:0>", "<P:1>", "<P:?>", "<Q:0>", "<Q:1>", "<Q:?>"}


def test_state_dict_round_trips_the_flag():
    tok = e7_tok()
    assert tok.state_dict()["include_fields"] is True
    back = URLTokenizer.from_state_dict(tok.state_dict())
    assert back.include_fields and back.itos == tok.itos


# ---- 3. presence is the tokenizer's own definition, and the gate's -------------------

@pytest.mark.parametrize("url, pth, qry", [
    ("example.com", False, False),
    ("example.com/", False, False),          # empty path body
    ("example.com/login", True, False),
    ("a.com/x?", True, False),               # empty query
    ("a.com/x?y=1", True, True),
    ("a.com?y=1", False, True),
    ("https://a.b.example.co.uk/x/y?q=1#f", True, True),
])
def test_presence_matches_the_gate_extractors_and_the_markers(url, pth, qry):
    tok = e7_tok()
    ids = tok.encode(url, "benign", frame="domain")
    pos = tok.field_control_positions()
    assert ids[pos["PTH"]] == tok.field_control_id("PTH", pth)
    assert ids[pos["QRY"]] == tok.field_control_id("QRY", qry)
    # the control agrees with what the gate sees ...
    assert bool(FIELDS["PTH"](url)) is pth
    assert bool(FIELDS["QRY"](url)) is qry
    # ... and with what the sequence actually contains
    assert (tok.marker_id("PTH") in ids) is pth
    assert (tok.marker_id("QRY") in ids) is qry


def test_controls_sit_after_class_and_frame():
    tok = e7_tok()
    assert tok.field_control_positions() == {"PTH": 3, "QRY": 4}
    ids = tok.encode("a.com/x?y=1", "phishing", frame="page")
    assert ids[:3] == [tok.bos_id, tok.stoi["<C:phishing>"], tok.stoi["<F:page>"]]


def test_hiding_encodes_the_question_mark_form():
    tok = e7_tok()
    ids = tok.encode("a.com/x?y=1", "benign", hide_fields=("PTH", "QRY"))
    assert ids[3:5] == [tok.stoi["<P:?>"], tok.stoi["<Q:?>"]]
    ids = tok.encode("a.com/x?y=1", "benign", hide_fields=("QRY",))
    assert ids[3:5] == [tok.stoi["<P:1>"], tok.stoi["<Q:?>"]]


def test_hiding_is_refused_where_it_cannot_mean_anything():
    with pytest.raises(KeyError):
        e7_tok().encode("a.com", "benign", hide_fields=("SUB",))
    with pytest.raises(ValueError):
        URLTokenizer(include_frame=True).fit(FIT).encode(
            "a.com", "benign", hide_fields=("QRY",))


# ---- 4. generation controls -----------------------------------------------------------

def test_generation_prefix_defaults_to_unconditional():
    tok = e7_tok()
    assert tok.field_control_ids() == [tok.stoi["<P:?>"], tok.stoi["<Q:?>"]]
    assert tok.field_control_ids({"QRY": True}) == [tok.stoi["<P:?>"],
                                                    tok.stoi["<Q:1>"]]


@pytest.mark.parametrize("bad", [{"SUB": True}, {"QRY": 1.5}, {"PTH": "yes"}])
def test_malformed_requests_are_refused(bad):
    with pytest.raises((KeyError, ValueError)):
        e7_tok().field_control_ids(bad)


def test_generate_runs_with_controls_and_never_leaks_them():
    tok = e7_tok()
    urls = generate(tiny_model(tok), tok, label="benign", n=4, seed=1,
                    fields={"QRY": True, "PTH": False})
    assert len(urls) == 4
    assert not any("<" in u for u in urls)


def test_a_pre_e7_checkpoint_refuses_a_presence_request():
    tok = URLTokenizer(include_frame=True).fit(FIT)
    with pytest.raises(ValueError):
        generate(tiny_model(tok), tok, label="benign", n=2, seed=1,
                 fields={"QRY": True})
    # asking for nothing is not a request, and stays allowed
    generate(tiny_model(tok), tok, label="benign", n=2, seed=1,
             fields={"QRY": None})


# ---- 5. training ----------------------------------------------------------------------

def test_control_targets_are_never_trained():
    """The controls are inputs; field_mask must give their positions weight 0."""
    tok = e7_tok()
    ids = tok.encode("a.com/x?y=1", "benign", frame="domain")
    w = field_mask(ids, tok, "structure")
    pos = tok.field_control_positions()
    for p in pos.values():
        assert w[p - 1] == 0.0          # weights align with ids[1:]


def _row(tok, url="a.com/x?y=1"):
    ids = tok.encode(url, "benign")
    return (ids, field_mask(ids, tok, "structure"))


def test_dropout_zero_leaves_the_row_untouched():
    tok = e7_tok()
    row = _row(tok)
    assert FieldDropout(tok, 0.0, seed=1)(row) is row


def test_dropout_one_hides_both_and_never_mutates_the_dataset():
    tok = e7_tok()
    row = _row(tok)
    before = list(row[0])
    ids, mask = FieldDropout(tok, 1.0, seed=1)(row)
    assert ids[3:5] == [tok.stoi["<P:?>"], tok.stoi["<Q:?>"]]
    assert row[0] == before              # the shared row is intact
    assert mask is row[1]
    assert len(ids) == len(before)       # bucketing is unaffected


def test_dropout_is_deterministic_and_near_its_rate():
    tok = e7_tok()
    row = _row(tok)
    a = [FieldDropout(tok, 0.5, seed=7)(row)[0][3:5] for _ in range(1)]
    d1, d2 = FieldDropout(tok, 0.5, seed=7), FieldDropout(tok, 0.5, seed=7)
    seq1 = [d1(row)[0][3:5] for _ in range(400)]
    seq2 = [d2(row)[0][3:5] for _ in range(400)]
    assert seq1 == seq2 and a
    hidden = sum(pair[1] == tok.stoi["<Q:?>"] for pair in seq1) / len(seq1)
    assert 0.4 < hidden < 0.6


def test_dropout_does_not_change_which_rows_are_batched():
    """It has its own random stream: same seed, same batches, controls aside."""
    tok = e7_tok()
    urls = [f"s{i}.com/p{i}?q={i}" for i in range(40)] + \
           [f"t{i}.com" for i in range(40)]
    enc, _ = encode_corpus(urls, ["benign"] * 80, tok, 64,
                           corpora=["structure"] * 80)
    ds = URLDataset(enc)
    cfg = TrainConfig(batch_size=8, bucket_window=2)
    plain = batches(ds, cfg, tok.pad_id, random.Random(3))
    dropped = batches(ds, cfg, tok.pad_id, random.Random(3),
                      FieldDropout(tok, 0.5, seed=9))
    for _ in range(6):
        (a, wa), (b, wb) = next(plain), next(dropped)
        keep = [i for i in range(a.shape[1]) if i not in (3, 4)]
        assert torch.equal(a[:, keep], b[:, keep])
        assert torch.equal(wa, wb)


def test_encode_corpus_can_hide_for_validation():
    tok = e7_tok()
    enc, _ = encode_corpus(["a.com/x?y=1"], ["benign"], tok, 64,
                           hide_fields=("PTH", "QRY"))
    ids = enc["benign"][0][0]
    assert ids[3:5] == [tok.stoi["<P:?>"], tok.stoi["<Q:?>"]]


def test_a_tiny_e7_training_run_completes():
    tok = e7_tok()
    urls = [f"s{i}.com/p{i}?q={i}" for i in range(30)] + \
           [f"t{i}.com/a" for i in range(30)]
    labels = ["benign"] * 30 + ["phishing"] * 30
    tr, _ = encode_corpus(urls, labels, tok, 64, corpora=["structure"] * 60)
    va, _ = encode_corpus(urls, labels, tok, 64, corpora=["structure"] * 60,
                          hide_fields=("PTH", "QRY"))
    model = tiny_model(tok)
    model.train()
    cfg = TrainConfig(steps=3, batch_size=8, bucket_window=2, eval_every=0,
                      eval_batches=2, field_dropout=0.5)
    out = train(model, URLDataset(tr), URLDataset(va), tok, cfg,
                device=torch.device("cpu"), log_every=0)
    assert set(out["final"]) >= {"benign", "phishing", "mean"}


def test_dropout_without_controls_is_refused():
    tok = URLTokenizer(include_frame=True).fit(FIT)
    with pytest.raises(ValueError):
        FieldDropout(tok, 0.5, seed=1)
