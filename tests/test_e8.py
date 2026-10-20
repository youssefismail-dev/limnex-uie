"""E8 (N1) -- the root-slash token and the presence plan (research/native/PREREG_E8.md).

Pins, per the registration:
- both flags off: every earlier tokenizer state, vocabulary and encoding is
  byte-identical, and earlier generation paths never meet the new code
  (the E7/HC1/DC1 fixtures in their own files prove the sampler);
- root slash: `host/` and `host/?q` carry `<RS>` where `<PTH>` would be, round
  trip exactly, and never count as a path -- the gate's reading is unchanged;
- presence plan: the TRUE plan follows the suffix, is never hidden, is trained
  as a target, and in generation is supplied (FULL CONTROL, both fields) or
  filled by the model from each slot's two legal tokens (FULL FREE), with the
  off-grammar mass recorded; partial plans, masking and a second prediction are
  refused.
"""
from types import SimpleNamespace

import pytest
import torch

from test_e7_fields import FIT, e7_tok, tiny_model
from uie.evaluation import FIELDS
from uie.model import ModelConfig, URLTransformer, generate
from uie.tokenizer import URLTokenizer
from uie.train import FieldDropout, TrainConfig, URLDataset, encode_corpus, field_mask, train

URLS = ["a.com/", "a.com", "a.com/?q=1", "a.com?q=1", "x.a.com/p/q?z=2", "a.co.uk//",
        "shop.b.de/cart", "c.org/?", "d.net/#top"]


def e8_tok(root_slash=True, presence_plan=True):
    return URLTokenizer(include_frame=True, include_fields=True, root_slash=root_slash,
                        presence_plan=presence_plan).fit(FIT + URLS)


def names(tok, ids):
    return [tok.itos[i] for i in ids]


class PlanStub(torch.nn.Module):
    """Asks for its suffix first (as the sampler requires before it injects
    one); uniform over the whole vocabulary otherwise, so a free plan slot sees
    both legal tokens AND mass elsewhere; likely to end once the label opened.
    A random tiny model often never reaches `<SUF>`, and then there is no
    suffix and no plan to test."""

    def __init__(self, tok):
        super().__init__()
        self.cfg = SimpleNamespace(max_len=64)
        self.anchor = torch.nn.Parameter(torch.zeros(1))
        self.suf, self.lbl, self.eos = tok.marker_id("SUF"), tok.marker_id("LBL"), tok.eos_id
        self.v = tok.vocab_size

    def forward(self, ids):
        out = torch.zeros(ids.shape[0], ids.shape[1], self.v)
        last = out[:, -1]
        last[~(ids == self.suf).any(dim=1), self.suf] = 50.0
        last[(ids == self.lbl).any(dim=1), self.eos] = 4.0
        return out


def normalised(u):
    """E3 section 7.3: an EMPTY query or fragment normalises away by design
    (`a.com/x?` is `a.com/x`). Unchanged by E8; only the root slash became exact."""
    return u[:-1] if u.endswith(("?", "#")) else u


# ---- flags off: nothing earlier moves ----------------------------------------------

def test_flags_off_leave_every_earlier_tokenizer_byte_identical():
    for kw in ({}, {"include_frame": True}, {"include_frame": True, "include_fields": True}):
        old = URLTokenizer(**kw).fit(FIT)
        assert "root_slash" not in old.state_dict() and "presence_plan" not in old.state_dict()
        back = URLTokenizer.from_state_dict(old.state_dict())
        assert back.itos == old.itos and not back.root_slash and not back.presence_plan
        for u in URLS:
            assert back.encode(u, "benign") == old.encode(u, "benign")
    assert "<RS>" not in e7_tok().stoi


def test_the_new_flags_are_recorded_only_when_on_and_round_trip():
    tok = e8_tok()
    state = tok.state_dict()
    assert state["root_slash"] is True and state["presence_plan"] is True
    back = URLTokenizer.from_state_dict(state)
    assert back.itos == tok.itos and back.root_slash and back.presence_plan
    assert tok.vocab_size == e8_tok(root_slash=False).vocab_size + 1


def test_presence_plan_needs_the_presence_tokens():
    with pytest.raises(ValueError):
        URLTokenizer(include_frame=True, presence_plan=True)


# ---- root slash ------------------------------------------------------------------------

def test_root_slash_encodes_where_the_path_would_be_and_round_trips():
    tok = e8_tok(presence_plan=False)
    rs, pth = tok.rs_id, tok.marker_id("PTH")
    for u in URLS:
        ids = tok.encode(u, "benign")
        assert tok.decode(ids) == normalised(u), u  # `a.com/` is exact now
        has_rs = rs in ids
        assert has_rs == (u.split("#")[0].split("?")[0].endswith("/")
                          and not FIELDS["PTH"](u)), u
        assert not (has_rs and pth in ids)          # a root slash is never a path
        assert bool(FIELDS["PTH"](tok.decode(ids))) == bool(FIELDS["PTH"](u))
    ids = tok.encode("a.com/?q=1", "benign")
    assert names(tok, ids[-6:]) == ["<RS>", "<QRY>", "q", "=", "1", "<eos>"]


def test_root_slash_is_structural_and_off_without_the_flag():
    tok = e8_tok(presence_plan=False)
    assert tok.rs_id in tok.struct_ids()
    plain = URLTokenizer(include_frame=True).fit(FIT + URLS)
    assert plain.encode("a.com/", "benign") == plain.encode("a.com", "benign")
    with pytest.raises(ValueError):
        plain.rs_id


# ---- presence plan: encoding ------------------------------------------------------------

def test_the_true_plan_follows_the_suffix_and_the_prefix_has_no_controls():
    tok = e8_tok()
    for u in URLS:
        ids = tok.encode(u, "phishing")
        toks = names(tok, ids)
        assert not any(t.startswith(("<P:", "<Q:")) for t in toks[:3])   # BOS, class, frame
        suf, lbl = toks.index("<SUF>"), toks.index("<LBL>")
        want = [f"<P:{int(bool(FIELDS['PTH'](u)))}>", f"<Q:{int(bool(FIELDS['QRY'](u)))}>"]
        assert toks[lbl - 2:lbl] == want, (u, toks)
        assert all(len(t) == 1 for t in toks[suf + 1:lbl - 2])            # suffix chars only
        assert "?" not in "".join(t for t in toks if t.startswith(("<P:", "<Q:")))


def test_a_plan_is_never_hidden_and_never_partial():
    tok = e8_tok()
    with pytest.raises(ValueError):
        tok.encode("a.com/x", "benign", hide_fields=("PTH",))
    with pytest.raises(ValueError):
        tok.field_control_positions()
    with pytest.raises(ValueError):
        FieldDropout(tok, 0.5, seed=1)
    for bad in ({"PTH": True}, {"PTH": True, "QRY": None}, {"PTH": 1, "QRY": 0}):
        with pytest.raises(ValueError):
            tok.plan_ids(bad)
    assert tok.plan_ids({"PTH": True, "QRY": False}) == [tok.plan_slots()[0][1],
                                                        tok.plan_slots()[1][0]]


def test_the_plan_and_the_root_slash_are_trained_as_shape():
    tok = e8_tok()
    ids = tok.encode("a.com/?q=1", "benign")
    mask = field_mask(ids, tok, "name")
    plan_tokens = {t for slot in tok.plan_slots() for t in slot}
    for pos, target in enumerate(ids[1:]):
        if target in plan_tokens or target == tok.rs_id:
            assert mask[pos] == 1.0, tok.itos[target]
    # E7's controls stay supplied inputs (weight 0) -- its mask is unchanged
    e7 = e7_tok()
    e7_ids = e7.encode("a.com/x?q=1", "benign")
    e7_mask = field_mask(e7_ids, e7, "name")
    ctrl = set(e7.field_control_positions().values())
    assert all(e7_mask[p - 1] == 0.0 for p in ctrl)


# ---- presence plan: generation ------------------------------------------------------------

def test_full_control_injects_the_supplied_plan_after_the_suffix():
    tok = e8_tok()
    model = PlanStub(tok)
    for plan in ({"PTH": True, "QRY": True}, {"PTH": False, "QRY": True},
                 {"PTH": True, "QRY": False}, {"PTH": False, "QRY": False}):
        ids = []
        trace = []
        generate(model, tok, label="benign", n=6, seed=3, suffix="com", fields=plan,
                 ids_out=ids, plan_trace=trace, batch_size=4)
        for row in ids:
            toks = names(tok, row)
            s = toks.index("<SUF>")
            assert toks[s + 1:s + 4] == ["c", "o", "m"]
            assert row[s + 4:s + 6] == tok.plan_ids(plan) and toks[s + 6] == "<LBL>"
        assert trace == [0.0] * 6


def test_full_free_fills_each_slot_from_its_two_legal_tokens_and_records_the_rest():
    tok = e8_tok()
    model = PlanStub(tok)
    ids, trace = [], []
    urls = generate(model, tok, label="phishing", n=24, seed=11, suffix="de",
                    ids_out=ids, plan_trace=trace, batch_size=5)
    (p0, p1), (q0, q1) = tok.plan_slots()
    seen = set()
    for row in ids:
        s = names(tok, row).index("<SUF>")
        assert row[s + 3] in (p0, p1) and row[s + 4] in (q0, q1)
        seen.add((row[s + 3], row[s + 4]))
        assert tok.itos[row[s + 5]] == "<LBL>"
    # uniform logits: each slot's pair holds 2/V, so 1 - 2/V is recorded per slot
    per_slot = 1.0 - 2.0 / tok.vocab_size
    assert len(trace) == 24 and all(abs(m - 2 * per_slot) < 1e-5 for m in trace)
    assert len(seen) > 1                     # the plan is sampled, not fixed
    again = generate(model, tok, label="phishing", n=24, seed=11, suffix="de", batch_size=5)
    assert again == urls                     # deterministic for a seed


def test_plan_mode_refuses_partial_plans_masking_and_a_second_prediction():
    tok = e8_tok()
    model = tiny_model(tok)
    kw = dict(label="benign", n=2, seed=1, suffix="com")
    with pytest.raises(ValueError):
        generate(model, tok, fields={"QRY": True}, **kw)
    with pytest.raises(ValueError):
        generate(model, tok, fields={"PTH": True, "QRY": True}, enforce="mask", **kw)
    with pytest.raises(ValueError):
        generate(model, tok, blend={"QRY": 0.5}, **kw)
    with pytest.raises(ValueError):
        generate(model, tok, label="benign", n=2, seed=1, fields={"PTH": True, "QRY": True})
    assert len(generate(model, tok, label="benign", n=3, seed=1)) == 3   # free, no suffix


def test_a_generated_root_slash_decodes_as_a_path_with_no_body():
    """At the token level: `<RS>` with no `<PTH>` is an empty-bodied path, which
    renders as the root slash. (From the STRING, an untrained model's garbage
    can hide a delimiter inside another field -- C1's known effect -- so the
    string-level claim is proven on real URLs above, not on random output.)"""
    tok = e8_tok()
    model = PlanStub(tok)
    ids = []
    generate(model, tok, label="benign", n=60, seed=5, suffix="com", ids_out=ids,
             fields={"PTH": False, "QRY": False})
    checked = 0
    for row in ids:
        if tok.rs_id in row and tok.marker_id("PTH") not in row:
            assert tok.decode_fields(row).path == ""
            checked += 1
    assert checked > 0


# ---- the generation calls (research/native/e8_common.py) --------------------------------

def _common():
    import importlib.util
    from pathlib import Path
    spec = importlib.util.spec_from_file_location(
        "e8_common_under_test",
        Path(__file__).resolve().parents[1] / "research" / "native" / "e8_common.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_the_job_list_and_the_seeds_are_as_registered():
    K = _common()
    jobs = K.jobs()
    assert len(jobs) == 4 * 2 * 5 + 5 * 2 * 4 * 5 + 2 * 5 == 250
    assert len({j["file"] for j in jobs}) == 250
    # free seeds ARE the canonical gate's: E7's HC1 C2 files pair with every arm
    assert [K.seed_free(g) for g in range(5)] == [20260807, 20261807, 20262807,
                                                  20263807, 20264807]
    ctrl = [K.seed_ctrl(l, s, g) for l in K.LABELS for s in K.STATES for g in range(5)]
    assert len(set(ctrl)) == len(ctrl)
    for seed in ctrl:                      # CR1's joint blocks, offset 9e8
        assert (seed - 20261001) % 1_000_000_000 == 900_000_000
    assert K.n_ctrl("phishing", "P1Q1") == K.n_ctrl("phishing", "P0Q1") == 1_000
    assert K.n_ctrl("benign", "P0Q1") == 3_000


def test_native_control_is_one_draw_per_url_with_the_plan_supplied():
    from uie.controlled import StateSuffixTables
    K = _common()
    tok = e8_tok()
    tables = StateSuffixTables({"benign": {"P1_Q1": {"com": 2, "de": 1}}})
    ids = []
    import uie.model as M
    real = M.generate
    calls = []

    def counting(*a, **kw):
        calls.append(kw["n"])
        kw["ids_out"] = ids
        return real(*a, **kw)

    M.generate = counting
    try:
        urls = K.control_urls(PlanStub(tok), tok, "benign", "P1Q1", 0, tables, n=12)
    finally:
        M.generate = real
    assert len(urls) == 12 and sum(calls) == 12      # nothing redrawn
    for row in ids:
        s = names(tok, row).index("<SUF>")
        after = names(tok, row)[s + 1:]
        assert tok.plan_ids({"PTH": True, "QRY": True}) == [
            row[s + 1 + after.index("<LBL>") - 2], row[s + 1 + after.index("<LBL>") - 1]]


# ---- training ----------------------------------------------------------------------------

def test_plan_and_root_slash_models_train_end_to_end_without_dropout():
    tok = e8_tok()
    enc, _ = encode_corpus(URLS * 4, ["benign"] * len(URLS * 4), tok, 64,
                           corpora=["name"] * len(URLS * 4))
    ds = URLDataset(enc)
    cfg = ModelConfig(vocab_size=tok.vocab_size, max_len=64, d_model=16, n_layers=1,
                      n_heads=2, d_ff=32)
    res = train(URLTransformer(cfg), ds, ds, tok,
                TrainConfig(steps=3, batch_size=4, warmup=1, eval_every=5, eval_batches=1))
    assert res is not None
    with pytest.raises(ValueError):
        train(URLTransformer(cfg), ds, ds, tok,
              TrainConfig(steps=1, batch_size=4, warmup=1, eval_every=5, eval_batches=1,
                          field_dropout=0.5))
