"""HC1 -- hard generation-time presence rules (research/hc1/PREREG_HC1.md).

The first test is the one that matters most: with `enforce` off, generate() must
be bit-identical to the code before HC1, checked against 96 configurations
captured from that code (tests/fixtures_e7_generate.json).

The rules are then tested on a STUB model with fixed preferences, so the
unconstrained generator can be made to misbehave on purpose -- a test that
cannot fail proves nothing.
"""
import json
import math
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from test_e7_fields import FIT, e7_tok, tiny_model
from uie.evaluation import FIELDS
from uie.model import generate
from uie.suffixes import SuffixTable, generate_with_suffixes
from uie.tokenizer import URLTokenizer

FIXTURE = json.loads((Path(__file__).parent / "fixtures_e7_generate.json")
                     .read_text(encoding="utf-8"))


# ---- 1. off means off -------------------------------------------------------------

@pytest.mark.reference_machine
def test_enforce_off_is_bit_identical_to_the_pre_hc1_code():
    tok = e7_tok()
    model = tiny_model(tok)
    for case in FIXTURE["cases"]:
        fb = {"PTH": {tok.eos_id: -0.3}} if case["field_bias"] else None
        kw = dict(label=case["label"], n=len(case["urls"]), seed=case["seed"],
                  fields=case["fields"], decoder=case["decoder"], field_bias=fb,
                  suffix=case["suffix"])
        assert generate(model, tok, **kw) == case["urls"], case
        trace = []                      # observing must not change the draw
        assert generate(model, tok, trace=trace, **kw) == case["urls"], case
        assert len(trace) == len(case["urls"])
        assert all(t["log_keep"] == 0.0 and t["masked_steps"] == 0 for t in trace)


# ---- 2. a stub whose preferences a test controls ----------------------------------

class Stub(torch.nn.Module):
    """Fixed next-token logits: -20 for everything except `prefer`. Until a row
    has its [SUF] it asks for one, so the supplied suffix is injected; markers
    in `once` cannot repeat."""

    def __init__(self, tok, prefer, once=(), max_len=64):
        super().__init__()
        self.cfg = SimpleNamespace(max_len=max_len)
        self.anchor = torch.nn.Parameter(torch.zeros(1))
        self.bias = torch.full((tok.vocab_size,), -20.0)
        for token, value in prefer.items():
            self.bias[token] = value
        self.suf = tok.marker_id("SUF")
        self.once = [tok.marker_id(m) for m in once]

    def forward(self, ids):
        out = self.bias.repeat(ids.shape[0], ids.shape[1], 1)
        out[~(ids == self.suf).any(dim=1), -1, self.suf] = 50.0
        for m in self.once:
            out[(ids == m).any(dim=1), -1, m] = -20.0
        return out


def stub(tok, once=(), **prefer):
    ids = {"EOS": tok.eos_id, "a": tok.stoi["a"]}
    ids.update({m: tok.marker_id(m) for m in ("PTH", "QRY", "FRG")})
    ids.update({k: tok.stoi[c] for k, c in (("slash", "/"), ("qmark", "?"),
                                            ("hash", "#")) if c in tok.stoi})
    return Stub(tok, {ids[k]: v for k, v in prefer.items()}, once=once)


def delim_tok():
    """A vocabulary holding all three delimiters as CONTENT characters: '?'
    inside a query, '#' inside a fragment, '/' inside a path."""
    return URLTokenizer(include_frame=True, include_fields=True).fit(
        FIT + ["https://a.example.com/x/y?q?r#f#g"])


def honours(url, fields):
    return all(bool(FIELDS[f](url)) == want for f, want in fields.items())


def run(model, tok, fields, enforce, n=48, seed=7):
    trace = []
    urls = generate(model, tok, label="benign", n=n, seed=seed, suffix="com",
                    fields=fields, enforce=enforce, trace=trace)
    return urls, trace


# ---- 3. the rules hold, where the model would break them --------------------------

# Each marker at most once (a repeated field decodes to its FIRST, empty value,
# which would let a stub honour a request by accident).
ONCE = ("PTH", "QRY", "FRG")
CASES = [  # (stub preferences, request) -- each stub breaks its request
    (dict(EOS=3.0, a=1.0, PTH=1.0, QRY=1.0), {"PTH": True}),       # ends early
    (dict(EOS=3.0, a=1.0, PTH=1.0, QRY=1.0), {"QRY": True}),
    (dict(EOS=3.0, a=1.0, PTH=1.0, QRY=1.0), {"PTH": True, "QRY": True}),
    (dict(EOS=2.0, a=3.0, PTH=5.0), {"PTH": False}),                # opens a path
    (dict(EOS=2.0, a=3.0, QRY=5.0), {"QRY": False}),                # opens a query
    (dict(EOS=3.0, a=1.0, PTH=5.0, QRY=1.0), {"PTH": False, "QRY": True}),
    (dict(EOS=3.0, a=1.0, PTH=1.0, QRY=5.0), {"PTH": True, "QRY": False}),
]


@pytest.mark.parametrize("prefer,fields", CASES)
def test_the_unconstrained_stub_breaks_the_request(prefer, fields):
    tok = e7_tok()
    urls, _ = run(stub(tok, once=ONCE, **prefer), tok, fields, enforce=None)
    assert sum(honours(u, fields) for u in urls) < len(urls) / 2


@pytest.mark.parametrize("prefer,fields", CASES)
def test_mask_makes_every_request_hold(prefer, fields):
    tok = e7_tok()
    urls, trace = run(stub(tok, once=ONCE, **prefer), tok, fields, enforce="mask")
    assert all(honours(u, fields) for u in urls), [u for u in urls
                                                   if not honours(u, fields)]
    assert not any(t["truncated"] for t in trace)
    assert all(t["satisfied"] for t in trace)
    assert any(t["log_keep"] < 0 for t in trace)     # something WAS removed


def test_an_opened_field_cannot_close_empty():
    tok = e7_tok()
    empty = stub(tok, once=("QRY",), EOS=4.0, QRY=5.0, a=0.0)  # [QRY] then EOS
    loose, _ = run(empty, tok, {"QRY": True}, enforce=None)
    assert sum(bool(FIELDS["QRY"](u)) for u in loose) < len(loose) / 2
    held, _ = run(empty, tok, {"QRY": True}, enforce="mask")
    assert all(FIELDS["QRY"](u) for u in held)


# ---- 3b. presence is what the gate reads from the STRING ------------------------------
# The parser starts the fragment at the first '#', the query at the first '?'
# before it, the path at the first '/' before both. A delimiter written inside
# another field creates or swallows a part with no marker at all.

DELIMITER_CASES = [  # (stub preferences, request) -- each breaks it by a delimiter
    (dict(slash=5.0, a=1.0, EOS=2.0), {"PTH": False}),               # '/' in the host
    (dict(PTH=5.0, qmark=4.0, a=2.0, EOS=1.0), {"QRY": False}),      # '?' in the path
    (dict(PTH=5.0, qmark=5.0, a=1.0, EOS=3.0), {"PTH": True}),       # path opens with '?'
    (dict(QRY=5.0, hash=5.0, a=1.0, EOS=3.0), {"QRY": True}),        # query opens with '#'
    (dict(hash=5.0, QRY=1.0, a=1.0, EOS=3.0), {"QRY": True}),        # '#' in the host
    (dict(qmark=5.0, PTH=1.0, a=1.0, EOS=3.0), {"PTH": True}),       # '?' in the host
]


@pytest.mark.parametrize("prefer,fields", DELIMITER_CASES)
def test_a_delimiter_in_the_wrong_field_breaks_the_unconstrained_stub(prefer, fields):
    tok = delim_tok()
    urls, _ = run(stub(tok, once=ONCE, **prefer), tok, fields, enforce=None)
    assert sum(honours(u, fields) for u in urls) < len(urls) / 2


@pytest.mark.parametrize("prefer,fields", DELIMITER_CASES)
def test_the_rules_hold_at_the_string_level(prefer, fields):
    tok = delim_tok()
    urls, trace = run(stub(tok, once=ONCE, **prefer), tok, fields, enforce="mask")
    assert all(honours(u, fields) for u in urls), [u for u in urls
                                                   if not honours(u, fields)]
    assert not any(t["truncated"] for t in trace)


# ---- 4. the weight is the model's own probability ---------------------------------

def test_mean_weight_equals_the_unconstrained_success_rate():
    """E_q[w] = P_p(rules hold). This stub makes both exactly 1/4: before [PTH],
    EOS, [PTH] and 'a' are equally likely; straight after it, EOS and 'a'. So a
    path survives with (1/2)(1/2); masking removes 1/3 per pre-path step and 1/2
    at the empty-path step, w = (2/3)^k / 2 with k ~ Geometric(1/2)."""
    tok = e7_tok()
    model = stub(tok, once=("PTH",), EOS=1.0, PTH=1.0, a=1.0)
    urls, _ = run(model, tok, {"PTH": True}, enforce=None, n=4000, seed=11)
    p_hat = sum(honours(u, {"PTH": True}) for u in urls) / len(urls)
    _, trace = run(model, tok, {"PTH": True}, enforce="mask", n=4000, seed=12)
    w = [math.exp(t["log_keep"]) for t in trace]
    mean_w = sum(w) / len(w)
    assert abs(mean_w - 0.25) < 0.01, mean_w
    assert abs(p_hat - 0.25) < 0.03, p_hat


# ---- 5. refusals, and the gate's own path --------------------------------------------

@pytest.mark.parametrize("enforce,fields", [("soft", {"QRY": True}),
                                            ("mask", None),
                                            ("mask", {"QRY": None})])
def test_meaningless_requests_are_refused(enforce, fields):
    tok = e7_tok()
    with pytest.raises(ValueError):
        generate(tiny_model(tok), tok, label="benign", n=2, seed=1,
                 fields=fields, enforce=enforce)


def test_a_checkpoint_without_controls_cannot_enforce():
    tok = URLTokenizer(include_frame=True).fit(FIT)
    with pytest.raises(ValueError):
        generate(tiny_model(tok), tok, label="benign", n=2, seed=1,
                 fields={"QRY": None}, enforce="mask")


def test_the_gate_path_carries_the_rules():
    tok = e7_tok()
    table = SuffixTable({"benign": {"com": 3, "org": 1}})
    urls = generate_with_suffixes(stub(tok, once=ONCE, EOS=3.0, a=1.0, PTH=1.0,
                                       QRY=1.0),
                                  tok, label="benign", n=24, table=table, seed=3,
                                  fields={"QRY": True}, enforce="mask")
    assert all(FIELDS["QRY"](u) for u in urls)
