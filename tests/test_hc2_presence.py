"""HC2 -- presence by resampling (research/hc2/PREREG_HC2.md).

With no control set, generate_with_presence must BE the gate's path, byte for
byte. The two resampling modes are then tested on a stub whose query propensity
depends on the suffix it was handed -- ~0.88 for "org", ~0.20 for "com" -- so
the modes' different targets are visible and computable:

    joint  p(suffix | query) ~ 0.5*0.88 / (0.5*0.88 + 0.5*0.20) ~ 0.81 "org"
    held   the suffix draw itself, exactly
"""
import collections
import random
from types import SimpleNamespace

import pytest
import torch

from test_e7_fields import FIT, e7_tok, tiny_model
from uie.evaluation import FIELDS
from uie.model import generate
from uie.presence import ROUND_STRIDE, PresenceUnmet, generate_with_presence, honours
from uie.suffixes import SuffixTable, generate_with_suffixes
from uie.tokenizer import URLTokenizer

TABLE = SuffixTable({"benign": {"com": 1, "org": 1}})


def tok_with_org():
    return URLTokenizer(include_frame=True, include_fields=True).fit(
        FIT + ["https://www.a.org/b?c=d"])


class SuffixStub(torch.nn.Module):
    """Asks for its suffix first; one 'a' of label; then a query with a
    propensity set by whether the supplied suffix contains 'g' ("org") or not
    ("com"); one 'a' of query; then it ends."""

    def __init__(self, tok, q_org=5.0, q_com=1.6):
        super().__init__()
        self.cfg = SimpleNamespace(max_len=64)
        self.anchor = torch.nn.Parameter(torch.zeros(1))
        self.base = torch.full((tok.vocab_size,), -20.0)
        self.base[tok.eos_id] = 3.0
        self.base[tok.stoi["a"]] = 1.0
        self.suf, self.lbl = tok.marker_id("SUF"), tok.marker_id("LBL")
        self.qry, self.a, self.g = tok.marker_id("QRY"), tok.stoi["a"], tok.stoi["g"]
        self.q_org, self.q_com = q_org, q_com

    def forward(self, ids):
        out = self.base.repeat(ids.shape[0], ids.shape[1], 1)
        last = out[:, -1]
        last[:, self.qry] = torch.where((ids == self.g).any(dim=1),
                                        torch.tensor(self.q_org), torch.tensor(self.q_com))
        last[(ids == self.qry).any(dim=1), self.qry] = -20.0          # one query
        last[~(ids == self.suf).any(dim=1), self.suf] = 50.0          # suffix first
        for marker in (self.lbl, self.qry):                           # never empty
            last[ids[:, -1] == marker, self.a] = 10.0
        return out


def suffix_of(url):
    return FIELDS["SUF"](url)


# ---- 1. no control: the gate's own path ---------------------------------------------

@pytest.mark.parametrize("fields", [None, {"QRY": None}, {"PTH": None, "QRY": None}])
def test_no_control_is_the_gate_path_byte_for_byte(fields):
    tok = e7_tok()
    model = tiny_model(tok)
    kw = dict(label="benign", n=12, table=TABLE, seed=5)
    assert generate_with_presence(model, tok, fields=fields, **kw) == \
        generate_with_suffixes(model, tok, **kw)


# ---- 2. both modes hold the control -----------------------------------------------

@pytest.mark.parametrize("hold", [False, True])
def test_every_url_honours_the_request(hold):
    tok = tok_with_org()
    stats = {}
    urls = generate_with_presence(SuffixStub(tok), tok, label="benign", n=200,
                                  table=TABLE, seed=3, fields={"QRY": True},
                                  hold_suffix=hold, stats=stats)
    assert len(urls) == 200 and all(FIELDS["QRY"](u) for u in urls)
    assert stats["draws"] > 200 and stats["first_pass"] < 200     # it DID redraw


def test_the_unconstrained_stub_breaks_the_request():
    tok = tok_with_org()
    urls = generate_with_suffixes(SuffixStub(tok), tok, label="benign", n=200,
                                  table=TABLE, seed=3, fields={"QRY": True})
    assert sum(bool(FIELDS["QRY"](u)) for u in urls) < 150


# ---- 3. what each mode samples ----------------------------------------------------

def test_joint_follows_the_models_suffix_dependence():
    tok = tok_with_org()
    urls = generate_with_presence(SuffixStub(tok), tok, label="benign", n=400,
                                  table=TABLE, seed=9, fields={"QRY": True})
    share = sum(suffix_of(u) == "org" for u in urls) / len(urls)
    assert 0.72 < share < 0.90, share          # p(org | query) ~ 0.81


def test_held_keeps_the_suffix_draw_exactly():
    tok = tok_with_org()
    n, seed = 400, 9
    urls = generate_with_presence(SuffixStub(tok), tok, label="benign", n=n,
                                  table=TABLE, seed=seed, fields={"QRY": True},
                                  hold_suffix=True)
    drawn = TABLE.sample("benign", n, random.Random(seed))
    assert collections.Counter(map(suffix_of, urls)) == collections.Counter(drawn)


def test_held_mode_is_hc1_h3s_rejection_algorithm():
    """An independent transcription of HC1 H3's exact arm must give the same
    URLs (as a multiset): R2 at unit scale."""
    tok = tok_with_org()
    model, n, seed, fields = SuffixStub(tok), 60, 21, {"QRY": True}
    rng = random.Random(seed)
    drawn = TABLE.sample("benign", n, rng)
    groups = sorted(collections.Counter(zip(["domain"] * n, drawn)).items())
    expect = []
    for i, ((frame, suffix), count) in enumerate(groups):
        kept, k = [], 0
        while len(kept) < count and k <= 100:
            got = generate(model, tok, label="benign", n=count - len(kept),
                           suffix=suffix, seed=seed + i + ROUND_STRIDE * k,
                           frame=frame, batch_size=32, fields=fields)
            kept.extend(u for u in got if FIELDS["QRY"](u))
            k += 1
        expect.extend(kept[:count])
    got = generate_with_presence(model, tok, label="benign", n=n, table=TABLE,
                                 seed=seed, fields=fields, hold_suffix=True)
    assert collections.Counter(got) == collections.Counter(expect)


# ---- 4. it fails loudly, is reproducible, and refuses to mix mechanisms -------------------

@pytest.mark.parametrize("hold", [False, True])
def test_an_unreachable_request_raises(hold):
    tok = tok_with_org()
    never = SuffixStub(tok, q_org=-20.0, q_com=-20.0)
    with pytest.raises(PresenceUnmet):
        generate_with_presence(never, tok, label="benign", n=8, table=TABLE, seed=1,
                               fields={"QRY": True}, hold_suffix=hold, max_rounds=3)


def test_same_seed_same_urls():
    tok = tok_with_org()
    kw = dict(label="benign", n=50, table=TABLE, fields={"QRY": True})
    a = generate_with_presence(SuffixStub(tok), tok, seed=4, **kw)
    assert a == generate_with_presence(SuffixStub(tok), tok, seed=4, **kw)
    assert a != generate_with_presence(SuffixStub(tok), tok, seed=5, **kw)


def test_masking_is_refused_inside_resampling():
    tok = tok_with_org()
    with pytest.raises(ValueError):
        generate_with_presence(SuffixStub(tok), tok, label="benign", n=4, table=TABLE,
                               fields={"QRY": True}, enforce="mask")


def test_honours_reads_the_string_as_the_gate_does():
    assert honours("a.com/x?y=1", {"PTH": True, "QRY": True})
    assert not honours("a.com/?y=1", {"PTH": True})      # "/" alone: empty body
    assert honours("a.com/x", {"QRY": False, "PTH": None})
