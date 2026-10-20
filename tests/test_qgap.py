"""QG1's query-hazard instrument (research/qgap/QG_PLAN.md section 5.7).

VQ3: on hand-built sequences under a model whose probabilities are known, y, h and
the routes come out exactly. VQ2: on a model small enough to sample heavily, the
summed hazard over its own raw samples equals its query count within Monte Carlo
error -- the identity G = S that QG1's sampler factor rests on, tested end to end on
the sampled token ids.
"""
import math
import sys
from pathlib import Path

import pytest
import torch

from test_e7_fields import e7_tok, tiny_model
from uie.model import generate

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "research" / "qgap"))
import qg_common as Q  # noqa: E402


class Flat(torch.nn.Module):
    """Every position: logit z on `<QRY>`, 0 elsewhere."""

    def __init__(self, vocab: int, qry: int, z: float = 1.5):
        super().__init__()
        self.vocab, self.qry, self.z = vocab, qry, z

    def forward(self, ids):
        out = torch.zeros(ids.shape[0], ids.shape[1], self.vocab)
        out[..., self.qry] = self.z
        return out

    def p(self) -> float:
        return math.exp(self.z) / (self.vocab - 1 + math.exp(self.z))


def _one(tok, model, url):
    seq = Q.encode_free(tok, url, "benign")
    return Q.hazards(model, tok, [seq], max_len=256)[0]


# host positions: the <LBL> marker and the label, then each <SUB> and its text;
# path positions: the <PTH> marker and the path body
CASES = [
    ("shop.com", 1 + 4, 0, 0, 0),                       # bare host
    ("shop.com/?a=1", 1 + 4, 0, 1, 0),                  # a query on the host route
    ("shop.com/p?a=1", 1 + 4, 1 + 1, 0, 1),             # a query on the path route
    ("www.shop.com/p/q", 1 + 4 + 1 + 3, 1 + 3, 0, 0),   # subdomain and path, no query
    ("shop.com#frag", 1 + 4, 0, 0, 0),                  # a fragment stops the decision
]


@pytest.mark.parametrize("url,host,path,y_host,y_path", CASES)
def test_vq3_hazards_and_routes_are_exact_on_known_probabilities(url, host, path,
                                                                  y_host, y_path):
    tok = e7_tok()
    model = Flat(tok.vocab_size, tok.marker_id("QRY"))
    r = _one(tok, model, url)
    # the model's softmax is float32: agreement to ~1e-7 relative, while a wrong
    # position count would be off by at least a fifth
    assert r["h_host"] == pytest.approx(host * model.p(), rel=1e-6)
    assert r["h_path"] == pytest.approx(path * model.p(), rel=1e-6, abs=1e-12)
    assert (r["y_host"], r["y_path"]) == (y_host, y_path)
    assert r["has_path"] == int(path > 0) and r["stopped"] == 1


def test_vq3_a_row_cut_at_max_len_stops_at_its_last_sampled_position():
    tok = e7_tok()
    model = Flat(tok.vocab_size, tok.marker_id("QRY"))
    seq = Q.encode_free(tok, "shop.com/pp", "benign")[:-1]          # no end token
    r = Q.hazards(model, tok, [seq], max_len=256)[0]
    assert r["stopped"] == 0 and r["y_host"] == r["y_path"] == 0
    assert r["h_host"] == pytest.approx(5 * model.p())               # <LBL> s h o p
    assert r["h_path"] == pytest.approx(2 * model.p())               # <PTH> p (not the last p)


def test_vq3_what_cannot_be_read_is_none():
    tok = e7_tok()
    model = Flat(tok.vocab_size, tok.marker_id("QRY"))
    long = Q.encode_free(tok, "shop.com/" + "p" * 300, "benign")
    no_label = [tok.bos_id, tok.eos_id]
    assert Q.hazards(model, tok, [None, long, no_label], max_len=256) == [None, None, None]


def test_vq3_forwarding_in_length_order_keeps_each_row_its_own():
    tok = e7_tok()
    model = tiny_model(tok)
    urls = ["shop.com/p?a=1", "shop.com", "www.shop.com/p/q", "shop.com/?a=1"]
    seqs = [Q.encode_free(tok, u, "benign") for u in urls]
    together = Q.hazards(model, tok, seqs, max_len=64, batch=3)
    alone = [Q.hazards(model, tok, [s], max_len=64)[0] for s in seqs]
    for a, b in zip(together, alone):
        assert a.keys() == b.keys()
        for key in a:
            assert a[key] == pytest.approx(b[key], rel=1e-6, abs=1e-9)


def test_vq2_the_summed_hazard_of_a_models_own_samples_equals_its_query_count():
    """Sampled at temperature 1.0 with nothing else, E[y] = E[h] (optional
    stopping). A wrong position, route or stop rule breaks this."""
    tok = e7_tok()
    model = tiny_model(tok, seed=3)
    ids = []
    for s in range(4):
        generate(model, tok, label="benign", n=500, suffix="com", seed=100 + s,
                 decoder="temperature", decoder_kwargs={"temperature": 1.0},
                 ids_out=ids)
    recs = [r for r in Q.hazards(model, tok, ids, max_len=model.cfg.max_len) if r]
    assert len(recs) > 1000
    d = [(r["y_host"] + r["y_path"]) - (r["h_host"] + r["h_path"]) for r in recs]
    mean = sum(d) / len(d)
    se = math.sqrt(sum((x - mean) ** 2 for x in d) / (len(d) - 1) / len(d))
    events = sum(r["y_host"] + r["y_path"] for r in recs)
    assert events > 100                                   # the test has power
    assert abs(mean) < 4 * se, (mean, se, events)


def _cat(name):
    return Q.CATS.index(name)


def test_qg2d_profile_counts_categories_and_snapshots_exactly():
    """PREREG_QG2D section 2 on a model with known probabilities: <QRY> gets p,
    every other token the same q."""
    tok = e7_tok()
    model = Flat(tok.vocab_size, tok.marker_id("QRY"))
    p = model.p()
    q = (1.0 - p) / (tok.vocab_size - 1)
    seq = Q.encode_free(tok, "shop.com/p?a=1", "benign")
    r = Q.profile(model, tok, [seq], max_len=256)[0]
    # host: <LBL> s h o p (5 positions); path: <PTH> p (2 positions)
    assert r["hp"][_cat("QRY")] == pytest.approx(5 * p, rel=1e-6)
    assert r["pp"][_cat("QRY")] == pytest.approx(2 * p, rel=1e-6)
    assert r["hp"][_cat("PTH")] == pytest.approx(5 * q, rel=1e-6)
    assert r["hp"][_cat("content")] == pytest.approx(5 * q * len(tok._chars), rel=1e-6)
    assert r["hy"][_cat("content")] == 4 and r["hy"][_cat("PTH")] == 1 and sum(r["hy"]) == 5
    assert r["py"][_cat("content")] == 1 and r["py"][_cat("QRY")] == 1 and sum(r["py"]) == 2
    assert r["hey"] == _cat("PTH") and r["pey"] == _cat("QRY")
    assert r["hep"][_cat("QRY")] == pytest.approx(p, rel=1e-6)
    assert r["has_path"] == 1 and r["dev"] < 1e-5              # VP2


def test_qg2d_profile_on_a_bare_host_has_no_path_snapshot():
    tok = e7_tok()
    model = Flat(tok.vocab_size, tok.marker_id("QRY"))
    r = Q.profile(model, tok, [Q.encode_free(tok, "shop.com", "benign")], max_len=256)[0]
    assert r["hy"][_cat("content")] == 4 and r["hy"][_cat("EOS")] == 1
    assert r["hey"] == _cat("EOS") and r["pep"] is None and r["pey"] is None
    assert sum(r["pp"]) == 0 and sum(r["py"]) == 0 and r["has_path"] == 0


def test_qg2d_profile_agrees_with_the_hazard_instrument():
    """The <QRY> column of the profile is QG1's hazard, route by route."""
    tok = e7_tok()
    model = tiny_model(tok)
    urls = ["shop.com/p?a=1", "www.shop.com/p/q", "shop.com/?a=1", "shop.com#frag"]
    seqs = [Q.encode_free(tok, u, "benign") for u in urls]
    for h, pr in zip(Q.hazards(model, tok, seqs, max_len=64),
                     Q.profile(model, tok, seqs, max_len=64)):
        assert pr["hp"][_cat("QRY")] == pytest.approx(h["h_host"], rel=1e-5, abs=1e-12)
        assert pr["pp"][_cat("QRY")] == pytest.approx(h["h_path"], rel=1e-5, abs=1e-12)
        assert pr["hy"][_cat("QRY")] == h["y_host"] and pr["py"][_cat("QRY")] == h["y_path"]


def test_qg3_regions_split_a_full_url_exactly():
    tok = e7_tok()
    seq = Q.encode_free(tok, "shop.com/p?a=1#f", "benign")
    g = Q.regions(seq, tok)
    sizes = {k: (v[1] - v[0]) if v else 0 for k, v in g.items()}
    # host: <LBL> s h o p; path: <PTH> p; query: <QRY> a = 1; fragment: <FRG> f
    assert sizes == {"host": 5, "path": 2, "query": 4, "fragment": 2}
    nxt = {k: [tok.itos[t] for t in seq[v[0] + 1:v[1] + 1]] for k, v in g.items()}
    assert nxt["host"][-1] == "<PTH>" and nxt["path"][-1] == "<QRY>"
    assert nxt["query"][-1] == "<FRG>" and nxt["fragment"][-1] == tok.itos[tok.eos_id]


def test_qg3_ledger_is_exact_and_matches_the_hazard_instrument():
    import numpy as np
    tok = e7_tok()
    flat = Flat(tok.vocab_size, tok.marker_id("QRY"))
    urls = ["shop.com/p?a=1#f", "shop.com", "www.shop.com/p/q", "shop.com/?a=1"]
    seqs = [Q.encode_free(tok, u, "benign") for u in urls]
    P, N, POS, read = Q.ledger(flat, tok, seqs, max_len=256, block=2)
    assert read == 4 and P.shape[0] == 2
    # VR2: every position predicts one next token, and its probabilities sum to 1
    assert np.allclose(N.sum(2), POS) and np.allclose(P.sum(2), POS, atol=1e-5)
    q = tok.marker_id("QRY")
    hz = Q.hazards(flat, tok, seqs, max_len=256)
    host, path = Q.REGIONS.index("host"), Q.REGIONS.index("path")
    assert P[:, host, q].sum() == pytest.approx(sum(h["h_host"] for h in hz), rel=1e-6)
    assert P[:, path, q].sum() == pytest.approx(sum(h["h_path"] for h in hz), rel=1e-6)
    assert N[:, host, q].sum() == 1 and N[:, path, q].sum() == 1      # the two queries


def test_suffix_groups_keep_the_tables_mass():
    groups, share = Q.suffix_groups({"com": 50, "org": 30, "net": 15, "io": 5}, top=2)
    assert groups == ["com", "org"]
    assert share == pytest.approx({"com": 0.5, "org": 0.3, Q.OTHER: 0.2})
    assert Q.group_of("io", groups) == Q.OTHER and Q.group_of("org", groups) == "org"
