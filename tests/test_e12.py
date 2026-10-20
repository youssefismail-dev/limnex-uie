"""E12's library changes (research/native/PREREG_E12.md D2-D5): the slash plan in
the tokenizer and in generation, the frame at every position in the model, the
E12 start, and the checkpoint layout. Flags off leave every earlier path as it was
(the E7-E11 suites run unchanged).
"""
import torch
import pytest

from test_e7_fields import FIT
from uie.free import weights_sha256
from uie.kaggle_train import (POS_ROWS_SEED, S_ROWS_SEED, new_model_e12,
                              new_model_shared)
from uie.model import ModelConfig, URLTransformer, generate
from uie.tokenizer import URLTokenizer
from uie.train import TrainConfig, load_checkpoint, save_checkpoint


def tok(slash_plan=False):
    return URLTokenizer(include_frame=True, include_fields=True, root_slash=True,
                        presence_plan=True, slash_plan=slash_plan).fit(FIT)


def cfg_for(t, **kw):
    base = dict(vocab_size=t.vocab_size, max_len=64, d_model=16, n_layers=1, n_heads=2, d_ff=32)
    return ModelConfig(**{**base, **kw})


# ---- the slash plan: tokenizer ---------------------------------------------------

def test_the_slash_plan_adds_two_tokens_after_the_root_slash():
    a, b = tok(), tok(slash_plan=True)
    assert b.vocab_size == a.vocab_size + 2
    i = b.itos.index("<RS>")
    assert b.itos[i + 1:i + 3] == ["<S:0>", "<S:1>"]
    assert [t for t in b.itos if t not in ("<S:0>", "<S:1>")] == a.itos


@pytest.mark.parametrize("url,s", [("https://example.com/", 1), ("https://example.com", 0),
                                   ("https://example.com?q=1", 0), ("https://example.com/?q=1", 1),
                                   ("https://example.com/a/b?x=2", 1), ("https://example.com#f", 0)])
def test_the_slash_plan_is_true_when_a_slash_follows_the_host(url, s):
    t = tok(slash_plan=True)
    ids = t.encode(url, "benign")
    names = [t.itos[i] for i in ids]
    k = names.index("<LBL>")
    assert names[k - 3:k - 1] in (["<P:0>", "<Q:0>"], ["<P:0>", "<Q:1>"], ["<P:1>", "<Q:0>"],
                                  ["<P:1>", "<Q:1>"])
    assert names[k - 1] == f"<S:{s}>"
    plain = tok()
    assert t.decode(ids) == plain.decode(plain.encode(url, "benign"))   # the same URL back


def test_flags_off_encodings_are_unchanged():
    a, b = tok(), tok(slash_plan=True)
    for url in ("https://example.com/", "https://example.com?x=1", "https://a.b.example.co.uk/p?q#f"):
        ea = [a.itos[i] for i in a.encode(url, "phishing")]
        eb = [b.itos[i] for i in b.encode(url, "phishing") if b.itos[i] not in ("<S:0>", "<S:1>")]
        assert ea == eb


def test_a_supplied_slash_plan_is_complete_and_consistent():
    t = tok(slash_plan=True)
    assert t.plan_fields() == ("PTH", "QRY", "SLASH")
    assert len(t.plan_slots()) == 3
    assert t.itos[t.plan_ids({"PTH": False, "QRY": True, "SLASH": False})[-1]] == "<S:0>"
    with pytest.raises(ValueError):
        t.plan_ids({"PTH": False, "QRY": True})                      # partial
    with pytest.raises(ValueError):
        t.plan_ids({"PTH": True, "QRY": False, "SLASH": False})      # a path implies the slash
    with pytest.raises(KeyError):
        tok().plan_ids({"PTH": False, "QRY": False, "SLASH": True})  # no slash plan
    with pytest.raises(ValueError):
        URLTokenizer(include_fields=True, presence_plan=True, slash_plan=True)  # no <RS>


def test_the_slash_plan_survives_the_state_round_trip():
    t = tok(slash_plan=True)
    r = URLTokenizer.from_state_dict(t.state_dict())
    assert r.slash_plan and r.itos == t.itos
    assert "slash_plan" not in tok().state_dict()


# ---- the slash plan: generation ---------------------------------------------------

def _after_suffix(t, row, suffix):
    """The tokens right after the injected suffix: where the plan is placed.
    (A random-weight model may emit stray markers before it asks for the suffix.)"""
    names = [t.itos[i] for i in row]
    j = names.index("<SUF>")
    assert names[j + 1:j + 1 + len(suffix)] == [t.itos[t.stoi.get(c, t.stoi["<unk>"])] for c in suffix]
    after = names[j + 1 + len(suffix):]
    assert len(after) >= 4, "the row hit max_len inside the injected block"
    return after


def test_generation_injects_a_three_token_plan_and_the_model_fills_three_slots():
    t = tok(slash_plan=True)
    torch.manual_seed(0)
    m = URLTransformer(cfg_for(t, max_len=256))   # room for a random model's detours
    for plan in ({"PTH": False, "QRY": False, "SLASH": True},
                 {"PTH": False, "QRY": True, "SLASH": False}):
        ids = []
        generate(m, t, label="benign", n=16, suffix="com", seed=1, fields=plan, ids_out=ids)
        reached = [r for r in ids if t.stoi["<SUF>"] in r]   # a row can end before asking
        assert reached
        for row in reached:
            after = _after_suffix(t, row, "com")
            assert after[:4] == [t.itos[i] for i in t.plan_ids(plan)] + ["<LBL>"]
    ids = []
    generate(m, t, label="benign", n=16, suffix="com", seed=1, ids_out=ids)  # the model's own plan
    reached = [r for r in ids if t.stoi["<SUF>"] in r]
    assert reached
    for row in reached:
        after = _after_suffix(t, row, "com")
        assert [x[:3] for x in after[:3]] == ["<P:", "<Q:", "<S:"] and after[3] == "<LBL>"


# ---- the frame at every position ------------------------------------------------

def test_the_frame_at_every_position_adds_no_parameters_and_leaves_earlier_positions():
    t = tok()
    torch.manual_seed(0)
    off = URLTransformer(cfg_for(t)).eval()
    on = URLTransformer(cfg_for(t, frame_every=True)).eval()
    on.load_state_dict(off.state_dict())
    assert sum(p.numel() for p in on.parameters()) == sum(p.numel() for p in off.parameters())
    x = torch.tensor([t.encode("https://example.com/a?b=1", "benign")])
    a, b = off(x), on(x)
    assert torch.equal(a[:, :2], b[:, :2])          # before the frame token: untouched
    assert not torch.allclose(a[:, 2:], b[:, 2:])   # from it onward: the frame is read


def test_flags_off_the_forward_pass_is_bit_identical():
    t = tok()
    torch.manual_seed(0)
    m = URLTransformer(cfg_for(t)).eval()
    x = torch.tensor([t.encode("https://example.com/a?b=1", "benign")])
    manual = m.drop(m.tok_emb(x) + m.pos_emb(torch.arange(x.shape[1])))
    for block in m.blocks:
        manual = block(manual)
    assert torch.equal(m(x), m.head(m.ln_f(manual)))


# ---- the E12 start ----------------------------------------------------------------

def test_the_e12_start_is_e11s_start_extended_by_seeded_rows():
    seed = TrainConfig().seed
    t11, t12 = tok(), tok(slash_plan=True)
    e11, _ = new_model_shared(cfg_for(t11, max_len=64), t11, seed)
    # an arm with neither addition starts exactly at E11's start
    same, prints = new_model_e12(cfg_for(t11, max_len=256), t11, seed)
    e11_256, _ = new_model_shared(cfg_for(t11, max_len=256), t11, seed)
    assert weights_sha256(same) == weights_sha256(e11_256) == prints["e11_start_sha256"]
    # the slash plan and 384 positions: every old row copied by token and position
    m, p = new_model_e12(cfg_for(t12, max_len=384, frame_every=True), t12, seed)
    old = e11_256.state_dict()
    new = m.state_dict()
    for tokname in t11.itos:
        assert torch.equal(new["tok_emb.weight"][t12.stoi[tokname]],
                           old["tok_emb.weight"][t11.stoi[tokname]])
    assert torch.equal(new["pos_emb.weight"][:256], old["pos_emb.weight"])
    s_gen = torch.Generator().manual_seed(S_ROWS_SEED)
    for s in ("<S:0>", "<S:1>"):
        want = torch.randn(old["tok_emb.weight"].shape[1:], generator=s_gen) * 0.02
        assert torch.equal(new["tok_emb.weight"][t12.stoi[s]], want)
    p_gen = torch.Generator().manual_seed(POS_ROWS_SEED)
    assert torch.equal(new["pos_emb.weight"][256:],
                       torch.randn((128,) + tuple(old["pos_emb.weight"].shape[1:]), generator=p_gen) * 0.02)
    assert m.head.weight is m.tok_emb.weight                      # still tied
    assert p["e11_start_sha256"] == prints["e11_start_sha256"]
    # and it is deterministic
    m2, _ = new_model_e12(cfg_for(t12, max_len=384, frame_every=True), t12, seed)
    assert weights_sha256(m2) == weights_sha256(m)


# ---- the research side: recipes, tables, calls, jobs ------------------------------

import collections  # noqa: E402
import sys  # noqa: E402
from pathlib import Path  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "research" / "native"))
import e11_common as K11  # noqa: E402
import e12_common as K  # noqa: E402
from uie.suffixes import SuffixTable  # noqa: E402


def test_the_arms_are_the_registered_recipes():
    base = K.recipe("e12_base")
    assert {k: v for k, v in base.items() if k not in ("shared_start", "e12_start")} == \
        {k: v for k, v in K11.recipe("e11_plan").items() if k != "shared_start"}
    assert base["e12_start"] and not base["shared_start"]
    diff = lambda a, b: {k for k in set(a) | set(b) if a.get(k) != b.get(k)}  # noqa: E731
    assert diff(base, K.recipe("e12_no_r1")) == {"slash_plan"}
    assert diff(K.recipe("e12_no_r1"), K.recipe("e12_final")) == {"max_len"}
    assert K.recipe("e12_final")["max_len"] == 384
    assert K.SLASH_ARMS == {"e12_final", "e12_no_r1"} and K.CANDIDATE == "e12_final"
    assert K.decode() == {"decoder": "temperature", "decoder_kwargs": {"temperature": 1.0}}


def _ps_table():
    # suffix -> counts of P0Q0S0, P0Q0S1, P0Q1S0, P0Q1S1, P1Q0S1, P1Q1S1
    return {"benign": {"com": [2, 98, 1, 9, 700, 190], "org": [0, 50, 0, 10, 30, 10]}}


def test_the_supplied_tables_are_the_shipped_table_times_the_datas_six_state_rates():
    t = K.supplied_tables(_ps_table(), {"benign": {"com": 0.75, "org": 0.25}}, "benign")
    assert t["P0Q0S0"] == {"com": 0.75 * 2 / 1000}
    assert t["P0Q1S1"] == pytest.approx({"com": 0.75 * 9 / 1000, "org": 0.25 * 10 / 100})
    assert sum(sum(v.values()) for v in t.values()) == pytest.approx(1.0)


class _States:
    def for_state(self, label, fields):
        return SuffixTable({label: {"com": 3.0, "org": 1.0}})


def test_control_mode_splits_each_suffix_by_its_own_slash_rate():
    t = K.control_tables(_ps_table(), _States(), "benign", "P0Q1")
    assert t == {"P0Q1S0": {"com": 3.0 * 1 / 10}, "P0Q1S1": {"com": 3.0 * 9 / 10, "org": 1.0}}
    assert K.control_tables(_ps_table(), _States(), "benign", "P1Q0") == \
        {"P1Q0S1": {"com": 3.0, "org": 1.0}}


def test_supplied_generation_tells_each_url_its_state_and_writes_it():
    t = tok(slash_plan=True)
    torch.manual_seed(0)
    m = URLTransformer(cfg_for(t, max_len=256))
    plans = []
    real_generate = K.generate_with_suffixes

    def spy(*a, **kw):
        plans.append((kw["fields"], kw["n"], kw["seed"]))
        return real_generate(*a, **kw)
    K.generate_with_suffixes = spy
    try:
        shipped = SuffixTable({"benign": {"com": 0.75, "org": 0.25}})
        urls = K.free_urls(m, t, "e12_final", "benign", 0, shipped, presence_slash=_ps_table(), n=40)
        again = K.free_urls(m, t, "e12_final", "benign", 0, shipped, presence_slash=_ps_table(), n=40)
    finally:
        K.generate_with_suffixes = real_generate
    assert len(urls) == 40 and urls == again                       # seeded end to end
    assert sum(n for _, n, _ in plans[:len(plans) // 2]) == 40
    assert {tuple(sorted(f.items())) for f, _, _ in plans} <= {
        tuple(sorted(K.fields_of(s).items())) for s in K.PS_STATES}


def test_the_identity_check_finds_every_url_of_a_supplied_file():
    import e12_verify as V
    t = tok(slash_plan=True)
    torch.manual_seed(0)
    m = URLTransformer(cfg_for(t, max_len=256))
    shipped = SuffixTable({"benign": {"com": 0.75, "org": 0.25}})
    urls = K.free_urls(m, t, "e12_final", "benign", 0, shipped, presence_slash=_ps_table(), n=30)
    same, compared = V.drawn_identity(m, t, K.supplied_tables(_ps_table(), shipped.counts, "benign"),
                                      urls, "benign", K.seed_free(0), 30)
    assert (same, compared) == (30, 30)
    urls = K.control_urls(m, t, "e12_final", "benign", "P0Q1", 0, _States(),
                          presence_slash=_ps_table(), n=20)
    same, compared = V.drawn_identity(m, t, K.control_tables(_ps_table(), _States(), "benign", "P0Q1"),
                                      urls, "benign", K.seed_ctrl("benign", "P0Q1", 0), 20)
    assert (same, compared) == (20, 20)


def test_state_positions_recover_every_urls_supplied_state_from_the_seeds():
    import e12_score as S
    real = K.generate_with_suffixes

    def stamped(model, tokenizer, *, label, n, table, seed, fields, **kw):
        tag = "".join(f"{k[0]}{int(v)}" for k, v in fields.items())    # e.g. P0Q1S0
        return [f"{tag}#{seed}#{i}" for i in range(n)]
    K.generate_with_suffixes = stamped
    try:
        shipped = SuffixTable({"benign": {"com": 0.75, "org": 0.25}})
        free = K.free_urls(None, None, "e12_final", "benign", 3, shipped, presence_slash=_ps_table(), n=400)
        ctrl = K.control_urls(None, None, "e12_final", "benign", "P0Q1", 2, _States(),
                              presence_slash=_ps_table(), n=300)
    finally:
        K.generate_with_suffixes = real
    # each stamp is the state's own name (P0Q1S0 ...), so position by position they must agree
    at = S.state_positions(K.supplied_tables(_ps_table(), shipped.counts, "benign"), 400, K.seed_free(3))
    assert [u.split("#")[0] for u in free] == at
    at = S.state_positions(K.control_tables(_ps_table(), _States(), "benign", "P0Q1"), 300,
                           K.seed_ctrl("benign", "P0Q1", 2))
    assert [u.split("#")[0] for u in ctrl] == at
    assert set(at) == {"P0Q1S0", "P0Q1S1"}


def test_the_jobs_are_the_registered_150_files():
    jobs = K.jobs()
    assert len(jobs) == 150 and len({j["file"] for j in jobs}) == 150
    assert collections.Counter(j["arm"] for j in jobs) == {a: 50 for a in K.ARMS}
    assert {j["plan"] for j in jobs if j["arm"] == "e12_base" and j["mode"] == "control"} == {"requested"}


# ---- the checkpoint layout ----------------------------------------------------------

def test_a_checkpoint_writes_the_frame_fields_only_when_on(tmp_path):
    t = tok()
    for every in (False, True):
        m = URLTransformer(cfg_for(t, frame_every=every))
        save_checkpoint(m, t, TrainConfig(), tmp_path / f"{every}.pt")
        blob = torch.load(tmp_path / f"{every}.pt", weights_only=False)
        assert ("frame_every" in blob["model_config"]) is every
        m2, _, cfg = load_checkpoint(tmp_path / f"{every}.pt")
        assert cfg.frame_every is every
