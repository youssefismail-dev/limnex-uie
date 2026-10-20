"""E11's code (research/native/PREREG_E11.md; research/native/e11_common.py;
uie/train.py and uie/kaggle_train.py's E11 flags).

Pins the registered design: the four recipes, each one change on its reference;
the shared start (every existing weight copied, `<RS>`'s row from its own
generator, the start without it recoverable and hashable); the clip schedule and
its record; joint dropout (both controls or neither); the supplied plan's tables
and calls; and the 200 registered files. Flags off leave every earlier path as it
was (the E7-E10 suites run unchanged).
"""
import json
import random
import sys
from pathlib import Path

import pytest
import torch

from test_e7_fields import FIT
from uie import baseline as B
from uie.free import weights_sha256
from uie.kaggle_train import RS_ROW_SEED, new_model, new_model_shared
from uie.model import ModelConfig
from uie.suffixes import SuffixTable, generate_with_suffixes
from uie.tokenizer import URLTokenizer
from uie.train import (FieldDropout, TrainConfig, URLDataset, _diagnostics,
                       encode_corpus, train)

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "research" / "native"))
import e11_common as K  # noqa: E402
import e8_common as E8  # noqa: E402


def tok(root_slash=False, plan=False):
    return URLTokenizer(include_frame=True, include_fields=True, root_slash=root_slash,
                        presence_plan=plan).fit(FIT)


def cfg_for(t):
    return ModelConfig(vocab_size=t.vocab_size, max_len=64, d_model=16, n_layers=1,
                       n_heads=2, d_ff=32)


# ---- the recipes ----------------------------------------------------------------

def test_each_arm_adds_one_registered_change_to_its_reference():
    # E11 was built on e9_ln and is frozen there; it must NOT follow BASELINE, which
    # has since moved to the adopted engine.
    base = dict(B.E9_LN.recipe)
    diff = lambda a, b: {k for k in set(a) | set(b) if a.get(k) != b.get(k)}  # noqa: E731
    assert diff(base, K.recipe("e11_rs")) == {"root_slash", "shared_start"}
    assert diff(K.recipe("e11_rs"), K.recipe("e11_ref")) == {"grad_clip_late"}
    assert diff(K.recipe("e11_ref"), K.recipe("e11_joint")) == {"field_dropout_joint"}
    assert diff(K.recipe("e11_ref"), K.recipe("e11_plan")) == {"presence_plan", "field_dropout"}
    assert K.recipe("e11_ref")["grad_clip_late"] == 5.0
    assert K.recipe("e11_plan")["field_dropout"] == 0.0
    assert K.REFERENCE == {"e11_rs": "e10_base", "e11_ref": "e11_rs",
                           "e11_joint": "e11_ref", "e11_plan": "e11_ref"}
    assert K.SUPPLIED == {"e11_plan"}
    assert K.decode() == {"decoder": "temperature", "decoder_kwargs": {"temperature": 1.0}}


# ---- the shared start (D2) ----------------------------------------------------------

def test_the_slash_vocabulary_is_the_old_one_with_one_token_inserted():
    old, new = tok(), tok(root_slash=True)
    assert new.vocab_size == old.vocab_size + 1
    assert [t for t in new.itos if t != "<RS>"] == old.itos
    assert tok(root_slash=True, plan=True).itos == new.itos        # the plan adds none


def test_the_shared_start_copies_every_weight_and_draws_only_the_new_row():
    old_t, new_t = tok(), tok(root_slash=True)
    model, base_sha = new_model_shared(cfg_for(new_t), new_t, seed=20260803)
    base = new_model(cfg_for(old_t), 20260803)
    assert base_sha == weights_sha256(base)
    rs = new_t.rs_id
    new_sd, old_sd = model.state_dict(), base.state_dict()
    for name, t in new_sd.items():
        b = old_sd[name]
        if t.shape == b.shape:
            assert torch.equal(t, b), name
        else:
            assert torch.equal(torch.cat([t[:rs], t[rs + 1:]]), b), name
    assert model.head.weight is model.tok_emb.weight               # still tied
    row = torch.randn(cfg_for(new_t).d_model, generator=torch.Generator().manual_seed(
        RS_ROW_SEED)) * 0.02
    assert torch.equal(model.tok_emb.weight[rs].detach(), row)
    # removing the row recovers the start without <RS>, hash for hash
    stripped = new_model(cfg_for(old_t), 0)
    sd = {k: (torch.cat([v[:rs], v[rs + 1:]]) if v.shape != old_sd[k].shape else v)
          for k, v in model.state_dict().items()}
    stripped.load_state_dict(sd)
    assert weights_sha256(stripped) == base_sha
    again, _ = new_model_shared(cfg_for(new_t), new_t, seed=20260803)
    assert weights_sha256(again) == weights_sha256(model)          # one start for all arms


def test_the_shared_start_needs_a_slash_vocabulary():
    t = tok()
    with pytest.raises(ValueError):
        new_model_shared(cfg_for(t), t, seed=1)


# ---- the clip schedule (D3) -------------------------------------------------------------

def test_the_diagnostics_count_each_step_against_its_own_cap():
    recs = [(s, 2.0, 100.0, 10, 1.0 if s < 2 else 5.0) for s in range(4)]
    d = _diagnostics(recs)
    assert d["all 0-30"]["clipped_share"] == 0.5 and "over_early_cap_share" not in d["all 0-30"]
    d = _diagnostics(recs, early_cap=1.0)
    assert d["all 0-30"]["over_early_cap_share"] == 1.0
    old = [(s, 2.0, 100.0, 10, 1.0) for s in range(4)]              # a pre-E11 run
    assert _diagnostics(old)["all 0-30"] == {"steps": 4, "mean_norm": 2.0,
                                             "clipped_share": 1.0, "nonfinite": 0}


def _tiny_run(t, **cfg):
    urls = [u for u in FIT if u][:24]
    enc, _ = encode_corpus(urls * 2, ["benign"] * len(urls) * 2, t, 64,
                           corpora=["name"] * len(urls) * 2)
    ds = URLDataset(enc)
    return train(new_model(cfg_for(t), 3), ds, ds, t,
                 TrainConfig(steps=4, batch_size=4, warmup=1, eval_every=10, eval_batches=1,
                             loss_norm="fixed", loss_constant=100.0, **cfg))


def test_the_cap_switches_at_the_registered_step():
    t = tok(root_slash=True)
    res = _tiny_run(t, grad_clip=1e-9, grad_clip_late=1e9, grad_clip_switch=2,
                    field_dropout=0.5)
    shares = [v for k, v in res["grad_diagnostics"].items() if k.startswith("all ")]
    steps = sum(v["steps"] for v in shares)
    clipped = sum(v["clipped_share"] * v["steps"] for v in shares)
    over = sum(v["over_early_cap_share"] * v["steps"] for v in shares)
    assert steps == 4 and round(clipped) == 2 and round(over) == 4


# ---- joint dropout (D4) -------------------------------------------------------------------

def test_joint_dropout_hides_both_controls_or_neither():
    t = tok()
    ids = t.encode("http://example.com/a?b=1", "benign", frame="domain")
    row = (ids, [1.0] * (len(ids) - 1))
    pos = t.field_control_positions()
    hidden = {t.field_control_id(f, None) for f in pos}
    drop = FieldDropout(t, 0.5, seed=5, joint=True)
    seen = []
    for _ in range(2000):
        out = drop(row)[0]
        states = [out[p] in hidden for p in pos.values()]
        assert len(set(states)) == 1                                # both or neither
        seen.append(states[0])
    assert 0.45 < sum(seen) / len(seen) < 0.55
    assert FieldDropout(t, 0.0, seed=1, joint=True)(row) is row
    assert all(FieldDropout(t, 1.0, seed=1, joint=True)(row)[0][p] in hidden
               for p in pos.values())


def test_joint_dropout_trains_and_needs_a_dropout_rate():
    t = tok(root_slash=True)
    res = _tiny_run(t, field_dropout=0.5, field_dropout_joint=True, grad_clip_late=5.0)
    assert res["loss_norm"] == "fixed"
    with pytest.raises(ValueError):
        _tiny_run(t, field_dropout=0.0, field_dropout_joint=True)


# ---- the supplied plan (D6) ---------------------------------------------------------------

PRESENCE = {"benign": {"com": [5, 1, 3, 1], "org": [0, 0, 4, 0]},
            "phishing": {"com": [2, 0, 2, 0]}}
SHIPPED = SuffixTable({"benign": {"com": 3, "org": 1}, "phishing": {"com": 1}})


def test_the_supplied_tables_are_the_shipped_table_times_the_datas_state_rates():
    w = K.supplied_tables(PRESENCE, SHIPPED.counts, "benign")
    assert w["P0Q0"] == {"com": 3 * 5 / 10} and w["P1Q0"] == {"com": 3 * 3 / 10, "org": 1.0}
    for s, c in SHIPPED.counts["benign"].items():
        assert sum(w[k].get(s, 0.0) for k in K.PRESENCE) == pytest.approx(c)
    with pytest.raises(KeyError):
        K.supplied_tables({"benign": {"com": [1, 0, 0, 0]}}, SHIPPED.counts, "benign")


def test_supplied_free_mode_is_control_modes_call_per_drawn_state():
    t = tok(root_slash=True, plan=True)
    model = new_model(cfg_for(t), 7).eval()
    trace = []
    got = K.free_urls(model, t, "e11_plan", "benign", 0, SHIPPED, presence=PRESENCE,
                      n=20, plan_trace=trace)
    assert len(got) == 20 and trace == [0.0] * 20                   # every plan supplied
    w = K.supplied_tables(PRESENCE, SHIPPED.counts, "benign")
    rng = random.Random(E8.seed_free(0))
    counts = __import__("collections").Counter(
        rng.choices(K.PRESENCE, weights=[sum(w[k].values()) for k in K.PRESENCE], k=20))
    want = []
    for i, k in enumerate(K.PRESENCE):
        if counts[k]:
            want += generate_with_suffixes(model, t, label="benign", n=counts[k],
                                           table=SuffixTable({"benign": w[k]}),
                                           seed=E8.seed_free(0) + K.STATE_SEED * (i + 1),
                                           fields=dict(E8.STATES[k]), **K.decode())
    rng.shuffle(want)
    assert got == want
    with pytest.raises(ValueError):
        K.free_urls(model, t, "e11_plan", "benign", 0, SHIPPED, n=4)


def test_the_supplied_plan_identity_check_finds_every_url_in_its_file():
    import e11_verify as V
    t = tok(root_slash=True, plan=True)
    model = new_model(cfg_for(t), 11).eval()
    urls = K.free_urls_supplied(model, t, "benign", 2, SHIPPED, PRESENCE, n=24)
    same, compared = V.supplied_identity(model, t, SHIPPED, PRESENCE, urls, "benign", 2, 24)
    assert compared == 24 and same == 24
    shuffled = urls[1:] + urls[:1]                                    # a wrong file fails
    same, compared = V.supplied_identity(model, t, SHIPPED, PRESENCE, shuffled, "benign", 2, 24)
    assert same < compared


def test_hidden_arms_generate_free_mode_as_e9_ln_does():
    t = tok(root_slash=True)
    model = new_model(cfg_for(t), 7).eval()
    got = K.free_urls(model, t, "e11_ref", "benign", 1, SHIPPED, n=10)
    want = generate_with_suffixes(model, t, label="benign", n=10, table=SHIPPED,
                                  seed=E8.seed_free(1), **K.decode())
    assert got == want


# ---- the readout's registered rules (section 2) -------------------------------------------------

def test_the_tolerance_has_the_registered_edge_rule():
    import e11_compare as C
    assert C.tolerance(0.73369, 8000) == pytest.approx(2 * (0.73369 * 0.26631 / 8000) ** 0.5)
    assert C.tolerance(1.0, 4710) == pytest.approx(2 / 4710)          # p = 1: one event
    assert C.tolerance(0.99576, 133) == pytest.approx(2 / 133)         # floored below 1/n
    assert C.tolerance(0.97944, 3000) == pytest.approx(2 * (0.97944 * 0.02056 / 3000) ** 0.5)


def test_k11a_and_k11b_read_the_registered_rows():
    import e11_compare as C
    targets = {lab: {"path_rate_free": 0.7, "query_rate_free": 0.06,
                     "slash_P0Q0": {"free": 0.98, "free_n_per_generation": 2000, "control": 0.98},
                     "slash_P0Q1": {"free": 1.0, "free_n_per_generation": 130, "control": 1.0}}
               for lab in K.LABELS}
    r = lambda x: {"rate": x}  # noqa: E731
    rep = {"emission": {"a": {lab: {"PTH": r(0.705), "QRY": r(0.06)} for lab in K.LABELS}},
           "slash": {"a": {**{f"free_bare_{lab}": r(0.98) for lab in K.LABELS},
                           **{f"free_host_query_{lab}": r(1.0) for lab in K.LABELS},
                           **{f"control_{lab}_P0Q0": r(0.978) for lab in K.LABELS},
                           **{f"control_{lab}_P0Q1": r(0.999) for lab in K.LABELS}}}}
    a = C.k11a(rep, targets, "a")
    assert a["pass"] and a["rows"]["benign PTH"]["diff"] == pytest.approx(0.005)
    rep["emission"]["a"]["phishing"]["QRY"] = r(0.05)                  # 1 point over at 6%
    assert not C.k11a(rep, targets, "a")["pass"]
    b = C.k11b(rep, targets, "a")
    assert b["rows"]["benign control P0Q1"]["pass"] is False           # 0.999 vs 1.0 at n 3,000
    assert b["rows"]["phishing control P0Q1"]["pass"] is True          # n 1,000: T = 0.002
    assert b["rows"]["benign free_bare"]["pass"] is True


def test_k11e_reads_the_clipping_after_step_1000(tmp_path):
    import e11_compare as C
    for arm, share in (("e11_ref", 0.004), ("e11_joint", 0.02)):
        (tmp_path / arm).mkdir()
        (tmp_path / arm / "train_result.json").write_text(json.dumps({"grad_diagnostics": {
            "after_1000 0-30": {"clipped_share": share, "steps": 4000},
            "after_1000 30-60": {"clipped_share": 0.0, "steps": 9000},
            "first_1000 0-30": {"clipped_share": 0.8, "steps": 200}}}))
    assert C.k11e(tmp_path, "e11_ref")["pass"] is True
    assert C.k11e(tmp_path, "e11_joint")["pass"] is False
    assert C.k11e(tmp_path, "e11_rs") is None                           # no repair, no K11e


def test_the_step_20000_notebook_reads_final_pt_and_changes_nothing_else():
    """PREREG_E11 3.3 (amendment 3): the same cells as E11_generate apart from the
    registered substitutions and the introduction."""
    import make_e11_notebooks as N
    base, final = N.generate_cells(), N.generate_final_cells()
    assert len(base) == len(final)
    for b, f in zip(base[1:], final[1:]):
        want = "".join(b["source"])
        for old, new in N.FINAL_SUBS:
            want = want.replace(old, new)
        assert want == "".join(f["source"])
    src = "".join("".join(c["source"]) for c in final[1:])
    assert '"best.pt"' not in src and "best_sha256" not in src
    assert '"checkpoint_file": "final.pt"' in src and "e11_gen_final" in src


# ---- the registered files (D8) --------------------------------------------------------------

def test_the_jobs_are_the_registered_200_files():
    jobs = K.jobs()
    assert len(jobs) == 200 and len({j["file"] for j in jobs}) == 200
    assert sum(j["n"] for j in jobs) == 720_000
    assert sum(j["n"] for j in jobs if j["mode"] == "free") == 320_000
    for j in jobs:
        want = (E8.seed_free(j["g"]) if j["mode"] == "free"
                else E8.seed_ctrl(j["label"], j["state"], j["g"]))
        assert j["seed"] == want
        if j["mode"] == "free":
            assert j["plan"] == ("supplied" if j["arm"] == "e11_plan" else "model")
