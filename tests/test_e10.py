"""E10's code (research/native/PREREG_E10.md; research/native/e10_common.py).

Pins the registered design: `e10_base` is `e9_ln`'s recipe exactly (the baseline at the
time, which this frozen arm stays bound to);
`e10_plan` differs by the presence plan and the dropout it implies, nothing else;
both decode raw; the jobs are the registered 100 files and 360,000 URLs at E8/E9's
seeds; the plan leaves the vocabulary (so the seeded start) unchanged; a plan
model trains with the fixed objective end to end; and generation for both arms is
the registered call.
"""
import sys
from pathlib import Path

import pytest

from test_e7_fields import FIT, e7_tok, tiny_model
from uie import baseline as B
from uie.controlled import StateSuffixTables
from uie.model import ModelConfig, URLTransformer
from uie.suffixes import SuffixTable, generate_with_suffixes
from uie.tokenizer import URLTokenizer
from uie.train import TrainConfig, URLDataset, encode_corpus, train

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "research" / "native"))
import e10_common as K  # noqa: E402
import e8_common as E8  # noqa: E402

TABLE = SuffixTable({"benign": {"com": 3, "org": 1}, "phishing": {"com": 1, "io": 2}})
STATE_TABLES = StateSuffixTables(
    {label: {code: {"com": 2, "net": 1} for code in ("P1_Q1", "P1_Q0", "P0_Q1", "P0_Q0")}
     for label in ("benign", "phishing")})


def test_the_base_arm_is_e9_lns_recipe_exactly():
    # E10 was built on e9_ln, the baseline at the time, and is frozen there. It must
    # NOT follow `uie.baseline.BASELINE`, which has since moved to the adopted engine.
    assert K.recipe("e10_base") == dict(B.E9_LN.recipe)


def test_the_plan_arm_differs_by_the_plan_and_its_dropout_only():
    base, plan = K.recipe("e10_base"), K.recipe("e10_plan")
    changed = {k for k in set(base) | set(plan) if base.get(k) != plan.get(k)}
    assert changed == {"presence_plan", "field_dropout"}
    assert plan["presence_plan"] is True and plan["field_dropout"] == 0.0
    assert plan["loss_norm"] == "fixed" and plan["loss_constant"] == base["loss_constant"]


def test_both_arms_decode_raw():
    assert K.decode() == {"decoder": "temperature", "decoder_kwargs": {"temperature": 1.0}}


def test_the_jobs_are_the_registered_100_files():
    jobs = K.jobs()
    assert len(jobs) == 100 and len({j["file"] for j in jobs}) == 100
    assert sum(j["n"] for j in jobs) == 360_000
    assert sum(j["n"] for j in jobs if j["mode"] == "free") == 160_000
    for j in jobs:
        want = (E8.seed_free(j["g"]) if j["mode"] == "free"
                else E8.seed_ctrl(j["label"], j["state"], j["g"]))
        assert j["seed"] == want


def test_the_plan_leaves_the_vocabulary_and_so_the_start_unchanged():
    plain = e7_tok()
    plan = URLTokenizer(include_frame=True, include_fields=True, presence_plan=True).fit(FIT)
    assert plan.itos == plain.itos and plan.vocab_size == plain.vocab_size


def test_a_plan_model_trains_end_to_end_with_the_fixed_objective():
    tok = URLTokenizer(include_frame=True, include_fields=True, presence_plan=True).fit(FIT)
    urls = [u for u in FIT if u][:24]
    enc, _ = encode_corpus(urls * 3, ["benign"] * len(urls) * 3, tok, 64,
                           corpora=["name"] * len(urls) * 3)
    ds = URLDataset(enc)
    cfg = ModelConfig(vocab_size=tok.vocab_size, max_len=64, d_model=16, n_layers=1,
                      n_heads=2, d_ff=32)
    res = train(URLTransformer(cfg), ds, ds, tok,
                TrainConfig(steps=3, batch_size=4, warmup=1, eval_every=5, eval_batches=1,
                            loss_norm="fixed", loss_constant=100.0))
    assert res["loss_norm"] == "fixed" and res["loss_constant"] == 100.0


def test_free_and_control_generation_are_the_registered_calls():
    tok = e7_tok()
    model = tiny_model(tok)
    got = K.free_urls(model, tok, "benign", 0, TABLE, n=12)
    want = generate_with_suffixes(model, tok, label="benign", n=12, table=TABLE,
                                  seed=E8.seed_free(0), decoder="temperature",
                                  decoder_kwargs={"temperature": 1.0})
    assert got == want and len(got) == 12
    got = K.control_urls(model, tok, "phishing", "P0Q1", 1, STATE_TABLES, n=10)
    want = generate_with_suffixes(model, tok, label="phishing", n=10,
                                  table=STATE_TABLES.for_state("phishing", E8.STATES["P0Q1"]),
                                  seed=E8.seed_ctrl("phishing", "P0Q1", 1),
                                  fields=dict(E8.STATES["P0Q1"]), decoder="temperature",
                                  decoder_kwargs={"temperature": 1.0})
    assert got == want and len(got) == 10


def test_a_plan_model_fills_its_own_plan_in_free_mode_and_takes_one_in_control():
    tok = URLTokenizer(include_frame=True, include_fields=True, presence_plan=True).fit(FIT)
    model = tiny_model(tok)
    trace = []
    urls = K.free_urls(model, tok, "benign", 0, TABLE, n=8, plan_trace=trace)
    assert len(urls) == 8 and len(trace) == 8 and all(0.0 <= t <= 2.0 for t in trace)
    trace = []
    urls = K.control_urls(model, tok, "benign", "P1Q1", 0, STATE_TABLES, n=6, plan_trace=trace)
    assert len(urls) == 6 and trace == [0.0] * 6            # a supplied plan is not sampled
