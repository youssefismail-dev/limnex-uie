"""The adopted engine and the previous baseline (uie/baseline.py;
research/native/DECISION_2026-10-08.md, DECISION_2026-09-28.md).

Pins two things at once.

`E9_LN` IS what E9 measured as `e9_ln` / raw: the recipe is E9's registered arm,
exactly; the decode is E9's raw decode; free and control generation are E9's calls, byte
for byte; the model is bound by its weights. **E10, E11 and QG1 derive their recipes and
their decode from it**, so its values must never move -- `test_the_frozen_arms_are_still_
built_on_e9_ln` is the guard against that happening by accident.

`BASELINE` IS `E16_B5`, the adopted engine: e16_cap's weights, decoded with
raw + D1 + D1' + B5. Its decode rules must survive the dataclass, and the native helpers
must refuse it rather than answer with the wrong decode.
"""
import dataclasses
import hashlib
import json
import sys
from pathlib import Path

import pytest
import torch

from test_e7_fields import e7_tok, tiny_model
from uie import baseline as B
from uie.controlled import StateSuffixTables
from uie.suffixes import SuffixTable, generate_with_suffixes
from uie.tokenizer import URLTokenizer
from uie.train import TrainConfig

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "research" / "native"))
import e9_common as K  # noqa: E402

CKPT = ROOT / "data" / "e9" / "train" / "e9_ln" / "best.pt"
PROVENANCE = ROOT / "data" / "e9" / "train" / "PROVENANCE.json"
E16_CKPT = ROOT / "data" / "e16" / "train" / "e16_cap" / "final.pt"
TABLE = SuffixTable({"benign": {"com": 3, "org": 1}, "phishing": {"com": 1, "io": 2}})
STATE_TABLES = StateSuffixTables(
    {label: {code: {"com": 2, "net": 1} for code in ("P1_Q1", "P1_Q0", "P0_Q1", "P0_Q0")}
     for label in ("benign", "phishing")})
UNBOUND = dataclasses.replace(B.E9_LN, weights_sha256=None)


# --------------------------------------------------------------------------- #
# E9_LN -- unchanged, and the frozen arms still rest on it
# --------------------------------------------------------------------------- #

def test_the_recipe_is_e9s_registered_e9_ln_arm():
    assert dict(B.E9_LN.recipe) == {**K.RECIPE, **K.ARMS["e9_ln"]}
    assert B.E9_LN.recipe["loss_constant"] == K.LOSS_CONSTANT
    assert B.E9_LN.recipe["loss_norm"] == "fixed" and B.E9_LN.recipe["seed_init"]


def test_the_decode_is_e9s_raw_decode_and_nothing_else():
    tok = e7_tok()
    for label in ("benign", "phishing"):
        assert B.E9_LN.decode_kwargs() == K.decode(tok, label, "raw")
        assert B.E9_LN.decode_kwargs() != K.decode(tok, label, "resting")
    assert B.E9_LN.rules == {}


def test_the_frozen_arms_are_still_built_on_e9_ln():
    """E10, E11 and QG1 take their recipe and decode from this module. Repointing
    what they import would silently redefine published arms, so pin it here."""
    import e10_common as E10
    import e11_common as E11
    for mod in (E10, E11):
        assert dict(mod.BASE) == dict(B.E9_LN.recipe), mod.__name__
        assert mod.decode() == B.E9_LN.decode_kwargs(), mod.__name__
        assert mod.BASE["d_model"] == 384 and mod.BASE["field_dropout"] == 0.5
    assert E10.recipe("e10_base") == dict(B.E9_LN.recipe)
    assert E11.recipe("e11_plan")["d_model"] == 384


def test_free_generation_is_e9s_raw_free_call_byte_for_byte():
    tok = e7_tok()
    model = tiny_model(tok)
    for label in ("benign", "phishing"):
        for seed in (5, 23):
            got = B.generate_free(model, tok, label=label, n=24, table=TABLE, seed=seed,
                                  baseline=UNBOUND)
            want = generate_with_suffixes(model, tok, label=label, n=24, table=TABLE,
                                          seed=seed, **K.decode(tok, label, "raw"))
            assert got == want and len(got) == 24


def test_control_generation_is_e9s_native_control_call_byte_for_byte():
    tok = e7_tok()
    model = tiny_model(tok)
    for label in ("benign", "phishing"):
        for state, fields in K.STATES.items():
            got = B.generate_control(model, tok, label=label, n=12, fields=fields,
                                     state_tables=STATE_TABLES, seed=7, baseline=UNBOUND)
            want = generate_with_suffixes(model, tok, label=label, n=12,
                                          table=STATE_TABLES.for_state(label, fields), seed=7,
                                          fields=dict(fields), **K.decode(tok, label, "raw"))
            assert got == want and len(got) == 12, state


def test_a_partial_state_is_refused():
    tok = e7_tok()
    with pytest.raises(ValueError, match="joint state"):
        B.generate_control(tiny_model(tok), tok, label="benign", n=2, fields={"PTH": True},
                           state_tables=STATE_TABLES, baseline=UNBOUND)


def test_any_other_model_is_refused_by_its_weights():
    tok = e7_tok()
    model = tiny_model(tok)
    with pytest.raises(ValueError, match="e9_ln"):
        B.generate_free(model, tok, label="benign", n=2, table=TABLE)
    with pytest.raises(ValueError, match="e9_ln"):
        B.generate_control(model, tok, label="benign", n=2, state_tables=STATE_TABLES,
                           fields={"PTH": True, "QRY": False})


def test_a_tokenizer_without_presence_controls_is_refused():
    tok = URLTokenizer(include_frame=True).fit(["a.com/x", "b.org/?q=1"])
    with pytest.raises(ValueError, match="presence controls"):
        B.generate_free(tiny_model(tok), tok, label="benign", n=2, table=TABLE,
                        baseline=UNBOUND)


# --------------------------------------------------------------------------- #
# E16_B5 -- the adopted engine
# --------------------------------------------------------------------------- #

def test_the_adopted_engine_is_e16_b5():
    assert B.BASELINE is B.E16_B5
    assert B.BASELINE.name == "e16_b5"
    assert B.BASELINE.file_sha256 == (
        "98b10b4663ab85ea4af4368b6a09ded1bd5e809d077aaf4f8f1fea013037d65f")
    assert B.BASELINE.weights_sha256 == (
        "14a27fc74630cf885e4702f2e34571237c9cd2cb5bc56dd3245d3c4530b301d4")


def test_the_adopted_decode_is_raw_plus_d1_d1p_b5():
    """The rules must survive the dataclass. Dropping an unrecognised key would
    quietly turn the adopted engine back into the raw decode."""
    kw = B.BASELINE.decode_kwargs()
    assert kw == {"decoder": "temperature", "decoder_kwargs": {"temperature": 1.0},
                  "slash_rule": True, "slash_paths": True, "query_rule": True}
    assert B.BASELINE.rules == {"slash_rule": True, "slash_paths": True, "query_rule": True}
    assert B.BASELINE.decode_kwargs() != B.E9_LN.decode_kwargs()


def test_the_adopted_recipe_is_e16s_registered_width_512_recipe():
    r = dict(B.BASELINE.recipe)
    assert r["d_model"] == 512 and r["n_layers"] == 8 and r["n_heads"] == 8
    assert r["loss_norm"] == "fixed" and r["loss_constant"] == 14397.885481960007
    assert r["presence_plan"] and r["slash_plan"] and r["root_slash"] and r["e12_start"]
    assert r["max_len"] == 384 and r["accum_steps"] == 2 and r["field_dropout"] == 0.0
    assert r["steps"] == 20_000 and r["batch_size"] == 256 and r["lr"] == 3e-4


def test_the_adopted_recipe_matches_e16s_own_module():
    for p in ("research/e16", "research/e15", "research/e13"):
        sys.path.insert(0, str(ROOT / p))
    try:
        import e16_common as E16
    except Exception as exc:                                   # pragma: no cover
        pytest.skip(f"e16_common not importable here: {exc}")
    assert dict(B.BASELINE.recipe) == E16.recipe_cap(Path(E16.STREAM_FILE))


def test_the_native_helpers_refuse_the_adopted_engine():
    """E16_B5 writes to a supplied presence-slash plan (research/e17). These
    helpers are E9's native calls and would answer with the wrong decode."""
    tok = e7_tok()
    model = tiny_model(tok)
    unbound = dataclasses.replace(B.E16_B5, weights_sha256=None)
    with pytest.raises(ValueError, match="decode rules"):
        B.generate_free(model, tok, label="benign", n=2, table=TABLE, baseline=unbound)
    with pytest.raises(ValueError, match="decode rules"):
        B.generate_control(model, tok, label="benign", n=2, state_tables=STATE_TABLES,
                           fields={"PTH": True, "QRY": False}, baseline=unbound)


# --------------------------------------------------------------------------- #
# both
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("name", ["E9_LN", "E16_B5"])
def test_the_settings_cannot_be_changed_by_accident(name):
    b = getattr(B, name)
    with pytest.raises(TypeError):
        b.recipe["lr"] = 1.0
    with pytest.raises(TypeError):
        b.decode["decoder_kwargs"]["temperature"] = 2.0
    with pytest.raises(TypeError):
        b.decode["decoder"] = "greedy"
    with pytest.raises(dataclasses.FrozenInstanceError):
        b.name = "e7"
    kwargs = b.decode_kwargs()
    kwargs["decoder_kwargs"]["temperature"] = 2.0
    assert b.decode_kwargs()["decoder_kwargs"]["temperature"] == 1.0


def test_no_library_default_moved():
    cfg = TrainConfig()
    assert cfg.loss_norm == "batch" and cfg.loss_constant is None


@pytest.mark.skipif(not CKPT.is_file(), reason="the e9_ln checkpoint is local data")
def test_e9_lns_bound_model_is_e9s_e9_ln():
    from uie.train import load_checkpoint
    assert hashlib.sha256(CKPT.read_bytes()).hexdigest() == B.E9_LN.file_sha256
    prov = json.loads(PROVENANCE.read_text(encoding="utf-8"))
    assert prov["arms"]["e9_ln"]["best_sha256"] == B.E9_LN.file_sha256
    model, tok, _ = load_checkpoint(CKPT, device=torch.device("cpu"))
    B.E9_LN.check_model(model)                         # must not raise
    assert tok.include_fields and not getattr(tok, "presence_plan", False)


@pytest.mark.skipif(not E16_CKPT.is_file(), reason="the e16_cap checkpoint is local data")
def test_the_adopted_engines_bound_model_is_e16_cap():
    from uie.train import load_checkpoint
    assert hashlib.sha256(E16_CKPT.read_bytes()).hexdigest() == B.E16_B5.file_sha256
    model, tok, _ = load_checkpoint(E16_CKPT, device=torch.device("cpu"))
    B.E16_B5.check_model(model)                        # must not raise
    assert model.cfg.d_model == 512
    assert tok.presence_plan and tok.slash_plan and tok.root_slash
