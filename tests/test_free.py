"""DC1's adoption for FULL FREE: free-mode generation with E7 (uie/free.py;
research/dc1/DC1_RESULTS.md).

Pins that the library IS what DC1 measured in free mode: the adopted settings are
DC1's confirmed free-mode settings, exactly (against the fitted file when it is
present); the call is DC1's free-arm call, byte for byte, and every setting in it
reaches the sampler; the settings are refused for any other model, by its weights,
and for a tokenizer without presence controls; and they cannot be changed by
accident.
"""
import dataclasses
import hashlib
import importlib.util
import json
from pathlib import Path

import pytest
import torch

from test_e7_fields import FIT, e7_tok, tiny_model
from uie.free import ADOPTED, FreeDecoding, generate_free, weights_sha256
from uie.suffixes import SuffixTable, generate_with_suffixes
from uie.tokenizer import URLTokenizer

ROOT = Path(__file__).resolve().parents[1]
SETTINGS = ROOT / "data" / "dc1" / "settings.json"
SETTINGS_SHA256 = "68850b79f83851cbc49dd14fd91cb20e8de404eacb208a04b3eada151efb5f35"
E7 = ROOT / "data" / "best_e7.pt"
E7_SHA256 = "916d03857d7d3db4101d1b106293a83905f4b317676ee7e144b018170cb36065"

TABLE = SuffixTable({"benign": {"com": 3, "org": 1}, "phishing": {"com": 1, "io": 2}})
DELTA = {"benign": 0.22, "phishing": 0.30}


def unbound(d=ADOPTED):
    """The same settings, released from E7's weights so a test model can run them."""
    return FreeDecoding(d.delta, d.temperature, d.stop_bias)


def test_generate_free_is_dc1s_free_arm_call_byte_for_byte():
    """DC1's notebook generated the free arm with generate_with_suffixes, the
    shipped table, temperature 1.0, the frozen delta on PTH's end and the
    confirmed per-field settings; the library must return exactly those URLs."""
    tok = e7_tok()
    model = tiny_model(tok)
    for label in ("benign", "phishing"):
        for seed in (3, 17):
            got = generate_free(model, tok, label=label, n=24, table=TABLE, seed=seed,
                                decoding=unbound())
            want = generate_with_suffixes(
                model, tok, label=label, n=24, table=TABLE, seed=seed,
                decoder="temperature", decoder_kwargs={"temperature": 1.0},
                field_bias={"PTH": {tok.eos_id: -DELTA[label]}},
                field_temperature=dict(ADOPTED.temperature[label]),
                field_stop_bias=dict(ADOPTED.stop_bias[label]))
            assert got == want and len(got) == 24
            assert got == generate_free(model, tok, label=label, n=24, table=TABLE,
                                        seed=seed, decoding=unbound())


def test_every_setting_reaches_the_sampler():
    """With the settings made large, removing any one part -- the frozen delta,
    the temperatures or the stop biases -- changes the URLs."""
    tok = e7_tok()
    model = tiny_model(tok)
    temp = {"benign": {"LBL": 1.8, "PTH": 0.6, "QRY": 1.5}}
    stop = {"benign": {"LBL": -1.5, "PTH": 1.0, "QRY": -1.0}}
    full = FreeDecoding({"benign": 2.0}, temp, stop)
    kw = dict(label="benign", n=48, table=TABLE, seed=5)
    got = generate_free(model, tok, decoding=full, **kw)
    for part in (FreeDecoding({"benign": 0.0}, temp, stop),
                 FreeDecoding({"benign": 2.0}, {}, stop),
                 FreeDecoding({"benign": 2.0}, temp, {})):
        assert generate_free(model, tok, decoding=part, **kw) != got, part


def test_the_adopted_settings_are_refused_for_any_other_model():
    tok = e7_tok()
    with pytest.raises(ValueError, match="fitted to the model"):
        generate_free(tiny_model(tok), tok, label="benign", n=4, table=TABLE)


def test_a_tokenizer_without_presence_controls_is_refused():
    e5_style = URLTokenizer(include_frame=True).fit(FIT)
    with pytest.raises(ValueError, match="presence controls"):
        generate_free(tiny_model(e5_style), e5_style, label="benign", n=4, table=TABLE,
                      decoding=unbound())


def test_an_unknown_class_is_refused():
    tok = e7_tok()
    with pytest.raises(ValueError, match="no free-mode decoding"):
        generate_free(tiny_model(tok), tok, label="spam", n=4, table=TABLE,
                      decoding=unbound())


def test_weights_fingerprint():
    tok = e7_tok()
    a, b = tiny_model(tok, seed=0), tiny_model(tok, seed=0)
    assert weights_sha256(a) == weights_sha256(a) == weights_sha256(b)
    assert weights_sha256(tiny_model(tok, seed=1)) != weights_sha256(a)
    with torch.no_grad():
        next(b.parameters()).view(-1)[0] += 1e-3
    assert weights_sha256(b) != weights_sha256(a)


def test_the_adopted_settings_cannot_be_changed_by_accident():
    tok = e7_tok()
    with pytest.raises(TypeError):
        ADOPTED.temperature["benign"]["LBL"] = 2.0
    with pytest.raises(TypeError):
        ADOPTED.delta["benign"] = 0.0
    with pytest.raises(dataclasses.FrozenInstanceError):
        ADOPTED.delta = {}
    kw = ADOPTED.kwargs(tok, "benign")
    kw["field_temperature"]["LBL"] = 9.0
    kw["decoder_kwargs"]["temperature"] = 9.0
    assert ADOPTED.temperature["benign"]["LBL"] == 1.0129424258915505
    assert ADOPTED.kwargs(tok, "benign")["decoder_kwargs"] == {"temperature": 1.0}


@pytest.mark.skipif(not SETTINGS.is_file(), reason="needs data/dc1/settings.json")
def test_the_adopted_settings_are_dc1s_confirmed_free_settings():
    """Exactly the free-mode fields whose fit passed validation, at their fitted
    values, from the registered settings file; and the same arguments DC1's
    notebook added (research/dc1/dc1_common.generation_kwargs)."""
    assert hashlib.sha256(SETTINGS.read_bytes()).hexdigest() == SETTINGS_SHA256
    s = json.loads(SETTINGS.read_text(encoding="utf-8"))
    assert dict(ADOPTED.delta) == s["frozen_delta"] == DELTA
    spec = importlib.util.spec_from_file_location(
        "dc1_common_for_free", ROOT / "research" / "dc1" / "dc1_common.py")
    common = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(common)
    tok = e7_tok()
    for label in ("benign", "phishing"):
        applied = {f: r for f, r in s["classes"][label]["free"].items() if r["applied"]}
        assert dict(ADOPTED.temperature[label]) == {f: r["T"] for f, r in applied.items()}
        assert dict(ADOPTED.stop_bias[label]) == {f: r["delta"] for f, r in applied.items()}
        lib = ADOPTED.kwargs(tok, label)
        assert common.generation_kwargs(s, label, "free") == {
            "field_temperature": lib["field_temperature"],
            "field_stop_bias": lib["field_stop_bias"]}
    assert s["checkpoint_sha256"] == E7_SHA256


@pytest.mark.skipif(not E7.is_file(), reason="needs data/best_e7.pt")
def test_the_adopted_settings_are_bound_to_e7s_weights():
    from uie.train import load_checkpoint
    assert hashlib.sha256(E7.read_bytes()).hexdigest() == E7_SHA256
    model, tok, _ = load_checkpoint(E7, device=torch.device("cpu"))
    assert tok.include_fields
    assert weights_sha256(model) == ADOPTED.weights_sha256
