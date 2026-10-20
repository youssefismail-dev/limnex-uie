"""research/e13/e13_fit.py (PREREG_E13 V6(d)): the fit scores exactly the steps the
sampler samples, with the sampler's own field tracker and transform. Proven the
DC1 way: URLs generated with field settings record their summed log-probability
(`logp`); the fit's Layout, applied to the same ids, must reproduce it.
"""
import sys
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "research" / "e13"))

from e13_fit import Layout, transformed  # noqa: E402
from test_e7_fields import FIT  # noqa: E402
from uie.model import ModelConfig, URLTransformer, generate  # noqa: E402
from uie.tokenizer import URLTokenizer  # noqa: E402


def tok():
    return URLTokenizer(include_frame=True, include_fields=True, root_slash=True,
                        presence_plan=True, slash_plan=True).fit(FIT)


@pytest.mark.parametrize("field,plan", [("QRY", {"PTH": False, "QRY": True, "SLASH": True}),
                                        ("PTH", {"PTH": True, "QRY": True, "SLASH": True})])
def test_the_fit_reproduces_the_samplers_log_probability(field, plan):
    t = tok()
    torch.manual_seed(1)
    m = URLTransformer(ModelConfig(vocab_size=t.vocab_size, max_len=96, d_model=16, n_layers=1,
                                   n_heads=2, d_ff=32)).eval()
    T, d = 0.8, 0.7
    ids_out, logp = [], []
    generate(m, t, label="benign", n=24, suffix="com", seed=3, fields=plan,
             field_temperature={field: T}, field_stop_bias={field: d}, ids_out=ids_out, logp=logp)
    layout = Layout(t)
    checked = 0
    for ids, lp in zip(ids_out, logp):
        if layout.suf not in ids[3:] or layout.lbl not in ids:
            continue                                     # the model never asked for the suffix
        z = m(torch.tensor([ids]))[0].float()
        total = 0.0
        for pos, f, y in layout.steps(ids):
            a = z[pos:pos + 1]
            a = transformed(a, layout, T, d) if f == field else a
            total += float(F.log_softmax(a, dim=-1)[0, y])
        assert abs(total - lp) < 1e-3, (total, lp)
        checked += 1
    assert checked >= 5


def test_the_state_is_read_from_the_encodings_own_plan():
    t = tok()
    layout = Layout(t)
    assert layout.state(t.encode("https://a.example.com/?q=1", "benign")) == "P0Q1S1"
    assert layout.state(t.encode("https://a.example.com?q=1", "benign")) == "P0Q1S0"
    assert layout.state(t.encode("https://a.example.com/x/y?q=1", "benign")) == "P1Q1S1"
    assert layout.state(t.encode("https://a.example.com", "benign")) == "P0Q0S0"
