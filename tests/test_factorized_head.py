"""E6: the factorised output head.

Targets a mechanism measured on E5 -- EOS shares one softmax with 84 content
characters, so mid-path EOS mass rises 0.111% -> 3.022% (27x) purely with the
model's uncertainty about the next CHARACTER. Over a 28-character path a ~1.8%
per-step leak compounds to 0.982**28 = 0.60 survival.

Splitting P(shape) from P(character | content) removes the coupling by
construction rather than correcting it after the fact -- which matters because
the correction was tried and rejected: a constant logit bias cannot track an
entropy-proportional leak, and fixing length that way inflated depth (1.8 -> 2.1)
and host length (17.2 -> 18.4).
"""
from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

from uie.model import ModelConfig, URLTransformer
from uie.tokenizer import URLTokenizer

STRUCT = (2, 3, 5)


def _cfg(struct=STRUCT):
    return ModelConfig(vocab_size=16, max_len=24, d_model=16, n_layers=1,
                       n_heads=2, d_ff=32, struct_ids=struct)


def test_output_is_a_normalised_distribution():
    m = URLTransformer(_cfg()).eval()
    p = torch.softmax(m(torch.randint(0, 16, (3, 7))), dim=-1)
    assert torch.allclose(p.sum(-1), torch.ones(3, 7), atol=1e-5)


def test_returned_values_are_log_probabilities():
    """Callers are unchanged because softmax(log p) == p. If raw logits were
    returned instead, every decoder and probe would silently renormalise across
    the factorisation and undo it."""
    m = URLTransformer(_cfg()).eval()
    out = m(torch.randint(0, 16, (2, 5)))
    assert bool((out <= 1e-5).all())
    assert torch.allclose(out.exp().sum(-1), torch.ones(2, 5), atol=1e-5)
    # log_softmax is idempotent on log-probs, so cross_entropy stays correct.
    assert torch.allclose(F.log_softmax(out, dim=-1), out, atol=1e-5)


def test_disabled_by_default_and_unchanged():
    """E5 must remain reproducible: no struct_ids means the classic head."""
    assert ModelConfig().struct_ids == ()
    m = URLTransformer(ModelConfig(vocab_size=16, max_len=24, d_model=16,
                                   n_layers=1, n_heads=2, d_ff=32)).eval()
    assert not hasattr(m, "gate")
    out = m(torch.randint(0, 16, (2, 5)))
    assert not torch.allclose(out.exp().sum(-1), torch.ones(2, 5), atol=1e-3)


def test_costs_only_the_gate():
    a = sum(p.numel() for p in URLTransformer(_cfg(())).parameters())
    b = sum(p.numel() for p in URLTransformer(_cfg()).parameters())
    assert b - a == 16 + 1              # d_model weights + one bias


def test_group_mass_is_the_gate_alone():
    """The property the whole change exists for.

    Total probability of the STRUCTURAL group must equal sigmoid(gate(h)) and
    nothing else -- so no amount of character-level uncertainty in the content
    head can move it. Under a shared softmax the group mass is a function of
    every content logit, which is exactly the leak measured on E5.

    Asserted against the hidden state directly. It cannot be tested by
    perturbing `head.weight`, because that tensor is TIED to `tok_emb.weight`:
    changing it also changes the input embedding and therefore h itself, so the
    premise "for a fixed h" would be violated by the test.
    """
    m = URLTransformer(_cfg()).eval()
    ids = torch.randint(0, 16, (4, 6))
    with torch.no_grad():
        _, t = ids.shape
        pos = torch.arange(t)
        x = m.tok_emb(ids) + m.pos_emb(pos)
        for block in m.blocks:
            x = block(x)
        h = m.ln_f(x)
        expected = torch.sigmoid(m.gate(h)).squeeze(-1)
        actual = torch.softmax(m(ids), -1)[..., list(STRUCT)].sum(-1)
    assert torch.allclose(expected, actual, atol=1e-5), (expected, actual)


def test_content_distribution_is_conditional_and_sums_to_one():
    """Within the content group the characters must form their own normalised
    distribution, so entropy there is fully contained."""
    m = URLTransformer(_cfg()).eval()
    p = torch.softmax(m(torch.randint(0, 16, (3, 5))), -1)
    content = p[..., list(m.content_idx)]
    assert torch.allclose(content.sum(-1) / (1 - p[..., list(STRUCT)].sum(-1)),
                          torch.ones(3, 5), atol=1e-4)


def test_loss_and_gradients_flow_through_the_gate():
    m = URLTransformer(_cfg())
    ids = torch.randint(1, 16, (3, 8))
    loss = m.loss(ids, pad_id=0)
    assert torch.isfinite(loss)
    loss.backward()
    assert m.gate.weight.grad is not None
    assert float(m.gate.weight.grad.abs().sum()) > 0


def test_tokenizer_supplies_the_structural_alphabet():
    tok = URLTokenizer(include_scheme=False, include_language=False,
                       include_frame=True).fit(["https://a.co.uk/x"], min_count=1)
    ids = tok.struct_ids()
    assert tok.eos_id in ids
    assert tok.marker_id("PTH") in ids
    assert tok.marker_id("LBL") in ids
    assert tok.bos_id not in ids          # never a target
    assert len(set(ids)) == len(ids)


def test_a_factorized_model_trains_end_to_end():
    from uie.train import TrainConfig, URLDataset, encode_corpus, train

    urls = ["https://a.co.uk/login/verify", "https://b.com/x", "https://c.org/"]
    tok = URLTokenizer(include_scheme=False, include_language=False,
                       include_frame=True).fit(urls, min_count=1)
    enc, _ = encode_corpus(urls * 8, ["benign"] * 24, tok, 64,
                           corpora=["name", "structure", "name"] * 8)
    ds = URLDataset(enc)
    cfg = ModelConfig(vocab_size=tok.vocab_size, max_len=64, d_model=16,
                      n_layers=1, n_heads=2, d_ff=32,
                      struct_ids=tok.struct_ids())
    res = train(URLTransformer(cfg), ds, ds, tok,
                TrainConfig(steps=3, batch_size=4, warmup=1, eval_every=5,
                            eval_batches=1))
    assert res is not None


def test_forward_and_loss_survive_autocast():
    """REGRESSION: the E6 launch crashed here, and every prior test missed it.

        RuntimeError: index_copy_(): self and source expected to have the same
        dtype, but got (self) Half and (source) Float

    Under autocast `head` returns fp16 while `log_softmax` and `logsigmoid` are
    promoted to fp32, so a buffer typed from `logits` is Half and the sources
    are Float. Training uses AMP on GPU; every test ran on CPU without it, so
    the code path that actually runs was never exercised. CPU autocast applies
    the same promotion policy and reproduces it without a GPU.
    """
    m = URLTransformer(_cfg())
    ids = torch.randint(1, 16, (3, 8))
    with torch.amp.autocast("cpu", dtype=torch.bfloat16):
        out = m(ids[:, :-1])
        assert out.dtype == torch.float32, out.dtype
        assert torch.allclose(out.exp().sum(-1), torch.ones(3, 7), atol=1e-2)
        loss = m.loss(ids, pad_id=0)
    assert torch.isfinite(loss)
    loss.backward()
    assert m.gate.weight.grad is not None


def test_unfactorized_head_also_survives_autocast():
    """The control: the classic head must be unaffected by the fix."""
    m = URLTransformer(ModelConfig(vocab_size=16, max_len=24, d_model=16,
                                   n_layers=1, n_heads=2, d_ff=32))
    ids = torch.randint(1, 16, (3, 8))
    with torch.amp.autocast("cpu", dtype=torch.bfloat16):
        loss = m.loss(ids, pad_id=0)
    assert torch.isfinite(loss)
