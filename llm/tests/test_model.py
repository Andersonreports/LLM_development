"""Correctness tests for the Transformer: shapes, causal masking, both attention paths,
initial loss, generation, checkpoint round-trip, and that it can actually learn."""
from __future__ import annotations

import math

import pytest
import torch

from tinygpt import GPT, GPTConfig, preset


def _model(**kw) -> GPT:
    torch.manual_seed(0)
    return GPT(GPTConfig(**{**dict(vocab_size=50, block_size=16, n_layer=2, n_head=2, n_embd=32, dropout=0.0), **kw}))


def test_output_shapes():
    m = _model()
    idx = torch.randint(0, 50, (3, 10))
    logits, loss = m(idx, idx)
    assert logits.shape == (3, 10, 50) and loss.ndim == 0
    logits, loss = m(idx)                      # inference: last position only
    assert logits.shape == (3, 1, 50) and loss is None


def test_initial_loss_is_close_to_uniform_guess():
    # an untrained model should be ~equally unsure about every token: loss ~ ln(vocab_size)
    m = _model()
    idx = torch.randint(0, 50, (8, 16))
    _, loss = m(idx, torch.randint(0, 50, (8, 16)))
    assert abs(loss.item() - math.log(50)) < 0.3


@pytest.mark.parametrize("fused", [True, False])
def test_causal_mask_hides_the_future(fused):
    # changing token t must not change the predictions at positions before t
    m = _model(fused_attention=fused).eval()
    a = torch.randint(0, 50, (1, 12))
    b = a.clone()
    b[0, 8:] = (b[0, 8:] + 1) % 50
    la, _ = m(a, a)
    lb, _ = m(b, b)
    assert torch.allclose(la[0, :8], lb[0, :8], atol=1e-5)
    assert not torch.allclose(la[0, 8:], lb[0, 8:], atol=1e-5)


def test_fused_and_manual_attention_agree():
    fused, manual = _model(fused_attention=True).eval(), _model(fused_attention=False).eval()
    manual.load_state_dict(fused.state_dict())
    idx = torch.randint(0, 50, (2, 16))
    assert torch.allclose(fused(idx, idx)[0], manual(idx, idx)[0], atol=1e-5)


def test_sequence_longer_than_context_is_rejected():
    with pytest.raises(ValueError):
        _model()(torch.zeros(1, 17, dtype=torch.long))


def test_generate_appends_tokens_and_crops_context():
    m = _model().eval()
    start = torch.zeros(2, 1, dtype=torch.long)
    out = m.generate(start, max_new_tokens=40, temperature=0.8, top_k=10)  # 41 > block_size 16
    assert out.shape == (2, 41) and int(out.max()) < 50
    greedy1 = m.generate(start, max_new_tokens=5, temperature=0)
    greedy2 = m.generate(start, max_new_tokens=5, temperature=0)
    assert torch.equal(greedy1, greedy2)      # temperature 0 is deterministic
    nucleus = m.generate(start, max_new_tokens=5, top_p=0.9)
    assert nucleus.shape == (2, 6)


def test_checkpoint_round_trip(tmp_path):
    m = _model().eval()
    path = tmp_path / "ckpt.pt"
    torch.save({"config": m.cfg.to_dict(), "model": m.state_dict()}, path)
    ck = torch.load(path, weights_only=True)
    m2 = GPT(GPTConfig(**ck["config"])).eval()
    m2.load_state_dict(ck["model"])
    idx = torch.randint(0, 50, (2, 16))
    assert torch.equal(m(idx, idx)[0], m2(idx, idx)[0])


def test_weights_are_tied_and_param_count_matches_preset():
    m = GPT(preset("small", vocab_size=4096))
    assert m.lm_head.weight is m.tok_emb.weight
    assert 10e6 < m.num_params() < 14e6


def test_model_can_memorise_a_sequence():
    # sanity check that gradients flow: a small model should overfit one fixed batch
    m = _model()
    opt = m.configure_optimizer(lr=3e-3, weight_decay=0.0)
    seq = torch.randint(0, 50, (4, 17))
    x, y = seq[:, :-1], seq[:, 1:]
    for _ in range(150):
        _, loss = m(x, y)
        opt.zero_grad()
        loss.backward()
        opt.step()
    assert loss.item() < 0.1
