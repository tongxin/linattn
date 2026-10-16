"""The compressed-state family: the chunked kernel must equal the recurrence.

This is the central numerical claim of ``variants/recurrent.py``: the chunkwise
algorithm (matmuls + one triangular solve per chunk) is not an approximation of
the sequential recurrence, it is an exact re-association of it.  If these tests
pass in float64, the chunked code is a valid kernel specification.
"""

from __future__ import annotations

import pytest
import torch

from linattn.variants.recurrent import LinearAttentionConfig, LinearAttentionLayer


@pytest.mark.parametrize("kind", ["ssd", "gdn", "kda"])
def test_chunked_equals_recurrent(kind):
    torch.manual_seed(0)
    cfg = LinearAttentionConfig(d_model=32, n_heads=2, d_state=16, chunk=16, kind=kind, decay_clamp=2.0)
    layer = LinearAttentionLayer(cfg).double()
    x = torch.randn(2, 64, cfg.d_model, dtype=torch.float64)
    with torch.no_grad():
        y_chunk, s_chunk = layer.forward_chunked(x)
        y_rec, s_rec = layer.forward_recurrent(x)
    assert torch.allclose(y_chunk, y_rec, atol=1e-8), f"{kind}: outputs differ"
    assert torch.allclose(s_chunk, s_rec, atol=1e-8), f"{kind}: states differ"


@pytest.mark.parametrize("kind", ["ssd", "gdn", "kda"])
def test_chunk_size_invariance(kind):
    torch.manual_seed(1)
    x = torch.randn(1, 40, 32, dtype=torch.float64)
    outs = []
    for chunk in (8, 20, 40):
        cfg = LinearAttentionConfig(d_model=32, n_heads=2, d_state=16, chunk=chunk, kind=kind, decay_clamp=2.0)
        layer = LinearAttentionLayer(cfg).double()
        with torch.no_grad():
            layer.q_proj.weight.zero_(); layer.k_proj.weight.zero_(); layer.v_proj.weight.zero_()
        # re-randomise deterministically so all chunk sizes see the same weights
        gen = torch.Generator().manual_seed(7)
        with torch.no_grad():
            for param in (layer.q_proj.weight, layer.k_proj.weight, layer.v_proj.weight,
                          layer.o_proj.weight, layer.decay_proj.weight, layer.beta_proj.weight):
                param.copy_(torch.randn(param.shape, generator=gen, dtype=torch.float64) * 0.3)
            layer.A_log.copy_(torch.randn(layer.A_log.shape, generator=gen, dtype=torch.float64))
        with torch.no_grad():
            out, _ = layer.forward_chunked(x)
        outs.append(out)
    assert torch.allclose(outs[0], outs[1], atol=1e-8)
    assert torch.allclose(outs[1], outs[2], atol=1e-8)


@pytest.mark.parametrize("kind", ["ssd", "gdn", "kda"])
def test_gradients_flow_and_match_between_paths(kind):
    """Chunked and recurrent paths must produce the same gradients too, otherwise
    the "trainable sparse/linear layer" claim would only hold for inference."""
    torch.manual_seed(2)
    cfg = LinearAttentionConfig(d_model=16, n_heads=2, d_state=8, chunk=8, kind=kind, decay_clamp=2.0)
    x = torch.randn(1, 16, cfg.d_model, dtype=torch.float64)

    def grads(mode):
        layer = LinearAttentionLayer(cfg).double()
        gen = torch.Generator().manual_seed(11)
        with torch.no_grad():
            for param in layer.parameters():
                param.copy_(torch.randn(param.shape, generator=gen, dtype=torch.float64) * 0.3)
        out, _ = layer.forward(x, mode=mode)
        loss = out.square().mean()
        layer.zero_grad()
        loss.backward()
        return {name: (None if param.grad is None else param.grad.clone())
                for name, param in layer.named_parameters()}

    g_chunk, g_rec = grads("chunked"), grads("recurrent")
    for name in g_chunk:
        if g_chunk[name] is None or g_rec[name] is None:
            continue          # e.g. A_log is unused by the channel-wise-decay variant
        assert torch.allclose(g_chunk[name], g_rec[name], atol=1e-7), f"{kind}: grad mismatch in {name}"


def test_state_is_constant_size():
    cfg = LinearAttentionConfig(d_model=32, n_heads=4, d_state=16, kind="kda")
    layer = LinearAttentionLayer(cfg)
    assert layer.state_bytes(1) == 4 * 4 * 16 * (cfg.d_value)
    # the state does not depend on sequence length: that is the entire point
    x1 = torch.randn(1, 8, 32)
    x2 = torch.randn(1, 128, 32)
    _, s1 = layer.forward_recurrent(x1)
    _, s2 = layer.forward_recurrent(x2)
    assert s1.shape == s2.shape
