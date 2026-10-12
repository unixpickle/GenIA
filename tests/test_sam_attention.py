"""SAM dense attention must use bounded MPS scores, including packed cross-attention."""
import os

os.environ.setdefault("LIDRA_SKIP_INIT", "1")
os.environ["ATTN_BACKEND"] = "sdpa"

import pytest
import torch
import torch.nn.functional as F
from sam3d_objects.model.backbone.tdfy_dit.modules.attention import full_attn


@pytest.mark.parametrize("device", ["cpu", "mps"])
@pytest.mark.parametrize("packing", ["qkv", "kv", "separate"])
def test_sam_attention_packing_values_and_gradients(device, packing):
    if device == "mps" and not torch.backends.mps.is_available():
        pytest.skip("requires Metal")
    torch.manual_seed(813)
    lengths = (7, 7, 7) if packing == "qkv" else (7, 11, 11)
    refs = [torch.randn(2, n, 3, 4, requires_grad=True) for n in lengths]
    q, k, v = [x.transpose(1, 2) for x in refs]
    expected = ((q @ k.transpose(-1, -2) / 2).softmax(-1) @ v).transpose(1, 2)
    inputs = [x.detach().to(device).requires_grad_() for x in refs]
    if packing == "qkv":
        actual = full_attn.scaled_dot_product_attention(torch.stack(inputs, dim=2))
    elif packing == "kv":
        actual = full_attn.scaled_dot_product_attention(inputs[0], torch.stack(inputs[1:], dim=2))
    else:
        actual = full_attn.scaled_dot_product_attention(*inputs)
    torch.testing.assert_close(actual.cpu(), expected, atol=3e-5, rtol=3e-5)
    upstream = torch.randn_like(expected)
    expected.backward(upstream)
    actual.backward(upstream.to(device))
    for got, ref in zip(inputs, refs):
        torch.testing.assert_close(got.grad.cpu(), ref.grad, atol=3e-5, rtol=3e-5)


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="requires Metal")
def test_sam_sixteen_frame_cross_attention_bounds_score_allocation(monkeypatch):
    torch.manual_seed(912)
    # Force the memory-aware inference budget to its 16M-element minimum.
    # Exercise the real packed KV entry point without recreating the pretrained
    # model's 14.7 GB allocation; a direct native alias must still fail this test.
    monkeypatch.setattr(torch.mps, "recommended_max_memory", lambda: 1024**3)
    monkeypatch.setattr(torch.mps, "current_allocated_memory", lambda: 0)
    q = torch.randn(16, 257, 4, 8)
    kv = torch.randn(16, 1031, 2, 4, 8)
    native = F.scaled_dot_product_attention
    calls = []

    def record(query, key, value, **kwargs):
        elements = query.shape[0] * query.shape[1] * query.shape[2] * key.shape[2]
        calls.append(elements)
        assert elements <= 16 * 1024 * 1024, "SAM bypassed bounded MPS attention"
        return native(query, key, value, **kwargs)

    k, v = kv.unbind(2)
    with torch.inference_mode():
        expected = native(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)).transpose(1, 2)
        monkeypatch.setattr(F, "scaled_dot_product_attention", record)
        # Patch the old direct alias too, so the regression fails before the fix.
        if full_attn.sdpa is native:
            monkeypatch.setattr(full_attn, "sdpa", record)
        actual = full_attn.scaled_dot_product_attention(q.to("mps"), kv.to("mps"))
    assert len(calls) > 1
    torch.testing.assert_close(actual.cpu(), expected, atol=3e-5, rtol=3e-5)
