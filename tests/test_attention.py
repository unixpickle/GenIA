"""Large-sequence SDPA dispatch: exact query partitions and bounded saved scores."""
import pytest
import torch
import torch.nn.functional as F

from genia.core.ops.attention import scaled_dot_product_attention, packed_self_attention

DEVICES = ["cpu"] + (["mps"] if torch.backends.mps.is_available() else [])


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("mask_kind", ["none", "additive", "broadcast", "boolean", "causal"])
def test_chunked_attention_values_and_gradients(device, dtype, mask_kind):
    torch.manual_seed(190)
    # Unequal Q/K lengths, broadcast batch, distinct value channels, strided inputs.
    base = [torch.randn(b, 2, c, n).transpose(-1, -2).to(dtype).requires_grad_()
            for b, c, n in [(2, 4, 7), (1, 4, 9), (1, 3, 9)]]
    mask = None
    if mask_kind in {"additive", "broadcast"}:
        mask = torch.randn(2, 1, 1 if mask_kind == "broadcast" else 7, 9,
                           dtype=dtype, requires_grad=True)
    elif mask_kind == "boolean":
        mask = torch.rand(7, 9) > .3
        mask[2] = False  # Fully masked rows must produce zero, not NaN.
    args = base + ([mask] if mask is not None else [])
    actual_args = [x.detach().to(device).requires_grad_(x.requires_grad) for x in args]
    expected = F.scaled_dot_product_attention(*base, attn_mask=mask,
                                              is_causal=mask_kind == "causal", scale=.4)
    actual = scaled_dot_product_attention(
        *actual_args[:3], attn_mask=actual_args[3] if mask is not None else None,
        is_causal=mask_kind == "causal", scale=.4, max_score_elements=72,
    )
    tolerance = {torch.float32: 3e-5, torch.float16: .004, torch.bfloat16: .035}[dtype]
    torch.testing.assert_close(actual.cpu(), expected, atol=tolerance, rtol=tolerance)
    upstream = torch.randn_like(expected)
    expected.backward(upstream)
    actual.backward(upstream.to(device))
    for got, ref in zip(actual_args, args):
        if ref.requires_grad:
            torch.testing.assert_close(got.grad.cpu(), ref.grad, atol=tolerance, rtol=tolerance)


def test_chunks_recompute_without_saving_score_matrices(monkeypatch):
    torch.manual_seed(834)
    q, k, v = [torch.randn(1, 2, n, 8, requires_grad=True) for n in (37, 43, 43)]
    original = F.scaled_dot_product_attention
    calls, saved = [], []

    def record(q, k, v, **kwargs):
        calls.append(q.shape[-2] * k.shape[-2] * q.shape[0] * q.shape[1])
        return original(q, k, v, **kwargs)

    def pack(tensor):
        saved.append(tuple(tensor.shape))
        return tensor

    monkeypatch.setattr(F, "scaled_dot_product_attention", record)
    with torch.autograd.graph.saved_tensors_hooks(pack, lambda tensor: tensor):
        result = scaled_dot_product_attention(q, k, v, max_score_elements=258)
    forward_calls = len(calls)
    result.sum().backward()
    assert forward_calls == 13 and len(calls) > forward_calls
    assert max(calls) <= 258
    assert not any(len(shape) >= 2 and shape[-1] == 43 for shape in saved)
    assert all(torch.isfinite(x.grad).all() for x in (q, k, v))


@pytest.mark.parametrize("length,key_length", [(0, 5), (5, 0), (1, 5)])
def test_empty_and_short_attention(length, key_length):
    q = torch.randn(1, 2, length, 4)
    k, v = [torch.randn(1, 2, key_length, 4) for _ in range(2)]
    torch.testing.assert_close(
        scaled_dot_product_attention(q, k, v, max_score_elements=2),
        F.scaled_dot_product_attention(q, k, v),
    )


def test_invalid_chunk_budget():
    q = torch.randn(1, 2, 3, 4)
    with pytest.raises(ValueError, match="positive"):
        scaled_dot_product_attention(q, q, q, max_score_elements=0)


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="Metal required")
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("mask_kind", ["none", "additive", "boolean", "causal"])
def test_mps_inference_autocast_reuses_cast_keys(monkeypatch, dtype, mask_kind):
    torch.manual_seed(408)
    q = torch.randn(2, 2, 8, 19, device="mps").transpose(-1, -2)
    k = torch.randn(1, 2, 8, 23, device="mps").transpose(-1, -2)
    v = torch.randn_like(k).to(dtype)
    mask = None
    if mask_kind == "additive":
        mask = torch.randn(19, 23, device="mps")
    elif mask_kind == "boolean":
        mask = torch.rand(19, 23, device="mps") > .3
        mask[2] = False
    original = F.scaled_dot_product_attention
    calls = []

    def record(q, k, v, **kwargs):
        bias = kwargs.get("attn_mask")
        calls.append((q.dtype, k.dtype, v.dtype, k.data_ptr(), v.data_ptr(),
                      None if bias is None else bias.dtype))
        return original(q, k, v, **kwargs)

    with torch.inference_mode(), torch.autocast("mps", dtype=dtype):
        cpu_mask = None if mask is None else mask.cpu().to(
            torch.bool if mask.dtype == torch.bool else dtype)
        expected = original(q.cpu().to(dtype), k.cpu().to(dtype), v.cpu(),
                            attn_mask=cpu_mask, is_causal=mask_kind == "causal")
        monkeypatch.setattr(F, "scaled_dot_product_attention", record)
        actual = scaled_dot_product_attention(
            q, k, v, attn_mask=mask, is_causal=mask_kind == "causal",
            max_score_elements=276,
        )
    tolerance = .004 if dtype == torch.float16 else .035
    torch.testing.assert_close(actual.cpu(), expected.cpu(), atol=tolerance, rtol=tolerance)
    assert len(calls) > 1
    assert all(call[:3] == (dtype, dtype, dtype) for call in calls)
    assert len({call[3] for call in calls}) == len({call[4] for call in calls}) == 1
    expected_mask_dtype = {"none": None, "additive": dtype,
                           "boolean": torch.bool, "causal": torch.bool}[mask_kind]
    assert all(call[5] == expected_mask_dtype for call in calls)


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="Metal required")
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("grad_source", ["all", "keys"])
def test_mps_autocast_keeps_per_chunk_gradient_casts(monkeypatch, dtype, grad_source):
    torch.manual_seed(507)
    tensors = [torch.randn(2, 2, n, 8, device="mps") for n in (19, 23, 23)]
    tensors[2] = tensors[2].to(dtype)
    tensors.append(torch.randn(19, 23, device="mps"))
    for index, tensor in enumerate(tensors):
        tensor.requires_grad_(grad_source == "all" or index == 1)
    refs = [tensor.detach().clone().requires_grad_(tensor.requires_grad) for tensor in tensors]
    original = F.scaled_dot_product_attention
    with torch.autocast("mps", dtype=dtype):
        expected = torch.cat([
            original(refs[0][..., start:start + 3, :], refs[1], refs[2],
                     attn_mask=refs[3][start:start + 3])
            for start in range(0, 19, 3)
        ], dim=-2)
    expected.float().square().mean().backward()
    seen = []

    def record(q, k, v, **kwargs):
        seen.append((q.dtype, k.dtype, kwargs["attn_mask"].dtype))
        return original(q, k, v, **kwargs)

    monkeypatch.setattr(F, "scaled_dot_product_attention", record)
    with torch.autocast("mps", dtype=dtype):
        actual = scaled_dot_product_attention(*tensors[:3], attn_mask=tensors[3],
                                               max_score_elements=276)
    actual.float().square().mean().backward()
    torch.testing.assert_close(actual.cpu(), expected.cpu(), atol=2e-5, rtol=.005)
    assert seen and all(dtypes == (torch.float32,) * 3 for dtypes in seen)
    for tensor, reference in zip(tensors, refs):
        if reference.requires_grad:
            torch.testing.assert_close(tensor.grad.cpu(), reference.grad.cpu(),
                                       atol=2e-5, rtol=.005)


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="Metal required")
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("chunked", [False, True])
def test_mps_attention_mask_only_gradient(dtype, chunked):
    torch.manual_seed(917)
    q, k, v = [torch.randn(2, 2, n, 8) for n in (7, 9, 9)]
    mask = torch.randn(1, 1, 7, 9, requires_grad=True)
    # Independent float32 softmax calculation on the autocast-rounded inputs.
    scores = q.to(dtype).float() @ k.to(dtype).float().transpose(-1, -2) / 8 ** .5
    expected = (scores + mask.to(dtype).float()).softmax(-1) @ v.to(dtype).float()
    expected = expected.to(dtype)
    expected.float().square().mean().backward()
    actual_mask = mask.detach().to("mps").requires_grad_()
    with torch.autocast("mps", dtype=dtype):
        actual = scaled_dot_product_attention(
            q.to("mps"), k.to("mps"), v.to("mps"), attn_mask=actual_mask,
            max_score_elements=72 if chunked else None,
        )
    actual.float().square().mean().backward()
    tolerance = .004 if dtype == torch.float16 else .035
    torch.testing.assert_close(actual.cpu(), expected, atol=tolerance, rtol=tolerance)
    torch.testing.assert_close(actual_mask.grad.cpu(), mask.grad, atol=2e-5, rtol=.05)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("lengths", [[3, 0, 1, 17, 4, 3, 9], [1, 1, 1], [0, 0]])
def test_packed_window_buckets_preserve_values_order_and_gradients(device, dtype, lengths):
    torch.manual_seed(445)
    source = torch.randn(sum(lengths), 3, 2, 4, dtype=dtype, requires_grad=True)
    refs, start = [], 0
    for length in lengths:
        if length:
            q, k, v = source[start:start+length].unbind(1)
            probs = (q.transpose(0, 1).float() @ k.transpose(0, 1).float().transpose(-1, -2) / 2).softmax(-1)
            refs.append((probs @ v.transpose(0, 1).float()).transpose(0, 1).to(dtype))
        start += length
    expected = torch.cat(refs) if refs else source[:, 0]
    x = source.detach().to(device).requires_grad_()
    actual = packed_self_attention(x, lengths, max_score_elements=64)
    tol = {torch.float32: 3e-5, torch.float16: .004, torch.bfloat16: .035}[dtype]
    torch.testing.assert_close(actual.cpu(), expected, atol=tol, rtol=tol)
    grad = torch.randn_like(expected)
    actual.backward(grad.to(device)); expected.backward(grad)
    torch.testing.assert_close(x.grad.cpu(), source.grad, atol=tol, rtol=tol)


def test_packed_window_invalid_lengths():
    x = torch.randn(3, 3, 2, 4)
    for lengths in [[1, 1], [4, -1]]:
        with pytest.raises(ValueError, match="lengths"):
            packed_self_attention(x, lengths)
