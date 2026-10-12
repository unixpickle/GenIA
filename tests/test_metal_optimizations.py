"""Compare optimized dispatch with the pre-optimization kernels and autograd."""
import importlib.util
from pathlib import Path

import pytest
import torch

from genia.core.ops.attention import scaled_dot_product_attention
from genia.core.ops.gaussian import _MetalSplat, gaussian_tile_intersections
from genia.core.ops.knn import knn_points
from genia.core.ops.sparse import submanifold_conv3d, _submanifold_conv3d_reference

pytestmark = pytest.mark.skipif(not torch.backends.mps.is_available(), reason="Metal required")


def reference(name):
    spec = importlib.util.spec_from_file_location(name, Path(__file__).parent / "reference" / (name + ".py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("norm", [1, 2])
def test_cooperative_knn_matches_generic_values_gradients_and_padding(dtype, norm):
    torch.manual_seed(124)
    # Non-multiple of eight queries, strided inputs, empty reference cloud,
    # ties across SIMD lanes, and both input gradients.
    a = torch.randn(3, 3, 13, device="mps").transpose(1, 2).to(dtype)
    b = torch.randn(3, 3, 291, device="mps").transpose(1, 2).to(dtype)
    b[0, 35] = b[0, 3]; a[0, 0] = b[0, 3]
    lengths = [torch.tensor([11, 13, 0]), torch.tensor([281, 0, 200])]
    fn = reference("knn_baseline").knn_points
    for grad in [False, True]:
        old = [x.detach().clone().requires_grad_(grad) for x in (a, b)]
        new = [x.detach().clone().requires_grad_(grad) for x in (a, b)]
        expected = fn(*old, *lengths, norm=norm, return_nn=True)
        actual = knn_points(*new, *lengths, norm=norm, return_nn=True)
        for got, ref in zip(actual, expected):
            torch.testing.assert_close(got, ref, atol=2e-6, rtol=2e-6)
        if grad:
            expected.dists.sum().backward(); actual.dists.sum().backward()
            for got, ref in zip(new, old):
                torch.testing.assert_close(got.grad, ref.grad, atol=.005, rtol=.005)


def test_cooperative_knn_nonfinite_fallback():
    a = torch.zeros(1, 8, 3, device="mps")
    b = torch.full((1, 256, 3), torch.nan, device="mps")
    old = reference("knn_baseline").knn_points(a, b)
    new = knn_points(a, b)
    torch.testing.assert_close(old.idx, new.idx)
    torch.testing.assert_close(old.dists, new.dists, equal_nan=True)


@pytest.mark.parametrize("channels", [8, 64, 192])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_grouped_sparse_convolution_matches_offset_loop(channels, dtype):
    torch.manual_seed(893)
    n = 311
    neighbors = torch.randint(-n, n, (27, n), device="mps").clamp_min(-1)
    f = torch.randn(n, channels, device="mps", dtype=dtype)
    w = torch.randn(17, 3, 3, 3, channels, device="mps", dtype=dtype) * .02
    bias = torch.randn(17, device="mps", dtype=dtype)
    old = [x.detach().clone().requires_grad_() for x in (f, w, bias)]
    new = [x.detach().clone().requires_grad_() for x in (f, w, bias)]
    expected = _submanifold_conv3d_reference(*old[:2], neighbors) + old[2]
    actual = submanifold_conv3d(*new[:2], neighbors, new[2])
    tol = {torch.float32: 3e-5, torch.float16: .006, torch.bfloat16: .06}[dtype]
    torch.testing.assert_close(actual, expected, atol=tol, rtol=tol)
    grad = torch.randn_like(actual) * .1
    actual.backward(grad); expected.backward(grad)
    for got, ref in zip(new, old):
        torch.testing.assert_close(got.grad, ref.grad, atol=tol, rtol=tol)


@pytest.mark.parametrize("radius,opacity", [(2, .1), (9, .3), (35, .95), (35, .01)])
def test_tiled_splat_matches_original_values_and_gradients(radius, opacity):
    torch.manual_seed(301)
    n, h, w = 353, 29, 37  # Partial tiles, reverse batches, and early termination.
    means = torch.rand(n, 2, device="mps") * torch.tensor([w, h], device="mps")
    conics = torch.tensor([1 / radius**2, 0, 1 / radius**2], device="mps").expand(n, 3).contiguous()
    inputs = [means, conics, torch.rand(n, 4, device="mps"),
              torch.full((n,), opacity, device="mps"), torch.rand(4, device="mps")]
    radii = torch.full((n, 2), radius * 3, dtype=torch.int32, device="mps")
    ranges, ids = gaussian_tile_intersections(means, radii, torch.rand(n, device="mps"), w, h)
    old = [x.detach().clone().requires_grad_() for x in inputs]
    new = [x.detach().clone().requires_grad_() for x in inputs]
    expected = reference("splat_baseline")._MetalSplat.apply(*old, ranges, ids, w, h, 16)
    actual = _MetalSplat.apply(*new, ranges, ids, w, h, 16)
    for got, ref in zip(actual, expected):
        torch.testing.assert_close(got, ref, atol=0, rtol=0)
    grads = [torch.randn_like(x) for x in actual]
    torch.autograd.backward(actual, grads); torch.autograd.backward(expected, grads)
    for got, ref in zip(new, old):
        torch.testing.assert_close(got.grad, ref.grad, atol=5e-4, rtol=2e-4)


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("mask_kind", ["none", "boolean", "causal"])
def test_larger_inference_attention_budget_matches_old_chunks(dtype, mask_kind):
    torch.manual_seed(990)
    # Old default must split queries; new default fits this entire invocation.
    q, k, v = [torch.randn(2, 4, 1537, 32, device="mps", dtype=dtype) for _ in range(3)]
    mask = torch.rand(1537, 1537, device="mps") > .2 if mask_kind == "boolean" else None
    if mask is not None:
        mask[15] = False
    with torch.inference_mode():
        expected = scaled_dot_product_attention(q, k, v, mask, is_causal=mask_kind == "causal",
                                                 max_score_elements=16 * 1024**2)
        actual = scaled_dot_product_attention(q, k, v, mask, is_causal=mask_kind == "causal")
    torch.testing.assert_close(actual, expected, atol=2e-5 if dtype == torch.float32 else .002,
                               rtol=2e-4 if dtype == torch.float32 else .02)


def test_grouped_sparse_multiple_chunks():
    torch.manual_seed(318)
    n, c = 5001, 64  # Exceeds the eight-million-element gather budget.
    neighbors = torch.randint(-n, n, (27, n), device="mps").clamp_min(-1)
    old = [torch.randn(n, c, device="mps", requires_grad=True),
           (torch.randn(5, 3, 3, 3, c, device="mps") * .01).requires_grad_()]
    new = [x.detach().clone().requires_grad_() for x in old]
    expected = _submanifold_conv3d_reference(*old, neighbors)
    actual = submanifold_conv3d(*new, neighbors)
    torch.testing.assert_close(actual, expected, atol=3e-5, rtol=3e-5)
    grad = torch.randn_like(actual) * .01
    actual.backward(grad); expected.backward(grad)
    for got, ref in zip(new, old):
        torch.testing.assert_close(got.grad, ref.grad, atol=3e-5, rtol=3e-5)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_grouped_sparse_autocast_preserves_output_dtype_and_gradients(dtype):
    torch.manual_seed(59)
    n, c = 311, 64
    neighbors = torch.randint(-n, n, (27, n), device="mps").clamp_min(-1)
    old = [torch.randn(n, c, device="mps", requires_grad=True),
           (torch.randn(5, 3, 3, 3, c, device="mps") * .01).requires_grad_()]
    new = [x.detach().clone().requires_grad_() for x in old]
    with torch.autocast("mps", dtype=dtype):
        expected = _submanifold_conv3d_reference(*old, neighbors)
        actual = submanifold_conv3d(*new, neighbors)
    assert actual.dtype == expected.dtype == torch.float32
    tolerance = .003 if dtype == torch.float16 else .025
    torch.testing.assert_close(actual, expected, atol=tolerance, rtol=tolerance)
    grad = torch.randn_like(actual) * .01
    actual.backward(grad); expected.backward(grad)
    for got, ref in zip(new, old):
        torch.testing.assert_close(got.grad, ref.grad, atol=tolerance, rtol=tolerance)


def test_attention_reduces_default_budget_under_memory_pressure(monkeypatch):
    import torch.nn.functional as F
    q = torch.randn(2, 4, 1537, 32, device="mps")
    calls = []
    original = F.scaled_dot_product_attention
    def record(q, k, v, **kwargs):
        calls.append(q.shape[-2])
        return original(q, k, v, **kwargs)
    monkeypatch.setattr(F, "scaled_dot_product_attention", record)
    monkeypatch.setattr(torch.mps, "recommended_max_memory", lambda: 1024)
    monkeypatch.setattr(torch.mps, "current_allocated_memory", lambda: 2048)
    with torch.inference_mode():
        scaled_dot_product_attention(q, q, q)
    assert len(calls) == 2 and max(calls) <= 16 * 1024**2 // (8 * 1537)
