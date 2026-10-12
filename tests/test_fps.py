"""Farthest sampling against PyTorch3D's unchanged Python reference."""
import importlib.util
from pathlib import Path

import pytest
import torch
from genia.core.ops.fps import masked_gather, sample_farthest_points

DEVICES = ["cpu"] + (["mps"] if torch.backends.mps.is_available() else [])
spec = importlib.util.spec_from_file_location("fps_reference", Path(__file__).parent / "reference/fps.py")
reference = importlib.util.module_from_spec(spec)
spec.loader.exec_module(reference)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("d", [3, 6, 17])
def test_ragged_sampling_values_indices_and_gradients(device, d):
    torch.manual_seed(79)
    points = torch.randn(3, 257, d, requires_grad=True)
    lengths = torch.tensor([257, 33, 101])
    k = [29, 47, 15]
    expected, indices = reference.sample_farthest_points_naive(points, lengths, k)
    values = points.detach().to(device).requires_grad_()
    actual, got_indices = sample_farthest_points(values, lengths.to(device), k)
    torch.testing.assert_close(got_indices.cpu(), indices, rtol=0, atol=0)
    torch.testing.assert_close(actual.cpu(), expected, rtol=0, atol=0)
    actual.square().sum().backward(); expected.square().sum().backward()
    torch.testing.assert_close(values.grad.cpu(), points.grad, rtol=0, atol=0)


@pytest.mark.parametrize("device", DEVICES)
def test_ties_random_start_empty_padding_and_half(device):
    points = torch.tensor([[[0., 0], [1, 0], [-1, 0], [0, 1], [0, -1]]], device=device)
    _, idx = sample_farthest_points(points, K=8)
    assert idx.cpu().tolist() == [[0, 1, 2, 3, 4, -1, -1, -1]]
    identical = torch.zeros(1, 7, 3, device=device)
    _, idx = sample_farthest_points(identical, K=7)
    assert idx.count_nonzero() == 0  # same tie convention as upstream
    torch.manual_seed(70)
    cloud = torch.randn(2, 89, 4, device=device, dtype=torch.float16, requires_grad=True)
    samples, idx = sample_farthest_points(cloud, lengths=[0, 67], K=[3, 9], random_start_point=True)
    assert samples.dtype == torch.float32
    assert (idx[0] == -1).all() and samples[0].count_nonzero() == 0
    assert 0 <= idx[1, 0] < 67 and idx[1].unique().numel() == 9
    selected = cloud[1, :67].float()
    nearest = torch.full((67,), float('inf'), device=device)
    for i in range(1, 9):
        nearest = torch.minimum(nearest, (selected - selected[idx[1, i-1]]).square().sum(-1))
        assert idx[1, i] == nearest.argmax()
    samples.sum().backward()
    assert cloud.grad.isfinite().all()
    for b, p in [(2, 0), (0, 4)]:
        empty = torch.empty(b, p, 3, device=device, requires_grad=True)
        result, idx = sample_farthest_points(empty, K=5)
        assert result.shape == (b, 5, 3) and idx.shape == (b, 5)
        assert result.count_nonzero() == 0
        result.sum().backward()
    assert sample_farthest_points(cloud, K=0)[0].shape == (2, 0, 4)


@pytest.mark.parametrize("device", DEVICES)
def test_masked_gather_neighbor_axis_and_strides(device):
    source = torch.arange(2*9*8).reshape(2, 9, 8).float().to(device).requires_grad_()
    points = source[:, ::2, ::2]
    indices = torch.tensor([[[1, -1], [3, 2]], [[-1, 4], [0, 1]]], device=device)
    got = masked_gather(points, indices)
    expected = reference.masked_gather(points.cpu(), indices.cpu())
    torch.testing.assert_close(got.cpu(), expected)
    got.sum().backward()
    assert source.grad.isfinite().all()
    with pytest.raises(IndexError):
        masked_gather(points, torch.tensor([[5], [0]], device=device))


def test_invalid_sampling_arguments():
    points = torch.zeros(2, 4, 3)
    for kwargs in [dict(K=-1), dict(lengths=[5, 2]), dict(K=[1]), dict(lengths=[-1, 3])]:
        with pytest.raises(ValueError):
            sample_farthest_points(points, **kwargs)
