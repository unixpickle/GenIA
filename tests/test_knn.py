import pytest
import torch

from genia.core.ops import knn_gather, knn_points

DEVICES = ["cpu"] + (["mps"] if torch.backends.mps.is_available() else [])


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("norm", [1, 2])
@pytest.mark.parametrize("k", [1, 4, 32])
def test_knn_reference_and_gradients(device, norm, k):
    torch.manual_seed(71)
    a = torch.randn(2, 7, 3, requires_grad=True)
    b = torch.randn(2, 11, 3, requires_grad=True)
    x = a.detach().to(device).requires_grad_()
    y = b.detach().to(device).requires_grad_()
    result = knn_points(x, y, K=k, norm=norm, return_nn=True)
    diff = a[:, :, None] - b[:, None]
    all_d = (diff.square() if norm == 2 else diff.abs()).sum(-1)
    expected, indices = all_d.sort(stable=True)
    n = min(k, b.shape[1])
    torch.testing.assert_close(result.dists[:, :, :n].cpu(), expected[:, :, :n])
    torch.testing.assert_close(result.idx[:, :, :n].cpu(), indices[:, :, :n])
    assert (result.dists[:, :, n:] == 0).all()
    torch.testing.assert_close(result.knn[:, :, :n].cpu(),
                               b[torch.arange(2)[:, None, None], indices[:, :, :n]])
    result.dists.sum().backward()
    expected[:, :, :n].sum().backward()
    torch.testing.assert_close(x.grad.cpu(), a.grad)
    torch.testing.assert_close(y.grad.cpu(), b.grad)


@pytest.mark.parametrize("device", DEVICES)
def test_lengths_ties_noncontiguous(device):
    a = torch.tensor([[[0., 2., 8.], [0., 0., 0.]],
                      [[3., 4., 5.], [0., 0., 0.]]], device=device).transpose(1, 2)
    b = torch.tensor([[[1., 0.], [-1., 0.], [4., 0.]],
                      [[0., 0.], [3., 0.], [5., 0.]]], device=device)
    got = knn_points(a, b, torch.tensor([2, 0]), torch.tensor([2, 3]), K=3, return_nn=True)
    torch.testing.assert_close(got.idx[0].cpu(), torch.tensor([[0, 1, 0], [0, 1, 0], [0, 0, 0]]))
    torch.testing.assert_close(got.dists[0].cpu(), torch.tensor([[1., 1., 0.], [1., 9., 0.], [0., 0., 0.]]))
    assert (got.dists[1] == 0).all() and (got.knn[1] == 0).all()


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("p,q", [(0, 3), (3, 0), (0, 0)])
def test_empty(device, p, q):
    a = torch.zeros(2, p, 3, device=device, requires_grad=True)
    b = torch.zeros(2, q, 3, device=device, requires_grad=True)
    got = knn_points(a, b, K=4, return_nn=True)
    assert got.dists.shape == (2, p, 4)
    assert (got.dists == 0).all() and (got.knn == 0).all()
    got.dists.sum().backward()
    assert a.grad is not None and b.grad is not None


def test_gradcheck():
    torch.manual_seed(12)
    a = torch.randn(1, 3, 2, dtype=torch.double, requires_grad=True)
    b = torch.randn(1, 5, 2, dtype=torch.double, requires_grad=True)
    assert torch.autograd.gradcheck(lambda a, b: knn_points(a, b, K=2).dists, (a, b))


@pytest.mark.parametrize("device", DEVICES)
def test_gather(device):
    x = torch.arange(12., device=device).reshape(2, 3, 2).requires_grad_()
    idx = torch.tensor([[[2, 0]], [[1, 2]]], device=device)
    got = knn_gather(x, idx, torch.tensor([3, 1]))
    torch.testing.assert_close(got.cpu(), torch.tensor([[[[4., 5.], [0., 1.]]], [[[8., 9.], [0., 0.]]]]))
    got.sum().backward()
    torch.testing.assert_close(x.grad.cpu(), torch.tensor([[[1., 1.], [0., 0.], [1., 1.]], [[0., 0.], [1., 1.], [0., 0.]]]))


@pytest.mark.parametrize("device", DEVICES)
def test_deformation_translation(device):
    from genia.core.utils.deformation import _warp_at_high_res
    torch.manual_seed(2)
    p = torch.randn(17, 3, device=device, requires_grad=True)
    v = torch.randn(13, 3, device=device)
    translation = torch.tensor([1., -2., 3.], device=device)
    rotations = torch.eye(3, device=device).repeat(13, 1, 1)
    warped, _ = _warp_at_high_res(p, v, v + translation, rotations, K=4, chunk_size=5)
    torch.testing.assert_close(warped, p + translation)
    warped.sum().backward()
    torch.testing.assert_close(p.grad, torch.ones_like(p), atol=2e-5, rtol=2e-5)


def test_invalid_arguments():
    a = torch.zeros(1, 3, 2)
    for kwargs in [{"K": 0}, {"norm": 3}, {"lengths1": torch.tensor([4])}]:
        with pytest.raises(ValueError):
            knn_points(a, a, **kwargs)
