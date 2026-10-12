from itertools import product

import pytest
import torch
import torch.nn.functional as F

from genia.core.ops.sparse import neighbor_indices, submanifold_conv3d

DEVICES = ["cpu"] + (["mps"] if torch.backends.mps.is_available() else [])


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("dilation", [1, 2])
def test_hash_neighbors(device, dilation):
    torch.manual_seed(4)
    coords = torch.tensor(list(product(range(2), range(-2, 3), range(-1, 4), range(3))), dtype=torch.int32)
    coords = coords[torch.randperm(len(coords))[:97]]
    lookup = {tuple(c): i for i, c in enumerate(coords.tolist())}
    offsets = list(product(range(-1, 2), repeat=3))
    expected = torch.tensor([[lookup.get((b, x + dx * dilation, y + dy * dilation, z + dz * dilation), -1)
                              for b, x, y, z in coords.tolist()] for dx, dy, dz in offsets])
    got = neighbor_indices(coords.to(device), dilation=dilation)
    torch.testing.assert_close(got.cpu(), expected)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("kernel,dilation", [(1, 1), (3, 1), (3, 2)])
def test_conv_dense_reference_and_gradients(device, kernel, dilation):
    torch.manual_seed(91)
    coords = torch.tensor(list(product(range(2), range(4), range(3), range(5))), dtype=torch.int32)
    coords = coords[torch.randperm(len(coords))[:63]]
    f = torch.randn(len(coords), 3, requires_grad=True)
    w = torch.randn(4, kernel, kernel, kernel, 3, requires_grad=True)
    bias = torch.randn(4, requires_grad=True)
    dense = torch.zeros(2, 4, 3, 5, 3)
    b, x, y, z = coords.long().T
    dense[b, x, y, z] = f
    ref = F.conv3d(dense.permute(0, 4, 1, 2, 3), w.permute(0, 4, 1, 2, 3), bias,
                   padding=(kernel // 2) * dilation, dilation=dilation).permute(0, 2, 3, 4, 1)[b, x, y, z]
    inputs = [v.detach().to(device).requires_grad_() for v in (f, w, bias)]
    neighbors = neighbor_indices(coords.to(device), kernel, dilation)
    got = submanifold_conv3d(inputs[0], inputs[1], neighbors, inputs[2])
    torch.testing.assert_close(got.cpu(), ref, atol=1e-5, rtol=1e-5)
    cotangent = torch.randn_like(ref)
    (ref * cotangent).sum().backward()
    (got * cotangent.to(device)).sum().backward()
    for actual, expected in zip(inputs, (f, w, bias)):
        torch.testing.assert_close(actual.grad.cpu(), expected.grad, atol=2e-5, rtol=2e-5)


@pytest.mark.parametrize("device", DEVICES)
def test_empty(device):
    coords = torch.empty(0, 4, dtype=torch.int32, device=device)
    f = torch.empty(0, 2, device=device, requires_grad=True)
    w = torch.randn(3, 3, 3, 3, 2, device=device, requires_grad=True)
    got = submanifold_conv3d(f, w, neighbor_indices(coords))
    assert got.shape == (0, 3)
    got.sum().backward()
    assert w.grad is not None and f.grad is not None


def test_gradcheck():
    coords = torch.tensor([[0, 0, 0, 0], [0, 0, 1, 0]], dtype=torch.int32)
    neighbors = neighbor_indices(coords)
    f = torch.randn(2, 2, dtype=torch.double, requires_grad=True)
    w = torch.randn(2, 3, 3, 3, 2, dtype=torch.double, requires_grad=True)
    assert torch.autograd.gradcheck(lambda f, w: submanifold_conv3d(f, w, neighbors), (f, w))


@pytest.mark.skipif(not torch.backends.mps.is_available(),reason='requires Metal')
@pytest.mark.parametrize('dtype',[torch.float16,torch.bfloat16])
def test_low_precision_sparse_conv_gradients(dtype):
    torch.manual_seed(190)
    coords=torch.tensor([[0,0,0,0],[0,1,0,0],[0,0,1,1],[0,1,1,1]],dtype=torch.int32)
    features=torch.randn(4,3,dtype=dtype,requires_grad=True)
    weights=(torch.randn(2,3,3,3,3,dtype=dtype)*.1).requires_grad_()
    mf=features.detach().to('mps').requires_grad_()
    mw=weights.detach().to('mps').requires_grad_()
    reference=submanifold_conv3d(features,weights,neighbor_indices(coords))
    actual=submanifold_conv3d(mf,mw,neighbor_indices(coords.to('mps')))
    torch.testing.assert_close(actual.cpu(),reference,atol=5e-3,rtol=2e-2)
    reference.float().square().sum().backward();actual.float().square().sum().backward()
    torch.testing.assert_close(mf.grad.cpu(),features.grad,atol=5e-3,rtol=3e-2)
    torch.testing.assert_close(mw.grad.cpu(),weights.grad,atol=5e-3,rtol=3e-2)
