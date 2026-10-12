import os

os.environ["LIDRA_SKIP_INIT"] = "true"
os.environ["SPARSE_BACKEND"] = "metal"
os.environ["SPARSE_ATTN_BACKEND"] = "sdpa"
os.environ["ATTN_BACKEND"] = "sdpa"

import pytest
import torch

from sam3d_objects.model.backbone.tdfy_dit.modules import sparse as sp

DEVICES = ["cpu"] + (["mps"] if torch.backends.mps.is_available() else [])


@pytest.mark.parametrize("device", DEVICES)
def test_model_conv_checkpoint_and_cache(device):
    coords = torch.tensor([[0, 0, 0, 0], [0, 1, 0, 0], [1, 0, 1, 0]], dtype=torch.int32, device=device)
    feats = torch.randn(3, 2, device=device, requires_grad=True)
    x = sp.SparseTensor(feats, coords)
    layer = sp.SparseConv3d(2, 4, 3).to(device)
    checkpoint = {"conv.weight": torch.randn(4, 3, 3, 3, 2, device=device),
                  "conv.bias": torch.randn(4, device=device)}
    layer.load_state_dict(checkpoint, strict=True)
    out = layer(x)
    dense = torch.nn.functional.conv3d(x.dense(), checkpoint["conv.weight"].permute(0, 4, 1, 2, 3),
                                      checkpoint["conv.bias"], padding=1).permute(0, 2, 3, 4, 1)
    torch.testing.assert_close(out.feats, dense[tuple(coords.long().T)])
    torch.testing.assert_close(out.coords, x.coords)
    assert out.layout == [slice(0, 2), slice(2, 3)]
    torch.testing.assert_close(layer(x).feats, out.feats)
    out.feats.sum().backward()
    assert torch.isfinite(feats.grad).all()
    assert len(x.get_spatial_cache()) == 1


@pytest.mark.parametrize("device", DEVICES)
def test_storage_and_subdivide(device):
    coords = torch.tensor([[0, 1, 0, 0], [1, 0, 1, 0]], dtype=torch.int32, device=device)
    feats = torch.randn(2, 4, device=device, requires_grad=True)
    x = sp.SparseTensor(feats, coords)
    y = sp.SparseSubdivide()(x)
    assert y.feats.shape == (16, 4) and y.coords.shape == (16, 4)
    assert y.layout == [slice(0, 8), slice(8, 16)]
    torch.testing.assert_close(y.feats, feats.repeat_interleave(8, dim=0))
    assert y.coords.cpu().unique(dim=0).shape == (16, 4)
    assert y._scale == (2, 2, 2)
    y.feats.sum().backward()
    torch.testing.assert_close(feats.grad, torch.full_like(feats, 8))
    assert x.cpu().device.type == "cpu"
    assert x.half().dtype == torch.float16


@pytest.mark.parametrize("device", DEVICES)
def test_sparse_attention(device):
    torch.manual_seed(72)
    coords = torch.tensor([[0, 0, 0, 0], [0, 1, 0, 0], [1, 0, 1, 0]], dtype=torch.int32, device=device)
    feats = torch.randn(3, 3, 2, 4, device=device, requires_grad=True)
    x = sp.SparseTensor(feats, coords)
    out = sp.sparse_scaled_dot_product_attention(x)
    expected = []
    for sl in x.layout:
        q, k, v = feats[sl].unbind(1)
        scores = torch.einsum("qhd,khd->hqk", q, k) / 2
        expected.append(torch.einsum("hqk,khd->qhd", scores.softmax(-1), v))
    torch.testing.assert_close(out.feats, torch.cat(expected), atol=1e-5, rtol=1e-5)
    out.feats.square().sum().backward()
    assert torch.isfinite(feats.grad).all()


@pytest.mark.parametrize("device", DEVICES)
def test_packed_cross_attention(device):
    from sam3d_objects.model.backbone.tdfy_dit.modules.sparse.attention.masked_sdpa import masked_sdpa
    torch.manual_seed(9)
    inputs = [torch.randn(1, n, 2, 4, requires_grad=True) for n in (5, 7, 7)]
    actual_inputs = [x.detach().to(device).requires_grad_() for x in inputs]
    mask = torch.full((5, 7), -torch.inf)
    mask[:2, :4] = 0
    mask[2:, 4:] = 0
    q, k, v = inputs
    ref = torch.nn.functional.scaled_dot_product_attention(q.transpose(1, 2), k.transpose(1, 2),
                                                           v.transpose(1, 2), attn_mask=mask).transpose(1, 2)[0]
    got = masked_sdpa(*actual_inputs, [2, 3], [4, 3])
    torch.testing.assert_close(got.cpu(), ref, atol=1e-5, rtol=1e-5)
    got.square().sum().backward()
    ref.square().sum().backward()
    for a, e in zip(actual_inputs, inputs):
        torch.testing.assert_close(a.grad.cpu(), e.grad, atol=1e-5, rtol=1e-5)
