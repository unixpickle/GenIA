"""Metal voxel coding compared with the unchanged upstream bit-array oracle."""
import importlib.util
import itertools
from pathlib import Path
import sys

import pytest
import torch
import vox2seq

DEVICES = ["cpu"] + (["mps"] if torch.backends.mps.is_available() else [])


def oracle():
    name = "_vox2seq_reference"
    if name not in sys.modules:
        path = Path(__file__).parent / "reference/vox2seq"
        spec = importlib.util.spec_from_file_location(name, path / "__init__.py",
                                                     submodule_search_locations=[str(path)])
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
    return sys.modules[name]


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("mode", ["z_order", "hilbert"])
def test_encode_decode_all_axis_permutations(device, mode):
    torch.manual_seed(4)
    corners = torch.tensor(list(itertools.product([0, 1, 511, 512, 1023], repeat=3)))
    coords = torch.cat([corners, torch.randint(0, 1024, (4096, 3))]).int()
    for permutation in itertools.permutations(range(3)):
        permutation = list(permutation)
        expected = oracle().encode(coords, mode=mode, permute=permutation)
        got = vox2seq.encode(coords.to(device), mode=mode, permute=permutation)
        assert got.dtype == torch.int32
        torch.testing.assert_close(got.cpu(), expected, rtol=0, atol=0)
        decoded = vox2seq.decode(got, mode=mode, permute=permutation)
        torch.testing.assert_close(decoded.cpu(), coords, rtol=0, atol=0)
    codes = torch.randint(0, 1 << 30, (1024,), dtype=torch.int32)
    torch.testing.assert_close(vox2seq.decode(codes.to(device), mode=mode).cpu(),
                               oracle().decode(codes, mode=mode).int(), rtol=0, atol=0)
    assert vox2seq.encode(coords[:0].to(device), mode=mode).shape == (0,)
    assert vox2seq.decode(codes[:0].to(device), mode=mode).shape == (0, 3)
    torch.testing.assert_close(vox2seq.encode(coords[::3].to(device), mode=mode).cpu(),
                               oracle().encode(coords[::3], mode=mode), rtol=0, atol=0)


def test_input_contract():
    with pytest.raises(ValueError):
        vox2seq.encode(torch.tensor([[0, -1, 2]]))
    with pytest.raises(ValueError):
        vox2seq.encode(torch.tensor([[0, 1024, 2]]))
    with pytest.raises(ValueError):
        vox2seq.decode(torch.tensor([1 << 30]))
    with pytest.raises(ValueError):
        vox2seq.encode(torch.zeros(2, 3, dtype=torch.int32), permute=[0, 0, 1])
    with pytest.raises(TypeError):
        vox2seq.encode(torch.zeros(2, 3))


@pytest.mark.parametrize("mode", range(4))
@pytest.mark.parametrize("device", DEVICES)
def test_sam_serialized_attention_values_and_gradients(mode, device):
    from sam3d_objects.model.backbone.tdfy_dit.modules.sparse import SparseTensor
    from sam3d_objects.model.backbone.tdfy_dit.modules.sparse.attention.serialized_attn import (
        SerializeModes, calc_serialization, sparse_serialized_scaled_dot_product_self_attention,
    )
    torch.manual_seed(39)
    batch = torch.cat([torch.zeros(27), torch.ones(3)]).int()
    xyz = torch.randperm(1000)[:30]
    coords = torch.stack([batch, xyz // 100, xyz // 10 % 10, xyz % 10], -1).int()
    feats = torch.randn(30, 3, 2, 8, requires_grad=True)
    ref_sparse = SparseTensor(feats, coords)
    args = (8, SerializeModes[mode], 3, (2, 1, 0))
    forward, backward, lengths, _ = calc_serialization(ref_sparse, *args)
    chunks = feats[forward].split(lengths)
    outputs = []
    for chunk in chunks:
        q, k, v = chunk.permute(1, 2, 0, 3).unbind(0)
        outputs.append(((q @ k.transpose(-1, -2) / 8**.5).softmax(-1) @ v).transpose(0, 1))
    expected = torch.cat(outputs)[backward]
    values = feats.detach().to(device).requires_grad_()
    sparse = SparseTensor(values, coords.to(device))
    actual = sparse_serialized_scaled_dot_product_self_attention(sparse, *args).feats
    torch.testing.assert_close(actual.cpu(), expected, atol=2e-6, rtol=2e-5)
    expected.square().sum().backward()
    actual.square().sum().backward()
    torch.testing.assert_close(values.grad.cpu(), feats.grad, atol=5e-6, rtol=1e-4)
    cached = sparse_serialized_scaled_dot_product_self_attention(sparse, *args).feats
    torch.testing.assert_close(actual, cached)


@pytest.mark.parametrize("device", DEVICES)
def test_serialized_attention_empty_batches(device):
    from sam3d_objects.model.backbone.tdfy_dit.modules.sparse import SparseTensor
    from sam3d_objects.model.backbone.tdfy_dit.modules.sparse.attention.serialized_attn import (
        calc_serialization, sparse_serialized_scaled_dot_product_self_attention as attention,
    )
    coords = torch.tensor([[1, 0, 0, 0], [1, 0, 0, 1]], dtype=torch.int32, device=device)
    values = torch.randn(2, 3, 2, 8, device=device, requires_grad=True)
    sparse = SparseTensor(values, coords, shape=torch.Size([3, 3, 2, 8]))
    out = attention(sparse, 8)
    q, k, v = values.permute(1, 2, 0, 3).unbind(0)
    expected = torch.nn.functional.scaled_dot_product_attention(q, k, v).transpose(0, 1)
    torch.testing.assert_close(out.feats, expected)
    empty_values = values[:0]
    empty = SparseTensor(empty_values, coords[:0], shape=torch.Size([3, 3, 2, 8]))
    forward, backward, lengths, batches = calc_serialization(empty, 8)
    assert forward.numel() == backward.numel() == len(lengths) == len(batches) == 0
    result = attention(empty, 8)
    assert result.feats.shape == (0, 2, 8)
    result.feats.sum().backward()
    torch.testing.assert_close(values.grad, torch.zeros_like(values))
    with pytest.raises(ValueError):
        attention(empty, 0)
