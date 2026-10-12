"""Small full MapAnything graph: two views, geometry inputs, and Metal autograd."""
import copy
import json
from pathlib import Path

import pytest
import torch

from mapanything.models.mapanything.model import MapAnything


@pytest.fixture(scope="module")
def model():
    previous = torch.get_num_threads()
    torch.set_num_threads(4)
    torch.manual_seed(125)
    config = json.loads((Path(__file__).parent / "fixtures/mapanything_small.json").read_text())
    model = MapAnything(**config, encoder_pretrained=False).eval().requires_grad_(False)
    yield model
    torch.set_num_threads(previous)


def views(geometry=False, count=2):
    torch.manual_seed(67)
    result = []
    for i in range(count):
        view = {"img": torch.randn(1, 3, 56, 56), "data_norm_type": ["dinov2"]}
        if geometry:
            pose = torch.eye(4)[None].clone()
            pose[:, 0, 3] = i * 0.2
            view.update(intrinsics=torch.tensor([[[60., 0, 28], [0, 60, 28], [0, 0, 1]]]),
                        depth_z=torch.full((1, 56, 56), 2.), camera_poses=pose,
                        is_metric_scale=torch.ones(1, dtype=torch.bool))
        result.append(view)
    return result


@pytest.fixture(params=[False, True], ids=["native", "chunked"])
def attention_chunks(request, monkeypatch):
    if request.param:
        from genia.core.ops.attention import scaled_dot_product_attention
        from uniception.models.utils import transformer_blocks

        def bounded(query, key, value, **kwargs):
            if query.device.type == "mps":
                kwargs["max_score_elements"] = 128
            return scaled_dot_product_attention(query, key, value, **kwargs)

        monkeypatch.setattr(transformer_blocks, "scaled_dot_product_attention", bounded)


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="requires Metal")
def test_two_view_forward_and_image_gradients(model, attention_chunks):
    metal = copy.deepcopy(model).to("mps")
    cpu_views = views()
    mps_views = []
    for v in cpu_views:
        v["img"].requires_grad_()
        mps_views.append({**v, "img": v["img"].detach().to("mps").requires_grad_()})
    expected = model(cpu_views)
    actual = metal(mps_views)
    for ref, got in zip(expected, actual):
        assert ref.keys() == got.keys()
        for key in ref:
            torch.testing.assert_close(got[key].cpu(), ref[key], atol=3e-4, rtol=2e-3)
    keys = ["pts3d", "cam_trans", "cam_quats", "conf"]
    sum(o[k].square().mean() for o in expected for k in keys).backward()
    sum(o[k].square().mean() for o in actual for k in keys).backward()
    for ref, got in zip(cpu_views, mps_views):
        assert torch.isfinite(got["img"].grad).all()
        assert got["img"].grad.abs().sum() > 0
        torch.testing.assert_close(got["img"].grad.cpu(), ref["img"].grad, atol=3e-5, rtol=5e-3)


@pytest.mark.parametrize("geometry", [False, True])
@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="requires Metal")
def test_inference_with_optional_geometry_and_minibatching(model, geometry, attention_chunks):
    metal = copy.deepcopy(model).to("mps")
    with torch.no_grad():
        expected = model.infer(views(geometry), use_amp=False, memory_efficient_inference=False)
        actual = metal.infer(views(geometry), use_amp=False, minibatch_size=1)
    for ref, got in zip(expected, actual):
        for key in ref:
            if not torch.is_tensor(ref[key]):
                continue
            if ref[key].dtype == torch.bool:
                assert (got[key].cpu() == ref[key]).float().mean() > .995
            else:
                left, right = got[key].cpu(), ref[key]
                if key in {"pts3d", "pts3d_cam", "depth_z", "depth_along_ray"}:
                    # Edge thresholds can differ at a few boundary pixels.
                    common = (got["mask"].cpu() & ref["mask"]).expand_as(right)
                    left, right = left[common], right[common]
                torch.testing.assert_close(left, right, atol=4e-4, rtol=3e-3,
                                           msg=lambda s: key + ": " + s)
        assert got["camera_poses"].shape == (1, 4, 4)
        assert got["intrinsics"].shape == (1, 3, 3)


@pytest.mark.parametrize("device", ["cpu"] + (["mps"] if torch.backends.mps.is_available() else []))
def test_sixteen_view_attention_chunking(model, monkeypatch, device):
    from genia.core.ops.attention import scaled_dot_product_attention
    from uniception.models.utils import transformer_blocks

    inputs = views(count=16)
    with torch.no_grad():
        expected = model(inputs)
    lengths = []

    def bounded(query, key, value, **kwargs):
        lengths.append(query.shape[-2])
        return scaled_dot_product_attention(
            query, key, value, max_score_elements=4096, **kwargs)

    monkeypatch.setattr(transformer_blocks, "scaled_dot_product_attention", bounded)
    target = copy.deepcopy(model).to(device)
    with torch.no_grad():
        actual = target([{**view, "img": view["img"].to(device)} for view in inputs])
    # A global attention call must actually include the patches from all views.
    assert max(lengths) >= 16 * 16
    assert len(actual) == len(expected) == 16
    for ref, got in zip(expected, actual):
        assert ref.keys() == got.keys()
        for key in ref:
            assert torch.isfinite(got[key]).all()
            torch.testing.assert_close(got[key].cpu(), ref[key], atol=3e-4, rtol=2e-3)


@pytest.mark.parametrize("device", ["cpu"] + (["mps"] if torch.backends.mps.is_available() else []))
def test_rope2d_against_explicit_rotation(device):
    from uniception.models.libs.croco.pos_embed import RoPE2D
    torch.manual_seed(8)
    tokens = torch.randn(2, 3, 11, 32, requires_grad=True)
    positions = torch.randint(0, 20, (2, 11, 2))
    halves = []
    for axis, half in enumerate(tokens.chunk(2, -1)):
        angle = positions[..., axis, None].float() / (100. ** (torch.arange(8) / 8))
        angle = torch.cat([angle, angle], -1)[:, None]
        a, b = half.chunk(2, -1)
        halves.append(half * angle.cos() + torch.cat([-b, a], -1) * angle.sin())
    expected = torch.cat(halves, -1)
    x = tokens.detach().to(device).requires_grad_()
    actual = RoPE2D()(x, positions.to(device))
    torch.testing.assert_close(actual.cpu(), expected, atol=2e-6, rtol=2e-5)
    weight = torch.randn_like(tokens)
    (expected * weight).sum().backward()
    (actual * weight.to(device)).sum().backward()
    torch.testing.assert_close(x.grad.cpu(), tokens.grad, atol=2e-6, rtol=2e-5)
