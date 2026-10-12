"""Exercise the local MoGe/DINO model and calibrated geometry on Metal."""
import copy
import os

import pytest
import torch
import utils3d

from moge.model.v1 import MoGeModel
from moge.utils.geometry_torch import normalized_view_plane_uv, recover_focal_shift


pytestmark = pytest.mark.skipif(not torch.backends.mps.is_available(), reason="requires Metal")


def test_sam_local_dino_register_tokens(monkeypatch):
    os.environ["LIDRA_SKIP_INIT"] = "1"
    from sam3d_objects.model.backbone.dit.embedder.dino import Dino

    def reject_remote_code(*args, **kwargs):
        raise AssertionError("DINO must not fetch or execute torch hub source")

    monkeypatch.setattr(torch.hub, "load", reject_remote_code)
    previous = torch.get_num_threads()
    torch.set_num_threads(4)
    try:
        model = Dino(input_size=56, dino_model="dinov2_vits14_reg",
                     backbone_kwargs={"pretrained": False})
        x = torch.rand(1, 1, 31, 47)
        with torch.no_grad():
            expected = model(x)
            actual = model.to("mps")(x.to("mps"))
        assert actual.shape == (1, 17, 384)
        torch.testing.assert_close(actual.cpu(), expected, atol=2e-5, rtol=2e-4)
    finally:
        torch.set_num_threads(previous)


def test_moge_forward_and_image_gradient():
    torch.manual_seed(51)
    previous = torch.get_num_threads()
    torch.set_num_threads(4)
    try:
        model = MoGeModel(encoder="dinov2_vits14", intermediate_layers=1,
            dim_proj=32, dim_upsample=[32, 32, 32], last_conv_channels=32).eval()
        model.requires_grad_(False)
        metal = copy.deepcopy(model).to("mps")
        x = torch.rand(1, 3, 32, 48, requires_grad=True)
        mx = x.detach().to("mps").requires_grad_()
        ref, got = model(x, num_tokens=16), metal(mx, num_tokens=16)
        for key in ref:
            torch.testing.assert_close(got[key].cpu(), ref[key], atol=2e-5, rtol=2e-4)
        sum(v.square().mean() for v in ref.values()).backward()
        sum(v.square().mean() for v in got.values()).backward()
        torch.testing.assert_close(mx.grad.cpu(), x.grad, atol=2e-6, rtol=3e-3)
        # Public inference includes autocast and camera recovery. Force all pixels
        # valid because random mask logits do not represent a trained model.
        metal.mask_threshold = -float("inf")
        from sam3d_objects.pipeline.depth_models.moge import MoGe
        wrapper = MoGe(metal, device="mps")
        assert wrapper.device.type == "mps" and wrapper.model is metal
        result = metal.infer(mx.detach()[0], num_tokens=16, fov_x=60, apply_mask=False)
        assert result["depth"].shape == (32, 48)
        for value in result.values():
            assert value.device.type == "mps" and torch.isfinite(value).all()
        assert result["points"].dtype == torch.float32
    finally:
        torch.set_num_threads(previous)


@pytest.mark.parametrize("known_focal", [False, True])
def test_moge_focal_recovery_and_unprojection(known_focal):
    h, w = 32, 48
    uv = normalized_view_plane_uv(w, h, device="mps")
    depth = 3 + uv[..., 0].square() + uv[..., 1]
    focal, shift = 1.4, .8
    points = torch.cat([uv * depth[..., None] / focal, (depth - shift)[..., None]], -1)
    recovered_focal, recovered_shift = recover_focal_shift(points,
        focal=torch.tensor(focal, device="mps") if known_focal else None)
    torch.testing.assert_close(recovered_focal.cpu(), torch.tensor(focal), atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(recovered_shift.cpu(), torch.tensor(shift), atol=1e-5, rtol=1e-5)
    diagonal = (h*h + w*w)**.5
    intrinsics = utils3d.pt.intrinsics_from_focal_center(
        torch.tensor(focal * diagonal / (2*w), device="mps"),
        torch.tensor(focal * diagonal / (2*h), device="mps"),
        torch.tensor(.5, device="mps"), torch.tensor(.5, device="mps"))
    got = utils3d.pt.depth_map_to_point_map(depth, intrinsics=intrinsics)
    expected = points + torch.tensor([0., 0., shift], device="mps")
    torch.testing.assert_close(got, expected, atol=1e-5, rtol=1e-5)


def test_pointmap_window_encoder_cpu_mps():
    from sam3d_objects.model.backbone.dit.embedder.pointmap import PointPatchEmbed
    torch.manual_seed(31)
    model = PointPatchEmbed(input_size=8, patch_size=2, embed_dim=32, remap_output='linear').eval()
    metal = copy.deepcopy(model).to('mps')
    x = torch.randn(2, 3, 9, 11)
    x[0, :, 0, 0] = torch.nan
    x[1, :, 3, 2] = torch.inf
    expected, actual = model(x), metal(x.to('mps'))
    torch.testing.assert_close(actual.cpu(), expected, atol=1e-5, rtol=1e-4)
    actual.square().mean().backward()
    expected.square().mean().backward()
    for (name, ref), (_, got) in zip(model.named_parameters(), metal.named_parameters()):
        if ref.grad is not None:
            torch.testing.assert_close(got.grad.cpu(), ref.grad, atol=1e-6, rtol=2e-3, msg=name)
