"""Exercise real local SAM model code at small dimensions, without checkpoints."""
import copy
import os

os.environ["LIDRA_SKIP_INIT"] = "true"
os.environ["SPARSE_BACKEND"] = "metal"
os.environ["ATTN_BACKEND"] = "sdpa"
os.environ["SPARSE_ATTN_BACKEND"] = "sdpa"

import pytest
import torch

from sam3d_objects.model.backbone.tdfy_dit.models import SLatGaussianDecoder
from sam3d_objects.model.backbone.tdfy_dit.modules import sparse as sp


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="requires Metal")
def test_dense_structure_decoder_mps_matches_cpu():
    from sam3d_objects.model.backbone.tdfy_dit.models import SparseStructureDecoder

    torch.manual_seed(63)
    previous = torch.get_num_threads()
    torch.set_num_threads(4)
    try:
        model = SparseStructureDecoder(out_channels=1, latent_channels=8,
            num_res_blocks=1, channels=[32, 16, 8], num_res_blocks_middle=1).eval()
        metal = copy.deepcopy(model).to("mps")
        x = torch.randn(1, 8, 4, 4, 4, requires_grad=True)
        mx = x.detach().to("mps").requires_grad_()
        expected, actual = model(x), metal(mx)
        assert actual.shape == (1, 1, 16, 16, 16)
        torch.testing.assert_close(actual.cpu(), expected, atol=3e-6, rtol=2e-5)
        actual.square().mean().backward()
        expected.square().mean().backward()
        torch.testing.assert_close(mx.grad.cpu(), x.grad, atol=1e-7, rtol=2e-4)
    finally:
        torch.set_num_threads(previous)


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="requires Metal")
@pytest.mark.parametrize("attention", ["full", "swin"])
@pytest.mark.parametrize("positions", ["ape", "rope"])
def test_decoder_mps_matches_cpu(attention, positions):
    torch.manual_seed(35)
    config = {"num_gaussians": 2, "voxel_size": 2, "3d_filter_kernel_size": 0.001,
              "scaling_bias": 0.01, "opacity_bias": 0.1, "scaling_activation": "exp",
              "perturb_offset": True,
              "lr": {k: 1. for k in ["_xyz", "_features_dc", "_scaling", "_rotation", "_opacity"]}}
    model = SLatGaussianDecoder(8, 24, 4, 2, num_heads=2, attn_mode=attention,
                                pe_mode=positions, window_size=2, representation_config=config)
    # The constructor zeroes the output layer; nonzero weights exercise every block.
    torch.nn.init.normal_(model.out_layer.weight, std=0.1)
    metal = copy.deepcopy(model).to("mps")
    coords = torch.tensor([[0, 0, 0, 0], [0, 1, 0, 0], [0, 5, 5, 5],
                           [1, 0, 1, 0], [1, 4, 4, 4]], dtype=torch.int32)
    feats = torch.randn(5, 4, requires_grad=True)
    mf = feats.detach().to("mps").requires_grad_()
    expected = model(sp.SparseTensor(feats, coords))
    actual = metal(sp.SparseTensor(mf, coords.to("mps")))
    losses = [0, 0]
    for g, ref in zip(actual, expected):
        for name in ["get_xyz", "get_features", "get_scaling", "get_rotation", "get_opacity"]:
            a, e = getattr(g, name), getattr(ref, name)
            torch.testing.assert_close(a.cpu(), e, rtol=2e-4, atol=2e-5)
            losses[0] = losses[0] + a.square().sum()
            losses[1] = losses[1] + e.square().sum()
        torch.testing.assert_close(g.get_covariance().cpu(), ref.get_covariance(), rtol=2e-4, atol=2e-5)
    losses[0].backward()
    losses[1].backward()
    torch.testing.assert_close(mf.grad.cpu(), feats.grad, rtol=3e-3, atol=2e-4)


@pytest.mark.parametrize("device", ["cpu"] + (["mps"] if torch.backends.mps.is_available() else []))
def test_local_sam_renderer(device):
    from sam3d_objects.model.backbone.tdfy_dit.renderers.gaussian_render import GaussianRenderer
    from sam3d_objects.model.backbone.tdfy_dit.representations.gaussian import Gaussian
    g = Gaussian(aabb=[-.5, -.5, -.5, 1., 1., 1.], scaling_bias=.2, opacity_bias=.5, device=device)
    g._xyz = torch.tensor([[.5, .5, 2.5]], device=device, requires_grad=True)
    g._features_dc = torch.zeros(1, 1, 3, device=device)
    g._scaling = torch.zeros(1, 3, device=device)
    g._rotation = torch.zeros(1, 4, device=device)
    g._opacity = torch.zeros(1, 1, device=device)
    renderer = GaussianRenderer({"resolution": 8, "near": .1, "far": 10., "bg_color": [0., 0., 0.]})
    K = torch.tensor([[1., 0., .5], [0., 1., .5], [0., 0., 1.]], device=device)
    result = renderer.render(g, torch.eye(4, device=device), K)
    assert result.color.shape == (3, 8, 8)
    torch.testing.assert_close(result.color, result.alpha.expand_as(result.color) * .5, atol=1e-6, rtol=1e-6)
    result.color.sum().backward()
    assert torch.isfinite(g._xyz.grad).all()


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="requires Metal")
def test_genia_decoder_bfloat16_autocast_gradients():
    from genia.core.utils.slat_decode import decode_tokens
    torch.manual_seed(64)
    config = {"num_gaussians": 2, "voxel_size": 2, "3d_filter_kernel_size": 0.001,
              "scaling_bias": 0.01, "opacity_bias": 0.1, "scaling_activation": "exp",
              "perturb_offset": True,
              "lr": {k: 1. for k in ["_xyz", "_features_dc", "_scaling", "_rotation", "_opacity"]}}
    model=SLatGaussianDecoder(8,24,4,1,num_heads=2,representation_config=config).to('mps')
    torch.nn.init.normal_(model.out_layer.weight,std=.02)
    coords=torch.tensor([[0,1,1,1],[0,2,2,2],[0,3,3,3]],device='mps',dtype=torch.int32)
    feats=torch.randn(3,4,device='mps',requires_grad=True)
    fp=decode_tokens(model,feats,coords)
    mixed=decode_tokens(model,feats,coords,autocast_bf16=True)
    for name in ['get_xyz','get_features','get_scaling','get_rotation','get_opacity']:
        actual,expected=getattr(mixed,name),getattr(fp,name)
        assert actual.dtype==torch.float32
        torch.testing.assert_close(actual,expected,atol=2e-3,rtol=2e-2)
    mixed.get_xyz.square().sum().backward()
    assert torch.isfinite(feats.grad).all() and feats.grad.abs().sum()>0


@pytest.mark.parametrize('device', ['cpu'] + (['mps'] if torch.backends.mps.is_available() else []))
def test_shape_inversion_optimization(device):
    from genia.core.shape_inversion import invert_decoder
    decoder=torch.nn.Conv3d(8,1,1,bias=False).to(device)
    torch.nn.init.constant_(decoder.weight,.2)
    decoder.requires_grad_(False)
    target=torch.ones(1,1,16,16,16,device=device)
    latent,history=invert_decoder(decoder,target,num_steps=5,lr=.2,prior_weight=0,
        seed=12,show_progress=False,autocast_dtype=torch.bfloat16)
    assert latent.device.type==device and torch.isfinite(latent).all()
    assert history[-1][0]<history[0][0]*.6
