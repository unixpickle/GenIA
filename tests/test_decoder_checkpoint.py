"""Memory checkpointing preserves the actual Gaussian decoder and adapter gradients."""
import copy
from types import SimpleNamespace

import pytest
import torch

from genia.core.finetuning import _ft_decoder_checkpoint_min_voxels, _set_decoder_checkpoint
from genia.core.utils.lora import apply_lora_to_decoder
from genia.core.utils.slat_decode import decode_tokens
from sam3d_objects.model.backbone.tdfy_dit.models import SLatGaussianDecoder


@pytest.mark.parametrize('device, expected', [('cpu', 30000), ('cuda', 30000), ('mps', 0)])
def test_checkpoint_threshold_default_and_override(device, expected):
    assert _ft_decoder_checkpoint_min_voxels(SimpleNamespace(decoder_checkpoint_min_voxels=None), device) == expected
    for threshold in [0, 12000, 45000]:
        cfg = SimpleNamespace(decoder_checkpoint_min_voxels=threshold)
        assert _ft_decoder_checkpoint_min_voxels(cfg, device) == threshold


@pytest.mark.parametrize('device', ['cpu'] + (['mps'] if torch.backends.mps.is_available() else []))
@pytest.mark.parametrize('mixed', [False, True])
def test_checkpointed_decoder_adapter_values_and_gradients(device, mixed):
    torch.manual_seed(138)
    config = {'num_gaussians': 2, 'voxel_size': 2, '3d_filter_kernel_size': .001,
              'scaling_bias': .01, 'opacity_bias': .1, 'scaling_activation': 'exp',
              'perturb_offset': True,
              'lr': {k: 1. for k in ['_xyz', '_features_dc', '_scaling', '_rotation', '_opacity']}}
    model = SLatGaussianDecoder(8, 24, 4, 2, num_heads=2, attn_mode='swin',
                               pe_mode='rope', window_size=2, representation_config=config)
    torch.nn.init.normal_(model.out_layer.weight, std=.03)
    model.requires_grad_(False)
    apply_lora_to_decoder(model, rank=2, alpha=.5, use_dora=True)
    model = model.to(device)
    checkpointed = copy.deepcopy(model)
    _set_decoder_checkpoint(checkpointed, True)
    flags = [m for m in checkpointed.modules() if hasattr(m, 'use_checkpoint')]
    assert flags and all(m.use_checkpoint for m in flags)
    coords = torch.tensor([[0, 0, 0, 0], [0, 1, 0, 0], [0, 5, 5, 5],
                           [0, 4, 4, 4], [0, 3, 2, 1]], dtype=torch.int32, device=device)
    features = torch.randn(5, 4, device=device, requires_grad=True)
    check_features = features.detach().clone().requires_grad_()
    reference = decode_tokens(model, features, coords, autocast_bf16=mixed)
    actual = decode_tokens(checkpointed, check_features, coords, autocast_bf16=mixed)
    losses = []
    for gs in [reference, actual]:
        losses.append(sum(getattr(gs, name).square().sum() for name in
                          ['get_xyz', 'get_features', 'get_scaling', 'get_rotation', 'get_opacity']))
    torch.testing.assert_close(losses[0], losses[1], rtol=0, atol=0)
    losses[0].backward()
    losses[1].backward()
    torch.testing.assert_close(check_features.grad, features.grad, rtol=1e-5, atol=1e-6)
    checked = dict(checkpointed.named_parameters())
    trainable = 0
    for name, parameter in model.named_parameters():
        if parameter.requires_grad:
            assert parameter.grad is not None and checked[name].grad is not None
            assert torch.isfinite(checked[name].grad).all()
            torch.testing.assert_close(checked[name].grad, parameter.grad, rtol=1e-5, atol=1e-6)
            trainable += 1
    assert trainable
    _set_decoder_checkpoint(checkpointed, False)
    assert not any(m.use_checkpoint for m in flags)
