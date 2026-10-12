"""Metal fine-tuning keeps FP32 decoder arithmetic across training and reload."""
import copy
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from omegaconf import OmegaConf

import genia.core
from genia.core.config import FinetuningConfig
from genia.core.finetuning import (
    _ft_decoder_autocast_bf16, _prepare_finetuning_decoder,
    _set_decoder_checkpoint, setup_shared_lora_decoder,
)
from genia.core.utils.lora import LoRAAdapter
from genia.core.utils.slat_decode import decode_tokens, decode_with_lora
from sam3d_objects.model.backbone.tdfy_dit.models import SLatGaussianDecoder
from sam3d_objects.model.backbone.tdfy_dit.modules.sparse import SparseTensor


def model():
    config = {'num_gaussians': 2, 'voxel_size': 2, '3d_filter_kernel_size': .001,
              'scaling_bias': .01, 'opacity_bias': .1, 'scaling_activation': 'exp',
              'perturb_offset': True,
              'lr': {k: 1. for k in ['_xyz', '_features_dc', '_scaling', '_rotation', '_opacity']}}
    decoder = SLatGaussianDecoder(8, 24, 4, 2, num_heads=2, attn_mode='swin',
                                 pe_mode='rope', window_size=2, use_fp16=True,
                                 representation_config=config)
    torch.nn.init.normal_(decoder.out_layer.weight, std=.04)
    return decoder.eval().requires_grad_(False)


@pytest.mark.parametrize('device, default', [('cpu', True), ('cuda', True), ('mps', False)])
def test_precision_policy_and_schema(device, default):
    assert FinetuningConfig().decoder_autocast_bf16 is None
    config = OmegaConf.load(Path(genia.core.__file__).parent / 'configs/finetuning/canonical.yaml')
    assert _ft_decoder_autocast_bf16(config, device) is default
    assert _ft_decoder_autocast_bf16(SimpleNamespace(), device) is default
    for enabled in [False, True]:
        config.decoder_autocast_bf16 = enabled
        assert _ft_decoder_autocast_bf16(config, device) is enabled


def test_promotion_invalidates_half_precision_adapter_template():
    torch.manual_seed(46)
    base = model()
    first = setup_shared_lora_decoder(base, 2, .5, use_dora=True)
    assert first.dtype == torch.float16
    # Policy can be exercised without allocating a GPU tensor.
    assert not _prepare_finetuning_decoder(base, SimpleNamespace(), 'mps')
    second = setup_shared_lora_decoder(base, 2, .5, use_dora=True)
    assert second is not first
    assert second.dtype == torch.float32
    assert all(p.dtype == torch.float32 for p in second.parameters())
    assert first.dtype == torch.float16
    assert setup_shared_lora_decoder(base, 2, .5, use_dora=True) is second


@pytest.mark.parametrize('device', ['cpu', 'mps'])
def test_explicit_mixed_keeps_checkpoint_precision(device):
    base = model()
    before = [p.dtype for p in base.parameters()]
    assert _prepare_finetuning_decoder(base, SimpleNamespace(decoder_autocast_bf16=True), device)
    assert base.dtype == torch.float16 and [p.dtype for p in base.parameters()] == before


def inputs(device):
    coords = torch.tensor([[0, 0, 0, 0], [0, 1, 0, 0], [0, 5, 5, 5],
                           [0, 4, 4, 4], [0, 3, 2, 1]], dtype=torch.int32, device=device)
    return torch.randn(5, 4, device=device), coords


def attributes(gs):
    return [getattr(gs, name) for name in
            ['get_xyz', 'get_features', 'get_scaling', 'get_rotation', 'get_opacity']]


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason='requires Metal')
def test_default_promoted_decoder_values_and_gradients_match_cpu():
    torch.manual_seed(48)
    base = model().to('mps')
    autocast = _prepare_finetuning_decoder(base, SimpleNamespace(), 'mps')
    actual_model = setup_shared_lora_decoder(base, 2, .5, use_dora=True)
    adapter = LoRAAdapter(actual_model, 2, .5, use_dora=True)
    adapter.activate()
    with torch.no_grad():
        for name, p in actual_model.named_parameters():
            if name.endswith('lora_B'):
                p.normal_(std=.015)
    _set_decoder_checkpoint(actual_model, True)
    reference_model = copy.deepcopy(actual_model).cpu()
    features, coords = inputs('cpu')
    features.requires_grad_()
    mfeatures = features.detach().to('mps').requires_grad_()
    expected = decode_tokens(reference_model, features, coords)
    actual = decode_tokens(actual_model, mfeatures, coords.to('mps'), autocast_bf16=autocast)
    for a, e in zip(attributes(actual), attributes(expected)):
        torch.testing.assert_close(a.cpu(), e, atol=2e-6, rtol=2e-5)
    def objective(gs):
        # A weighted probe also exercises orientation: the unweighted squared
        # norm of a normalized quaternion is constant and has only roundoff grad.
        return sum((a * torch.linspace(-.7, 1.1, a.numel(), device=a.device)
                    .reshape(a.shape)).sum() for a in attributes(gs))
    objective(actual).backward()
    objective(expected).backward()
    torch.testing.assert_close(mfeatures.grad.cpu(), features.grad, atol=2e-6, rtol=2e-4)
    refs = dict(reference_model.named_parameters())
    for name, parameter in actual_model.named_parameters():
        if parameter.requires_grad:
            assert parameter.grad is not None
            torch.testing.assert_close(parameter.grad.cpu(), refs[name].grad, atol=3e-6, rtol=3e-4)


@pytest.mark.parametrize('device', ['cpu'] + (['mps'] if torch.backends.mps.is_available() else []))
def test_saved_full_precision_adapter_reloads_with_half_precision_base(device):
    torch.manual_seed(49)
    fresh = model().to(device)
    trained = copy.deepcopy(fresh).float()
    trained.dtype = torch.float32
    wrapped = setup_shared_lora_decoder(trained, 2, .5, use_dora=True)
    adapter = LoRAAdapter(wrapped, 2, .5, use_dora=True)
    adapter.activate()
    with torch.no_grad():
        for name, p in wrapped.named_parameters():
            if name.endswith('lora_B'):
                p.normal_(std=.02)
    features, coords = inputs(device)
    info = {'lora_rank': 2, 'lora_alpha': .5, 'use_dora': True,
            'decoder_full_precision': True, 'state_dict': adapter.state_dict()}
    with torch.no_grad():
        expected = decode_tokens(wrapped, features, coords)
        actual = decode_with_lora(SimpleNamespace(models={'slat_decoder_gs': fresh}),
                                  SparseTensor(features, coords), info)
    for a, e in zip(attributes(actual), attributes(expected)):
        torch.testing.assert_close(a, e, atol=2e-6, rtol=3e-5)
    assert fresh.dtype == torch.float16  # loading must not mutate the shared base
