"""Local HF-layout DINO weights against the pinned Transformers implementation."""
import copy

import pytest
import torch
from transformers import Dinov2Config, Dinov2Model as ReferenceDINO
from genia.core.models.hf_dinov2 import Dinov2Model
from genia.core.models.hf_dinov2 import Dinov2Embeddings
from transformers.models.dinov2.modeling_dinov2 import Dinov2Embeddings as ReferenceEmbeddings

DEVICES = ['cpu'] + (['mps'] if torch.backends.mps.is_available() else [])


@pytest.mark.parametrize('device', DEVICES)
@pytest.mark.parametrize('swiglu', [False, True])
def test_checkpoint_keys_rectangular_masked_images_and_gradients(tmp_path, device, swiglu):
    torch.manual_seed(141)
    config = Dinov2Config(hidden_size=32, num_attention_heads=4, num_hidden_layers=2,
                          image_size=28, patch_size=7, use_swiglu_ffn=swiglu)
    config._attn_implementation = 'eager'
    reference = ReferenceDINO(config).eval().requires_grad_(False)
    reference.save_pretrained(tmp_path)
    model = Dinov2Model.from_pretrained(tmp_path, local_files_only=True).to(device).eval()
    model.requires_grad_(False)
    assert set(model.state_dict()) == set(reference.state_dict())
    x = torch.randn(2, 3, 28, 42, requires_grad=True)
    mx = x.detach().to(device).requires_grad_()
    mask = torch.rand(2, 24) > .8
    a = reference(x, bool_masked_pos=mask, output_hidden_states=True)
    b = model(mx, bool_masked_pos=mask.to(device), output_hidden_states=True)
    for ref, got in zip((a.last_hidden_state, a.pooler_output, *a.hidden_states),
                        (b.last_hidden_state, b.pooler_output, *b.hidden_states)):
        torch.testing.assert_close(got.cpu(), ref, atol=4e-6, rtol=2e-4)
    weight = torch.randn_like(a.last_hidden_state)
    (a.last_hidden_state * weight).sum().backward()
    (b.last_hidden_state * weight.to(device)).sum().backward()
    torch.testing.assert_close(mx.grad.cpu(), x.grad, atol=1e-5, rtol=3e-4)


@pytest.mark.parametrize('device', DEVICES)
@pytest.mark.parametrize('dtype', [torch.float16, torch.bfloat16])
def test_low_precision_position_interpolation_and_gradients(device, dtype):
    # Real TripoSG DINO shape: dropping CLS leaves a nonzero storage offset;
    # permuting the grid makes it noncontiguous before the FP32 resize cast.
    torch.manual_seed(312)
    config = Dinov2Config(hidden_size=1024, image_size=518, patch_size=14)
    reference = ReferenceEmbeddings(config).to(dtype=dtype)
    model = Dinov2Embeddings(config).to(dtype=dtype)
    model.load_state_dict(reference.state_dict())
    model.to(device)
    embeddings = torch.empty(1, 16 * 20 + 1, 1024, dtype=dtype)
    expected = reference.interpolate_pos_encoding(embeddings, 224, 280)
    actual = model.interpolate_pos_encoding(embeddings.to(device), 224, 280)
    tolerance = 2e-3 if dtype == torch.float16 else 2e-2
    torch.testing.assert_close(actual.cpu(), expected, atol=tolerance, rtol=tolerance)
    weight = torch.randn(expected.shape).to(dtype)
    (expected * weight).sum().backward()
    (actual * weight.to(device)).sum().backward()
    torch.testing.assert_close(model.position_embeddings.grad.cpu(),
        reference.position_embeddings.grad, atol=tolerance, rtol=tolerance)
