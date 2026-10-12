"""Full local RMBG graph, including the layouts used by image preprocessing."""
import copy

import numpy as np
import pytest
import torch

from genia.core.models.rmbg import BriaRMBG
from actionmesh.preprocessing.background_removal import BackgroundRemover


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="requires Metal")
@pytest.mark.parametrize("channels_last", [False, True])
def test_rmbg_side_outputs_features_and_input_gradient(channels_last):
    torch.manual_seed(413)
    model = BriaRMBG().eval().requires_grad_(False)
    metal = copy.deepcopy(model).to("mps")
    x = torch.randn(1, 3, 64, 80)
    if channels_last:
        x = x.contiguous(memory_format=torch.channels_last)
    x.requires_grad_()
    mx = x.detach().to("mps").requires_grad_()
    expected = model(x)
    actual = metal(mx)
    for es, gs in zip(expected, actual):
        for e, g in zip(es, gs):
            torch.testing.assert_close(g.cpu(), e, atol=3e-6, rtol=2e-4)
    weight = torch.randn_like(expected[0][0])
    (expected[0][0] * weight).sum().backward()
    (actual[0][0] * weight.to("mps")).sum().backward()
    torch.testing.assert_close(mx.grad.cpu(), x.grad, atol=3e-6, rtol=3e-3)


def test_local_checkpoint_background_preprocess_and_mask(tmp_path):
    torch.manual_seed(532)
    model = BriaRMBG().eval()
    model.save_pretrained(tmp_path)
    remover = BackgroundRemover(str(tmp_path), model_input_size=(64, 80)).eval()
    assert type(remover.model) is BriaRMBG
    im = np.random.default_rng(20).integers(0, 256, (39, 57, 3), dtype=np.uint8)
    preprocessed = remover._preprocess_image(im)
    with torch.no_grad():
        expected = model(preprocessed)[0][0]
        torch.testing.assert_close(remover.model(preprocessed)[0][0], expected)
    rgba = remover(im, refine=False)
    assert rgba.shape == (39, 57, 4) and rgba.dtype == np.uint8
    np.testing.assert_array_equal(rgba[:, :, :3], im)
    assert rgba[:, :, 3].min() == 0 and rgba[:, :, 3].max() == 255
