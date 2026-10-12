import copy

import pytest
import torch
from torch.nn import functional as F

from genia.core.ops.convolution import Conv3d


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason='requires Metal')
@pytest.mark.parametrize('dtype', [torch.float16, torch.bfloat16])
@pytest.mark.parametrize('bias', [False, True])
def test_mixed_conv3d_values_and_parameter_input_gradients(dtype, bias):
    torch.manual_seed(342)
    model=Conv3d(4,6,3,padding=1,groups=2,bias=bias)
    metal=copy.deepcopy(model).to('mps')
    x=torch.randn(2,4,5,6,7,requires_grad=True)
    mx=x.detach().to('mps').requires_grad_()
    expected=F.conv3d(x.to(dtype),model.weight.to(dtype),
                      model.bias.to(dtype) if bias else None,padding=1,groups=2)
    with torch.autocast('mps',dtype=dtype):actual=metal(mx)
    assert actual.dtype==dtype
    torch.testing.assert_close(actual.cpu(),expected,atol=.008,rtol=.025)
    expected.float().square().mean().backward();actual.float().square().mean().backward()
    torch.testing.assert_close(mx.grad.cpu(),x.grad,atol=2e-5,rtol=.04)
    for a,b in zip(metal.parameters(),model.parameters()):
        torch.testing.assert_close(a.grad.cpu(),b.grad,atol=4e-4,rtol=.04)


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason='requires Metal')
@pytest.mark.parametrize('dtype', [torch.float16, torch.bfloat16])
def test_sam_structure_decoder_mixed_precision(dtype):
    from sam3d_objects.model.backbone.tdfy_dit.models import SparseStructureDecoder
    torch.manual_seed(64)
    model=SparseStructureDecoder(out_channels=1,latent_channels=8,num_res_blocks=1,
        channels=[32,16,8],num_res_blocks_middle=1).eval().to('mps')
    x=torch.randn(1,8,4,4,4,device='mps',requires_grad=True)
    expected=model(x).detach()
    with torch.autocast('mps',dtype=dtype):actual=model(x)
    assert actual.shape==(1,1,16,16,16) and actual.dtype==dtype
    torch.testing.assert_close(actual.float(),expected,atol=.035,rtol=.06)
    actual.float().square().mean().backward()
    assert torch.isfinite(x.grad).all() and x.grad.abs().sum()>0
