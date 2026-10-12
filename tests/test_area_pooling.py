import pytest
import torch
from torch.nn import functional as F

from genia.core.ops.pooling import adaptive_avg_pool2d
from genia.core.ops.resize import interpolate

pytestmark=pytest.mark.skipif(not torch.backends.mps.is_available(),reason='requires Metal')


@pytest.mark.parametrize('dtype',[torch.float32,torch.float16,torch.bfloat16])
@pytest.mark.parametrize('shape,size',[
    ((1,3,518,518),(129,129)), ((2,4,17,23),(8,6)),
    ((1,2,3,5),(7,11)), ((1,2,1,7),(1,3)), ((2,3,9,11),(1,1)),
])
def test_adaptive_pooling_values_and_gradient(shape,size,dtype):
    torch.manual_seed(456)
    x=torch.randn(shape,dtype=dtype).contiguous(memory_format=torch.channels_last).requires_grad_()
    mx=x.detach().to('mps').requires_grad_()
    expected=F.adaptive_avg_pool2d(x,size)
    actual=adaptive_avg_pool2d(mx,size)
    tolerance=2e-6 if dtype==torch.float32 else .004 if dtype==torch.float16 else .025
    torch.testing.assert_close(actual.cpu(),expected,atol=tolerance,rtol=tolerance)
    grad=torch.randn_like(expected)
    expected.backward(grad);actual.backward(grad.to('mps'))
    torch.testing.assert_close(mx.grad.cpu(),x.grad,atol=tolerance,rtol=tolerance)


def test_area_interpolate_scale_factor_and_multiscale_loss():
    from genia.core.utils.refinement import _compute_multiscale_rgb_loss
    from genia.core.utils.config import LossConfig
    torch.manual_seed(861)
    x=torch.rand(518,518,3,requires_grad=True)
    y=torch.rand_like(x)
    mx=x.detach().to('mps').requires_grad_()
    config=LossConfig(rgb_multiscale_scales=[1.,.5,.25],rgb_multiscale_weights=[.5,.3,.2])
    reference=0.
    for s,w in zip(config.rgb_multiscale_scales,config.rgb_multiscale_weights):
        a,b=[F.interpolate(t.permute(2,0,1)[None],scale_factor=s,mode='area') for t in [x,y]]
        reference=reference+w*(a-b).abs().mean()
        actual=interpolate(mx.permute(2,0,1)[None],scale_factor=s,mode='area')
        torch.testing.assert_close(actual.cpu(),a,atol=2e-6,rtol=2e-6)
    got=_compute_multiscale_rgb_loss(mx,y.to('mps'),config)
    torch.testing.assert_close(got.cpu(),reference)
    reference.backward();got.backward()
    torch.testing.assert_close(mx.grad.cpu(),x.grad,atol=1e-10,rtol=3e-5)


def test_unbatched_none_axis_and_empty_batch():
    x=torch.randn(3,9,13)
    torch.testing.assert_close(adaptive_avg_pool2d(x.to('mps'),(None,5)).cpu(),
                               F.adaptive_avg_pool2d(x,(None,5)))
    x=torch.empty(0,3,9,13,device='mps',requires_grad=True)
    y=adaptive_avg_pool2d(x,(5,6));assert y.shape==(0,3,5,6)
    y.sum().backward();assert x.grad.shape==x.shape
