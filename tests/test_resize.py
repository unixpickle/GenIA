import pytest
import torch
import torch.nn.functional as F

from genia.core.ops.resize import interpolate

pytestmark = pytest.mark.skipif(not torch.backends.mps.is_available(), reason="requires Metal")


@pytest.mark.parametrize("mode", ["bilinear", "bicubic"])
@pytest.mark.parametrize("shape,size", [((2, 3, 13, 19), (5, 7)), ((1, 2, 5, 7), (17, 23)),
                                       ((1, 1, 17, 23), (1, 1)), ((1, 2, 1, 7), (5, 3)),
                                       ((1, 1, 11, 7), (11, 13)), ((1, 2, 9, 13), (9, 13))])
def test_resize_values_gradients(mode, shape, size):
    torch.manual_seed(73)
    x = torch.randn(shape, requires_grad=True)
    m = x.detach().to("mps").requires_grad_()
    expected = F.interpolate(x, size, mode=mode, align_corners=False, antialias=True)
    actual = interpolate(m, size, mode=mode, align_corners=False, antialias=True)
    torch.testing.assert_close(actual.cpu(), expected, atol=5e-6, rtol=5e-6)
    weight = torch.randn_like(expected)
    (actual * weight.to("mps")).sum().backward()
    (expected * weight).sum().backward()
    torch.testing.assert_close(m.grad.cpu(), x.grad, atol=8e-6, rtol=1e-5)


@pytest.mark.parametrize("mode", ["bilinear", "bicubic"])
@pytest.mark.parametrize("recompute", [True, False])
@pytest.mark.parametrize("align", [True, False])
@pytest.mark.parametrize("scale", [(.72, 1.38), (1.03, .85), (.08, .06)])
def test_scale_factors(mode, recompute, align, scale):
    x = torch.randn(1, 2, 13, 17, requires_grad=True)
    m = x.detach().to("mps").requires_grad_()
    options = dict(scale_factor=scale, mode=mode, align_corners=align,
                   recompute_scale_factor=recompute, antialias=True)
    ref, got = F.interpolate(x, **options), interpolate(m, **options)
    torch.testing.assert_close(got.cpu(), ref, atol=8e-6, rtol=8e-6)
    got.square().sum().backward()
    ref.square().sum().backward()
    torch.testing.assert_close(m.grad.cpu(), x.grad, atol=2e-5, rtol=2e-5)


def test_noncontiguous_half():
    x = torch.randn(1, 3, 9, 15).half().transpose(-1, -2)
    m = x.to("mps").requires_grad_()
    ref = F.interpolate(x.float(), (8, 12), mode="bicubic", align_corners=False, antialias=True).half()
    got = interpolate(m, (8, 12), mode="bicubic", align_corners=False, antialias=True)
    torch.testing.assert_close(got.cpu(), ref, atol=1e-3, rtol=1e-3)
    got.sum().backward()
    assert m.grad is not None and torch.isfinite(m.grad).all()
