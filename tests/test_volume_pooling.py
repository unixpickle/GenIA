import pytest
import torch
import torch.nn.functional as F

from genia.core.ops.pooling import max_pool3d_stride1

DEVICES = ['cpu'] + (['mps'] if torch.backends.mps.is_available() else [])


@pytest.mark.parametrize('device', DEVICES)
@pytest.mark.parametrize('kernel', [1, 3, 5])
@pytest.mark.parametrize('ties', [False, True])
def test_volume_pool_values_and_gradients(device, kernel, ties):
    torch.manual_seed(151)
    x = torch.randn(7, 5, 11).transpose(0, 2)
    if ties:
        x = x.round()
    x.requires_grad_()
    mx = x.detach().to(device).requires_grad_()
    expected = F.max_pool3d(x[None, None], kernel, stride=1, padding=kernel//2)[0, 0]
    actual = max_pool3d_stride1(mx, kernel)
    torch.testing.assert_close(actual.cpu(), expected, atol=0, rtol=0)
    weights = torch.randn_like(expected)
    (expected*weights).sum().backward()
    (actual*weights.to(device)).sum().backward()
    torch.testing.assert_close(mx.grad.cpu(), x.grad, atol=4e-6, rtol=2e-5)


@pytest.mark.parametrize('device', DEVICES)
def test_inference_half_pool_and_nearest_grid_upsampling(device):
    from triposg.inference_utils import parallel_zoom
    torch.manual_seed(752)
    x = torch.randn(7, 5, 9).half().transpose(1, 2)
    mx = x.to(device)
    with torch.no_grad():
        for kernel in (3, 5):
            expected = F.max_pool3d(x.float()[None, None], kernel, stride=1, padding=kernel//2)[0, 0].half()
            torch.testing.assert_close(max_pool3d_stride1(mx, kernel).cpu(), expected, atol=0, rtol=0)
    x.requires_grad_(); mx.requires_grad_()
    expected = F.interpolate(x.float()[None, None], scale_factor=2, mode='nearest')[0, 0].half()
    actual = parallel_zoom(mx, 2)
    torch.testing.assert_close(actual.cpu(), expected, atol=0, rtol=0)
    expected.sum().backward(); actual.sum().backward()
    torch.testing.assert_close(mx.grad.cpu(), x.grad, atol=0, rtol=0)
