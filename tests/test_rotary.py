import pytest
import torch

from genia.core.ops.rotary import rotary_embedding

DEVICES = ["cpu"] + (["mps"] if torch.backends.mps.is_available() else [])


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
def test_complex_reference_and_gradients(device, dtype):
    torch.manual_seed(2)
    x = torch.randn(2, 3, 4, 8, dtype=dtype, requires_grad=True)
    a = torch.randn(3, 4, 4, requires_grad=True)
    z = torch.view_as_complex(x.float().reshape(2, 3, 4, 4, 2))
    ref = torch.view_as_real(z * torch.polar(torch.ones_like(a), a)).flatten(-2).to(dtype)
    xm = x.detach().to(device).requires_grad_()
    am = a.detach().to(device).requires_grad_()
    got = rotary_embedding(xm, am)
    tol = 2e-3 if dtype == torch.float16 else 2e-6
    torch.testing.assert_close(got.cpu(), ref, atol=tol, rtol=tol)
    cotangent = torch.randn_like(ref)
    (got * cotangent.to(device)).sum().backward()
    (ref * cotangent).sum().backward()
    torch.testing.assert_close(xm.grad.cpu(), x.grad, atol=tol, rtol=tol)
    torch.testing.assert_close(am.grad.cpu(), a.grad, atol=tol, rtol=tol)
