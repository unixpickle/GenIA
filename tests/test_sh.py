import numpy as np
import pytest
import torch

from genia.core.ops.sh import eval_sh


@pytest.mark.parametrize("degree", range(5))
def test_sh_basis_orthonormality(degree):
    # Independent spherical quadrature checks normalization, all coefficient
    # indices, and orthogonality, rather than reproducing the SH polynomials.
    z, weights = np.polynomial.legendre.leggauss(16)
    azimuth = np.arange(32) * 2 * np.pi / 32
    r = np.sqrt(1 - z*z)
    dirs = np.stack(np.broadcast_arrays(r[:, None] * np.cos(azimuth),
                                        r[:, None] * np.sin(azimuth), z[:, None]), -1)
    dirs = torch.from_numpy(dirs.reshape(-1, 3))
    size = (degree + 1)**2
    coefficients = torch.eye(size, dtype=torch.double).expand(len(dirs), size, size)
    values = eval_sh(degree, coefficients, dirs)
    quadrature = torch.from_numpy(np.repeat(weights, 32) * 2 * np.pi / 32)
    gram = values.T @ (quadrature[:, None] * values)
    torch.testing.assert_close(gram, torch.eye(size, dtype=torch.double), atol=2e-13, rtol=2e-13)


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="requires Metal")
@pytest.mark.parametrize("degree", range(5))
def test_sh_mps_gradients(degree):
    torch.manual_seed(80)
    coeff = torch.randn(7, 3, (degree + 1)**2, requires_grad=True)
    directions = torch.nn.functional.normalize(torch.randn(7, 3), dim=-1).requires_grad_()
    mc = coeff.detach().to("mps").requires_grad_()
    md = directions.detach().to("mps").requires_grad_()
    ref, actual = eval_sh(degree, coeff, directions), eval_sh(degree, mc, md)
    torch.testing.assert_close(actual.cpu(), ref, atol=2e-6, rtol=2e-6)
    ref.sum().backward()
    actual.sum().backward()
    torch.testing.assert_close(mc.grad.cpu(), coeff.grad, atol=2e-6, rtol=2e-6)
    if degree:
        torch.testing.assert_close(md.grad.cpu(), directions.grad, atol=5e-6, rtol=5e-6)
