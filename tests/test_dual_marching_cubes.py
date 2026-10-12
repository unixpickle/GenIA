import importlib.util
from pathlib import Path

import numpy as np
import pytest
import torch
from diso import DiffDMC

DEVICES = ['cpu'] + (['mps'] if torch.backends.mps.is_available() else [])
spec = importlib.util.spec_from_file_location('dual_oracle', Path(__file__).parent/'reference/diso/dual_reference.py')
oracle = importlib.util.module_from_spec(spec); spec.loader.exec_module(oracle)


@pytest.mark.parametrize('device', DEVICES)
@pytest.mark.parametrize('deformed', [False, True])
def test_dual_mc_reference(device, deformed):
    rng = np.random.default_rng(322)
    grid = rng.normal(size=(5, 7, 6)).astype(np.float32)
    deform = rng.uniform(-.2, .2, (*grid.shape, 3)).astype(np.float32) if deformed else None
    expected, quads = oracle.extract(grid, deform, iso=.1)
    g = torch.tensor(grid, device=device, requires_grad=True)
    d = None if deform is None else torch.tensor(deform, device=device, requires_grad=True)
    vertices, faces = DiffDMC()(g, d, isovalue=.1, return_quads=True)
    torch.testing.assert_close(vertices.cpu(), torch.tensor(expected, dtype=torch.float32), atol=3e-7, rtol=2e-6)
    torch.testing.assert_close(faces.cpu(), torch.tensor(quads))
    vertices.square().sum().backward()
    assert g.grad.isfinite().all() and g.grad.abs().sum() > 0
    if deformed:
        assert d.grad.isfinite().all() and d.grad.abs().sum() > 0


@pytest.mark.parametrize('device', DEVICES)
def test_all_256_cases_ambiguity_and_nan_grid(device):
    corners = oracle.tables()[0]
    grid = np.ones((256*3+1, 4, 4), dtype=np.float32)
    for code in range(256):
        for c, (x, y, z) in enumerate(corners):
            grid[code*3+x+1, y+1, z+1] = .75 if code & (1 << c) else -.25
    # TripoSG marks unevaluated adaptive cells NaN. Their edges emit no quads.
    grid[0, 0, :] = np.nan
    expected, quads = oracle.extract(grid)
    vertices, faces = DiffDMC()(torch.tensor(grid, device=device), return_quads=True)
    torch.testing.assert_close(vertices.cpu(), torch.tensor(expected, dtype=torch.float32), atol=2e-7, rtol=2e-6)
    torch.testing.assert_close(faces.cpu(), torch.tensor(quads))
    assert vertices.isfinite().all()


def test_dual_mc_triangulation_and_finite_differences():
    torch.manual_seed(99)
    g = (torch.randn(3, 4, 3, dtype=torch.double).sign()*(.5+torch.rand(3, 4, 3, dtype=torch.double))).requires_grad_()
    d = (torch.rand(3, 4, 3, 3, dtype=torch.double)*.1).requires_grad_()
    mc = DiffDMC(dtype=torch.double)
    assert torch.autograd.gradcheck(lambda a, b: mc(a, b, return_quads=True)[0], (g, d), fast_mode=True)
    v, q = oracle.extract(g.detach().numpy(), d.detach().numpy())
    faces = oracle.triangulate(v, q)
    actual_v, actual_f = mc(g, d)
    torch.testing.assert_close(actual_f, torch.tensor(faces))
    torch.testing.assert_close(actual_v, torch.tensor(v), atol=1e-14, rtol=1e-14)


@pytest.mark.parametrize('device', DEVICES)
def test_dual_mc_equality_normalization_and_empty(device):
    grid = torch.arange(5, device=device, dtype=torch.float32)[:, None, None].expand(5, 4, 3)-2
    v, q = DiffDMC()(grid, normalize=False, return_quads=True)
    nv, nq = DiffDMC()(grid, return_quads=True)
    torch.testing.assert_close(nv, v/v.new_tensor([4, 3, 2]))
    torch.testing.assert_close(nq, q)
    expected, eq = oracle.extract(grid.cpu().numpy(), normalize=False)
    torch.testing.assert_close(v.cpu(), torch.tensor(expected, dtype=torch.float32))
    torch.testing.assert_close(q.cpu(), torch.tensor(eq))
    for sign in [-1., 0., 1.]:
        g = torch.full((3, 4, 5), sign, device=device, requires_grad=True)
        v, f = DiffDMC()(g)
        assert v.shape == (0, 3) and f.shape == (0, 4)
        v.sum().backward()
        assert (g.grad == 0).all()


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason='requires Metal')
def test_dual_mc_cpu_mps_gradient_and_triangle_order():
    torch.manual_seed(425)
    g = torch.randn(9, 11, 7, requires_grad=True)
    d = (torch.rand(9, 11, 7, 3)*.1).requires_grad_()
    mg = g.detach().to('mps').requires_grad_()
    md = d.detach().to('mps').requires_grad_()
    v, f = DiffDMC()(g, d)
    mv, mf = DiffDMC()(mg, md)
    torch.testing.assert_close(mf.cpu(), f)
    torch.testing.assert_close(mv.cpu(), v)
    v.square().sum().backward(); mv.square().sum().backward()
    torch.testing.assert_close(mg.grad.cpu(), g.grad, atol=1e-5, rtol=2e-5)
    torch.testing.assert_close(md.grad.cpu(), d.grad, atol=1e-6, rtol=2e-5)
