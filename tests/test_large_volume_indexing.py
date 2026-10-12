"""Regressions for MPS nonzero's float32 rounding of large coordinates."""
import pytest
import torch

from genia.core.ops.indexing import flat_nonzero, unravel_indices
from genia.core.ops.marching_cubes import DiffMC
from genia.core.ops.dual_marching_cubes import DiffDMC


@pytest.mark.parametrize('device', ['cpu', 'mps'])
def test_flat_nonzero_empty_and_strided(device):
    if device == 'mps' and not torch.backends.mps.is_available():
        pytest.skip('requires Metal')
    data = torch.tensor([[0, 1, 0], [2, 0, 3]], device=device).T
    torch.testing.assert_close(flat_nonzero(data).cpu(), torch.tensor([1, 2, 5]))
    assert flat_nonzero(torch.empty(0, device=device)).shape == (0,)


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason='requires Metal')
def test_flat_nonzero_indices_across_float32_precision_boundary():
    n = (1 << 25) + 19
    expected = torch.tensor([0, (1 << 24)-1, (1 << 24)+1, (1 << 24)+3,
                             (1 << 25)-1, (1 << 25)+1, n-1])
    mask = torch.zeros(n, device='mps', dtype=torch.bool)
    mask[expected.to('mps')] = True
    torch.testing.assert_close(flat_nonzero(mask).cpu(), expected)


@pytest.mark.parametrize('device', ['cpu', 'mps'])
def test_unravel_large_indices_with_strides_and_empty_input(device):
    if device == 'mps' and not torch.backends.mps.is_available():
        pytest.skip('requires Metal')
    indices = torch.arange((1 << 27)-31, (1 << 27)+33).reshape(8,8).T
    shape = (1024,513,511)
    expected = torch.stack(torch.unravel_index(indices, shape), dim=-1)
    torch.testing.assert_close(unravel_indices(indices.to(device), shape).cpu(), expected)
    empty = torch.empty((0,3), device=device, dtype=torch.long)
    assert unravel_indices(empty, shape).shape == (0,3,3)


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason='requires Metal')
@pytest.mark.parametrize('operator', [DiffMC, DiffDMC])
def test_mesh_extraction_above_float32_index_precision(operator):
    # Active cells and edges are above 2**24 in the padded volume. Small-grid
    # topology tests cannot expose rounding of their flattened coordinates.
    grid = torch.ones(67, 512, 512)
    grid[-4:-1, 240:244, 247:251] = -.75
    with torch.no_grad():
        expected_v, expected_f = operator()(grid, normalize=False)
        vertices, faces = operator()(grid.to('mps'), normalize=False)
    torch.testing.assert_close(vertices.cpu(), expected_v, atol=1e-5, rtol=1e-6)
    torch.testing.assert_close(faces.cpu(), expected_f)
