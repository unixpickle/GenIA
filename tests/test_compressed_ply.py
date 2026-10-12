"""GenIA Gaussian export/reload across chunk and spherical-harmonic layouts."""
import numpy as np
import pytest
import torch

from genia.core.utils.gaussian import create_gaussians_object
from genia.core.utils.io_utils import save_gaussian_compressed_ply, save_gaussian_ply
from genia.core.utils.render_final_results import load_gaussian_ply
from gsconverter.formats.compressed_ply import CompressedPlyFormat
from plyfile import PlyData


@pytest.mark.parametrize("device", ["cpu"] + (["mps"] if torch.backends.mps.is_available() else []))
@pytest.mark.parametrize("degree", [0, 3, 4])
def test_full_gaussian_compressed_export_and_reload(tmp_path, device, degree):
    torch.manual_seed(48)
    n = 273  # a full 256-splat chunk and a partial chunk
    xyz = torch.rand(n, 3, device=device) * 2 - 1
    features = torch.rand(n, (degree+1)**2, 3, device=device) * .4 - .2
    scales = torch.rand(n, 3, device=device) * .1 + .01
    rotations = torch.randn(n, 4, device=device)
    rotations /= rotations.norm(dim=-1, keepdim=True)
    opacity = torch.rand(n, 1, device=device) * .8 + .1
    model = create_gaussians_object(xyz, features, scales, rotations, opacity)
    ordinary = tmp_path / 'ordinary.ply'
    compressed = tmp_path / 'cloud.compressed.ply'
    save_gaussian_ply(model, str(ordinary))
    save_gaussian_compressed_ply(model, str(compressed))
    data = PlyData.read(ordinary)['vertex'].data
    order = np.arange(n, dtype=np.uint32)
    CompressedPlyFormat()._sort_morton_order(data, order)
    order = torch.from_numpy(order.astype(np.int64)).to(device)
    loaded = load_gaussian_ply(compressed, device=device)
    assert loaded.means.device.type == device
    torch.testing.assert_close(loaded.means, xyz[order], atol=.002, rtol=0)
    torch.testing.assert_close(loaded.scales, scales[order], atol=.0002, rtol=.002)
    torch.testing.assert_close(loaded.opacities, opacity[order, 0], atol=.004, rtol=0)
    # Smallest-three quaternion packing can reverse a quaternion's sign.
    agreement = (loaded.quats * rotations[order]).sum(-1).abs()
    assert agreement.min() > .9999
    torch.testing.assert_close(loaded.sh[:, :1], features[order, :1], atol=.014, rtol=0)
    if degree:
        torch.testing.assert_close(loaded.sh[:, 1:], features[order, 1:], atol=.032, rtol=0)
    assert loaded.sh.shape == features.shape
    assert compressed.stat().st_size < ordinary.stat().st_size
    # Public fallback reader also accepts an ordinary binary Gaussian PLY.
    fallback = CompressedPlyFormat().read(str(ordinary))
    np.testing.assert_array_equal(fallback['x'], data['x'])
