import numpy as np
import pytest

from genia.core.utils.colmap_export import write_colmap_model
from mapanything.utils.colmap import read_model


@pytest.mark.parametrize('dtype', [np.float32, np.float64])
def test_colmap_text_and_binary_are_readable_and_agree(tmp_path, dtype):
    xyz = np.array([[.125, -1.3, 2.7], [1e-5, 1024., -17.]], dtype=dtype)
    rgb = np.array([[0, 128, 255], [31, 57, 191]], dtype=np.uint8)
    k = np.array([[230., 0, 50.], [0, 231., 40.], [0, 0, 1.]])
    pose = np.eye(4); pose[:3, 3] = [.3, -.2, 1.]
    records = [(k, np.eye(4), 'first.png'), (k, pose, 'second.png')]
    assert write_colmap_model(tmp_path, records, 100, 80, xyz, rgb) == (1, 2)
    binary = read_model(str(tmp_path), '.bin')
    text = read_model(str(tmp_path), '.txt')
    for model in (binary, text):
        cameras, images, points = model
        np.testing.assert_array_equal(cameras[1].params, [230., 231., 50., 40.])
        for i, point in points.items():
            np.testing.assert_array_equal(point.xyz, xyz[i-1].astype(np.float64))
            np.testing.assert_array_equal(point.rgb, rgb[i-1])
        np.testing.assert_allclose(images[2].tvec, -pose[:3, 3])
        assert images[1].name == 'first.png' and images[2].camera_id == 1
