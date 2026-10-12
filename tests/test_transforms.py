import math

import pytest
import torch

from pytorch3d import transforms as tr

DEVICES = ["cpu"] + (["mps"] if torch.backends.mps.is_available() else [])


@pytest.mark.parametrize("device", DEVICES)
def test_transform_composition_inverse_and_normals(device):
    points = torch.tensor([[.1, .2, .3], [1., 2., 3.]], device=device, requires_grad=True)
    R = tr.axis_angle_to_matrix(torch.tensor([.2, .3, -.4], device=device))
    scale = torch.tensor([2., 3., 4.], device=device)
    offset = torch.tensor([1., -2., .5], device=device)
    transform = tr.Transform3d(device=device).scale(scale[None]).rotate(R).translate(offset[None])
    got = transform.transform_points(points)
    expected = (points * scale) @ R + offset
    torch.testing.assert_close(got, expected)
    torch.testing.assert_close(transform.inverse().transform_points(got), points, atol=1e-6, rtol=1e-6)
    normals = torch.nn.functional.normalize(points.detach(), dim=-1)
    torch.testing.assert_close(transform.transform_normals(normals), (normals / scale) @ R)
    got.sum().backward()
    torch.testing.assert_close(points.grad, (scale * R.sum(1)).expand_as(points))


@pytest.mark.parametrize("device", DEVICES)
def test_rotation_conversion_roundtrips(device):
    angles = torch.tensor([[0., 0., 0.], [.2, -.3, .1], [math.pi - .001, 0., 0.]], device=device)
    q = tr.axis_angle_to_quaternion(angles)
    R = tr.quaternion_to_matrix(q)
    eye = torch.eye(3, device=device).expand(3, 3, 3)
    torch.testing.assert_close(R @ R.transpose(-1, -2), eye, atol=1e-6, rtol=1e-6)
    torch.testing.assert_close(tr.quaternion_to_matrix(tr.matrix_to_quaternion(R)), R, atol=1e-6, rtol=1e-6)
    torch.testing.assert_close(tr.quaternion_to_axis_angle(q), angles, atol=1e-6, rtol=1e-6)
    torch.testing.assert_close(tr.axis_angle_to_matrix(tr.matrix_to_axis_angle(R)), R, atol=1e-6, rtol=1e-6)
    torch.testing.assert_close(tr.rotation_6d_to_matrix(tr.matrix_to_rotation_6d(R)), R, atol=1e-6, rtol=1e-6)
    torch.testing.assert_close(tr.quaternion_to_matrix(q * 2), R)
    identity = tr.quaternion_multiply(q, tr.quaternion_invert(q))
    torch.testing.assert_close(identity, torch.tensor([1., 0., 0., 0.], device=device).expand_as(q), atol=1e-6, rtol=1e-6)
    p = torch.tensor([[1., 2., 3.]], device=device).expand(3, 3)
    torch.testing.assert_close(tr.quaternion_apply(q, p), (R @ p[..., None]).squeeze(-1))


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("composed", [True, False])
def test_inverse_noncontiguous_matrix(device, composed):
    matrix = torch.tensor([[2., .1, .4, 0.], [.2, 3., .1, 0.], [-.1, .3, 1., 0.], [1., 2., 3., 1.]], device=device)
    matrix = matrix.T.contiguous().T
    assert not matrix.is_contiguous()
    transform = tr.Transform3d(matrix=matrix)
    inverse = transform.inverse(invert_composed=composed).get_matrix()[0]
    torch.testing.assert_close(inverse @ matrix, torch.eye(4, device=device), atol=1e-6, rtol=1e-6)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("convention", ["XYZ", "ZYX", "ZXZ"])
def test_euler_conversion(device, convention):
    e = torch.tensor([[.2, .4, .1], [-.2, .6, .8]], device=device)
    R = tr.euler_angles_to_matrix(e, convention)
    restored = tr.euler_angles_to_matrix(tr.matrix_to_euler_angles(R, convention), convention)
    torch.testing.assert_close(restored, R, atol=1e-6, rtol=1e-6)


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="requires Metal")
@pytest.mark.parametrize("function,shape", [
    (tr.axis_angle_to_matrix, (4, 3)), (tr.axis_angle_to_quaternion, (4, 3)),
    (tr.quaternion_to_matrix, (4, 4)), (tr.quaternion_to_axis_angle, (4, 4)),
    (tr.rotation_6d_to_matrix, (4, 6)), (tr.so3_exp_map, (4, 3)),
    (tr.se3_exp_map, (4, 6)),
])
def test_transform_mps_gradients(function, shape):
    torch.manual_seed(20)
    x = (torch.randn(shape) * .2).requires_grad_()
    m = x.detach().to("mps").requires_grad_()
    expected, actual = function(x), function(m)
    torch.testing.assert_close(actual.cpu(), expected, atol=1e-6, rtol=1e-5)
    weight = torch.randn_like(expected)
    (actual * weight.to("mps")).sum().backward()
    (expected * weight).sum().backward()
    torch.testing.assert_close(m.grad.cpu(), x.grad, atol=2e-5, rtol=2e-5)
