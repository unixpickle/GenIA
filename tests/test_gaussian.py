import importlib.util
import math
from pathlib import Path

import pytest
import torch

from genia.core.ops.gaussian import (
    gaussian_tile_intersections, project_gaussians, rasterization, rasterize_projected,
)
from genia.core.ops.metal import library

DEVICES = ["cpu"] + (["mps"] if torch.backends.mps.is_available() else [])
METAL = pytest.mark.skipif(not torch.backends.mps.is_available(), reason="requires Metal")


def _reference():
    spec = importlib.util.spec_from_file_location("gsplat_reference", Path(__file__).parent / "reference/gsplat_torch.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.mark.parametrize("device", DEVICES)
def test_projection_against_upstream(device):
    torch.manual_seed(21)
    n = 17
    means = torch.randn(n, 3)
    means[:, 2] = torch.rand(n) + 2
    quats = torch.randn(n, 4)
    scales = torch.rand(n, 3) * .4 + .03
    view = torch.eye(4)
    view[:3, 3] = torch.tensor([.1, -.2, .5])
    K = torch.tensor([[15., 0., 8.], [0., 13., 5.], [0., 0., 1.]])
    inputs = [x.requires_grad_() for x in (means, quats, scales, view, K)]
    ref = _reference()
    cov, _ = ref._quat_scale_to_covar_preci(quats, scales, compute_preci=False)
    _, xy, depth, conic, _ = ref._fully_fused_projection(means, cov, view[None], K[None], 17, 11)
    actual_inputs = [x.detach().to(device).requires_grad_() for x in inputs]
    actual = project_gaussians(*actual_inputs[:3], *actual_inputs[3:], 17, 11, torch.ones(n, device=device))
    for a, e in zip(actual[:3], (xy[0], conic[0], depth[0])):
        torch.testing.assert_close(a.cpu(), e, atol=2e-5, rtol=2e-5)
    weights = [torch.randn_like(x) for x in (xy[0], conic[0], depth[0])]
    sum((a * w.to(device)).sum() for a, w in zip(actual[:3], weights)).backward()
    sum((a * w).sum() for a, w in zip((xy[0], conic[0], depth[0]), weights)).backward()
    # The gsplat CPU reference multiplies K as a full matrix, unlike CUDA's
    # pinhole-only fx/fy/cx/cy use. Compare intrinsics' meaningful entries only.
    for a, e in zip(actual_inputs[:4], inputs[:4]):
        torch.testing.assert_close(a.grad.cpu(), e.grad, atol=5e-5, rtol=5e-5)
    for row, col in ((0, 0), (1, 1), (0, 2), (1, 2)):
        torch.testing.assert_close(actual_inputs[4].grad[row, col].cpu(), K.grad[row, col], atol=5e-5, rtol=5e-5)


@pytest.mark.parametrize("device", DEVICES)
def test_projection_culls_and_isotropic_footprint(device):
    means = torch.tensor([[0., 0., 2.], [0., 0., -.5], [0., 0., 0.], [100., 0., 2.]], device=device)
    q = torch.tensor([[1., 0., 0., 0.]], device=device).repeat(4, 1)
    s = torch.full((4, 3), .2, device=device)
    K = torch.tensor([[10., 0., 4.], [0., 10., 3.], [0., 0., 1.]], device=device)
    xy, conic, z, radius = project_gaussians(means, q, s, torch.eye(4, device=device), K, 8, 6, torch.ones(4, device=device))
    torch.testing.assert_close(xy[0], torch.tensor([4., 3.], device=device))
    torch.testing.assert_close(conic[0], torch.tensor([1 / 1.3, 0., 1 / 1.3], device=device))
    assert radius[0].tolist() == [4, 4]
    assert (radius[1:] == 0).all()
    assert torch.isfinite(conic).all()


@METAL
def test_tile_kernels():
    means = torch.tensor([[.1, .2], [15.9, 16.1], [28., 5.], [7., 8.], [-200., 3.], [19., 17.]])
    radii = torch.tensor([[2, 3], [3, 4], [20, 12], [0, 0], [1, 1], [2, 2]], dtype=torch.int32)
    depths = torch.tensor([3., 1., 4., 1., 5., 1.])
    bounds = torch.empty(6, 4, dtype=torch.int32, device="mps")
    counts = torch.empty(6, dtype=torch.int32, device="mps")
    library("gaussian").tile_counts(means.to("mps"), radii.to("mps"), bounds, counts, 8, 4, 3, threads=6)
    expected_bounds = torch.tensor([[0, 0, 1, 1], [1, 1, 3, 3], [1, 0, 4, 3],
                                    [0, 1, 0, 1], [0, 0, 0, 1], [2, 1, 3, 3]], dtype=torch.int32)
    torch.testing.assert_close(bounds.cpu(), expected_bounds)
    torch.testing.assert_close(counts.cpu(), torch.tensor([1, 4, 9, 0, 0, 2], dtype=torch.int32))
    cr, ci = gaussian_tile_intersections(means, radii, depths, 29, 19, 8)
    mr, mi = gaussian_tile_intersections(means.to("mps"), radii.to("mps"), depths.to("mps"), 29, 19, 8)
    torch.testing.assert_close(mi.cpu(), ci)
    for a, b in zip(mr.cpu().tolist(), cr.tolist()):
        torch.testing.assert_close(mi.cpu()[a[0]:a[1]], ci[b[0]:b[1]])


@METAL
def test_tile_range_writes_do_not_race():
    torch.manual_seed(65)
    means = torch.rand(400, 2) * 65
    radii = torch.randint(1, 13, (400, 2), dtype=torch.int32)
    depth = torch.rand(400)
    cr, ci = gaussian_tile_intersections(means, radii, depth, 65, 57, 4)
    for _ in range(8):
        mr, mi = gaussian_tile_intersections(means.to("mps"), radii.to("mps"), depth.to("mps"), 65, 57, 4)
        torch.testing.assert_close(mi.cpu(), ci)
        torch.testing.assert_close(mr.cpu(), cr)


@pytest.mark.parametrize("device", DEVICES)
def test_genia_rendering_entry_point(device):
    from genia.core.utils.rendering import render_gaussian_params
    means = torch.tensor([[0., 0., 2.]], device=device, requires_grad=True)
    quats = torch.tensor([[1., 0., 0., 0.]], device=device)
    scales = torch.full((1, 3), .2, device=device)
    opacity = torch.tensor([.5], device=device)
    sh = torch.zeros(1, 1, 3, device=device)
    camera = torch.eye(4, device=device, requires_grad=True)
    K = torch.tensor([[8., 0., 3.], [0., 8., 3.], [0., 0., 1.]], device=device)
    rgb, alpha, depth = render_gaussian_params(means, quats, scales, opacity, sh, camera, K, 6, 6)
    torch.testing.assert_close(rgb, alpha[..., None].expand_as(rgb) * .5, atol=1e-6, rtol=1e-6)
    torch.testing.assert_close(depth[alpha > .01], torch.full_like(depth[alpha > .01], 2.), atol=2e-5, rtol=2e-5)
    (rgb.sum() + alpha.sum() + depth.sum()).backward()
    assert torch.isfinite(means.grad).all() and torch.isfinite(camera.grad).all()


@METAL
def test_noncontiguous_camera():
    from genia.core.utils.rendering import render_gaussian_params
    from pytorch3d.transforms import axis_angle_to_matrix
    camera = torch.eye(4)
    camera[:3, :3] = axis_angle_to_matrix(torch.tensor([.1, -.2, .3]))
    camera[:3, 3] = torch.tensor([.1, .3, -.4])
    camera = camera.T.contiguous().T
    means = torch.tensor([[0., 0., 2.]])
    quats = torch.tensor([[1., 0., 0., 0.]])
    scales = torch.full((1, 3), .3)
    opacity = torch.tensor([.6])
    sh = torch.zeros(1, 1, 3)
    K = torch.tensor([[8., 0., 3.], [0., 8., 3.], [0., 0., 1.]])
    args = (means, quats, scales, opacity, sh, camera, K)
    ref = render_gaussian_params(*args, 6, 6)
    actual = render_gaussian_params(*(a.to("mps") for a in args), 6, 6)
    for a, e in zip(actual, ref):
        torch.testing.assert_close(a.cpu(), e, atol=3e-5, rtol=3e-5)


def _projected_scene(dtype=torch.float32):
    return [torch.tensor(x, dtype=dtype) for x in (
        [[1.5, 2.5], [3.2, 1.7], [-.2, 3.1]],
        [[.8, .12, .7], [.3, -.1, .5], [1.2, .2, .9]],
        [[.9, .2, .4, 2.], [.1, .7, .3, 3.], [.3, .2, .8, 4.]],
        [.8, .6, .9], [.03, .2, .1, 0.])]


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("channels", [1, 3, 4])
def test_compositor_values_and_gradients(device, channels):
    inputs = _projected_scene()
    inputs[2] = inputs[2][:, :channels]
    inputs[4] = inputs[4][:channels]
    inputs = [x.requires_grad_() for x in inputs]
    actual_inputs = [x.detach().to(device).requires_grad_() for x in inputs]
    radii = torch.full((3, 2), 5, dtype=torch.int32)
    depths = torch.tensor([2., 3., 4.])
    def run(xs, dev):
        m, c, f, o, bg = xs
        return rasterize_projected(m, c, f, o, radii.to(dev), depths.to(dev), bg, 7, 5, 2)
    expected = run(inputs, "cpu")
    actual = run(actual_inputs, device)
    torch.manual_seed(4)
    weights = [torch.randn_like(e) for e in expected]
    for a, e in zip(actual, expected):
        torch.testing.assert_close(a.cpu(), e, atol=2e-6, rtol=2e-6)
    sum((a * w.to(device)).sum() for a, w in zip(actual, weights)).backward()
    sum((e * w).sum() for e, w in zip(expected, weights)).backward()
    for a, e in zip(actual_inputs, inputs):
        torch.testing.assert_close(a.grad.cpu(), e.grad, atol=1e-5, rtol=2e-5)


@pytest.mark.parametrize("device", DEVICES)
def test_single_pixel_occlusion_saturation_and_termination(device):
    # Coincident means isolate alpha compositing from projection/footprint math.
    xy = torch.full((5, 2), .5, device=device, requires_grad=True)
    conic = torch.tensor([[1., 0., 1.]] * 5, device=device, requires_grad=True)
    colors = torch.eye(5, 4, device=device, requires_grad=True)
    opacity = torch.tensor([.5, .25, .002, .9999, .9999], device=device, requires_grad=True)
    bg = torch.tensor([.1, .2, .3, .4], device=device, requires_grad=True)
    image, alpha = rasterize_projected(xy, conic, colors, opacity,
        torch.ones(5, 2, dtype=torch.int32, device=device), torch.arange(5., device=device), bg, 1, 1)
    # Third splat is below threshold; fourth clamps alpha; fifth terminates
    # exclusively, before its contribution is added.
    T = .5 * .75 * .001
    expected = torch.tensor([.5, .125, 0., .5 * .75 * .999], device=device) + T * bg
    torch.testing.assert_close(image[0, 0], expected, atol=2e-7, rtol=2e-6)
    torch.testing.assert_close(alpha[0, 0, 0], torch.tensor(1 - T, device=device))
    (image.sum() + alpha.sum()).backward()
    assert opacity.grad[2:].tolist() == [0., 0., 0.]
    assert (colors.grad[4] == 0).all()
    assert torch.isfinite(opacity.grad).all()


@pytest.mark.parametrize("device", DEVICES)
def test_empty_scene(device):
    m = torch.empty(0, 2, device=device, requires_grad=True)
    c = torch.empty(0, 3, device=device, requires_grad=True)
    f = torch.empty(0, 3, device=device, requires_grad=True)
    o = torch.empty(0, device=device, requires_grad=True)
    bg = torch.tensor([.2, .4, .1], device=device, requires_grad=True)
    out, alpha = rasterize_projected(m, c, f, o, torch.empty(0, 2, dtype=torch.int32, device=device),
                                    torch.empty(0, device=device), bg, 3, 2)
    torch.testing.assert_close(out, bg.expand(2, 3, 3))
    assert (alpha == 0).all()
    (out.sum() + alpha.sum()).backward()
    for x in (m, c, f, o):
        assert x.grad is not None
    torch.testing.assert_close(bg.grad, torch.full_like(bg, 6))


def test_compositor_finite_differences():
    inputs = [x.requires_grad_() for x in _projected_scene(torch.double)]
    def run(m, c, f, o, bg):
        return rasterize_projected(m, c, f, o, torch.full((3, 2), 5, dtype=torch.int32),
                                  torch.tensor([2., 3., 4.], dtype=torch.double), bg, 3, 2)
    assert torch.autograd.gradcheck(run, tuple(inputs), fast_mode=True)


def test_reject_unsafe_tile_inputs():
    means = torch.zeros(2, 2)
    depths = torch.ones(2)
    with pytest.raises(ValueError, match="int32"):
        gaussian_tile_intersections(means, torch.ones(2, 2, dtype=torch.int64), depths, 4, 4)
    with pytest.raises(ValueError, match="positive"):
        gaussian_tile_intersections(means, torch.ones(2, 2, dtype=torch.int32), depths, 0, 4)


@METAL
@pytest.mark.parametrize("mode", ["RGB", "D", "ED", "RGB+D", "RGB+ED"])
@pytest.mark.parametrize("degree", [None, 0, 2])
def test_full_renderer_gradients(mode, degree):
    torch.manual_seed(52)
    n = 4
    means = torch.randn(n, 3) * .2
    means[:, 2] += 3
    q = torch.randn(n, 4)
    s = torch.rand(n, 3) * .3 + .2
    o = torch.rand(n) * .3 + .2
    colors = torch.rand(n, 3) if degree is None else torch.rand(n, (degree + 1)**2, 3) * .2
    view = torch.eye(4).repeat(2, 1, 1)
    view[1, :3, 3] = torch.tensor([.1, -.2, .3])
    K = torch.tensor([[8., 0., 3.], [0., 9., 2.], [0., 0., 1.]]).repeat(2, 1, 1)
    bg = torch.rand(2, 3)
    inputs = [x.requires_grad_() for x in (means, q, s, o, colors, view, K, bg)]
    actual_inputs = [x.detach().to("mps").requires_grad_() for x in inputs]
    def run(xs):
        return rasterization(*xs[:7], 6, 5, backgrounds=xs[7], render_mode=mode, sh_degree=degree)[:2]
    expected, actual = run(inputs), run(actual_inputs)
    for a, e in zip(actual, expected):
        torch.testing.assert_close(a.cpu(), e, atol=3e-6, rtol=3e-6)
    weights = [torch.randn_like(e) for e in expected]
    sum((a * w.to("mps")).sum() for a, w in zip(actual, weights)).backward()
    sum((e * w).sum() for e, w in zip(expected, weights)).backward()
    for a, e in zip(actual_inputs, inputs):
        if e.grad is None:
            assert a.grad is None
        else:
            torch.testing.assert_close(a.grad.cpu(), e.grad, atol=5e-5, rtol=5e-4)
