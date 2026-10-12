"""Local differentiable 3D Gaussian rasterization for Apple Metal and CPU.

Projection and SH use PyTorch autograd. Metal handles tile binning and front to
back compositing, with a reverse compositing kernel for gradients. Pixel centers,
opacity cutoffs, depth order and expected depth follow gsplat 1.5.3's classic
pinhole renderer. This is the subset used by GenIA, not the whole gsplat API.
"""
import math

import torch
import torch.nn.functional as F
from torch.autograd.function import once_differentiable

from .metal import library
from .sh import eval_sh


def project_gaussians(means, quats, scales, viewmat, K, width, height, opacities,
                      eps2d=0.3, near_plane=0.01, far_plane=1e10, radius_clip=0.0):
    """Project one camera; return centers, inverse conics, depths and integer radii."""
    q = F.normalize(quats, dim=-1)
    w, x, y, z = q.unbind(-1)
    R = torch.stack((1 - 2 * (y*y + z*z), 2 * (x*y - w*z), 2 * (x*z + w*y),
                     2 * (x*y + w*z), 1 - 2 * (x*x + z*z), 2 * (y*z - w*x),
                     2 * (x*z - w*y), 2 * (y*z + w*x), 1 - 2 * (x*x + y*y)), -1).reshape(-1, 3, 3)
    M = R * scales[:, None, :]
    cov = M @ M.transpose(-1, -2)
    cam_R = viewmat[:3, :3]
    camera = means @ cam_R.T + viewmat[:3, 3]
    cov = cam_R @ cov @ cam_R.T
    depth = camera[:, 2]
    # Invalid depths never contribute. Avoid infinities before culling so zero
    # upstream gradients do not become NaN in the projection backward pass.
    safe_z = torch.where(depth.abs() > 1e-8, depth, torch.ones_like(depth))
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    tx = (camera[:, 0] / safe_z).clamp(-cx / fx - 0.15 * width / fx,
                                      (width - cx) / fx + 0.15 * width / fx)
    ty = (camera[:, 1] / safe_z).clamp(-cy / fy - 0.15 * height / fy,
                                      (height - cy) / fy + 0.15 * height / fy)
    zero = torch.zeros_like(depth)
    J = torch.stack((fx / safe_z, zero, -fx * tx / safe_z,
                     zero, fy / safe_z, -fy * ty / safe_z), -1).reshape(-1, 2, 3)
    cov2 = J @ cov @ J.transpose(-1, -2)
    a, b, c = cov2[:, 0, 0] + eps2d, cov2[:, 0, 1], cov2[:, 1, 1] + eps2d
    det = a * c - b * b
    denom = det.clamp_min(1e-10)
    conics = torch.stack((c / denom, -b / denom, a / denom), -1)
    centers = torch.stack((fx * camera[:, 0] / safe_z + cx, fy * camera[:, 1] / safe_z + cy), -1)
    with torch.no_grad():
        extend = (2 * (opacities.clamp_min(1 / 255) * 255).log()).sqrt().clamp_max(3.33)
        radii = torch.ceil(extend[:, None] * torch.stack((a, c), -1).clamp_min(0).sqrt())
        valid = ((depth >= near_plane) & (depth <= far_plane) & (det > 0) &
                 (opacities >= 1 / 255) & (radii.amax(-1) > radius_clip) &
                 torch.isfinite(centers).all(-1) & torch.isfinite(radii).all(-1) &
                 (centers[:, 0] + radii[:, 0] > 0) & (centers[:, 0] - radii[:, 0] < width) &
                 (centers[:, 1] + radii[:, 1] > 0) & (centers[:, 1] - radii[:, 1] < height))
        radii = torch.where(valid[:, None], radii, 0).to(torch.int32)
    return centers, conics, depth, radii


@torch.no_grad()
def gaussian_tile_intersections(means2d, radii, depths, width, height, tile_size=16):
    """Return tile ranges and Gaussian ids, ordered by increasing depth per tile."""
    if width < 1 or height < 1 or tile_size < 1:
        raise ValueError("image dimensions and tile_size must be positive")
    n = len(means2d)
    if means2d.shape != (n, 2) or radii.shape != (n, 2) or depths.shape != (n,):
        raise ValueError("expected centers=(N,2), radii=(N,2), depths=(N,)")
    if radii.dtype != torch.int32 or radii.device != means2d.device or depths.device != means2d.device:
        raise ValueError("radii must be int32; all tile inputs must share a device")
    if not means2d.is_floating_point() or not depths.is_floating_point():
        raise ValueError("centers and depths must be floating point")
    if means2d.device.type == "mps" and means2d.dtype != torch.float32:
        raise ValueError("Metal tile binning requires float32 centers")
    tw, th = math.ceil(width / tile_size), math.ceil(height / tile_size)
    ranges = torch.zeros((tw * th, 2), device=means2d.device, dtype=torch.int32)
    if not n:
        return ranges, torch.empty(0, device=means2d.device, dtype=torch.int32)
    order = depths.argsort(stable=True)
    if means2d.device.type == "mps":
        lib = library("gaussian")
        bounds = torch.empty((n, 4), device=means2d.device, dtype=torch.int32)
        counts = torch.empty(n, device=means2d.device, dtype=torch.int32)
        lib.tile_counts(means2d.contiguous(), radii.contiguous(), bounds, counts, tile_size, tw, th, threads=n)
        cumulative = counts.cumsum(0)
        total = int(cumulative[-1])
        if total >= 2**31:
            raise RuntimeError("Gaussian tile intersections exceed the int32 index limit")
        if not total:
            return ranges, torch.empty(0, device=means2d.device, dtype=torch.int32)
        offsets = (cumulative - counts).int()
        ranks = torch.empty(n, device=means2d.device, dtype=torch.int32)
        ranks[order] = torch.arange(n, device=means2d.device, dtype=torch.int32)
        keys = torch.empty(total, device=means2d.device, dtype=torch.int64)
        ids = torch.empty(total, device=means2d.device, dtype=torch.int32)
        lib.tile_emit(bounds, offsets, ranks, keys, ids, n, tw, threads=n)
        permutation = keys.argsort()
        keys, ids = keys[permutation], ids[permutation]
        lib.tile_ranges(keys, ranges, n, total, threads=total)
        return ranges, ids
    bins = [[] for _ in range(tw * th)]
    for i in order.tolist():
        x, y = means2d[i].tolist()
        rx, ry = radii[i].tolist()
        if rx <= 0 or ry <= 0:
            continue
        for ty in range(max(0, math.floor((y - ry) / tile_size)), min(th, math.ceil((y + ry) / tile_size))):
            for tx in range(max(0, math.floor((x - rx) / tile_size)), min(tw, math.ceil((x + rx) / tile_size))):
                bins[ty * tw + tx].append(i)
    flat = []
    for t, items in enumerate(bins):
        ranges[t, 0] = len(flat)
        flat.extend(items)
        ranges[t, 1] = len(flat)
    return ranges, torch.tensor(flat, device=means2d.device, dtype=torch.int32)


class _MetalSplat(torch.autograd.Function):
    @staticmethod
    def forward(ctx, means, conics, colors, opacities, background, ranges, ids, width, height, tile_size):
        output = colors.new_empty((height, width, 4))
        trans = colors.new_empty((height, width))
        last = torch.empty((height, width), dtype=torch.int32, device=colors.device)
        tw = math.ceil(width / tile_size)
        ctx.tiled = tile_size == 16
        if ctx.tiled:
            library("gaussian").splat_forward_tiled(means, conics, colors, opacities, background, ranges, ids,
                output, trans, last, width, height, tw,
                threads=tw * math.ceil(height / 16) * 256, group_size=256)
        else:
            library("gaussian").splat_forward(means, conics, colors, opacities, background, ranges, ids,
                                              output, trans, last, width, tile_size, tw, threads=width * height)
        ctx.save_for_backward(means, conics, colors, opacities, background, ranges, ids, trans, last)
        ctx.parameters = width, tile_size, tw
        return output, 1 - trans

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_output, grad_alpha):
        means, conics, colors, opacities, background, ranges, ids, trans, last = ctx.saved_tensors
        grads = [torch.zeros_like(x) for x in (means, conics, colors, opacities)]
        if ctx.tiled:
            height, width = trans.shape
            tw = ctx.parameters[2]
            library("gaussian").splat_backward_tiled(means, conics, colors, opacities, background, ranges, ids,
                trans, last, grad_output.contiguous(), grad_alpha.contiguous(), *grads, width, height, tw,
                threads=tw * math.ceil(height / 16) * 256, group_size=256)
        else:
            library("gaussian").splat_backward(means, conics, colors, opacities, background, ranges, ids,
                                               trans, last, grad_output.contiguous(), grad_alpha.contiguous(),
                                               *grads, *ctx.parameters, threads=trans.numel())
        grad_background = (grad_output * trans[..., None]).sum((0, 1))
        return *grads, grad_background, None, None, None, None, None


def _reference_splat(means, conics, colors, opacities, background, ranges, ids, width, height, tile_size):
    """Slow, differentiable CPU reference; no Metal/CUDA extension needed."""
    rows = []
    alphas = []
    tw = math.ceil(width / tile_size)
    # Keep empty scenes connected to all differentiable inputs.
    anchor = sum(x.sum() * 0 for x in (means, conics, colors, opacities))
    for y in range(height):
        row, alpha_row = [], []
        for x in range(width):
            T = means.new_ones(())
            color = background * 0 + anchor
            start, stop = ranges[(y // tile_size) * tw + x // tile_size].tolist()
            for idx in ids[start:stop].tolist():
                dx, dy = means[idx] - means.new_tensor((x + .5, y + .5))
                a, b, c = conics[idx]
                sigma = .5 * (a * dx.square() + c * dy.square()) + b * dx * dy
                opacity = (opacities[idx] * (-sigma).exp()).clamp_max(.999)
                if sigma.detach() < 0 or opacity.detach() < 1 / 255:
                    continue
                next_T = T * (1 - opacity)
                if next_T.detach() <= 1e-4:
                    break
                color = color + T * opacity * colors[idx]
                T = next_T
            row.append(color + T * background)
            alpha_row.append(1 - T + anchor)
        rows.append(torch.stack(row))
        alphas.append(torch.stack(alpha_row))
    return torch.stack(rows), torch.stack(alphas)


def rasterize_projected(means2d, conics, colors, opacities, radii, depths, background,
                        width, height, tile_size=16):
    """Composite projected splats; supports 1–4 feature channels and first gradients."""
    n = len(means2d)
    if means2d.shape != (n, 2) or conics.shape != (n, 3) or colors.ndim != 2 or colors.shape[0] != n:
        raise ValueError("expected centers=(N,2), conics=(N,3), colors=(N,D)")
    if opacities.shape != (n,) or radii.shape != (n, 2) or depths.shape != (n,):
        raise ValueError("invalid opacity, radius or depth shape")
    channels = colors.shape[1]
    if not 1 <= channels <= 4 or background.shape != (channels,):
        raise ValueError("rasterizer supports 1 to 4 channels and one matching background")
    tensors = (means2d, conics, colors, opacities, depths, background)
    if any(x.device != means2d.device or x.dtype != means2d.dtype for x in tensors):
        raise ValueError("floating inputs must share dtype and device")
    if means2d.device.type == "mps" and means2d.dtype != torch.float32:
        raise ValueError("Metal splatting requires float32")
    if radii.dtype != torch.int32 or radii.device != means2d.device:
        raise ValueError("radii must be int32 on the input device")
    ranges, ids = gaussian_tile_intersections(means2d, radii, depths, width, height, tile_size)
    args = [x.contiguous() for x in (means2d, conics, F.pad(colors, (0, 4 - channels)),
                                    opacities, F.pad(background, (0, 4 - channels)))]
    impl = _MetalSplat.apply if means2d.device.type == "mps" else _reference_splat
    rendered, alpha = impl(*args, ranges, ids, width, height, tile_size)
    return rendered[..., :channels], alpha[..., None]


def rasterization(means, quats, scales, opacities, colors, viewmats, Ks, width, height,
                  near_plane=.01, far_plane=1e10, radius_clip=0., eps2d=.3, sh_degree=None,
                  packed=False, tile_size=16, backgrounds=None, render_mode="RGB",
                  rasterize_mode="classic", distributed=False, camera_model="pinhole"):
    """GenIA's gsplat-compatible rendering entry point (classic pinhole cameras).

    ``packed`` controls upstream intermediates, not the rendered result; local
    metadata is always unpacked. Unsupported rendering models fail explicitly.
    """
    if rasterize_mode != "classic" or camera_model != "pinhole" or distributed:
        raise NotImplementedError("local splatting supports classic pinhole rendering on one device")
    if render_mode not in ("RGB", "D", "ED", "RGB+D", "RGB+ED"):
        raise ValueError("unsupported render_mode")
    n, cameras = len(means), len(viewmats)
    if means.shape != (n, 3) or quats.shape != (n, 4) or scales.shape != (n, 3) or opacities.shape != (n,):
        raise ValueError("invalid Gaussian parameter shapes")
    if viewmats.shape != (cameras, 4, 4) or Ks.shape != (cameras, 3, 3) or not cameras:
        raise ValueError("expected at least one viewmat=(C,4,4) and K=(C,3,3)")
    if sh_degree is not None and (not 0 <= sh_degree <= 4 or colors.shape != (n, (sh_degree + 1)**2, 3)):
        raise ValueError("expected SH coefficients (N,(degree+1)**2,3) with degree 0..4")
    if sh_degree is None and colors.shape != (n, 3):
        raise ValueError("expected colors=(N,3)")
    # GenIA decodes in mixed precision but rasterizes and optimizes in float32.
    if means.device.type == "mps":
        means, quats, scales, opacities, colors, viewmats, Ks = [x.float() for x in
            (means, quats, scales, opacities, colors, viewmats, Ks)]
    if backgrounds is None:
        backgrounds = means.new_zeros((cameras, 3))
    else:
        backgrounds = backgrounds.to(device=means.device, dtype=means.dtype)
        if backgrounds.shape == (3,):
            backgrounds = backgrounds.expand(cameras, 3)
        if backgrounds.shape != (cameras, 3):
            raise ValueError("backgrounds must have shape (C,3) or (3,)")
    images, alphas, projections = [], [], []
    for camera in range(cameras):
        xy, conics, depths, radii = project_gaussians(means, quats, scales, viewmats[camera], Ks[camera],
            width, height, opacities, eps2d, near_plane, far_plane, radius_clip)
        rgb = colors
        if sh_degree is not None:
            # Match gsplat's camera-center derivatives even when callers
            # differentiate matrix entries before applying rigid constraints.
            center = torch.linalg.inv(viewmats[camera].contiguous())[:3, 3]
            direction = F.normalize(means - center, dim=-1)
            rgb = (eval_sh(sh_degree, colors.transpose(-1, -2), direction) + .5).clamp_min(0)
        bg = backgrounds[camera]
        if render_mode in ("D", "ED"):
            features, bg = depths[:, None], bg[:1] * 0
        elif render_mode in ("RGB+D", "RGB+ED"):
            features, bg = torch.cat((rgb, depths[:, None]), -1), F.pad(bg, (0, 1))
        else:
            features = rgb
        output, alpha = rasterize_projected(xy, conics, features, opacities, radii, depths, bg,
                                            width, height, tile_size)
        if render_mode in ("ED", "RGB+ED"):
            output = torch.cat((output[..., :-1], output[..., -1:] / alpha.clamp_min(1e-10)), -1)
        images.append(output)
        alphas.append(alpha)
        projections.append((xy, conics, depths, radii))
    meta = {name: torch.stack([p[i] for p in projections]) for i, name in
            enumerate(("means2d", "conics", "depths", "radii"))}
    return torch.stack(images), torch.stack(alphas), meta
