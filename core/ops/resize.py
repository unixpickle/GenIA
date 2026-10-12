"""Antialiased image resizing on Metal, with a PyTorch-compatible entry point."""
import math

import torch
import torch.nn.functional as F
from torch.autograd.function import once_differentiable

from .metal import library


def _axis(x, source, target, inner, scale, cubic, backward=False):
    # Match PyTorch's per-axis forward shortcut. Its backward still uses the
    # requested scale when rounding gives the same input and output sizes.
    if source == target and not backward:
        return x.contiguous().clone()
    if backward:
        count = x.numel() // (target * inner)
        out = x.new_empty(count * source * inner)
        kernel = library("resize").resize_axis_backward
    else:
        count = x.numel() // (source * inner)
        out = x.new_empty(count * target * inner)
        kernel = library("resize").resize_axis
    if out.numel():
        kernel(x.contiguous(), out, source, target, inner, float(scale), cubic, threads=out.numel())
    return out


class _ResizeAA(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, height, width, sy, sx, cubic):
        n, c, h, w = x.shape
        # Each axis is contiguous conceptually; 'inner' sets its element stride.
        horizontal = _axis(x, w, width, 1, sx, cubic).reshape(n, c, h, width)
        vertical = _axis(horizontal, h, height, width, sy, cubic)
        ctx.params = x.shape, height, width, sy, sx, cubic
        return vertical.reshape(n, c, height, width)

    @staticmethod
    @once_differentiable
    def backward(ctx, grad):
        (n, c, h, w), height, width, sy, sx, cubic = ctx.params
        vertical = _axis(grad, h, height, width, sy, cubic, backward=True)
        horizontal = _axis(vertical, w, width, 1, sx, cubic, backward=True)
        return horizontal.reshape(n, c, h, w), None, None, None, None, None


def interpolate(input, size=None, scale_factor=None, mode="nearest", align_corners=None,
                recompute_scale_factor=None, antialias=False):
    """Use Metal for antialiased bilinear/bicubic NCHW; native torch otherwise.

    MPS half/bfloat16 inputs accumulate in float32 and retain their output dtype.
    Both explicit sizes and scale factors retain PyTorch's pixel-center mapping.
    """
    if input.device.type == 'mps' and mode == 'area' and input.ndim == 4 and not antialias:
        from .pooling import adaptive_avg_pool2d
        if align_corners is not None:
            raise ValueError('align_corners is not supported for area interpolation')
        if (size is None) == (scale_factor is None):
            raise ValueError('specify exactly one of size or scale_factor')
        if size is not None and recompute_scale_factor:
            raise ValueError('recompute_scale_factor is not meaningful with explicit size')
        if size is None:
            scales = (scale_factor, scale_factor) if isinstance(scale_factor, (int, float)) else scale_factor
            size = tuple(math.floor(dim * scale) for dim, scale in zip(input.shape[-2:], scales))
        return adaptive_avg_pool2d(input, size)
    if input.device.type != "mps" or not antialias or mode not in ("bilinear", "bicubic"):
        return F.interpolate(input, size=size, scale_factor=scale_factor, mode=mode,
            align_corners=align_corners, recompute_scale_factor=recompute_scale_factor, antialias=antialias)
    if input.ndim != 4 or input.dtype not in (torch.float16, torch.bfloat16, torch.float32):
        raise ValueError("Metal antialiased interpolation expects floating NCHW input")
    if (size is None) == (scale_factor is None):
        raise ValueError("specify exactly one of size or scale_factor")
    h, w = input.shape[-2:]
    if not h or not w:
        raise ValueError("input spatial dimensions must be positive")
    if size is not None:
        if recompute_scale_factor:
            raise ValueError("recompute_scale_factor is not meaningful with explicit size")
        oh, ow = (size, size) if isinstance(size, int) else size
        sy, sx = h / oh if oh else 0, w / ow if ow else 0
    else:
        fy, fx = (scale_factor, scale_factor) if isinstance(scale_factor, (int, float)) else scale_factor
        if fy <= 0 or fx <= 0:
            raise ValueError("scale factors must be positive")
        oh, ow = math.floor(h * fy), math.floor(w * fx)
        sy, sx = (h / oh if oh else 0, w / ow if ow else 0) if recompute_scale_factor else (1 / fy, 1 / fx)
    if oh < 1 or ow < 1:
        raise ValueError("output dimensions must be positive")
    if align_corners:
        sy, sx = ((h - 1) / (oh - 1) if oh > 1 else 0, (w - 1) / (ow - 1) if ow > 1 else 0)
        # PyTorch's AA filter keeps half-pixel centers even for align_corners.
    if (oh, ow) == (h, w):
        return input.clone()
    return _ResizeAA.apply(input.float(), oh, ow, sy, sx, int(mode == "bicubic")).to(input.dtype)
