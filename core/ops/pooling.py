"""Metal-compatible stride-one pooling for dense scalar volumes."""
import torch
import torch.nn.functional as F
from torch.autograd.function import once_differentiable
from .metal import library


class _AreaPool2d(torch.autograd.Function):
    @staticmethod
    def forward(ctx, image, oh, ow):
        n, c, ih, iw = image.shape
        out = image.new_empty((n, c, oh, ow))
        ctx.shape = image.shape
        if out.numel():
            library('pooling').area_pool2d(image.contiguous(), out, ih, iw, oh, ow, n*c,
                                           threads=out.numel())
        return out

    @staticmethod
    @once_differentiable
    def backward(ctx, grad):
        out = grad.new_empty(ctx.shape)
        ih, iw = ctx.shape[-2:]
        oh, ow = grad.shape[-2:]
        if out.numel():
            library('pooling').area_pool2d_backward(grad.contiguous(), out, ih, iw, oh, ow, ctx.shape[0]*ctx.shape[1],
                                                    threads=out.numel())
        return out, None, None


def adaptive_avg_pool2d(image, output_size):
    """Native Metal adaptive area pooling, including nondivisible dimensions."""
    if image.device.type != 'mps':
        return F.adaptive_avg_pool2d(image, output_size)
    if image.ndim not in (3, 4) or image.dtype not in (torch.float16, torch.bfloat16, torch.float32):
        raise ValueError('Metal area pooling expects floating CHW or NCHW input')
    sizes = (output_size, output_size) if isinstance(output_size, int) else output_size
    oh, ow = [old if new is None else new for old, new in zip(image.shape[-2:], sizes)]
    if min(*image.shape[-2:], oh, ow) < 1:
        raise ValueError('area pooling spatial dimensions must be positive')
    x = image.unsqueeze(0) if image.ndim == 3 else image
    out = _AreaPool2d.apply(x.float(), oh, ow).to(image.dtype)
    return out.squeeze(0) if image.ndim == 3 else out


def max_pool3d_stride1(volume, kernel_size):
    """Same-size odd cubic max pooling of an (X,Y,Z) volume, with gradients.

    A cubic maximum is separable. Two native 2D pools avoid PyTorch 2.7's
    unavailable MPS max_pool3d kernel. When gradients are enabled, a Metal
    argmax keeps CPU's tie convention (native MPS pooling uses different ties).
    """
    if volume.ndim != 3 or kernel_size < 1 or kernel_size % 2 != 1:
        raise ValueError('expected an (X,Y,Z) volume and positive odd kernel size')
    pad = kernel_size // 2
    if volume.device.type == 'mps' and volume.requires_grad and torch.is_grad_enabled():
        indices = torch.empty(volume.numel(), device=volume.device, dtype=torch.long)
        library('pooling').volume_argmax(volume.detach().float().contiguous(), indices,
            *volume.shape, pad, threads=volume.numel())
        return volume.reshape(-1).index_select(0, indices).reshape(volume.shape)
    yz = F.max_pool2d(volume[None].contiguous(), kernel_size, stride=1, padding=pad)
    xyz = F.max_pool2d(yz.permute(0, 2, 1, 3).contiguous(), (kernel_size, 1),
                       stride=1, padding=(pad, 0))
    return xyz.permute(0, 2, 1, 3)[0]
