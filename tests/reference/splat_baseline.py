"""Original per-pixel Metal compositor, retained for optimization comparisons."""
import math
import torch
from torch.autograd.function import once_differentiable
from genia.core.ops.metal import library

class _MetalSplat(torch.autograd.Function):
    @staticmethod
    def forward(ctx, means, conics, colors, opacities, background, ranges, ids, width, height, tile_size):
        output = colors.new_empty((height, width, 4))
        trans = colors.new_empty((height, width))
        last = torch.empty((height, width), dtype=torch.int32, device=colors.device)
        tw = math.ceil(width / tile_size)
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
        library("gaussian").splat_backward(means, conics, colors, opacities, background, ranges, ids,
                                           trans, last, grad_output.contiguous(), grad_alpha.contiguous(),
                                           *grads, *ctx.parameters, threads=trans.numel())
        grad_background = (grad_output * trans[..., None]).sum((0, 1))
        return *grads, grad_background, None, None, None, None, None

