"""Reductions missing from PyTorch's Metal backend."""
import torch


def _dim_nanmedian(x, dim, keepdim):
    scalar = x.ndim == 0
    if scalar:
        x = x.reshape(1)
    if not -x.ndim <= dim < x.ndim:
        raise IndexError("nanmedian dimension out of range")
    dim %= x.ndim
    if x.shape[dim] == 0:
        raise IndexError("nanmedian reduction dimension must have non-zero size")
    # PyTorch ascending sort places NaNs after all numbers, including +inf.
    # The half/bfloat conversion is exact and avoids unsupported sort dtypes.
    keys = x.detach().float() if x.dtype in (torch.float16, torch.bfloat16) else x.detach()
    order = keys.argsort(dim=dim, stable=True)
    count = (~x.isnan()).sum(dim=dim, keepdim=True)
    rank = ((count - 1) // 2).clamp_min(0)
    index = order.gather(dim, rank)
    result = x.gather(dim, index)
    if scalar:
        return result.reshape(())
    return result if keepdim else result.squeeze(dim)


class _GlobalNanmedian(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        value = _dim_nanmedian(x.flatten(), 0, False) if x.numel() else x.new_full((), torch.nan)
        ctx.save_for_backward(x, value)
        return value

    @staticmethod
    def backward(ctx, grad):
        x, value = ctx.saved_tensors
        # The global reduction shares gradients among equal medians (and all
        # NaNs for an all-NaN input); dimensional median selects one index.
        mask = (x == value) | (x.isnan() & value.isnan())
        return mask.to(grad.dtype) * (grad / mask.sum())


def nanmedian(x: torch.Tensor, dim=None, keepdim=False) -> torch.Tensor:
    """Lower median ignoring NaNs, returning values, on CPU/CUDA/Metal.

    As with torch.nanmedian, a dimensional reduction may choose any tied median
    for its derivative; this implementation uses stable input order on Metal.
    """
    if x.device.type != 'mps':
        return torch.nanmedian(x) if dim is None else torch.nanmedian(x, dim=dim, keepdim=keepdim).values
    if dim is None:
        return _GlobalNanmedian.apply(x)
    return _dim_nanmedian(x, dim, keepdim)
