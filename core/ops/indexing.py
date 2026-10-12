"""Exact large-volume index extraction on Metal."""
import torch
from .metal import library


def flat_nonzero(values):
    """Return ordered int64 indices of nonzero elements in flattened order.

    PyTorch 2.7's MPS nonzero stores coordinates through float32, which rounds
    one-dimensional indices above 2**24. Keep local coordinates within the
    exactly representable range, then add each chunk's offset as int64.
    """
    flat = values.reshape(-1)
    chunk_size = 1 << 24
    if values.device.type != 'mps' or flat.numel() <= chunk_size:
        return flat.nonzero().flatten()
    return torch.cat([
        flat[start:start + chunk_size].nonzero().flatten() + start
        for start in range(0, flat.numel(), chunk_size)
    ])


def unravel_indices(indices, shape):
    """Convert nonnegative int64 linear indices to exact integer coordinates.

    MPS integer division/remainder in torch 2.7 also rounds through float32.
    Use Metal integer arithmetic before coordinates are converted to geometry.
    """
    if indices.dtype != torch.long or not shape or any(size <= 0 for size in shape):
        raise ValueError('expected int64 indices and positive dimensions')
    if indices.device.type != 'mps':
        result = []
        remaining = indices
        for size in reversed(shape):
            result.append(remaining % size)
            remaining = remaining // size
        return torch.stack(result[::-1], dim=-1)
    result = torch.empty((*indices.shape, len(shape)), dtype=torch.long, device=indices.device)
    if indices.numel():
        dims = torch.tensor(shape, dtype=torch.long, device=indices.device)
        library('indexing').unravel_indices(indices.contiguous(), dims, result,
            len(shape), indices.numel(), threads=indices.numel())
    return result
