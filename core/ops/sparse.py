"""Submanifold sparse convolution with a Metal coordinate hash table.

Coordinates are (batch, x, y, z). Weights retain spconv's checkpoint layout
(out_channels, kx, ky, kz, in_channels). Only occupied output sites are computed.
"""

from itertools import product

import torch

from .metal import library


def triple(value):
    result = (value,) * 3 if isinstance(value, int) else tuple(value)
    if len(result) != 3 or any(not isinstance(v, int) or v < 1 for v in result):
        raise ValueError("expected three positive integers")
    return result


def neighbor_indices(coords, kernel_size=3, dilation=1):
    """Return (kernel_volume, N) neighbors, using -1 for unoccupied sites.

    Coordinates must be unique int32 sites; negative spatial positions are valid.
    The returned discrete map can be reused across convolutions on the same grid.
    """
    kernel_size, dilation = triple(kernel_size), triple(dilation)
    if any(k % 2 == 0 for k in kernel_size):
        raise ValueError("submanifold kernels must have odd dimensions")
    if coords.ndim != 2 or coords.shape[1] != 4 or coords.dtype != torch.int32:
        raise ValueError("coords must be int32 with shape (N,4)")
    offsets = [(0, *[(i - k // 2) * d for i, k, d in zip(pos, kernel_size, dilation)])
               for pos in product(*(range(k) for k in kernel_size))]
    n, volume = len(coords), len(offsets)
    if not n:
        return torch.empty((volume, 0), dtype=torch.int64, device=coords.device)
    if coords.device.type == "mps":
        capacity = 1 << (2 * n - 1).bit_length()
        table = torch.full((capacity,), -1, dtype=torch.int32, device=coords.device)
        result = torch.empty((volume, n), dtype=torch.int32, device=coords.device)
        coords = coords.contiguous()
        off = torch.tensor(offsets, dtype=torch.int32, device=coords.device)
        lib = library("sparse")
        lib.build_table(coords, table, capacity, threads=n)
        lib.neighbor_table(coords, off, table, result, n, capacity, threads=n * volume)
        return result.long()
    rows = coords.cpu().tolist()
    lookup = {tuple(c): i for i, c in enumerate(rows)}
    if len(lookup) != n:
        raise ValueError("sparse coordinates must be unique")
    result = [[lookup.get(tuple(a + b for a, b in zip(c, off)), -1) for c in rows]
              for off in offsets]
    return torch.tensor(result, dtype=torch.int64, device=coords.device)


def submanifold_conv3d(features, weight, neighbors, bias=None):
    """Apply a neighbor map with autograd through features, weights, and bias."""
    if features.ndim != 2 or weight.ndim != 5 or features.shape[1] != weight.shape[-1]:
        raise ValueError("expected features=(N,Cin), weight=(Cout,Kx,Ky,Kz,Cin)")
    cout, kx, ky, kz, cin = weight.shape
    if neighbors.shape != (kx * ky * kz, len(features)):
        raise ValueError("neighbor map shape does not match features and kernel")
    if neighbors.dtype != torch.int64 or neighbors.device != features.device:
        raise ValueError("neighbors must be int64 on the feature device")
    if weight.device != features.device or weight.dtype != features.dtype:
        raise ValueError("weights and features must have matching dtype/device")
    if bias is not None and (bias.shape != (cout,) or bias.device != features.device or bias.dtype != features.dtype):
        raise ValueError("bias must have shape (Cout,) and match features dtype/device")
    if features.device.type == "mps" and len(features) and cin:
        # Gather all offsets into the GEMM reduction dimension. Bound each
        # temporary to 32 MiB for small channels, 256 MiB for wide decoder layers.
        # index_select retains the supported half/bfloat16 gradient path.
        budget = (8 if cin <= 128 else 64) * 1024 * 1024
        rows = max(1, budget // (neighbors.shape[0] * cin))
        padded = torch.cat([features.new_zeros((1, cin)), features])
        matrix = weight.reshape(cout, -1).T
        outputs = []
        for start in range(0, len(features), rows):
            indices = (neighbors[:, start:start + rows].T + 1).reshape(-1)
            gathered = padded.index_select(0, indices).reshape(-1, matrix.shape[0])
            outputs.append((gathered @ matrix).to(features.dtype))
        out = torch.cat(outputs)
    else:
        out = _submanifold_conv3d_reference(features, weight, neighbors)
    return out if bias is None else out + bias


def _submanifold_conv3d_reference(features, weight, neighbors):
    """Original offset-by-offset implementation, also used for CPU/empty inputs."""
    cin, cout = features.shape[1], weight.shape[0]
    padded = torch.cat([features.new_zeros((1, cin)), features])
    kernels = weight.reshape(cout, -1, cin)
    out = features.new_zeros((len(features), cout))
    for k in range(neighbors.shape[0]):
        out = out + padded.index_select(0, neighbors[k] + 1) @ kernels[:, k].T
    return out
