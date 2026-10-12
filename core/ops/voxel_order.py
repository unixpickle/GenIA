"""30-bit Morton/Hilbert voxel codes, compatible with TRELLIS vox2seq.

Coordinates have ten bits per axis. Adapted from Microsoft TRELLIS (MIT);
see vendor/licenses/vox2seq.LICENSE. Integer ordering is nondifferentiable.
"""
import torch
from .metal import library


def _validate(x, permute, mode, encode):
    if sorted(permute) != [0, 1, 2]:
        raise ValueError("permute must be a permutation of [0, 1, 2]")
    if mode not in {"z_order", "hilbert"}:
        raise ValueError(f"Unknown encoding mode: {mode}")
    if x.dtype not in {torch.int32, torch.int64}:
        raise TypeError("voxel coordinates/codes must be int32 or int64")
    if (encode and (x.ndim != 2 or x.shape[1] != 3)) or (not encode and x.ndim != 1):
        raise ValueError("expected coordinates (N, 3) or codes (N,)")
    limit = 1024 if encode else 1 << 30
    if x.numel() and bool(((x < 0) | (x >= limit)).any()):
        raise ValueError(f"values must be in [0, {limit})")


@torch.no_grad()
def encode(coords, permute=(0, 1, 2), mode="z_order"):
    _validate(coords, permute, mode, True)
    coords = coords[:, list(permute)].to(torch.int32).contiguous()
    n = len(coords)
    if coords.device.type == "mps":
        out = torch.empty(n, dtype=torch.int32, device=coords.device)
        if n:
            library("voxel_order").encode_voxels(coords, out, int(mode == "hilbert"), threads=n)
        return out
    point = list(coords.unbind(1))
    if mode == "hilbert":
        for bit in range(9, 0, -1):
            q, p = 1 << bit, (1 << bit) - 1
            for i in range(3):
                invert = (point[i] & q) != 0
                t = torch.where(invert, 0, (point[0] ^ point[i]) & p)
                point[0] = point[0] ^ torch.where(invert, p, t)
                point[i] = point[i] ^ t
        point[1] = point[1] ^ point[0]
        point[2] = point[2] ^ point[1]
        t = torch.zeros_like(point[0])
        for bit in range(9, 0, -1):
            t = t ^ torch.where((point[2] & (1 << bit)) != 0, (1 << bit) - 1, 0)
        point = [p ^ t for p in point]
    out = torch.zeros(n, dtype=torch.int32, device=coords.device)
    for axis in range(3):
        for bit in range(10):
            out |= ((point[axis] >> bit) & 1) << (3 * bit + 2 - axis)
    return out


@torch.no_grad()
def decode(code, permute=(0, 1, 2), mode="z_order"):
    _validate(code, permute, mode, False)
    code = code.to(torch.int32).contiguous()
    n = len(code)
    if code.device.type == "mps":
        out = torch.empty((n, 3), dtype=torch.int32, device=code.device)
        if n:
            library("voxel_order").decode_voxels(code, out, int(mode == "hilbert"), threads=n)
    else:
        point = [torch.zeros_like(code) for _ in range(3)]
        for axis in range(3):
            for bit in range(10):
                point[axis] |= ((code >> (3 * bit + 2 - axis)) & 1) << bit
        if mode == "hilbert":
            t = point[2] >> 1
            point[2] = point[2] ^ point[1]
            point[1] = point[1] ^ point[0]
            point[0] = point[0] ^ t
            for bit in range(1, 10):
                q, p = 1 << bit, (1 << bit) - 1
                for i in range(2, -1, -1):
                    invert = (point[i] & q) != 0
                    t = torch.where(invert, 0, (point[0] ^ point[i]) & p)
                    point[0] = point[0] ^ torch.where(invert, p, t)
                    point[i] = point[i] ^ t
        out = torch.stack(point, -1)
    return out[:, [permute.index(i) for i in range(3)]]
