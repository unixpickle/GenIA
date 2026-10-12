"""Farthest-point selection on Metal, with native autograd for gathered values.

Follows PyTorch3D v0.7.9 CUDA sampling (including its 1e10 distance cap).
See vendor/licenses/pytorch3d.LICENSE for upstream attribution.
"""
import torch
from .metal import library


def masked_gather(points, idx):
    if points.ndim != 3 or idx.ndim not in (2, 3) or len(points) != len(idx):
        raise ValueError("expected points (B, P, D) and indices (B, K) or (B, Q, K)")
    b, p, d = points.shape
    idx = idx.to(device=points.device, dtype=torch.long)
    if idx.numel() and bool(((idx < -1) | (idx >= p)).any()):
        raise IndexError("point index outside [-1, P)")
    shape = (*idx.shape, d)
    if p == 0:
        if idx.numel() and bool((idx != -1).any()):
            raise ValueError("only padding indices are valid for empty point clouds")
        return points.sum(1).reshape(b, *([1] * (idx.ndim - 1)), d).expand(shape) * 0
    offset = torch.arange(b, device=points.device).reshape(b, *([1] * (idx.ndim - 1))) * p
    gathered = points.reshape(b * p, d).index_select(0, (idx.clamp_min(0) + offset).flatten()).reshape(shape)
    return torch.where((idx != -1)[..., None], gathered, 0)


def sample_farthest_points(points, lengths=None, K=50, random_start_point=False):
    if points.ndim != 3 or not points.is_floating_point() or points.shape[-1] == 0:
        raise ValueError("points must be a floating tensor (B, P, D), with D > 0")
    b, p, d = points.shape
    points = points.float()  # PyTorch3D's public API selects and returns float32.
    lengths = (torch.full((b,), p, dtype=torch.long, device=points.device) if lengths is None
               else torch.as_tensor(lengths, dtype=torch.long, device=points.device))
    counts = (torch.full((b,), K, dtype=torch.long, device=points.device) if isinstance(K, int)
              else torch.as_tensor(K, dtype=torch.long, device=points.device))
    if lengths.shape != (b,) or counts.shape != (b,):
        raise ValueError("lengths and K must have one entry per batch")
    if bool(((lengths < 0) | (lengths > p)).any()) or bool((counts < 0).any()):
        raise ValueError("invalid lengths or negative sample count")
    max_k = K if isinstance(K, int) else (int(counts.max()) if b else 0)
    indices = torch.full((b, max_k), -1, device=points.device, dtype=torch.long)
    with torch.no_grad():
        starts = ((torch.rand(b, device=points.device) * lengths).long() if random_start_point
                  else torch.zeros(b, dtype=torch.long, device=points.device))
        if b and p and max_k:
            if points.device.type == "mps":
                distances = torch.empty((b, p), dtype=torch.float32, device=points.device)
                library("fps").farthest_points(points.detach().contiguous(), lengths.contiguous(),
                    counts.contiguous(), starts, distances, indices, p, d, max_k,
                    threads=b * 256, group_size=256)
            else:
                for batch in range(b):
                    n, k = int(lengths[batch]), min(int(counts[batch]), int(lengths[batch]))
                    if not k:
                        continue
                    selected = int(starts[batch])
                    indices[batch, 0] = selected
                    closest = points.new_full((n,), 1e10)
                    for sample in range(1, k):
                        dist = ((points[batch, :n] - points[batch, selected]) ** 2).sum(-1)
                        closest = torch.minimum(closest, dist)
                        selected = int(closest.argmax())
                        indices[batch, sample] = selected
    return masked_gather(points, indices), indices
