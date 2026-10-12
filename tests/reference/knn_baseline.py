"""Differentiable KNN with Metal selection and ordinary PyTorch gradient paths.

Distances follow PyTorch3D's convention: squared L2 (or L1), zero padding for
missing neighbors/queries, and int64 indices. No all-pairs GPU tensor is built.
"""

from collections import namedtuple

import torch

from genia.core.ops.metal import library

KNN = namedtuple("KNN", "dists idx knn")


def _lengths(value, batch, count, device):
    if value is None:
        return torch.full((batch,), count, dtype=torch.int32, device=device)
    if value.shape != (batch,) or value.dtype not in (torch.int32, torch.int64):
        raise ValueError("lengths must be an integer tensor of shape (batch,)")
    if bool(((value < 0) | (value > count)).any()):
        raise ValueError("lengths must lie within the padded point count")
    return value.to(device=device, dtype=torch.int32).contiguous()


def knn_gather(x, idx, lengths=None):
    """Gather (B, P, K) indices from (B, Q, D), zeroing padded neighbors."""
    if x.ndim != 3 or idx.ndim != 3 or x.shape[0] != idx.shape[0]:
        raise ValueError("expected x=(B,Q,D), idx=(B,P,K)")
    if idx.device != x.device or idx.dtype != torch.int64:
        raise ValueError("indices must be int64 and on the input device")
    b, p, k = idx.shape
    lens = _lengths(lengths, b, x.shape[1], x.device)
    if x.shape[1] == 0:
        return x.sum() * 0 + x.new_zeros((b, p, k, x.shape[-1]))
    out = x[torch.arange(b, device=x.device)[:, None, None], idx]
    valid = torch.arange(k, device=x.device)[None, None, :] < lens[:, None, None]
    return out.masked_fill(~valid[..., None], 0)


def knn_points(p1, p2, lengths1=None, lengths2=None, norm=2, K=1,
               version=-1, return_nn=False, return_sorted=True):
    """K nearest points, with piecewise gradients to both input clouds.

    Metal supports K <= 32 and float32 selection. Half/bfloat16 inputs are
    promoted for selection and distances. CPU float64 is kept for gradcheck.
    ``version`` is accepted for API compatibility; results are always sorted.
    """
    if p1.ndim != 3 or p2.ndim != 3 or p1.shape[0] != p2.shape[0] or p1.shape[2] != p2.shape[2]:
        raise ValueError("point clouds must have shapes (B,P,D) and (B,Q,D)")
    if p1.device != p2.device or p1.dtype != p2.dtype or not p1.is_floating_point():
        raise ValueError("point clouds must have the same floating dtype and device")
    if norm not in (1, 2) or not isinstance(K, int) or K < 1:
        raise ValueError("norm must be 1 or 2 and K must be a positive integer")
    if p1.device.type == "mps" and K > 32:
        raise ValueError("Metal KNN supports K <= 32")
    b, p, d = p1.shape
    q = p2.shape[1]
    l1 = _lengths(lengths1, b, p, p1.device)
    l2 = _lengths(lengths2, b, q, p1.device)
    a = p1.float() if p1.dtype in (torch.float16, torch.bfloat16) else p1
    z = p2.float() if p2.dtype in (torch.float16, torch.bfloat16) else p2
    with torch.no_grad():
        if p1.device.type == "mps" and b * p and q:
            idx32 = torch.empty((b, p, K), device=p1.device, dtype=torch.int32)
            library("knn").knn_indices(a.contiguous(), z.contiguous(), l1, l2,
                                       idx32, p, q, d, K, norm, threads=b * p)
            idx = idx32.long()
        else:
            idx = torch.zeros((b, p, K), device=p1.device, dtype=torch.int64)
            for bi in range(b):
                n = min(K, int(l2[bi]))
                if not n:
                    continue
                # Bound reference implementation working memory as well.
                chunk = max(1, 1048576 // max(q * d, 1))
                for start in range(0, int(l1[bi]), chunk):
                    stop = min(start + chunk, int(l1[bi]))
                    delta = a[bi, start:stop, None] - z[bi, None, :int(l2[bi])]
                    distances = (delta.square() if norm == 2 else delta.abs()).sum(-1)
                    idx[bi, start:stop, :n] = distances.argsort(stable=True)[:, :n]
    valid = ((torch.arange(p, device=a.device)[None, :, None] < l1[:, None, None]) &
             (torch.arange(K, device=a.device)[None, None, :] < l2[:, None, None]))
    idx = idx.masked_fill(~valid, 0)
    neighbors = knn_gather(z, idx, l2)
    delta = a[:, :, None] - neighbors
    distances = (delta.square() if norm == 2 else delta.abs()).sum(-1).masked_fill(~valid, 0)
    nn = knn_gather(p2, idx, l2).masked_fill(~valid[..., None], 0) if return_nn else None
    return KNN(distances, idx, nn)
