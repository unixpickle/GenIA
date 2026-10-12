"""DISO 0.1.4 dual marching cubes, with Metal topology and native autograd.

Preserves manifold ambiguity resolution, patch/edge ordering, quad winding and
the original minimum-angle triangulation. See vendor/licenses/diso.LICENSE.
"""
from functools import lru_cache

import torch
from torch import nn
from torch.nn import functional as F

from .metal import library
from .indexing import flat_nonzero, unravel_indices
from .dual_marching_cubes_tables import (
    CORNERS, EDGES, FIRST_PATCH, FIRST_EDGE, EDGE_INDEX, PROBLEMATIC, EDGE_OFFSET, QUAD,
)


@lru_cache(None)
def _tables(device):
    return tuple(torch.tensor(t, device=device, dtype=torch.int32) for t in
                 (CORNERS, EDGES, FIRST_PATCH, FIRST_EDGE, EDGE_INDEX, PROBLEMATIC, EDGE_OFFSET, QUAD))


@torch.no_grad()
def _topology(grid, iso):
    nx, ny, nz = grid.shape
    corners, edges, first_patch, first_edge, edge_index, problematic, edge_offset, quad = _tables(grid.device)
    if grid.device.type == 'mps':
        codes = torch.empty(grid.numel(), device=grid.device, dtype=torch.uint8)
        bits = torch.empty_like(codes)
        library('dual_marching_cubes').classify_dual_cells(
            grid.contiguous(), corners, codes, bits, nx, ny, nz, float(iso), threads=grid.numel())
    else:
        codes = torch.zeros_like(grid, dtype=torch.uint8)
        bits = torch.zeros_like(codes)
        for c, (x, y, z) in enumerate(CORNERS):
            codes[:-1, :-1, :-1] |= (grid[x:x+nx-1, y:y+ny-1, z:z+nz-1] >= iso).to(torch.uint8) << c
        a = grid[:-1, :-1, :-1]
        for axis in range(3):
            other = [slice(0, -1)] * 3
            other[axis] = slice(1, None)
            b = grid[tuple(other)]
            bits[:-1, :-1, :-1] |= (((a < iso) & (b >= iso)) | ((b < iso) & (a >= iso))).to(torch.uint8) << axis
        codes, bits = codes.flatten(), bits.flatten()
    cells = flat_nonzero((codes > 0) & (codes < 255))
    good = torch.empty(len(cells), device=grid.device, dtype=torch.int32)
    counts = torch.empty_like(good)
    if len(cells) and grid.device.type == 'mps':
        library('dual_marching_cubes').resolve_dual_cells(
            cells, codes, problematic, first_patch, good, counts, nx, ny, nz, len(cells), threads=len(cells))
    else:
        good = codes[cells].long()
        direction = problematic[good].long()
        axis = (direction >> 1).clamp_max(2)
        delta = (direction & 1) * 2 - 1
        stride = torch.tensor([ny*nz, nz, 1], device=grid.device)
        xyz = torch.stack([cells//(ny*nz), (cells//nz) % ny, cells % nz], -1)
        next_coord = xyz.gather(1, axis[:, None]).flatten() + delta
        neighbor = (cells + delta*stride[axis]).clamp(0, grid.numel()-1)
        dims = torch.tensor(grid.shape, device=grid.device)
        flip = ((direction != 255) & (next_coord >= 0) & (next_coord < dims[axis]-1)
                & (problematic[codes[neighbor].long()] != 255))
        good = good ^ (flip.long()*255)
        counts = first_patch[good+1] - first_patch[good]
    first = counts.cumsum(0) - counts
    slot = torch.arange(4, device=grid.device)
    valid = slot[None] < counts[:, None]
    patch_cells = cells[:, None].expand(-1, 4)[valid]
    patches = (first_patch[good.long(), None] + slot[None])[valid].long()
    selected = flat_nonzero(((bits[cells, None].long() >> torch.arange(3, device=grid.device)) & 1) > 0)
    if len(cells):
        used, axes = unravel_indices(selected, (len(cells), 3)).unbind(-1)
    else:
        used = axes = selected
    origins = cells[used]
    quads = torch.empty((len(origins), 4), device=grid.device, dtype=torch.long)
    cell_to_used = torch.empty(grid.numel(), device=grid.device, dtype=torch.int32)
    cell_to_used[cells] = torch.arange(len(cells), device=grid.device, dtype=torch.int32)
    if len(origins) and grid.device.type == 'mps':
        library('dual_marching_cubes').emit_dual_quads(
            origins.contiguous(), axes.contiguous(), grid.contiguous(), cell_to_used, good,
            first, quad, edge_offset, quads, ny, nz, float(iso), len(origins), threads=len(origins))
    elif len(origins):
        types = axes + (grid.flatten()[origins] >= iso).long()*3
        q = quad[types].long()
        neighbor = origins[:, None] + q[..., 0]*ny*nz + q[..., 1]*nz + q[..., 2]
        used = cell_to_used[neighbor].long()
        quads = first[used] + edge_offset[good[used], q[..., 3]]
    return patch_cells, patches, quads


def _vertices(grid, deform, cells, patches, iso):
    _, ny, nz = grid.shape
    _, edges, _, first_edge, edge_index, *_ = _tables(grid.device)
    slots = torch.arange(12, device=grid.device)
    count = (first_edge[patches+1] - first_edge[patches]).long()
    valid = slots[None] < count[:, None]
    at = first_edge[patches, None] + slots[None]
    e = edges[edge_index[at.clamp_max(len(edge_index)-1)].long()].long()
    origin = cells[:, None] + e[..., 0]*ny*nz + e[..., 1]*nz + e[..., 2]
    axis = e[..., 3]
    stride = torch.tensor([ny*nz, nz, 1], device=grid.device)
    other = origin + stride[axis]
    p0 = unravel_indices(origin, grid.shape).to(grid.dtype)
    p1 = p0 + F.one_hot(axis, 3).to(grid.dtype)
    if deform is not None:
        p0 = p0 + deform[origin]
        p1 = p1 + deform[other]
    a, b = grid.flatten()[origin], grid.flatten()[other]
    denominator = b-a
    t = (iso-a) / torch.where(denominator == 0, 1, denominator)
    # DISO's scalar clamp maps NaN to zero. Equal endpoints use the midpoint.
    t = torch.where(denominator == 0, .5, torch.nan_to_num(t, nan=0.)).clamp(0, 1)
    positions = p0 + (p1-p0)*t[..., None]
    return torch.where(valid[..., None], positions, 0).sum(1) / count[:, None]


@torch.no_grad()
def _triangulate(vertices, quads):
    scores = []
    for pattern in (((0, 1, 3), (1, 2, 3)), ((0, 1, 2), (0, 2, 3))):
        angles = []
        for face in pattern:
            v0, v1, v2 = vertices[quads[:, list(face)]].unbind(-2)
            cosines = []
            for a, b, c in ((v0, v1, v2), (v1, v2, v0), (v2, v0, v1)):
                cosines.append((F.normalize(b-a, dim=-1)*F.normalize(c-a, dim=-1)).sum(-1))
            angles.append(torch.stack(cosines, -1).max(-1).values)
        scores.append(torch.stack(angles, -1).max(-1).values)
    use_first = scores[0] < scores[1]
    return torch.cat((quads[use_first][:, [0, 1, 3, 1, 2, 3]].reshape(-1, 3),
                      quads[~use_first][:, [0, 1, 2, 0, 2, 3]].reshape(-1, 3)))


class DiffDMC(nn.Module):
    """Extract DISO dual vertices and triangles (or quads) from an (X,Y,Z) grid."""
    def __init__(self, dtype=torch.float32):
        super().__init__()
        if dtype not in (torch.float32, torch.float64):
            raise ValueError('DiffDMC supports float32 and CPU float64')
        self.dtype = dtype

    def forward(self, grid, deform=None, isovalue=0., return_quads=False, normalize=True):
        if grid.ndim != 3 or min(grid.shape) < 2:
            raise ValueError('grid must be (X,Y,Z) with each dimension at least 2')
        if grid.dtype != self.dtype:
            raise ValueError('grid dtype must match DiffDMC dtype')
        if deform is not None and (deform.shape != (*grid.shape, 3) or deform.device != grid.device or deform.dtype != grid.dtype):
            raise ValueError('deform must have shape (*grid.shape,3), matching grid dtype/device')
        if grid.min() >= isovalue or grid.max() <= isovalue:
            anchor = grid.sum()*0 + (0 if deform is None else deform.sum()*0)
            # Upstream returns four columns on empty input even in triangle mode.
            return grid.new_zeros((0, 3))+anchor, torch.empty((0, 4), device=grid.device, dtype=torch.long)
        padded = F.pad(grid, (1, 1, 1, 1, 1, 1), value=float(isovalue)+1)
        pd = None if deform is None else F.pad(deform.permute(3, 0, 1, 2), (1, 1, 1, 1, 1, 1)).permute(1, 2, 3, 0).reshape(-1, 3)
        cells, patches, quads = _topology(padded, isovalue)
        pieces = [_vertices(padded, pd, cells[start:start+65536], patches[start:start+65536], isovalue)
                  for start in range(0, len(cells), 65536)]
        vertices = torch.cat(pieces) - 1 if pieces else grid.new_empty((0, 3))
        if normalize:
            vertices = vertices / (grid.new_tensor(grid.shape)-1)
        return vertices, quads if return_quads else _triangulate(vertices, quads)
