"""DISO 0.1.4 differentiable marching cubes with local Metal topology kernels.

Table order, threshold equality, boundary padding and normalization retain the
upstream conventions. Geometry and deformation derivatives use native autograd.
Derived from Xinyue Wei's DISO, CC BY-NC 4.0; see vendor/licenses/diso.LICENSE.
"""
from functools import lru_cache

import torch
from torch import nn
from torch.nn import functional as F

from .metal import library
from .indexing import flat_nonzero, unravel_indices
from .marching_cubes_tables import CORNERS, EDGES, FIRST, TRIANGLES


@lru_cache(None)
def _tables(device):
    return tuple(torch.tensor(t,device=device,dtype=torch.int32) for t in (CORNERS,EDGES,FIRST,TRIANGLES))


@torch.no_grad()
def _topology(grid,iso):
    nx,ny,nz=grid.shape
    corners,edges,first,triangles=_tables(grid.device)
    n=grid.numel()
    if grid.device.type=='mps':
        codes=torch.empty(n,device=grid.device,dtype=torch.int32)
        bits=torch.empty_like(codes);counts=torch.empty_like(codes)
        library('marching_cubes').classify_cells(grid.contiguous(),corners,codes,bits,counts,
            nx,ny,nz,float(iso),threads=n)
    else:
        codes=torch.zeros_like(grid,dtype=torch.int32)
        for c,(x,y,z) in enumerate(CORNERS):
            codes[:-1,:-1,:-1] |= (grid[x:x+nx-1,y:y+ny-1,z:z+nz-1]>=iso).int()<<c
        sign=grid>=iso
        bits=torch.zeros_like(codes)
        for axis in range(3):
            slices=[slice(0,-1)]*3;other=slices.copy();other[axis]=slice(1,None)
            bits[:-1,:-1,:-1] |= (sign[tuple(slices)]!=sign[tuple(other)]).int()<<axis
        codes,bits=codes.flatten(),bits.flatten()
        counts=sum((bits>>axis)&1 for axis in range(3))
    vertex_first=counts.cumsum(0)-counts
    cells=flat_nonzero((codes>0)&(codes<255))
    count=(first[codes[cells].long()+1]-first[codes[cells].long()]).long()
    face_first=count.cumsum(0)-count
    faces=torch.empty(int(count.sum()),device=grid.device,dtype=torch.long)
    if len(cells) and grid.device.type=='mps':
        library('marching_cubes').emit_faces(cells,codes,bits,vertex_first,face_first,first,triangles,edges,
            faces,ny,nz,len(cells),threads=len(cells))
    elif len(cells):
        offsets=torch.arange(15,device=grid.device)
        valid=offsets[None]<count[:,None]
        table_at=first[codes[cells].long(),None]+offsets[None]
        e=edges[triangles[table_at.clamp_max(len(triangles)-1)].long()].long()
        origin=cells[:,None]+e[...,0]*ny*nz+e[...,1]*nz+e[...,2]
        axis=e[...,3]
        lower=bits[origin]&((1<<axis)-1)
        ids=vertex_first[origin]+(lower&1)+((lower>>1)&1)
        faces=ids[valid]
    selected=flat_nonzero(((bits[:,None]>>torch.arange(3,device=grid.device))&1)>0)
    origins,axes=unravel_indices(selected,(bits.numel(),3)).unbind(-1)
    return origins,axes,faces.reshape(-1,3)


class DiffMC(nn.Module):
    """Extract a differentiable triangle mesh from a dense (X,Y,Z) scalar grid."""
    def __init__(self,dtype=torch.float32):
        super().__init__()
        if dtype not in (torch.float32,torch.float64):
            raise ValueError('DiffMC supports float32 and CPU float64')
        self.dtype=dtype

    def forward(self,grid,deform=None,isovalue=0.,normalize=True):
        if grid.ndim!=3 or min(grid.shape)<2:
            raise ValueError('grid must be (X,Y,Z) with each dimension at least 2')
        if grid.dtype!=self.dtype:
            raise ValueError('grid dtype must match DiffMC dtype')
        if deform is not None and (deform.shape!=(*grid.shape,3) or deform.device!=grid.device or deform.dtype!=grid.dtype):
            raise ValueError('deform must have shape (*grid.shape,3), matching grid dtype/device')
        if not torch.isfinite(grid).all(): raise ValueError('grid must be finite')
        if grid.min()>=isovalue or grid.max()<=isovalue:
            anchor=grid.sum()*0+(0 if deform is None else deform.sum()*0)
            return grid.new_zeros((0,3))+anchor,torch.empty((0,3),device=grid.device,dtype=torch.long)
        shape=grid.shape
        padded=F.pad(grid,(1,1,1,1,1,1),value=float(isovalue)+1)
        origin,axis,faces=_topology(padded,isovalue)
        _,ny,nz=padded.shape
        stride=torch.tensor([ny*nz,nz,1],device=grid.device)
        other=origin+stride[axis]
        pos=unravel_indices(origin,padded.shape).to(grid.dtype)
        direction=F.one_hot(axis,3).to(grid.dtype)
        p0=pos;p1=pos+direction
        if deform is not None:
            pd=F.pad(deform.permute(3,0,1,2),(1,1,1,1,1,1)).permute(1,2,3,0).reshape(-1,3)
            p0=p0+pd[origin];p1=p1+pd[other]
        a,b=padded.flatten()[origin],padded.flatten()[other]
        t=((isovalue-a)/(b-a)).clamp(0,1)
        vertices=p0+(p1-p0)*t[:,None]-1
        if normalize: vertices=vertices/(grid.new_tensor(shape)-1)
        return vertices,faces
