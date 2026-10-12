# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

"""Clip-space rasterization with nvdiffrast-compatible buffers on CPU and Metal.

Barycentric and differential equations follow nvdiffrast v0.4.0 (NVIDIA), whose
licence is retained under vendor/licenses/nvdiffrast.LICENSE. Discrete visibility
uses the local Metal triangle selector; continuous values use native autograd.
"""
import torch
from .mesh import _select


class RasterizeCudaContext:
    """Compatibility context. Operations follow input devices and require no CUDA."""
    def __init__(self, device=None):
        self.device = None if device is None else torch.device(device)


RasterizeGLContext = RasterizeCudaContext


def _clip_faces(face):
    """Clip homogeneous polygons to all six frustum planes; return triangles/ids.

    This operation is used only for visibility. Original homogeneous vertices are
    used to recompute differentiable barycentrics after selecting visible faces.
    """
    with torch.no_grad():
        n=len(face)
        source=torch.arange(n,device=face.device)
        if not n: return face,source
        planes=face.new_tensor([[1,0,0,1],[-1,0,0,1],[0,1,0,1],[0,-1,0,1],[0,0,1,1],[0,0,-1,1]])
        distance=face@planes.T
        visible=(distance>=0).any(1).all(-1)
        face,source=face[visible],source[visible]
        if not len(face): return face,source
        if (distance[visible]>=0).all(): return face,source
        n=len(face); max_vertices=9
        polygon=torch.nn.functional.pad(face,(0,0,0,max_vertices-3))
        counts=torch.full((n,),3,device=face.device,dtype=torch.long)
        column=torch.arange(max_vertices,device=face.device)[None].expand(n,-1)
        row=torch.arange(n,device=face.device)[:,None]
        for plane in planes:
            previous=polygon[row,(column-1)%counts.clamp_min(1)[:,None]]
            curr_d=(polygon*plane).sum(-1); prev_d=(previous*plane).sum(-1)
            active=column<counts[:,None]
            inside=curr_d>=0; was_inside=prev_d>=0
            crossing=(inside!=was_inside)&active
            denominator=torch.where(crossing,prev_d-curr_d,1)
            t=prev_d/denominator
            intersection=previous+t[...,None]*(polygon-previous)
            values=torch.stack([intersection,polygon],2).flatten(1,2)
            keep=torch.stack([crossing,inside&active],2).flatten(1,2)
            offsets=keep.long().cumsum(1)-1
            new_counts=keep.sum(1)
            if (new_counts>max_vertices).any(): raise RuntimeError('frustum clipping polygon overflow')
            output=torch.zeros_like(polygon)
            output.scatter_add_(1,offsets.clamp(0,max_vertices-1)[...,None].expand(-1,-1,4),
                                torch.where(keep[...,None],values,0))
            polygon,counts=output,new_counts
        j=torch.arange(1,max_vertices-1,device=face.device)
        triangles=torch.stack([polygon[:,0,None].expand(-1,len(j),-1),polygon[:,j],polygon[:,j+1]],2)
        valid=j[None]+1<counts[:,None]
        return triangles[valid],source[:,None].expand(-1,len(j))[valid]


def _barycentrics(face, pixel, h, w):
    # Homogeneous edge equations avoid dividing vertices behind the eye by w.
    shifted=face[...,:2]-pixel[:,None,:]*face[...,3:4]
    a,b,c=shifted.unbind(1)
    cross=lambda p,q:p[...,0]*q[...,1]-p[...,1]*q[...,0]
    areas=torch.stack([cross(b,c),cross(c,a),cross(a,b)],-1)
    denominator=areas.sum(-1)
    inv=1/torch.where(denominator.abs()>1e-20,denominator,torch.ones_like(denominator))
    bary=areas*inv[:,None]
    b01=bary[:,:2].clamp(0,1)
    b01=b01/b01.sum(-1,keepdim=True).clamp_min(1)
    # A negative denominator is valid for clockwise triangles.
    depth_den=(face[...,3]*areas).sum(-1)
    depth=(face[...,2]*areas).sum(-1)/torch.where(depth_den.abs()>1e-20,depth_den,torch.ones_like(depth_den))
    p0,p1,p2=face.unbind(1)
    dx=torch.stack([p2[:,1]*p1[:,3]-p1[:,1]*p2[:,3],p0[:,1]*p2[:,3]-p2[:,1]*p0[:,3],p1[:,1]*p0[:,3]-p0[:,1]*p1[:,3]],-1)
    dy=torch.stack([p1[:,0]*p2[:,3]-p2[:,0]*p1[:,3],p2[:,0]*p0[:,3]-p0[:,0]*p2[:,3],p0[:,0]*p1[:,3]-p1[:,0]*p0[:,3]],-1)
    dbx=(2/w)*inv[:,None]*(b01*dx.sum(-1,keepdim=True)-dx[:,:2])
    dby=(2/h)*inv[:,None]*(b01*dy.sum(-1,keepdim=True)-dy[:,:2])
    return b01,depth.clamp(-1,1).detach(),torch.stack([dbx[:,0],dby[:,0],dbx[:,1],dby[:,1]],-1)


def rasterize(glctx,pos,tri,resolution,ranges=None,grad_db=True):
    h,w=map(int,resolution)
    if h<1 or w<1 or pos.shape[-1]!=4 or tri.ndim!=2 or tri.shape[-1]!=3:
        raise ValueError('Expected positive resolution, clip-space vertices (...,4), triangles (F,3)')
    if pos.ndim not in (2,3): raise ValueError('pos must be (V,4) or (B,V,4)')
    if pos.device!=tri.device: raise ValueError('positions and triangles must share a device')
    if pos.ndim==2:
        if ranges is None: raise ValueError('range mode requires (start,count) ranges')
        batches=[(pos,int(start),int(count)) for start,count in ranges.tolist()]
    else:
        if ranges is not None and ranges.numel(): raise ValueError('ranges are only used for unbatched positions')
        batches=[(v,0,len(tri)) for v in pos]
    y,x=torch.meshgrid(torch.arange(h,device=pos.device,dtype=pos.dtype),torch.arange(w,device=pos.device,dtype=pos.dtype),indexing='ij')
    pixels=torch.stack([(x+.5)*2/w-1,(y+.5)*2/h-1],-1).reshape(-1,2)
    outputs=[]; derivatives=[]
    for vertices,start,count in batches:
        clipped,original=_clip_faces(vertices[tri[start:start+count].long()])
        clip_w=clipped[...,3:4].clamp_min(1e-12)
        ndc=clipped[...,:3]/clip_w
        selection=torch.cat([-ndc[...,:2]*pos.new_tensor([w/min(h,w),h/min(h,w)]),ndc[...,2:3]+2],-1)
        selected=_select(selection,torch.full((len(selection),),-1,device=pos.device,dtype=torch.long),h,w,0.,1,False,False,False,inclusive=True)[:,0]
        valid=selected>=0
        if len(clipped):
            ids=original[selected.clamp_min(0)]+start
            faces=vertices[tri[ids].long()]
            b,depth,db=_barycentrics(faces,pixels,h,w)
            result=torch.cat([b,depth[:,None],(ids+1).to(pos.dtype)[:,None]],-1)
            result=torch.where(valid[:,None],result,0)
            db=torch.where(valid[:,None],db,0)
        else:
            anchor=pos.sum()*0
            result=pos.new_zeros((h*w,4))+anchor;db=result.clone()
        outputs.append(result.reshape(h,w,4));derivatives.append(db.reshape(h,w,4))
    rast,db=torch.stack(outputs),torch.stack(derivatives)
    return rast,db if grad_db else db.detach()


def interpolate(attr,rast,tri,rast_db=None,diff_attrs=None):
    if attr.ndim==2: attr=attr[None]
    if attr.ndim!=3 or attr.shape[0] not in (1,rast.shape[0]): raise ValueError('attribute batch mismatch')
    selected=list(range(attr.shape[-1])) if diff_attrs=='all' else list(diff_attrs or [])
    if selected and rast_db is None: raise ValueError('attribute derivatives require rast_db')
    values=[];derivatives=[]
    for batch in range(len(rast)):
        ids=rast[batch,...,3].long()-1; valid=ids>=0
        if len(tri):
            attrs=attr[min(batch,len(attr)-1)][tri[ids.clamp_min(0)].long()]
            b=torch.cat([rast[batch,...,:2],1-rast[batch,...,:2].sum(-1,keepdim=True)],-1)
            value=(attrs*b[...,None]).sum(-2)
            value=torch.where(valid[...,None],value,0)
            if selected:
                delta=attrs[...,:2,selected]-attrs[...,2:3,selected]
                db=rast_db[batch].reshape(*ids.shape,2,2)
                deriv=torch.einsum('...ia,...ij->...aj',delta,db).flatten(-2)
                deriv=torch.where(valid[...,None],deriv,0)
            else: deriv=value[...,:0]
        else:
            value=attr.new_zeros((*ids.shape,attr.shape[-1]))+attr.sum()*0+rast.sum()*0
            deriv=attr.new_zeros((*ids.shape,2*len(selected)))+value.sum()*0
        values.append(value);derivatives.append(deriv)
    return torch.stack(values),torch.stack(derivatives)
