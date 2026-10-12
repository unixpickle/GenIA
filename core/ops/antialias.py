# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

"""Silhouette edge antialiasing on CPU/MPS following nvdiffrast v0.4.0.

The edge-selection, coverage and stabilized gradient equations are derived from
NVIDIA's antialias.cu; its source-code licence is in vendor/licenses/nvdiffrast.LICENSE.
This implementation uses device-native tensor operations and autograd scatter.
"""
from dataclasses import dataclass
import torch
from torch.autograd.function import once_differentiable
from .unique import unique


@dataclass
class TopologyHash:
    opposites: torch.Tensor
    triangles: torch.Tensor


@torch.no_grad()
def antialias_construct_topology_hash(tri):
    if tri.ndim!=2 or tri.shape[1]!=3: raise ValueError('triangles must be (F,3)')
    # Each row describes an edge and its opposite vertex.
    edges=tri[:,[[1,2],[2,0],[0,1]]].reshape(-1,2).sort(-1).values
    opposite=tri.reshape(-1)
    _,inverse,counts=unique(edges,dim=0,return_inverse=True,return_counts=True)
    order=inverse.argsort(stable=True)
    starts=counts.cumsum(0)-counts
    # Manifold edges have one or two distinct opposite vertices. Retain the
    # first two on nonmanifold edges, matching the upstream hash's capacity.
    first=order[starts] if len(order) else order
    second=order[(starts+1).minimum((starts+counts-1))] if len(order) else order
    if len(order):
        a,b=opposite[first[inverse]],opposite[second[inverse]]
        neighbor=torch.where(opposite==a,b,torch.where(opposite==b,a,-1))
        neighbor=torch.where((counts[inverse]>1)&(a!=b)&(edges[:,0]!=edges[:,1]),neighbor,-1)
    else: neighbor=opposite
    return TopologyHash(neighbor.reshape(-1,3).long(),tri.clone())


class _Coverage(torch.autograd.Function):
    @staticmethod
    def forward(ctx,a,b,ds):
        dx,dy=(b-a).unbind(-1)
        determinant=a[:,0]*dy-a[:,1]*dx
        alpha=ds*(.5-(ds*determinant/dy).clamp(0,1))
        ctx.save_for_backward(a,b,alpha)
        return alpha

    @staticmethod
    @once_differentiable
    def backward(ctx,grad):
        a,b,alpha=ctx.saved_tensors
        dx,dy=(b-a).unbind(-1)
        determinant=a[:,0]*dy-a[:,1]*dx
        inv=1/(dy+torch.copysign(torch.full_like(dy,1e-3),dy))
        grad=torch.where((alpha.abs()<.5)&(alpha!=0),grad,0)
        v=determinant*inv
        da=torch.stack([-inv*b[:,1],-inv*(v-b[:,0])],-1)*grad[:,None]
        db=torch.stack([ inv*a[:,1], inv*(v-a[:,0])],-1)*grad[:,None]
        return da,db,None


def _same_sign(a,b): return torch.signbit(a)==torch.signbit(b)


def _rational_gt(n0,n1,d0,d1): return (n0*d1>n1*d0)==_same_sign(d0,d1)


def antialias(color,rast,pos,tri,topology_hash=None,pos_gradient_boost=1.):
    if color.shape[:3]!=rast.shape[:3] or rast.shape[-1]!=4: raise ValueError('color and raster shapes must agree')
    if not len(tri): return color+pos.sum()*0
    topology=antialias_construct_topology_hash(tri) if topology_hash is None else topology_hash
    if topology.opposites.shape!=tri.shape or not torch.equal(topology.triangles,tri):
        raise ValueError('topology hash does not match the triangles')
    n,h,w,_=color.shape
    if pos.ndim==2: pos=pos[None].expand(n,-1,-1)
    if pos.shape[0]!=n: raise ValueError('position and image batches must agree')
    pos=pos.detach()+(pos-pos.detach())*pos_gradient_boost
    ids=rast[...,3].long()-1
    flat_color=color.reshape(-1,color.shape[-1])
    updates=[];targets=[]
    scale=pos.new_tensor([w/2,h/2])
    grid=torch.arange(n*h*w,device=pos.device).reshape(n,h,w)
    for down in (False,True):
        if down:
            base=grid[:,:-1]; other=grid[:,1:]; changes=ids[:,:-1]!=ids[:,1:]
        else:
            base=grid[:,:,:-1]; other=grid[:,:,1:]; changes=ids[:,:,:-1]!=ids[:,:,1:]
        pixel0,pixel1=base[changes],other[changes]
        if not len(pixel0): continue
        r0,r1=rast.reshape(-1,4)[pixel0],rast.reshape(-1,4)[pixel1]
        t0,t1=r0[:,3].long()-1,r1[:,3].long()-1
        choose1=(t0<0)|((t1>=0)&(r0[:,2]>=r1[:,2]))
        triangle=torch.where(choose1,t1,t0)
        selected_pixel=torch.where(choose1,pixel1,pixel0)
        batch=selected_pixel//(h*w)
        center=torch.stack([selected_pixel%w,(selected_pixel//w)%h],-1).to(pos.dtype)+.5-scale
        vertex_ids=tri[triangle].long()
        face=pos[batch[:,None],vertex_ids]
        opposite=topology.opposites[triangle]
        wing=pos[batch[:,None],torch.where(opposite>=0,opposite,vertex_ids)]
        face_xy=face[...,:2]/face[...,3:4]*scale-center[:,None]
        wing_xy=wing[...,:2]/wing[...,3:4]*scale-center[:,None]
        cross=lambda a,b:a[...,0]*b[...,1]-a[...,1]*b[...,0]
        a,b,c=face_xy.unbind(1)
        normal=cross(b-a,c-a)
        wing_area=torch.stack([cross(b-wing_xy[:,0],c-wing_xy[:,0]),
            cross(c-wing_xy[:,1],a-wing_xy[:,1]),cross(a-wing_xy[:,2],b-wing_xy[:,2])],-1)
        silhouette=_same_sign(wing_area,normal[:,None])
        if down: face_xy=face_xy.flip(-1)
        a=face_xy[:,[1,2,0]];b=face_xy[:,[2,0,1]]
        delta=b-a
        ds=torch.where(choose1,-torch.ones_like(normal),torch.ones_like(normal))
        numerator=ds[:,None]*cross(a,delta)
        denominator=delta[...,1]
        crossing=~_same_sign(a[...,1],b[...,1])
        numerator=torch.where(crossing,numerator,torch.finfo(pos.dtype).min)
        denominator=torch.where(crossing,denominator,1)
        g10=_rational_gt(numerator[:,1],numerator[:,0],denominator[:,1],denominator[:,0])
        g20=_rational_gt(numerator[:,2],numerator[:,0],denominator[:,2],denominator[:,0])
        g21=_rational_gt(numerator[:,2],numerator[:,1],denominator[:,2],denominator[:,1])
        edge=torch.where(g20&g21,2,torch.where(g10,1,0))
        rows=torch.arange(len(edge),device=pos.device)
        den=denominator[rows,edge]
        distance=numerator[rows,edge]/torch.where(den!=0,den,1)
        valid=(silhouette[rows,edge] & crossing[rows,edge] & (den.abs()>=delta[rows,edge,0].abs()) &
               (den!=0) & (distance>-.0625) & (distance<1.0625))
        if not valid.any(): continue
        alpha=_Coverage.apply(a[rows,edge][valid],b[rows,edge][valid],ds[valid])
        p0,p1=pixel0[valid],pixel1[valid]
        targets.append(torch.where(alpha>0,p0,p1))
        updates.append(alpha[:,None]*(flat_color[p1]-flat_color[p0]))
    if not updates: return color+pos.sum()*0
    output=flat_color.index_add(0,torch.cat(targets),torch.cat(updates))
    return output.reshape_as(color)
