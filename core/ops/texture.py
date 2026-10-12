# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

"""Native autograd 2D texture sampling and mipmaps for the local rasterizer.

Mip footprint equations follow nvdiffrast v0.4.0; see its retained NVIDIA licence.
"""
import torch
import torch.nn.functional as F


def texture_construct_mip(tex,max_mip_level=None,cube_mode=False):
    if cube_mode: raise NotImplementedError('cube-map textures are not used by GenIA')
    levels=[]; current=tex
    maximum=100 if max_mip_level is None else max_mip_level
    if maximum<0: raise ValueError('max_mip_level must be nonnegative')
    for _ in range(maximum):
        h,w=current.shape[1:3]
        if (h,w)==(1,1): break
        if (h>1 and h%2) or (w>1 and w%2):
            raise ValueError('mipmapped texture dimensions must halve evenly; set max_mip_level to stop earlier')
        kernel=(min(h,2),min(w,2))
        current=F.avg_pool2d(current.permute(0,3,1,2),kernel).permute(0,2,3,1)
        levels.append(current)
    return levels


def _sample(tex,uv,boundary,nearest=False):
    n,h,w,c=tex.shape; batch=uv.shape[0]
    if n not in (1,batch): raise ValueError('texture batch must match UVs or be 1')
    if boundary not in ('wrap','clamp','zero'): raise ValueError('boundary_mode must be wrap, clamp or zero')
    xy=uv*uv.new_tensor([w,h])-(0 if nearest else .5)
    integer=xy.floor().long(); fraction=xy-integer
    tex=tex.expand(batch,-1,-1,-1).reshape(batch,h*w,c)
    def fetch(ix,iy):
        valid=(ix>=0)&(ix<w)&(iy>=0)&(iy<h)
        if boundary=='wrap': ix,iy=ix%w,iy%h
        else: ix,iy=ix.clamp(0,w-1),iy.clamp(0,h-1)
        indices=(iy*w+ix).reshape(batch,-1)
        result=tex.gather(1,indices[...,None].expand(-1,-1,c)).reshape(*uv.shape[:-1],c)
        return torch.where(valid[...,None],result,0) if boundary=='zero' else result
    x,y=integer.unbind(-1)
    if nearest: return fetch(x,y)
    fx,fy=fraction[...,0:1],fraction[...,1:2]
    return ((1-fx)*fetch(x,y)+fx*fetch(x+1,y))*(1-fy)+((1-fx)*fetch(x,y+1)+fx*fetch(x+1,y+1))*fy


def texture(tex,uv,uv_da=None,mip_level_bias=None,mip=None,filter_mode='auto',boundary_mode='wrap',max_mip_level=None):
    if tex.ndim!=4 or uv.ndim!=4 or uv.shape[-1]!=2: raise ValueError('expected NHWC texture and NHW2 UVs')
    if filter_mode=='auto': filter_mode='linear-mipmap-linear' if uv_da is not None or mip_level_bias is not None else 'linear'
    if filter_mode in ('linear','nearest'):
        return _sample(tex,uv,boundary_mode,filter_mode=='nearest')
    if filter_mode not in ('linear-mipmap-linear','linear-mipmap-nearest'): raise ValueError('unsupported texture filter')
    if uv_da is None and mip_level_bias is None: raise ValueError('mipmapping requires UV derivatives or mip-level bias')
    levels=[tex]+(texture_construct_mip(tex,max_mip_level) if mip is None else list(mip))
    maxlevel=len(levels)-1
    if uv_da is None: lod=uv.new_zeros(uv.shape[:-1])
    else:
        if uv_da.shape!=(*uv.shape[:-1],4): raise ValueError('uv_da must have four derivatives per pixel')
        h,w=tex.shape[1:3]
        dx,dy,ex,ey=(uv_da*uv_da.new_tensor([w,w,h,h])).unbind(-1)
        a,b,c=dx*dx+ex*ex,dy*dy+ey*ey,dx*dy+ex*ey
        q=.25*(a-b).square()+c.square()
        # Isotropic footprints have an undefined principal-axis derivative.
        # Like nvdiffrast, return zero derivative for this degenerate axis term.
        root=torch.where(q>0,q.clamp_min(1e-30).sqrt(),0)
        major=.5*(a+b)+root
        major=torch.where(q>0,major,major.detach())
        lod=.5*major.clamp_min(1e-20).log2()
    if mip_level_bias is not None: lod=lod+mip_level_bias
    lod=lod.clamp(0,maxlevel)
    lower=lod.floor().long(); upper=(lower+1).clamp_max(maxlevel)
    fraction=lod-lower
    result=tex.new_zeros((*uv.shape[:-1],tex.shape[-1]))
    for i,level in enumerate(levels):
        sampled=_sample(level,uv,boundary_mode)
        if filter_mode=='linear-mipmap-nearest': weight=(lower==i).to(lod.dtype)
        else: weight=(lower==i)*(1-fraction)+(upper==i)*fraction
        result=result+sampled*weight[...,None]
    return result
