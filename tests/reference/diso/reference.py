"""Scalar NumPy transcription of DISO 0.1.4's CUDA topology, for tests only.

Read the unchanged source tables independently of the production implementation.
Xinyue Wei, CC BY-NC 4.0 (LICENSE in this directory).
"""
from pathlib import Path
import ast
import re
import numpy as np


def tables():
    source=re.sub(r'//[^\n]*','',Path(__file__).with_name('cumc.cu').read_text())
    result=[]
    for name in ['mcCorners','mcEdgeLocations','firstMarchingCubesId','marchingCubesIds']:
        text=re.search(r'__constant__ int '+name+r'[^=]*=\s*(\{.*?\});',source,re.S).group(1)
        result.append(ast.literal_eval(text.replace('{','[').replace('}',']')))
    return result


def extract(grid,deform=None,iso=0.,normalize=True):
    corners,edges,first,triangles=tables()
    shape=grid.shape
    data=np.pad(grid,1,constant_values=iso+1)
    deformed=None if deform is None else np.pad(deform,[(1,1)]*3+[(0,0)])
    used=[];vertices=[];mapping={}
    for xyz in np.ndindex(tuple(n-1 for n in data.shape)):
        code=sum((1<<c) for c,offset in enumerate(corners) if data[tuple(a+b for a,b in zip(xyz,offset))]>=iso)
        if code in (0,255):continue
        used.append((xyz,code))
        for axis in range(3):
            end=list(xyz);end[axis]+=1;end=tuple(end)
            a,b=data[xyz],data[end]
            if (a<iso)==(b<iso):continue
            t=np.clip((iso-a)/(b-a),0,1)
            p0=np.array(xyz,dtype=grid.dtype);p1=np.array(end,dtype=grid.dtype)
            if deformed is not None:p0+=deformed[xyz];p1+=deformed[end]
            mapping[(*xyz,axis)]=len(vertices)
            vertices.append(p0+(p1-p0)*t)
    faces=[]
    for xyz,code in used:
        for start in range(first[code],first[code+1],3):
            face=[]
            for edge in triangles[start:start+3]:
                *delta,axis=edges[edge]
                origin=tuple(a+b for a,b in zip(xyz,delta))
                face.append(mapping[(*origin,axis)])
            faces.append(face)
    vertices=np.asarray(vertices).reshape(-1,3)-1
    if normalize:vertices=vertices/(np.array(shape)-1)
    return vertices,np.asarray(faces,dtype=np.int64).reshape(-1,3)
