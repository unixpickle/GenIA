import importlib.util
from pathlib import Path

import numpy as np
import pytest
import torch
from genia.core.ops.marching_cubes import DiffMC,_topology

DEVICES=['cpu']+(['mps'] if torch.backends.mps.is_available() else [])
spec=importlib.util.spec_from_file_location('diso_oracle',Path(__file__).parent/'reference/diso/reference.py')
oracle=importlib.util.module_from_spec(spec);spec.loader.exec_module(oracle)


@pytest.mark.parametrize('device',DEVICES)
@pytest.mark.parametrize('deformed',[False,True])
def test_marching_cubes_reference(device,deformed):
    rng=np.random.default_rng(126)
    grid=rng.normal(size=(5,7,6)).astype(np.float32)
    deform=rng.uniform(-.2,.2,(*grid.shape,3)).astype(np.float32) if deformed else None
    expected,ef=oracle.extract(grid,deform,iso=.1)
    g=torch.tensor(grid,device=device,requires_grad=True)
    d=None if deform is None else torch.tensor(deform,device=device,requires_grad=True)
    verts,faces=DiffMC()(g,d,isovalue=.1)
    torch.testing.assert_close(verts.cpu(),torch.tensor(expected,dtype=torch.float32),atol=2e-7,rtol=2e-6)
    torch.testing.assert_close(faces.cpu(),torch.tensor(ef))
    verts.square().sum().backward()
    assert torch.isfinite(g.grad).all() and g.grad.abs().sum()>0
    if deformed:assert torch.isfinite(d.grad).all() and d.grad.abs().sum()>0


@pytest.mark.parametrize('device',DEVICES)
def test_all_256_cube_cases(device):
    # Separate every case by an outer positive layer; all cases share one dispatch.
    corners,edges,first,triangles=oracle.tables()
    grid=np.ones((256*3+1,4,4),dtype=np.float32)
    for code in range(256):
        for c,offset in enumerate(corners):
            x,y,z=offset;grid[code*3+x+1,y+1,z+1]=.75 if code&(1<<c) else -.25
    expected,ef=oracle.extract(grid)
    verts,faces=DiffMC()(torch.tensor(grid,device=device))
    torch.testing.assert_close(verts.cpu(),torch.tensor(expected,dtype=torch.float32),atol=1e-7,rtol=1e-6)
    torch.testing.assert_close(faces.cpu(),torch.tensor(ef))


def test_marching_cubes_finite_differences():
    torch.manual_seed(15)
    grid=(torch.randn(3,3,3,dtype=torch.double).sign()*(.5+torch.rand(3,3,3,dtype=torch.double))).requires_grad_()
    deform=(torch.rand(3,3,3,3,dtype=torch.double)*.1).requires_grad_()
    mc=DiffMC(dtype=torch.double)
    assert torch.autograd.gradcheck(lambda g,d:mc(g,d)[0],(grid,deform),fast_mode=True)


@pytest.mark.parametrize('device',DEVICES)
def test_plane_equality_normalization_empty(device):
    grid=torch.arange(5,device=device,dtype=torch.float32)[:,None,None].expand(5,4,3).clone()-2
    verts,faces=DiffMC()(grid,normalize=False)
    # Interior crossings lie exactly on the isovalue-equality samples.
    interior=(verts[:,1]>0)&(verts[:,1]<3)&(verts[:,2]>0)&(verts[:,2]<2)
    assert (verts[interior,0][verts[interior,0]>=0]==2).all()
    normalized,nf=DiffMC()(grid)
    torch.testing.assert_close(normalized,verts/verts.new_tensor([4,3,2]))
    torch.testing.assert_close(faces,nf)
    for sign in [-1.,0.,1.]:
        g=torch.full((3,4,5),sign,device=device,requires_grad=True)
        v,f=DiffMC()(g)
        assert v.shape==f.shape==(0,3)
        v.sum().backward();assert (g.grad==0).all()


@pytest.mark.skipif(not torch.backends.mps.is_available(),reason='requires Metal')
def test_marching_cubes_cpu_mps_gradients():
    torch.manual_seed(125)
    g=torch.randn(9,11,7,requires_grad=True)
    d=(torch.rand(9,11,7,3)*.1).requires_grad_()
    mg=g.detach().to('mps').requires_grad_();md=d.detach().to('mps').requires_grad_()
    v,f=DiffMC()(g,d);mv,mf=DiffMC()(mg,md)
    torch.testing.assert_close(mf.cpu(),f)
    torch.testing.assert_close(mv.cpu(),v)
    v.square().sum().backward();mv.square().sum().backward()
    torch.testing.assert_close(mg.grad.cpu(),g.grad,atol=1e-5,rtol=1e-5)
    torch.testing.assert_close(md.grad.cpu(),d.grad,atol=1e-6,rtol=1e-5)


@pytest.mark.parametrize('device',DEVICES)
def test_pipeline_occupancy_mesh_to_differentiable_render(device):
    from genia.core.utils.mesh_rendering import diffmc_mesh,render_depth_and_alpha
    axis=torch.linspace(-.5,.5,12,device=device)
    xyz=torch.stack(torch.meshgrid(axis,axis,axis,indexing='ij'),-1)
    occupancy=torch.sigmoid(30*(.3-xyz.norm(dim=-1))).requires_grad_()
    verts,faces=diffmc_mesh(occupancy,device)
    assert len(verts)>100 and len(faces)>100
    # A closed surface has exactly two triangles at every edge.
    edges=torch.cat([faces[:,[0,1]],faces[:,[1,2]],faces[:,[2,0]]]).sort(-1).values.cpu()
    _,counts=torch.unique(edges,dim=0,return_counts=True)
    assert (counts==2).all()
    posed=verts+verts.new_tensor([0.,0.,2.])
    depth,alpha=render_depth_and_alpha(posed,faces,None,32.,32.,8.,8.,16,16)
    assert alpha.sum()>10 and depth.max()>1
    (alpha.sum()+depth.sum()).backward()
    assert torch.isfinite(occupancy.grad).all() and occupancy.grad.abs().sum()>0
