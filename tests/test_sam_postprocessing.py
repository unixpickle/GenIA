"""SAM mesh cleanup and legacy geometry helpers on CPU and Metal."""
import os
os.environ.setdefault('LIDRA_SKIP_INIT','1')

import numpy as np
import pytest
import torch
import utils3d

DEVICES=['cpu']+(['mps'] if torch.backends.mps.is_available() else [])


@pytest.mark.parametrize('device',DEVICES)
def test_legacy_face_and_edge_components(device):
    # Faces 0/1 share an edge. Face 2 only touches a vertex, so it stays separate.
    faces=torch.tensor([[0,1,2],[2,1,3],[3,4,5],[8,9,10]],device=device)
    edges,mapping,counts=utils3d.torch.compute_edges(faces)
    assert len(edges)==11 and (counts==2).sum()==1
    components=utils3d.torch.compute_connected_components(faces,edges,mapping)
    assert [sorted(c.cpu().tolist()) for c in components]==[[0,1],[2],[3]]
    dual,primal=utils3d.torch.compute_dual_graph(mapping)
    assert dual.cpu().tolist()==[[0,1]]
    assert edges[primal].cpu().tolist()==[[1,2]]
    edge_components=utils3d.torch.compute_edge_connected_components(edges)
    assert sorted(map(len,edge_components))==[3,8]
    empty=faces[:0]
    assert len(utils3d.torch.compute_connected_components(empty))==0
    assert len(utils3d.torch.compute_edge_connected_components(edges[:0]))==0
    dual,primal=utils3d.torch.compute_dual_graph(mapping[:0])
    assert dual.shape==(0,2) and primal.shape==(0,)


@pytest.mark.parametrize('device',DEVICES)
def test_sam_cameras_and_legacy_raster(device):
    from sam3d_objects.model.backbone.tdfy_dit.utils.render_utils import yaw_pitch_r_fov_to_extrinsics_intrinsics
    views,intrinsics=yaw_pitch_r_fov_to_extrinsics_intrinsics([0.,.7],[0.,.3],2.,40.,device=device)
    for extr,intr in zip(views,intrinsics):
        assert extr.device.type==device and intr.device.type==device
        assert extr.shape==(4,4) and intr.shape==(3,3)
        center=extr@extr.new_tensor([0,0,0,1])
        torch.testing.assert_close(center,extr.new_tensor([0,0,2,1]),atol=1e-6,rtol=1e-6)
        torch.testing.assert_close(intr[:2,2],intr.new_tensor([.5,.5]))
    verts=torch.tensor([[[-.75,-.5,0],[.5,-.5,0],[0,.75,0]]],device=device,requires_grad=True)
    faces=torch.tensor([[0,1,2]],device=device,dtype=torch.int32)
    attrs=torch.ones(1,3,3,device=device,requires_grad=True)
    result=utils3d.torch.rasterize_triangle_faces(utils3d.torch.RastContext(device=device),verts,faces,12,10,attr=attrs)
    assert result['image'].shape==(1,3,10,12) and result['mask'].sum()>0
    result['image'].sum().backward()
    assert torch.isfinite(verts.grad).all() and attrs.grad.sum()>0


@pytest.mark.parametrize('device',DEVICES)
def test_sam_hole_fill_removes_hidden_inner_surface(device):
    import trimesh
    from sam3d_objects.model.backbone.tdfy_dit.utils.postprocessing_utils import _fill_holes
    outer=trimesh.creation.icosphere(subdivisions=1,radius=.65)
    inner=trimesh.creation.icosphere(subdivisions=1,radius=.12)
    mesh=trimesh.util.concatenate([outer,inner])
    verts=torch.tensor(mesh.vertices,device=device,dtype=torch.float32)
    faces=torch.tensor(mesh.faces,device=device,dtype=torch.int32)
    cleaned,cf=_fill_holes(verts,faces,resolution=32,num_views=20)
    assert cleaned.device.type==device and cf.device.type==device
    assert len(cf)<len(faces) and len(cf)>0
    assert cleaned.norm(dim=-1).min()>.5


@pytest.mark.parametrize('device',DEVICES)
def test_sam_fast_texture_baking(device):
    from sam3d_objects.model.backbone.tdfy_dit.utils.postprocessing_utils import bake_texture
    verts=np.array([[-.6,-.6,2],[.6,-.6,2],[.6,.6,2],[-.6,.6,2]],np.float32)
    faces=np.array([[0,1,2],[0,2,3]],np.int32)
    uv=np.array([[.05,.05],[.95,.05],[.95,.95],[.05,.95]],np.float32)
    obs=np.full((16,16,3),[40,100,200],np.uint8)
    intr=np.array([[1,0,.5],[0,1,.5],[0,0,1]],np.float32)
    texture=bake_texture(verts,faces,uv,[obs],[np.ones((16,16),bool)],
        [np.eye(4,dtype=np.float32)],[intr],texture_size=8,mode='fast',device=device)
    assert texture.shape==(8,8,3)
    assert np.abs(texture.astype(float).mean((0,1))-[40,100,200]).max()<5
