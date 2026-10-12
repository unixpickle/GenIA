"""Metal mesh visibility and derivatives against PyTorch3D's Python oracle."""
import copy
from types import SimpleNamespace
import pytest
import torch
from pytorch3d.structures import Meshes
from pytorch3d.renderer import (rasterize_meshes, PerspectiveCameras, MeshRasterizer,
    RasterizationSettings, SoftSilhouetteShader, BlendParams, TexturesVertex, MeshRenderer,
    HardFlatShader, look_at_view_transform)
from pytorch3d.renderer.mesh.rasterize_meshes import rasterize_meshes_python
from pytorch3d.ops import mesh_face_areas_normals, interpolate_face_attributes

DEVICES=['cpu']+(['mps'] if torch.backends.mps.is_available() else [])


def mesh_data(dtype=torch.float32):
    verts=torch.tensor([[.85,.75,1.2],[-.7,.6,1.8],[.1,-.75,1.4],
                         [.9,.2,2.4],[-.9,.1,2.8],[.2,-.9,2.1]],dtype=dtype)
    faces=torch.tensor([[0,1,2],[5,4,3]])
    return verts,faces


@pytest.mark.parametrize('device',DEVICES)
@pytest.mark.parametrize('options', [dict(),dict(perspective_correct=True),
    dict(blur_radius=.04,clip_barycentric_coords=True),
    dict(blur_radius=.03,perspective_correct=True,clip_barycentric_coords=True),dict(cull_backfaces=True)])
def test_raster_values_and_gradients(device,options):
    v,f=mesh_data(); v.requires_grad_(); mv=v.detach().to(device).requires_grad_()
    ref=rasterize_meshes_python(Meshes([v],[f]),image_size=(9,13),faces_per_pixel=3,
                                cull_to_frustum=False,**options)
    got=rasterize_meshes(Meshes([mv],[f.to(device)]),image_size=(9,13),faces_per_pixel=3,**options)
    torch.testing.assert_close(got[0].cpu(),ref[0])
    for a,b in zip(got[1:],ref[1:]): torch.testing.assert_close(a.cpu(),b,atol=2e-6,rtol=2e-5)
    torch.manual_seed(69)
    weights=[torch.randn_like(x) for x in ref[1:]]
    sum((a*w.to(device)).sum() for a,w in zip(got[1:],weights)).backward()
    sum((a*w).sum() for a,w in zip(ref[1:],weights)).backward()
    torch.testing.assert_close(mv.grad.cpu(),v.grad,atol=3e-5,rtol=3e-4)


@pytest.mark.parametrize('device',DEVICES)
def test_empty_degenerate_and_batch(device):
    v,f=mesh_data(); v=v.to(device).requires_grad_(); f=f.to(device)
    meshes=Meshes([v,v[:0],torch.zeros(3,3,device=device)], [f,f[:0],f[:1]])
    ids,z,bary,dist=rasterize_meshes(meshes, image_size=8, faces_per_pixel=50,blur_radius=.001)
    assert (ids[1:]==-1).all() and (z[1:]==-1).all()
    assert ids.shape==(3,8,8,50)
    z.sum().backward()
    assert torch.isfinite(v.grad).all()


@pytest.mark.parametrize('device',DEVICES)
def test_attributes_normals_and_silhouette(device):
    v,f=mesh_data(); v=v.to(device).requires_grad_(); f=f.to(device)
    areas,normals=mesh_face_areas_normals(v,f)
    torch.testing.assert_close(torch.linalg.vector_norm(normals,dim=-1),torch.ones(2,device=device))
    assert (areas>0).all()
    meshes=Meshes([v],[f],textures=TexturesVertex(verts_features=[torch.ones_like(v)*.6]))
    cameras=PerspectiveCameras(device=device)
    rasterizer=MeshRasterizer(cameras=cameras,raster_settings=RasterizationSettings(image_size=12,
        blur_radius=.03, faces_per_pixel=3, perspective_correct=True))
    frag=rasterizer(meshes)
    color=meshes.sample_textures(frag)
    torch.testing.assert_close(color[frag.pix_to_face>=0],torch.full_like(color[frag.pix_to_face>=0],.6))
    result=SoftSilhouetteShader(blend_params=BlendParams(sigma=.01))(frag,meshes)
    expected=1-(1-torch.sigmoid(-frag.dists/.01)*(frag.pix_to_face>=0)).prod(-1)
    torch.testing.assert_close(result[...,3],expected)
    result[...,3].sum().backward()
    assert torch.isfinite(v.grad).all() and v.grad.abs().sum()>0


@pytest.mark.parametrize('device',DEVICES)
def test_raster_near_plane_clipping(device):
    v,f=mesh_data(); v[0,2]=-.5
    meshes=Meshes([v.to(device)],[f.to(device)])
    result=rasterize_meshes(meshes,image_size=10,faces_per_pixel=3, perspective_correct=True,z_clip_value=.2)
    assert torch.isfinite(result[1]).all()
    # A split quadrilateral contributes once at a pixel, not two silhouettes.
    assert ((result[0]==0).sum(-1)<=1).all()
    assert (result[0]>=0).any()


def test_raster_cpu_gradcheck():
    from genia.core.ops.mesh import rasterize_face_verts
    v,f=mesh_data(torch.float64); fv=v[f].requires_grad_()
    def op(x): return rasterize_face_verts(x,torch.tensor([0]),torch.tensor([2]),
        torch.tensor([-1,-1]),(3,4),.01,2,perspective_correct=True)[1:]
    assert torch.autograd.gradcheck(op,(fv,),atol=1e-4,rtol=1e-3)


@pytest.mark.parametrize('device',DEVICES)
def test_face_normals_attributes_gradients(device):
    v,f=mesh_data(); v.requires_grad_(); mv=v.detach().to(device).requires_grad_()
    ref=mesh_face_areas_normals(v,f)
    got=mesh_face_areas_normals(mv,f.to(device))
    for a,b in zip(got,ref): torch.testing.assert_close(a.cpu(),b)
    sum(a.square().sum() for a in got).backward()
    sum(a.square().sum() for a in ref).backward()
    torch.testing.assert_close(mv.grad.cpu(),v.grad)
    deg=torch.zeros(3,3,device=device,requires_grad=True)
    a,n=mesh_face_areas_normals(deg,torch.tensor([[0,1,2]],device=device))
    (a.sum()+n.sum()).backward()
    assert torch.isfinite(deg.grad).all()
    attrs=torch.randn(2,3,5,device=device,requires_grad=True)
    bary=torch.tensor([[[[[.2,.3,.5],[-1.,-1.,-1.]]]]],device=device,requires_grad=True)
    ids=torch.tensor([[[[1,-1]]]],device=device)
    out=interpolate_face_attributes(ids,bary,attrs)
    torch.testing.assert_close(out[0,0,0,0],(attrs[1]*bary[0,0,0,0,:,None]).sum(0))
    assert (out[0,0,0,1]==0).all()
    out.sum().backward()
    assert torch.isfinite(attrs.grad).all() and (bary.grad[0,0,0,1]==0).all()


@pytest.mark.skipif(not torch.backends.mps.is_available(),reason='requires Metal')
def test_obj_camera_and_flat_shader(tmp_path):
    from genia.core.utils.mesh_rendering import PyTorch3DMeshRenderer
    obj=tmp_path/'triangle.obj'
    obj.write_text('v -0.6 -0.4 0 1 0 0\nv 0.6 -0.4 0 0 1 0\nv 0 0.7 0 0 0 1\nf 1 2 3\n')
    images=[]
    for device in ['cpu','mps']:
        renderer=PyTorch3DMeshRenderer(torch.device(device))
        mesh=renderer.load_mesh_with_vertex_colors(str(obj))
        R,T=renderer.get_camera_positions(distance=2.)['front']
        images.append(torch.from_numpy(renderer.render_from_view(mesh,R,T,image_size=16)))
    torch.testing.assert_close(images[1],images[0],atol=1e-5,rtol=1e-5)
    assert images[0].min()<.9


@pytest.mark.parametrize('device',DEVICES)
def test_hard_blend_background_values_gradients(device):
    from pytorch3d.renderer.blending import hard_rgb_blend
    colors=torch.randn(1,3,4,2,3,device=device,requires_grad=True)
    background=torch.tensor([.2,.5,.8],device=device,requires_grad=True)
    ids=torch.zeros(1,3,4,2,device=device,dtype=torch.long)
    ids[:,::2]=-1
    image=hard_rgb_blend(colors,SimpleNamespace(pix_to_face=ids),BlendParams(background_color=background))
    torch.testing.assert_close(image[0,0,:,:3],background.expand(4,3))
    torch.testing.assert_close(image[0,1,:,:3],colors[0,1,:,0])
    image.sum().backward()
    torch.testing.assert_close(background.grad,torch.full((3,),8.,device=device))
    assert (colors.grad[ids<0]==0).all()


@pytest.mark.skipif(not torch.backends.mps.is_available(),reason='requires Metal')
@pytest.mark.parametrize('size,k', [((33,65),1),((65,33),8),((19,37),50)])
def test_tile_boundaries_many_faces(size,k):
    from genia.core.ops.mesh import rasterize_face_verts
    torch.manual_seed(16)
    centers=torch.randn(60,1,3)*.9
    faces=centers+torch.randn(60,3,3)*.3
    faces[...,2]=faces[...,2].abs()+.2
    first=torch.tensor([0]); count=torch.tensor([60]); neighbors=torch.full((60,),-1)
    args=dict(image_size=size,blur_radius=.02,faces_per_pixel=k,perspective_correct=True,
              clip_barycentric_coords=True)
    ref=rasterize_face_verts(faces,first,count,neighbors,**args)
    got=rasterize_face_verts(faces.to('mps'),first.to('mps'),count.to('mps'),neighbors.to('mps'),**args)
    torch.testing.assert_close(got[0].cpu(),ref[0])
    for a,b in zip(got[1:],ref[1:]): torch.testing.assert_close(a.cpu(),b,atol=3e-5,rtol=2e-4)
