"""Exported assets can be reopened on CPU and Metal in one process."""
import numpy as np
import pytest
import torch

DEVICES=['cpu']+(['mps'] if torch.backends.mps.is_available() else [])


@pytest.mark.parametrize('device', DEVICES)
@pytest.mark.parametrize('in_place', [False, True])
def test_final_scene_assembly_without_notebook_visualizer(monkeypatch, device, in_place):
    import sys
    from genia.core.utils.gaussian import create_gaussians_object
    from genia.core.utils.slat_decode import make_scene

    # The actual final-evaluation path must not load optional notebook tooling.
    monkeypatch.setitem(sys.modules, 'sam3d_objects.utils.visualization', None)
    points = torch.tensor([[.1, .2, .3], [-.2, .1, -.1]], device=device)
    rotation = torch.tensor([[2**-.5, 0, 0, 2**-.5]], device=device)
    scale = torch.tensor([[2., 3., 4.]], device=device)
    translation = torch.tensor([[.5, -.5, 2.]], device=device)
    objects = []
    for offset in [0., .1]:
        gs = create_gaussians_object(
            points + offset, torch.full((2, 1, 3), offset, device=device),
            torch.full((2, 3), .03, device=device),
            torch.tensor([[1., 0, 0, 0]] * 2, device=device),
            torch.full((2, 1), .7, device=device), aabb=[-1, -1, -1, 2, 2, 2])
        objects.append(dict(gaussian=[gs], rotation=rotation, scale=scale,
                            translation=translation))
    scene = make_scene(*objects, in_place=in_place)
    local = torch.cat([points, points + .1]) * scale
    # SAM uses row vectors: positive quarter-turn about Z sends (x,y) to (y,-x).
    expected = torch.stack([local[:, 1], -local[:, 0], local[:, 2]], -1) + translation
    torch.testing.assert_close(scene.get_xyz, expected, atol=1e-6, rtol=1e-5)
    torch.testing.assert_close(scene.get_scaling, (.03 * scale).expand(4, 3))
    torch.testing.assert_close(scene.get_opacity, torch.full((4, 1), .7, device=device))
    expected_rotation = torch.tensor([[2**-.5, 0, 0, -2**-.5]], device=device).expand(4, 4)
    torch.testing.assert_close(scene.get_rotation, expected_rotation)
    if in_place:
        assert scene is objects[0]['gaussian'][0]
    else:
        torch.testing.assert_close(objects[0]['gaussian'][0].get_xyz, points)


@pytest.mark.parametrize('device',DEVICES)
def test_exported_mesh_and_gaussian_device(tmp_path,monkeypatch,device):
    from genia.core.utils import render_final_results as rendering
    import trimesh
    from plyfile import PlyData,PlyElement
    rendering._lazy_imports()
    monkeypatch.setenv('GENIA_DEVICE',device)
    mesh=trimesh.creation.icosphere(subdivisions=0)
    path=tmp_path/'mesh.glb';mesh.export(path)
    cpu=rendering.load_mesh(path,'cpu')
    selected=rendering.load_mesh(path)
    assert cpu.verts.device.type=='cpu' and selected.verts.device.type==device
    torch.testing.assert_close(selected.verts.cpu(),cpu.verts)
    fields=['x','y','z','opacity']+[f'rot_{i}' for i in range(4)]+[f'scale_{i}' for i in range(3)]+[f'f_dc_{i}' for i in range(3)]
    data=np.zeros(2,dtype=[(name,'f4') for name in fields])
    data['rot_0']=1;data['z']=2
    ply=tmp_path/'gaussian.ply'
    PlyData([PlyElement.describe(data,'vertex')]).write(str(ply))
    cloud=rendering.load_gaussian_ply(ply)
    assert cloud.means.device.type==device
    torch.testing.assert_close(cloud.scales,torch.ones(2,3,device=device))
    torch.testing.assert_close(cloud.opacities,torch.full((2,),.5,device=device))


@pytest.mark.parametrize('device',DEVICES)
def test_pipeline_gaussian_render_cache_and_camera_gradients(monkeypatch,device):
    from genia.core.utils.gaussian import create_gaussians_object
    from genia.core.utils.rendering import render_gaussians_scene
    from genia.core.utils.pipeline_state import _serialize_value,_deserialize_value
    from sam3d_objects.model.backbone.tdfy_dit.modules.sparse import SparseTensor
    monkeypatch.setenv('GENIA_DEVICE',device)
    xyz=torch.tensor([[-.1,0.,2.],[.2,.1,2.2]],device=device,requires_grad=True)
    colors=torch.zeros(2,1,3,device=device,requires_grad=True)
    scales=torch.full((2,3),.12,device=device)
    rot=torch.tensor([[1.,0,0,0]]*2,device=device)
    gs=create_gaussians_object(xyz,colors,scales,rot,torch.full((2,1),.7,device=device),
                              aabb=[-1,-1,1,2,2,2])
    camera=torch.eye(4,requires_grad=True) # Intentionally starts on CPU.
    intrinsic=torch.tensor([[12.,0,8.],[0,12.,8.],[0,0,1.]],requires_grad=True)
    rgb,alpha,depth=render_gaussians_scene(gs,camera,intrinsic,16,16)
    assert rgb.device.type==device and alpha.max()>0
    (rgb.square().sum()+alpha.sum()+depth.sum()).backward()
    for value in [xyz,colors,camera,intrinsic]:
        assert torch.isfinite(value.grad).all() and value.grad.abs().sum()>0
    sparse=SparseTensor(torch.ones(2,3,device=device),torch.tensor([[0,0,0,0],[0,1,1,1]],device=device,dtype=torch.int32))
    restored=_deserialize_value(_serialize_value({'gaussian':gs,'sparse':sparse}))
    rg=restored['gaussian'];rs=restored['sparse']
    for attr in ['get_xyz','get_rotation','get_scaling','get_opacity','get_features']:
        torch.testing.assert_close(getattr(rg,attr),getattr(gs,attr))
    assert rg.get_xyz.device.type==rs.device.type==device
    torch.testing.assert_close(rs.feats,sparse.feats)


@pytest.mark.parametrize('device',DEVICES)
def test_pointmap_background_uses_selected_device(monkeypatch,device):
    from genia.core.utils.gaussian import create_gaussians_from_pointmap
    monkeypatch.setenv('GENIA_DEVICE',device)
    pointmap=np.array([[[-.1,-.1,2],[.1,-.1,2.1]],
                       [[-.1,.1,2.2],[.1,.1,2.3]]],np.float32)
    image=np.full((2,2,3),128,np.uint8)
    intrinsics=np.array([[5,0,1],[0,5,1],[0,0,1]],np.float32)
    gs=create_gaussians_from_pointmap(image,pointmap,intrinsics)
    assert gs.get_xyz.device.type==device
    torch.testing.assert_close(gs.get_xyz.cpu(),torch.tensor(pointmap.reshape(-1,3)))
    assert torch.isfinite(gs.get_scaling).all() and (gs.get_scaling>0).all()
