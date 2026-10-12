"""Local SAM FlexiCubes geometry, topology, color and gradient parity."""
import os
os.environ['LIDRA_SKIP_INIT']='1'
import pytest
import torch
from sam3d_objects.model.backbone.tdfy_dit.representations.mesh.flexicubes.flexicubes import FlexiCubes
from sam3d_objects.model.backbone.tdfy_dit.representations.mesh.utils_cube import construct_dense_grid, cube_corners
from sam3d_objects.model.backbone.tdfy_dit.representations.mesh.cube2mesh import SparseFeatures2Mesh
from sam3d_objects.model.backbone.tdfy_dit.modules.sparse import SparseTensor

pytestmark=pytest.mark.skipif(not torch.backends.mps.is_available(),reason='requires Metal')


@pytest.mark.parametrize('training',[False,True])
def test_flexicubes_matches_cpu(training):
    grid,cubes=construct_dense_grid(8,'cpu')
    vertices=(grid.float()/8-.5).requires_grad_()
    sdf=(vertices.detach().norm(dim=-1)-.31).requires_grad_()
    colors=(vertices.detach()+.5).requires_grad_()
    inputs=[vertices,sdf,colors]
    metal=[x.detach().to('mps').requires_grad_() for x in inputs]
    ref=FlexiCubes('cpu')(vertices,sdf,cubes,8,voxelgrid_colors=colors,training=training)
    got=FlexiCubes('mps')(metal[0],metal[1],cubes.to('mps'),8,voxelgrid_colors=metal[2],training=training)
    assert ref[0].numel()>0
    for a,b in zip(got,ref): torch.testing.assert_close(a.cpu(),b,atol=2e-6,rtol=2e-5)
    sum(ref[i].square().sum() for i in [0,2,3]).backward()
    sum(got[i].square().sum() for i in [0,2,3]).backward()
    for a,b in zip(metal,inputs): torch.testing.assert_close(a.grad.cpu(),b.grad,atol=2e-5,rtol=2e-4)


def test_sparse_features_mesh_moves_with_input():
    extractor=SparseFeatures2Mesh(res=4,use_color=False)
    coords=torch.stack(torch.meshgrid(*[torch.arange(4)]*3,indexing='ij'),-1).reshape(-1,3)
    corner_positions=(coords[:,None]+cube_corners[None]).float()/4-.5
    feats=torch.zeros(len(coords),extractor.feats_channels)
    feats[:,:8]=corner_positions.norm(dim=-1)-.35+.25
    # Break exact duplicate-corner ties in the absolute-value regularizer;
    # its derivative at zero is sensitive to CPU/MPS reduction roundoff.
    torch.manual_seed(12)
    feats += torch.randn_like(feats)*.002
    feats.requires_grad_()
    indices=torch.cat([torch.zeros(len(coords),1,dtype=torch.long),coords],1).int()
    ref=extractor(SparseTensor(feats,indices),training=True)
    mf=feats.detach().to('mps').requires_grad_()
    got=extractor(SparseTensor(mf,indices.to('mps')),training=True)
    assert got.success and extractor.reg_v.device.type=='mps'
    torch.testing.assert_close(got.faces.cpu(),ref.faces)
    torch.testing.assert_close(got.vertices.cpu(),ref.vertices,atol=2e-6,rtol=2e-5)
    (got.vertices.square().sum()+got.reg_loss).backward()
    (ref.vertices.square().sum()+ref.reg_loss).backward()
    torch.testing.assert_close(mf.grad.cpu(),feats.grad,atol=1e-5,rtol=1e-4)


def test_slat_mesh_decoder_on_mps():
    import copy
    from sam3d_objects.model.backbone.tdfy_dit.models.structured_latent_vae.decoder_mesh import SLatMeshDecoder
    torch.manual_seed(93)
    previous=torch.get_num_threads(); torch.set_num_threads(4)
    try:
        model=SLatMeshDecoder(resolution=2,model_channels=256,latent_channels=8,
            num_blocks=1,num_heads=4,attn_mode='full',representation_config={'use_color':True}).eval()
        torch.nn.init.normal_(model.out_layer.weight,std=.01)
        metal=copy.deepcopy(model).to('mps')
        coords=torch.tensor([[0,0,0,0],[0,1,1,1]],dtype=torch.int32)
        feats=torch.randn(2,8,requires_grad=True)
        mf=feats.detach().to('mps').requires_grad_()
        ref=model(SparseTensor(feats,coords))[0]
        got=metal(SparseTensor(mf,coords.to('mps')))[0]
        assert got.success
        torch.testing.assert_close(got.faces.cpu(),ref.faces)
        torch.testing.assert_close(got.vertices.cpu(),ref.vertices,atol=2e-5,rtol=2e-4)
        torch.testing.assert_close(got.vertex_attrs.cpu(),ref.vertex_attrs,atol=2e-5,rtol=2e-4)
        (got.vertices.square().sum()+got.vertex_attrs.square().sum()).backward()
        (ref.vertices.square().sum()+ref.vertex_attrs.square().sum()).backward()
        torch.testing.assert_close(mf.grad.cpu(),feats.grad,atol=2e-4,rtol=3e-3)
    finally: torch.set_num_threads(previous)
