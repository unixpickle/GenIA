"""Clip-space rasterization/attributes and silhouette coverage on CPU and MPS."""
import pytest
import torch
import nvdiffrast.torch as dr

DEVICES=['cpu']+(['mps'] if torch.backends.mps.is_available() else [])


def triangle(device='cpu',dtype=torch.float32):
    # Unequal w exercises perspective barycentrics and their pixel derivatives.
    pos=torch.tensor([[-.75,-.7,.2,1],[1.2,-.9,.6,1.5],[.0,.8,-.3,1]],device=device,dtype=dtype)
    tri=torch.tensor([[0,1,2]],device=device,dtype=torch.int32)
    return pos,tri


@pytest.mark.parametrize('device',DEVICES)
def test_raster_barycentric_derivatives(device):
    pos,tri=triangle(device)
    pos.requires_grad_()
    rast,db=dr.rasterize(dr.RasterizeCudaContext(),pos[None],tri,[13,17])
    valid=rast[0,...,3]>0
    y,x=valid.nonzero().unbind(-1)
    p=torch.stack([(x+.5)*2/17-1,(y+.5)*2/13-1],-1)
    bary=torch.cat([rast[0,y,x,:2],1-rast[0,y,x,:2].sum(-1,keepdim=True)],-1)
    clip=bary@pos
    torch.testing.assert_close(clip[:,:2]/clip[:,3:],p,atol=1e-6,rtol=1e-6)
    # Analytical raster pixel derivatives against finite differences in point position.
    from genia.core.ops.raster import _barycentrics
    face=pos[None].expand(len(p),-1,-1)
    eps=1e-3
    for axis,scale in [(0,2/17),(1,2/13)]:
        shift=torch.zeros_like(p); shift[:,axis]=eps*scale
        plus=_barycentrics(face,p+shift,13,17)[0]
        minus=_barycentrics(face,p-shift,13,17)[0]
        finite=(plus-minus)/(2*eps)
        torch.testing.assert_close(db[0,y,x][:,[axis,axis+2]],finite,atol=1e-4,rtol=2e-3)
    depth_grad=torch.autograd.grad(rast[...,2].sum(),pos,retain_graph=True)[0]
    assert (depth_grad==0).all()
    (rast[...,:2].square().sum()+db.square().sum()).backward()
    assert torch.isfinite(pos.grad).all() and pos.grad.abs().sum()>0


@pytest.mark.parametrize('device',DEVICES)
def test_interpolation_and_attribute_derivatives(device):
    pos,tri=triangle(device)
    rast,db=dr.rasterize(None,pos[None],tri,[7,9])
    attr=torch.tensor([[1.,2.,3.],[2.,4.,8.],[3.,1.,5.]],device=device,requires_grad=True)
    values,deriv=dr.interpolate(attr,rast,tri,db,diff_attrs=[0,2])
    b0,b1=rast[...,0:1],rast[...,1:2]
    expected=b0*attr[0]+b1*attr[1]+(1-b0-b1)*attr[2]
    expected=torch.where(rast[...,3:]>0,expected,0)
    torch.testing.assert_close(values,expected)
    for j,index in enumerate([0,2]):
        expected_da=(attr[0,index]-attr[2,index])*db[...,:2]+(attr[1,index]-attr[2,index])*db[...,2:]
        torch.testing.assert_close(deriv[...,2*j:2*j+2],expected_da)
    (values.square().mean()+deriv.square().mean()).backward()
    assert torch.isfinite(attr.grad).all()


@pytest.mark.parametrize('device',DEVICES)
def test_clip_all_frustum_planes_and_eye(device):
    pos,tri=triangle(device)
    cases=[pos.clone() for _ in range(5)]
    cases[0][0,3]=-.3
    cases[1][:,2]=-2 # fully behind near plane
    cases[2][:,2]=2.5 # fully beyond far plane
    cases[3][0,0]=-5
    cases[4][1,2]=3
    for i,p in enumerate(cases):
        rast,db=dr.rasterize(None,p[None],tri,[12,15])
        assert torch.isfinite(rast).all() and torch.isfinite(db).all()
        if i in [1,2]: assert (rast==0).all()
        else:
            mask=rast[...,3]>0
            assert mask.any()
            assert ((rast[...,2][mask]>=-1)&(rast[...,2][mask]<=1)).all()


def quad(boundary,device='cpu',dtype=torch.float32):
    pos=torch.tensor([[-1,-1,0,1],[boundary/2-1,-1,0,1],[boundary/2-1,1,0,1],[-1,1,0,1]],device=device,dtype=dtype)
    tri=torch.tensor([[0,1,2],[0,2,3]],dtype=torch.int32,device=device)
    return pos,tri


@pytest.mark.parametrize('device',DEVICES)
@pytest.mark.parametrize('boundary',[1.8,2.2])
@pytest.mark.parametrize('transpose',[False,True])
def test_antialias_rectangle_analytic_coverage(device,boundary,transpose):
    pos,tri=quad(boundary,device)
    # Put corners beyond the viewport so a second silhouette does not enter
    # the one-pixel footprint used by the upstream antialias approximation.
    pos[:,1]*=10
    if transpose: pos=pos[:,[1,0,2,3]]
    pos.requires_grad_()
    rast,_=dr.rasterize(None,pos[None],tri,[4,4])
    color=(rast[...,3:]>0).float().requires_grad_()
    out=dr.antialias(color,rast,pos[None],tri)
    expected=torch.tensor([1.,.8,0,0] if boundary<2 else [1.,1.,.2,0],device=device)
    expected=expected[None,:].expand(4,4)
    if transpose: expected=expected.T
    torch.testing.assert_close(out[0,...,0],expected,atol=2e-6,rtol=2e-6)
    out.sum().backward()
    assert torch.isfinite(pos.grad).all() and pos.grad.abs().sum()>0
    # Translating the right edge by one NDC unit adds W/2 pixels per row.
    torch.testing.assert_close(pos.grad[[1,2],int(transpose)].sum(),
                               pos.new_tensor(8.),atol=5e-4,rtol=1e-4)
    assert torch.isfinite(color.grad).all()


@pytest.mark.parametrize('device',DEVICES)
def test_antialias_internal_edge_and_empty(device):
    pos,tri=quad(4.,device)
    rast,_=dr.rasterize(None,pos[None],tri,[4,4])
    assert (rast[...,3]>0).all() # shared-edge pixel centers must have coverage
    colors=(rast[...,3:]==1).float()
    out=dr.antialias(colors,rast,pos[None],tri)
    torch.testing.assert_close(out,colors) # coplanar mesh edges are not silhouettes
    empty=tri[:0]
    er,ed=dr.rasterize(None,pos[None],empty,[4,4])
    assert not er.any() and not ed.any()
    values,da=dr.interpolate(pos[:,:2],er,empty,ed,'all')
    assert not values.any() and not da.any()


@pytest.mark.skipif(not torch.backends.mps.is_available(),reason='requires Metal')
def test_raster_and_antialias_cpu_mps_gradients():
    pos,tri=triangle();pos.requires_grad_()
    mp=pos.detach().to('mps').requires_grad_()
    outputs=[]
    for p,t in [(pos,tri),(mp,tri.to('mps'))]:
        r,d=dr.rasterize(None,p[None],t,[19,27])
        attrs=p[:,:3]*.2+.5
        color,_=dr.interpolate(attrs,r,t)
        outputs.append((r,d,dr.antialias(color,r,p[None],t)))
    for a,b in zip(outputs[0],outputs[1]): torch.testing.assert_close(a,b.cpu(),atol=2e-5,rtol=2e-4)
    for output in outputs: sum(v.square().sum() for v in output).backward()
    torch.testing.assert_close(mp.grad.cpu(),pos.grad,atol=2e-4,rtol=5e-4)


@pytest.mark.parametrize('device',DEVICES)
def test_range_mode_matches_instances(device):
    pos,tri=triangle(device)
    second=pos.clone();second[:,0]+=.2
    batched=torch.stack([pos,second]).requires_grad_()
    expected,ed=dr.rasterize(None,batched,tri,[8,11],grad_db=False)
    combined=batched.reshape(-1,4)
    faces=torch.cat([tri,tri+3])
    actual,ad=dr.rasterize(None,combined,faces,[8,11],
                         ranges=torch.tensor([[0,1],[1,1]]),grad_db=False)
    actual=actual.clone()
    actual[1,...,3]=torch.where(actual[1,...,3]>0,actual[1,...,3]-1,0)
    torch.testing.assert_close(actual,expected)
    torch.testing.assert_close(ad,ed)
    assert not ad.requires_grad and not ed.requires_grad


@pytest.mark.parametrize('device',DEVICES)
@pytest.mark.parametrize('textured',[False,True])
def test_pipeline_rgb_depth_image_coordinates(device,textured):
    from genia.core.utils.mesh_rendering import render_depth_and_alpha,render_rgba_and_depth
    # A frontoparallel patch in the top half of the image tests the GL/image Y flip.
    vertices=torch.tensor([[-.75,-.75,2.],[.75,-.75,2.],[.75,-.125,2.],[-.75,-.125,2.]],
                          device=device,requires_grad=True)
    faces=torch.tensor([[0,1,2],[0,2,3]],device=device,dtype=torch.int32)
    color=torch.tensor([[.2,.4,.8]]*4,device=device,requires_grad=True)
    kwargs={}
    if textured:
        uv=torch.tensor([[0.,0.],[1.,0.],[1.,1.],[0.,1.]],device=device,requires_grad=True)
        tex=torch.tensor([.2,.4,.8],device=device).expand(8,8,3).clone().requires_grad_()
        kwargs={'uv':uv,'tex':tex}
    args=(None,8.,8.,4.,4.,8,8)
    rgb,alpha,depth=render_rgba_and_depth(vertices,faces,color,*args,**kwargs)
    other_depth,other_alpha=render_depth_and_alpha(vertices,faces,*args)
    torch.testing.assert_close(depth,other_depth)
    torch.testing.assert_close(alpha,other_alpha)
    torch.testing.assert_close(depth,2*alpha)
    torch.testing.assert_close(rgb,alpha[...,None]*color[0],atol=1e-6,rtol=1e-6)
    assert alpha[:4].sum()>0 and alpha[4:].sum()==0
    weights=torch.arange(64,device=device).reshape(8,8)/64
    ((rgb.sum(-1)+depth+alpha)*weights).sum().backward()
    assert torch.isfinite(vertices.grad).all() and vertices.grad.abs().sum()>0
    if textured:
        assert torch.isfinite(uv.grad).all()
        assert torch.isfinite(tex.grad).all() and tex.grad.abs().sum()>0
    else:
        assert torch.isfinite(color.grad).all() and color.grad.abs().sum()>0
