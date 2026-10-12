import pytest
import torch
import torch.nn.functional as F
import nvdiffrast.torch as dr

DEVICES=['cpu']+(['mps'] if torch.backends.mps.is_available() else [])


@pytest.mark.parametrize('device',DEVICES)
@pytest.mark.parametrize('boundary,padding',[('clamp','border'),('zero','zeros')])
def test_bilinear_texture_grid_reference(device,boundary,padding):
    torch.manual_seed(52)
    tex=torch.rand(1,4,8,3,requires_grad=True)
    uv=(torch.rand(2,3,5,2)*1.4-.2).requires_grad_()
    mt=tex.detach().to(device).requires_grad_();mu=uv.detach().to(device).requires_grad_()
    got=dr.texture(mt,mu,filter_mode='linear',boundary_mode=boundary)
    ref=F.grid_sample(tex.permute(0,3,1,2).expand(2,-1,-1,-1),uv*2-1,padding_mode=padding,align_corners=False).permute(0,2,3,1)
    torch.testing.assert_close(got.cpu(),ref,atol=2e-6,rtol=2e-6)
    got.square().sum().backward();ref.square().sum().backward()
    torch.testing.assert_close(mt.grad.cpu(),tex.grad,atol=3e-6,rtol=3e-5)
    torch.testing.assert_close(mu.grad.cpu(),uv.grad,atol=5e-6,rtol=3e-5)


@pytest.mark.parametrize('device',DEVICES)
def test_texture_wrap_and_nearest(device):
    tex=torch.tensor([[[[0.],[1.]],[[2.],[3.]]]],device=device)
    uv=torch.tensor([[[[0.,0.],[1.,1.],[-1.,2.],[.25,.25],[.75,.75]]]],device=device)
    actual=dr.texture(tex,uv,filter_mode='linear')
    torch.testing.assert_close(actual.flatten(),torch.tensor([1.5,1.5,1.5,0.,3.],device=device))
    actual=dr.texture(tex,uv,filter_mode='nearest')
    torch.testing.assert_close(actual.flatten(),torch.tensor([0.,0.,0.,0.,3.],device=device))


@pytest.mark.parametrize('device',DEVICES)
def test_mipmap_levels_and_trilinear(device):
    tex=torch.arange(64,device=device,dtype=torch.float32).reshape(1,8,8,1).requires_grad_()
    uv=torch.tensor([[[[.35,.45],[.61,.23]]]],device=device,requires_grad=True)
    bias=torch.tensor([[[1.25,2.5]]],device=device,requires_grad=True)
    levels=dr.texture_construct_mip(tex)
    assert [tuple(x.shape[1:3]) for x in levels]==[(4,4),(2,2),(1,1)]
    got=dr.texture(tex,uv,mip_level_bias=bias)
    l1=dr.texture(levels[0],uv,filter_mode='linear')
    l2=dr.texture(levels[1],uv,filter_mode='linear')
    l3=dr.texture(levels[2],uv,filter_mode='linear')
    expected=torch.cat([(.75*l1+.25*l2)[:,:,:1],(.5*l2+.5*l3)[:,:,1:]],dim=2)
    torch.testing.assert_close(got,expected)
    got.sum().backward()
    assert torch.isfinite(tex.grad).all() and torch.isfinite(uv.grad).all() and torch.isfinite(bias.grad).all()


def test_texture_derivative_gradcheck():
    torch.manual_seed(9)
    tex=torch.rand(1,8,8,2,dtype=torch.float64,requires_grad=True)
    uv=torch.tensor([[[[.37,.43]]]],dtype=torch.float64,requires_grad=True)
    da=torch.tensor([[[[.31,.03,-.02,.17]]]],dtype=torch.float64,requires_grad=True)
    assert torch.autograd.gradcheck(lambda t,u,d:dr.texture(t,u,d),(tex,uv,da),atol=1e-5,rtol=1e-4)


@pytest.mark.skipif(not torch.backends.mps.is_available(),reason='requires Metal')
def test_mipmap_cpu_mps_derivatives():
    torch.manual_seed(8)
    cpu=[torch.rand(1,8,16,3,requires_grad=True),torch.rand(2,5,7,2,requires_grad=True),
         (torch.rand(2,5,7,4)*.2).requires_grad_()]
    metal=[x.detach().to('mps').requires_grad_() for x in cpu]
    a,b=dr.texture(*cpu),dr.texture(*metal)
    torch.testing.assert_close(a,b.cpu(),atol=1e-6,rtol=1e-5)
    a.square().sum().backward();b.square().sum().backward()
    for x,y in zip(cpu,metal):torch.testing.assert_close(x.grad,y.grad.cpu(),atol=3e-5,rtol=2e-4)


@pytest.mark.parametrize('device',DEVICES)
def test_isotropic_mip_footprint_gradient(device):
    torch.manual_seed(51)
    tex=torch.rand(1,8,8,1,device=device)
    uv=torch.tensor([[[[.3,.4]]]],device=device)
    derivatives=torch.tensor([[[[.3,0.,0.,.3]]]],device=device,requires_grad=True)
    out=dr.texture(tex,uv,derivatives)
    out.sum().backward()
    # Upstream defines zero derivative when the principal-axis equation is singular.
    torch.testing.assert_close(derivatives.grad,torch.zeros_like(derivatives))
