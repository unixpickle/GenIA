"""Local perceptual models against LPIPS 0.1.4 with torchvision backbones."""
import copy
import importlib.util
from pathlib import Path
import sys

import pytest
import torch
import lpips

DEVICES=['cpu']+(['mps'] if torch.backends.mps.is_available() else [])


@pytest.fixture(scope='module',autouse=True)
def bounded_threads():
    previous=torch.get_num_threads();torch.set_num_threads(4)
    yield
    torch.set_num_threads(previous)


def reference():
    root=Path(__file__).parent/'reference/lpips'
    name='_genia_lpips_reference'
    if name+'.lpips' in sys.modules:return sys.modules[name+'.lpips']
    for module,path,search in [(name,root/'__init__.py',[str(root)]),
                                (name+'.lpips',root/'lpips.py',None)]:
        spec=importlib.util.spec_from_file_location(module,path,submodule_search_locations=search)
        loaded=importlib.util.module_from_spec(spec);sys.modules[module]=loaded
        spec.loader.exec_module(loaded)
    return loaded


@pytest.mark.parametrize('channels_last',[False,True])
@pytest.mark.parametrize('net',['vgg','alex','squeeze'])
@pytest.mark.parametrize('device',DEVICES)
def test_local_backbone_metric_and_image_gradients(net,device,channels_last):
    torch.manual_seed(232)
    expected=reference().LPIPS(net=net,pretrained=False,pnet_rand=True,verbose=False)
    actual=lpips.LPIPS(net=net,pretrained=False,pnet_rand=True,verbose=False)
    actual.load_state_dict(expected.state_dict(),strict=True)
    actual=actual.to(device)
    x=torch.rand(1,3,64,80,requires_grad=True)*2-1;x=x.detach().requires_grad_()
    if channels_last:
        x=x.detach().contiguous(memory_format=torch.channels_last).requires_grad_()
    target=torch.rand_like(x)*2-1
    mx=x.detach().to(device).requires_grad_()
    ref,rlayers=expected(x,target,retPerLayer=True)
    got,glayers=actual(mx,target.to(device),retPerLayer=True)
    for a,b in zip([got]+glayers,[ref]+rlayers):
        torch.testing.assert_close(a.cpu(),b,atol=3e-6,rtol=2e-4)
    ref.sum().backward();got.sum().backward()
    torch.testing.assert_close(mx.grad.cpu(),x.grad,atol=2e-6,rtol=2e-3)
    assert torch.isfinite(mx.grad).all() and mx.grad.abs().sum()>0
    # Full spatial output and the uncalibrated feature baseline share the trunk.
    for model in [expected,actual]:model.spatial=True;model.lpips=False
    torch.testing.assert_close(actual(mx,target.to(device)).cpu(),expected(x,target),atol=2e-5,rtol=3e-4)
    assert actual(mx,mx).count_nonzero()==0


@pytest.mark.parametrize('device',DEVICES)
def test_perceptual_wrapper_frozen_parameters_and_normalization(monkeypatch,device):
    from genia.core.utils.perceptual_loss import PerceptualLoss,crop_to_mask_bbox
    ctor=lpips.LPIPS
    monkeypatch.setattr(lpips,'LPIPS',lambda **kw:ctor(**kw,pretrained=False,pnet_rand=True))
    model=PerceptualLoss().to(device)
    assert not any(p.requires_grad for p in model.parameters())
    image=torch.rand(1,3,32,32,device=device,requires_grad=True)
    other=torch.rand_like(image)
    torch.testing.assert_close(model(image,other),model.model(image,other,normalize=True).mean())
    model(image,other).backward()
    assert torch.isfinite(image.grad).all() and image.grad.abs().sum()>0
    mask=torch.zeros(32,32,device=device);mask[5:25,7:27]=1
    crop,_=crop_to_mask_bbox(image,other,mask,margin=0)
    assert crop.shape==(1,3,20,20)
