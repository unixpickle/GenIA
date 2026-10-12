import pytest
import torch

from genia.core.ops.reductions import nanmedian

DEVICES = ['cpu'] + (['mps'] if torch.backends.mps.is_available() else [])


@pytest.mark.parametrize('device', DEVICES)
@pytest.mark.parametrize('dtype', [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize('dim', [None, -1, 0])
def test_nanmedian_values_and_gradients(device, dtype, dim):
    # Noncontiguous axes, odd/even valid counts, infinities and all-NaN slices.
    x = torch.tensor([[3., float('nan'), 2., 5., 1., 7.],
                      [float('nan')]*6,
                      [float('inf'), 10., 9., 12., 14., -float('inf')]], dtype=dtype).T.requires_grad_()
    mx = x.detach().to(device).requires_grad_()
    ref = torch.nanmedian(x) if dim is None else torch.nanmedian(x, dim=dim, keepdim=True).values
    got = nanmedian(mx, dim=dim, keepdim=True)
    torch.testing.assert_close(got.cpu(), ref, equal_nan=True)
    ref.sum().backward();got.sum().backward()
    torch.testing.assert_close(mx.grad.cpu(), x.grad)


@pytest.mark.parametrize('device', DEVICES)
def test_nanmedian_ties_empty_and_scalar(device):
    for values in ([1., 2., 2., 3.], [float('nan')]*3, []):
        x=torch.tensor(values,requires_grad=True)
        mx=x.detach().to(device).requires_grad_()
        ref=torch.nanmedian(x);got=nanmedian(mx)
        torch.testing.assert_close(got.cpu(),ref,equal_nan=True)
        ref.backward();got.backward()
        torch.testing.assert_close(mx.grad.cpu(),x.grad)
    x=torch.tensor(3.,device=device,requires_grad=True)
    got=nanmedian(x,dim=0,keepdim=True)
    assert got.shape==()
    got.backward();assert x.grad==1
    with pytest.raises(IndexError):nanmedian(torch.empty(2,0,device=device),dim=1)
    assert nanmedian(torch.empty(0,2,device=device),dim=1).shape==(0,)


@pytest.mark.parametrize('device', DEVICES)
def test_sam_pointmap_normalizer(device):
    from sam3d_objects.data.dataset.tdfy.img_and_mask_transforms import ObjectCentricSSI
    torch.manual_seed(518)
    pointmap=torch.randn(3,12,16)
    pointmap[:,0,:]=torch.nan
    mask=torch.ones(1,12,16)
    normalizer=ObjectCentricSSI(use_scene_scale=True)
    expected=normalizer.normalize(pointmap,mask)
    actual=normalizer.normalize(pointmap.to(device),mask.to(device))
    for a,b in zip(actual,expected):
        torch.testing.assert_close(a.cpu(),b,equal_nan=True)
