import pytest
import torch
from genia.core.ops.unique import unique

pytestmark=pytest.mark.skipif(not torch.backends.mps.is_available(),reason='requires MPS')


@pytest.mark.parametrize('dtype',[torch.int64,torch.int32,torch.float32])
@pytest.mark.parametrize('shape,dim',[((23,3),0),((3,23),1),((2,9,3),1),((0,3),0),((9,),0)])
def test_unique(dtype,shape,dim):
    torch.manual_seed(26)
    x=torch.randint(-2,3,shape).to(dtype)
    ref=torch.unique(x,dim=dim,return_inverse=True,return_counts=True)
    got=unique(x.to('mps'),dim=dim,return_inverse=True,return_counts=True)
    for a,b in zip(got,ref): torch.testing.assert_close(a.cpu(),b)
    torch.testing.assert_close(got[0].index_select(dim,got[1]).cpu(),x)


def test_unique_large_integer_coordinates():
    x=torch.tensor([[2**45,3],[-2**42,4],[2**45,3],[2**45,2]])
    ref=torch.unique(x,dim=0,return_inverse=True,return_counts=True)
    got=unique(x.to('mps'),dim=0,return_inverse=True,return_counts=True)
    for a,b in zip(got,ref): torch.testing.assert_close(a.cpu(),b)
