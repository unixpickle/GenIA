import pytest
import torch

from sam3d_objects.model.backbone.tdfy_dit.modules.sparse.attention.masked_sdpa import masked_sdpa
from sam3d_objects.model.backbone.tdfy_dit.modules import sparse as sp

DEVICES=['cpu']+(['mps'] if torch.backends.mps.is_available() else [])


@pytest.mark.parametrize('device',DEVICES)
@pytest.mark.parametrize('dtype',[torch.float32,torch.float16,torch.bfloat16])
@pytest.mark.parametrize('stacked',[False,True])
@pytest.mark.parametrize('heads',[1,2])
def test_visibility_bias_blocks_chunking_and_gradients(device,dtype,stacked,heads):
    torch.manual_seed(416)
    inputs=[torch.randn(1,n,2,c,dtype=dtype,requires_grad=True) for n,c in [(5,4),(7,4),(7,3)]]
    bias=torch.randn((2,heads,3,4) if stacked else (1,heads,5,7),requires_grad=True)
    actual_inputs=[t.detach().to(device).requires_grad_() for t in inputs]
    actual_bias=bias.detach().to(device).requires_grad_()
    outputs=[];qs=ks=0
    q,k,v=inputs
    for i,(nq,nk) in enumerate(zip([2,3],[4,3])):
        b=bias[i:i+1,:,:nq,:nk] if stacked else bias[:,:,qs:qs+nq,ks:ks+nk]
        qv,kv,vv=[t[:,start:start+n].transpose(1,2).float() for t,start,n in [(q,qs,nq),(k,ks,nk),(v,ks,nk)]]
        probabilities=(qv@kv.transpose(-1,-2)/2+b.to(dtype).float()).softmax(-1)
        outputs.append((probabilities@vv).transpose(1,2).to(dtype))
        qs+=nq;ks+=nk
    reference=torch.cat(outputs,dim=1)[0]
    actual=masked_sdpa(*actual_inputs,[2,3],[4,3],attn_bias=actual_bias,query_chunk_size=2)
    tol=2e-5 if dtype==torch.float32 else .005 if dtype==torch.float16 else .04
    torch.testing.assert_close(actual.cpu(),reference,atol=tol,rtol=tol)
    grad=torch.randn_like(reference)
    reference.backward(grad);actual.backward(grad.to(device))
    for a,b in zip(actual_inputs+[actual_bias],inputs+[bias]):
        torch.testing.assert_close(a.grad.cpu(),b.grad,atol=tol,rtol=tol)


@pytest.mark.parametrize('device',DEVICES)
def test_sparse_cross_attention_public_biased_path(device):
    torch.manual_seed(573)
    coords=torch.tensor([[0,0,0,0],[0,1,0,0],[1,0,0,0],[1,1,0,0],[1,2,0,0]],dtype=torch.int32,device=device)
    q=torch.randn(5,2,4,device=device,requires_grad=True)
    kv=torch.randn(2,4,2,2,4,device=device,requires_grad=True)
    bias=torch.randn(2,1,3,4,device=device,requires_grad=True)
    sparse=sp.SparseTensor(q,coords)
    got=sp.sparse_scaled_dot_product_attention(sparse,kv,attn_bias=bias)
    expected=[]
    for i,sl in enumerate(sparse.layout):
        key,value=kv[i].unbind(1)
        score=torch.einsum('qhd,khd->hqk',q[sl],key)/2+bias[i,:,:sl.stop-sl.start,:]
        expected.append(torch.einsum('hqk,khd->qhd',score.softmax(-1),value))
    torch.testing.assert_close(got.feats,torch.cat(expected),atol=2e-5,rtol=2e-5)
    got.feats.square().sum().backward()
    assert all(t.grad is not None and torch.isfinite(t.grad).all() for t in [q,kv,bias])
