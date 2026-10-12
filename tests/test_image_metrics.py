"""Evaluation borders/reductions and optimization SSIM values/gradients."""
import ast
from collections.abc import Sequence
from pathlib import Path
from typing import List, Optional, Union

import pytest
import torch
from torch import Tensor
from torch.nn import functional as F

from genia.core.ops.image_metrics import psnr, ssim
from pytorch_msssim import ssim as loss_ssim

DEVICES = ['cpu'] + (['mps'] if torch.backends.mps.is_available() else [])


def original_ssim(x, y):
    # Execute unmodified upstream functions without importing TorchMetrics and
    # its unrelated model dependencies. The preserved sources include provenance.
    namespace = dict(torch=torch, Tensor=Tensor, F=F, Sequence=Sequence,
                     List=List, Optional=Optional, Union=Union)
    root = Path(__file__).parent / 'reference/image_metrics'
    names = {'_gaussian', '_gaussian_kernel_2d', '_ssim_update'}
    for filename in ('torchmetrics_utils.py', 'torchmetrics_ssim.py'):
        tree = ast.parse((root / filename).read_text())
        tree.body = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in names]
        exec(compile(tree, str(root / filename), 'exec'), namespace)
    return namespace['_ssim_update'](x, y, data_range=1.).mean()


@pytest.fixture(scope='module', autouse=True)
def bounded_threads():
    before = torch.get_num_threads()
    torch.set_num_threads(4)
    yield
    torch.set_num_threads(before)


@pytest.mark.parametrize('device', DEVICES)
@pytest.mark.parametrize('flat', [False, True])
def test_evaluation_matches_original_values_and_gradients(device, flat):
    torch.manual_seed(837)
    x = torch.rand(2, 3, 17, 23)
    y = torch.rand_like(x)
    if flat:
        x[:, :, 4:13, 4:18] = .5
        y[:, :, 4:13, 4:18] = .7
    x = x.contiguous(memory_format=torch.channels_last).requires_grad_()
    mx = x.detach().to(device).requires_grad_()
    ref, actual = original_ssim(x, y), ssim(mx, y.to(device))
    torch.testing.assert_close(actual.cpu(), ref, atol=3e-6, rtol=3e-5)
    ref.backward(); actual.backward()
    torch.testing.assert_close(mx.grad.cpu(), x.grad, atol=2e-7, rtol=2e-4)
    expected_psnr = 10 * torch.log10(torch.tensor(x.numel()) / (x-y).square().sum())
    torch.testing.assert_close(psnr(mx, y.to(device)).cpu(), expected_psnr)
    assert psnr(mx, mx).isinf()
    torch.testing.assert_close(ssim(mx, mx), torch.ones((), device=device))


@pytest.mark.parametrize('device', DEVICES)
def test_optimization_valid_window_and_refinement_gradient(device, monkeypatch):
    from genia.core.utils.pose_refinement import loss_terms as refinement
    # Having the optional CUDA package installed must not dispatch MPS/CPU to it.
    def no_cuda(*args, **kwargs):
        raise AssertionError('CUDA-only SSIM called on a non-CUDA device')
    monkeypatch.setattr(refinement, '_fused_ssim', no_cuda)
    torch.manual_seed(922)
    x = torch.rand(19, 23, 3, requires_grad=True)
    y = .8*x.detach() + .2*torch.rand_like(x)
    # Independent full 2D valid convolution, compared with the upstream separable
    # convolution used in the actual loss. No reflected border or variance clamp.
    g = torch.exp(-torch.arange(-5, 6).float().square() / (2*1.5**2))
    g /= g.sum()
    kernel = (g[:, None]*g[None, :]).expand(3, 1, 11, 11)
    a, b = [v.permute(2, 0, 1)[None].contiguous() for v in (x, y)]
    ax, by, xx, yy, xy = [F.conv2d(v, kernel, groups=3) for v in (a, b, a*a, b*b, a*b)]
    ref = 1 - (((2*ax*by+.0001)/(ax*ax+by*by+.0001)) *
               ((2*(xy-ax*by)+.0009)/(xx-ax*ax+yy-by*by+.0009))).mean()
    mx = x.detach().to(device).requires_grad_()
    actual = refinement._compute_ssim_loss(mx, y.to(device))
    torch.testing.assert_close(actual.cpu(), ref, atol=3e-6, rtol=3e-5)
    ref.backward(); actual.backward()
    torch.testing.assert_close(mx.grad.cpu(), x.grad, atol=3e-7, rtol=3e-4)
    torch.testing.assert_close(loss_ssim(a.detach(), a.detach(), data_range=1.), torch.tensor(1.))


@pytest.mark.parametrize('device', DEVICES)
def test_evaluator_local_lpips_batch_reduction_and_sequence(device, monkeypatch):
    import lpips
    from genia.core.evaluator import Evaluator
    ctor = lpips.LPIPS
    monkeypatch.setattr(lpips, 'LPIPS', lambda **kw: ctor(**kw, pretrained=False, pnet_rand=True))
    evaluator = Evaluator(torch.device(device))
    assert evaluator._lpip_metric is None
    x, y = torch.rand(2, 3, 64, 64, device=device), torch.rand(2, 3, 64, 64, device=device)
    expected = evaluator.lpip_metric(y, x, normalize=False).mean().item()
    got = evaluator.evaluate_frame(x, y)
    assert got['lpip'] == pytest.approx(expected)
    assert not any(p.requires_grad for p in evaluator.lpip_metric.parameters())
    sequence = evaluator.evaluate_sequence([x, x], [y, y])
    assert sequence['lpip_values'] == pytest.approx([expected, expected])
    assert sequence['lpip_std'] == 0
    with pytest.raises(ValueError, match='empty frame lists'):
        evaluator.evaluate_sequence([], [])
