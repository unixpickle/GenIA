"""GenIA's image metrics, preserving TorchMetrics 1.8.2 defaults.

SSIM is adapted from Lightning-AI/torchmetrics (Apache-2.0; see
vendor/licenses/torchmetrics.LICENSE). Evaluation uses reflected borders and
clamped variances, unlike the valid-window SSIM used by the optimization loss.
"""
import torch
from torch.nn import functional as F


def psnr(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Whole-batch PSNR for images in [0, 1]."""
    return -10 * torch.log10((pred - target).square().mean())


def ssim(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Mean RGB SSIM: 11x11 Gaussian, sigma=1.5, data_range=1."""
    if pred.ndim != 4 or pred.shape != target.shape:
        raise ValueError("SSIM expects matching BCHW images")
    pred = pred.contiguous()
    target = target.to(pred.dtype).contiguous()
    dist = torch.arange(-5, 6, dtype=pred.dtype, device=pred.device)
    gaussian = torch.exp(-(dist / 1.5).square() / 2)
    gaussian = gaussian / gaussian.sum()
    kernel = (gaussian[:, None] @ gaussian[None, :]).expand(pred.shape[1], 1, 11, 11)
    x, y = (F.pad(t, (5, 5, 5, 5), mode="reflect") for t in (pred, target))
    moments = F.conv2d(torch.cat((x, y, x*x, y*y, x*y)), kernel, groups=x.shape[1])
    mx, my, xx, yy, xy = moments.split(x.shape[0])
    vx, vy = (xx - mx.square()).clamp_min(0), (yy - my.square()).clamp_min(0)
    covariance = xy - mx*my
    value = ((2*mx*my + 0.01**2) * (2*covariance + 0.03**2)) / (
        (mx.square() + my.square() + 0.01**2) * (vx + vy + 0.03**2))
    return value.reshape(pred.shape[0], -1).mean(-1).mean()
