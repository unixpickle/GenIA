# Copyright (c) Meta Platforms, Inc. and affiliates.

"""Differentiable image, silhouette, depth, and normal loss primitives."""

from __future__ import annotations

from typing import Optional, Tuple

import torch
from genia.core.ops.resize import interpolate
from genia.core.utils.config import LossConfig

try:
    from fused_ssim import fused_ssim as _fused_ssim
except Exception:
    _fused_ssim = None


def _compute_center_of_mass_loss(
    alpha: torch.Tensor,
    mask: torch.Tensor,
    occlusion_robust: bool = False,
) -> torch.Tensor:
    """
    Compute center-of-mass alignment loss between rendered alpha and GT mask.

    Standard mode: compares centroid of the full predicted alpha vs centroid
    of the GT mask. Provides useful gradients even with zero overlap.

    Occlusion-robust mode: compares centroid of the INTERSECTION (predicted
    alpha masked by GT mask) vs centroid of GT mask. This prevents the
    predicted centroid from being biased by alpha in occluded regions where
    the GT mask is absent.

    Parameters
    ----------
    alpha : torch.Tensor
        Rendered alpha/opacity map, shape (H, W), values in [0, 1].
    mask : torch.Tensor
        Ground truth binary mask, shape (H, W), boolean or float.
    occlusion_robust : bool
        If True, compute render centroid from intersection of alpha and GT mask.

    Returns
    -------
    torch.Tensor
        Scalar loss representing squared distance between centroids,
        normalized by image diagonal squared.
    """
    H, W = alpha.shape
    device = alpha.device

    # Create coordinate grids
    y_coords, x_coords = torch.meshgrid(
        torch.arange(H, device=device, dtype=torch.float32),
        torch.arange(W, device=device, dtype=torch.float32),
        indexing="ij",
    )

    # Compute GT mask center of mass (same in both modes)
    mask_float = mask.float()
    mask_sum = mask_float.sum() + 1e-6
    gt_cx = (mask_float * x_coords).sum() / mask_sum
    gt_cy = (mask_float * y_coords).sum() / mask_sum

    if occlusion_robust:
        # Compute render centroid from intersection only (alpha * mask)
        visible_render = alpha * mask_float
        visible_sum = visible_render.sum() + 1e-6
        render_cx = (visible_render * x_coords).sum() / visible_sum
        render_cy = (visible_render * y_coords).sum() / visible_sum
    else:
        # Standard: centroid of full predicted alpha
        alpha_sum = alpha.sum() + 1e-6
        render_cx = (alpha * x_coords).sum() / alpha_sum
        render_cy = (alpha * y_coords).sum() / alpha_sum

    # Squared distance between centers, normalized by image diagonal
    diagonal = (H**2 + W**2) ** 0.5
    com_loss = ((render_cx - gt_cx) ** 2 + (render_cy - gt_cy) ** 2) / (diagonal**2)

    return com_loss


def _compute_signed_distance_transform(
    mask: torch.Tensor,
) -> torch.Tensor:
    """
    Compute signed distance transform of a binary mask.

    The SDT is positive outside the mask (distance to nearest mask pixel)
    and negative inside the mask (distance to nearest non-mask pixel).
    This creates a smooth field that guides optimization toward the mask.

    Parameters
    ----------
    mask : torch.Tensor
        Binary mask, shape (H, W), boolean or float.

    Returns
    -------
    torch.Tensor
        Signed distance transform, shape (H, W). Positive outside mask,
        negative inside mask.

    Notes
    -----
    This function uses scipy.ndimage.distance_transform_edt and is not
    differentiable. The SDT should be precomputed once per frame.
    """
    from scipy.ndimage import distance_transform_edt

    mask_np = mask.detach().cpu().numpy().astype(bool)

    # Distance from outside points to nearest inside point
    dist_outside = distance_transform_edt(~mask_np)

    # Distance from inside points to nearest outside point
    dist_inside = distance_transform_edt(mask_np)

    # Signed: positive outside, negative inside
    sdt = dist_outside - dist_inside

    return torch.from_numpy(sdt).float().to(mask.device)


def _compute_sdt_loss(
    alpha: torch.Tensor,
    sdt: torch.Tensor,
    occlusion_robust: bool = False,
) -> torch.Tensor:
    """
    Compute signed distance transform loss for silhouette alignment.

    Standard mode: penalizes rendered alpha weighted by the signed distance
    to the GT mask boundary. Pixels outside the mask contribute positive loss,
    pixels inside contribute negative loss (reward).

    Occlusion-robust mode ("interior-pull"): only penalizes missing coverage
    inside the GT mask. For each pixel inside the GT mask (sdt < 0), the loss
    is ``(1 - alpha) * |sdt|``, pulling alpha toward 1 proportionally to depth
    inside the mask. Pixels outside the GT mask are ignored entirely, so
    predicted alpha extending into occluded regions is not penalized.

    Parameters
    ----------
    alpha : torch.Tensor
        Rendered alpha/opacity map, shape (H, W), values in [0, 1].
    sdt : torch.Tensor
        Precomputed signed distance transform of GT mask, shape (H, W).
        Positive outside mask, negative inside mask.
    occlusion_robust : bool
        If True, use asymmetric interior-pull loss.

    Returns
    -------
    torch.Tensor
        Scalar loss. Lower values indicate better alignment.
    """
    H, W = alpha.shape
    diagonal = (H**2 + W**2) ** 0.5
    sdt_normalized = sdt / diagonal

    if occlusion_robust:
        # Interior-pull: for pixels INSIDE the GT mask (sdt < 0),
        # penalize (1 - alpha) proportional to how deep inside the mask they are.
        # Pixels outside the GT mask are ignored entirely.
        inside_mask = (sdt < 0).float()
        interior_depth = (-sdt_normalized).clamp(min=0)  # |sdt|/diagonal, nonzero only inside
        inside_sum = inside_mask.sum() + 1e-6
        sdt_loss = ((1.0 - alpha) * interior_depth * inside_mask).sum() / inside_sum
        return sdt_loss

    # Standard: alpha-weighted average of SDT values
    alpha_sum = alpha.sum() + 1e-6
    sdt_loss = (alpha * sdt_normalized).sum() / alpha_sum
    return sdt_loss


def _compute_soft_iou_loss(
    alpha: torch.Tensor,
    mask: torch.Tensor,
    occlusion_robust: bool = False,
) -> torch.Tensor:
    """
    Compute soft intersection-over-union loss for silhouette matching.

    Soft IoU provides a differentiable approximation to the IoU metric,
    treating alpha values as soft membership. This loss is effective for
    fine-grained shape matching once the rendered object is roughly
    aligned with the target mask.

    When ``occlusion_robust=True``, uses a coverage loss instead: only
    penalizes GT mask pixels not covered by the prediction, without
    penalizing predicted alpha that extends beyond the GT mask (which may
    correspond to occluded regions of the object).

    Parameters
    ----------
    alpha : torch.Tensor
        Rendered alpha/opacity map, shape (H, W), values in [0, 1].
    mask : torch.Tensor
        Ground truth binary mask, shape (H, W), boolean or float.
    occlusion_robust : bool
        If True, use asymmetric coverage loss instead of symmetric IoU.

    Returns
    -------
    torch.Tensor
        Scalar loss in [0, 1]. 0 = perfect overlap/coverage, 1 = no overlap.
    """
    mask_float = mask.float()

    if occlusion_robust:
        # Coverage loss: penalizes GT pixels not covered by prediction.
        # Does NOT penalize prediction extending beyond GT (occluded regions).
        intersection = (alpha * mask_float).sum()
        coverage = intersection / (mask_float.sum() + 1e-6)
        return 1.0 - coverage

    # Standard symmetric soft IoU
    intersection = (alpha * mask_float).sum()
    union = alpha.sum() + mask_float.sum() - intersection + 1e-6
    iou = intersection / union
    return 1.0 - iou


def _compute_ssim_loss(
    rgb: torch.Tensor,
    gt_masked: torch.Tensor,
) -> torch.Tensor:
    """
    Compute SSIM (Structural Similarity) loss between rendered and GT images.

    SSIM captures perceptual similarity by comparing luminance, contrast, and
    structure. It's more robust to small misalignments than pixel-wise losses
    and provides better gradients for pose optimization.

    Parameters
    ----------
    rgb : torch.Tensor
        Rendered RGB image, shape (H, W, 3), values in [0, 1].
    gt_masked : torch.Tensor
        Ground truth RGB image with background masked to black,
        shape (H, W, 3), values in [0, 1].

    Returns
    -------
    torch.Tensor
        Scalar SSIM loss in [0, 1]. 0 = identical, 1 = completely different.

    Notes
    -----
    Uses the fused-CUDA SSIM kernel (fused_ssim) when available — 5-10x faster
    than pytorch_msssim and numerically identical with ``padding="valid"``
    (same 11x11 sigma=1.5 window, valid convolution). Falls back to
    pytorch_msssim if fused_ssim isn't installed.
    """
    # Convert from (H, W, C) to (B, C, H, W). fused_ssim needs contiguous input.
    rgb_bchw = rgb.permute(2, 0, 1).unsqueeze(0).contiguous()
    gt_bchw = gt_masked.permute(2, 0, 1).unsqueeze(0).contiguous()

    if _fused_ssim is not None and rgb_bchw.device.type == "cuda":
        # padding="valid" matches pytorch_msssim's default (no border padding);
        # train= gates the backward buffers to when a backward can actually run.
        ssim_value = _fused_ssim(
            rgb_bchw, gt_bchw, padding="valid", train=torch.is_grad_enabled()
        )
    else:
        from pytorch_msssim import ssim

        ssim_value = ssim(rgb_bchw, gt_bchw, data_range=1.0, size_average=True)

    ssim_loss = torch.clamp(1.0 - ssim_value, 0.0, 1.0)

    return ssim_loss


def _l1_loss(rgb: torch.Tensor, gt_image: torch.Tensor) -> torch.Tensor:
    """
    Compute L1 loss between rendered and ground truth images.

    Parameters
    ----------
    rgb : torch.Tensor
        Rendered RGB image, shape (H, W, 3), values in [0, 1].
    gt_image : torch.Tensor
        Ground truth RGB image, shape (H, W, 3), values in [0, 1].

    Returns
    -------
    torch.Tensor
        Scalar L1 loss.
    """
    return (rgb - gt_image).abs().mean()


def _depth_l1_term(
    pred_depth: torch.Tensor,
    gt_depth: torch.Tensor,
    gt_mask: torch.Tensor,
    *,
    mask_only: bool = False,
    valid_mask: Optional[torch.Tensor] = None,
    return_error_map: bool = False,
):
    """
    Unweighted absolute depth L1 term.

    Zeroes GT depth outside ``gt_mask`` *before* the subtract, so
    out-of-mask pixels contribute ``|pred_depth − 0|`` (containment
    pressure).  When ``mask_only`` is ``True`` the error is additionally
    masked to in-mask pixels (bg contribution dropped entirely) and the
    reduction is a **proper masked mean** — denominator is the count of
    pixels that actually contributed, not ``H·W``.  When ``valid_mask``
    is provided, the error is further restricted to dataset-valid pixels
    (e.g. depth holes masked out) and the denominator counts only
    in-mask AND in-valid pixels.

    Caller multiplies by the loss weight.  Shape contract: all tensors
    are ``(H, W)``; ``gt_mask`` and ``valid_mask`` are bool or float
    (cast internally).

    When ``return_error_map`` is true returns ``(scalar, error)`` where
    ``error`` is the per-pixel post-masking absolute error tensor that
    was actually summed into the scalar — for diagnostic visualizations
    that must mirror the loss pixel-for-pixel.

    This is the shared primitive for the ``absolute_l1`` depth mode used
    by both ``_compute_frame_loss`` (pose refinement) and the Stage-2
    rendering guidance (``core/rendering_guidance.py``).
    """
    gt_mask_f = gt_mask.float()
    gd_masked = gt_depth * gt_mask_f
    error = (pred_depth - gd_masked).abs()
    if valid_mask is not None:
        error = error * valid_mask.float()
    if mask_only:
        error = error * gt_mask_f
        if valid_mask is not None:
            denom = (gt_mask_f * valid_mask.float()).sum().clamp(min=1.0)
        else:
            denom = gt_mask_f.sum().clamp(min=1.0)
        scalar = error.sum() / denom
    else:
        scalar = error.mean()
    if return_error_map:
        return scalar, error
    return scalar


def _match_depth_grid(
    rendered: torch.Tensor,
    gt_hw: Tuple[int, int],
    *,
    K_matrix: Optional[torch.Tensor] = None,
    mask: Optional[torch.Tensor] = None,
):
    """Nearest-downsample a RENDERED depth/normals channel (and its accompanying
    render-space K/mask) down to ``gt_hw``.

    GT depth/normals/valid_mask are at the reconstruction backbone's (H', W'); when
    the render happens at another resolution, the depth and normals loss terms must
    compare at the GT's own grid rather than inventing detail that was never predicted.

    Nearest, NOT area-average: averaging would blend a soft-alpha render's
    silhouette-edge depth discontinuity into its interior before the loss sees it --
    and, on the normals branch, would blend two unit vectors into a non-unit
    "average" direction. Nearest also never spreads a NaN to its neighbours the way
    bilinear/area would, so an invalid depth pixel stays exactly one pixel invalid.

    Identity when ``rendered`` already matches ``gt_hw``, so this is a safe,
    unconditional call at every site. Nearest resizes correctly either way
    (block-replication on upsample, strided sampling on downsample).

    Parameters
    ----------
    rendered : torch.Tensor
        ``(H, W)`` depth or ``(H, W, 3)`` normals, at render resolution. Bool is
        accepted too (some callers reuse this function to resize a mask via the
        ``rendered`` slot, discarding the first return value) -- cast to float for
        the interpolation and back, which is lossless since nearest never blends.
    gt_hw : (int, int)
        Target ``(H, W)`` -- the GT depth/normals grid.
    K_matrix : torch.Tensor, optional
        ``(3, 3)`` at render resolution; rescaled by the pure per-axis ratio (same
        field of view as the render, so no offset correction is needed).
    mask : torch.Tensor, optional
        ``(H, W)`` bool at render resolution; nearest-resized and re-cast to bool.

    Returns
    -------
    (rendered_at_gt_hw, K_at_gt_hw_or_None, mask_at_gt_hw_or_None)
    """
    render_hw = tuple(rendered.shape[:2])
    gt_hw = (int(gt_hw[0]), int(gt_hw[1]))
    if render_hw == gt_hw:
        return rendered, K_matrix, mask

    import torch.nn.functional as F

    was_bool = rendered.dtype == torch.bool
    src = rendered.float() if was_bool else rendered
    if src.dim() == 2:
        rendered_d = F.interpolate(
            src[None, None], size=gt_hw, mode="nearest",
        )[0, 0]
    else:
        rendered_d = F.interpolate(
            src.permute(2, 0, 1)[None], size=gt_hw, mode="nearest",
        )[0].permute(1, 2, 0)
    if was_bool:
        rendered_d = rendered_d.bool()

    K_d = None
    if K_matrix is not None:
        K_d = K_matrix.clone()
        K_d[0, :] = K_d[0, :] * (gt_hw[1] / render_hw[1])
        K_d[1, :] = K_d[1, :] * (gt_hw[0] / render_hw[0])

    mask_d = None
    if mask is not None:
        mask_d = F.interpolate(
            mask.float()[None, None], size=gt_hw, mode="nearest",
        )[0, 0].bool()

    return rendered_d, K_d, mask_d


def _compute_multiscale_rgb_loss(
    rgb: torch.Tensor,
    gt_image: torch.Tensor,
    losses: "LossConfig",
) -> torch.Tensor:
    """
    Compute multi-scale RGB loss for robust pose optimization.

    Multi-scale loss helps escape local minima by computing the loss at
    multiple resolutions. Lower resolutions capture global alignment errors
    while higher resolutions refine fine details.

    Parameters
    ----------
    rgb : torch.Tensor
        Rendered RGB image, shape (H, W, 3), values in [0, 1].
    gt_image : torch.Tensor
        Ground truth RGB image with background masked to black,
        shape (H, W, 3), values in [0, 1].
    losses : LossConfig
        Configuration with rgb_multiscale_scales and rgb_multiscale_weights.

    Returns
    -------
    torch.Tensor
        Scalar RGB loss averaged across all scales.

    Notes
    -----
    L1 per scale.  Downsampling uses area interpolation for anti-aliasing.
    """
    import torch.nn.functional as F

    device = rgb.device
    total_loss = torch.tensor(0.0, device=device)

    # Prepare tensors for interpolation: (H, W, C) -> (1, C, H, W)
    rgb_nchw = rgb.permute(2, 0, 1).unsqueeze(0)
    gt_nchw = gt_image.permute(2, 0, 1).unsqueeze(0)

    for scale, weight in zip(losses.rgb_multiscale_scales, losses.rgb_multiscale_weights):
        if scale == 1.0:
            # Full resolution - no interpolation needed
            rgb_scaled = rgb
            gt_scaled = gt_image
        else:
            # Downsample using area interpolation (anti-aliased)
            rgb_scaled = interpolate(
                rgb_nchw, scale_factor=scale, mode="area"
            ).squeeze(0).permute(1, 2, 0)
            gt_scaled = interpolate(
                gt_nchw, scale_factor=scale, mode="area"
            ).squeeze(0).permute(1, 2, 0)

        diff = (rgb_scaled - gt_scaled).abs()

        # Mean over all pixels (full image comparison)
        n_pixels = rgb_scaled.shape[0] * rgb_scaled.shape[1]
        scale_loss = diff.sum() / (n_pixels * 3)

        total_loss = total_loss + weight * scale_loss

    return total_loss


def depth_to_normals(
    depth: torch.Tensor,
    K_matrix: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Compute surface normals from a z-depth map via finite differences.

    Unprojects z-depth to 3D camera-space points using the pinhole camera
    model, then computes cross product of horizontal and vertical tangent
    vectors from central finite differences.

    Both gsplat rendered depth (``RGB+ED`` mode, which outputs expected
    z-depth ``sum(w_i * z_i) / sum(w_i)``) and MoGe/GT depth maps are
    z-depth (distance along optical axis), so no conversion is needed.

    Parameters
    ----------
    depth : torch.Tensor
        Z-depth map (H, W), distance along optical axis.
    K_matrix : torch.Tensor
        Camera intrinsics (3, 3).
    mask : torch.Tensor, optional
        Boolean mask (H, W). Normals outside mask are zeroed.

    Returns
    -------
    torch.Tensor
        Unit surface normals (H, W, 3). Zero at boundaries and outside mask.
        Normals point toward camera for front-facing surfaces (negative z
        in R3 convention where Z points into the scene).
    """
    H, W = depth.shape
    device = depth.device

    # Pixel grid
    v_coords, u_coords = torch.meshgrid(
        torch.arange(H, device=device, dtype=torch.float32),
        torch.arange(W, device=device, dtype=torch.float32),
        indexing="ij",
    )

    fx, fy = K_matrix[0, 0], K_matrix[1, 1]
    cx, cy = K_matrix[0, 2], K_matrix[1, 2]

    x_norm = (u_coords - cx) / fx
    y_norm = (v_coords - cy) / fy

    # Unproject z-depth to 3D camera-space points via pinhole model
    points = torch.stack([x_norm * depth, y_norm * depth, depth], dim=-1)

    # Central finite differences for tangent vectors
    du = torch.zeros_like(points)
    dv = torch.zeros_like(points)
    du[1:-1, 1:-1] = points[1:-1, 2:] - points[1:-1, :-2]
    dv[1:-1, 1:-1] = points[2:, 1:-1] - points[:-2, 1:-1]

    # Normal = dv x du (normals point toward camera for front-facing surfaces)
    normals = torch.linalg.cross(dv, du)
    normals = normals / normals.norm(dim=-1, keepdim=True).clamp(min=1e-8)

    # Zero out boundary pixels (finite differences invalid there)
    normals[0, :] = 0
    normals[-1, :] = 0
    normals[:, 0] = 0
    normals[:, -1] = 0

    if mask is not None:
        normals = normals * mask.unsqueeze(-1).float()

    return normals


def _erode_bool_mask(mask: torch.Tensor, px: int) -> torch.Tensor:
    """Binary erosion by a ``(2*px+1)`` square.  ``mask`` is ``(H, W)``.

    A pixel survives only if every pixel in its kxk neighborhood is True.
    Implemented as dilation of the inverted mask via max_pool2d so it runs
    on GPU without pulling in scipy / skimage.  ``px <= 0`` is a no-op.
    """
    if px <= 0:
        return mask
    inv = (~mask).float()[None, None]
    k = 2 * px + 1
    dilated_inv = torch.nn.functional.max_pool2d(inv, k, stride=1, padding=px)
    return dilated_inv.squeeze(0).squeeze(0) < 0.5


def _compute_depth_discontinuity_mask(
    depth: torch.Tensor,
    threshold: float,
) -> torch.Tensor:
    """Compute a boolean mask that is False at depth discontinuities.

    A pixel is marked as discontinuous if the maximum absolute difference
    between it and any of its 4-connected neighbours exceeds ``threshold``
    times the local depth value. This relative criterion adapts to both
    near and far regions.

    Parameters
    ----------
    depth : torch.Tensor
        Z-depth map (H, W).
    threshold : float
        Relative depth-gradient threshold. Typical value: 0.02 (2%).

    Returns
    -------
    torch.Tensor
        Boolean mask (H, W), True where depth is smooth (no discontinuity).
    """
    # Pad depth to handle boundaries (replicate edges)
    z_pad = torch.nn.functional.pad(depth.unsqueeze(0).unsqueeze(0), (1, 1, 1, 1), mode="replicate")
    z_pad = z_pad.squeeze(0).squeeze(0)

    # Differences to 4-connected neighbours
    local = z_pad[1:-1, 1:-1]
    local_safe = local.clamp(min=1e-6)
    diff_l = (local - z_pad[1:-1, :-2]).abs() / local_safe
    diff_r = (local - z_pad[1:-1, 2:]).abs() / local_safe
    diff_u = (local - z_pad[:-2, 1:-1]).abs() / local_safe
    diff_d = (local - z_pad[2:, 1:-1]).abs() / local_safe

    max_diff = torch.max(torch.max(diff_l, diff_r), torch.max(diff_u, diff_d))
    return max_diff <= threshold
