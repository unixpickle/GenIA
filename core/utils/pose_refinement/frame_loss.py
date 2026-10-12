# Copyright (c) Meta Platforms, Inc. and affiliates.

"""Assemble reconstruction losses and prepare observations for one frame."""

from __future__ import annotations

from typing import Any, Dict, Optional

import torch
from genia.core.utils.config import LossConfig
from genia.core.utils.pose_refinement.loss_terms import (
    _compute_center_of_mass_loss,
    _compute_depth_discontinuity_mask,
    _compute_multiscale_rgb_loss,
    _compute_sdt_loss,
    _compute_signed_distance_transform,
    _compute_soft_iou_loss,
    _compute_ssim_loss,
    _depth_l1_term,
    _l1_loss,
    _match_depth_grid,
    depth_to_normals,
)


def _compute_frame_loss(
    rgb: torch.Tensor,
    alpha: torch.Tensor,
    gt_image: torch.Tensor,
    mask: torch.Tensor,
    losses: LossConfig,
    sdt: Optional[torch.Tensor] = None,
    rendered_depth: Optional[torch.Tensor] = None,
    gt_depth: Optional[torch.Tensor] = None,
    valid_mask: Optional[torch.Tensor] = None,
    bg_color: Optional[torch.Tensor] = None,
    perceptual_scale: float = 1.0,
    K_matrix: Optional[torch.Tensor] = None,
    return_pixelwise: bool = False,
    has_background: bool = False,
    depth_mask_only: Optional[bool] = None,
) -> Dict[str, torch.Tensor]:
    """
    Compute loss for a single frame.

    The loss consists of:
    - RGB loss: L1 between rendered and GT image, optionally multi-scale
    - SSIM loss: Structural similarity loss (optional, weight > 0 to enable)
    - Perceptual loss: LPIPS (VGG) between rendered and masked GT (optional)
    - Silhouette loss: Combined loss for pose-aware silhouette alignment
      - Center-of-mass: Aligns centroids (guides translation)
      - SDT: Signed distance transform (guides all params, escapes local minima)
      - IoU: Soft intersection-over-union (fine shape matching)
    - Depth loss: L1 or scale-shift invariant between rendered and GT/estimated depth (optional)

    GT background is masked to match the rendering background color for all
    appearance losses (RGB, SSIM, perceptual).

    Parameters
    ----------
    rgb : torch.Tensor
        Rendered RGB image (H, W, 3).
    alpha : torch.Tensor
        Rendered alpha/opacity (H, W).
    gt_image : torch.Tensor
        Ground truth image (H, W, 3).
    mask : torch.Tensor
        Object mask (H, W), boolean.
    losses : LossConfig
        Per-phase loss weights and RGB loss settings.
    sdt : torch.Tensor, optional
        Precomputed signed distance transform of the mask, shape (H, W).
        If None and silhouette_sdt_weight > 0, it will be computed here.
    rendered_depth : torch.Tensor, optional
        Rendered depth map (H, W). Required when depth_weight > 0.
    gt_depth : torch.Tensor, optional
        Ground truth depth map (H, W). Required when depth_weight > 0.
    K_matrix : torch.Tensor, optional
        Camera intrinsics (3, 3). Required when normals_weight > 0
        (used to unproject rendered depth to 3D for normal computation).
    valid_mask : torch.Tensor, optional
        MoGe/sensor depth validity mask (H, W), boolean.
    bg_color : torch.Tensor, optional
        Background color (3,). Used to mask GT image background.
    perceptual_scale : float
        Scale factor for perceptual loss input resolution.

    Returns
    -------
    dict
        Dictionary with loss tensors:
        - 'rgb_loss': Scalar RGB loss (multi-scale if enabled)
        - 'ssim_loss': Scalar SSIM loss
        - 'perceptual_loss': LPIPS perceptual loss (0 if disabled)
        - 'silhouette_com': Center-of-mass loss component
        - 'silhouette_sdt': SDT loss component
        - 'silhouette_iou': IoU loss component
        - 'depth_loss': Scalar depth loss (0 if disabled)
        - 'normals_loss': Scalar normals loss (0 if disabled)
    """
    device = rgb.device
    H, W = rgb.shape[:2]

    # Mask GT background to match rendering background color
    mask_f = mask.float().unsqueeze(-1)
    if bg_color is not None:
        gt_masked = gt_image * mask_f + bg_color.view(1, 1, 3) * (1.0 - mask_f)
    else:
        gt_masked = gt_image * mask_f

    # Per-pixel debug maps (populated only when return_pixelwise=True)
    px = {}
    if return_pixelwise:
        px["px_rgb_error"] = (rgb - gt_masked).abs().mean(dim=-1).detach().cpu().numpy()
        px["px_alpha"] = (alpha.squeeze(0) if alpha.dim() == 3 else alpha).detach().cpu().numpy()

    # RGB loss: L1, optionally multi-scale
    if losses.rgb_multiscale:
        # Multi-scale loss for robust optimization
        rgb_loss_value = _compute_multiscale_rgb_loss(rgb, gt_masked, losses)
    else:
        # Standard single-scale L1 loss
        rgb_loss_value = _l1_loss(rgb, gt_masked)

    # SSIM loss (optional, enabled when weight > 0)
    ssim_loss_value = torch.tensor(0.0, device=device)
    if losses.rgb_ssim_weight > 0:
        ssim_loss_value = _compute_ssim_loss(rgb, gt_masked)

    # Perceptual loss (LPIPS, optional)
    perceptual_loss_value = torch.tensor(0.0, device=device)
    if losses.perceptual_weight > 0:
        from genia.core.utils.model_cache import ModelCache
        rgb_bchw = rgb.permute(2, 0, 1).unsqueeze(0)
        gt_bchw = gt_masked.permute(2, 0, 1).unsqueeze(0)
        # Crop to the object before any downscale: VGG then sees only the
        # object, not the (white) background.  `mask` matches the un-scaled
        # (H, W) here, so crop-then-scale keeps mask/tensor resolution aligned.
        if losses.perceptual_crop_to_mask:
            from genia.core.utils.perceptual_loss import crop_to_mask_bbox
            rgb_bchw, gt_bchw = crop_to_mask_bbox(
                rgb_bchw, gt_bchw, mask, losses.perceptual_crop_margin)
        if perceptual_scale < 1.0:
            rgb_bchw = torch.nn.functional.interpolate(
                rgb_bchw, scale_factor=perceptual_scale,
                mode="bilinear", align_corners=False)
            gt_bchw = torch.nn.functional.interpolate(
                gt_bchw, scale_factor=perceptual_scale,
                mode="bilinear", align_corners=False)
        # Optional bf16 autocast on the LPIPS/VGG forward to halve its retained
        # activation memory; `.float()` restores fp32 for the downstream
        # weighting/sum. enabled=False is a no-op (and stays a no-op under the
        # rendering-guidance autocast(enabled=False) caller).
        with torch.autocast(torch.device(device).type, dtype=torch.bfloat16,
                            enabled=losses.perceptual_autocast_bf16):
            perceptual_loss_value = ModelCache.get().perceptual_model(
                rgb_bchw, gt_bchw).float()

    # Silhouette loss components
    alpha_squeezed = alpha.squeeze(0) if alpha.dim() == 3 else alpha

    # Initialize loss components
    com_loss_value = torch.tensor(0.0, device=device)
    sdt_loss_value = torch.tensor(0.0, device=device)
    iou_loss_value = torch.tensor(0.0, device=device)

    if losses.silhouette_weight > 0:

        # Center-of-mass loss: directly guides translation
        if losses.silhouette_com_weight > 0:
            com_loss_value = _compute_center_of_mass_loss(alpha_squeezed, mask, occlusion_robust=losses.occlusion_robust_silhouette)

        # Signed distance transform loss: guides all params, escapes local minima
        if losses.silhouette_sdt_weight > 0:
            if sdt is None:
                sdt = _compute_signed_distance_transform(mask)
            sdt_loss_value = _compute_sdt_loss(alpha_squeezed, sdt, occlusion_robust=losses.occlusion_robust_silhouette)

        # Soft IoU loss: fine-grained shape matching
        if losses.silhouette_iou_weight > 0:
            iou_loss_value = _compute_soft_iou_loss(alpha_squeezed, mask, occlusion_robust=losses.occlusion_robust_silhouette)

    # Everything below this point is depth/normals: GT depth/normals are at the
    # reconstruction backbone's resolution, so when the render is at a different
    # resolution, shadow rendered_depth/K_matrix/mask with their GT-grid counterparts
    # for the rest of this function.  A no-op whenever the shapes already match.  This
    # also keeps `rendered_normals` (below) at the SAME grid as `target_normals`, so
    # `rendered_normals[normals_valid]` indexes a tensor of the mask's shape.
    if gt_depth is not None and rendered_depth is not None:
        rendered_depth, K_matrix, mask = _match_depth_grid(
            rendered_depth, gt_depth.shape[-2:], K_matrix=K_matrix, mask=mask)

    # Depth loss (optional, enabled when weight > 0)
    # Only GT depth is masked with the object mask (background → 0).
    # valid_mask (MoGe reliability) is used to mask the error, not the inputs.
    depth_loss_value = torch.tensor(0.0, device=device)
    if losses.depth_weight > 0 and rendered_depth is not None and gt_depth is not None:
        # Partial / sparse GT depth (e.g. a mesh-rasterized depth that is NaN
        # outside the silhouette) would poison the loss: masking can't remove NaNs
        # because NaN * 0 = NaN.  Fold finiteness into valid_mask and zero the NaNs
        # so masked-out pixels contribute 0.  No-op when GT depth is already dense
        # (GSO / MoGe / map-anything cover every pixel).
        _finite = torch.isfinite(gt_depth)
        if not bool(_finite.all()):
            valid_mask = _finite if valid_mask is None else (valid_mask & _finite)
            gt_depth = torch.nan_to_num(gt_depth, nan=0.0)
        # Mask GT depth with object mask (same convention as RGB masked to black)
        gt_depth_masked = gt_depth * mask.float()
        # Per-call override wins over the config default: the shared-world MV path
        # forces masked depth on frames whose fused background came out empty (no
        # valid depth outside the object → full-image supervision would be noise).
        _depth_mask_only = (
            losses.depth_mask_only if depth_mask_only is None else depth_mask_only
        )
        # depth_mask_only=False supervises depth over the full image; that is only
        # well-defined when the render covers every GT depth pixel — i.e. a
        # background model (pointmap Gaussians or equivalent) is rendered alongside
        # the foreground. Without it, outside-mask pixels have no valid rendered
        # depth and the loss becomes noise.
        assert _depth_mask_only or has_background, (
            "losses.depth_mask_only=False requires has_background=True: "
            "full-image depth supervision needs a background model to produce "
            "valid depth outside the GT mask. Either set depth_mask_only=true, "
            "or render a background and pass has_background=True."
        )
        if losses.depth_loss_type == "scale_shift_invariant":
            # Scale-and-shift invariant depth loss for monocular depth (e.g. MoGe).
            # Fit (s, t) on reliable pixels (mask & valid_mask), then compute
            # error on mask-only or full image depending on depth_mask_only.
            fit_mask = mask & valid_mask if valid_mask is not None else mask
            rd_fit = rendered_depth[fit_mask]
            gd_fit = gt_depth[fit_mask]
            if gd_fit.numel() > 1:
                with torch.no_grad():
                    g_mean = gd_fit.mean()
                    d_mean = rd_fit.mean()
                    g_var = ((gd_fit - g_mean) ** 2).mean()
                    if g_var > 1e-8:
                        g_cov = ((gd_fit - g_mean) * (rd_fit - d_mean)).mean()
                        s = g_cov / g_var
                        t = d_mean - s * g_mean
                    else:
                        s = torch.ones(1, device=device)
                        t = d_mean - g_mean
                mask_f = mask.float()
                aligned_gt = (s * gt_depth + t) * mask_f
                error = (rendered_depth - aligned_gt).abs()
                if valid_mask is not None:
                    error = error * valid_mask.float()
                if _depth_mask_only:
                    error = error * mask_f
                    # Proper masked mean — denominator counts contributing
                    # pixels (in-mask AND in-valid), not H·W.
                    if valid_mask is not None:
                        _denom = (mask_f * valid_mask.float()).sum().clamp(min=1.0)
                    else:
                        _denom = mask_f.sum().clamp(min=1.0)
                    depth_loss_value = error.sum() / _denom
                else:
                    depth_loss_value = error.mean()
                if return_pixelwise:
                    px["px_depth_error"] = error.detach().cpu().numpy()
        else:
            # Absolute L1 depth loss (for GT depth with known scale).
            # Delegates to ``_depth_l1_term`` — the shared primitive also
            # used by the Stage-2 rendering guidance.  When
            # depth_mask_only=False, rendered depth outside
            # the mask is penalized against 0 (silhouette-like leakage
            # penalty); when True, only mask pixels contribute.
            depth_loss_value = _depth_l1_term(
                rendered_depth, gt_depth, mask,
                mask_only=_depth_mask_only,
                valid_mask=valid_mask,
            )
            if return_pixelwise:
                # Re-compute the unreduced error map for the pixelwise
                # debug plot (``_depth_l1_term`` returns the scalar mean).
                err = (rendered_depth - gt_depth_masked).abs()
                if valid_mask is not None:
                    err = err * valid_mask.float()
                if _depth_mask_only:
                    err = err * mask.float()
                px["px_depth_error"] = err.detach().cpu().numpy()

    # Surface normals loss (optional, enabled when normals_weight > 0).
    # Pseudo-GT normals are derived from gt_depth via finite differences (always
    # consistent with the pointmap).  Discontinuity masking: exclude pixels near large depth gradients where
    # finite-difference normals are unreliable.
    normals_loss_value = torch.tensor(0.0, device=device)
    if losses.normals_weight > 0 and rendered_depth is not None and K_matrix is not None:
        # Rendered normals (from gsplat z-depth, differentiable)
        rendered_normals = depth_to_normals(rendered_depth, K_matrix, mask=mask)

        # Pseudo-GT normals from gt_depth (z-depth)
        target_normals = None
        if gt_depth is not None:
            with torch.no_grad():
                target_normals = depth_to_normals(gt_depth, K_matrix, mask=mask)

        if target_normals is not None:
            # Valid region: object mask, non-zero target normals, interior pixels
            target_norm = target_normals.norm(dim=-1)
            normals_valid = mask & (target_norm > 0.5)
            normals_valid[0, :] = False
            normals_valid[-1, :] = False
            normals_valid[:, 0] = False
            normals_valid[:, -1] = False
            if valid_mask is not None:
                normals_valid = normals_valid & valid_mask

            # Discontinuity masking: exclude pixels near large depth gradients
            if losses.normals_discontinuity_threshold > 0 and gt_depth is not None:
                disc_mask = _compute_depth_discontinuity_mask(
                    gt_depth, losses.normals_discontinuity_threshold,
                )
                normals_valid = normals_valid & disc_mask

            if normals_valid.sum() > 0:
                rn = rendered_normals[normals_valid]  # (M, 3)
                gn = target_normals[normals_valid]    # (M, 3)
                # Sign-invariant cosine: 1 - |cos(angle)|
                cos_sim = (rn * gn).sum(dim=-1)
                normals_loss_value = (1.0 - cos_sim.abs()).mean()
                if return_pixelwise:
                    # normals_valid/mask may be at the GT-depth grid (render
                    # resampled by _match_depth_grid above), not (H, W).
                    px_normals = torch.zeros(mask.shape, device=device)
                    px_normals[normals_valid] = (1.0 - cos_sim.abs()).detach()
                    px["px_normals_error"] = px_normals.cpu().numpy()

    result = {
        "rgb_loss": rgb_loss_value,  # Scalar RGB loss
        "ssim_loss": ssim_loss_value,  # Scalar SSIM loss (0 if disabled)
        "perceptual_loss": perceptual_loss_value,  # LPIPS perceptual loss (0 if disabled)
        "silhouette_com": com_loss_value,  # Center-of-mass loss component (0 if disabled)
        "silhouette_sdt": sdt_loss_value,  # SDT loss component (0 if disabled)
        "silhouette_iou": iou_loss_value,  # IoU loss component (0 if disabled)
        "depth_loss": depth_loss_value,   # Scalar depth loss (0 if disabled)
        "normals_loss": normals_loss_value,  # Surface normals loss (0 if disabled)
    }
    if return_pixelwise:
        result.update(px)
    return result


def _prepare_frame_data_for_refinement(
    sequence: Any,
    frame_idx: int,
    obj_idx: int,
) -> Optional[Dict[str, Any]]:
    """
    Prepare frame data for refinement from a cached Sequence.

    Parameters
    ----------
    sequence : Sequence
        Cached scene data.
    frame_idx : int
        Frame index.
    obj_idx : int
        Object index.

    Returns
    -------
    dict or None
        Dictionary with 'image', 'mask', 'K_matrix', 'gt_depth',
        'valid_mask', 'H', 'W', or None if the object is not present
        in this frame.
    """
    frame = sequence[frame_idx]

    image, render_masks, K_matrix = frame.image, frame.masks, frame.K_matrix
    mask = render_masks[obj_idx]
    if not mask.any():
        return None  # Object not present in this frame

    return {
        "image": image,
        "mask": mask,
        "K_matrix": K_matrix,
        "gt_depth": frame.depth_map_z,
        "valid_mask": frame.valid_mask,
        "H": image.shape[0],
        "W": image.shape[1],
    }
