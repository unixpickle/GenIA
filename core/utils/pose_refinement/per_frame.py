# Copyright (c) Meta Platforms, Inc. and affiliates.

"""Optimize a single frame’s object pose."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict, Optional

import numpy as np
import torch
from genia.core.utils.config import LossConfig, PipelineConfig, resolve_chamfer_gt_trim
from genia.core.utils.pose_refinement.chamfer import (
    _chamfer_gt_points_from_depth,
    _chamfer_gt_to_model,
    _chamfer_model_subsample_idx,
)
from genia.core.utils.pose_refinement.decoding import differentiable_pose_decode
from genia.core.utils.pose_refinement.frame_loss import _compute_frame_loss
from genia.core.utils.pose_refinement.loss_terms import (
    _compute_signed_distance_transform,
    _match_depth_grid,
    depth_to_normals,
)
from genia.core.utils.pose_refinement.rendering import _get_bg_color, make_posed_object_renderer
from genia.core.utils.timing import get_timer

if TYPE_CHECKING:
    from genia.core.utils.pose_refinement.rendering import PosedObjectRenderer



if TYPE_CHECKING:
    from sam3d_objects.model.backbone.tdfy_dit.representations.gaussian.gaussian_model import (
        Gaussian,
    )


def refine_pose_for_frame(
    canonical_gs: "Gaussian | Dict[str, Any]",
    initial_rotation: torch.Tensor,
    initial_translation: torch.Tensor,
    initial_scale: torch.Tensor,
    gt_image: torch.Tensor,
    mask: torch.Tensor | np.ndarray,
    K_matrix: np.ndarray,
    losses: LossConfig,
    pipeline: PipelineConfig,
    refine_scale: str = "perframe",
    gt_depth: Optional[np.ndarray] = None,
    valid_mask: Optional[np.ndarray] = None,
    raw_modalities: Optional[Dict[str, torch.Tensor]] = None,
    scene_scale: Optional[torch.Tensor] = None,
    scene_shift: Optional[torch.Tensor] = None,
    downsample_factor: float = 1.0,
    optimize_pose_tokens: bool = False,
    output_dir: Optional[str] = None,
    save_renders: bool = True,
    obj_idx: int = 0,
    frame_idx: int = 0,
    *,
    renderer: Optional[PosedObjectRenderer] = None,
) -> Dict[str, Any]:
    """
    Refine pose parameters using differentiable rendering.

    Parameters
    ----------
    canonical_gs : Gaussian
        The canonical object (frozen), rendered through
        :func:`make_posed_object_renderer`.
    initial_rotation : torch.Tensor
        Initial quaternion rotation (1, 4).
    initial_translation : torch.Tensor
        Initial translation (1, 3).
    initial_scale : torch.Tensor
        Initial scale (1, 3) or (1,).
    gt_image : torch.Tensor
        Ground truth image (H, W, 3) in [0, 1].
    mask : np.ndarray or torch.Tensor
        Object mask (H, W), boolean.
    K_matrix : np.ndarray
        Camera intrinsics (3, 3).
    losses : LossConfig
        Per-phase loss weights and iteration count.
    pipeline : PipelineConfig
        Cross-phase pipeline control flags.
    refine_scale : str
        Scale refinement mode: "perframe" (optimize per-frame) or "none" (freeze).
    gt_depth : np.ndarray, optional
        Ground truth z-depth map (H, W). When provided with depth_weight > 0,
        depth loss is computed in the masked region.
    valid_mask : np.ndarray, optional
        Boolean mask of valid depth pixels (H, W). When provided, intersected
        with the object mask for depth loss (excludes unreliable MoGe pixels).
    raw_modalities : dict, optional
        Raw Stage 1 pose tokens: ``6drotation_normalized``, ``translation``,
        ``scale``. When provided with ``optimize_pose_tokens=True``, these are
        optimized instead of the final pose, with gradients flowing through
        the differentiable pose decoder.
    scene_scale : torch.Tensor, optional
        Pointmap scale for PoseTargetConverter, shape ``(3,)``.
    scene_shift : torch.Tensor, optional
        Pointmap shift for PoseTargetConverter, shape ``(3,)``.
    downsample_factor : float
        Scale post-multiplier from sparse-structure downsampling.
    optimize_pose_tokens : bool
        When True and raw_modalities are available, optimize raw Stage 1 pose
        tokens instead of final pose parameters.
    renderer : PosedObjectRenderer or None, keyword-only
        Prebuilt adapter for ``canonical_gs``; built here when omitted. Callers
        looping over frames should build it ONCE per object and pass it.

    Returns
    -------
    dict
        Refined pose parameters with keys:
        - rotation: refined quaternion (1, 4)
        - translation: refined translation (1, 3)
        - scale: refined uniform scale (1, 3)
        - loss_history: list of loss values at each iteration
        - best_iteration: iteration index with lowest loss
    """
    # Handle num_iterations == 0 (pose init search only, no optimization)
    if losses.num_iterations == 0:
        initial_scale_flat = initial_scale.view(-1)
        # Collapse to uniform scale
        if initial_scale_flat.shape[0] == 3:
            s = initial_scale_flat.mean().item()
        else:
            s = initial_scale_flat[0].item()
        scale_out = torch.tensor([[s, s, s]],
                                  device=initial_rotation.device, dtype=initial_rotation.dtype)
        return {
            "rotation": initial_rotation,
            "translation": initial_translation,
            "scale": scale_out,
            "loss_history": [],
            "best_iteration": 0,
        }

    device = initial_rotation.device
    H, W = gt_image.shape[:2]
    if renderer is None:
        renderer = make_posed_object_renderer(canonical_gs, device)

    # Prepare ground truth
    if isinstance(gt_image, np.ndarray):
        gt_image = torch.from_numpy(gt_image).float().to(device)
    else:
        gt_image = gt_image.float().to(device)

    # Prepare mask
    if isinstance(mask, np.ndarray):
        mask = torch.from_numpy(mask).bool().to(device)
    else:
        mask = mask.bool().to(device)

    # Prepare GT depth tensor and valid mask (for depth loss or depth-derived normals)
    gt_depth_tensor = None
    valid_mask_tensor = None
    need_depth = losses.depth_weight > 0 or losses.normals_weight > 0
    if gt_depth is not None and need_depth:
        gt_depth_tensor = torch.from_numpy(gt_depth).float().to(device)
        if valid_mask is not None:
            valid_mask_tensor = torch.from_numpy(valid_mask).bool().to(device)

    # K_matrix tensor (for normals loss)
    K_tensor = None
    if losses.normals_weight > 0:
        K_tensor = torch.from_numpy(K_matrix).float().to(device)
        if valid_mask is not None and valid_mask_tensor is None:
            valid_mask_tensor = torch.from_numpy(valid_mask).bool().to(device)

    # Precompute signed distance transform for silhouette loss (expensive, do once)
    sdt = None
    if losses.silhouette_weight > 0 and losses.silhouette_sdt_weight > 0:
        sdt = _compute_signed_distance_transform(mask)

    # Determine whether to use pose token optimization
    # Only enabled for per-frame refinement (raw modalities may be stale otherwise)
    use_pose_tokens = (
        optimize_pose_tokens
        and refine_scale == "perframe"
        and raw_modalities is not None
        and "6drotation_normalized" in raw_modalities
        and scene_scale is not None
        and scene_shift is not None
    )
    if optimize_pose_tokens and not use_pose_tokens:
        if pipeline.verbose:
            print("      [Pose tokens] Raw modalities or scene context unavailable (or non-perframe mode), falling back to direct pose optimization")
    # Also check that raw_modalities has the expected "translation" key (not confuse
    # with the final translation from the token dict which has the same name)
    if use_pose_tokens and "translation" not in raw_modalities:
        use_pose_tokens = False
        if pipeline.verbose:
            print("      [Pose tokens] Missing raw translation modality, falling back to direct pose optimization")

    if use_pose_tokens:
        # --- Pose token optimization path ---
        # Optimize raw Stage 1 outputs and forward through differentiable pose decoder
        _rot_lr = float(losses.lr_rotation)
        _trans_lr = float(losses.lr_translation)
        _scale_lr = float(losses.lr_scale)
        opt_6drot = raw_modalities["6drotation_normalized"].clone().detach().float().requires_grad_(_rot_lr > 0)
        opt_raw_trans = raw_modalities["translation"].clone().detach().float().requires_grad_(_trans_lr > 0)
        opt_raw_scale = raw_modalities["scale"].clone().detach().float().requires_grad_(
            refine_scale == "perframe" and _scale_lr > 0)

        # Keep initial values for regularization
        initial_6drot = opt_6drot.clone().detach()
        initial_raw_trans = opt_raw_trans.clone().detach()
        initial_raw_scale = opt_raw_scale.clone().detach()

        param_groups = []
        if _rot_lr > 0:
            param_groups.append({"params": [opt_6drot], "lr": _rot_lr})
        if _trans_lr > 0:
            param_groups.append({"params": [opt_raw_trans], "lr": _trans_lr})
        if refine_scale == "perframe" and _scale_lr > 0:
            param_groups.append({"params": [opt_raw_scale], "lr": _scale_lr})

        # Dummy direct-opt variables (not used but referenced in shared code paths)
        opt_rotation = None
        opt_translation = None
        opt_scale = None
        initial_scale_param = None

        if pipeline.verbose:
            print("      [Pose tokens] Optimizing raw Stage 1 modalities (6D rotation, translation, log-scale)")
    else:
        # --- Direct pose optimization path ---
        opt_6drot = None
        opt_raw_trans = None
        opt_raw_scale = None
        initial_6drot = None
        initial_raw_trans = None
        initial_raw_scale = None

        _rot_lr = float(losses.lr_rotation)
        _trans_lr = float(losses.lr_translation)
        _scale_lr = float(losses.lr_scale)
        opt_rotation = initial_rotation.clone().detach().requires_grad_(_rot_lr > 0)
        opt_translation = initial_translation.clone().detach().requires_grad_(_trans_lr > 0)

        # Isotropic scale parameter (scalar)
        initial_scale_flat = initial_scale.view(-1)
        if initial_scale_flat.shape[0] == 3:
            initial_scale_param = initial_scale_flat.mean().view(1)
        else:
            initial_scale_param = initial_scale_flat[:1]

        _scale_grad = refine_scale == "perframe" and _scale_lr > 0
        opt_scale = initial_scale_param.clone().detach().requires_grad_(_scale_grad)

        param_groups = []
        if _rot_lr > 0:
            param_groups.append({"params": [opt_rotation], "lr": _rot_lr})
        if _trans_lr > 0:
            param_groups.append({"params": [opt_translation], "lr": _trans_lr})
        if refine_scale == "perframe" and _scale_lr > 0:
            param_groups.append({"params": [opt_scale], "lr": _scale_lr})

    optimizer = torch.optim.Adam(param_groups)

    bg_color = _get_bg_color(pipeline, device)

    best_loss = float("inf")
    best_params = {}
    best_iteration = 0
    loss_history = []

    # Optional Chamfer (GT-depth → posed-model) — precompute the object's
    # visible GT surface points (unproject GT depth to a camera-space pointmap,
    # keep mask ∩ valid ∩ finite) + a fixed model-point subsample, once.
    gt_chamfer_pts = None
    chamfer_model_idx = None
    # Resolved ONCE, outside the loop: a property of the config, not of the frame.
    _chamfer_trim = resolve_chamfer_gt_trim(losses, "global_pose_refine")
    if getattr(losses, "chamfer_weight", 0.0) > 0 and gt_depth is not None:
        # mask (already a torch tensor by this point) and K_matrix (still the
        # function's numpy param) may be render-resolution; gt_depth never
        # exceeds the backbone's resolution -- downsample to match before use.
        _, _K_chamfer, _mask_chamfer = _match_depth_grid(
            mask, gt_depth.shape,
            K_matrix=torch.as_tensor(K_matrix, dtype=torch.float32, device=device),
            mask=mask)
        gt_chamfer_pts = _chamfer_gt_points_from_depth(
            gt_depth, _K_chamfer.cpu().numpy(), _mask_chamfer, valid_mask,
            int(getattr(losses, "chamfer_max_gt_points", 2048)), device,
            trim_factor=_chamfer_trim)
        chamfer_model_idx = _chamfer_model_subsample_idx(
            renderer.num_model_points(),
            int(getattr(losses, "chamfer_max_model_points", 8192)), device)

    for iteration in range(losses.num_iterations):
        optimizer.zero_grad()

        # Derive current pose (either from raw tokens or direct params)
        if use_pose_tokens:
            _, cur_rotation, cur_translation, cur_scale = differentiable_pose_decode(
                opt_6drot, opt_raw_trans, opt_raw_scale,
                scene_scale, scene_shift, downsample_factor,
            )
        else:
            cur_rotation = opt_rotation
            cur_translation = opt_translation
            cur_scale = opt_scale

        # Render frame with current pose
        rgb, alpha, depth = renderer.render(
            cur_rotation, cur_translation, cur_scale, K_matrix, W, H, device,
            bg_color=bg_color,
        )

        # One-directional Chamfer (GT-depth → posed model means), optional.
        chamfer_loss_value = torch.tensor(0.0, device=device)
        if gt_chamfer_pts is not None:
            _model_xyz = renderer.model_points(
                cur_rotation, cur_translation, cur_scale, device)
            chamfer_loss_value = _chamfer_gt_to_model(
                gt_chamfer_pts, _model_xyz, model_sub_idx=chamfer_model_idx)

        # Compute RGB, SSIM, silhouette, depth, and normals loss
        want_pixelwise = (
            save_renders
            and output_dir is not None
            and iteration % 50 == 0
        )
        losses_dict = _compute_frame_loss(
            rgb, alpha, gt_image, mask, losses, sdt,
            rendered_depth=depth, gt_depth=gt_depth_tensor,
            valid_mask=valid_mask_tensor,
            bg_color=bg_color,
            K_matrix=K_tensor,
            return_pixelwise=want_pixelwise,
            # This path renders the object alone (no background model), so depth
            # outside the GT mask has no valid rendered counterpart -- force masked
            # depth regardless of the config default.  Same auto-fall-back the
            # shared-world composite path applies on background-less frames; lets a
            # global-refine LossConfig (depth_mask_only=false + render_with_background)
            # drive per-frame refinement.
            depth_mask_only=True,
        )

        # Save per-pixel debug plot every 50 iterations
        if want_pixelwise:
            _K = K_tensor
            if _K is None:
                _K = torch.from_numpy(K_matrix).float().to(device)
            mask_f = mask.float().unsqueeze(-1)
            gt_masked = gt_image * mask_f + bg_color.view(1, 1, 3) * (1.0 - mask_f)
            frame_px: Dict[str, Any] = {
                "rendered_rgb": rgb.detach().cpu().numpy(),
                "gt_rgb": gt_masked.detach().cpu().numpy(),
                "mask": mask.cpu().numpy(),
            }
            if depth is not None:
                depth_sq = depth.squeeze(0) if depth.dim() == 3 else depth
                frame_px["rendered_depth"] = depth_sq.detach().cpu().numpy()
                frame_px["rendered_normals"] = depth_to_normals(depth_sq, _K).detach().cpu().numpy()
            if gt_depth_tensor is not None:
                # mask/_K are render-resolution; gt_depth_tensor never exceeds the
                # backbone's resolution -- downsample to match before use.
                _, _K_gt, _mask_gt = _match_depth_grid(
                    mask, gt_depth_tensor.shape[-2:], K_matrix=_K, mask=mask)
                gt_depth_masked = gt_depth_tensor * _mask_gt.float()
                frame_px["gt_depth"] = gt_depth_masked.cpu().numpy()
                frame_px["gt_normals"] = depth_to_normals(gt_depth_tensor, _K_gt, mask=_mask_gt).detach().cpu().numpy()
            for k, v in losses_dict.items():
                if k.startswith("px_"):
                    frame_px[k] = v
            from genia.core.utils.visualization import save_pixelwise_loss_plot
            with get_timer().exclude():
                save_pixelwise_loss_plot({frame_idx: frame_px}, iteration, obj_idx, output_dir)

        # RGB and SSIM losses are separate scalars for logging
        rgb_loss_value = losses_dict["rgb_loss"]  # Excludes SSIM
        ssim_loss_value = losses_dict["ssim_loss"]  #  (0 if disabled)
        depth_loss_value = losses_dict["depth_loss"]  #  (0 if disabled)
        normals_loss_value = losses_dict["normals_loss"]  #  (0 if disabled)

        # Extract individual silhouette components for logging
        silh_com = losses_dict["silhouette_com"]
        silh_sdt = losses_dict["silhouette_sdt"]
        silh_iou = losses_dict["silhouette_iou"]

        # Combined silhouette loss
        silhouette_loss_value = (
            losses.silhouette_com_weight * silh_com
            + losses.silhouette_sdt_weight * silh_sdt
            + losses.silhouette_iou_weight * silh_iou
        )

        # Combine weighted losses
        weighted_rgb_loss = losses.rgb_weight * rgb_loss_value
        weighted_ssim_loss_value = losses.rgb_ssim_weight * ssim_loss_value
        weighted_silhouette_loss_value = losses.silhouette_weight * silhouette_loss_value
        weighted_depth_loss = losses.depth_weight * depth_loss_value
        weighted_normals_loss = losses.normals_weight * normals_loss_value
        weighted_perceptual_loss = losses.perceptual_weight * losses_dict["perceptual_loss"]

        base_loss = weighted_rgb_loss + weighted_ssim_loss_value + weighted_silhouette_loss_value + weighted_perceptual_loss + weighted_depth_loss + weighted_normals_loss + getattr(losses, "chamfer_weight", 0.0) * chamfer_loss_value

        # Debug: Check if rendered object overlaps with mask
        if iteration == 0 and pipeline.verbose:
            with torch.no_grad():
                rendered_in_mask = rgb[mask]
                bg_in_mask = bg_color.expand(rendered_in_mask.shape[0], -1)
                non_bg_mask = (rendered_in_mask - bg_in_mask).abs().sum(dim=-1) > 0.01
                print(
                    f"      [Overlap check] Mask pixels: {mask.sum().item()}, "
                    f"Non-background in mask: {non_bg_mask.sum().item()} "
                    f"({100*non_bg_mask.float().mean().item():.1f}%)"
                )

        # Optional: add regularization to prevent large deviations from initial pose
        if losses.regularization_weight > 0:
            rot_w = losses.regularization_rotation_weight
            trans_w = losses.regularization_translation_weight
            scale_w = losses.regularization_scale_weight
            if use_pose_tokens:
                # Regularize in raw Stage 1 space
                reg_rot = ((opt_6drot - initial_6drot) ** 2).sum()
                reg_trans = ((opt_raw_trans - initial_raw_trans) ** 2).sum()
                if refine_scale == "perframe":
                    reg_scale = ((opt_raw_scale - initial_raw_scale) ** 2).sum()
                    weighted_reg_loss = losses.regularization_weight * (rot_w * reg_rot + trans_w * reg_trans + scale_w * reg_scale)
                else:
                    weighted_reg_loss = losses.regularization_weight * (rot_w * reg_rot + trans_w * reg_trans)
            else:
                opt_rotation_normalized = opt_rotation / opt_rotation.norm()
                reg_rot = (
                    (opt_rotation_normalized - initial_rotation / initial_rotation.norm()) ** 2
                ).sum()
                reg_trans = ((opt_translation - initial_translation) ** 2).sum()
                if refine_scale == "perframe":
                    reg_scale = ((opt_scale - initial_scale_param) ** 2).sum()
                    weighted_reg_loss = losses.regularization_weight * (rot_w * reg_rot + trans_w * reg_trans + scale_w * reg_scale)
                else:
                    weighted_reg_loss = losses.regularization_weight * (rot_w * reg_rot + trans_w * reg_trans)
        else:
            weighted_reg_loss = torch.tensor(0.0, device=device)

        # Total loss
        loss = base_loss + weighted_reg_loss

        # Record loss for this iteration (all values are weighted)
        loss_history.append(
            {
                "total": loss.item(),
                "rgb": weighted_rgb_loss.item(),
                "ssim": weighted_ssim_loss_value.item(),
                "silhouette": weighted_silhouette_loss_value.item(),
                "silhouette_com": silh_com.item(),
                "silhouette_sdt": silh_sdt.item(),
                "silhouette_iou": silh_iou.item(),
                "depth": weighted_depth_loss.item(),
                "normals": weighted_normals_loss.item(),
                "perceptual": weighted_perceptual_loss.item() if isinstance(weighted_perceptual_loss, torch.Tensor) else weighted_perceptual_loss,
                "chamfer": (getattr(losses, "chamfer_weight", 0.0) * chamfer_loss_value).item(),
                "regularization": weighted_reg_loss.item(),
            }
        )

        # Track best BEFORE backward/step so saved params match the evaluated loss
        if loss.item() < best_loss:
            best_loss = loss.item()
            best_iteration = iteration
            if use_pose_tokens:
                # Derive final pose from current raw tokens
                with torch.no_grad():
                    _, best_rot, best_trans, best_scl = differentiable_pose_decode(
                        opt_6drot, opt_raw_trans, opt_raw_scale,
                        scene_scale, scene_shift, downsample_factor,
                    )
                best_params = {
                    "rotation": best_rot.detach(),
                    "translation": best_trans.detach(),
                    "scale": best_scl.detach(),
                    "raw_modalities": {
                        "6drotation_normalized": opt_6drot.clone().detach(),
                        "translation": opt_raw_trans.clone().detach(),
                        "scale": opt_raw_scale.clone().detach(),
                    },
                }
            else:
                # Store scale as 3D tensor (1, 3) — expand scalar if isotropic
                scale_3d = opt_scale.clone().detach().expand(3).reshape(1, 3)
                best_params = {
                    "rotation": opt_rotation.clone().detach(),
                    "translation": opt_translation.clone().detach(),
                    "scale": scale_3d,
                }

        # Backprop
        loss.backward()

        # Step optimizer
        optimizer.step()

        # Project scale back to isotropic (uniform) (direct path only)
        if not use_pose_tokens and refine_scale == "perframe":
            with torch.no_grad():
                opt_scale.fill_(opt_scale.mean())

        # Early stopping: break if no improvement for patience iterations
        if losses.early_stop_patience > 0 and (iteration - best_iteration) >= losses.early_stop_patience:
            if pipeline.verbose:
                print(
                    f"      Early stopping at iteration {iteration} "
                    f"(no improvement for {losses.early_stop_patience} iters, best at {best_iteration})"
                )
            break

        if pipeline.verbose and (
            iteration % pipeline.log_interval == 0 or iteration == losses.num_iterations - 1
        ):
            silh_details = ""
            if losses.silhouette_weight > 0:
                silh_details = (
                    f" (com={silh_com.item():.4f}, sdt={silh_sdt.item():.4f}, "
                    f"iou={silh_iou.item():.4f})"
                )
            percept_str = ""
            if losses.perceptual_weight > 0:
                percept_val = weighted_perceptual_loss.item() if isinstance(weighted_perceptual_loss, torch.Tensor) else weighted_perceptual_loss
                percept_str = f"percept={percept_val:.6f} "
            depth_str = ""
            if losses.depth_weight > 0:
                depth_str = f"depth={weighted_depth_loss.item():.6f} "
            print(
                f"      Iteration {iteration}: total={loss.item():.6f} "
                f"rgb={weighted_rgb_loss.item():.6f} "
                f"ssim={weighted_ssim_loss_value.item():.6f} "
                f"{percept_str}"
                f"{depth_str}"
                f"silh={weighted_silhouette_loss_value.item():.6f}{silh_details} "
                f"reg={weighted_reg_loss.item():.8f} "
            )

    if pipeline.verbose:
        print(f"      Best loss: {best_loss:.6f} (iteration {best_iteration})")

    # Normalize the rotation in best_params
    best_params["rotation"] = best_params["rotation"] / best_params["rotation"].norm()

    # Add loss history and best iteration to the result
    best_params["loss_history"] = loss_history
    best_params["best_iteration"] = best_iteration

    return best_params
