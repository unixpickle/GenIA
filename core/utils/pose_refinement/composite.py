# Copyright (c) Meta Platforms, Inc. and affiliates.

"""Jointly refine object poses against composited scene renders."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from genia.core.device import get_device
from genia.core.utils.config import (
    LossConfig,
    PipelineConfig,
    resolve_chamfer_gt_trim,
    resolve_correction_granularity,
    resolve_correction_scale_control,
)
from genia.core.utils.frame_sampling import EpochFrameSampler
from genia.core.utils.pose_params import build_correction, freeze_params, natives_to_freeze
from genia.core.utils.pose_refinement.chamfer import (
    _chamfer_gt_points_for_object,
    _chamfer_gt_to_model,
    _chamfer_model_subsample_idx,
)
from genia.core.utils.pose_refinement.context import (
    _carry_forward_decoder_context,
    _sync_raw_from_decoded,
)
from genia.core.utils.pose_refinement.decoding import differentiable_pose_decode
from genia.core.utils.pose_refinement.frame_loss import _compute_frame_loss
from genia.core.utils.pose_refinement.loss_terms import _match_depth_grid, depth_to_normals
from genia.core.utils.pose_refinement.rendering import (
    _concat_fg_bg_params,
    _get_bg_color,
    _transform_object_to_r3,
)
from genia.core.utils.pose_refinement.shared_world import _apply_gs_correction
from genia.core.utils.pose_refinement.warping import _precompute_perframe_gaussian_warps
from genia.core.utils.rendering import render_gaussian_params
from genia.core.utils.timing import get_timer
from tqdm import tqdm


def refine_poses_global_composite(
    canonical_gaussians: Dict[int, Any],
    tokens_by_object: Dict[int, List[Tuple[int, Dict[str, Any]]]],
    sequence: Any,
    losses: LossConfig,
    pipeline: PipelineConfig,
    perframe_raw_modalities: Optional[Dict[int, Dict[int, Dict[str, Any]]]] = None,
    output_dir: Optional[str] = None,
    save_renders: bool = True,
    # Per-canonical-mesh-vertex deformation field (actionmesh mono-dynamic).
    # See ``refine_poses_for_sequence`` for full kwarg semantics.  Warps
    # are evaluated once per (obj, frame) under ``no_grad`` and cached on
    # ``frame_data[frame_idx]["overrides_per_obj"]``; the composite render
    # injects them via ``_transform_object_to_r3``'s ``means_override`` /
    # ``rotation_override`` kwargs.
    canonical_mesh_verts_per_obj: Optional[Dict[int, torch.Tensor]] = None,
    per_frame_mesh_verts_per_obj: Optional[Dict[int, Dict[int, torch.Tensor]]] = None,
    per_frame_mesh_rotations_per_obj: Optional[Dict[int, Dict[int, torch.Tensor]]] = None,
    canonical_mesh_faces_per_obj: Optional[Dict[int, torch.Tensor]] = None,
    warp_knn_k: int = 4,
    warp_knn_eps: float = 1.0e-8,
    warp_knn_chunk_size: int = 8192,
) -> Dict[int, List[Tuple[int, Dict[str, Any]]]]:
    """
    Refine poses for ALL objects jointly with composite rendering.

    Renders all objects together in a single gsplat pass per frame.
    Losses are computed on the full composite image vs the full GT image
    (union mask for silhouette, union-masked GT for appearance).

    Each object has its own global scale, per-frame rotation, and per-frame
    translation.  All are optimized jointly in a single optimizer.

    Parameters
    ----------
    canonical_gaussians : dict
        {obj_idx: Gaussian} — canonical Gaussian per object.
    tokens_by_object : dict
        {obj_idx: [(frame_idx, decoder_input), ...]} per object.
    sequence : Sequence
        Cached scene data.
    losses : LossConfig
        Per-phase loss weights and iteration count.
    pipeline : PipelineConfig
        Cross-phase pipeline control flags.

    Returns
    -------
    dict
        Refined tokens_by_object with updated poses for each object.
    """
    device = get_device()
    all_obj_indices = sorted(tokens_by_object.keys())
    num_objects = len(all_obj_indices)

    print(f"\n    Composite rendering: {num_objects} objects jointly")

    # ── Determine valid frames (union of all objects' frame sets) ──
    frame_to_obj_inputs = {}  # {frame_idx: {obj_idx: decoder_input}}
    for obj_idx in all_obj_indices:
        for frame_idx, decoder_input in tokens_by_object[obj_idx]:
            if frame_idx not in frame_to_obj_inputs:
                frame_to_obj_inputs[frame_idx] = {}
            frame_to_obj_inputs[frame_idx][obj_idx] = decoder_input

    # ── Load frame data with union masks ──
    frame_data = {}
    valid_frame_indices = []

    for frame_idx in sorted(frame_to_obj_inputs.keys()):
        frame = sequence[frame_idx]
        render_image, render_masks, render_K = frame.image, frame.masks, frame.K_matrix
        gt_image = torch.from_numpy(render_image).float().to(device) / 255.0

        # Compute union mask and per-object masks
        H, W = render_image.shape[0], render_image.shape[1]
        union_mask = torch.zeros(H, W, dtype=torch.bool, device=device)
        per_obj_masks = {}
        present_objects = []

        for obj_idx in all_obj_indices:
            m = torch.from_numpy(render_masks[obj_idx]).bool().to(device)
            per_obj_masks[obj_idx] = m
            if m.any():
                union_mask = union_mask | m
                present_objects.append(obj_idx)

        if not union_mask.any():
            print(f"      Warning: No objects visible in frame {frame_idx}, skipping")
            continue

        # GT depth and valid mask (needed for depth loss or depth-derived normals)
        gt_depth_tensor = None
        valid_mask_tensor = None
        need_depth = losses.depth_weight > 0 or losses.normals_weight > 0
        if frame.depth_map_z is not None and need_depth:
            gt_depth_tensor = torch.from_numpy(frame.depth_map_z).float().to(device)
            if frame.valid_mask is not None:
                valid_mask_tensor = torch.from_numpy(frame.valid_mask).bool().to(device)

        # K_matrix tensor (for normals loss)
        K_tensor = None
        if losses.normals_weight > 0:
            K_tensor = torch.from_numpy(render_K).float().to(device)
            if frame.valid_mask is not None and valid_mask_tensor is None:
                valid_mask_tensor = torch.from_numpy(frame.valid_mask).bool().to(device)

        frame_data[frame_idx] = {
            "gt_image": gt_image,
            "union_mask": union_mask,
            "per_obj_masks": per_obj_masks,
            "present_objects": present_objects,
            "gt_depth": gt_depth_tensor,
            "valid_mask": valid_mask_tensor,
            "K_tensor": K_tensor,
            "K_matrix": render_K,
            "H": H,
            "W": W,
            "overrides_per_obj": {},
        }
        valid_frame_indices.append(frame_idx)

    if len(valid_frame_indices) == 0:
        print("      No valid frames for composite refinement")
        return tokens_by_object

    print(f"      {len(valid_frame_indices)} valid frames, "
          f"{num_objects} objects")

    # ── Optional background Gaussians for full-image depth supervision ──
    # depth_mask_only=False scores the depth loss over the WHOLE image, which
    # is only well-defined if the render has valid depth OUTSIDE the object
    # union.  Build a pointmap background per frame and append it to the
    # composite render so has_background=True is truthful.  The background is a
    # FIXED depth reference: every param is DETACHED, so no gradient ever flows
    # into it (pose grads come only through the foreground objects).
    # NOTE: full-image mode passes an all-ones loss mask (so GT depth is not
    # zeroed outside the objects), which makes the silhouette term inert —
    # coverage is meaningless once the whole background is rendered.  Drop
    # silhouette_weight to 0 if you don't want that near-constant term.
    background_params_per_frame: Optional[Dict[int, Tuple[torch.Tensor, ...]]] = None
    if getattr(losses, "render_with_background", False):
        from genia.core.utils.gaussian import create_background_gaussians
        from genia.core.utils.rendering import transform_gaussian_params_world_to_cam
        background_params_per_frame = {}
        print(f"      render_with_background=True → building DETACHED background "
              f"Gaussians for {len(valid_frame_indices)} frames "
              f"(full-image depth; silhouette term goes inert)...")
        for frame_idx in valid_frame_indices:
            frame = sequence[frame_idx]
            # Background = everything NOT covered by any present object.
            obj_masks = [frame.masks[oi] for oi in frame_data[frame_idx]["present_objects"]]
            bg_gs = create_background_gaussians(
                frame.image, frame.pointmap, obj_masks, frame.K_matrix, c2w=frame.c2w,
            )
            if bg_gs is None:
                continue
            # World→camera (composite renders camera-space with identity c2w).
            # Detach BEFORE and AFTER the transform so the background is a pure
            # fixed reference — no autograd graph is retained on it.
            bg_xyz = bg_gs.get_xyz.detach().to(device)
            bg_rot = bg_gs.get_rotation.detach().to(device)
            bg_xyz, bg_rot = transform_gaussian_params_world_to_cam(bg_xyz, bg_rot, frame.c2w)
            background_params_per_frame[frame_idx] = (
                bg_xyz.detach(),
                bg_rot.detach(),
                bg_gs.get_scaling.detach().to(device),
                bg_gs.get_opacity.squeeze(-1).detach().to(device),
                bg_gs.get_features.detach().to(device),
            )
        print(f"      Background Gaussians ready "
              f"({len(background_params_per_frame)}/{len(valid_frame_indices)} frames)")

    # ── Optional Chamfer (GT-depth → posed-model) alignment targets ──
    # Per (frame, obj): the object's visible GT surface points from the depth
    # pointmap.  Per obj: a fixed model-point subsample.  Both cheap to reuse
    # every iter.  camera-space R3 on both sides (composite renders identity c2w).
    gt_chamfer_points = None
    chamfer_model_idx = None
    _chamfer_trim = resolve_chamfer_gt_trim(losses, "global_pose_refine")
    if getattr(losses, "chamfer_weight", 0.0) > 0:
        _max_gt = int(getattr(losses, "chamfer_max_gt_points", 2048))
        _max_model = int(getattr(losses, "chamfer_max_model_points", 8192))
        gt_chamfer_points = {}
        for frame_idx in valid_frame_indices:
            pm = torch.from_numpy(sequence[frame_idx].pointmap).float().to(device)
            _vm = frame_data[frame_idx].get("valid_mask")
            # sequence[frame_idx].masks (NOT frame_data[...]["per_obj_masks"],
            # which may be render-resolution) -- pm is always at the backbone's
            # (H', W'), and pointmap-aligned masks must match it exactly.
            gt_chamfer_points[frame_idx] = {
                oi: _chamfer_gt_points_for_object(
                    pm, torch.from_numpy(
                        np.asarray(sequence[frame_idx].masks[oi])).bool().to(device),
                    _max_gt, valid_mask=_vm, trim_factor=_chamfer_trim)
                for oi in frame_data[frame_idx]["present_objects"]
            }
        chamfer_model_idx = {
            oi: _chamfer_model_subsample_idx(
                canonical_gaussians[oi].get_xyz.shape[0], _max_model, device)
            for oi in all_obj_indices
        }
        print(f"      Chamfer alignment ON (GT→model, weight={losses.chamfer_weight}, "
              f"GT cap={_max_gt}, model cap={_max_model})")

    # ── Pre-compute per-(obj, frame) deformation warps (actionmesh) ──
    # Identity-default no-op when the deformation field is unsupplied.
    # Composite render iterates frame-major, so we invert the (frame ->
    # objects) layout into (obj -> frames) for the helper, then re-key
    # the result back into ``frame_data[frame_idx]["overrides_per_obj"]``.
    _frames_per_obj_for_warp: Dict[int, List[Any]] = {}
    for frame_idx in valid_frame_indices:
        for obj_idx in frame_data[frame_idx]["present_objects"]:
            _frames_per_obj_for_warp.setdefault(obj_idx, []).append(frame_idx)
    warp_cache = _precompute_perframe_gaussian_warps(
        canonical_gaussians, _frames_per_obj_for_warp,
        canonical_mesh_verts_per_obj,
        per_frame_mesh_verts_per_obj,
        per_frame_mesh_rotations_per_obj,
        canonical_mesh_faces_per_obj,
        warp_knn_k=warp_knn_k, warp_knn_eps=warp_knn_eps,
        warp_knn_chunk_size=warp_knn_chunk_size,
    )
    for obj_idx, per_frame in warp_cache.items():
        for frame_idx, override in per_frame.items():
            frame_data[frame_idx]["overrides_per_obj"][obj_idx] = override
    _n_warped = sum(len(v) for v in warp_cache.values())
    if _n_warped:
        print(f"      Deformation warp: cached {_n_warped} (obj, frame) overrides")

    # ── Determine per-object pose-token availability ──
    use_pose_tokens_per_obj = {}  # {obj_idx: bool}
    if pipeline.optimize_pose_tokens and perframe_raw_modalities:
        for obj_idx in all_obj_indices:
            obj_raw = perframe_raw_modalities.get(obj_idx, {})
            ok = True
            for fi in valid_frame_indices:
                if obj_idx not in frame_to_obj_inputs.get(fi, {}):
                    continue
                raw = obj_raw.get(fi)
                if (raw is None
                        or "raw_ss_modalities" not in raw
                        or "6drotation_normalized" not in raw["raw_ss_modalities"]
                        or "translation" not in raw["raw_ss_modalities"]
                        or "scale" not in raw["raw_ss_modalities"]
                        or raw.get("pointmap_scale") is None
                        or raw.get("pointmap_shift") is None):
                    ok = False
                    break
            use_pose_tokens_per_obj[obj_idx] = ok
    else:
        for obj_idx in all_obj_indices:
            use_pose_tokens_per_obj[obj_idx] = False

    any_pose_tokens = any(use_pose_tokens_per_obj.values())
    if any_pose_tokens:
        pt_objs = [oi for oi, v in use_pose_tokens_per_obj.items() if v]
        print(f"      [Pose tokens] Enabled for objects: {pt_objs}")

    # ── Initialize optimizable parameters ──
    opt_rotations = {}      # {obj_idx: {frame_idx: tensor}}
    opt_translations = {}   # {obj_idx: {frame_idx: tensor}}
    opt_global_scale = {}   # {obj_idx: tensor}
    initial_rotations = {}
    initial_translations = {}
    initial_global_scales = {}

    # Pose-token specific dicts
    opt_6drots = {}         # {obj_idx: {frame_idx: tensor}}
    opt_raw_trans = {}      # {obj_idx: {frame_idx: tensor}}
    initial_6drots = {}
    initial_raw_trans = {}
    frozen_raw_scales = {}  # {obj_idx: {frame_idx: tensor}}
    scene_scales_all = {}   # {obj_idx: {frame_idx: tensor/value}}
    scene_shifts_all = {}
    downsample_factors_all = {}

    param_groups = []

    for obj_idx in all_obj_indices:
        opt_rotations[obj_idx] = {}
        opt_translations[obj_idx] = {}
        initial_rotations[obj_idx] = {}
        initial_translations[obj_idx] = {}
        opt_6drots[obj_idx] = {}
        opt_raw_trans[obj_idx] = {}
        initial_6drots[obj_idx] = {}
        initial_raw_trans[obj_idx] = {}
        frozen_raw_scales[obj_idx] = {}
        scene_scales_all[obj_idx] = {}
        scene_shifts_all[obj_idx] = {}
        downsample_factors_all[obj_idx] = {}

        # Initialize per-object global scale from first available frame
        first_di = None
        for fi in valid_frame_indices:
            if obj_idx in frame_to_obj_inputs.get(fi, {}):
                first_di = frame_to_obj_inputs[fi][obj_idx]
                break
        if first_di is None:
            continue

        scale_flat = first_di["scale"].view(-1)
        init_scale = scale_flat.mean().view(1) if scale_flat.shape[0] >= 3 else scale_flat[:1].clone()
        _scale_lr = float(losses.lr_scale)
        opt_global_scale[obj_idx] = init_scale.detach().to(device).requires_grad_(_scale_lr > 0)
        initial_global_scales[obj_idx] = opt_global_scale[obj_idx].clone().detach()
        if _scale_lr > 0:
            param_groups.append({"params": [opt_global_scale[obj_idx]], "lr": _scale_lr})

        scale_str = f"{opt_global_scale[obj_idx].item():.6f}"
        print(f"      Object {obj_idx}: initial scale = {scale_str}")

        # Per-frame rotation and translation
        obj_raw = (perframe_raw_modalities or {}).get(obj_idx, {})
        for frame_idx in valid_frame_indices:
            if obj_idx not in frame_to_obj_inputs.get(frame_idx, {}):
                continue

            _rot_lr = float(losses.lr_rotation)
            _trans_lr = float(losses.lr_translation)
            if use_pose_tokens_per_obj[obj_idx]:
                raw = obj_raw[frame_idx]
                mods = raw["raw_ss_modalities"]
                opt_6drots[obj_idx][frame_idx] = mods["6drotation_normalized"].clone().detach().float().to(device).requires_grad_(_rot_lr > 0)
                opt_raw_trans[obj_idx][frame_idx] = mods["translation"].clone().detach().float().to(device).requires_grad_(_trans_lr > 0)
                frozen_raw_scales[obj_idx][frame_idx] = mods["scale"].clone().detach().float().to(device)
                scene_scales_all[obj_idx][frame_idx] = raw["pointmap_scale"]
                scene_shifts_all[obj_idx][frame_idx] = raw["pointmap_shift"]
                downsample_factors_all[obj_idx][frame_idx] = raw.get("downsample_factor", 1.0)
                initial_6drots[obj_idx][frame_idx] = opt_6drots[obj_idx][frame_idx].clone().detach()
                initial_raw_trans[obj_idx][frame_idx] = opt_raw_trans[obj_idx][frame_idx].clone().detach()
                if _rot_lr > 0:
                    param_groups.append({"params": [opt_6drots[obj_idx][frame_idx]], "lr": _rot_lr})
                if _trans_lr > 0:
                    param_groups.append({"params": [opt_raw_trans[obj_idx][frame_idx]], "lr": _trans_lr})
            else:
                di = frame_to_obj_inputs[frame_idx][obj_idx]
                initial_rotations[obj_idx][frame_idx] = di["rotation"].clone().detach().to(device)
                initial_translations[obj_idx][frame_idx] = di["translation"].clone().detach().to(device)
                opt_rotations[obj_idx][frame_idx] = (
                    di["rotation"].clone().detach().to(device).requires_grad_(_rot_lr > 0)
                )
                opt_translations[obj_idx][frame_idx] = (
                    di["translation"].clone().detach().to(device).requires_grad_(_trans_lr > 0)
                )
                if _rot_lr > 0:
                    param_groups.append({"params": [opt_rotations[obj_idx][frame_idx]], "lr": _rot_lr})
                if _trans_lr > 0:
                    param_groups.append({"params": [opt_translations[obj_idx][frame_idx]], "lr": _trans_lr})

    # ---- Shared correction: ONE Sim(3) per object, composed onto every frame's pose ----
    # `pose_i o dT` in the object's CANONICAL frame, so per-frame root MOTION survives --
    # unlike the shared-world path, which DERIVES each frame's pose from one reference and
    # thereby replaces it.  Orthogonal to `strategy`: this is the photometric loss driving
    # the shared correction.
    # On MV data this solver is reached ONLY with mv_shared_world_pose off (
    # `refine_poses_for_sequence` short-circuits to the shared-world path first), where its
    # per-FrameKey natives are per-VIEW -- which is what the resolver refuses.
    correction_granularity_mode = resolve_correction_granularity(
        losses, "global_pose_refine", is_mv=sequence.is_mv,
        mv_shared_world_pose=pipeline.mv_shared_world_pose)
    opt_delta_aa, opt_delta_logs, opt_delta_t = {}, {}, {}
    if correction_granularity_mode != "per_frame":
        for obj_idx in all_obj_indices:
            (opt_delta_aa[obj_idx], opt_delta_logs[obj_idx], opt_delta_t[obj_idx],
             _dg) = build_correction(
                float(losses.lr_rotation), float(losses.lr_scale),
                float(losses.lr_translation), device)
            param_groups.extend(_dg)
        _freeze = natives_to_freeze(
            correction_granularity_mode,
            resolve_correction_scale_control(losses, "global_pose_refine"))
        if _freeze == "all":
            # Freeze the native params so the delta is the ONLY thing that moves.  That
            # is what makes the fit well-posed, and what lets the result be READ as "the
            # systematic error": with both free (`correction_granularity: both`, which skips
            # this freeze) any delta is absorbable into the per-frame poses, so their
            # split is a gauge choice rather than a result.
            _delta_ids = {id(p) for d in (opt_delta_aa, opt_delta_logs, opt_delta_t)
                          for p in d.values()}
            _is_delta = [any(id(p) in _delta_ids for p in g["params"])
                         for g in param_groups]
            for g, keep in zip(param_groups, _is_delta):
                if not keep:
                    for prm in g["params"]:
                        prm.requires_grad_(False)
            param_groups = [g for g, keep in zip(param_groups, _is_delta) if keep]
        elif _freeze == "scale":
            # The object has ONE size, so the shared correction owns it and the native
            # scale is held.  Dropped from the groups rather than given lr=0, because
            # `losses.lr_scale` also drives the correction's scale and would freeze that.
            param_groups = freeze_params(param_groups, opt_global_scale.values())
        print(f"      Pose correction: {correction_granularity_mode} "
              f"({len(opt_delta_aa)} object(s), one Sim(3) each)"
              + ("; native scale FROZEN" if _freeze == "scale" else ""))

    optimizer = torch.optim.Adam(param_groups)

    # Batch size
    num_frames = len(valid_frame_indices)
    batch_size = losses.batch_size if losses.batch_size > 0 else num_frames
    batch_size = min(batch_size, num_frames)
    effective_num_iterations = losses.num_iterations

    print(f"      Batch size: {batch_size} (out of {num_frames} frames)")

    # ── Optimization loop ──
    best_loss = float("inf")
    best_params = {}
    best_iteration = 0
    loss_history = []

    if effective_num_iterations == 0:
        print("      Skipping optimization (num_iterations=0)")
        return tokens_by_object

    frame_sampler = EpochFrameSampler(valid_frame_indices)

    pbar = tqdm(
        range(effective_num_iterations),
        desc="      Composite opt",
        leave=True,
    )
    for iteration in pbar:
        optimizer.zero_grad()

        # Sample batch of frames (epoch-based: all frames seen before any repeats)
        if batch_size >= len(valid_frame_indices):
            batch_frame_indices = valid_frame_indices
        else:
            batch_frame_indices = frame_sampler.sample(batch_size)

        total_rgb_loss = 0.0
        total_ssim_loss = 0.0
        total_silhouette_loss = 0.0
        total_depth_loss = 0.0
        total_normals_loss = 0.0
        total_perceptual_loss = 0.0
        total_reg_loss = 0.0
        total_chamfer_loss = 0.0
        total_chamfer_weight = 0.0   # per-(frame,object) contributions → per-object mean, not object-count-scaled
        total_weight = 0.0

        want_pixelwise = (
            save_renders
            and output_dir is not None
            and iteration % 50 == 0
        )
        px_data: Optional[Dict[int, Dict[str, Any]]] = (
            {} if want_pixelwise else None
        )

        for frame_idx in batch_frame_indices:
            data = frame_data[frame_idx]
            frame_weight = 1.0

            # Transform and concatenate all present objects' Gaussians
            all_xyz, all_rot, all_scales, all_opac, all_feats = [], [], [], [], []

            for obj_idx in data["present_objects"]:
                # Determine current rotation and translation
                if use_pose_tokens_per_obj.get(obj_idx, False) and frame_idx in opt_6drots.get(obj_idx, {}):
                    _, cur_rot, cur_trans, _ = differentiable_pose_decode(
                        opt_6drots[obj_idx][frame_idx],
                        opt_raw_trans[obj_idx][frame_idx],
                        frozen_raw_scales[obj_idx][frame_idx],
                        scene_scales_all[obj_idx][frame_idx],
                        scene_shifts_all[obj_idx][frame_idx],
                        downsample_factors_all[obj_idx][frame_idx],
                    )
                elif frame_idx in opt_rotations.get(obj_idx, {}):
                    cur_rot = opt_rotations[obj_idx][frame_idx]
                    cur_trans = opt_translations[obj_idx][frame_idx]
                else:
                    continue
                # Shared correction, in the object's CANONICAL frame.  Identity when this
                # object has no delta (`per_frame`), so no branch is needed.
                cur_rot, cur_trans, cur_scale = _apply_gs_correction(
                    opt_delta_aa.get(obj_idx), opt_delta_logs.get(obj_idx),
                    opt_delta_t.get(obj_idx),
                    cur_rot, cur_trans, opt_global_scale[obj_idx])
                _override = data["overrides_per_obj"].get(obj_idx)
                _means_ovr, _rot_ovr = _override if _override is not None else (None, None)
                xyz, rot, scales, opac, feats = _transform_object_to_r3(
                    canonical_gaussians[obj_idx],
                    cur_rot,
                    cur_trans,
                    cur_scale,
                    device,
                    means_override=_means_ovr,
                    rotation_override=_rot_ovr,
                )
                all_xyz.append(xyz)
                all_rot.append(rot)
                all_scales.append(scales)
                all_opac.append(opac)
                all_feats.append(feats)

                # One-directional Chamfer (GT-depth → this object's posed means).
                if gt_chamfer_points is not None:
                    _gt_pts = gt_chamfer_points.get(frame_idx, {}).get(obj_idx)
                    if _gt_pts is not None:
                        total_chamfer_loss = total_chamfer_loss + frame_weight * _chamfer_gt_to_model(
                            _gt_pts, xyz, model_sub_idx=chamfer_model_idx.get(obj_idx))
                        total_chamfer_weight = total_chamfer_weight + frame_weight

            if not all_xyz:
                continue

            cat_xyz = torch.cat(all_xyz, dim=0)
            cat_rot = torch.cat(all_rot, dim=0)
            cat_scales = torch.cat(all_scales, dim=0)
            cat_opac = torch.cat(all_opac, dim=0)
            cat_feats = torch.cat(all_feats, dim=0)

            # Append the (detached) background so the render has valid depth
            # outside the object union (full-image depth mode).  All bg params
            # are detached → they add geometry to the render but carry no
            # gradient.  ``_num_fg`` marks the fg/bg split for foreground-weight.
            _num_fg = cat_xyz.shape[0]
            has_bg = (
                background_params_per_frame is not None
                and frame_idx in background_params_per_frame
            )
            if has_bg:
                cat_xyz, cat_rot, cat_scales, cat_opac, cat_feats = _concat_fg_bg_params(
                    cat_xyz, cat_rot, cat_scales, cat_opac, cat_feats,
                    background_params_per_frame[frame_idx], device,
                )

            # Single composite render (camera-space poses → identity c2w)
            _c2w = torch.eye(4, device=device, dtype=torch.float32).unsqueeze(0)
            bg_color = _get_bg_color(pipeline, device)

            rgb, alpha, depth = render_gaussian_params(
                cat_xyz, cat_rot, cat_scales, cat_opac, cat_feats,
                _c2w, data["K_matrix"], data["W"], data["H"],
                bg_color=bg_color,
            )

            # Compute losses on composite.  Foreground-only mode masks to the
            # union of per-object masks (silhouette / depth see the object
            # footprint).  Full-image mode (has_bg) passes an all-ones mask so
            # GT depth is NOT zeroed outside the objects — the background render
            # supplies valid depth there, which is what makes has_background=True
            # meaningful (at the cost of neutralising the silhouette term).
            assert "union_mask" in data, (
                "refine_poses_global_composite: composite rendering requires "
                "frame_data['union_mask'] (union of all per-object masks)."
            )
            _loss_mask = (
                torch.ones(data["H"], data["W"], device=device, dtype=torch.bool)
                if has_bg else data["union_mask"]
            )
            losses_dict = _compute_frame_loss(
                rgb, alpha, data["gt_image"], _loss_mask,
                losses,
                rendered_depth=depth, gt_depth=data.get("gt_depth"),
                valid_mask=data.get("valid_mask"),
                bg_color=bg_color,
                K_matrix=data.get("K_tensor"),
                return_pixelwise=want_pixelwise,
                has_background=has_bg,
                # Frames without a fused background (e.g. GSO/OAB foreground-only GT
                # depth) fall back to masked depth so empty-bg pixels aren't
                # supervised as noise.
                depth_mask_only=losses.depth_mask_only or not has_bg,
            )

            # Collect per-pixel data for debug visualization
            if want_pixelwise:
                _K = data.get("K_tensor")
                if _K is None:
                    _K = torch.from_numpy(data["K_matrix"]).float().to(device)
                # Mask the GT panels with the SAME mask the loss used, so the
                # visualization matches what was actually supervised: object-only
                # in foreground mode, full-image (background included) in
                # full-image / has_bg mode.
                _mask = _loss_mask
                _mask_f = _mask.float().unsqueeze(-1)
                gt_masked = data["gt_image"] * _mask_f + bg_color.view(1, 1, 3) * (1.0 - _mask_f)
                frame_px: Dict[str, Any] = {
                    "rendered_rgb": rgb.detach().cpu().numpy(),
                    "gt_rgb": gt_masked.detach().cpu().numpy(),
                    "mask": _mask.cpu().numpy(),
                }
                if depth is not None:
                    depth_sq = depth.squeeze(0) if depth.dim() == 3 else depth
                    frame_px["rendered_depth"] = depth_sq.detach().cpu().numpy()
                    frame_px["rendered_normals"] = depth_to_normals(depth_sq, _K).detach().cpu().numpy()
                gt_d = data.get("gt_depth")
                if gt_d is not None:
                    _, _K_gt, _mask_gt = _match_depth_grid(
                        _mask, gt_d.shape[-2:], K_matrix=_K, mask=_mask)
                    gt_depth_masked = gt_d * _mask_gt.float()
                    frame_px["gt_depth"] = gt_depth_masked.cpu().numpy()
                    frame_px["gt_normals"] = depth_to_normals(gt_d, _K_gt, mask=_mask_gt).detach().cpu().numpy()
                for k, v in losses_dict.items():
                    if k.startswith("px_"):
                        frame_px[k] = v
                px_data[frame_idx] = frame_px

            rgb_loss_value = losses_dict["rgb_loss"]
            ssim_loss_value = losses_dict["ssim_loss"]
            depth_loss_value = losses_dict["depth_loss"]
            normals_loss_value = losses_dict["normals_loss"]

            # Accumulate weighted losses
            total_rgb_loss += frame_weight * losses.rgb_weight * rgb_loss_value
            total_ssim_loss += frame_weight * losses.rgb_ssim_weight * ssim_loss_value
            total_depth_loss += frame_weight * losses.depth_weight * depth_loss_value
            total_normals_loss += frame_weight * losses.normals_weight * normals_loss_value
            total_perceptual_loss += frame_weight * losses.perceptual_weight * losses_dict["perceptual_loss"]

            # Silhouette loss (composite alpha vs union mask)
            total_silhouette_loss += frame_weight * losses.silhouette_weight * (
                losses.silhouette_com_weight * losses_dict["silhouette_com"]
                + losses.silhouette_sdt_weight * losses_dict["silhouette_sdt"]
                + losses.silhouette_iou_weight * losses_dict["silhouette_iou"]
            )

            total_weight += frame_weight

            # Per-object regularization
            if losses.regularization_weight > 0:
                for obj_idx in data["present_objects"]:
                    if use_pose_tokens_per_obj.get(obj_idx, False) and frame_idx in opt_6drots.get(obj_idx, {}):
                        reg_rot = ((opt_6drots[obj_idx][frame_idx] - initial_6drots[obj_idx][frame_idx]) ** 2).sum()
                        reg_trans = ((opt_raw_trans[obj_idx][frame_idx] - initial_raw_trans[obj_idx][frame_idx]) ** 2).sum()
                    elif frame_idx in opt_rotations.get(obj_idx, {}):
                        opt_rot_n = opt_rotations[obj_idx][frame_idx] / opt_rotations[obj_idx][frame_idx].norm()
                        init_rot_n = initial_rotations[obj_idx][frame_idx] / initial_rotations[obj_idx][frame_idx].norm()
                        reg_rot = ((opt_rot_n - init_rot_n) ** 2).sum()
                        reg_trans = ((opt_translations[obj_idx][frame_idx] - initial_translations[obj_idx][frame_idx]) ** 2).sum()
                    else:
                        continue
                    total_reg_loss += losses.regularization_weight * (
                        losses.regularization_rotation_weight * reg_rot
                        + losses.regularization_translation_weight * reg_trans
                    )

        if total_weight == 0:
            continue

        # Normalize by total weight
        batch_rgb_loss = total_rgb_loss / total_weight
        batch_ssim_loss = total_ssim_loss / total_weight
        batch_silhouette_loss = total_silhouette_loss / total_weight
        batch_depth_loss = total_depth_loss / total_weight
        batch_normals_loss = total_normals_loss / total_weight
        batch_perceptual_loss = total_perceptual_loss / total_weight
        batch_reg_loss = (
            total_reg_loss / len(batch_frame_indices)
            if losses.regularization_weight > 0
            else torch.tensor(0.0, device=device)
        )

        # Global scale regularization (per-object)
        if losses.regularization_weight > 0:
            for obj_idx in all_obj_indices:
                scale_reg = ((opt_global_scale[obj_idx] - initial_global_scales[obj_idx]) ** 2).sum()
                batch_reg_loss = batch_reg_loss + losses.regularization_weight * losses.regularization_scale_weight * scale_reg

        batch_chamfer_loss = (
            total_chamfer_loss / total_chamfer_weight
            if isinstance(total_chamfer_loss, torch.Tensor) and total_chamfer_weight > 0
            else torch.tensor(0.0, device=device)
        )

        total_loss = (
            batch_rgb_loss
            + batch_ssim_loss
            + batch_silhouette_loss
            + batch_depth_loss
            + batch_normals_loss
            + batch_perceptual_loss
            + batch_reg_loss
            + getattr(losses, "chamfer_weight", 0.0) * batch_chamfer_loss
        )

        # Record loss
        loss_history.append({
            "total": total_loss.item(),
            "rgb": batch_rgb_loss.item(),
            "ssim": batch_ssim_loss.item(),
            "silhouette": batch_silhouette_loss.item(),
            "depth": batch_depth_loss.item() if isinstance(batch_depth_loss, torch.Tensor) else batch_depth_loss,
            "normals": batch_normals_loss.item() if isinstance(batch_normals_loss, torch.Tensor) else batch_normals_loss,
            "perceptual": batch_perceptual_loss.item() if isinstance(batch_perceptual_loss, torch.Tensor) else batch_perceptual_loss,
            "regularization": batch_reg_loss.item() if isinstance(batch_reg_loss, torch.Tensor) else batch_reg_loss,
            "chamfer": batch_chamfer_loss.item(),
        })

        # Track best BEFORE backward/step so saved params match the evaluated loss
        if total_loss.item() < best_loss:
            best_loss = total_loss.item()
            best_iteration = iteration
            bp_rotations = {}
            bp_translations = {}
            bp_raw_6drots = {}
            bp_raw_translations = {}
            for oi in all_obj_indices:
                bp_rotations[oi] = {}
                bp_translations[oi] = {}
                if use_pose_tokens_per_obj.get(oi, False):
                    bp_raw_6drots[oi] = {fi: opt_6drots[oi][fi].clone().detach() for fi in opt_6drots[oi]}
                    bp_raw_translations[oi] = {fi: opt_raw_trans[oi][fi].clone().detach() for fi in opt_raw_trans[oi]}
                    with torch.no_grad():
                        for fi in opt_6drots[oi]:
                            _, dec_rot, dec_trans, _ = differentiable_pose_decode(
                                bp_raw_6drots[oi][fi], bp_raw_translations[oi][fi],
                                frozen_raw_scales[oi][fi],
                                scene_scales_all[oi][fi], scene_shifts_all[oi][fi],
                                downsample_factors_all[oi][fi],
                            )
                            bp_rotations[oi][fi] = dec_rot.detach()
                            bp_translations[oi][fi] = dec_trans.detach()
                else:
                    bp_rotations[oi] = {fi: opt_rotations[oi][fi].clone().detach()
                                        for fi in opt_rotations[oi]}
                    bp_translations[oi] = {fi: opt_translations[oi][fi].clone().detach()
                                           for fi in opt_translations[oi]}
            best_params = {
                "global_scales": {oi: opt_global_scale[oi].clone().detach() for oi in all_obj_indices},
                "rotations": bp_rotations,
                "translations": bp_translations,
            }
            # The shared correction rides along with the params it was evaluated
            # against.  Snapshotted HERE, inside the best-tracking branch, so it cannot
            # come from a later iteration than the poses it is composed onto.
            if opt_delta_aa:
                best_params["delta"] = {
                    oi: (opt_delta_aa[oi].clone().detach(),
                         opt_delta_logs[oi].clone().detach(),
                         opt_delta_t[oi].clone().detach())
                    for oi in opt_delta_aa
                }
            if bp_raw_6drots:
                best_params["raw_6drots"] = bp_raw_6drots
                best_params["raw_translations"] = bp_raw_translations

        # Backprop
        total_loss.backward()
        optimizer.step()

        # Save per-pixel debug plot every 50 iterations
        if want_pixelwise and px_data:
            from genia.core.utils.visualization import save_pixelwise_loss_plot
            with get_timer().exclude():
                save_pixelwise_loss_plot(px_data, iteration, 0, output_dir)

        # Progress bar
        postfix = {
            "loss": f"{total_loss.item():.4f}",
            "rgb": f"{batch_rgb_loss.item():.4f}",
        }
        if losses.perceptual_weight > 0:
            pval = batch_perceptual_loss.item() if isinstance(batch_perceptual_loss, torch.Tensor) else batch_perceptual_loss
            postfix["lpips"] = f"{pval:.4f}"
        if losses.silhouette_weight > 0:
            postfix["sil"] = f"{batch_silhouette_loss.item():.4f}"
        pbar.set_postfix(postfix)

        # Early stopping: break if no improvement for patience iterations
        if losses.early_stop_patience > 0 and (iteration - best_iteration) >= losses.early_stop_patience:
            if pipeline.verbose:
                print(
                    f"      Early stopping at iteration {iteration} "
                    f"(no improvement for {losses.early_stop_patience} iters, best at {best_iteration})"
                )
            break

    if pipeline.verbose:
        print(f"      Best loss: {best_loss:.6f} (iteration {best_iteration})")
        for oi in all_obj_indices:
            s = best_params["global_scales"][oi]
            s_str = f"{s.item():.6f}"
            print(f"      Object {oi} final scale: {s_str}")

    # ── Build refined tokens ──
    refined_tokens = {}
    for obj_idx in all_obj_indices:
        refined_tokens[obj_idx] = []
        global_scale_3d = best_params["global_scales"][obj_idx].expand(3).reshape(1, 3)

        for frame_idx, decoder_input in tokens_by_object[obj_idx]:
            has_frame = (
                (use_pose_tokens_per_obj.get(obj_idx, False) and frame_idx in opt_6drots.get(obj_idx, {}))
                or frame_idx in opt_rotations.get(obj_idx, {})
            )
            if has_frame and frame_idx in best_params["rotations"].get(obj_idx, {}):
                refined_rotation = best_params["rotations"][obj_idx][frame_idx]
                refined_rotation = refined_rotation / refined_rotation.norm()
                refined_translation = best_params["translations"][obj_idx][frame_idx]
                refined_scale_3d = global_scale_3d

                # Compose the shared correction IN.  Applied per frame inside the render
                # loop but onto a copy, so without this the fitted correction never
                # reaches the tokens: under `correction_granularity: shared` the native params
                # are frozen, and the block would return its input poses unchanged.
                # Before `_sync_raw_from_decoded` below, so raw and decoded agree.
                refined_rotation, refined_translation, refined_scale_3d = _apply_gs_correction(
                    *(best_params.get("delta", {}).get(obj_idx) or (None, None, None)),
                    refined_rotation, refined_translation, refined_scale_3d)

                refined_decoder_input = {
                    "rotation": refined_rotation,
                    "translation": refined_translation,
                    "scale": refined_scale_3d.clone(),
                    "refinement_loss_history": loss_history,
                    "refinement_batch_loss_history": loss_history,
                    "refinement_best_iteration": best_iteration,
                }

                # Carry forward raw modalities — re-derived via the SSI
                # inverse of the optimized decoded pose so the round-trip
                # ``decode(raw) == (rotation, translation, scale)`` holds
                # exactly.  The optimizer here treats global_scale as an
                # independent decoded variable (decoupled from
                # ``frozen_raw_scales``), so the optimized raw scale is
                # stale relative to the final decoded scale, and the
                # optimized raw translation was paired with the frozen raw
                # scale.  ``camera_pose_to_raw_tokens`` produces raw tokens
                # consistent with the final decoded (R, t, global_scale_3d).
                if (use_pose_tokens_per_obj.get(obj_idx, False)
                        and "raw_6drots" in best_params
                        and obj_idx in best_params.get("raw_6drots", {})):
                    obj_raw = (perframe_raw_modalities or {}).get(obj_idx, {})
                    raw_source = obj_raw.get(frame_idx, {})
                    original_raw = dict(raw_source.get("raw_ss_modalities", {}))
                    _sync_raw_from_decoded(
                        original_raw,
                        refined_decoder_input["rotation"],
                        refined_decoder_input["translation"],
                        refined_decoder_input["scale"],
                        raw_source,
                        fallback_raw_6d=best_params["raw_6drots"][obj_idx][frame_idx],
                        fallback_raw_trans=best_params["raw_translations"][obj_idx][frame_idx],
                    )
                    refined_decoder_input["raw_ss_modalities"] = original_raw
                    if raw_source.get("pointmap_scale") is not None:
                        refined_decoder_input["pointmap_scale"] = raw_source["pointmap_scale"]
                        refined_decoder_input["pointmap_shift"] = raw_source["pointmap_shift"]
                    refined_decoder_input["downsample_factor"] = raw_source.get("downsample_factor", 1.0)
                else:
                    _carry_forward_decoder_context(decoder_input, refined_decoder_input)

                refined_tokens[obj_idx].append((frame_idx, refined_decoder_input))
            else:
                refined_tokens[obj_idx].append((frame_idx, decoder_input))

    return refined_tokens
