# Copyright (c) Meta Platforms, Inc. and affiliates.

"""Visibility masks and deformation inputs for Gaussian fine-tuning."""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from genia.core.config import FinetuningConfig
from genia.core.utils.timing import get_timer


def _flag_visible_by_depth(
    xyz: torch.Tensor,
    alpha_map: torch.Tensor,
    depth_map: torch.Tensor,
    K_matrix: Any,
    W: int,
    H: int,
    margin: float,
    alpha_eps: float = 0.01,
) -> torch.Tensor:
    """Single-view per-Gaussian visibility by depth occlusion.

    Projects each Gaussian centre with *K_matrix* (identity-c2w camera space) and
    compares its camera z to the rendered expected-depth at that pixel.  A
    Gaussian is occluded (invisible) iff a surface is present there
    (``alpha > alpha_eps``) AND it sits behind it by more than *margin* (relative:
    ``z > z_surface * (1 + margin)``).  Out-of-frame / behind-camera Gaussians and
    pixels with no surface default to not-occluded-this-view; cross-view
    OR-accumulation in the caller handles "seen somewhere".

    Returns (n_gauss,) bool on ``xyz.device``.
    """
    K = torch.as_tensor(K_matrix, dtype=torch.float32, device=xyz.device).reshape(3, 3)
    depth_map = depth_map.reshape(H, W)
    alpha_map = alpha_map.reshape(H, W)
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    x, y, z = xyz[:, 0], xyz[:, 1], xyz[:, 2]
    eps = 1e-6
    zc = z.clamp_min(eps)
    u = (fx * x / zc + cx).round().long()
    v = (fy * y / zc + cy).round().long()
    in_view = (z > eps) & (u >= 0) & (u < W) & (v >= 0) & (v < H)
    uc = u.clamp(0, W - 1)
    vc = v.clamp(0, H - 1)
    z_surf = depth_map[vc, uc]
    has_surface = alpha_map[vc, uc] > alpha_eps
    occluded = has_surface & (z > z_surf * (1.0 + margin))
    return in_view & ~occluded


def _compute_gaussian_visibility(
    canonical_gs: Any,
    poses: Dict[int, Dict[str, torch.Tensor]],
    frames: List[int],
    sequence: Any,
    obj_idx: int,
    device: torch.device,
    ft_config: FinetuningConfig,
    warp: Optional[Dict[str, Any]] = None,
    per_frame_out: Optional[list] = None,
) -> Optional[torch.Tensor]:
    """Per-Gaussian train-view visibility, OR-accumulated over *frames*.

    Occlusion test: render the object's expected-depth map and flag a Gaussian
    invisible iff its projected centre lies behind the surface by more than
    ``visibility_depth_margin`` (relative) in every frame (see
    :func:`_flag_visible_by_depth`).

    Self-occlusion only (no inter-object compositing).  When *warp* is given
    (per-frame mesh deformation bundle from :func:`_visibility_warp_bundle`), each
    frame's Gaussians are warped to that frame's deformed shape BEFORE the rigid
    pose, so the OR over frames means "occluded across the whole DEFORMING
    sequence" — a voxel whose back rotates/articulates into view at any frame is
    NOT flagged invisible.  Without *warp* (static objects) the rigid canonical is
    used.  Cross-object occlusion is not modelled.

    Returns
    -------
    (n_gauss,) bool tensor on *device*, or ``None`` when no frame had a valid
    pose/frame_data (caller should then skip whatever consumes this).
    """
    from genia.core.utils.pose_refinement.frame_loss import _prepare_frame_data_for_refinement
    from genia.core.utils.pose_refinement.rendering import _transform_object_to_r3
    from genia.core.utils.rendering import render_gaussian_params
    if warp is not None:
        from genia.core.utils.deformation import warp_gaussians_high_res

    c2w = torch.eye(4, device=device, dtype=torch.float32).unsqueeze(0)

    visible_gauss: Optional[torch.Tensor] = None  # (n_gauss,) bool, OR-accumulated
    for frame_idx in frames:
        if frame_idx not in poses:
            continue
        frame_data = _prepare_frame_data_for_refinement(sequence, frame_idx, obj_idx)
        if frame_data is None:
            continue

        pose = poses[frame_idx]
        # Per-frame deformation warp (dynamic objects): warp the canonical to this
        # frame's deformed shape BEFORE the rigid pose, so occlusion is tested on
        # the geometry as it actually appears this frame.  None ⇒ rigid canonical.
        means_override = rotation_override = None
        if warp is not None:
            fi_int = frame_idx.frame if hasattr(frame_idx, "frame") else int(frame_idx)
            if fi_int in warp["per_frame_mesh_verts"]:
                means_override, rotation_override = warp_gaussians_high_res(
                    canonical_gs, warp["canonical_mesh_verts"],
                    warp["per_frame_mesh_verts"][fi_int],
                    warp["per_frame_mesh_rotations"][fi_int],
                    K=int(warp["knn_k"]), eps=float(warp["knn_eps"]),
                    chunk_size=int(warp["knn_chunk"]),
                    faces=warp["canonical_mesh_faces"],
                )
        with torch.no_grad():
            xyz, rot, scales, opac, feats = _transform_object_to_r3(
                canonical_gs, pose["rotation"], pose["translation"], pose["scale"], device,
                means_override=means_override, rotation_override=rotation_override,
            )

        # Forward-only occlusion test against the rendered expected-depth — no
        # backward, so it all runs under no_grad (pass-through attrs may still
        # carry the freed decode graph; no_grad keeps it off).
        with torch.no_grad():
            _, alpha, depth = render_gaussian_params(
                xyz, rot, scales, opac, feats,
                c2w, frame_data["K_matrix"], frame_data["W"], frame_data["H"],
            )
            vis = _flag_visible_by_depth(
                xyz, alpha, depth, frame_data["K_matrix"],
                frame_data["W"], frame_data["H"], ft_config.visibility_depth_margin,
            )

        if per_frame_out is not None:
            per_frame_out.append((frame_idx, vis))
        visible_gauss = vis if visible_gauss is None else (visible_gauss | vis)

    return visible_gauss


_GAUSSIAN_RAW_ATTRS = ("_xyz", "_features_dc", "_features_rest",
                       "_scaling", "_rotation", "_opacity")


def _visibility_warp_bundle(
    canonical_mesh_verts: Any,
    per_frame_mesh_verts: Any,
    per_frame_mesh_rotations: Any,
    canonical_mesh_faces: Any,
    knn_k: int,
    knn_eps: float,
    knn_chunk: int,
) -> Optional[Dict[str, Any]]:
    """Bundle the per-frame deformation-warp inputs for the visibility mask, or
    ``None`` when no deformation field is loaded (static objects → rigid canonical).
    With it, the visibility OR-over-frames means 'never visible across the DEFORMING
    sequence' (see :func:`_compute_gaussian_visibility`)."""
    if per_frame_mesh_verts is None or canonical_mesh_verts is None:
        return None
    return {
        "canonical_mesh_verts": canonical_mesh_verts,
        "per_frame_mesh_verts": per_frame_mesh_verts,
        "per_frame_mesh_rotations": per_frame_mesh_rotations,
        "canonical_mesh_faces": canonical_mesh_faces,
        "knn_k": knn_k, "knn_eps": knn_eps, "knn_chunk": knn_chunk,
    }


def _setup_visibility_mask(
    initial_gs: Any,
    poses: Dict[int, Dict[str, torch.Tensor]],
    frames: List[int],
    sequence: Any,
    obj_idx: int,
    n_tokens: int,
    device: torch.device,
    ft_config: FinetuningConfig,
    warp: Optional[Dict[str, Any]] = None,
    coords: Optional[torch.Tensor] = None,
    output_dir: Optional[str] = None,
    scene_name: Optional[str] = None,
) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
    """One-time (stable) visibility for the FINETUNE gradient mask.

    Computes per-Gaussian train-view visibility ONCE (depth occlusion test;
    recomputing per-iter would let boundary voxels flicker in and accumulate gradient)
    and returns two stable masks:

    - ``invisible_gauss`` (n_gauss,): Gaussians occluded in EVERY train view.  Their
      decoded attributes are stop-gradient'd before each render
      (:func:`_detach_invisible_gaussians`) — closes the decoder leak and kills
      their own wrong gradient, to neither the tokens nor the shared LoRA decoder.
    - ``frozen_tokens`` (n_tokens,): voxels whose 32 Gaussians are ALL occluded.
      Their ``opt_feats.grad`` row is zeroed each iter so the token itself stays at
      the appearance-init prior.  The detach alone cannot freeze a token — it is
      SHARED by 32 spread Gaussians, so one visible sibling drags the whole voxel
      (incl. its occluded Gaussians), and ``token_drift`` nudges it too.

    A voxel with even one visible Gaussian is NOT frozen (it carries visible
    content); its occluded Gaussians still follow the token — the irreducible
    shared-token cost.  Returns ``(None, None)`` when no train frame had a valid
    pose.  Self-occlusion only (see :func:`_compute_gaussian_visibility`).
    """
    pf = [] if (output_dir is not None and coords is not None) else None
    vis = _compute_gaussian_visibility(
        initial_gs, poses, [f for f in frames if f in poses],
        sequence, obj_idx, device, ft_config, warp=warp,
        per_frame_out=pf,
    )
    if vis is None:
        return None, None
    invisible = ~vis
    frozen_tokens = ~vis.view(n_tokens, -1).any(dim=1)
    print(f"  Visibility mask (obj {obj_idx}): "
          f"{int(invisible.sum())}/{invisible.numel()} Gaussians occluded, "
          f"{int(frozen_tokens.sum())}/{n_tokens} voxels frozen "
          f"({100.0 * frozen_tokens.float().mean().item():.1f}%)")
    if output_dir is not None and coords is not None and pf:
        try:
            from genia.core.visualization import save_perframe_voxel_visibility_plot
            with get_timer().exclude():
                coords_np = (coords[:, 1:] if coords.shape[1] == 4 else coords).detach().cpu().numpy()
                vis_per_frame = np.stack([
                    v.view(n_tokens, -1).any(dim=1).float().cpu().numpy() for _f, v in pf])
                frame_poses = [(f, poses[f]) for f, _v in pf]
                save_perframe_voxel_visibility_plot(
                    coords_np, vis_per_frame, frame_poses, sequence, obj_idx,
                    output_path=os.path.join(output_dir, f"{scene_name}_obj{obj_idx}_visibility.png"),
                    title=f"{scene_name} obj {obj_idx} — FINETUNE visibility (occluded across the whole sequence)")
            print(f"  Visibility mask (obj {obj_idx}): saved per-frame visibility plot")
        except Exception as e:
            print(f"  Visibility mask (obj {obj_idx}): plot skipped ({e})")
    return invisible, frozen_tokens
