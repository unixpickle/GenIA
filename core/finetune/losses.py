# Copyright (c) Meta Platforms, Inc. and affiliates.

"""Fine-tuning reconstruction losses and background sampling."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from genia.core.finetune.poses import compose_pose_with_root_delta
from genia.core.utils.config import LossConfig
from genia.core.utils.gaussian import attach_sh_rest


def _sample_random_bg_color(
    device: torch.device,
    *,
    seed_base: Optional[int],
    iter_idx: int,
    frame_idx: Any,
) -> torch.Tensor:
    """Per-(iter, frame) uniform RGB used for ``random_background``.

    With ``seed_base=None`` (default) we draw from the **global** torch
    RNG.  When ``seed_base`` is set
    (typically the run's master seed), the colour is derived from
    ``hash((seed_base, iter_idx, frame))``: identical across reruns of
    the same config, but still varies per (iter, frame) so the augmentation
    intent — preventing fixed-BG bake-in — is preserved.

    ``frame_idx`` may be a :class:`FrameKey`, an int, or any hashable.
    """
    if seed_base is None:
        return torch.rand(3, device=device)
    fi_key: Any
    if hasattr(frame_idx, "frame") and hasattr(frame_idx, "view"):
        fi_key = (int(frame_idx.frame), int(frame_idx.view))
    else:
        fi_key = int(frame_idx)
    # Stable across PYTHONHASHSEED runs because tuple/int hashing is
    # itself deterministic; modulo keeps the seed inside a Generator's
    # accepted range.
    seed = (hash((int(seed_base), int(iter_idx), fi_key)) & 0xFFFFFFFF)
    g = torch.Generator(device="cpu").manual_seed(seed)
    return torch.rand(3, generator=g).to(device)


def compute_total_loss(
    canonical_gs: Any,
    poses: Dict[int, Dict[str, torch.Tensor]],
    frame_indices: List[int],
    sequence: Any,
    obj_idx: int,
    losses: LossConfig,
    device: torch.device,
    perceptual_scale: float = 0.5,
    sh_rest_per_frame: Optional[Dict[int, torch.Tensor]] = None,
    dc_offset_per_frame: Optional[Dict[int, torch.Tensor]] = None,
    bg_color: Optional[torch.Tensor] = None,
    return_pixelwise: bool = False,
    microbatch_size: int = 0,
    root_delta: Optional[Dict[str, torch.Tensor]] = None,
    # ── actionmesh: high-res per-canonical-mesh-vertex deformation field ──
    # All three must be supplied together; ``None`` falls back to the
    # rigid-Sim(3)-only render path.  Identical contract to
    # ``build_appearance_rendering_guidance_transform``.
    canonical_mesh_verts: "torch.Tensor | None" = None,
    per_frame_mesh_verts: "Dict[int, torch.Tensor] | None" = None,
    per_frame_mesh_rotations: "Dict[int, torch.Tensor] | None" = None,
    canonical_mesh_faces: "torch.Tensor | None" = None,
    warp_knn_k: int = 4,
    warp_knn_eps: float = 1.0e-8,
    warp_knn_chunk_size: int = 131072,
    # ── per-render BG randomization ──
    random_background: bool = False,
    random_background_seed: Optional[int] = None,
    random_background_iter: int = 0,
) -> Tuple[torch.Tensor, Dict[str, float], Optional[Dict[int, Dict[str, np.ndarray]]]]:
    """Render each frame and accumulate loss.

    Uses the shared functions from ``refinement.py``:
    - ``_prepare_frame_data_for_refinement`` for GT data loading
    - ``_render_frame_with_pose`` for differentiable rendering
    - ``_compute_frame_loss`` for per-frame loss computation

    Parameters
    ----------
    return_pixelwise : bool
        When True, also return per-frame per-pixel error maps (as numpy
        arrays) for debug visualization.

    Returns
    -------
    tuple
        ``(total_loss, metrics, pixelwise_data)`` where *pixelwise_data* is
        ``None`` when *return_pixelwise* is False, or a dict
        ``{frame_idx: {"rendered_rgb": (H,W,3), "gt_rgb": (H,W,3),
        "mask": (H,W), "px_rgb_error": (H,W), ...}}`` when True.
    """
    from genia.core.utils.pose_refinement.aggregation import _aggregate_frame_loss
    from genia.core.utils.pose_refinement.frame_loss import (
        _compute_frame_loss,
        _prepare_frame_data_for_refinement,
    )
    from genia.core.utils.pose_refinement.rendering import _render_frame_with_pose

    # Validate deformation kwargs (asymmetric supply raises early so a
    # half-wired call site doesn't silently fall back to rigid-only).
    _has_canon_mesh = canonical_mesh_verts is not None
    _has_pf_verts = per_frame_mesh_verts is not None
    _has_pf_R = per_frame_mesh_rotations is not None
    if not (_has_canon_mesh == _has_pf_verts == _has_pf_R):
        raise ValueError(
            "compute_total_loss: canonical_mesh_verts, per_frame_mesh_verts "
            "and per_frame_mesh_rotations must be supplied together "
            f"(got canonical_mesh_verts={'set' if _has_canon_mesh else 'None'}, "
            f"per_frame_mesh_verts={'set' if _has_pf_verts else 'None'}, "
            f"per_frame_mesh_rotations={'set' if _has_pf_R else 'None'})."
        )
    use_warp = _has_canon_mesh
    if use_warp:
        from genia.core.utils.deformation import warp_gaussians_high_res

    # Gradient accumulation: process frames in micro-batches, backward per
    # chunk so only one chunk's rendering+VGG graph is in memory at a time.
    # The decode_tokens graph (shared across frames) is retained until the
    # last chunk via retain_graph=True.
    valid_frames = [fi for fi in frame_indices if fi in poses]
    n_total_valid = len(valid_frames)
    use_grad_accum = microbatch_size > 0 and n_total_valid > microbatch_size

    total_loss = torch.tensor(0.0, device=device, requires_grad=True)
    metrics = {
        "rgb": 0.0, "ssim": 0.0, "silhouette": 0.0, "depth": 0.0,
        "normals": 0.0, "perceptual": 0.0,
        "total": 0.0,
    }
    n_frames = 0
    pixelwise_data: Optional[Dict[int, Dict[str, np.ndarray]]] = (
        {} if return_pixelwise else None
    )

    _chunk_loss: Optional[torch.Tensor] = None
    _frames_in_chunk = 0

    iter_frames = valid_frames if use_grad_accum else frame_indices
    for vi, frame_idx in enumerate(iter_frames):
        if not use_grad_accum and frame_idx not in poses:
            continue

        # Start a new micro-batch
        if use_grad_accum and _frames_in_chunk == 0:
            _chunk_loss = torch.tensor(0.0, device=device, requires_grad=True)

        frame_data = _prepare_frame_data_for_refinement(sequence, frame_idx, obj_idx)
        if frame_data is None:
            continue

        pose = poses[frame_idx]
        if root_delta is not None:
            # ONE Sim(3) per object composed onto this frame's own pose, so the
            # per-frame motion is preserved rather than replaced.
            pose = compose_pose_with_root_delta(pose, root_delta)
        gt_image = torch.from_numpy(frame_data["image"]).float().to(device) / 255.0
        mask_tensor = torch.from_numpy(frame_data["mask"]).bool().to(device)

        # Apply per-frame DC offset (base color varies per frame)
        _saved_dc = None
        if dc_offset_per_frame is not None and frame_idx in dc_offset_per_frame:
            _saved_dc = canonical_gs._features_dc
            canonical_gs._features_dc = _saved_dc + dc_offset_per_frame[frame_idx]

        # Attach per-frame SH coefficients before rendering — save state
        # so it can be restored after.  The mutation otherwise leaks into
        # the next frame's render in this same iter, which silently
        # corrupts the loss when SH coverage is partial across frames.
        _saved_sh: Optional[Tuple[Optional[torch.Tensor], int, int]] = None
        if sh_rest_per_frame is not None and frame_idx in sh_rest_per_frame:
            _saved_sh = (
                getattr(canonical_gs, "_features_rest", None),
                getattr(canonical_gs, "sh_degree", 0),
                getattr(canonical_gs, "active_sh_degree", 0),
            )
            attach_sh_rest(canonical_gs, sh_rest_per_frame[frame_idx])

        # actionmesh: warp Gaussians via per-canonical-mesh-vertex Φ + R
        # BEFORE applying Stage-1 rigid pose.  Per-frame DC / SH mutations
        # above stay valid — only means / rotation are overridden in
        # apply_pose_to_gaussian; scales, opacity, features pass through
        # unchanged.  Identity Φ collapses to canonical-bit-identical.
        # The deformation-field dicts are keyed by plain ``int`` frame
        # indices (built by ``compute_canonical_mesh_correspondence``);
        # ``frame_idx`` may be a FrameKey for multi-view actionmesh, so
        # extract ``.frame`` before lookup — same idiom used in the
        # rendering-guidance / DDA / viz call sites.
        means_override_i = None
        rotation_override_i = None
        if use_warp:
            fi_int = frame_idx.frame if hasattr(frame_idx, "frame") else int(frame_idx)
            if fi_int in per_frame_mesh_verts:
                means_override_i, rotation_override_i = warp_gaussians_high_res(
                    canonical_gs,
                    canonical_mesh_verts,
                    per_frame_mesh_verts[fi_int],
                    per_frame_mesh_rotations[fi_int],
                    K=int(warp_knn_k),
                    eps=float(warp_knn_eps),
                    chunk_size=int(warp_knn_chunk_size),
                    faces=canonical_mesh_faces,
                )

        # Per-render BG color: optionally sample a fresh uniform RGB so the
        # model can't bake a fixed BG color into the Gaussians.  The same color
        # must be used for both the rasterizer AND the GT-image masking inside
        # ``_compute_frame_loss`` — otherwise foreground gradients pick up a
        # spurious "match-background" signal.
        bg_color_frame = bg_color
        if random_background:
            bg_color_frame = _sample_random_bg_color(
                device,
                seed_base=random_background_seed,
                iter_idx=random_background_iter,
                frame_idx=frame_idx,
            )
        rgb, alpha, depth = _render_frame_with_pose(
            canonical_gs,
            pose["rotation"], pose["translation"], pose["scale"],
            frame_data["K_matrix"],
            frame_data["W"], frame_data["H"],
            device,
            bg_color=bg_color_frame,
            means_override=means_override_i,
            rotation_override=rotation_override_i,
        )
        # Restore DC after rendering (DC offset mode)
        if _saved_dc is not None:
            canonical_gs._features_dc = _saved_dc

        # Mirror DC: restore SH state so the next frame's render starts
        # from the canonical, not the previous frame's mutated, state.
        if _saved_sh is not None:
            prev_rest, prev_degree, prev_active = _saved_sh
            if prev_rest is None:
                # Original Gaussian had no _features_rest — drop the
                # attribute we just set in attach_sh_rest.
                if hasattr(canonical_gs, "_features_rest"):
                    delattr(canonical_gs, "_features_rest")
            else:
                canonical_gs._features_rest = prev_rest
            canonical_gs.sh_degree = prev_degree
            canonical_gs.active_sh_degree = prev_active

        gt_depth_tensor = None
        valid_mask_tensor = None
        need_depth = losses.depth_weight > 0 or losses.normals_weight > 0
        if need_depth and frame_data["gt_depth"] is not None:
            gt_depth_tensor = torch.from_numpy(frame_data["gt_depth"]).float().to(device)
            if frame_data["valid_mask"] is not None:
                valid_mask_tensor = torch.from_numpy(frame_data["valid_mask"]).bool().to(device)

        K_tensor = None
        if losses.normals_weight > 0:
            K_tensor = torch.from_numpy(frame_data["K_matrix"]).float().to(device)
            if valid_mask_tensor is None and frame_data["valid_mask"] is not None:
                valid_mask_tensor = torch.from_numpy(frame_data["valid_mask"]).bool().to(device)

        losses_dict = _compute_frame_loss(
            rgb, alpha, gt_image, mask_tensor, losses,
            rendered_depth=depth, gt_depth=gt_depth_tensor,
            valid_mask=valid_mask_tensor,
            bg_color=bg_color_frame,
            perceptual_scale=perceptual_scale,
            K_matrix=K_tensor,
            return_pixelwise=return_pixelwise,
            has_background=False,
        )

        # Collect per-pixel data for debug visualization
        if return_pixelwise:
            from genia.core.utils.pose_refinement.loss_terms import depth_to_normals as _d2n

            frame_px = {
                "rendered_rgb": rgb.detach().cpu().numpy(),
                "gt_rgb": gt_image.cpu().numpy(),
                "mask": mask_tensor.cpu().numpy(),
            }
            _K = torch.from_numpy(frame_data["K_matrix"]).float().to(device)

            # Rendered depth + depth-derived normals
            if depth is not None:
                depth_sq = depth.squeeze(0) if depth.dim() == 3 else depth
                frame_px["rendered_depth"] = depth_sq.detach().cpu().numpy()
                normals = _d2n(depth_sq, _K)  # (H, W, 3)
                frame_px["rendered_normals"] = normals.detach().cpu().numpy()

            # GT depth + depth-derived GT normals
            if frame_data["gt_depth"] is not None:
                gt_d = frame_data["gt_depth"]  # numpy (H, W)
                frame_px["gt_depth"] = gt_d
                gt_d_t = torch.from_numpy(gt_d).float().to(device)
                frame_px["gt_normals"] = _d2n(gt_d_t, _K).cpu().numpy()

            for k, v in losses_dict.items():
                if k.startswith("px_"):
                    frame_px[k] = v
            pixelwise_data[frame_idx] = frame_px

        # The same aggregator rendering guidance uses; `terms_out` supplies the
        # per-term breakdown for `metrics`.  It also isolates non-finite terms,
        # which matters here: the backward below has no finiteness gate, so a
        # NaN term would reach Adam's moments and never wash out.
        _g_terms: dict = {}
        frame_loss = _aggregate_frame_loss(
            losses_dict, losses, terms_out=_g_terms,
        )

        if use_grad_accum:
            _chunk_loss = _chunk_loss + frame_loss
            _frames_in_chunk += 1
        else:
            total_loss = total_loss + frame_loss
        n_frames += 1

        # Per-term metrics; their sum is what the grad-accum path below
        # reconstructs as the reported total.
        for _k in ("rgb", "ssim", "silhouette", "depth", "normals",
                   "perceptual"):
            metrics[_k] += _g_terms[_k].item()

        # Backward at micro-batch boundary (frees chunk's rendering graph)
        if use_grad_accum and _frames_in_chunk >= microbatch_size:
            more_frames = (vi < len(iter_frames) - 1)
            (_chunk_loss / max(n_total_valid, 1)).backward(
                retain_graph=more_frames,
            )
            _chunk_loss = None
            _frames_in_chunk = 0

    # Backward remaining partial micro-batch (last chunk, release graph)
    if use_grad_accum and _chunk_loss is not None and _frames_in_chunk > 0:
        (_chunk_loss / max(n_total_valid, 1)).backward(retain_graph=False)
        _chunk_loss = None

    if n_frames > 0:
        if not use_grad_accum:
            total_loss = total_loss / n_frames
        for k in metrics:
            metrics[k] /= n_frames

    if use_grad_accum:
        # Backward already done per chunk — return detached scalar
        total_val = sum(metrics[k] for k in metrics if k != "total")
        total_loss = torch.tensor(total_val, device=device)

    metrics["total"] = total_loss.item()
    return total_loss, metrics, pixelwise_data
