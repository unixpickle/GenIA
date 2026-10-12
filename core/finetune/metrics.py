# Copyright (c) Meta Platforms, Inc. and affiliates.

"""Evaluate a decoded object across its training frames."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch


def _compute_mean_psnr_ssim(
    gs: Any,
    poses: Dict[Any, Dict[str, torch.Tensor]],
    frame_indices: List[Any],
    sequence: Any,
    obj_idx: int,
    device: torch.device,
    *,
    bg_color: Optional[torch.Tensor] = None,
    # actionmesh deformation (same contract as ``compute_total_loss``)
    canonical_mesh_verts: "torch.Tensor | None" = None,
    per_frame_mesh_verts: "Dict[int, torch.Tensor] | None" = None,
    per_frame_mesh_rotations: "Dict[int, torch.Tensor] | None" = None,
    canonical_mesh_faces: "torch.Tensor | None" = None,
    warp_knn_k: int = 4,
    warp_knn_eps: float = 1.0e-8,
    warp_knn_chunk_size: int = 131072,
) -> Tuple[Optional[float], Optional[float]]:
    """Mean masked PSNR / SSIM of ``gs`` across all frames.

    Renders ``gs`` per-frame with the supplied poses (optionally warped
    via the per-canonical-mesh-vertex deformation field), then computes
    PSNR / SSIM against the masked GT image — same masking rule the
    FINETUNE loss uses. Returns ``(None, None)`` if no frame contributes.
    """
    from genia.core.utils.pose_refinement.frame_loss import _prepare_frame_data_for_refinement
    from genia.core.utils.pose_refinement.loss_terms import _compute_ssim_loss
    from genia.core.utils.pose_refinement.rendering import _render_frame_with_pose
    use_warp = (
        canonical_mesh_verts is not None
        and per_frame_mesh_verts is not None
        and per_frame_mesh_rotations is not None
    )
    if use_warp:
        from genia.core.utils.deformation import warp_gaussians_high_res

    psnrs: List[float] = []
    ssims: List[float] = []
    with torch.no_grad():
        for fi in frame_indices:
            if fi not in poses:
                continue
            data = _prepare_frame_data_for_refinement(sequence, fi, obj_idx)
            if data is None:
                continue

            means_override = rotation_override = None
            if use_warp:
                fi_int = fi.frame if hasattr(fi, "frame") else int(fi)
                if fi_int in per_frame_mesh_verts:
                    means_override, rotation_override = warp_gaussians_high_res(
                        gs,
                        canonical_mesh_verts,
                        per_frame_mesh_verts[fi_int],
                        per_frame_mesh_rotations[fi_int],
                        K=int(warp_knn_k),
                        eps=float(warp_knn_eps),
                        chunk_size=int(warp_knn_chunk_size),
                        faces=canonical_mesh_faces,
                    )

            rgb, _alpha, _depth = _render_frame_with_pose(
                gs,
                poses[fi]["rotation"], poses[fi]["translation"], poses[fi]["scale"],
                data["K_matrix"], data["W"], data["H"], device,
                bg_color=bg_color,
                means_override=means_override,
                rotation_override=rotation_override,
            )

            # _prepare_frame_data_for_refinement returns numpy ndarrays
            # (uint8 image in [0, 255]; bool mask) — convert + normalize.
            gt = torch.from_numpy(data["image"]).float().to(device) / 255.0
            mask = torch.from_numpy(np.asarray(data["mask"])).bool().to(device)
            mask3 = mask.unsqueeze(-1).float()
            bg = (
                bg_color.view(1, 1, 3)
                if bg_color is not None
                else torch.zeros(1, 1, 3, device=device)
            )
            gt_masked = gt * mask3 + bg * (1.0 - mask3)
            rgb_masked = rgb * mask3 + bg * (1.0 - mask3)

            mse = ((rgb_masked - gt_masked) ** 2).mean()
            psnr = -10.0 * torch.log10(mse + 1e-8)
            ssim = 1.0 - _compute_ssim_loss(rgb_masked, gt_masked)
            psnrs.append(psnr.item())
            ssims.append(ssim.item())

    if not psnrs:
        return None, None
    return sum(psnrs) / len(psnrs), sum(ssims) / len(ssims)
