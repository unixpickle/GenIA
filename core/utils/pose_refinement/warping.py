# Copyright (c) Meta Platforms, Inc. and affiliates.

"""Cache fixed Gaussian deformations for multi-frame pose refinement."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import torch


def _precompute_perframe_gaussian_warps(
    canonical_gaussians: Dict[int, Any],
    frames_per_obj: Dict[int, List[Any]],
    canonical_mesh_verts_per_obj: Optional[Dict[int, torch.Tensor]],
    per_frame_mesh_verts_per_obj: Optional[Dict[int, Dict[int, torch.Tensor]]],
    per_frame_mesh_rotations_per_obj: Optional[Dict[int, Dict[int, torch.Tensor]]],
    canonical_mesh_faces_per_obj: Optional[Dict[int, torch.Tensor]] = None,
    *,
    warp_knn_k: int,
    warp_knn_eps: float,
    warp_knn_chunk_size: int,
) -> Dict[int, Dict[Any, Tuple[torch.Tensor, torch.Tensor]]]:
    """Compose ``_lookup_per_frame_deformation`` + ``warp_gaussians_high_res``
    over every (obj, frame) pair, returning ``{obj: {frame: (means_w, quats_w)}}``.

    Used by global pose refinement, where canonical Gaussians are held
    fixed across the optimization loop, so the rigid-LBS warp is constant
    and can be cached once with ``no_grad``.  Returns ``{}`` when the
    deformation field is unsupplied or empty (callers should fall through
    to the static-canonical path).  Per-(obj, frame) miss inside
    ``_lookup_per_frame_deformation`` is a per-cell skip — same behaviour
    as the keyframes / evaluation / interpolation call sites.
    """
    if (
        canonical_mesh_verts_per_obj is None
        or per_frame_mesh_verts_per_obj is None
        or per_frame_mesh_rotations_per_obj is None
        or not canonical_mesh_verts_per_obj
    ):
        return {}

    from genia.core.utils.deformation import _lookup_per_frame_deformation, warp_gaussians_high_res

    cache: Dict[int, Dict[Any, Tuple[torch.Tensor, torch.Tensor]]] = {}
    for obj_idx, frame_list in frames_per_obj.items():
        if obj_idx not in canonical_mesh_verts_per_obj:
            continue
        gs_obj = canonical_gaussians.get(obj_idx)
        if gs_obj is None:
            continue
        per_obj: Dict[Any, Tuple[torch.Tensor, torch.Tensor]] = {}
        for frame_idx in frame_list:
            fi_int = (
                frame_idx.frame if hasattr(frame_idx, "frame") else int(frame_idx)
            )
            resolved = _lookup_per_frame_deformation(
                canonical_mesh_verts_per_obj,
                per_frame_mesh_verts_per_obj,
                per_frame_mesh_rotations_per_obj,
                obj_idx, fi_int, gs_obj.get_xyz.device,
                canonical_mesh_faces_per_obj,
            )
            if resolved is None:
                continue
            *_warp_core, _faces = resolved
            with torch.no_grad():
                means_w, quats_w = warp_gaussians_high_res(
                    gs_obj, *_warp_core,
                    K=int(warp_knn_k),
                    eps=float(warp_knn_eps),
                    chunk_size=int(warp_knn_chunk_size),
                    faces=_faces,
                )
            per_obj[frame_idx] = (means_w.detach(), quats_w.detach())
        if per_obj:
            cache[obj_idx] = per_obj
    return cache


def _perframe_warp_overrides(
    canonical_src: Any,
    obj_idx: int,
    frames: List[Any],
    device: torch.device,
    canonical_mesh_verts_per_obj: Optional[Dict[int, torch.Tensor]],
    per_frame_mesh_verts_per_obj: Optional[Dict[int, Dict[int, torch.Tensor]]],
    per_frame_mesh_rotations_per_obj: Optional[Dict[int, Dict[int, torch.Tensor]]],
    canonical_mesh_faces_per_obj: Optional[Dict[int, torch.Tensor]] = None,
    *,
    warp_knn_k: int,
    warp_knn_eps: float,
    warp_knn_chunk_size: int,
) -> Dict[Any, Tuple[torch.Tensor, Optional[torch.Tensor]]]:
    """``{frame: (means_override, rotation_override)}`` for ONE object — what
    :class:`PosedObjectRenderer` takes per call.

    ``{}`` when no deformation field is supplied — every lookup then misses and the
    caller renders the static canonical.
    """
    warp_kw = dict(
        warp_knn_k=warp_knn_k, warp_knn_eps=warp_knn_eps,
        warp_knn_chunk_size=warp_knn_chunk_size,
    )
    field = (canonical_mesh_verts_per_obj, per_frame_mesh_verts_per_obj,
             per_frame_mesh_rotations_per_obj, canonical_mesh_faces_per_obj)
    return _precompute_perframe_gaussian_warps(
        {obj_idx: canonical_src}, {obj_idx: frames}, *field, **warp_kw,
    ).get(obj_idx, {})
