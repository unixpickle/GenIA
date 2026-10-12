# Copyright (c) Meta Platforms, Inc. and affiliates.

"""Select and coordinate pose-refinement strategies for a sequence."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

import torch
from genia.core.utils.config import LossConfig, PipelineConfig
from genia.core.utils.pose_refinement.composite import refine_poses_global_composite
from genia.core.utils.pose_refinement.per_frame import refine_pose_for_frame
from genia.core.utils.pose_refinement.rendering import make_posed_object_renderer
from genia.core.utils.pose_refinement.shared_world import refine_poses_shared_world_for_sequence
from tqdm import tqdm

if TYPE_CHECKING:
    from genia.core.utils.pose_refinement.rendering import PosedObjectRenderer



def refine_poses_for_sequence(
    object_gaussians: Dict[int, Any],
    tokens_by_object: Dict[int, List[Tuple[int, Dict[str, Any]]]],
    sequence: Any,
    losses: LossConfig,
    pipeline: PipelineConfig,
    refine_scale: str = "perframe",
    per_frame_canonical: bool = False,
    perframe_raw_modalities: Optional[Dict[int, Dict[int, Dict[str, Any]]]] = None,
    output_dir: Optional[str] = None,
    save_renders: bool = True,
    canon_frame_per_object: Optional[Dict[int, int]] = None,
    # Per-canonical-mesh-vertex deformation field (actionmesh mono-dynamic).
    # State-shaped dicts: canonical_mesh_verts_per_obj[obj_idx] -> (V, 3),
    # per_frame_mesh_verts_per_obj[obj_idx][frame_int] -> (V, 3),
    # per_frame_mesh_rotations_per_obj[obj_idx][frame_int] -> (V, 3, 3).
    # When supplied, the canonical Gaussian is warped per (obj, frame)
    # via ``warp_gaussians_high_res`` BEFORE the Stage-1 rigid pose is
    # applied — consumed by the global-scale, composite and shared-world
    # paths.  Identity-default ``None`` collapses to the rigid render.
    canonical_mesh_verts_per_obj: Optional[Dict[int, torch.Tensor]] = None,
    per_frame_mesh_verts_per_obj: Optional[Dict[int, Dict[int, torch.Tensor]]] = None,
    per_frame_mesh_rotations_per_obj: Optional[Dict[int, Dict[int, torch.Tensor]]] = None,
    canonical_mesh_faces_per_obj: Optional[Dict[int, torch.Tensor]] = None,
    warp_knn_k: int = 4,
    warp_knn_eps: float = 1.0e-8,
    warp_knn_chunk_size: int = 8192,
) -> Dict[int, List[Tuple[int, Dict[str, Any]]]]:
    """
    Refine per-frame poses for all objects using differentiable rendering.

    Parameters
    ----------
    object_gaussians : dict
        The Gaussians rendered to drive pose refinement. NOT necessarily
        canonical: with ``per_frame_canonical=True`` (the per-frame
        ``refine_geometry: own_frame`` scope) this is the per-frame predictions
        ``{obj_idx: {frame_idx: Gaussian}}``; with
        ``per_frame_canonical=False`` it is one shared Gaussian per object
        ``{obj_idx: Gaussian}``.
    tokens_by_object : dict
        Dictionary mapping object_index -> list of (frame_index, decoder_input).
    sequence : Sequence
        Cached scene data.
    losses : LossConfig
        Per-phase loss weights and iteration count.
    pipeline : PipelineConfig
        Cross-phase pipeline control flags.
    refine_scale : str
        Scale refinement mode: "perframe" (per-frame scale optimization) or
        "global" (shared scale across all frames via batch optimization).
    per_frame_canonical : bool, optional
        If True, use per-frame canonical Gaussians (standard mode).
        If False, use shared canonical Gaussians across frames (averaged-tokens mode).
    perframe_raw_modalities : dict, optional
        Raw Stage 1 modalities per frame, structured as
        ``{obj_idx: {frame_idx: {"raw_ss_modalities": {...}, "pointmap_scale": ...,
        "pointmap_shift": ..., "downsample_factor": ...}}}``.
        Used for pose token optimization in global refinement mode.
    save_renders : bool
        False skips the per-iteration ``debug_pixelwise/`` PNGs (and the CPU copies
        feeding them) in every branch.  Pass the block's RESOLVED flag,
        ``get_block_output_flag(cfg, block, "save_renders")`` — raw
        ``cfg.output.save_renders`` ignores ``suppress_intermediate_renders``.

    Returns
    -------
    dict
        Refined tokens_by_object with updated poses.
    """
    if object_gaussians is None:
        raise RuntimeError(
            "refine_poses_for_sequence: no render source is available "
            "(object_gaussians is None). "
            "Per-frame refinement renders decoded per-frame Gaussians, but none could be "
            "decoded."
        )

    # ── Deformation-field validation ──
    # The three deformation dicts must be supplied together (matches the
    # FINETUNE / Stage-2 rendering-guidance contract).  Asymmetric supply
    # would silently fall back to rigid rendering inside
    # ``_lookup_per_frame_deformation`` — raise here so misuse surfaces
    # at the entry point.
    _has_canon_mesh = canonical_mesh_verts_per_obj is not None
    _has_pf_verts = per_frame_mesh_verts_per_obj is not None
    _has_pf_R = per_frame_mesh_rotations_per_obj is not None
    if not (_has_canon_mesh == _has_pf_verts == _has_pf_R):
        raise ValueError(
            "refine_poses_for_sequence: canonical_mesh_verts_per_obj, "
            "per_frame_mesh_verts_per_obj and per_frame_mesh_rotations_per_obj "
            "must be supplied together "
            f"(got canonical_mesh_verts_per_obj={'set' if _has_canon_mesh else 'None'}, "
            f"per_frame_mesh_verts_per_obj={'set' if _has_pf_verts else 'None'}, "
            f"per_frame_mesh_rotations_per_obj={'set' if _has_pf_R else 'None'})."
        )
    # The global-scale, composite AND shared-world paths consume the deformation field.
    # The per-frame path does not, so reject silently-dropped supply there.
    _deformation_supplied = _has_canon_mesh and bool(canonical_mesh_verts_per_obj)
    if _deformation_supplied:
        if refine_scale != "global" and not pipeline.mv_shared_world_pose:
            raise NotImplementedError(
                "refine_poses_for_sequence: per-canonical-mesh-vertex "
                "deformation field is consumed by refine_scale='global' and by the "
                f"shared-world path, but not by the per-frame one (got "
                f"refine_scale={refine_scale!r}, mv_shared_world_pose=False).  The "
                "per-frame path uses per-frame canonical Gaussians directly and does "
                "not warp."
            )

    # MV shared-world-pose: collapse per-camera params to one shared pose per
    # object; aggregated per-frame loss drives a single optimizer. Short-circuits
    # before the per-frame / global dispatch since the shared path supersedes both.
    # (is_mv is asserted at the sequence-construction call site; not redundantly
    # checked here.)
    if pipeline.mv_shared_world_pose:
        # NOTE: ``or {}`` would drop an empty-but-present dict (falsy),
        # breaking the write-back's mutation back into state. Use ``is None``.
        pf_raw = perframe_raw_modalities if perframe_raw_modalities is not None else {}
        return refine_poses_shared_world_for_sequence(
            object_gaussians, tokens_by_object, sequence, losses, pipeline,
            save_renders=save_renders,
            perframe_raw_modalities=pf_raw,
            canon_frame_per_object=canon_frame_per_object,
            per_frame_canonical=per_frame_canonical,
            output_dir=output_dir,
            canonical_mesh_verts_per_obj=canonical_mesh_verts_per_obj,
            per_frame_mesh_verts_per_obj=per_frame_mesh_verts_per_obj,
            per_frame_mesh_rotations_per_obj=per_frame_mesh_rotations_per_obj,
            canonical_mesh_faces_per_obj=canonical_mesh_faces_per_obj,
            warp_knn_k=warp_knn_k,
            warp_knn_eps=warp_knn_eps,
            warp_knn_chunk_size=warp_knn_chunk_size,
        )

    # Global scale: all objects rendered jointly (composite)
    if refine_scale == "global":
        print("\n  Refining poses with COMPOSITE rendering (all objects jointly)...")
        return refine_poses_global_composite(
            object_gaussians, tokens_by_object, sequence, losses, pipeline,
            save_renders=save_renders,
            perframe_raw_modalities=perframe_raw_modalities,
            output_dir=output_dir,
            canonical_mesh_verts_per_obj=canonical_mesh_verts_per_obj,
            per_frame_mesh_verts_per_obj=per_frame_mesh_verts_per_obj,
            per_frame_mesh_rotations_per_obj=per_frame_mesh_rotations_per_obj,
            canonical_mesh_faces_per_obj=canonical_mesh_faces_per_obj,
            warp_knn_k=warp_knn_k,
            warp_knn_eps=warp_knn_eps,
            warp_knn_chunk_size=warp_knn_chunk_size,
        )

    # Per-frame refinement
    print("\n  Refining per-frame poses with differentiable rendering...")

    refined_tokens = {}
    total_frames = sum(len(tl) for tl in tokens_by_object.values())
    n_objects = len(tokens_by_object)
    pbar = tqdm(
        total=total_frames,
        desc="  Per-frame refine",
        leave=True,
        ncols=120,
        bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} frames [{elapsed}<{remaining}]",
    )

    # One adapter per distinct render source, not per frame: a single canonical
    # Gaussian is shared by every frame.  Keying on `id` is safe here because every
    # source belongs to a caller-owned dict that outlives the loop, so none can be freed
    # and have its id reused -- unlike `_block_icp_frames`, which splats its own.
    _renderers: Dict[int, PosedObjectRenderer] = {}

    def _renderer_for(source, device):
        if id(source) not in _renderers:
            _renderers[id(source)] = make_posed_object_renderer(source, device)
        return _renderers[id(source)]

    for obj_idx in sorted(tokens_by_object.keys()):
        refined_tokens[obj_idx] = []

        # An object with no Gaussians passes its tokens through unrefined; one with
        # Gaussians for most frames SKIPS the frames that failed to decode.
        # ``obj_gaussians`` is a {frame: Gaussian} dict under `per_frame_canonical` and a
        # single Gaussian otherwise; test emptiness only on the dict, never with a bare
        # truth test on a Gaussian.
        obj_gaussians = (object_gaussians or {}).get(obj_idx)
        has_gaussians = obj_gaussians is not None and (
            bool(obj_gaussians) if isinstance(obj_gaussians, dict) else True)
        if not has_gaussians:
            refined_tokens[obj_idx].extend(tokens_by_object[obj_idx])
            pbar.update(len(tokens_by_object[obj_idx]))
            continue

        for frame_idx, decoder_input in tokens_by_object[obj_idx]:
            desc = (f"  obj {obj_idx} f{frame_idx}" if n_objects > 1
                    else f"  frame {frame_idx}")
            pbar.set_description(desc)

            # Resolve the render source for this (obj, frame)
            if per_frame_canonical:
                if frame_idx not in obj_gaussians:
                    refined_tokens[obj_idx].append((frame_idx, decoder_input))
                    pbar.update(1)
                    continue
                canonical_gs = obj_gaussians[frame_idx]
            else:
                canonical_gs = obj_gaussians

            frame = sequence[frame_idx]
            render_image, render_masks, K_matrix = frame.image, frame.masks, frame.K_matrix
            mask = render_masks[obj_idx]
            if not mask.any():
                refined_tokens[obj_idx].append((frame_idx, decoder_input))
                pbar.update(1)
                continue

            depth_map_z = frame.depth_map_z

            # Ground truth image follows this frame's pose tensors.
            device = decoder_input["rotation"].device
            gt_image = torch.from_numpy(render_image).float().to(device) / 255.0

            # Initial pose
            initial_rotation = decoder_input["rotation"]
            initial_translation = decoder_input["translation"]
            initial_scale = decoder_input["scale"]

            # Extract raw modalities and scene context for pose token optimization
            raw_modalities = decoder_input.get("raw_ss_modalities")
            di_scene_scale = decoder_input.get("pointmap_scale")
            di_scene_shift = decoder_input.get("pointmap_shift")
            di_downsample_factor = decoder_input.get("downsample_factor", 1.0)

            # Refine pose
            refined_pose = refine_pose_for_frame(
                canonical_gs,
                initial_rotation,
                initial_translation,
                initial_scale,
                gt_image,
                mask,
                K_matrix,
                losses=losses,
                pipeline=pipeline,
                refine_scale=refine_scale,
                gt_depth=depth_map_z,
                valid_mask=frame.valid_mask,
                raw_modalities=raw_modalities,
                scene_scale=di_scene_scale,
                scene_shift=di_scene_shift,
                downsample_factor=di_downsample_factor,
                optimize_pose_tokens=pipeline.optimize_pose_tokens,
                output_dir=output_dir,
                save_renders=save_renders,
                obj_idx=obj_idx,
                frame_idx=frame_idx,
                renderer=_renderer_for(canonical_gs, initial_rotation.device),
            )

            # Create refined decoder input
            refined_decoder_input = {
                "rotation": refined_pose["rotation"],
                "translation": refined_pose["translation"],
                "scale": refined_pose["scale"],
                "refinement_loss_history": refined_pose["loss_history"],
                "refinement_best_iteration": refined_pose["best_iteration"],
            }
            # Preserve raw modalities (updated if pose token optimization was used)
            if "raw_modalities" in refined_pose:
                # Merge updated pose keys back into original dict (keeps shape, translation_scale)
                original_raw = decoder_input.get("raw_ss_modalities", {})
                merged = dict(original_raw)
                merged.update(refined_pose["raw_modalities"])
                refined_decoder_input["raw_ss_modalities"] = merged
            elif "raw_ss_modalities" in decoder_input:
                refined_decoder_input["raw_ss_modalities"] = decoder_input["raw_ss_modalities"]
            # Carry forward scene context + slat
            for _cfk in ("pointmap_scale", "pointmap_shift",
                         "downsample_factor", "decoder_input_slat"):
                if _cfk in decoder_input:
                    refined_decoder_input[_cfk] = decoder_input[_cfk]

            refined_tokens[obj_idx].append((frame_idx, refined_decoder_input))
            pbar.update(1)

    pbar.close()
    return refined_tokens
