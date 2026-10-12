# Copyright (c) Meta Platforms, Inc. and affiliates.

"""Coordinate per-frame and canonical fine-tuning across objects."""

from __future__ import annotations

import os
from typing import Any, Dict, List, Tuple

import torch
from genia.core.config import FinetuningConfig
from genia.core.finetune.decoder import _ft_lora_targets, _ft_rs_scaling, _ft_use_dora
from genia.core.finetune.object_optimizer import _finetune_single_object
from genia.core.utils.config import LossConfig, PipelineConfig


def finetune_perframe_tokens(
    tokens_by_object: Dict[int, List[Tuple[int, Dict]]],
    decoder: torch.nn.Module,
    sequence: Any,
    losses: "LossConfig",
    pipeline: "PipelineConfig",
    ft_config: "FinetuningConfig",
    device: torch.device,
    output_dir: str,
    scene_name: str,
    save_renders: bool = True,
    save_metrics: bool = True,
) -> Tuple[Dict[int, List[Tuple[int, Dict]]], Dict[int, Dict]]:
    """Fine-tune each frame's OWN SLAT against that frame's GT.

    The per-frame counterpart of :func:`finetune_canonical_tokens`, for runs
    that never built a canonical object — a dynamic sequence reconstructed from
    SAM3D's per-frame shapes (``PipelineState.has_canonical`` is False, so the
    canonical entry point has nothing to optimize).

    Each ``(object, frame)`` is a **one-frame instance of the same optimizer**:
    ``_finetune_single_object`` receives that frame's SLAT plus a single-element
    ``frame_indices``, so it supervises exactly that frame's image / mask /
    depth and refines that frame's pose.  Nothing about the loss, LoRA, or
    visibility machinery is duplicated here.

    No deformation warp is threaded through — a per-frame SLAT already **is**
    the deformed geometry at its own timestamp, so the ``canonical_mesh_*``
    kwargs the canonical path uses have no meaning.

    ``num_iterations`` is spent **per frame**, so wall-clock scales with the
    frame count.

    Returns
    -------
    tuple
        ``(tokens_by_object, perframe_gaussians)`` — each entry's finetuned
        SLAT rides along in its ``decoder_input_slat``, and the Gaussians
        (``{obj_idx: {frame_idx: Gaussian}}``) are decoded with the per-frame
        LoRA decoder when one was used.  The LoRA weights themselves are
        **not** returned — only ``set_canonical_slat`` consumes
        ``PipelineState.lora_state_dicts``, and there is no canonical here, so
        the decoded Gaussians are what carries the adaptation.
    """
    from genia.core.utils.frame_key import frame_key_stem
    from sam3d_objects.model.backbone.tdfy_dit.modules import sparse as sp

    if (
        ft_config.sh_degree is not None and ft_config.sh_degree > 0
        and not (ft_config.lora_decoder and ft_config.optimize_decoder)
    ):
        print("WARNING: sh_degree > 0 requires lora_decoder + optimize_decoder. Setting sh_degree=0 (DC only).")
        ft_config.sh_degree = 0

    new_perframe_gaussians: Dict[int, Dict] = {}
    new_tokens_by_object: Dict[int, List[Tuple[int, Dict]]] = dict(tokens_by_object)

    for obj_idx in sorted(tokens_by_object.keys()):
        updated_tokens: List[Tuple[int, Dict]] = []
        entries = tokens_by_object[obj_idx]
        n_slats = sum(1 for _, di in entries if di.get("decoder_input_slat") is not None)
        print(f"\n{'='*60}")
        print(f"Fine-tuning object {obj_idx} per-frame ({n_slats} frame(s))")
        print(f"{'='*60}")

        for frame_idx, di in entries:
            slat = di.get("decoder_input_slat")
            if slat is None:
                # No per-frame SLAT for this frame (never decoded, or purged) —
                # nothing to optimize; carry the entry through untouched.
                updated_tokens.append((frame_idx, di))
                continue

            poses = {
                frame_idx: {
                    "rotation": di["rotation"].clone().to(device),
                    "translation": di["translation"].clone().to(device),
                    "scale": di["scale"].clone().to(device),
                }
            }

            # Per-frame artifact subdir: every file _finetune_single_object
            # writes is named {scene}_obj{obj}_* , so sibling frames would
            # otherwise overwrite each other.
            frame_output_dir = os.path.join(
                output_dir, "perframe", frame_key_stem(frame_idx),
            )

            print(f"\n  --- object {obj_idx}, frame {frame_idx} ---")
            result = _finetune_single_object(
                canonical_feats=slat.feats.clone().to(device),
                canonical_coords=slat.coords.clone().to(device),
                frame_indices=[frame_idx],
                poses=poses,
                obj_idx=obj_idx,
                decoder=decoder,
                sequence=sequence,
                losses=losses,
                pipeline=pipeline,
                ft_config=ft_config,
                device=device,
                output_dir=frame_output_dir,
                scene_name=scene_name,
                save_renders=save_renders,
                save_metrics=save_metrics,
            )

            new_di = dict(di)
            new_di["decoder_input_slat"] = sp.SparseTensor(
                feats=result["feats"].to(device), coords=result["coords"].to(device),
            )
            if frame_idx in result["poses"]:
                bp = result["poses"][frame_idx]
                new_di["rotation"] = bp["rotation"]
                new_di["translation"] = bp["translation"]
                new_di["scale"] = bp["scale"]
            if result["dc_per_frame"] and frame_idx in result["dc_per_frame"]:
                new_di["dc_offset"] = result["dc_per_frame"][frame_idx]
            if result["sh_per_frame"] and frame_idx in result["sh_per_frame"]:
                new_di["sh_rest"] = result["sh_per_frame"][frame_idx]
            new_di["refinement_loss_history"] = result["loss_history"]
            new_di["refinement_batch_loss_history"] = result["loss_history"]
            new_di["refinement_best_iteration"] = result["best_iter"]
            updated_tokens.append((frame_idx, new_di))
            new_perframe_gaussians.setdefault(obj_idx, {})[frame_idx] = result["gaussian"]

        new_tokens_by_object[obj_idx] = updated_tokens

    return new_tokens_by_object, new_perframe_gaussians


def finetune_canonical_tokens(
    canonical_slats: Dict[int, Any],
    tokens_by_object: Dict[int, List[Tuple[int, Dict]]],
    decoder: torch.nn.Module,
    sequence: Any,
    losses: "LossConfig",
    pipeline: "PipelineConfig",
    ft_config: "FinetuningConfig",
    device: torch.device,
    output_dir: str,
    scene_name: str,
    save_renders: bool = True,
    save_metrics: bool = True,
    canon_frame_per_object: "Dict[int, Any] | None" = None,
    *,
    # ── actionmesh: per-obj per-canonical-mesh-vertex deformation field ──
    # When supplied, the per-frame render path warps decoded Gaussians via
    # ``warp_gaussians_high_res`` BEFORE applying Stage-1 rigid pose, so
    # FINETUNE supervises against deformed-frame GT images.  ``None``
    # falls back to the rigid-Sim(3)-only path.  Identical contract
    # to ``stage2_mv``'s ``canonical_mesh_verts`` kwargs.
    canonical_mesh_verts_per_obj: "Dict[int, torch.Tensor] | None" = None,
    per_frame_mesh_verts_per_obj: "Dict[int, Dict[int, torch.Tensor]] | None" = None,
    per_frame_mesh_rotations_per_obj: "Dict[int, Dict[int, torch.Tensor]] | None" = None,
    canonical_mesh_faces_per_obj: "Dict[int, torch.Tensor] | None" = None,
    warp_knn_k: int = 4,
    warp_knn_eps: float = 1.0e-8,
    warp_knn_chunk_size: int = 131072,
) -> Tuple[Dict[int, Any], Dict[int, List[Tuple[int, Dict]]], Dict[int, Any],
           Dict[int, dict]]:
    """Fine-tune canonical tokens for all objects.

    Parameters
    ----------
    canonical_slats : dict
        ``{obj_idx: SparseTensor}`` — canonical SLAT tokens per object.
    tokens_by_object : dict
        ``{obj_idx: [(frame_idx, decoder_input_dict), ...]}`` with poses.
    decoder : nn.Module
        Frozen Gaussian decoder (LoRA-wrapped once and shared across objects,
        each with its own adapter, if LoRA is used).
    sequence : Sequence
        Loaded scene data (GT images, masks, depth).
    losses : LossConfig
        Phase-specific loss weights and learning rates.
    pipeline : PipelineConfig
        Cross-phase pipeline control flags.
    ft_config : FinetuningConfig
        Token fine-tuning specific parameters.
    device : torch.device
    output_dir : str
        Directory for output files.
    scene_name : str
    save_renders : bool

    Returns
    -------
    tuple
        ``(canonical_slats, tokens_by_object, canonical_gaussians, lora_info)``
        — updated.  ``canonical_gaussians`` are decoded with the LoRA
        decoder (when used).  ``lora_info`` is ``{obj_idx: {"lora_rank",
        "lora_alpha", "use_dora", "rs_scaling", "lora_targets",
        "state_dict"}}`` for cache persistence.
    """
    from sam3d_objects.model.backbone.tdfy_dit.modules import sparse as sp

    if (
        ft_config.sh_degree is not None and ft_config.sh_degree > 0
        and not (ft_config.lora_decoder and ft_config.optimize_decoder)
    ):
        print("WARNING: sh_degree > 0 requires lora_decoder + optimize_decoder. Setting sh_degree=0 (DC only).")
        ft_config.sh_degree = 0

    new_canonical_gaussians: Dict[int, Any] = {}
    new_canonical_slats: Dict[int, Any] = {}
    new_tokens_by_object: Dict[int, List[Tuple[int, Dict]]] = dict(tokens_by_object)
    lora_info: Dict[int, dict] = {}

    for obj_idx in sorted(canonical_slats.keys()):
        slat = canonical_slats[obj_idx]
        feats = slat.feats.clone().to(device)
        coords = slat.coords.clone().to(device)

        # Extract frame indices and poses from tokens_by_object
        frame_indices = []
        poses: Dict[int, Dict[str, torch.Tensor]] = {}
        for frame_idx, di in tokens_by_object[obj_idx]:
            frame_indices.append(frame_idx)
            poses[frame_idx] = {
                "rotation": di["rotation"].clone().to(device),
                "translation": di["translation"].clone().to(device),
                "scale": di["scale"].clone().to(device),
            }

        print(f"\n{'='*60}")
        print(f"Fine-tuning object {obj_idx} ({len(frame_indices)} frames)")
        print(f"{'='*60}")

        # Per-obj deformation lookup (None ⇒ rigid-only for this obj).
        _cm_obj = (
            canonical_mesh_verts_per_obj.get(obj_idx)
            if canonical_mesh_verts_per_obj is not None else None
        )
        _pfv_obj = (
            per_frame_mesh_verts_per_obj.get(obj_idx)
            if per_frame_mesh_verts_per_obj is not None else None
        )
        _pfR_obj = (
            per_frame_mesh_rotations_per_obj.get(obj_idx)
            if per_frame_mesh_rotations_per_obj is not None else None
        )
        _faces_obj = (
            canonical_mesh_faces_per_obj.get(obj_idx)
            if canonical_mesh_faces_per_obj is not None else None
        )
        # The object's shared-world reference, resolved the ONE way the rebase resolves
        # it (`canon_frame_per_object`, else the lowest FrameKey).  Two rules for one
        # reference silently produce a wrong rebase.
        #
        # Gated on the flag rather than resolved unconditionally: `resolve_reference_frame`
        # sorts with `frame_key_sort_key`, which needs real FrameKeys, and a per-frame
        # caller may legitimately carry bare ints.  MV data always has
        # FrameKeys, so the resolution is only reachable where it is well defined.
        _sw_ref = None
        if bool(getattr(pipeline, "mv_shared_world_pose", False)):
            from genia.core.utils.pipeline_state import resolve_reference_frame
            _sw_ref = resolve_reference_frame(
                tokens_by_object[obj_idx], canon_frame_per_object or {}, obj_idx)

        result = _finetune_single_object(
            canonical_feats=feats,
            canonical_coords=coords,
            frame_indices=frame_indices,
            poses=poses,
            obj_idx=obj_idx,
            shared_world_ref_key=_sw_ref,
            decoder=decoder,
            sequence=sequence,
            losses=losses,
            pipeline=pipeline,
            ft_config=ft_config,
            device=device,
            output_dir=output_dir,
            scene_name=scene_name,
            save_renders=save_renders,
            save_metrics=save_metrics,
            canonical_mesh_verts=_cm_obj,
            per_frame_mesh_verts=_pfv_obj,
            per_frame_mesh_rotations=_pfR_obj,
            canonical_mesh_faces=_faces_obj,
            warp_knn_k=warp_knn_k,
            warp_knn_eps=warp_knn_eps,
            warp_knn_chunk_size=warp_knn_chunk_size,
        )

        new_canonical_gaussians[obj_idx] = result["gaussian"]
        new_canonical_slats[obj_idx] = sp.SparseTensor(
            feats=result["feats"].to(device), coords=result["coords"].to(device),
        )

        # LoRA state dict of the per-object Gaussian-decoder adapter.
        gs_adapter_i = result.get("gs_adapter")
        if gs_adapter_i is not None:
            lora_info[obj_idx] = {
                "lora_rank": ft_config.lora_rank,
                "lora_alpha": ft_config.lora_alpha,
                "use_dora": _ft_use_dora(ft_config),
                "rs_scaling": _ft_rs_scaling(ft_config),
                "lora_targets": _ft_lora_targets(ft_config),
                "decoder_full_precision": all(
                    p.dtype == torch.float32 for p in result["decoder"].parameters()
                ),
                "state_dict": gs_adapter_i.state_dict(),
            }

        updated_tokens = []
        for frame_idx, di in tokens_by_object[obj_idx]:
            new_di = dict(di)
            if frame_idx in result["poses"]:
                bp = result["poses"][frame_idx]
                new_di["rotation"] = bp["rotation"]
                new_di["translation"] = bp["translation"]
                new_di["scale"] = bp["scale"]
            if result["dc_per_frame"] and frame_idx in result["dc_per_frame"]:
                new_di["dc_offset"] = result["dc_per_frame"][frame_idx]
            if result["sh_per_frame"] and frame_idx in result["sh_per_frame"]:
                new_di["sh_rest"] = result["sh_per_frame"][frame_idx]
            new_di["refinement_loss_history"] = result["loss_history"]
            new_di["refinement_batch_loss_history"] = result["loss_history"]
            new_di["refinement_best_iteration"] = result["best_iter"]
            updated_tokens.append((frame_idx, new_di))
        new_tokens_by_object[obj_idx] = updated_tokens

    return (new_canonical_slats, new_tokens_by_object, new_canonical_gaussians,
            lora_info)
