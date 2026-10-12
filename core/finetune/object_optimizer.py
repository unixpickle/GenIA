# Copyright (c) Meta Platforms, Inc. and affiliates.

"""Optimize one object’s latent tokens, decoder adapters, and poses."""

from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Optional

import numpy as np
import torch
from genia.core.config import FinetuningConfig
from genia.core.finetune.anchors import (
    _apply_decoder_anchors,
    _detach_invisible_gaussians,
    _setup_opacity_anchor,
)
from genia.core.finetune.decoder import (
    _ft_decoder_checkpoint,
    _ft_decoder_checkpoint_min_voxels,
    _ft_lora_init,
    _ft_lora_targets,
    _ft_random_bg_seed,
    _ft_rs_scaling,
    _ft_use_dora,
    _prepare_finetuning_decoder,
    _set_decoder_checkpoint,
    setup_shared_lora_decoder,
)
from genia.core.finetune.losses import compute_total_loss
from genia.core.finetune.metrics import _compute_mean_psnr_ssim
from genia.core.finetune.poses import derive_shared_world_poses, snapshot_poses_with_correction
from genia.core.finetune.visibility import _setup_visibility_mask, _visibility_warp_bundle
from genia.core.utils.config import (
    LossConfig,
    PipelineConfig,
    resolve_correction_granularity,
    resolve_correction_scale_control,
)
from genia.core.utils.frame_sampling import EpochFrameSampler
from genia.core.utils.model_cache import ModelCache
from genia.core.utils.slat_decode import decode_tokens
from genia.core.utils.timing import get_timer
from tqdm import tqdm


def _finetune_single_object(
    canonical_feats: torch.Tensor,
    canonical_coords: torch.Tensor,
    frame_indices: List[int],
    poses: Dict[int, Dict[str, torch.Tensor]],
    obj_idx: int,
    decoder: torch.nn.Module,
    sequence: Any,
    losses: LossConfig,
    pipeline: PipelineConfig,
    ft_config: FinetuningConfig,
    device: torch.device,
    output_dir: str,
    scene_name: str,
    save_renders: bool = True,
    save_metrics: bool = True,
    shared_world_ref_key: Any = None,
    *,
    canonical_mesh_verts: "torch.Tensor | None" = None,
    per_frame_mesh_verts: "Dict[int, torch.Tensor] | None" = None,
    per_frame_mesh_rotations: "Dict[int, torch.Tensor] | None" = None,
    canonical_mesh_faces: "torch.Tensor | None" = None,
    warp_knn_k: int = 4,
    warp_knn_eps: float = 1.0e-8,
    warp_knn_chunk_size: int = 131072,
) -> Dict[str, Any]:
    """Run token fine-tuning optimization for a single object.

    Returns a dict with keys: ``gaussian``, ``feats``, ``coords``, ``poses``,
    ``loss_history``, ``best_iter``, ``decoder``,
    ``gs_adapter``.
    """
    from genia.core.utils.lora import LoRAAdapter

    _autocast_bf16 = _prepare_finetuning_decoder(decoder, ft_config, device)
    print(f"  Decoder precision (obj {obj_idx}): body={decoder.dtype}, "
          f"autocast_bf16={_autocast_bf16}")

    # ------------------------------------------------------------------
    # 1. Optionally LoRA-adapt the decoder via a single shared copy +
    #    per-object adapter.
    # ------------------------------------------------------------------
    lora_params: List[torch.nn.Parameter] = []
    obj_decoder = decoder            # default: shared frozen Gaussian decoder
    gs_adapter: Optional[LoRAAdapter] = None
    if ft_config.lora_decoder and ft_config.optimize_decoder:
        obj_decoder = setup_shared_lora_decoder(
            decoder,
            ft_config.lora_rank,
            ft_config.lora_alpha,
            init=_ft_lora_init(ft_config),
            use_dora=_ft_use_dora(ft_config),
            rs_scaling=_ft_rs_scaling(ft_config),
            lora_targets=_ft_lora_targets(ft_config),
        )
        gs_adapter = LoRAAdapter(
            obj_decoder,
            ft_config.lora_rank,
            ft_config.lora_alpha,
            init=_ft_lora_init(ft_config),
            use_dora=_ft_use_dora(ft_config),
            rs_scaling=_ft_rs_scaling(ft_config),
        )
        gs_adapter.activate()
        lora_params = gs_adapter.parameters
        lora_trainable = sum(p.numel() for p in lora_params)
        total_params = sum(p.numel() for p in obj_decoder.parameters())
        method = "DoRA" if _ft_use_dora(ft_config) else "LoRA"
        print(f"  {method} gauss-decoder (obj {obj_idx}): rank={ft_config.lora_rank} "
              f"alpha={ft_config.lora_alpha} init={_ft_lora_init(ft_config)} — "
              f"{lora_trainable} params "
              f"({lora_trainable / total_params:.2%} of decoder)")

    # Gradient-checkpoint the decoder torso to free retained activations.
    # Metal defaults to every object; other devices and explicit overrides
    # can limit recomputation to objects above a voxel-count threshold.
    _n_voxels = canonical_coords.shape[0]
    _checkpoint = (_ft_decoder_checkpoint(ft_config)
                   and _n_voxels >= _ft_decoder_checkpoint_min_voxels(ft_config, device))
    # LoRA decoders are shared across objects: reset the flag as well as
    # enabling it, so a large first object does not force every later one
    # to pay for recomputation.
    _set_decoder_checkpoint(obj_decoder, _checkpoint)
    if _checkpoint:
        print(f"  Decoder checkpointing (obj {obj_idx}): ON ({_n_voxels} voxels)")

    # ------------------------------------------------------------------
    # 2. Decode initial tokens (for before/after comparison)
    # ------------------------------------------------------------------
    with torch.no_grad():
        initial_gs = decode_tokens(obj_decoder, canonical_feats.float(), canonical_coords)
    n_points = initial_gs.get_xyz.shape[0]
    print(f"  Initial Gaussian (obj {obj_idx}): {n_points} points")

    # Snapshot of iter-0 raw _rotation for the soft anchor regulariser.
    # Detached + cloned so it can't get rebound when the per-iter decode
    # mutates canonical_gs.
    rotation_anchor_init: Optional[torch.Tensor] = None
    if ft_config.gauss_rotation_anchor_weight > 0:
        rotation_anchor_init = initial_gs._rotation.detach().clone()

    # Invisible-Gaussian opacity anchor: snapshot iter-0 opacity + the set of
    # Gaussians not visible from any train view, both computed ONCE here (the
    # invisible set is a stable geometric property; recomputing per-iter would
    # also re-flag already-collapsed Gaussians).  Reused every iter to penalise
    # opacity drops below init on the back-side.  See FinetuningConfig.
    opacity_anchor_init: Optional[torch.Tensor] = None
    opacity_anchor_invisible: Optional[torch.Tensor] = None
    if ft_config.invisible_opacity_anchor_weight > 0:
        opacity_anchor_init, opacity_anchor_invisible = _setup_opacity_anchor(
            initial_gs, poses, frame_indices, sequence, obj_idx, device, ft_config,
        )

    # Visibility gradient mask (stable, computed once): per-Gaussian occluded set
    # (detached before each render — leak-free) + per-token frozen set
    # (opt_feats.grad zeroed each iter so fully-occluded voxels stay at the prior).
    vis_invisible_gauss: Optional[torch.Tensor] = None
    vis_frozen_tokens: Optional[torch.Tensor] = None
    if ft_config.token_grad_visibility_mask:
        vis_invisible_gauss, vis_frozen_tokens = _setup_visibility_mask(
            initial_gs, poses, frame_indices, sequence, obj_idx,
            canonical_coords.shape[0], device, ft_config,
            warp=_visibility_warp_bundle(
                canonical_mesh_verts, per_frame_mesh_verts, per_frame_mesh_rotations,
                canonical_mesh_faces, warp_knn_k, warp_knn_eps,
                warp_knn_chunk_size),
            coords=canonical_coords, output_dir=(output_dir if save_renders else None),
            scene_name=scene_name,
        )

    # Per-frame appearance parameters:
    #   sh_degree=None → disabled (no per-frame appearance)
    #   sh_degree=0    → DC offsets only (base color varies per frame)
    #   sh_degree>0    → DC offsets + SH rest (base color + view-dependent vary per frame)
    dc_offset_per_frame: Optional[Dict[int, torch.nn.Parameter]] = None
    sh_rest_per_frame: Optional[Dict[int, torch.nn.Parameter]] = None

    if ft_config.sh_degree is not None:
        # Per-frame DC offsets (always when sh_degree is not None)
        dc_offset_per_frame = {}
        for fi in frame_indices:
            if fi in poses:
                dc_offset_per_frame[fi] = torch.nn.Parameter(
                    torch.zeros(n_points, 1, 3, device=device, dtype=torch.float32)
                )
        total_dc_params = sum(p.numel() for p in dc_offset_per_frame.values())
        print(f"  Per-frame DC offsets: {len(dc_offset_per_frame)} frames, "
              f"{total_dc_params} params (lr={ft_config.sh_lr})")

        # Per-frame SH rest (only when sh_degree > 0)
        if ft_config.sh_degree > 0:
            extra_sh = (ft_config.sh_degree + 1) ** 2 - 1
            sh_rest_per_frame = {}
            for fi in frame_indices:
                if fi in poses:
                    sh_rest_per_frame[fi] = torch.nn.Parameter(
                        torch.zeros(n_points, extra_sh, 3, device=device, dtype=torch.float32)
                    )
            total_sh_params = sum(p.numel() for p in sh_rest_per_frame.values())
            print(f"  Per-frame SH degree {ft_config.sh_degree}: "
                  f"{len(sh_rest_per_frame)} frames, {total_sh_params} params "
                  f"(lr={ft_config.sh_lr_rest})")

    # Warm up perceptual models if needed (lazy-loaded via ModelCache)
    if losses.perceptual_weight > 0:
        print("  Loading LPIPS perceptual model...")
        _ = ModelCache.get().perceptual_model

    # ------------------------------------------------------------------
    # 3. Setup optimizable parameters
    # ------------------------------------------------------------------
    _token_lr = float(ft_config.token_lr)
    _optimize_tokens = bool(ft_config.optimize_tokens) and _token_lr > 0
    opt_feats = canonical_feats.clone().detach().float().requires_grad_(_optimize_tokens)

    original_feats = None
    if ft_config.token_drift_weight > 0:
        original_feats = canonical_feats.clone().detach().float()

    # Resolved HERE, before the shared-world gate below reads it, and reused by the
    # correction setup further down -- it is a pure config read with no dependencies.
    # With mv_shared_world_pose ON, the gate below already frees one pose per
    # (object, TIMESTAMP) and derives each view by a c2w rebase -- exactly what the
    # per-frame axis means.  With it OFF the natives are per-FrameKey, i.e. per VIEW, so
    # the resolver refuses per_frame/both there.
    _correction_granularity = resolve_correction_granularity(
        ft_config, "finetuning", is_mv=sequence.is_mv,
        mv_shared_world_pose=pipeline.mv_shared_world_pose)

    # ---- MV shared world pose ------------------------------------------------------
    from genia.core.state import timestamp_reference_keys
    from genia.core.utils.frame_key import as_frame_key, frame_key_sort_key, group_by_frame
    from genia.core.utils.pose_params import (
        build_correction,
        build_decoded_shared_world_pose_params,
        freeze_params,
        natives_to_freeze,
        precompute_c2w_rebases,
    )

    # ONE pose per (object, TIMESTAMP); every view of that timestamp is DERIVED from it
    # by a fixed c2w rebase, so the object has a single world placement per instant
    # instead of V independent camera-space ones.  Gated on the pipeline flag alone --
    # no block knob -- exactly as `refine_poses_for_sequence` dispatches, so
    # `pipeline=mv_ours` gets it and monocular runs (where the flag is forced
    # off) are untouched.  FINETUNE is the LAST block to touch pose, so without this it
    # undoes the invariant every upstream block maintained.
    _shared_world = bool(getattr(pipeline, "mv_shared_world_pose", False)) and \
        bool(ft_config.refine_poses)
    _sw_ts_ref: Dict[Any, Any] = {}      # timestamp -> reference FrameKey
    _sw_groups: Dict[Any, list] = {}     # timestamp -> that timestamp's keys, view-sorted
    _sw_rebase: Dict[Any, torch.Tensor] = {}   # frame key -> constant 4x4 M_i
    _sw_q: Dict[Any, torch.Tensor] = {}        # timestamp -> shared quaternion leaf
    _sw_t: Dict[Any, torch.Tensor] = {}        # timestamp -> shared translation leaf
    if _shared_world:
        if shared_world_ref_key is None or shared_world_ref_key not in poses:
            raise ValueError(
                "finetuning: mv_shared_world_pose is on but no usable reference frame "
                f"was supplied for object {obj_idx} (got {shared_world_ref_key!r}). "
                "The caller must resolve it with `resolve_reference_frame`.")
        for fk in poses:
            if sequence[fk].c2w is None:
                raise ValueError(
                    f"finetuning: mv_shared_world_pose needs every frame's c2w to derive "
                    f"the per-view poses; frame {fk} has none.")
        _sw_ts_ref = timestamp_reference_keys(list(poses.keys()), shared_world_ref_key)
        _sw_groups = group_by_frame({as_frame_key(fk): None for fk in poses})
        for _ts, _ref_fk in _sw_ts_ref.items():
            _sw_rebase.update(
                precompute_c2w_rebases(sequence, _sw_groups[_ts], _ref_fk, device))
        print(f"  MV shared world pose: {len(_sw_ts_ref)} timestamp(s) x "
              f"{len(poses)} frame(s), reference {shared_world_ref_key} "
              f"(per-timestamp refs {sorted(_sw_ts_ref.values(), key=frame_key_sort_key)})")

    # Save original poses for before/after comparison
    original_poses = {}
    rot_params, trans_params = [], []

    # Create a single global scale shared across all frames (shape: (1, 3) or (3,))
    # Use the first available frame's scale as the initial value.
    opt_global_scale: Optional[torch.Tensor] = None
    initial_global_scale: Optional[torch.Tensor] = None

    for fi in frame_indices:
        if fi not in poses:
            continue
        original_poses[fi] = {
            "rotation": poses[fi]["rotation"].clone().detach(),
            "translation": poses[fi]["translation"].clone().detach(),
            "scale": poses[fi]["scale"].clone().detach(),
        }
        if ft_config.refine_poses:
            _rot_lr = float(losses.lr_rotation)
            _trans_lr = float(losses.lr_translation)
            if not _shared_world:
                poses[fi]["rotation"] = poses[fi]["rotation"].clone().detach().float().requires_grad_(_rot_lr > 0)
                poses[fi]["translation"] = poses[fi]["translation"].clone().detach().float().requires_grad_(_trans_lr > 0)
                rot_params.append(poses[fi]["rotation"])
                trans_params.append(poses[fi]["translation"])
            # Initialize global scale from first frame.  Isotropic mode
            # holds a single scalar Parameter (Adam tracks one moment);
            # we expand to (3,) only when handing the value to the
            # renderer below.
            if opt_global_scale is None:
                scale_val = poses[fi]["scale"].clone().detach().float().view(-1)
                if scale_val.shape[0] == 1:
                    scale_val = scale_val.expand(3).clone()
                _scale_lr = float(losses.lr_scale)
                opt_global_scale = (
                    scale_val.mean().reshape(1)
                    .clone().to(device).requires_grad_(_scale_lr > 0)
                )
                initial_global_scale = opt_global_scale.clone().detach()

    # Shared-world leaves: one (q, t) per TIMESTAMP, cloned off that timestamp's
    # reference view.  `build_decoded_shared_world_pose_params` is the decoded-mode
    # builder (FINETUNE carries no raw tokens and no SSI, i.e. the
    # `optimize_pose_tokens=False` path).  Its scale leaf is deliberately DISCARDED:
    # FINETUNE keeps its own single `opt_global_scale` per object, which is stricter than
    # the refine block's per-timestamp scale and is what the rebase's `s_i = s_ref`
    # already assumes.  The param groups are rebuilt here rather than reused because
    # FINETUNE's carry `tag`/`name`, which drive the pose freeze and grad logging.
    if _shared_world:
        for _ts, _ref_fk in _sw_ts_ref.items():
            _q, _t, _s_unused, _ = build_decoded_shared_world_pose_params(
                poses[_ref_fk], _rot_lr, _trans_lr, 0.0, device)
            _sw_q[_ts], _sw_t[_ts] = _q, _t
        rot_params = list(_sw_q.values())
        trans_params = list(_sw_t.values())

    # Point all frames' scale to the shared global scale tensor.
    # Isotropic case expands the scalar to (3,) for the renderer; the
    # expanded view shares storage with the underlying Parameter so
    # gradients flow back to the (1,) Adam-tracked leaf.
    scale_for_render: Optional[torch.Tensor] = None
    if ft_config.refine_poses and opt_global_scale is not None:
        scale_for_render = opt_global_scale.expand(3)
        for fi in frame_indices:
            if fi in poses:
                # Store as (1, 3) to match expected shape in compute_total_loss
                poses[fi]["scale"] = scale_for_render.unsqueeze(0)

    # Build optimizer param groups — skip groups with lr=0 (their params
    # already have requires_grad=False, so no gradients are wasted).
    param_groups = []
    if _optimize_tokens:
        param_groups.append({"params": [opt_feats], "lr": _token_lr, "tag": "appearance", "name": "tokens"})
    if ft_config.refine_poses:
        if _rot_lr > 0 and rot_params:
            param_groups.append({"params": rot_params, "lr": _rot_lr, "tag": "pose", "name": "rotation"})
        if _trans_lr > 0 and trans_params:
            param_groups.append({"params": trans_params, "lr": _trans_lr, "tag": "pose", "name": "translation"})
        if _scale_lr > 0 and opt_global_scale is not None:
            param_groups.append({"params": [opt_global_scale], "lr": _scale_lr, "tag": "pose", "name": "scale"})
    if lora_params:
        _lora_lr = float(ft_config.lora_lr)
        if _lora_lr > 0:
            lora_group = {"params": lora_params, "lr": _lora_lr, "tag": "appearance", "name": "lora"}
            if ft_config.lora_weight_decay > 0:
                lora_group["weight_decay"] = float(ft_config.lora_weight_decay)
            param_groups.append(lora_group)
    if dc_offset_per_frame is not None:
        _dc_lr = float(ft_config.sh_lr)
        if _dc_lr > 0:
            param_groups.append({
                "params": list(dc_offset_per_frame.values()),
                "lr": _dc_lr, "tag": "appearance", "name": "dc_offset",
            })
    if sh_rest_per_frame is not None:
        _sh_lr = float(ft_config.sh_lr_rest)
        if _sh_lr > 0:
            param_groups.append({
                "params": list(sh_rest_per_frame.values()),
                "lr": _sh_lr, "tag": "appearance", "name": "sh_rest",
            })

    # ---- Shared correction: ONE Sim(3) per object, composed onto every frame ----
    # In the object's CANONICAL frame, so per-frame root MOTION survives.
    _root_delta = None
    if _correction_granularity != "per_frame":
        if not ft_config.refine_poses:
            raise ValueError(
                "finetuning: correction_granularity requires refine_poses=true — the "
                "correction IS a pose parameter, and pose optimisation is off.")
        _rd_aa, _rd_logs, _rd_dt, _rd_groups = build_correction(
            float(losses.lr_rotation), float(losses.lr_scale),
            float(losses.lr_translation), device,
            # `tag` drives the pose freeze; `name` is per component, which is why
            # this is a callable rather than one dict of extras.
            group_extra=lambda k: {"tag": "pose", "name": f"correction_granularity_{k}"})
        _root_delta = {"aa": _rd_aa, "log_ds": _rd_logs, "dt": _rd_dt}
        param_groups.extend(_rd_groups)
        _freeze = natives_to_freeze(
            _correction_granularity,
            resolve_correction_scale_control(ft_config, "finetuning"))
        if _freeze == "all":
            # Freeze the native per-frame pose params so the correction is the only
            # thing that moves.  `correction_granularity: both` skips this freeze and leaves
            # them live alongside the correction; the split between the two is then a
            # gauge choice rather than a result, which is harmless only because
            # `snapshot_poses_with_correction` composes them before anything reads them.
            _delta_ids = {id(v) for v in _root_delta.values()}
            _keep = [any(id(p) in _delta_ids for p in g["params"])
                     or g.get("tag") != "pose"
                     for g in param_groups]
            for g, keep in zip(param_groups, _keep):
                if not keep:
                    for prm in g["params"]:
                        prm.requires_grad_(False)
            param_groups = [g for g, keep in zip(param_groups, _keep) if keep]
        elif _freeze == "scale" and opt_global_scale is not None:
            # The object has ONE size, so the shared correction owns it and the native
            # scale is held.  Dropped from the groups rather than given lr=0, because
            # `losses.lr_scale` also drives the correction's `log_ds` above.
            param_groups = freeze_params(param_groups, [opt_global_scale])
        print(f"  Pose correction: {_correction_granularity} (one Sim(3) for obj {obj_idx})"
              + ("; native scale FROZEN" if _freeze == "scale" else ""))

    if not param_groups:
        # All LRs are zero ⇒ nothing to optimise.  Return the same dict
        # contract as the optimised path: caller (``finetune_canonical_tokens``)
        # consumes ``result["gaussian"]`` / ``result["feats"]`` / ... — the
        # optimised values are simply the inputs unchanged.
        print("  WARNING: all LRs are 0.0 — nothing to optimize, skipping finetuning")
        with torch.no_grad():
            unchanged_gs = decode_tokens(
                obj_decoder, canonical_feats.float().to(device), canonical_coords,
            )
        return {
            "gaussian": unchanged_gs,
            "initial_gaussian": initial_gs,
            "feats": canonical_feats,
            "coords": canonical_coords,
            "poses": poses,
            "original_poses": original_poses,
            "dc_per_frame": dc_offset_per_frame,
            "sh_per_frame": sh_rest_per_frame,
            "loss_history": [],
            "best_iter": 0,
            "decoder": obj_decoder,
            "gs_adapter": gs_adapter,
        }

    optimizer = torch.optim.AdamW(param_groups)

    # Batch size
    num_frames = len(frame_indices)
    batch_size = ft_config.batch_size if ft_config.batch_size > 0 else num_frames
    batch_size = min(batch_size, num_frames)
    effective_num_iterations = ft_config.num_iterations

    # Linear LR warmup (multiplier min(step / W, 1)) on every param group.
    scheduler = None
    warmup_steps = max(0, int(ft_config.lr_warmup_steps))
    if warmup_steps > 0:
        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer,
            lr_lambda=lambda step, _W=warmup_steps: min(step / _W, 1.0),
        )

    parts = []
    if _optimize_tokens:
        parts.append("token")
    if ft_config.lora_decoder and ft_config.optimize_decoder:
        parts.append("lora")
    if ft_config.sh_degree is not None:
        parts.append("dc_offset")
        if ft_config.sh_degree > 0:
            parts.append(f"sh{ft_config.sh_degree}")
    if ft_config.refine_poses:
        parts.append("pose")
    mode_str = " + ".join(parts) if parts else "no-op"

    if ft_config.refine_poses and opt_global_scale is not None:
        print(f"  Global scale (isotropic): {opt_global_scale.data.tolist()}")

    print(f"\n  Starting {mode_str} fine-tuning (obj {obj_idx})")
    print(f"    Iterations: {effective_num_iterations}  Batch: {batch_size}/{num_frames} frames")
    print(f"    Token LR: {ft_config.token_lr}  warmup={warmup_steps}")
    if lora_params:
        print(f"    LoRA LR: {ft_config.lora_lr}  rank={ft_config.lora_rank}"
              f"  weight_decay={ft_config.lora_weight_decay}")
    if dc_offset_per_frame is not None:
        print(f"    DC offset LR: {ft_config.sh_lr}")
    if sh_rest_per_frame is not None:
        print(f"    SH rest LR: {ft_config.sh_lr_rest}  degree={ft_config.sh_degree}")
    if ft_config.refine_poses:
        print(f"    Pose LR: rot={losses.lr_rotation}  trans={losses.lr_translation}"
              f"  scale={losses.lr_scale}")

    # ------------------------------------------------------------------
    # 4. Optimization loop
    # ------------------------------------------------------------------
    loss_history: List[Dict[str, float]] = []
    # No best-tracking: the final iter's state is what we save.  The
    # ``best_*`` variables below carry the iter-0 snapshot for the zero-iter
    # edge case; they are overwritten in-place with the final state after
    # the loop exits.  ``final_loss`` carries the last reported total loss.
    final_loss = float("nan")
    best_feats = opt_feats.clone().detach()
    # Derive once before the loop as well, so the ZERO-iteration path (and the iter-0
    # snapshot it falls back to) writes derived poses rather than the per-view input.  A
    # no-op in practice on input the upstream rebase already made consistent, and a
    # correction when it did not.
    if _shared_world:
        with torch.no_grad():
            derive_shared_world_poses(poses, _sw_ts_ref, _sw_groups, _sw_rebase,
                                      _sw_q, _sw_t, scale_for_render)
    best_poses = snapshot_poses_with_correction(poses, _root_delta)
    best_dc_per_frame: Optional[Dict[int, torch.Tensor]] = None
    if dc_offset_per_frame is not None:
        best_dc_per_frame = {fi: p.clone().detach() for fi, p in dc_offset_per_frame.items()}
    best_sh_per_frame: Optional[Dict[int, torch.Tensor]] = None
    if sh_rest_per_frame is not None:
        best_sh_per_frame = {fi: p.clone().detach() for fi, p in sh_rest_per_frame.items()}
    best_iter = 0
    frame_sampler = EpochFrameSampler(frame_indices)

    pbar = tqdm(
        range(effective_num_iterations),
        desc=f"  Finetune obj {obj_idx}",
        leave=True,
        disable=effective_num_iterations == 0,
    )
    for iteration in pbar:
        optimizer.zero_grad()

        # Every frame's pose re-derived from its timestamp's shared (q, t).  Inside the
        # loop by necessity -- see `derive_shared_world_poses`.  All frames, not just the
        # batch, and the cost is a few small matmuls.
        if _shared_world:
            derive_shared_world_poses(poses, _sw_ts_ref, _sw_groups, _sw_rebase,
                                      _sw_q, _sw_t, scale_for_render)

        canonical_gs = decode_tokens(obj_decoder, opt_feats, canonical_coords,
                                     autocast_bf16=_autocast_bf16)

        # Sample frame batch (epoch-based: all frames seen before any repeats)
        if batch_size >= len(frame_indices):
            batch_frames = frame_indices
        else:
            batch_frames = frame_sampler.sample(batch_size)

        # Stop-gradient the occluded Gaussians before the render (stable mask) —
        # closes the decoder leak + kills their own wrong gradient.
        if vis_invisible_gauss is not None:
            _detach_invisible_gaussians(canonical_gs, vis_invisible_gauss)

        # save_renders is the block's resolved flag: no debug_pixelwise/ PNGs (nor
        # the CPU copies feeding them) when the run asked for no renders.
        want_pixelwise = save_renders and iteration % 50 == 0
        total_loss, metrics, px_data = compute_total_loss(
            canonical_gs, poses, batch_frames, sequence, obj_idx, losses, device,
            perceptual_scale=ft_config.perceptual_scale,
            sh_rest_per_frame=sh_rest_per_frame,
            dc_offset_per_frame=dc_offset_per_frame,
            bg_color=torch.ones(3, device=device) if pipeline.white_background else None,
            return_pixelwise=want_pixelwise,
            microbatch_size=ft_config.microbatch_size,
            canonical_mesh_verts=canonical_mesh_verts,
            per_frame_mesh_verts=per_frame_mesh_verts,
            per_frame_mesh_rotations=per_frame_mesh_rotations,
            canonical_mesh_faces=canonical_mesh_faces,
            warp_knn_k=warp_knn_k,
            warp_knn_eps=warp_knn_eps,
            warp_knn_chunk_size=warp_knn_chunk_size,
            random_background=ft_config.random_background,
            random_background_seed=_ft_random_bg_seed(ft_config),
            random_background_iter=iteration,
            root_delta=_root_delta,
        )

        # Token drift regularization
        drift_loss_val = 0.0
        if ft_config.token_drift_weight > 0 and original_feats is not None:
            drift_loss = ft_config.token_drift_weight * torch.nn.functional.mse_loss(opt_feats, original_feats)
            total_loss = total_loss + drift_loss
            drift_loss_val = drift_loss.item()

        # DC offset regularization: L2 toward zero
        dc_reg_val = 0.0
        if dc_offset_per_frame is not None and ft_config.sh_reg_weight > 0:
            batch_dc = [dc_offset_per_frame[fi] for fi in batch_frames if fi in dc_offset_per_frame]
            if batch_dc:
                dc_reg = ft_config.sh_reg_weight * torch.stack(batch_dc).pow(2).mean()
                total_loss = total_loss + dc_reg
                dc_reg_val = dc_reg.item()

        # DC offset cross-frame consistency: penalize deviation from mean
        dc_consistency_val = 0.0
        if dc_offset_per_frame is not None and ft_config.sh_consistency_weight > 0:
            batch_dc = [dc_offset_per_frame[fi] for fi in batch_frames if fi in dc_offset_per_frame]
            if len(batch_dc) > 1:
                dc_stack = torch.stack(batch_dc)
                dc_mean = dc_stack.mean(dim=0)
                dc_cons = ft_config.sh_consistency_weight * (dc_stack - dc_mean.unsqueeze(0)).pow(2).mean()
                total_loss = total_loss + dc_cons
                dc_consistency_val = dc_cons.item()

        # SH regularization: L2 toward zero (per-frame, protects base color)
        sh_reg_val = 0.0
        if sh_rest_per_frame is not None and ft_config.sh_reg_weight > 0:
            batch_sh = [sh_rest_per_frame[fi] for fi in batch_frames if fi in sh_rest_per_frame]
            if batch_sh:
                sh_reg = ft_config.sh_reg_weight * torch.stack(batch_sh).pow(2).mean()
                total_loss = total_loss + sh_reg
                sh_reg_val = sh_reg.item()

        # SH cross-frame consistency: penalize deviation from mean
        sh_consistency_val = 0.0
        if sh_rest_per_frame is not None and ft_config.sh_consistency_weight > 0:
            batch_sh = [sh_rest_per_frame[fi] for fi in batch_frames if fi in sh_rest_per_frame]
            if len(batch_sh) > 1:
                sh_stack = torch.stack(batch_sh)
                sh_mean = sh_stack.mean(dim=0)
                consistency_loss = ft_config.sh_consistency_weight * (sh_stack - sh_mean.unsqueeze(0)).pow(2).mean()
                total_loss = total_loss + consistency_loss
                sh_consistency_val = consistency_loss.item()

        metrics["drift"] = drift_loss_val
        metrics["dc_reg"] = dc_reg_val
        metrics["dc_consistency"] = dc_consistency_val
        metrics["sh_reg"] = sh_reg_val
        metrics["sh_consistency"] = sh_consistency_val
        if vis_frozen_tokens is not None:
            metrics["vis_masked_frac"] = vis_frozen_tokens.float().mean().item()
        metrics["total"] = total_loss.item()
        final_loss = metrics["total"]

        # Backward: when grad_accum was used, rendering gradients are already
        # accumulated and total_loss is detached.  Adding reg losses re-enables
        # requires_grad only through the reg terms, so backward() propagates
        # only the regularization gradients.  If no reg losses were added,
        # total_loss stays detached and we skip backward entirely.
        if total_loss.requires_grad:
            total_loss.backward()

        # Freeze fully-occluded voxels: zero their token grad — keeps the back-side
        # at the prior, which the per-Gaussian detach alone cannot (shared token).
        if vis_frozen_tokens is not None and opt_feats.grad is not None:
            opt_feats.grad[vis_frozen_tokens] = 0.0

        # Opacity + rotation anchors — gradient to the decoder's LoRA only
        # (tokens detached), accumulated on top of the main loss's LoRA grad.
        _apply_decoder_anchors(
            obj_decoder, opt_feats, canonical_coords,
            opacity_anchor_init, opacity_anchor_invisible,
            ft_config.invisible_opacity_anchor_weight,
            rotation_anchor_init, ft_config.gauss_rotation_anchor_weight,
            _autocast_bf16, metrics,
        )

        # Per-group gradient norms
        grad_norms = {}
        for g in optimizer.param_groups:
            gname = g.get("name", "?")
            gnorm = 0.0
            for p in g["params"]:
                if p.grad is not None:
                    gnorm += p.grad.data.norm(2).item() ** 2
            grad_norms[gname] = gnorm ** 0.5

        optimizer.step()
        if scheduler is not None:
            scheduler.step()

        # Save per-pixel debug plot every 50 iterations
        if want_pixelwise and px_data:
            from genia.core.utils.visualization import save_pixelwise_loss_plot
            with get_timer().exclude():
                save_pixelwise_loss_plot(px_data, iteration, obj_idx, output_dir)

        # Track parameter drift (after step — measures actual drift from initial)
        with torch.no_grad():
            token_l2 = (opt_feats - canonical_feats.float().to(device)).norm().item()
            token_rel = token_l2 / max(canonical_feats.float().norm().item(), 1e-8)
            metrics["token_l2"] = token_l2
            metrics["token_rel"] = token_rel
            if ft_config.refine_poses:
                rot_deltas, trans_deltas = [], []
                for fi in frame_indices:
                    if fi not in poses or fi not in original_poses:
                        continue
                    rot_deltas.append((poses[fi]["rotation"] - original_poses[fi]["rotation"].to(device)).norm().item())
                    trans_deltas.append((poses[fi]["translation"] - original_poses[fi]["translation"].to(device)).norm().item())
                metrics["pose_rot_l2"] = sum(rot_deltas) / max(len(rot_deltas), 1)
                metrics["pose_trans_l2"] = sum(trans_deltas) / max(len(trans_deltas), 1)
                # Global scale delta (single shared scale vs first frame's original)
                if opt_global_scale is not None and initial_global_scale is not None:
                    metrics["pose_scale_l2"] = (opt_global_scale - initial_global_scale.to(device)).norm().item()
                else:
                    metrics["pose_scale_l2"] = 0.0

        # Store per-group gradient norms
        for gname, gnorm in grad_norms.items():
            metrics[f"grad_{gname}"] = gnorm

        loss_history.append(metrics)

        # Update tqdm postfix (match global refinement display)
        postfix = {
            "loss": f"{metrics['total']:.4f}",
        }
        if losses.rgb_weight > 0:
            postfix["rgb"] = f"{metrics['rgb']:.4f}"
        if losses.rgb_ssim_weight > 0:
            postfix["ssim"] = f"{metrics['ssim']:.4f}"
        if losses.silhouette_weight > 0:
            postfix["silh"] = f"{metrics['silhouette']:.4f}"
        if losses.perceptual_weight > 0:
            postfix["lpips"] = f"{metrics['perceptual']:.4f}"
        if losses.depth_weight > 0:
            postfix["depth"] = f"{metrics['depth']:.4f}"
        if drift_loss_val > 0:
            postfix["drift"] = f"{drift_loss_val:.4f}"
        if dc_reg_val > 0:
            postfix["dc_reg"] = f"{dc_reg_val:.4f}"
        if dc_consistency_val > 0:
            postfix["dc_cons"] = f"{dc_consistency_val:.4f}"
        if sh_reg_val > 0:
            postfix["sh_reg"] = f"{sh_reg_val:.4f}"
        if sh_consistency_val > 0:
            postfix["sh_cons"] = f"{sh_consistency_val:.4f}"
        if "vis_masked_frac" in metrics:
            postfix["vmask"] = f"{metrics['vis_masked_frac']:.2f}"
        if metrics.get("opacity_anchor", 0) > 0:
            postfix["op_anch"] = f"{metrics['opacity_anchor']:.4f}"
        for gname, gnorm in grad_norms.items():
            if gnorm > 0:
                postfix[f"∇{gname}"] = f"{gnorm:.1e}"
        pbar.set_postfix(postfix)

    # ------------------------------------------------------------------
    # 5. Snapshot FINAL-iter state + save outputs
    # ------------------------------------------------------------------
    # No best-tracking: take the final iter's state.  ``best_*``
    # variables keep their names downstream (history JSON, save paths,
    # PSNR/SSIM eval) but hold the FINAL-iter values.
    if effective_num_iterations > 0:
        best_feats = opt_feats.clone().detach()
        if ft_config.refine_poses:
            best_poses = snapshot_poses_with_correction(poses, _root_delta)
        if dc_offset_per_frame is not None:
            best_dc_per_frame = {fi: dc_offset_per_frame[fi].clone().detach() for fi in dc_offset_per_frame}
        if sh_rest_per_frame is not None:
            best_sh_per_frame = {fi: p.clone().detach() for fi, p in sh_rest_per_frame.items()}
        best_iter = effective_num_iterations - 1

    print(f"  Final loss: {final_loss:.4f} at iteration {best_iter+1}")

    # Report the fitted correction, not just that one was requested: "correction_granularity:
    # shared" alone cannot distinguish "ran and moved the object 3 degrees" from "ran and
    # did nothing".  Same three numbers `run_icp_refine_block` prints.
    if _root_delta is not None:
        with torch.no_grad():
            _aa = _root_delta["aa"].reshape(3)
            _ang = float(torch.rad2deg(_aa.norm()))
            print(f"  Pose correction fitted: ΔR {_ang:.2f}°, "
                  f"Δs {float(torch.exp(_root_delta['log_ds']).reshape(())):.4f}, "
                  f"|Δt| {float(_root_delta['dt'].reshape(3).norm()):.4f}")

    # Decode final Gaussian (pure decoded tokens, no per-frame appearance baked in)
    with torch.no_grad():
        best_gs = decode_tokens(obj_decoder, best_feats.float().to(device), canonical_coords)

    # Before/after PSNR + SSIM headline (mean over all frames, masked to obj).
    _eval_bg = torch.ones(3, device=device) if pipeline.white_background else None
    _psnr_before, _ssim_before = _compute_mean_psnr_ssim(
        initial_gs, original_poses, frame_indices, sequence, obj_idx, device,
        bg_color=_eval_bg,
        canonical_mesh_verts=canonical_mesh_verts,
        per_frame_mesh_verts=per_frame_mesh_verts,
        per_frame_mesh_rotations=per_frame_mesh_rotations,
        canonical_mesh_faces=canonical_mesh_faces,
        warp_knn_k=warp_knn_k, warp_knn_eps=warp_knn_eps,
        warp_knn_chunk_size=warp_knn_chunk_size,
    )
    _psnr_after, _ssim_after = _compute_mean_psnr_ssim(
        best_gs, best_poses, frame_indices, sequence, obj_idx, device,
        bg_color=_eval_bg,
        canonical_mesh_verts=canonical_mesh_verts,
        per_frame_mesh_verts=per_frame_mesh_verts,
        per_frame_mesh_rotations=per_frame_mesh_rotations,
        canonical_mesh_faces=canonical_mesh_faces,
        warp_knn_k=warp_knn_k, warp_knn_eps=warp_knn_eps,
        warp_knn_chunk_size=warp_knn_chunk_size,
    )
    if _psnr_before is not None and _psnr_after is not None:
        print(
            f"  FINETUNE obj {obj_idx}: PSNR {_psnr_before:.2f} → {_psnr_after:.2f} "
            f"(Δ {_psnr_after - _psnr_before:+.2f})  "
            f"SSIM {_ssim_before:.4f} → {_ssim_after:.4f} "
            f"(Δ {_ssim_after - _ssim_before:+.4f})"
        )

    from genia.core.visualization import (
        plot_finetune_losses,
        plot_gradient_norms,
        plot_parameter_drift,
        plot_sh_magnitudes,
        render_color_shift_debug,
        visualize_slat_voxels_before_after,
    )

    os.makedirs(output_dir, exist_ok=True)

    # The finetuned token, LoRA adapter and loss curve: a record of this pass that
    # nothing reads back (final/ holds the result), so gated like the block's metrics.
    if save_metrics:
        # Save finetuned token
        finetuned_path = os.path.join(output_dir, f"{scene_name}_obj{obj_idx}_finetuned_token.npz")
        save_dict = dict(
            slat_feats=best_feats.cpu().numpy(),
            slat_coords=canonical_coords.cpu().numpy(),
        )
        if best_dc_per_frame:
            save_dict["dc_frame_indices"] = np.array(sorted(best_dc_per_frame.keys()))
            for fi, dc_tensor in best_dc_per_frame.items():
                save_dict[f"dc_offset_frame_{fi}"] = dc_tensor.cpu().numpy()
        if best_sh_per_frame:
            save_dict["sh_degree"] = np.array(ft_config.sh_degree)
            save_dict["sh_frame_indices"] = np.array(sorted(best_sh_per_frame.keys()))
            for fi, sh_tensor in best_sh_per_frame.items():
                save_dict[f"sh_rest_frame_{fi}"] = sh_tensor.cpu().numpy()
        np.savez(finetuned_path, **save_dict)
        print(f"  Saved fine-tuned token: {finetuned_path}")

        # Save LoRA weights (read from the adapter directly — independent of
        # which adapter is currently activated on the shared decoder)
        if gs_adapter is not None:
            lora_path = os.path.join(output_dir, f"{scene_name}_obj{obj_idx}_lora_weights.pt")
            torch.save({
                "lora_rank": ft_config.lora_rank,
                "lora_alpha": ft_config.lora_alpha,
                "use_dora": _ft_use_dora(ft_config),
                "rs_scaling": _ft_rs_scaling(ft_config),
                "lora_targets": _ft_lora_targets(ft_config),
                "state_dict": gs_adapter.state_dict(),
            }, lora_path)
            print(f"  Saved LoRA weights: {lora_path}")

        # Save loss history
        history_path = os.path.join(output_dir, f"{scene_name}_obj{obj_idx}_loss_history.json")
        history_data = {
            "num_iterations": ft_config.num_iterations,
            "lr_tokens": ft_config.token_lr,
            "refine_poses": ft_config.refine_poses,
            "batch_size": batch_size,
            "best_iteration": best_iter + 1,
            "best_loss": final_loss,
            "history": loss_history,
        }
        if ft_config.lora_decoder and ft_config.optimize_decoder:
            history_data.update({
                "lora_decoder": True,
                "lora_rank": ft_config.lora_rank,
                "lora_alpha": ft_config.lora_alpha,
                "lora_lr": ft_config.lora_lr,
            })
        if ft_config.sh_degree is not None:
            history_data.update({"sh_degree": ft_config.sh_degree, "sh_lr": ft_config.sh_lr,
                                 "sh_reg_weight": ft_config.sh_reg_weight,
                                 "sh_consistency_weight": ft_config.sh_consistency_weight})
            if ft_config.sh_degree > 0:
                history_data["sh_lr_rest"] = ft_config.sh_lr_rest
        with open(history_path, "w") as f:
            json.dump(history_data, f, indent=2)
        print(f"  Saved loss history: {history_path}")

    # Plot loss curves and parameter drift.  Diagnostics, so gated on the block's
    # resolved save_renders — 4-6 PNGs PER OBJECT otherwise, in a run that asked for
    # none.
    with get_timer().exclude():
        if save_renders:
            plot_path = os.path.join(output_dir, f"{scene_name}_obj{obj_idx}_loss_plot.png")
            plot_finetune_losses(loss_history, best_iter, plot_path, config=losses)

            drift_plot_path = os.path.join(output_dir, f"{scene_name}_obj{obj_idx}_param_drift.png")
            plot_parameter_drift(loss_history, best_iter, drift_plot_path, has_poses=ft_config.refine_poses)

            grad_plot_path = os.path.join(output_dir, f"{scene_name}_obj{obj_idx}_grad_norms.png")
            plot_gradient_norms(loss_history, best_iter, grad_plot_path)

            # SH magnitude plot (per-frame appearance parameters at each degree)
            if best_dc_per_frame or best_sh_per_frame:
                sh_mag_path = os.path.join(output_dir, f"{scene_name}_obj{obj_idx}_sh_magnitudes.png")
                plot_sh_magnitudes(
                    dc_per_frame=best_dc_per_frame,
                    sh_per_frame=best_sh_per_frame,
                    sh_degree=ft_config.sh_degree if ft_config.sh_degree is not None else 0,
                    output_path=sh_mag_path,
                )

                # Color shift debug: amplified diff + offset-only renders
                shift_debug_path = os.path.join(output_dir, f"{scene_name}_obj{obj_idx}_color_shift_debug.png")
                render_color_shift_debug(
                    canonical_gs=best_gs,
                    dc_per_frame=best_dc_per_frame or {},
                    sh_per_frame=best_sh_per_frame,
                    poses=best_poses,
                    sequence=sequence,
                    obj_idx=obj_idx,
                    output_path=shift_debug_path,
                    white_background=pipeline.white_background,
                )

            # SLAT voxel before/after visualization (shared PCA)
            voxel_path = os.path.join(output_dir, f"{scene_name}_obj{obj_idx}_slat_before_after.png")
            visualize_slat_voxels_before_after(
                feats_before=canonical_feats,
                feats_after=best_feats,
                coords=canonical_coords,
                output_path=voxel_path,
                title=f"SLAT Tokens — Object {obj_idx} (before vs after fine-tuning)",
            )

    # Feature change summary
    delta = (best_feats.cpu() - canonical_feats.float().cpu()).norm().item()
    rel_delta = delta / max(canonical_feats.float().cpu().norm().item(), 1e-8)
    print(f"  Feature L2 change: {delta:.4f} (relative: {rel_delta:.4%})")

    return {
        "gaussian": best_gs,
        "initial_gaussian": initial_gs,
        "feats": best_feats,
        "coords": canonical_coords,
        "poses": best_poses,
        "original_poses": original_poses,
        "dc_per_frame": best_dc_per_frame,
        "sh_per_frame": best_sh_per_frame,
        "loss_history": loss_history,
        "best_iter": best_iter,
        "decoder": obj_decoder,
        # The adapter carries the trained (A, B) Parameters; the caller reads
        # its state dict into ``PipelineState.lora_state_dicts``.  ``None``
        # when LoRA is off.
        "gs_adapter": gs_adapter,
    }
