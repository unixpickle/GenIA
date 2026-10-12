# Copyright (c) Meta Platforms, Inc. and affiliates.

"""Optimize shared world-space corrections across synchronized views."""

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
from genia.core.utils.pose_params import build_correction, freeze_params, natives_to_freeze
from genia.core.utils.pose_refinement.aggregation import _aggregate_frame_loss
from genia.core.utils.pose_refinement.chamfer import (
    _chamfer_gt_points_from_depth,
    _chamfer_gt_to_model,
    _chamfer_model_subsample_idx,
)
from genia.core.utils.pose_refinement.decoding import differentiable_pose_decode
from genia.core.utils.pose_refinement.frame_loss import (
    _compute_frame_loss,
    _prepare_frame_data_for_refinement,
)
from genia.core.utils.pose_refinement.loss_terms import (
    _compute_signed_distance_transform,
    _match_depth_grid,
    depth_to_normals,
)
from genia.core.utils.pose_refinement.rendering import _get_bg_color, make_posed_object_renderer
from genia.core.utils.pose_refinement.warping import _perframe_warp_overrides
from genia.core.utils.quaternion_ops import matrix_to_quaternion, quaternion_to_matrix
from genia.core.utils.timing import get_timer
from tqdm import tqdm


def _prepare_shared_world_frame_data(
    sequence: Any,
    obj_idx: int,
    frame_indices: List[int],
    losses: LossConfig,
    device: torch.device,
) -> Dict[int, Dict[str, Any]]:
    """Per-frame GT tensors needed by ``_compute_frame_loss``, keyed by frame idx.

    Skips frames where the object is absent (no mask coverage). Moves arrays
    onto ``device`` once to avoid per-iteration transfer cost.
    """
    frame_data: Dict[int, Dict[str, Any]] = {}
    need_depth = losses.depth_weight > 0 or losses.normals_weight > 0
    for fi in frame_indices:
        data = _prepare_frame_data_for_refinement(sequence, fi, obj_idx)
        if data is None:
            continue
        mask = torch.from_numpy(np.asarray(data["mask"])).bool().to(device)
        gt_image = torch.from_numpy(data["image"]).float().to(device) / 255.0
        sdt = None
        if losses.silhouette_weight > 0 and losses.silhouette_sdt_weight > 0:
            sdt = _compute_signed_distance_transform(mask)
        gt_depth_t = None
        valid_mask_t = None
        if data["gt_depth"] is not None and need_depth:
            gt_depth_t = torch.from_numpy(data["gt_depth"]).float().to(device)
            if data["valid_mask"] is not None:
                valid_mask_t = torch.from_numpy(data["valid_mask"]).bool().to(device)
        K_tensor = None
        if losses.normals_weight > 0:
            K_tensor = torch.from_numpy(data["K_matrix"]).float().to(device)
            if data["valid_mask"] is not None and valid_mask_t is None:
                valid_mask_t = torch.from_numpy(data["valid_mask"]).bool().to(device)
        # Optional Chamfer: object's visible GT surface points (unproject GT
        # depth → camera-space pointmap, keep mask ∩ valid ∩ finite), once.
        chamfer_gt_pts = None
        if getattr(losses, "chamfer_weight", 0.0) > 0 and data["gt_depth"] is not None:
            # sequence[fi].masks/.K_matrix (NOT data[...], which may be
            # render-resolution) -- data["gt_depth"] is always at the backbone's
            # (H', W'), and depth-unprojected points must match it exactly.
            chamfer_gt_pts = _chamfer_gt_points_from_depth(
                data["gt_depth"], sequence[fi].K_matrix,
                sequence[fi].masks[obj_idx], data["valid_mask"],
                int(getattr(losses, "chamfer_max_gt_points", 2048)), device,
                trim_factor=resolve_chamfer_gt_trim(losses, "global_pose_refine"))
        frame_data[fi] = {
            "gt_image": gt_image,
            "mask": mask,
            "sdt": sdt,
            "gt_depth": gt_depth_t,
            "valid_mask": valid_mask_t,
            "K_tensor": K_tensor,
            "K_matrix": data["K_matrix"],
            "H": data["H"],
            "W": data["W"],
            "chamfer_gt_pts": chamfer_gt_pts,
        }
    return frame_data


_SHARED_WORLD_BG_MAX_POINTS = 150000


def _build_fused_mv_background(
    sequence: Any, frame_data: Dict[int, Dict[str, Any]], obj_idx: int,
    device: torch.device,
) -> Dict[int, Tuple[torch.Tensor, ...]]:
    """Fuse every view's pointmap background (this object excluded) into one
    shared-world cloud, then re-project it into each view's camera.

    The shared-world path renders ONE object foreground-only, so (unlike the
    composite path) it has no background — full-image depth (depth_mask_only=
    False) needs one.  Building it from the union of all views fills the holes a
    single view can't see; all params are DETACHED (fixed depth reference, no
    pose gradient).  Capped at ``_SHARED_WORLD_BG_MAX_POINTS``.

    Returns ``{frame_idx: (xyz, rot, scale, opac, feat)}`` in each view's camera
    space, or ``{}`` when no view has background geometry (→ masked-depth
    fallback in the caller).
    """
    from genia.core.utils.gaussian import create_background_gaussians
    from genia.core.utils.rendering import transform_gaussian_params_world_to_cam
    w_xyz, w_rot, w_scale, w_opac, w_feat = [], [], [], [], []
    for fi in frame_data:
        frame = sequence[fi]
        bg_gs = create_background_gaussians(
            frame.image, frame.pointmap, [frame.masks[obj_idx]],
            frame.K_matrix, c2w=frame.c2w,
        )
        if bg_gs is None:
            continue
        w_xyz.append(bg_gs.get_xyz.detach().to(device))
        w_rot.append(bg_gs.get_rotation.detach().to(device))
        w_scale.append(bg_gs.get_scaling.detach().to(device))
        w_opac.append(bg_gs.get_opacity.squeeze(-1).detach().to(device))
        w_feat.append(bg_gs.get_features.detach().to(device))
    if not w_xyz:
        print(f"      obj {obj_idx}: no background geometry in any view "
              f"→ masked-depth fallback")
        return {}
    f_xyz, f_rot = torch.cat(w_xyz), torch.cat(w_rot)
    f_scale, f_opac, f_feat = torch.cat(w_scale), torch.cat(w_opac), torch.cat(w_feat)
    if f_xyz.shape[0] > _SHARED_WORLD_BG_MAX_POINTS:
        sel = torch.randperm(f_xyz.shape[0], device=device)[:_SHARED_WORLD_BG_MAX_POINTS]
        f_xyz, f_rot, f_scale, f_opac, f_feat = (
            f_xyz[sel], f_rot[sel], f_scale[sel], f_opac[sel], f_feat[sel])
    # Re-project the shared-world union into each view's camera once (fixed → no
    # per-iteration transform); the shared-world loop renders in camera space.
    bg_cam_per_frame: Dict[int, Tuple[torch.Tensor, ...]] = {}
    for fi in frame_data:
        c_xyz, c_rot = transform_gaussian_params_world_to_cam(
            f_xyz, f_rot, sequence[fi].c2w)
        bg_cam_per_frame[fi] = (c_xyz.detach(), c_rot.detach(), f_scale, f_opac, f_feat)
    print(f"      obj {obj_idx}: fused MV background {f_xyz.shape[0]} pts "
          f"→ full-image depth on {len(bg_cam_per_frame)} view(s)")
    return bg_cam_per_frame


def _apply_shared_correction(delta, R_ref, t_ref, s_ref):
    """Compose the shared Sim(3) onto a reference pose, in the object's CANONICAL frame.

    ``delta`` is the ``(aa, log_ds, dt)`` triple, or a triple of ``None`` when
    ``correction_granularity: per_frame`` -- in which case this is the identity and returns its
    inputs unchanged, so the caller needs no branch.

    Applied to the REFERENCE rather than to each derived frame because an object-frame
    correction COMMUTES with the c2w derivation: ``derive_perframe_pose_from_shared_decoded``
    is ``R_i = R_ref @ M33.T`` / ``t_i = M33 @ t_ref + M3``, and ``v @ A.T == A @ v`` for a
    1-D ``v``, so composing before or after the derive gives the same ``(R, t, s)``.  That
    is also why the correction preserves the shared-world invariant: every frame receives
    the identical object-frame transform.

    Row-convention throughout and NO transposes -- ``R_ref`` here is already the matrix the
    renderer right-multiplies by, which is what ``compose_root_delta`` documents.
    """
    aa, log_ds, dt = delta if delta is not None else (None, None, None)
    if aa is None:
        return R_ref, t_ref, s_ref
    # Imported here rather than at module scope: pose_refit reaches back into this module
    # (for the renderer and the rendered-chamfer), so a top-level import would make the
    # pair mutually importable and order-dependent.
    from genia.core.utils.pose_refit import compose_root_delta
    from pytorch3d.transforms import axis_angle_to_matrix

    R_new, s_new, t_new = compose_root_delta(
        axis_angle_to_matrix(aa), torch.exp(log_ds).reshape(()), dt,
        R_ref, s_ref.reshape(-1).mean(), t_ref.reshape(3),
    )
    return R_new, t_new, s_new.reshape(1).expand(3).contiguous()


def _apply_gs_correction(aa, log_ds, dt, q, t, s):
    """Quaternion-flavoured wrapper around :func:`_apply_shared_correction`.

    The composition lives in ONE place; this only adapts the shapes the per-frame solvers
    carry (wxyz quaternion + ``(1,3)``/``(3,)`` translation and scale) to the row-matrix
    form ``compose_root_delta`` takes, and back.  Identity when ``aa is None``
    (``correction_granularity: per_frame``), so callers need no branch.
    """
    if aa is None:
        return q, t, s
    R_new, t_new, s_new = _apply_shared_correction(
        (aa, log_ds, dt),
        quaternion_to_matrix(q.reshape(1, 4))[0], t.reshape(3), s.reshape(-1),
    )
    return (matrix_to_quaternion(R_new.contiguous().unsqueeze(0)).reshape(1, 4),
            t_new.reshape(1, 3), s_new.reshape(1, -1))


def prefer_present(primary, fallback):
    """First of `primary`, `fallback` that is present -- WITHOUT a truth test.

    `primary or fallback` forces `bool(primary)`, which RAISES on a multi-element
    tensor. The SSI params (`pointmap_scale` / `pointmap_shift`) are exactly that
    on an MV run once APPEARANCE_INIT has populated `perframe_raw_modalities`
    (one entry per view).

    An empty dict still falls through, preserving the `or` behaviour the
    `raw_ss_modalities` lookup relied on.
    """
    if primary is None:
        return fallback
    if isinstance(primary, dict) and not primary:
        return fallback
    return primary


def refine_poses_shared_world_for_sequence(
    canonical_gaussians: Dict[int, Any],
    tokens_by_object: Dict[int, List[Tuple[int, Dict[str, Any]]]],
    sequence: Any,
    losses: LossConfig,
    pipeline: PipelineConfig,
    perframe_raw_modalities: Dict[int, Dict[int, Dict[str, Any]]],
    canon_frame_per_object: Optional[Dict[int, int]] = None,
    per_frame_canonical: bool = False,
    output_dir: Optional[str] = None,
    save_renders: bool = True,
    canonical_mesh_verts_per_obj: Optional[Dict[int, torch.Tensor]] = None,
    per_frame_mesh_verts_per_obj: Optional[Dict[int, Dict[int, torch.Tensor]]] = None,
    per_frame_mesh_rotations_per_obj: Optional[Dict[int, Dict[int, torch.Tensor]]] = None,
    canonical_mesh_faces_per_obj: Optional[Dict[int, torch.Tensor]] = None,
    warp_knn_k: int = 4,
    warp_knn_eps: float = 1.0e-8,
    warp_knn_chunk_size: int = 8192,
) -> Dict[int, List[Tuple[int, Dict[str, Any]]]]:
    """MV shared-world-pose refinement: one shared pose per object, all frames
    contribute aggregated loss per iteration.

    Each object is optimized independently. Per iteration: resolve the shared
    reference cam_ref-space pose, derive every frame's cam_i-space pose via a
    fixed c2w rebase, render each frame, sum the per-frame losses, and backprop
    into the shared pose. At the end, best-loss state is propagated to all
    frames via :func:`rebase_object_poses_from_reference`.

    Two modes, by ``pipeline.optimize_pose_tokens``:

    - ``True`` (raw-token chain): the shared pose is the reference frame's raw
      Stage-1 tokens, decoded each step via SSI. Requires
      ``raw_ss_modalities`` + SSI params on the ref-frame decoder_input.
    - ``False`` (decoded): the shared pose is the reference frame's decoded
      ``rotation`` / ``translation`` / ``scale``, optimized directly (no raw
      tokens / SSI).

    An object is rendered from its canonical Gaussian through
    :func:`make_posed_object_renderer`.

    DEFORMING assets are supported.  When the per-canonical-mesh-vertex
    field is supplied, each frame's warped geometry is precomputed ONCE per (obj, frame)
    under ``no_grad`` — the canonical shape is fixed across the solve, so the warp is
    constant — and handed to the renderer per call as ``means_override`` /
    ``rotation_override`` (``_precompute_perframe_gaussian_warps``).  This is what lets a
    deforming object's ONE shared correction be fit from EVERY frame instead of from the
    canonical timestamp alone.

    ``losses.correction_granularity`` is honoured here as on every other solver:
    ``per_frame`` optimises the per-timestamp reference poses, ``shared`` freezes them and
    fits ONE Sim(3) per object in its canonical frame.  The correction is composed onto the
    REFERENCE pose -- it commutes with the c2w derivation, so that is identical to
    composing it on every derived frame, and it preserves the shared-world invariant by
    construction.

    Preconditions: ``pipeline.mv_shared_world_pose=True``. (``sequence.is_mv``
    is enforced at the call site; single-view data degenerates to an identity
    rebase.)
    """
    from genia.core.utils.pipeline_state import (
        rebase_object_poses_from_reference,
        resolve_reference_frame,
    )
    from genia.core.utils.pose_params import (
        build_decoded_shared_world_pose_params,
        build_shared_world_pose_params,
        derive_perframe_pose_from_shared_decoded,
        precompute_c2w_rebases,
    )

    if not pipeline.mv_shared_world_pose:
        raise RuntimeError(
            "refine_poses_shared_world_for_sequence called with "
            "mv_shared_world_pose=False; dispatch logic is inconsistent."
        )
    # Two modes: raw-token chain (optimize_pose_tokens=True; the reference frame
    # must carry raw_ss_modalities + SSI params) and decoded-pose (False; the
    # shared pose is the reference frame's decoded rotation/translation/scale,
    # no raw tokens / SSI needed).
    decoded_mode = not pipeline.optimize_pose_tokens

    canon_frame_per_object = canon_frame_per_object or {}
    device = get_device()
    bg_color = _get_bg_color(pipeline, device)

    all_objs = sorted(tokens_by_object.keys())

    if losses.num_iterations == 0:
        # No optimization: return tokens untouched.
        return {o: list(tokens_by_object[o]) for o in all_objs}

    # ── Per-object setup ──────────────────────────────────────────────
    obj_setup: Dict[int, Dict[str, Any]] = {}
    for obj_idx in all_objs:
        entries = tokens_by_object[obj_idx]
        if not entries:
            continue

        ref_frame = resolve_reference_frame(entries, canon_frame_per_object, obj_idx)

        # Source ref-frame raw tokens + SSI params. Priority: the snapshot in
        # perframe_raw_modalities; otherwise the live decoder_input on the ref
        # frame itself (populated by Stage-1 pose init).
        _pf_map = perframe_raw_modalities if perframe_raw_modalities is not None else {}
        ref_raw_info = _pf_map.get(obj_idx, {}).get(ref_frame)
        ref_di_tmp = next((di for fi, di in entries if fi == ref_frame), None)

        def _pull(source, key):
            return source.get(key) if source is not None else None

        ref_raw_mods = prefer_present(
            _pull(ref_raw_info, "raw_ss_modalities"),
            _pull(ref_di_tmp, "raw_ss_modalities"),
        )
        ref_ps = prefer_present(_pull(ref_raw_info, "pointmap_scale"),
                               _pull(ref_di_tmp, "pointmap_scale"))
        ref_psh = prefer_present(_pull(ref_raw_info, "pointmap_shift"),
                                _pull(ref_di_tmp, "pointmap_shift"))
        ref_dsf = (
            ref_raw_info.get("downsample_factor", 1.0) if ref_raw_info is not None
            else (ref_di_tmp.get("downsample_factor", 1.0) if ref_di_tmp is not None else 1.0)
        )

        # Decoded mode reads the always-present decoded pose off ref_di_tmp; only
        # the raw-token path needs raw_ss_modalities + SSI params validated.
        if not decoded_mode and (ref_raw_mods is None or ref_ps is None or ref_psh is None
                or "6drotation_normalized" not in ref_raw_mods
                or "translation" not in ref_raw_mods
                or "scale" not in ref_raw_mods):
            raise ValueError(
                f"obj {obj_idx}: reference frame {ref_frame} is missing raw_ss_modalities "
                f"(6drotation_normalized / translation / scale) or SSI params "
                f"(pointmap_scale / pointmap_shift). Ensure pose init "
                f"populates raw tokens on the ref-frame decoder_input, or set "
                f"pipeline.optimize_pose_tokens=false."
            )

        # Canonical source selection (shared mode: always use a single canonical,
        # the reference frame's when per_frame_canonical is set).
        if per_frame_canonical:
            per_fr = canonical_gaussians.get(obj_idx, {})
            canonical_gs = per_fr.get(ref_frame) if isinstance(per_fr, dict) else None
        else:
            canonical_gs = canonical_gaussians.get(obj_idx)
        canonical_src = canonical_gs
        if canonical_src is None:
            print(f"      Skipping obj {obj_idx}: no canonical Gaussian "
                  f"for ref frame {ref_frame}")
            continue
        renderer = make_posed_object_renderer(canonical_src, device)

        # Deformation warp, precomputed ONCE per frame: the canonical shape is fixed for
        # the whole solve, so the warp is constant -- the same reasoning (and the same
        # helpers) as the global-scale / composite paths.  Warping inside the render
        # would instead run a kNN per frame per iteration.
        frame_indices = [fi for fi, _ in entries]
        warp_overrides = _perframe_warp_overrides(
            canonical_src, obj_idx, frame_indices, device,
            canonical_mesh_verts_per_obj,
            per_frame_mesh_verts_per_obj,
            per_frame_mesh_rotations_per_obj,
            canonical_mesh_faces_per_obj,
            warp_knn_k=warp_knn_k, warp_knn_eps=warp_knn_eps,
            warp_knn_chunk_size=warp_knn_chunk_size,
        )
        if warp_overrides:
            print(f"      obj {obj_idx}: deformation warp cached for "
                  f"{len(warp_overrides)}/{len(frame_indices)} frame(s)")

        # ``opt_6d`` holds the 6D rotation token (raw mode) or the wxyz quaternion
        # (decoded mode); the loop / write-back branch on ``decoded_mode``.
        if decoded_mode:
            opt_6d, opt_t, opt_s, param_groups = build_decoded_shared_world_pose_params(
                ref_di_tmp,
                float(losses.lr_rotation), float(losses.lr_translation), float(losses.lr_scale),
                device,
            )
        else:
            opt_6d, opt_t, opt_s, param_groups = build_shared_world_pose_params(
                ref_raw_mods,
                float(losses.lr_rotation), float(losses.lr_translation), float(losses.lr_scale),
                device,
            )
        init_6d = opt_6d.clone().detach()
        init_t = opt_t.clone().detach()
        init_s = opt_s.clone().detach()

        # ---- Shared correction: ONE Sim(3) per object, in its CANONICAL frame ----------
        # Same knob, same three values and the same composer as every other solver.  It is
        # applied to the REFERENCE pose, once, rather than to each derived frame: an
        # object-frame correction COMMUTES with the c2w derivation
        # (`derive_perframe_pose_from_shared_decoded` is `R_i = R_ref @ M33.T`,
        # `t_i = M33 @ t_ref + M3`, and `v @ A.T == A @ v` for a 1-D v), so the two orders
        # are identical -- which is also why it preserves the shared-world invariant by
        # construction.  The scale is isotropic (the decoded builder means-collapses
        # it, the raw branch goes through `differentiable_pose_decode`).
        # Note: `opt_6d`/`opt_t`/`opt_s` are ONE pose per object for the whole
        # SEQUENCE (a single `ref_frame`), so on MV-DYNAMIC data `per_frame` here means
        # per-sequence, not per-timestamp; on MV-STATIC data the two coincide.  That is
        # the TEMPORAL axis; the view axis is handled by
        # `pose_refit._timestamp_delta_index`.
        # (The guard cannot fire here: mv_shared_world_pose=True is a precondition.)
        _pc_mode = resolve_correction_granularity(
            losses, "global_pose_refine", is_mv=sequence.is_mv,
            mv_shared_world_pose=pipeline.mv_shared_world_pose)
        delta_aa = delta_logs = delta_t = None
        if _pc_mode != "per_frame":
            delta_aa, delta_logs, delta_t, _delta_groups = build_correction(
                float(losses.lr_rotation), float(losses.lr_scale),
                float(losses.lr_translation), device)
            _freeze = natives_to_freeze(
                _pc_mode,
                resolve_correction_scale_control(losses, "global_pose_refine"))
            if _freeze == "all":
                # Freeze the natives so the correction is the only thing that moves --
                # what makes the fit well-posed and readable as "the systematic error".
                for _p in (opt_6d, opt_t, opt_s):
                    _p.requires_grad_(False)
                param_groups = _delta_groups
            else:
                if _freeze == "scale":
                    # The shared correction owns SIZE; hold the native scale.
                    param_groups = freeze_params(param_groups, [opt_s])
                param_groups = param_groups + _delta_groups
            print(f"      obj {obj_idx}: pose correction {_pc_mode} (one Sim(3))")

        c2w_rebase = precompute_c2w_rebases(sequence, frame_indices, ref_frame, device)
        frame_data = _prepare_shared_world_frame_data(
            sequence, obj_idx, frame_indices, losses, device,
        )
        if not frame_data:
            print(f"      Skipping obj {obj_idx}: no frames with mask coverage")
            continue

        # Optional fused multi-view background so the (foreground-only) shared-
        # world render carries valid depth outside the object → full-image depth
        # (depth_mask_only=False).  Empty → masked-depth fallback in the loop.
        _want_bg = bool(getattr(losses, "render_with_background", False))
        bg_cam_per_frame = (
            _build_fused_mv_background(sequence, frame_data, obj_idx, device)
            if _want_bg else {}
        )

        optimizer = torch.optim.Adam(param_groups) if param_groups else None

        obj_setup[obj_idx] = {
            "ref_frame": ref_frame,
            "renderer": renderer,
            "opt_6d": opt_6d, "opt_t": opt_t, "opt_s": opt_s,
            "delta": (delta_aa, delta_logs, delta_t),
            "init_6d": init_6d, "init_t": init_t, "init_s": init_s,
            "ref_ps": ref_ps, "ref_psh": ref_psh, "ref_dsf": ref_dsf,
            "c2w_rebase": c2w_rebase,
            "warp_overrides": warp_overrides,
            "frame_data": frame_data,
            "bg_cam_per_frame": bg_cam_per_frame,
            "optimizer": optimizer,
            "best_loss": float("inf"),
            "best_iteration": 0,
            "best_state": None,
            "loss_history": [],
        }

    if not obj_setup:
        return {o: list(tokens_by_object[o]) for o in all_objs}

    print(f"\n  Shared-world-pose refinement: {len(obj_setup)} object(s), "
          f"{losses.num_iterations} iters/obj")

    # ── Optimization loop (per-object, independent) ──────────────────
    total_steps = losses.num_iterations * len(obj_setup)
    pbar = tqdm(total=total_steps, desc="  Shared-world refine", leave=True, ncols=120)
    for obj_idx, setup in obj_setup.items():
        pbar.set_description(f"  obj {obj_idx}")
        chamfer_model_idx = (
            _chamfer_model_subsample_idx(
                setup["renderer"].num_model_points(),
                int(getattr(losses, "chamfer_max_model_points", 8192)), device)
            if getattr(losses, "chamfer_weight", 0.0) > 0 else None
        )
        for iteration in range(losses.num_iterations):
            if setup["optimizer"] is not None:
                setup["optimizer"].zero_grad()

            # Shared reference cam-space pose (row-convention matrix). Raw mode
            # decodes the tokens via SSI; decoded mode uses the optimized
            # quaternion/translation/scale directly.
            if decoded_mode:
                q_ref = torch.nn.functional.normalize(setup["opt_6d"].reshape(1, 4), dim=-1)
                R_ref = quaternion_to_matrix(q_ref).squeeze(0)
                t_ref = setup["opt_t"].reshape(3)
                s_ref = setup["opt_s"].reshape(1).expand(3).contiguous()  # isotropic
            else:
                matrix_ref, _quat_ref, trans_ref, scale_ref = differentiable_pose_decode(
                    setup["opt_6d"], setup["opt_t"], setup["opt_s"],
                    setup["ref_ps"], setup["ref_psh"], setup["ref_dsf"],
                )
                R_ref = matrix_ref.squeeze(0)
                t_ref = trans_ref.squeeze(0)
                s_ref = scale_ref.squeeze(0)

            # ONE site for the correction, covering both branches: composing it here and
            # deriving is identical to deriving and composing on every frame (see the
            # commuting note at the parameter setup).
            R_ref, t_ref, s_ref = _apply_shared_correction(
                setup["delta"], R_ref, t_ref, s_ref)

            sum_rgb = torch.zeros((), device=device)
            sum_ssim = torch.zeros((), device=device)
            sum_silh = torch.zeros((), device=device)
            sum_depth = torch.zeros((), device=device)
            sum_normals = torch.zeros((), device=device)
            sum_perceptual = torch.zeros((), device=device)
            sum_chamfer = torch.zeros((), device=device)

            want_pixelwise = (save_renders and output_dir is not None
                              and iteration % 50 == 0)
            px_data: Optional[Dict[int, Dict[str, Any]]] = (
                {} if want_pixelwise else None
            )

            total_loss = torch.zeros((), device=device)
            n_frames = 0
            for fi, data in setup["frame_data"].items():
                R_i, t_i, s_i = derive_perframe_pose_from_shared_decoded(
                    R_ref, t_ref, s_ref, setup["c2w_rebase"][fi],
                )
                q_i = matrix_to_quaternion(R_i.unsqueeze(0))  # (1, 4)

                _bg = setup["bg_cam_per_frame"].get(fi)
                # This frame's pre-warped geometry when the object deforms; (None, None)
                # otherwise, which is the static canonical and bit-identical to omitting
                # the arguments.  Precomputed once per frame -- see the setup above.
                _ovr = setup["warp_overrides"].get(fi)
                _means_ovr, _rot_ovr = _ovr if _ovr is not None else (None, None)
                rgb, alpha, depth = setup["renderer"].render(
                    q_i, t_i.unsqueeze(0), s_i.unsqueeze(0),
                    data["K_matrix"], data["W"], data["H"], device, bg_color=bg_color,
                    background_params=_bg,
                    means_override=_means_ovr, rotation_override=_rot_ovr,
                )
                # Full-image depth only where a fused background exists this frame:
                # all-ones loss mask (GT depth not zeroed outside the object) +
                # has_background=True.  Without a background, fall back to masked
                # depth (object mask + depth_mask_only=True) so empty-bg pixels
                # aren't supervised as noise.  Silhouette is inert either way in
                # full-image mode (config sets silhouette_weight=0).
                _has_bg = _bg is not None
                _loss_mask = (
                    torch.ones(data["H"], data["W"], device=device, dtype=torch.bool)
                    if _has_bg else data["mask"]
                )
                losses_dict = _compute_frame_loss(
                    rgb, alpha, data["gt_image"], _loss_mask, losses,
                    data["sdt"], rendered_depth=depth, gt_depth=data["gt_depth"],
                    valid_mask=data["valid_mask"], bg_color=bg_color,
                    K_matrix=data["K_tensor"],
                    return_pixelwise=want_pixelwise,
                    has_background=_has_bg,
                    depth_mask_only=not _has_bg,
                )
                total_loss = total_loss + _aggregate_frame_loss(losses_dict, losses)
                n_frames += 1

                # One-directional Chamfer (GT-depth → posed model means), optional.
                if getattr(losses, "chamfer_weight", 0.0) > 0 and data.get("chamfer_gt_pts") is not None:
                    # The SAME override the render used: a Chamfer term measuring the
                    # undeformed canonical while the photometric terms measure the
                    # deformed frame would be two halves of one loss disagreeing about
                    # the object's shape.
                    _model_xyz = setup["renderer"].model_points(
                        q_i, t_i.unsqueeze(0), s_i.unsqueeze(0), device,
                        means_override=_means_ovr)
                    _cham = getattr(losses, "chamfer_weight", 0.0) * _chamfer_gt_to_model(
                        data["chamfer_gt_pts"], _model_xyz, model_sub_idx=chamfer_model_idx)
                    total_loss = total_loss + _cham
                    sum_chamfer = sum_chamfer + _cham

                if want_pixelwise:
                    _K = data["K_tensor"]
                    if _K is None:
                        _K = torch.from_numpy(data["K_matrix"]).float().to(device)
                    # Mask the GT panels with the SAME mask the loss used, so the
                    # viz matches what was supervised: object-only in masked mode,
                    # full-image (background included) in full-image / has_bg mode.
                    _mask_f = _loss_mask.float().unsqueeze(-1)
                    gt_masked = (
                        data["gt_image"] * _mask_f
                        + bg_color.view(1, 1, 3) * (1.0 - _mask_f)
                    )
                    frame_px: Dict[str, Any] = {
                        "rendered_rgb": rgb.detach().cpu().numpy(),
                        "gt_rgb": gt_masked.detach().cpu().numpy(),
                        "mask": _loss_mask.cpu().numpy(),
                    }
                    if depth is not None:
                        depth_sq = depth.squeeze(0) if depth.dim() == 3 else depth
                        frame_px["rendered_depth"] = depth_sq.detach().cpu().numpy()
                        frame_px["rendered_normals"] = depth_to_normals(
                            depth_sq, _K,
                        ).detach().cpu().numpy()
                    gt_d = data["gt_depth"]
                    if gt_d is not None:
                        _, _K_gt, _mask_gt = _match_depth_grid(
                            _loss_mask, gt_d.shape[-2:], K_matrix=_K, mask=_loss_mask)
                        frame_px["gt_depth"] = (gt_d * _mask_gt.float()).cpu().numpy()
                        frame_px["gt_normals"] = depth_to_normals(
                            gt_d, _K_gt, mask=_mask_gt,
                        ).detach().cpu().numpy()
                    for k, v in losses_dict.items():
                        if k.startswith("px_"):
                            frame_px[k] = v
                    px_data[fi] = frame_px

                sum_rgb = sum_rgb + losses.rgb_weight * losses_dict["rgb_loss"]
                sum_ssim = sum_ssim + losses.rgb_ssim_weight * losses_dict["ssim_loss"]
                sum_silh = sum_silh + losses.silhouette_weight * (
                    losses.silhouette_com_weight * losses_dict["silhouette_com"]
                    + losses.silhouette_sdt_weight * losses_dict["silhouette_sdt"]
                    + losses.silhouette_iou_weight * losses_dict["silhouette_iou"]
                )
                sum_depth = sum_depth + losses.depth_weight * losses_dict["depth_loss"]
                sum_normals = sum_normals + losses.normals_weight * losses_dict["normals_loss"]
                sum_perceptual = sum_perceptual + losses.perceptual_weight * losses_dict["perceptual_loss"]

            denom = max(n_frames, 1)
            total_loss = total_loss / denom

            reg_val = torch.zeros((), device=device)
            if losses.regularization_weight > 0:
                reg = (
                    losses.regularization_rotation_weight * ((setup["opt_6d"] - setup["init_6d"]) ** 2).sum()
                    + losses.regularization_translation_weight * ((setup["opt_t"] - setup["init_t"]) ** 2).sum()
                    + losses.regularization_scale_weight * ((setup["opt_s"] - setup["init_s"]) ** 2).sum()
                )
                reg_val = losses.regularization_weight * reg
                total_loss = total_loss + reg_val

            if setup["optimizer"] is not None:
                total_loss.backward()
                setup["optimizer"].step()

            if want_pixelwise and px_data:
                from genia.core.utils.visualization import save_pixelwise_loss_plot
                with get_timer().exclude():
                    save_pixelwise_loss_plot(px_data, iteration, obj_idx, output_dir)

            loss_val = float(total_loss.detach())
            _to_float = lambda v: v.item() if isinstance(v, torch.Tensor) else float(v)
            setup["loss_history"].append({
                "iteration": iteration,
                "total": loss_val,
                "rgb": _to_float(sum_rgb) / denom,
                "ssim": _to_float(sum_ssim) / denom,
                "silhouette": _to_float(sum_silh) / denom,
                "depth": _to_float(sum_depth) / denom,
                "normals": _to_float(sum_normals) / denom,
                "perceptual": _to_float(sum_perceptual) / denom,
                "chamfer": _to_float(sum_chamfer) / denom,
                "regularization": _to_float(reg_val),
            })
            if loss_val < setup["best_loss"]:
                setup["best_loss"] = loss_val
                setup["best_iteration"] = iteration
                # The correction rides along with the params it was evaluated against,
                # so it can never come from a later iteration than the pose it composes
                # onto.  ``None`` entries under `per_frame` keep the shape uniform.
                _d = setup["delta"]
                setup["best_state"] = (
                    setup["opt_6d"].detach().clone(),
                    setup["opt_t"].detach().clone(),
                    setup["opt_s"].detach().clone(),
                    tuple(None if _x is None else _x.detach().clone() for _x in _d),
                )
            pbar.update(1)
    pbar.close()

    # ── Write-back: apply best state, propagate via rebase ──────────
    refined_tokens: Dict[int, List[Tuple[int, Dict[str, Any]]]] = {}
    for obj_idx in all_objs:
        if obj_idx not in obj_setup:
            refined_tokens[obj_idx] = list(tokens_by_object[obj_idx])
            continue
        setup = obj_setup[obj_idx]
        # A 4-tuple; the fallback fires when every iteration scored NaN
        # (``NaN < inf`` is False), so it must carry the live correction too.
        best_6d, best_t, best_s, best_delta = setup["best_state"] or (
            setup["opt_6d"].detach(), setup["opt_t"].detach(), setup["opt_s"].detach(),
            setup["delta"],
        )
        ref_frame = setup["ref_frame"]

        entries = list(tokens_by_object[obj_idx])
        ref_di = next(di for fi, di in entries if fi == ref_frame)

        pf_raw_obj = perframe_raw_modalities.setdefault(obj_idx, {}) if perframe_raw_modalities is not None else {}
        if decoded_mode:
            # best_6d is the wxyz quaternion; write the decoded ref pose straight
            # into the ref entry (no raw tokens / SSI). rebase_object_poses_from_
            # reference reads it and refreshes raw only where SSI is present.
            #
            # BAKE the correction in.  It was composed onto the reference inside the loop
            # but onto a local, so without this the block returns its INPUT pose under
            # `shared` -- where the natives are frozen -- while its loss history still
            # shows improvement.  Composing on the reference is enough for every frame:
            # the rebase below derives them all from it.
            _q_ref = torch.nn.functional.normalize(best_6d.reshape(1, 4), dim=-1).detach()
            _R_b, _t_b, _s_b = _apply_shared_correction(
                best_delta, quaternion_to_matrix(_q_ref)[0], best_t.reshape(3),
                best_s.reshape(-1))
            ref_di["rotation"] = matrix_to_quaternion(
                _R_b.contiguous().unsqueeze(0)).detach()
            ref_di["translation"] = _t_b.reshape(1, 3).detach()
            ref_di["scale"] = _s_b.reshape(1, -1).detach()
        else:
            # Update reference raw tokens in both tokens_by_object and perframe_raw_modalities.
            ref_raw_mods = ref_di.setdefault("raw_ss_modalities", {})
            ref_raw_mods["6drotation_normalized"] = best_6d
            ref_raw_mods["translation"] = best_t
            ref_raw_mods["scale"] = best_s
            pf_raw_obj.setdefault(ref_frame, {
                "pointmap_scale": setup["ref_ps"],
                "pointmap_shift": setup["ref_psh"],
                "downsample_factor": setup["ref_dsf"],
                "raw_ss_modalities": {},
            })
            pf_ref_raw = pf_raw_obj[ref_frame].setdefault("raw_ss_modalities", {})
            pf_ref_raw["6drotation_normalized"] = best_6d
            pf_ref_raw["translation"] = best_t
            pf_ref_raw["scale"] = best_s

            # Decode ref pose into the ref entry (rebase helper will read this).
            with torch.no_grad():
                _, q_final, t_final, s_final = differentiable_pose_decode(
                    best_6d, best_t, best_s,
                    setup["ref_ps"], setup["ref_psh"], setup["ref_dsf"],
                )
            # Same bake, on the decoded pose.  The raw tokens written above stay
            # UNcorrected, which is harmless: the rebase re-derives them from this decoded
            # pose via `camera_pose_to_raw_tokens`, so raw and decoded end up consistent.
            _R_b, _t_b, _s_b = _apply_shared_correction(
                best_delta, quaternion_to_matrix(q_final.reshape(1, 4))[0],
                t_final.reshape(3), s_final.reshape(-1))
            ref_di["rotation"] = matrix_to_quaternion(
                _R_b.contiguous().unsqueeze(0)).detach()
            ref_di["translation"] = _t_b.reshape(1, 3).detach()
            ref_di["scale"] = _s_b.reshape(1, -1).detach()

        # Propagate reference pose to every frame (updates decoded + raw tokens).
        # scope="all": this solve produces ONE Sim(3) for the whole sequence -- that is what
        # "shared world" means here -- so it reaches every frame, not just the reference's
        # timestamp.  pose_init's rebase is the other axis (motion in the per-frame Sim(3)).
        rebase_object_poses_from_reference(
            entries, pf_raw_obj, sequence, ref_frame, scope="all")

        # Annotate entries with loss history for save_and_plot_loss_history.
        # All frames share one optimization, so every frame points at the same
        # per-object history; setting batch_loss_history selects one-row-per-object
        # plotting in plot_refinement_history.
        for _, di in entries:
            di["refinement_loss_history"] = setup["loss_history"]
            di["refinement_batch_loss_history"] = setup["loss_history"]
            di["refinement_best_iteration"] = setup["best_iteration"]

        refined_tokens[obj_idx] = entries

    return refined_tokens
