# Copyright (c) Meta Platforms, Inc. and affiliates.

"""Compose root corrections and derive shared world-space object poses."""

from __future__ import annotations

import torch


def compose_pose_with_root_delta(pose, root_delta):
    """``pose_i o dT`` for one frame, returning a new pose dict.

    The delta is ONE Sim(3) shared by every frame, applied in the object's CANONICAL
    frame, so per-frame root MOTION survives -- on a dynamic scene this is the
    canonical-frame correction propagating outward.

    NO TRANSPOSES.  The renderer poses points as ``x @ quaternion_to_matrix(q)``
    (``apply_pose_to_gaussian``), so ``quaternion_to_matrix(q)`` IS the row-vector matrix
    ``compose_root_delta`` documents, and ``matrix_to_quaternion`` takes it straight back.
    Transposing on the way in and out would compose the delta in CAMERA space instead --
    ``Q_new = Q @ dR`` rather than ``dR @ Q`` -- which is a different transform per frame
    the moment two frames disagree in rotation, cannot express a shared canonical-frame
    correction at all when they do, and breaks the shared-world invariant.
    """
    from genia.core.utils.pose_refit import compose_root_delta
    from genia.core.utils.quaternion_ops import matrix_to_quaternion, quaternion_to_matrix
    from pytorch3d.transforms import axis_angle_to_matrix

    R_row = quaternion_to_matrix(pose["rotation"].reshape(1, 4))[0]
    dR_row = axis_angle_to_matrix(root_delta["aa"])
    s_i = pose["scale"].reshape(-1)
    R_new, s_new, t_new = compose_root_delta(
        dR_row, torch.exp(root_delta["log_ds"]).reshape(()), root_delta["dt"],
        R_row, s_i.mean(), pose["translation"].reshape(3))
    out = dict(pose)
    out["rotation"] = matrix_to_quaternion(R_new.contiguous().unsqueeze(0)).squeeze(0)
    out["translation"] = t_new
    out["scale"] = s_new.reshape(1).expand(s_i.shape[0]) if s_i.shape[0] > 1 else s_new.reshape(1)
    return out


def derive_shared_world_poses(poses, ts_ref, groups, rebase, sw_q, sw_t, scale_for_render):
    """Rewrite every frame's pose from its TIMESTAMP's shared (q, t), in place.

    One shared pose per (object, timestamp); each view of that timestamp is derived by the
    fixed c2w rebase ``M_i``, so gradients from every view land on the one shared leaf and
    the object keeps a single world placement per instant.

    Must run INSIDE the optimization loop, once per iteration.  The existing per-frame
    scale alias (``expand(3).unsqueeze(0)``) may be built once before the loop only
    because ``expand``/``unsqueeze`` save no tensors for backward; the quaternion
    normalize and the matmuls here do, so a graph built once would raise "backward through
    the graph a second time" on iteration 2.

    The normalize is load-bearing.  ``quaternion_ops.quaternion_to_matrix`` does NOT
    normalize, while ``apply_pose_to_gaussian`` (the per-frame path) does -- so an
    Adam-drifted ``|q| != 1`` would silently scale the derived rotation matrix by
    ``|q|^2``, i.e. become an extra scale factor fighting ``opt_global_scale``.

    Rebinding ``poses[fk]`` is safe: every in-loop consumer reads it at call time, and the
    dict-splat preserves the scale alias.
    """
    from genia.core.utils.pose_params import derive_perframe_pose_from_shared_decoded
    from genia.core.utils.quaternion_ops import matrix_to_quaternion, quaternion_to_matrix

    for ts in ts_ref:
        q = torch.nn.functional.normalize(sw_q[ts].reshape(1, 4), dim=-1)
        R_ref = quaternion_to_matrix(q)[0]
        t_ref = sw_t[ts].reshape(3)
        for fk in groups[ts]:
            R_i, t_i, _ = derive_perframe_pose_from_shared_decoded(
                R_ref, t_ref, scale_for_render, rebase[fk])
            poses[fk] = {
                **poses[fk],
                "rotation": matrix_to_quaternion(R_i.unsqueeze(0)).reshape(1, 4),
                "translation": t_i.reshape(1, 3),
            }


def snapshot_poses_with_correction(poses, root_delta=None):
    """Detached copy of ``poses``, with the shared correction COMPOSED IN when there is one.

    The correction is applied per frame at loss time (``compute_total_loss``) but onto a
    COPY, so without this it never reaches ``tokens_by_object``: under
    ``correction_granularity: shared`` the native per-frame params are frozen, so the block would
    return its INPUT poses unchanged and its own before/after PSNR would show nothing
    happened.  ``pose_refit.apply_icp_refine`` is the path this mirrors.

    Restores the STORED shapes.  ``compose_pose_with_root_delta`` returns
    ``(4,)/(3,)/(1,|3,)`` while every consumer downstream expects the
    ``(1,4)/(1,3)/(1,3)`` that ``decode_perframe_poses_from_raw`` writes -- bare 1-D
    survives everything until ``make_scene`` hands the translation to PyTorch3D's
    ``Translate``, which reads it as the X coordinate with Y and Z missing.
    """
    out = {}
    for fi, pose in poses.items():
        if root_delta is None:
            out[fi] = {k: v.clone().detach() for k, v in pose.items()}
            continue
        composed = compose_pose_with_root_delta(pose, root_delta)
        snap = {k: v.clone().detach() for k, v in composed.items()}
        snap["rotation"] = snap["rotation"].reshape(1, 4)
        snap["translation"] = snap["translation"].reshape(1, 3)
        snap["scale"] = snap["scale"].reshape(-1).expand(3).reshape(1, 3)
        out[fi] = snap
    return out
