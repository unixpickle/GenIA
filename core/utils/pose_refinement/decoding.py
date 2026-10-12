# Copyright (c) Meta Platforms, Inc. and affiliates.

"""Decode normalized rotation and scale-shift-invariant pose tokens."""

from __future__ import annotations

from typing import Tuple

import torch
from genia.core.utils.quaternion_ops import matrix_to_quaternion

_ROTATION_6D_MEAN = torch.tensor([
    -0.06366084883674913, 0.008438224692279752, 0.00017084786438302483,
    0.0007126610473540038, -0.0030916726538816417, 0.5166093753457688,
])


_ROTATION_6D_STD = torch.tensor([
    0.6656971967514863, 0.6787012271867754, 0.30345010594844524,
    0.4394504420678794, 0.39817973931717104, 0.6176286868761914,
])


def differentiable_rotation_decode(
    raw_6drot_normalized: torch.Tensor,
) -> torch.Tensor:
    """Decode raw normalized 6D rotation tokens to rotation matrices.

    Steps: denormalize → Gram-Schmidt orthogonalization.
    Matches SAM3D's pose_decoder rotation chain exactly.

    Parameters
    ----------
    raw_6drot_normalized : torch.Tensor
        Normalized 6D rotation from Stage 1, shape ``(..., 6)``.

    Returns
    -------
    torch.Tensor
        Rotation matrix/matrices ``(..., 3, 3)``.
    """
    device = raw_6drot_normalized.device
    mean = _ROTATION_6D_MEAN.to(device)
    std = _ROTATION_6D_STD.to(device)
    rot_6d = raw_6drot_normalized * std + mean
    a1 = rot_6d[..., 0:3]
    a2 = rot_6d[..., 3:6]
    b1 = torch.nn.functional.normalize(a1, dim=-1)
    b2 = a2 - (b1 * a2).sum(dim=-1, keepdim=True) * b1
    b2 = torch.nn.functional.normalize(b2, dim=-1)
    b3 = torch.cross(b1, b2, dim=-1)
    return torch.stack([b1, b2, b3], dim=-1)  # (..., 3, 3)


def differentiable_pose_decode(
    raw_6drot_normalized: torch.Tensor,
    raw_translation: torch.Tensor,
    raw_scale: torch.Tensor,
    scene_scale: torch.Tensor,
    scene_shift: torch.Tensor,
    downsample_factor: float = 1.0,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Convert raw Stage 1 pose tokens to final pose via differentiable chain.

    Uses ``ScaleShiftInvariant.to_instance_pose`` directly.

    Accepts either single-token ``(D,)`` or multi-token ``(T, D)`` inputs.
    When multi-token, each token is decoded independently through the full
    SSI chain (matching SAM3D's ``_broadcast_postcompose``).

    Returns a camera-space pose (no c2w composition).

    Parameters
    ----------
    raw_6drot_normalized : torch.Tensor
        Normalized 6D rotation, shape ``(6,)`` or ``(T, 6)``.
    raw_translation : torch.Tensor
        Raw translation, shape ``(3,)`` or ``(T, 3)``.
    raw_scale : torch.Tensor
        Raw log-scale, shape ``(1,)``, ``(3,)``, ``(T, 1)``, or ``(T, 3)``.
    scene_scale : torch.Tensor
        Pointmap scale, shape ``(3,)`` or ``(1, 3)``.
    scene_shift : torch.Tensor
        Pointmap shift, shape ``(3,)`` or ``(1, 3)``.
    downsample_factor : float
        Post-multiplier for scale (from sparse-structure downsampling).

    Returns
    -------
    rotation_matrix : torch.Tensor
        Rotation matrix ``(1, 3, 3)`` or ``(T, 3, 3)``.
    quaternion : torch.Tensor
        Quaternion ``(1, 4)`` or ``(T, 4)`` in wxyz format.
    translation : torch.Tensor
        Translation ``(1, 3)`` or ``(T, 3)``.
    scale : torch.Tensor
        Isotropic scale ``(1, 3)`` or ``(T, 3)``.
    """
    from pytorch3d.transforms import quaternion_to_matrix as _q2m
    from sam3d_objects.data.dataset.tdfy.pose_target import PoseTarget, ScaleShiftInvariant

    device = raw_6drot_normalized.device

    # Raw modalities stored by inference carry a (batch=1, T, D) shape
    # (see elem_raw in core/inference.py). Squeeze the leading batch dim so
    # downstream logic treats the remaining leading dim as the token count T.
    def _squeeze_leading_batch(x: torch.Tensor) -> torch.Tensor:
        return x.squeeze(0) if x.dim() == 3 and x.shape[0] == 1 else x

    raw_6drot_normalized = _squeeze_leading_batch(raw_6drot_normalized)
    raw_translation = _squeeze_leading_batch(raw_translation)
    raw_scale = _squeeze_leading_batch(raw_scale)

    # Rotation: denormalize + Gram-Schmidt → matrix → quaternion
    rotation_matrix = differentiable_rotation_decode(raw_6drot_normalized)
    # Ensure 3D: (T, 3, 3) — unsqueeze if single-token (3, 3)
    if rotation_matrix.dim() == 2:
        rotation_matrix = rotation_matrix.unsqueeze(0)
    quaternion = matrix_to_quaternion(rotation_matrix)  # (T, 4)

    # Scale: exp
    raw_scale_2d = raw_scale if raw_scale.dim() >= 2 else raw_scale.unsqueeze(0)
    instance_scale = torch.exp(raw_scale_2d)  # (T, D_scale)

    # Translation: ensure 2D
    translation = raw_translation if raw_translation.dim() >= 2 else raw_translation.unsqueeze(0)

    # SSI decode via ScaleShiftInvariant.to_instance_pose
    # (same path as training's decode_poses_pipeline).
    # Loop per-token because _broadcast_postcompose treats the leading dim
    # as batch and the SSI transform is per-frame (shared across tokens).
    T = quaternion.shape[0]
    ssi_scale = scene_scale.flatten().to(device)
    ssi_shift = scene_shift.flatten().to(device)

    dec_Rs, dec_qs, dec_ts, dec_ss = [], [], [], []
    for i in range(T):
        pose_target = PoseTarget(
            x_instance_scale=instance_scale[i:i+1],
            x_instance_rotation=quaternion[i:i+1],
            x_instance_translation=translation[i:i+1],
            x_scene_scale=ssi_scale,
            x_scene_center=ssi_shift,
            x_translation_scale=torch.ones(1, 1, device=device),
            pose_target_convention=ScaleShiftInvariant.pose_target_convention,
        )
        ip = ScaleShiftInvariant.to_instance_pose(pose_target)
        q = ip.instance_quaternion_l2c.squeeze(0)   # (4,)
        dec_qs.append(q)
        dec_Rs.append(_q2m(q.unsqueeze(0)).squeeze(0))  # (3, 3)
        dec_ts.append(ip.instance_position_l2c.squeeze(0))  # (3,)
        s = ip.instance_scale_l2c.squeeze(0)  # (D,)
        dec_ss.append(s)

    dec_R = torch.stack(dec_Rs)  # (T, 3, 3)
    dec_q = torch.stack(dec_qs)  # (T, 4)
    dec_t = torch.stack(dec_ts)  # (T, 3)
    dec_s = torch.stack(dec_ss)  # (T, D)

    # Apply downsample_factor + make isotropic (always return 3-component scale)
    final_scale = dec_s * downsample_factor
    final_scale = final_scale.mean(dim=-1, keepdim=True).expand(*dec_s.shape[:-1], 3)

    return dec_R, dec_q, dec_t, final_scale
