# Copyright (c) Meta Platforms, Inc. and affiliates.

"""Preserve decoder metadata and synchronize raw pose tokens."""

from __future__ import annotations

from typing import Any, Dict

import torch
from genia.core.utils.quaternion_ops import quaternion_to_matrix

_CARRY_FORWARD_KEYS = (
    "raw_ss_modalities", "pointmap_scale", "pointmap_shift",
    "downsample_factor", "decoder_input_slat",
)


def _carry_forward_decoder_context(
    source: Dict[str, Any], target: Dict[str, Any],
) -> None:
    """Copy non-pose context fields from *source* to *target* decoder_input dict."""
    for key in _CARRY_FORWARD_KEYS:
        if key in source:
            target[key] = source[key]


def _sync_raw_from_decoded(
    raw_mods: Dict[str, torch.Tensor],
    rotation_quat: torch.Tensor,
    translation: torch.Tensor,
    scale: torch.Tensor,
    raw_source: Dict[str, Any],
    fallback_raw_6d: torch.Tensor,
    fallback_raw_trans: torch.Tensor,
) -> None:
    """Overwrite raw layout tokens in *raw_mods* so that the SSI decode of the
    raw tokens round-trips to (``rotation_quat``, ``translation``, ``scale``).

    The global-scale optimizers in this module treat ``opt_global_scale`` as
    an independent decoded variable (decoupled from ``frozen_raw_scales``),
    so the optimized raw scale token is stale relative to the final decoded
    scale, and the optimized raw translation was paired with the frozen raw
    scale.  Inverting via ``camera_pose_to_raw_tokens`` produces raw tokens
    consistent with the decoded triplet.

    Falls back to the optimized raw rotation/translation when SSI params are
    missing on *raw_source* (no SSI snapshot).  In that case the
    raw scale stays whatever was already in *raw_mods*: no consumer can
    re-decode pose without SSI params anyway, so the contract is moot.
    """
    from genia.core.utils.pose_token_gt import camera_pose_to_raw_tokens

    ps = raw_source.get("pointmap_scale")
    psh = raw_source.get("pointmap_shift")
    if ps is None or psh is None:
        raw_mods["6drotation_normalized"] = fallback_raw_6d
        raw_mods["translation"] = fallback_raw_trans
        return

    dsf = raw_source.get("downsample_factor", 1.0)
    R_mat = quaternion_to_matrix(rotation_quat.reshape(1, 4)).squeeze(0)
    t_vec = translation.reshape(-1).float()
    s_vec = scale.reshape(-1).float()
    if s_vec.numel() == 1:
        s_vec = s_vec.expand(3)
    device = R_mat.device
    with torch.no_grad():
        new_raw = camera_pose_to_raw_tokens(
            R_mat, t_vec.to(device), s_vec.to(device),
            ps.flatten().to(device), psh.flatten().to(device),
            downsample_factor=dsf,
        )
    raw_mods["6drotation_normalized"] = new_raw["6drotation_normalized"]
    raw_mods["translation"] = new_raw["translation"]
    raw_mods["scale"] = new_raw["scale"]
