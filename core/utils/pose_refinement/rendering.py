# Copyright (c) Meta Platforms, Inc. and affiliates.

"""Apply object poses and render differentiable foreground/background scenes."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, NamedTuple, Optional, Tuple

import numpy as np
import torch
from genia.core.utils.config import PipelineConfig
from genia.core.utils.quaternion_ops import (
    p3d_to_r3_positions,
    p3d_to_r3_quaternions,
    quaternion_invert,
    quaternion_multiply,
    quaternion_to_matrix,
)
from genia.core.utils.rendering import render_gaussian_params

if TYPE_CHECKING:
    from sam3d_objects.model.backbone.tdfy_dit.representations.gaussian.gaussian_model import (
        Gaussian,
    )


def _get_bg_color(pipeline: Optional[PipelineConfig], device: torch.device) -> torch.Tensor:
    """Return rendering background color from pipeline config (black or white)."""
    if pipeline is not None and pipeline.white_background:
        return torch.ones(3, device=device)
    return torch.zeros(3, device=device)


def apply_pose_to_gaussian(
    canonical_gs: "Gaussian",
    rotation: torch.Tensor,
    translation: torch.Tensor,
    scale: torch.Tensor,
    *,
    means_override: "torch.Tensor | None" = None,
    rotation_override: "torch.Tensor | None" = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Apply a pose transformation to Gaussian positions and rotations.

    Parameters
    ----------
    canonical_gs : Gaussian
        The canonical Gaussian object (frozen, not modified).
    rotation : torch.Tensor
        Quaternion rotation (1, 4) or (4,).
    translation : torch.Tensor
        Translation (1, 3) or (3,).
    scale : torch.Tensor
        Scale factor (1, 3) or (3,) or (1,) or scalar.
    means_override : torch.Tensor or None, keyword-only
        When provided, used in place of ``canonical_gs.get_xyz`` for the
        pose math.  Shape ``(N, 3)``.  Used by the keyframes renderer to
        inject per-frame warped Gaussian means without bypassing this
        function.  Other attributes (rotation, scaling, opacity, features)
        still come from ``canonical_gs`` unless ``rotation_override`` is
        also set.
    rotation_override : torch.Tensor or None, keyword-only
        When provided, used in place of ``canonical_gs.get_rotation``.
        Shape ``(N, 4)`` wxyz.  Companion to ``means_override``.

    Returns
    -------
    tuple
        (transformed_means, transformed_quats, scales, opacities, features)
    """
    # Get canonical Gaussian attributes (overrides applied selectively).
    xyz_local = (
        means_override if means_override is not None else canonical_gs.get_xyz
    )                                                                # (N, 3)
    rot_local = (
        rotation_override
        if rotation_override is not None
        else canonical_gs.get_rotation
    )                                                                # (N, 4)
    scales_local = canonical_gs.get_scaling  # (N, 3)
    opacities = canonical_gs.get_opacity  # (N, 1)
    features = canonical_gs.get_features  # (N, 1, 3) or (N, K, 3)

    # Ensure rotation is (4,)
    if rotation.dim() == 2:
        rotation = rotation.squeeze(0)

    # Ensure translation is (3,)
    if translation.dim() == 2:
        translation = translation.squeeze(0)

    # Ensure scale is (3,)
    if scale.dim() == 0:
        scale = scale.expand(3)
    elif scale.dim() == 1 and scale.shape[0] == 1:
        scale = scale.expand(3)
    elif scale.dim() == 2:
        scale = scale.squeeze(0)
        if scale.shape[0] == 1:
            scale = scale.expand(3)

    # Normalize quaternion
    rotation = rotation / rotation.norm()

    # Convert quaternion to rotation matrix
    # PyTorch3D convention: quaternion_to_matrix gives R such that points are transformed as p @ R
    R = quaternion_to_matrix(rotation.unsqueeze(0)).squeeze(0)  # (3, 3)

    # Transform positions following compose_transform convention:
    # tfm = Scale(scale).compose(Rotate(R)).compose(Translate(trans))
    # transformed = points * scale @ R + trans
    scaled_xyz = xyz_local * scale
    transformed_xyz = torch.mm(scaled_xyz, R) + translation  # Note: @ R, not @ R.T

    # Transform rotations: rot_world = quaternion_multiply(rotation_inv, rot_local)
    # Note: Using inverse because of the convention in make_scene
    rotation_inv = quaternion_invert(rotation.unsqueeze(0)).squeeze(0)
    transformed_rot = quaternion_multiply(
        rotation_inv.unsqueeze(0).expand(rot_local.shape[0], -1), rot_local
    )

    # Transform scales
    transformed_scales = scales_local * scale

    return transformed_xyz, transformed_rot, transformed_scales, opacities, features


def _transform_object_to_r3(
    canonical_gs: "Gaussian",
    rotation: torch.Tensor,
    translation: torch.Tensor,
    scale: torch.Tensor,
    device: torch.device,
    *,
    means_override: "torch.Tensor | None" = None,
    rotation_override: "torch.Tensor | None" = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Apply pose to canonical Gaussians and transform to R3 coordinate system.

    Returns raw tensor tuples ready for rendering or concatenation across objects.

    Parameters
    ----------
    canonical_gs : Gaussian
        The canonical Gaussian object.
    rotation : torch.Tensor
        Quaternion rotation.
    translation : torch.Tensor
        Translation vector.
    scale : torch.Tensor
        Scale factor.
    device : torch.device
        Device to use.
    means_override, rotation_override : torch.Tensor or None, keyword-only
        Forwarded to :func:`apply_pose_to_gaussian`.  Used by the actionmesh
        FINETUNE path to pass already-warped Gaussian means / quats from
        ``warp_gaussians_high_res``.  Identity-default leaves the canonical
        means / rotations untouched.

    Returns
    -------
    tuple
        (transformed_xyz, transformed_rot, transformed_scales, opacities, features)
    """
    # Apply pose to canonical Gaussian
    transformed_xyz, transformed_rot, transformed_scales, opacities, features = (
        apply_pose_to_gaussian(
            canonical_gs=canonical_gs, rotation=rotation,
            translation=translation, scale=scale,
            means_override=means_override,
            rotation_override=rotation_override,
        )
    )

    # Transform from PyTorch3D convention to R3 convention
    # P3D↔R3 is diag(-1,-1,1): negate X and Y for positions, apply 180° Z rotation for quaternions
    transformed_xyz = p3d_to_r3_positions(transformed_xyz)
    transformed_rot = p3d_to_r3_quaternions(transformed_rot)

    # Handle opacities
    opacities_flat = opacities.squeeze(-1) if opacities.dim() == 2 else opacities

    return transformed_xyz, transformed_rot, transformed_scales, opacities_flat, features


def _concat_fg_bg_params(
    fg_xyz, fg_rot, fg_scales, fg_opac, fg_feats, bg_params, device,
):
    """Concatenate a DETACHED background Gaussian tuple onto the foreground params
    for a full-image render, matching the foreground SH layout/degree first.

    ``bg_params`` is ``(xyz, rot, scales, opac, feats)`` in the SAME (camera)
    space as the foreground.  Returns the concatenated ``(xyz, rot, scales, opac,
    feats)``.  Mirrors the fg+bg concat in ``refine_poses_global_composite``.
    """
    bg_xyz, bg_rot, bg_scales, bg_opac, bg_feats = bg_params
    if fg_feats.dim() == 2:
        fg_feats = fg_feats.unsqueeze(1)
    if bg_feats.dim() == 2:
        bg_feats = bg_feats.unsqueeze(1)
    if fg_feats.shape[1] > bg_feats.shape[1]:
        _pad = torch.zeros(
            bg_feats.shape[0], fg_feats.shape[1] - bg_feats.shape[1], 3,
            device=device, dtype=bg_feats.dtype,
        )
        bg_feats = torch.cat([bg_feats, _pad], dim=1)
    return (
        torch.cat([fg_xyz, bg_xyz], dim=0),
        torch.cat([fg_rot, bg_rot], dim=0),
        torch.cat([fg_scales, bg_scales], dim=0),
        torch.cat([fg_opac, bg_opac], dim=0),
        torch.cat([fg_feats, bg_feats], dim=0),
    )


def _render_frame_with_pose(
    canonical_gs: "Gaussian",
    rotation: torch.Tensor,
    translation: torch.Tensor,
    scale: torch.Tensor,
    K_matrix: np.ndarray,
    W: int,
    H: int,
    device: torch.device,
    bg_color: Optional[torch.Tensor] = None,
    *,
    means_override: "torch.Tensor | None" = None,
    rotation_override: "torch.Tensor | None" = None,
    background_params: "Tuple[torch.Tensor, ...] | None" = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Render a single frame with given pose parameters.

    Assumes poses are in camera space; renders with identity c2w.

    Parameters
    ----------
    canonical_gs : Gaussian
        The canonical Gaussian object.
    rotation : torch.Tensor
        Quaternion rotation (camera space).
    translation : torch.Tensor
        Translation vector (camera space).
    scale : torch.Tensor
        Scale factor.
    K_matrix : np.ndarray
        Camera intrinsics.
    W : int
        Image width.
    H : int
        Image height.
    device : torch.device
        Device to use.
    bg_color : torch.Tensor or None, optional
        Background color (3,). Defaults to black (0,0,0).
    means_override, rotation_override : torch.Tensor or None, keyword-only
        Forwarded to :func:`_transform_object_to_r3` /
        :func:`apply_pose_to_gaussian` — used by the actionmesh FINETUNE
        path to inject warped Gaussian means + quats from
        ``warp_gaussians_high_res`` before applying Stage-1 pose.

    Returns
    -------
    tuple
        (rgb, alpha, depth) where rgb is (H, W, 3) and alpha is (H, W), depth is (H, W).
    """
    xyz, rot, scales, opac, feats = _transform_object_to_r3(
        canonical_gs, rotation, translation, scale, device,
        means_override=means_override,
        rotation_override=rotation_override,
    )

    # Optional DETACHED background (same camera space) for full-image depth:
    # concat behind the posed foreground so the render carries valid depth
    # outside the object.  The bg carries no gradient (pose grads flow only
    # through the foreground); the composite depth is fg where the object is,
    # bg elsewhere.
    if background_params is not None:
        xyz, rot, scales, opac, feats = _concat_fg_bg_params(
            xyz, rot, scales, opac, feats, background_params, device,
        )

    c2w_tensor = torch.eye(4, device=device, dtype=torch.float32).unsqueeze(0)
    if bg_color is None:
        bg_color = torch.zeros(3, device=device)

    return render_gaussian_params(
        xyz, rot, scales, opac, feats,
        c2w_tensor, K_matrix, W, H, bg_color=bg_color,
    )


class PosedObjectRenderer(NamedTuple):
    """One canonical object behind a representation-agnostic render interface.

    Built by :func:`make_posed_object_renderer`. Members:

    ``render(q, t, s, K, W, H, device, bg_color=None, background_params=None, *,
    means_override=None, rotation_override=None)``
        Rasterise the object under one camera-space Sim(3) → ``(rgb (H,W,3),
        alpha (H,W), depth (H,W))``. Differentiable w.r.t. ``q``/``t``/``s``.
        The two overrides mean: *use this
        pre-warped geometry instead of the canonical* — how a DEFORMING object is
        rendered at one frame. Both ``None`` is the static canonical, bit-identical
        to a call that omits them.
    ``model_points(q, t, s, device, *, means_override=None)``
        The posed object's points in R3 camera space ``(N, 3)`` (Chamfer target).
        Takes the same override, and a caller that warps ``render`` must pass it
        here too — otherwise the Chamfer term scores the undeformed surface while
        the photometric terms score the deformed one.
    ``num_model_points()``
        ``N`` for the above, computed lazily (``Gaussian.get_xyz`` recomputes).
    """

    render: Any
    model_points: Any
    num_model_points: Any


def make_posed_object_renderer(source: Any, device: torch.device) -> PosedObjectRenderer:
    """Adapt a canonical reconstruction (a repo ``Gaussian``) to
    :class:`PosedObjectRenderer`, so pose optimisation is written once.

    Renders through gsplat via :func:`_render_frame_with_pose` (the call is forwarded
    unchanged, so numerics are identical).
    """
    def render(q, t, s, K, W, H, dev, bg_color=None, background_params=None,
               *, means_override=None, rotation_override=None):
        return _render_frame_with_pose(
            source, q, t, s, K, W, H, dev,
            bg_color=bg_color, background_params=background_params,
            means_override=means_override, rotation_override=rotation_override,
        )

    def model_points(q, t, s, dev, *, means_override=None):
        return _transform_object_to_r3(
            source, q, t, s, dev, means_override=means_override)[0]

    return PosedObjectRenderer(
        render, model_points, lambda: int(source.get_xyz.shape[0]),
    )
