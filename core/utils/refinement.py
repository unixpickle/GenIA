# Copyright (c) Meta Platforms, Inc. and affiliates.

"""Compatibility imports; implementations live in ``genia.core.utils.pose_refinement``."""

from genia.core.utils.config import LossConfig, PipelineConfig
from genia.core.utils.frame_sampling import EpochFrameSampler
from genia.core.utils.pose_refinement.aggregation import (
    _NONFINITE_TERM_REPORTED,
    _aggregate_frame_loss,
    _any_device,
    _group_contribution,
    _term_groups,
)
from genia.core.utils.pose_refinement.chamfer import (
    _chamfer_gt_points_for_object,
    _chamfer_gt_points_from_depth,
    _chamfer_gt_to_model,
    _chamfer_model_subsample_idx,
    _subsample_idx,
    rendered_surface_cloud,
    rendered_surface_keep,
    trim_and_subsample,
)
from genia.core.utils.pose_refinement.composite import (
    refine_poses_global_composite,
)
from genia.core.utils.pose_refinement.context import (
    _CARRY_FORWARD_KEYS,
    _carry_forward_decoder_context,
    _sync_raw_from_decoded,
)
from genia.core.utils.pose_refinement.decoding import (
    _ROTATION_6D_MEAN,
    _ROTATION_6D_STD,
    differentiable_pose_decode,
    differentiable_rotation_decode,
)
from genia.core.utils.pose_refinement.frame_loss import (
    _compute_frame_loss,
    _prepare_frame_data_for_refinement,
)
from genia.core.utils.pose_refinement.loss_terms import (
    _compute_center_of_mass_loss,
    _compute_depth_discontinuity_mask,
    _compute_multiscale_rgb_loss,
    _compute_sdt_loss,
    _compute_signed_distance_transform,
    _compute_soft_iou_loss,
    _compute_ssim_loss,
    _depth_l1_term,
    _erode_bool_mask,
    _fused_ssim,
    _l1_loss,
    _match_depth_grid,
    depth_to_normals,
)
from genia.core.utils.pose_refinement.per_frame import (
    refine_pose_for_frame,
)
from genia.core.utils.pose_refinement.rendering import (
    PosedObjectRenderer,
    _concat_fg_bg_params,
    _get_bg_color,
    _render_frame_with_pose,
    _transform_object_to_r3,
    apply_pose_to_gaussian,
    make_posed_object_renderer,
)
from genia.core.utils.pose_refinement.sequence import (
    refine_poses_for_sequence,
)
from genia.core.utils.pose_refinement.shared_world import (
    _SHARED_WORLD_BG_MAX_POINTS,
    _apply_gs_correction,
    _apply_shared_correction,
    _build_fused_mv_background,
    _prepare_shared_world_frame_data,
    prefer_present,
    refine_poses_shared_world_for_sequence,
)
from genia.core.utils.pose_refinement.warping import (
    _perframe_warp_overrides,
    _precompute_perframe_gaussian_warps,
)

__all__ = [
    "LossConfig",
    "PipelineConfig",
    "apply_pose_to_gaussian",
    "refine_pose_for_frame",
    "refine_poses_global_composite",
    "refine_poses_for_sequence",
]
