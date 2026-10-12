# Copyright (c) Meta Platforms, Inc. and affiliates.

"""Compatibility imports; implementations live in ``genia.core.finetune``."""

from genia.core.finetune.anchors import (
    _apply_decoder_anchors,
    _detach_invisible_gaussians,
    _setup_opacity_anchor,
)
from genia.core.finetune.decoder import (
    _ft_decoder_autocast_bf16,
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
from genia.core.finetune.losses import (
    _sample_random_bg_color,
    compute_total_loss,
)
from genia.core.finetune.metrics import (
    _compute_mean_psnr_ssim,
)
from genia.core.finetune.object_optimizer import (
    _finetune_single_object,
)
from genia.core.finetune.poses import (
    compose_pose_with_root_delta,
    derive_shared_world_poses,
    snapshot_poses_with_correction,
)
from genia.core.finetune.strategies import (
    finetune_canonical_tokens,
    finetune_perframe_tokens,
)
from genia.core.finetune.visibility import (
    _GAUSSIAN_RAW_ATTRS,
    _compute_gaussian_visibility,
    _flag_visible_by_depth,
    _setup_visibility_mask,
    _visibility_warp_bundle,
)
from genia.core.utils.frame_sampling import EpochFrameSampler
