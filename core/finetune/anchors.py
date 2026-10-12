# Copyright (c) Meta Platforms, Inc. and affiliates.

"""Freeze invisible Gaussian attributes and apply decoder anchors."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import torch
from genia.core.config import FinetuningConfig
from genia.core.finetune.visibility import _GAUSSIAN_RAW_ATTRS, _compute_gaussian_visibility
from genia.core.utils.slat_decode import decode_tokens


def _detach_invisible_gaussians(gs: Any, invisible: torch.Tensor) -> None:
    """Stop-gradient the occluded Gaussians' decoded attributes IN PLACE, before the
    render, so backward sends no gradient through them to the tokens OR the shared
    (LoRA'd) decoder.  *invisible* is the stable per-Gaussian mask from
    :func:`_setup_visibility_mask`.  Visible rows keep full gradient."""
    for name in _GAUSSIAN_RAW_ATTRS:
        attr = getattr(gs, name, None)
        if attr is None or not attr.requires_grad:  # _features_rest may be None
            continue
        m = invisible.view(-1, *([1] * (attr.dim() - 1)))
        setattr(gs, name, torch.where(m, attr.detach(), attr))


def _setup_opacity_anchor(
    initial_gs: Any,
    poses: Dict[int, Dict[str, torch.Tensor]],
    frames: List[int],
    sequence: Any,
    obj_idx: int,
    device: torch.device,
    ft_config: FinetuningConfig,
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """One-time setup for the invisible-Gaussian opacity anchor.

    Returns ``(init_opacity, invisible)``: the iter-0 per-Gaussian opacity (always
    captured) and the boolean train-invisible mask (``None`` when no train frame
    had a valid pose, leaving the anchor disabled).  Both are stable for the whole
    run — recomputing per-iter would re-flag already-collapsed Gaussians.  See
    ``FinetuningConfig.invisible_opacity_anchor_weight``.
    """
    init_opacity = initial_gs.get_opacity.detach().reshape(-1).clone()
    vis = _compute_gaussian_visibility(
        initial_gs, poses, [fi for fi in frames if fi in poses],
        sequence, obj_idx, device, ft_config,
    )
    if vis is None:
        print(f"  Invisible-opacity anchor (obj {obj_idx}): no valid train "
              f"frame for visibility — anchor disabled this run")
        return init_opacity, None
    invisible = ~vis
    print(f"  Invisible-opacity anchor (obj {obj_idx}): "
          f"weight={ft_config.invisible_opacity_anchor_weight}, "
          f"{int(invisible.sum())}/{invisible.numel()} Gaussians train-invisible")
    return init_opacity, invisible


def _apply_decoder_anchors(
    decoder: Any,
    opt_feats: torch.Tensor,
    coords: torch.Tensor,
    init_opacity: Optional[torch.Tensor],
    invisible: Optional[torch.Tensor],
    opacity_weight: float,
    init_rotation: Optional[torch.Tensor],
    rotation_weight: float,
    autocast_bf16: bool,
    metrics: Dict[str, float],
) -> None:
    """Opacity + rotation anchors — gradient routed to the decoder's LoRA ONLY.

    Both regularize a DECODED Gaussian attribute back toward its iter-0 value, so
    they adapt the (LoRA'd) decoder's token→attr mapping rather than rewrite the
    SLAT tokens — perturbing the tokens corrupts the shape representation (the
    opacity anchor demonstrably backfires that way, driving back-side opacity
    DOWN).  A single ``opt_feats.detach()`` re-decode serves both; ``backward()``
    runs on this independent graph (detached tokens) and accumulates into the
    LoRA params' ``.grad`` on top of the main loss's, leaving ``opt_feats.grad``
    untouched.

    - opacity (``invisible_opacity_anchor_weight``): one-sided ``relu(init - cur)``
      on the train-invisible Gaussians.
    - rotation (``gauss_rotation_anchor_weight``): L2 ``(cur - init)`` on every
      Gaussian's raw ``_rotation``.

    No-op when both are off, and also when the decoder is FROZEN
    (``lora_decoder``/``optimize_decoder`` off): the tokens are detached, so a
    frozen decoder leaves no grad-requiring parameter in this graph — the decode
    carries no ``grad_fn`` and ``backward()`` would raise.  Anchoring a decoder
    that cannot move is meaningless by construction, so skip it.  Call AFTER the
    main ``total_loss.backward()``, before ``optimizer.step()``.
    """
    op_on = (opacity_weight > 0 and invisible is not None
             and init_opacity is not None and bool(invisible.any()))
    rot_on = rotation_weight > 0 and init_rotation is not None
    if not (op_on or rot_on):
        return
    # Checked before the decode so a frozen decoder costs nothing here.
    if not any(p.requires_grad for p in decoder.parameters()):
        return
    gs = decode_tokens(decoder, opt_feats.detach(), coords, autocast_bf16=autocast_bf16)
    loss = None
    if op_on:
        deficit = torch.relu(init_opacity - gs.get_opacity.reshape(-1))[invisible]
        op_term = opacity_weight * deficit.pow(2).mean()
        metrics["opacity_anchor"] = op_term.item()
        loss = op_term
    if rot_on:
        rot_term = rotation_weight * (gs._rotation - init_rotation).pow(2).mean()
        metrics["rotation_anchor"] = rot_term.item()
        loss = rot_term if loss is None else loss + rot_term
    loss.backward()
