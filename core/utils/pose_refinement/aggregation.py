# Copyright (c) Meta Platforms, Inc. and affiliates.

"""Weight frame losses and report nonfinite terms consistently."""

from __future__ import annotations

from typing import Dict

import torch
from genia.core.utils.config import LossConfig

_NONFINITE_TERM_REPORTED: set = set()


def _term_groups(losses: LossConfig) -> "list[tuple[str, float, list]]":
    """``(metric key, outer weight, [(losses_dict key, inner weight), ...])``.

    THE single place naming which loss terms exist and what each is worth: the
    summed expression, the drop list, and the per-term breakdown all read it.

    A GROUP is exactly one entry of FINETUNE's ``metrics`` dict, which is what
    makes the two granularities fall out of one table:

    * the SUM uses the nesting ``outer * (Σ inner_w * val)``.  Do NOT
      "simplify" this by folding ``outer`` into the inner weights: float
      addition is not associative, so with a non-trivial master weight AND
      non-trivial sub-weights folding changes the number.
    * the DROP test is per inner term on its full contribution
      ``outer * inner_w * val``, so one bad sub-term leaves its siblings intact
      and a zero weight (outer OR inner) cannot smuggle a NaN in via ``0.0*nan``.
    """
    return [
        ("rgb",        1.0, [("rgb_loss",        losses.rgb_weight)]),
        ("ssim",       1.0, [("ssim_loss",       losses.rgb_ssim_weight)]),
        ("silhouette", losses.silhouette_weight, [
            ("silhouette_com", losses.silhouette_com_weight),
            ("silhouette_sdt", losses.silhouette_sdt_weight),
            ("silhouette_iou", losses.silhouette_iou_weight)]),
        ("perceptual", 1.0, [("perceptual_loss", losses.perceptual_weight)]),
        ("depth",      1.0, [("depth_loss",      losses.depth_weight)]),
        ("normals",    1.0, [("normals_loss",    losses.normals_weight)]),
    ]


def _group_contribution(losses_dict, outer, inner, drop=()):
    """``outer * (Σ inner_w * val)`` over the members not in ``drop``.

    Returns ``None`` when every member was dropped, so the caller can leave the
    group out of the sum entirely rather than adding a zero.  ``outer == 1.0``
    skips the multiply: exact either way, but it avoids adding a node to the
    autograd graph per single-member group.
    """
    acc = None
    for key, w in inner:
        if key in drop:
            continue
        term = w * losses_dict[key]
        acc = term if acc is None else acc + term
    if acc is None:
        return None
    return acc if outer == 1.0 else outer * acc


def _aggregate_frame_loss(
    losses_dict: Dict[str, torch.Tensor],
    losses: LossConfig,
    report: "list | None" = None,
    terms_out: "dict | None" = None,
) -> torch.Tensor:
    """Sum the weighted loss terms from ``_compute_frame_loss`` output.

    Mirrors the scalar aggregation used in ``refine_pose_for_frame`` so that
    MV shared-pose optimization uses the same loss surface per frame.

    A NON-FINITE TERM IS DROPPED, NOT PROPAGATED.  The terms are summed, so one
    bad term would make the frame total NaN, and from there the whole chunk and
    the whole step: callers gate their backward on ``loss.item() >= 1e-10``,
    which is False for NaN, so ``torch.autograd.grad`` would never be called and
    EVERY term of EVERY frame in that step would silently lose its gradient.
    FINETUNE calls ``.backward()`` with no finiteness check at all, so a NaN
    would reach ``.grad`` and from there Adam's moments, which never recover.

    Parameters
    ----------
    report
        If given, receives the names of dropped terms, so a caller can attach
        richer context (inputs, ranges) to its own diagnostic.
    terms_out
        If given, receives the per-term weighted tensors at METRIC granularity
        (one entry per group in :func:`_term_groups`, silhouette pre-combined).
        A dropped term reports ``0.0`` here, never NaN: FINETUNE sums
        ``metrics`` to rebuild its returned total, so a NaN would re-poison the
        return value even though the gradient was saved -- and ``0.0`` is the
        honest figure, being what the term actually contributed.
    """
    groups = _term_groups(losses)

    def _assemble(drop=()):
        contribs, total = {}, None
        for key, outer, inner in groups:
            c = _group_contribution(losses_dict, outer, inner, drop)
            contribs[key] = c
            if c is not None:
                total = c if total is None else total + c
        return contribs, total

    contribs, total = _assemble()
    if total is None:                       # no terms at all; nothing to sum
        total = torch.zeros((), device=_any_device(losses_dict))

    def _fill(contribs):
        if terms_out is None:
            return
        zero = torch.zeros((), device=total.device, dtype=total.dtype)
        terms_out.update({k: (zero if c is None else c)
                          for k, c in contribs.items()})

    # Populated BEFORE the fast-path return: consumers need the breakdown on
    # every call, not only when something is non-finite.  Returning early
    # without it would empty FINETUNE's metrics on exactly the HEALTHY path.
    _fill(contribs)

    # ONE sync on the healthy path.  Only when it fails do we pay a test per
    # term to find out which one -- ten `__bool__` calls per frame is not a
    # price worth paying when nothing is wrong.
    if torch.isfinite(total):
        return total

    dropped = [k for _, outer, inner in groups for k, w in inner
               if not torch.isfinite(outer * w * losses_dict[k]).all()]
    if report is not None:
        report.extend(dropped)
    for name in dropped:
        if name not in _NONFINITE_TERM_REPORTED:
            _NONFINITE_TERM_REPORTED.add(name)
            print(f"    [loss] NON-FINITE term '{name}' DROPPED from the frame "
                  f"total (reported once per term per process). The remaining "
                  f"terms keep their gradient.")
    contribs, kept_total = _assemble(drop=set(dropped))
    _fill(contribs)
    if kept_total is None:
        return torch.zeros((), device=total.device, dtype=total.dtype)
    return kept_total


def _any_device(losses_dict: Dict[str, torch.Tensor]) -> torch.device:
    for v in losses_dict.values():
        if torch.is_tensor(v):
            return v.device
    return torch.device("cpu")
