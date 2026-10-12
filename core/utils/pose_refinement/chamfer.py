# Copyright (c) Meta Platforms, Inc. and affiliates.

"""Build and sample point clouds for geometric alignment losses."""

from __future__ import annotations

from typing import Any, Optional

import numpy as np
import torch
from genia.core.utils.config import CHAMFER_GT_TRIM_DEFAULT
from genia.core.utils.pose_refinement.loss_terms import _erode_bool_mask, depth_to_normals
from genia.core.utils.rendering import bulk_keep_mask


def trim_and_subsample(pts, trim_factor, max_points, seed=None, *,
                       return_index: bool = False):
    """Outlier trim then optional cap on an ``(M, 3)`` cloud -- shared by BOTH ICP clouds.

    Shared so the observed and the rendered cloud get the same post-processing from the
    same code, keeping the two sides of a Chamfer comparable.

    ``bulk_keep_mask`` is a median-centred radius cut at ``trim_factor * p95``, so it keeps
    at least 95% of any cloud by construction and cannot empty the target.  The empty guard
    is mandatory, not defensive: ``median`` and ``quantile`` both raise on an empty cloud,
    and an empty cloud is a REAL path here (a mask landing entirely on invalid depth, or a
    pose that renders nothing) which callers handle by SKIPPING the frame.

    ``max_points`` None / <= 0 means NO cap: every kept pixel enters the cloud.  The cap is
    a real time budget -- ``_chamfer_gt_to_model`` runs a KNN per frame per ICP iteration
    (~4800 calls a solve at 300 steps x 16 frames) -- and also a SAMPLING bias: 4000 points
    over a full-resolution silhouette is a few percent, drawn uniformly over PIXELS, so thin
    structure (legs, tails) is represented in proportion to its image area rather than its
    importance.  Uncapped removes that bias and pays for it in wall clock.

    Does NOT detach: the rendered cloud must stay differentiable w.r.t. the pose.  Both
    steps are boolean-mask/index selections, so gradients reach the surviving points and
    the GT caller detaches for itself.

    ``return_index`` additionally returns the COMPOSED index into the input, so a caller
    can carry a per-point attribute (the ICP's depth-map normals) through the identical
    selection instead of re-deriving it.  Re-deriving is not merely redundant, it is
    WRONG: ``_subsample_idx`` is a ``randperm`` prefix, so the cap PERMUTES as well as
    subsets and the surviving order is not recoverable from the masks alone.
    """
    import torch

    idx = (torch.arange(pts.shape[0], device=pts.device)
           if return_index else None)
    if trim_factor is not None and float(trim_factor) > 0 and pts.shape[0] > 0:
        keep = bulk_keep_mask(pts, float(trim_factor))
        pts = pts[keep]
        if idx is not None:
            idx = idx[keep]
    if max_points and pts.shape[0] > int(max_points):
        take = _subsample_idx(pts.shape[0], int(max_points), pts.device, seed)
        pts = pts[take]
        if idx is not None:
            idx = idx[take]
    return (pts, idx) if return_index else pts


def _chamfer_gt_points_for_object(
    pointmap: torch.Tensor,
    obj_mask: torch.Tensor,
    max_points: Optional[int],
    valid_mask: Optional[torch.Tensor] = None,
    seed=None,
    trim_factor: Optional[float] = CHAMFER_GT_TRIM_DEFAULT,
    return_index: bool = False,
):
    """Object's visible GT surface points from the camera-space depth pointmap.

    ``mask ∩ valid ∩ trimmed ∩ finite-pointmap`` pixels → ``(M, 3)``, subsampled to
    ``max_points`` and detached (a fixed alignment target, no gradient).
    ``valid_mask`` (depth-reliability) is folded in so finite-but-unreliable
    depth (dense predictors flag it separately from NaN) doesn't pollute the
    target.  ``pointmap`` is ``(H, W, 3)`` in the same camera-space R3 frame the
    model is posed into.

    ``seed`` makes the subsample REPRODUCIBLE.  The cap is a real time budget --
    ``_chamfer_gt_to_model`` runs a KNN per frame per ICP iteration -- but the randomness
    only needs to avoid aliasing with raster order, which a seeded draw does just as well.
    Unseeded, two runs of the SAME config would optimise against different targets.  Pass
    something stable per (object, frame, view); ``None`` draws unseeded for callers that
    want a fresh sample.

    ``trim_factor`` drops BACKGROUND points, which is what the two masks above cannot do:
    a segmentation mask that bleeds onto the wall behind the subject contributes pixels
    whose depth is perfectly VALID and perfectly FINITE, just wrong -- ``valid_mask``
    carries ``recon_confidence_min`` / ``recon_mask_edges``, and both are about whether a
    depth measurement is TRUSTWORTHY, not about whether it belongs to this object.  The
    Chamfer that consumes this cloud is a mean of SQUARED distances, so a stray 100x
    further than the bulk carries 10,000x, and a handful of such points can dominate the
    gradient and inflate the object.

    The cut is :func:`rendering.bulk_keep_mask` -- a median-centred radius at
    ``factor * p95`` -- reused rather than reimplemented, and the only one of
    ``trim_outliers``' four cuts that needs no opacity or scale, which a depth cloud has
    not got.  The default factor (3.0) is looser than that function's own 1.5, so clean
    GT-depth clouds pass untouched.  Cleaning the cloud rather than reweighting the loss
    keeps the objective exactly L2, runs ONCE per frame instead of once per iteration (the
    observation is loop-invariant), and makes both the `icp_clouds/` dumps and the logged
    residual show what the solver optimises.

    Assumes the strays are a MINORITY: ``p95`` is by construction blind to a blob holding
    more than 5% of the masked pixels.  This is outlier rejection, NOT segmentation
    repair.  ``None`` (or <= 0) disables the trim.
    """
    sel = obj_mask.bool() & torch.isfinite(pointmap).all(dim=-1)
    if valid_mask is not None:
        sel = sel & valid_mask.bool()
    pts = pointmap[sel]
    if return_index:
        # Row-major `sel.nonzero()` order, which is exactly the order `pointmap[sel]`
        # produced, so composing the trim/cap index below lands each surviving point on
        # the pixel it came from.  The ICP's depth-map normals ride on this.
        pix = sel.nonzero(as_tuple=False)
        pts, idx = trim_and_subsample(pts, trim_factor, max_points, seed=seed,
                                      return_index=True)
        return pts.detach(), pix[idx]
    # BEFORE the subsample, so the cap spends its whole budget on real object points --
    # and the guard is mandatory, not defensive: `bulk_keep_mask` raises on an empty cloud
    # (`median` and `quantile` both do), while an empty target is a REAL path here (a mask
    # landing entirely on invalid depth) that `icp_frame_observation` handles by SKIPPING
    # the frame.  Unguarded this turns a handled skip into a crash on exactly the
    # bad-depth scenes it exists to rescue.
    return trim_and_subsample(pts, trim_factor, max_points, seed=seed).detach()


def _chamfer_gt_points_from_depth(
    gt_depth: np.ndarray,
    K_matrix: np.ndarray,
    obj_mask: Any,
    valid_mask: Any,
    max_points: Optional[int],
    device: torch.device,
    seed=None,
    trim_factor: Optional[float] = CHAMFER_GT_TRIM_DEFAULT,
    with_normals: bool = False,
):
    """Unproject GT depth → camera-space pointmap, then the object's visible
    surface points (``_chamfer_gt_points_for_object``).  Shared by the paths
    that carry a depth map rather than a precomputed pointmap.

    ``with_normals`` returns ``(points, normals)``: the SURFACE NORMAL at each kept point,
    taken from the depth map's own grid (``depth_to_normals``) rather than by PCA over the
    scattered cloud.  The grid is far better conditioned, and the points are carried
    through the identical trim/cap selection so a normal cannot land on the wrong point.
    A point whose normal is unreliable -- the 1px border, an eroded mask edge, a
    degenerate patch -- comes back as the ZERO vector, which the ICP's normal-compatibility
    gate rejects on its own; that is the intended behaviour, not a hole to fill.
    """
    from genia.core.utils.depth import depth_to_pointmap
    pm = torch.from_numpy(depth_to_pointmap(gt_depth, K_matrix)).float().to(device)
    om = obj_mask if isinstance(obj_mask, torch.Tensor) else torch.from_numpy(np.asarray(obj_mask)).to(device)
    vm = None
    if valid_mask is not None:
        vm = valid_mask if isinstance(valid_mask, torch.Tensor) else torch.from_numpy(np.asarray(valid_mask)).to(device)
    if not with_normals:
        return _chamfer_gt_points_for_object(pm, om, max_points, valid_mask=vm,
                                             seed=seed, trim_factor=trim_factor)
    pts, pix = _chamfer_gt_points_for_object(
        pm, om, max_points, valid_mask=vm, seed=seed, trim_factor=trim_factor,
        return_index=True)
    # `depth_to_normals` needs a DENSE FINITE map: NaN spreads to the 4-neighbours through
    # the central differences and masking does not clean it (`NaN * 0 = NaN`).  Hence
    # `nan_to_num` + a 1px erode; the erode is what drops the contaminated ring rather
    # than trusting the mask alone.
    K_t = torch.as_tensor(np.asarray(K_matrix), dtype=torch.float32, device=device)
    valid = om.bool() if vm is None else (om.bool() & vm.bool())
    nmap = depth_to_normals(
        torch.nan_to_num(torch.as_tensor(np.asarray(gt_depth), dtype=torch.float32,
                                         device=device)),
        K_t, mask=_erode_bool_mask(valid, 1))
    return pts, nmap[pix[:, 0], pix[:, 1]].detach()


def _subsample_idx(n_points: int, max_points: int, device, seed=None):
    """``max_points`` indices out of ``n_points``, reproducibly when ``seed`` is given.

    One definition for both Chamfer draws (the GT target and the model subset), because
    they have the same requirement: bound the KNN cost without aliasing against the raster
    order the points arrive in, and give the SAME answer on a re-run so two configs can be
    compared at all.
    """
    g = None
    if seed is not None:
        g = torch.Generator(device="cpu").manual_seed(int(seed) & 0x7FFFFFFF)
        return torch.randperm(n_points, generator=g)[:max_points].to(device)
    return torch.randperm(n_points, device=device)[:max_points]


def _chamfer_model_subsample_idx(
    n_points: int, max_points: int, device: torch.device, seed=None
) -> Optional[torch.Tensor]:
    """Fixed random subset of model-point indices (stable across iterations), or ``None``
    to use all points.  Precompute once per object so the Chamfer target set doesn't jitter
    every step -- and pass ``seed`` so it does not jitter between RUNS either."""
    if n_points > max_points:
        return _subsample_idx(n_points, max_points, device, seed)
    return None


def _chamfer_gt_to_model(
    gt_points: torch.Tensor,
    model_points: torch.Tensor,
    model_sub_idx: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """One-directional Chamfer: mean squared distance from each GT point to its
    nearest posed-model point.

    GT→model only — single-view GT depth is the visible front surface, so each
    observed point pulls toward its nearest model point and the model's unseen
    back is never penalized.  Differentiable w.r.t. ``model_points`` (hence the
    pose).  Returns a scalar (0 if either set is empty).

    A mean of SQUARES, so a target point 100x further than the bulk carries 10,000x --
    which is why the outliers are removed when the target CLOUD is built
    (``_chamfer_gt_points_for_object``'s ``trim_factor``) rather than reweighted here.
    """
    if gt_points.shape[0] == 0 or model_points.shape[0] == 0:
        return torch.zeros((), device=model_points.device)
    mp = model_points if model_sub_idx is None else model_points[model_sub_idx]
    # knn_points requires matching float dtypes; GT is float32 (see builders).
    mp = mp.float()
    from genia.core.ops import knn_points
    knn = knn_points(gt_points.unsqueeze(0), mp.unsqueeze(0), K=1)
    # knn.dists is squared distance (B, M, 1).
    return knn.dists.squeeze(0).squeeze(-1).mean()


def rendered_surface_keep(depth, alpha, K_t, *, alpha_threshold: float = 0.5):
    """The pixels that enter the rendered cloud, as a boolean ``(H, W)`` mask.

    Split out of :func:`rendered_surface_cloud` so an INSPECTION site (the `icp_clouds/`
    depth dump) can show exactly the pixels the loss consumed without restating
    ``alpha > 0.5`` for itself.  That restatement is the specific failure this file already
    warns about -- "a second ``alpha > 0.5`` at an inspection site would let the picture
    drift from the thing being optimised" -- so the mask is shared rather than recomputed.

    See :func:`rendered_surface_cloud` for why the alpha threshold cannot be relaxed to a
    finiteness test (gsplat returns a finite ``0.0`` at uncovered pixels).
    """
    return (alpha > alpha_threshold) & torch.isfinite(depth) & (depth > 0)


def rendered_surface_cloud(
    depth: torch.Tensor,
    alpha: torch.Tensor,
    uu: torch.Tensor,
    vv: torch.Tensor,
    K_t: torch.Tensor,
    *,
    alpha_threshold: float = 0.5,
    trim_factor: Optional[float] = None,
    return_index: bool = False,
):
    """The rendered object's visible surface as a camera-space cloud, or ``None``.

    Built by the SAME steps as the observed cloud (``_chamfer_gt_points_for_object``) --
    coverage mask, one shared unprojection
    (``depth.unproject_z_depth``), then ``trim_and_subsample`` -- so the only asymmetry
    between the two sides of the ICP Chamfer is which mask the caller supplies: the object
    segmentation there, rendered coverage here.

    THE definition of "what the ICP term matches against", so the ICP solver and anything
    inspecting that match (the debug cloud dump) read one implementation.  A
    second ``alpha > 0.5`` at an inspection site would let the picture drift from the thing
    being optimised.

    ``keep = alpha > alpha_threshold`` cannot be relaxed to a finiteness test.  gsplat
    renders expected depth as ``accum_depth / alpha.clamp(min=1e-10)`` (``RGB+ED``, see
    ``rendering.render_gaussian_params``), so an UNCOVERED pixel comes back as exactly
    ``0.0`` -- finite, not NaN.  Unprojecting ``z = 0`` puts the point at the CAMERA CENTRE,
    and in a mean-of-SQUARES Chamfer a cloud of those swamps the objective -- worse the more
    wrong the pose is, since more of the frame goes uncovered.  The threshold is the only
    thing rejecting them.

    ``keep`` is a HARD mask throughout -- no gradient reaches the silhouette -- while the
    kept pixels' depth stays differentiable, hence so does the pose.

    Note the point COUNT grows with the object's rendered area: one point per kept pixel,
    not a fixed budget like the GT side's ``max_gt_points``.
    """
    from genia.core.utils.depth import unproject_z_depth

    keep = rendered_surface_keep(depth, alpha, K_t, alpha_threshold=alpha_threshold)
    if not bool(keep.any()):
        return (None, None, keep) if return_index else None
    if return_index:
        # The cloud, its index into the KEEP-ordered unprojection, AND the keep mask --
        # so a caller holding a dense per-pixel map writes `dense[keep][idx]` and lands on
        # the right point even under a trim, without recomputing `keep`.
        pts, idx = trim_and_subsample(unproject_z_depth(depth, keep, K_t, uu, vv),
                                      trim_factor, None, return_index=True)
        return pts, idx, keep
    return trim_and_subsample(unproject_z_depth(depth, keep, K_t, uu, vv),
                              trim_factor, None)
