# Copyright (c) Meta Platforms, Inc. and affiliates.

"""Select, sample, project, and draw tracked point trajectories."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
from genia.core.utils.final_rendering import runtime as _runtime
from genia.core.utils.final_rendering.projection import project_r3
from genia.core.utils.final_rendering.runtime import log
from genia.core.utils.frame_key import FrameKey

if TYPE_CHECKING:
    from genia.core.utils.final_rendering.context import RenderContext
    from genia.core.utils.final_rendering.run import FinalRun



def track_colors(n: int) -> list[tuple[int, int, int]]:
    """``n`` visually distinct RGB triples (golden-angle hue walk)."""
    import colorsys
    return [
        tuple(int(255 * c) for c in colorsys.hsv_to_rgb((i * 0.618033988) % 1.0, 0.85, 1.0))
        for i in range(n)
    ]


def draw_tracks(rgba: np.ndarray, trails: np.ndarray, c2w: np.ndarray, K: np.ndarray,
                depth: np.ndarray | None, colors: list[tuple[int, int, int]],
                width: int = 2, head: int = 3, opacity: float = 1.0) -> np.ndarray:
    """Draw 3D point trails onto an RGBA frame, in place of a 3D line renderer.

    ``trails`` is ``(P, L, 3)`` -- one polyline per tracked point, oldest first,
    in the same R3 space the frame was rendered from.  When a ``depth`` buffer
    is given, vertices behind the rendered surface are dropped so trails go
    *behind* the object instead of always floating on top.  The trail fades
    towards its tail and ends in a filled dot at the current position.
    ``opacity`` scales the whole overlay (0.5 = half-transparent tracks).
    """
    from PIL import ImageDraw

    P, L, _ = trails.shape
    if P == 0 or L == 0:
        return rgba
    H, W = rgba.shape[:2]
    flat = project_r3(trails.reshape(-1, 3), c2w, K).reshape(P, L, 3)

    ui = np.clip(flat[..., 0].astype(int), 0, W - 1)
    vi = np.clip(flat[..., 1].astype(int), 0, H - 1)
    on_screen = (flat[..., 0] >= 0) & (flat[..., 0] < W) & \
                (flat[..., 1] >= 0) & (flat[..., 1] < H) & (flat[..., 2] > 0)
    if depth is not None:
        surf = depth[vi, ui]
        # depth == 0 is background (nothing rendered there) -> always visible.
        on_screen &= (surf <= 0) | (flat[..., 2] <= surf + 0.02 * np.abs(surf) + 1e-3)

    img = _runtime.Image.fromarray(rgba).convert("RGBA")
    layer = _runtime.Image.new("RGBA", (W, H), (0, 0, 0, 0))
    draw = ImageDraw.Draw(layer)
    for p in range(P):
        rgb = colors[p % len(colors)]
        for l in range(1, L):
            if not (on_screen[p, l] and on_screen[p, l - 1]):
                continue
            alpha = int(60 + 195 * (l / max(1, L - 1)))  # fade towards the tail
            draw.line(
                [(flat[p, l - 1, 0], flat[p, l - 1, 1]), (flat[p, l, 0], flat[p, l, 1])],
                fill=(*rgb, alpha), width=width,
            )
        if on_screen[p, L - 1] and head > 0:
            x, y = flat[p, L - 1, 0], flat[p, L - 1, 1]
            draw.ellipse([x - head, y - head, x + head, y + head], fill=(*rgb, 255))
    if opacity < 1.0:  # scale the whole overlay's alpha, keeping the fade shape
        r, g, b, a = layer.split()
        layer = _runtime.Image.merge("RGBA", (r, g, b, a.point(lambda v: int(v * opacity))))
    return np.asarray(_runtime.Image.alpha_composite(img, layer))


_TRACK_HEAD_PX = 3


def overlay_trails(frame: np.ndarray, trails: np.ndarray | None, upto: int,
                   c2w: np.ndarray, K: np.ndarray, depth: Any,
                   colors: list, opts: dict) -> np.ndarray:
    """Draw track trails up to frame index ``upto`` onto ``frame`` — the shared
    body of the ``orbit`` / ``synth_nvs`` overlay.  ``trails is None`` (overlay
    off) returns the frame untouched, so the depth read-back is skipped too."""
    if trails is None:
        return frame
    lo = 0 if not opts["track_trail"] else max(0, upto + 1 - opts["track_trail"])
    return draw_tracks(
        frame, trails[lo : upto + 1].transpose(1, 0, 2), c2w, K,
        depth.detach().cpu().numpy(), colors, width=opts["track_width"],
        head=opts.get("track_head", _TRACK_HEAD_PX), opacity=opts["track_opacity"],
    )


def farthest_point_indices(pts: np.ndarray, k: int) -> np.ndarray:
    """``k`` indices of ``pts`` (N,3) spread evenly through space by greedy
    farthest-point sampling.  Deterministic (seeded at index 0), so the same
    tracks are kept across frames and reruns.  Returns sorted indices."""
    n = len(pts)
    if k >= n:
        return np.arange(n)
    pts = np.asarray(pts, dtype=np.float64)
    chosen = [0]
    dist = np.linalg.norm(pts - pts[0], axis=1)
    for _ in range(1, k):
        i = int(dist.argmax())
        chosen.append(i)
        dist = np.minimum(dist, np.linalg.norm(pts - pts[i], axis=1))
    return np.array(sorted(chosen))


_ANCHOR_K = 64


_ANCHOR_SPARSITY = 2.5


def _anchor_keep_mask(traj: np.ndarray, pose_bounds) -> np.ndarray:
    """Which anchors stay ON the object, as a boolean mask over ``traj`` ``(T, P, 3)``.

    ``pose_bounds`` maps a trajectory row to the ``bounding_points`` of the geometry
    the caller will actually DRAW at that row -- an anchor sits on a canonical Gaussian
    or voxel, so a stray one lands wherever that asset has stray parts, and the
    anchors' own spread is no evidence about where the object is.

    An anchor must pass at EVERY row that has bounds, not just the reference one.  A
    noisy trajectory can start on the object and wander off, and the drawn dot is the
    trail's HEAD at the current frame -- so filtering on the reference frame alone
    would leave exactly those strays on screen.

    Falls back to the anchors' own bulk at the reference row
    (``rendering.bulk_keep_mask``) when the caller has no bounds to offer -- the
    renders that draw over a real camera, where the geometry is untrimmed and a stray
    dot is cosmetic rather than a framing error.
    """
    if not pose_bounds:
        # Imported here, not through `_lazy_imports`: this branch is the only part of
        # the track path that needs the rule, and a `--dry-run` never binds the GPU
        # globals.  Same reason as the scipy import below.
        import torch as _torch
        from genia.core.utils.rendering import bulk_keep_mask as _bulk_keep

        return _bulk_keep(
            _torch.as_tensor(np.ascontiguousarray(traj[0]))).numpy()
    from scipy.spatial import cKDTree

    keep = np.ones(traj.shape[1], dtype=bool)
    for t, bounds in pose_bounds.items():
        if bounds is None or t >= len(traj) or not keep.any():
            continue
        pts = np.asarray(bounds[0], np.float64)
        if not len(pts):
            continue
        # `k=[k]` asks for the k-th neighbour alone, and the whole anchor set is
        # queried rather than just the survivors: the threshold is the median over
        # all of them, so narrowing the set would move the bar.
        k = [min(_ANCHOR_K, len(pts))]
        d = cKDTree(pts).query(traj[t], k=k)[0][:, 0]
        keep &= d < _ANCHOR_SPARSITY * float(np.median(d))
    return keep if keep.any() else np.ones(traj.shape[1], dtype=bool)


def _track_trails(run: FinalRun, ctx: RenderContext, keys: list[FrameKey],
                  ref_key: FrameKey | None = None,
                  pose_bounds=None) -> np.ndarray | None:
    """``(len(keys), P, 3)`` track positions in ``compose(space="world")`` space.

    Reads :attr:`FinalRun.tracks` (posed R3 points).  ``ref_key=None`` keeps
    each frame's own posed position — for renders that keep root motion
    (``synth_nvs``).  A ``ref_key`` rebases every frame onto that key's pose,
    stripping root motion the same way the pinned Gaussians have it stripped
    (``orbit``).  Returns None when the run has no tracks, or none for the
    objects being rendered.
    """
    tracks = run.tracks
    if tracks is None:
        log("    (no 3D tracks on disk — rendering without them)")
        return None
    wanted = set(run.objects if ctx.objects is None else ctx.objects)
    per_obj: list[np.ndarray] = []
    for obj_idx in sorted(set(tracks.xyz) & wanted):
        obj = run.objects.get(obj_idx)
        if obj is None:
            continue
        xyz = tracks.xyz[obj_idx]                      # (T, P, 3)
        # EVERY anchor's whole trajectory first, then filter, then sample.  In that
        # order because both later steps depend on it: an anchor is judged by where it
        # goes, not only by where it starts, and farthest-point sampling MAXIMISES
        # spread -- it seeks out the extremes, which is exactly where a stray sits, so
        # sampling before filtering makes an outlier nearly certain to be among the
        # few drawn rather than merely possible.
        rows = []
        for k in keys:
            t = tracks.row_for(k.frame)
            if t is None:
                return None                            # tracks don't cover these frames
            rows.append(xyz[t] if ref_key is None
                        else run.rebase_points(obj, xyz[t], k, ref_key))
        traj = np.stack(rows)                          # (len(keys), P, 3)
        inside = np.flatnonzero(_anchor_keep_mask(traj, pose_bounds))
        if not len(inside):                            # degenerate — keep them all
            inside = np.arange(traj.shape[1])
        # How many to draw: the track_points cap, thinned by track_subsample.
        # farthest_point_indices clamps to the available count, so no upper bound
        # here even if track_subsample is set above 1.
        base = min(max(1, ctx.opts["track_points"]), len(inside))
        n_pts = max(1, round(base * ctx.opts["track_subsample"]))
        # Sampled on the reference frame's positions, so the kept tracks are spread
        # evenly through space rather than by anchor index.
        sel = inside[farthest_point_indices(traj[0][inside], n_pts)]
        per_obj.append(traj[:, sel])                   # (len(keys), n_pts, 3)
    if not per_obj:
        return None
    out = np.concatenate(per_obj, axis=1)
    log(f"    3D tracks: {out.shape[1]} point(s) over {out.shape[0]} frame(s)")
    return out
