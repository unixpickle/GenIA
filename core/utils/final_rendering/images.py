# Copyright (c) Meta Platforms, Inc. and affiliates.

"""Convert render outputs, composite alpha, and crop image sequences."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from genia.core.utils.final_rendering import runtime as _runtime
from genia.core.utils.final_rendering.runtime import log
from genia.core.utils.final_rendering.tracks import _TRACK_HEAD_PX

if TYPE_CHECKING:
    from genia.core.utils.final_rendering.context import RenderContext



def to_rgba(rgb, alpha) -> np.ndarray:
    """``(H,W,3)+(H,W)`` float tensors -> RGBA uint8 (coverage in alpha).

    Packed and cast on device: one uint8 transfer instead of two float32 ones.
    Truncating (not rounding) keeps this bit-identical to the pipeline's own
    ``(x * 255).astype(uint8)`` in ``save_canonical_renders_perframe``.
    """
    rgba = _runtime.torch.cat([rgb, alpha[..., None]], dim=-1)
    return rgba.mul(255).to(_runtime.torch.uint8).cpu().numpy()


def flatten_alpha(arr: np.ndarray, bg: np.ndarray) -> np.ndarray:
    """RGBA uint8 -> RGB uint8, compositing the transparent part onto ``bg``."""
    bg_u8 = np.asarray(bg, np.float32) * 255.0
    a = arr[..., 3:4].astype(np.float32) / 255.0
    return (arr[..., :3].astype(np.float32) * a + bg_u8 * (1.0 - a)).astype(np.uint8)


def alpha_bbox(a: np.ndarray) -> tuple[int, int, int, int] | None:
    """``(x0, y0, x1, y1)`` of what is opaque enough to count as foreground, or None."""
    ys, xs = np.where(a > 8)
    return (xs.min(), ys.min(), xs.max(), ys.max()) if len(xs) else None


def sequence_crop_box(boxes, pad: float, H: int, W: int) -> tuple[int, int, int, int]:
    """``(x0, y0, x1, y1)``: one crop covering every box in ``boxes``, at the frame's
    aspect ratio.

    ``boxes`` are ``(x0, y0, x1, y1)`` foreground extents (see :func:`alpha_bbox`), one
    per frame of a sequence, or one per CELL of a figure row when the caller wants a box
    every run in the row shares. Their union, padded by ``pad`` x its longer side,
    then GROWN (never squeezed) to ``W/H`` so the crop rescales without distorting, and
    shifted -- not shrunk -- back inside the frame when that growth overhangs an edge.
    One box for every frame is the point: a per-frame box would make the object jitter
    in scale across a row of tiles meant to differ only in timestep.

    An empty ``boxes`` (nothing rendered at all) gives the full frame back.
    """
    boxes = list(boxes)
    if not boxes:
        return 0, 0, W - 1, H - 1
    x0 = min(b[0] for b in boxes); y0 = min(b[1] for b in boxes)
    x1 = max(b[2] for b in boxes); y1 = max(b[3] for b in boxes)
    cw, ch = x1 - x0 + 1, y1 - y0 + 1
    m = pad * max(cw, ch)
    cw, ch = cw + 2 * m, ch + 2 * m
    ar = W / H
    if cw / ch < ar:                      # too tall -> widen
        cw = ch * ar
    else:                                 # too wide -> heighten
        ch = cw / ar
    s = min(1.0, W / cw, H / ch)          # a box bigger than the frame scales down whole
    cw, ch = cw * s, ch * s
    cx, cy = (x0 + x1 + 1) / 2, (y0 + y1 + 1) / 2
    nx = min(max(cx - cw / 2, 0), W - cw)
    ny = min(max(cy - ch / 2, 0), H - ch)
    return int(round(nx)), int(round(ny)), int(round(nx + cw)) - 1, int(round(ny + ch)) - 1


def fit_to_native(frame: np.ndarray, W: int, H: int) -> np.ndarray:
    """Downscale a cropped frame back to the run's own ``W x H``. NEVER upscales.

    Supersampling exists to feed the crop, not to ship a bigger file: rendering at 2x
    and resolving back down is ordinary supersampling, so the tile lands at the run's
    resolution with the aliasing of a 2x render rather than a 1x one. Output is
    therefore always <= 1.0x native, and a crop already at or below it (a small object,
    where the box is tighter than the supersample covers) is passed through untouched
    rather than blown up to fill the size.

    The crop carries the frame's aspect ratio by construction, so resizing to exactly
    ``(W, H)`` is not a distortion -- and it removes the sub-pixel aspect differences
    that integer rounding leaves between one sequence's box and another's, which would
    otherwise print as slightly unequal tile heights across a figure row.
    """
    if frame.shape[1] <= W:
        return frame
    return np.asarray(_runtime.Image.fromarray(frame).resize((W, H), _runtime.Image.LANCZOS))


def _supersampled_size(ctx: RenderContext, trails) -> tuple[int, int, dict]:
    """``(H, W, draw_opts)`` to render at, for this render's ``supersample`` factor.

    Rendering above native and resolving back down in :func:`fit_to_native` is ordinary
    supersampling: the tile ships at the run's own resolution with the aliasing of a 2x
    render.  Track width and head scale with the factor, or the trails would come out
    proportionally thinner once the frame is resolved down.
    """
    ss = float(ctx.opts.get("supersample", 1.0))
    if ss == 1.0:
        return ctx.H, ctx.W, ctx.opts
    H, W = max(1, round(ctx.H * ss)), max(1, round(ctx.W * ss))
    log(f"    supersample {ss:g}x: rendering {W}x{H} (native {ctx.W}x{ctx.H})")
    if trails is None:
        return H, W, ctx.opts
    return H, W, {**ctx.opts,
                  "track_width": max(1, round(ctx.opts["track_width"] * ss)),
                  "track_head": max(1, round(_TRACK_HEAD_PX * ss))}


def _write_cropped(ctx: RenderContext, pending: list, H: int, W: int) -> None:
    """Fit ONE box to every held frame, crop to it, resolve to native, write.

    The box is fitted to the alpha AS DRAWN, trails included -- fitting it to the bare
    silhouette would slice the tails off the very overlay these renders exist for.
    """
    if not pending:
        return
    x0, y0, x1, y1 = sequence_crop_box(
        [b for b in (alpha_bbox(f[..., 3]) for _, f in pending) if b],
        ctx.opts["crop_pad"], H, W)
    cw, ch = x1 - x0 + 1, y1 - y0 + 1
    out_w, out_h = (ctx.W, ctx.H) if cw > ctx.W else (cw, ch)
    log(f"    crop: {cw}x{ch} of {W}x{H} at ({x0},{y0}) — "
        f"{cw * ch / (W * H):.0%} of the frame, {cw / ctx.W:.2f}x native "
        f"-> written at {out_w}x{out_h}")
    for out, frame in pending:
        ctx.write_png(out, fit_to_native(frame[y0:y1 + 1, x0:x1 + 1], ctx.W, ctx.H))
