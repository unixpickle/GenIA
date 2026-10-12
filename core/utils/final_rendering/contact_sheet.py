# Copyright (c) Meta Platforms, Inc. and affiliates.

"""Build contact sheets from images already written by the pipeline."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import numpy as np
from genia.core.utils.final_rendering import runtime as _runtime
from genia.core.utils.final_rendering.images import flatten_alpha
from genia.core.utils.final_rendering.registry import Option, renderer
from genia.core.utils.final_rendering.runtime import log

if TYPE_CHECKING:
    from genia.core.utils.final_rendering.context import RenderContext
    from genia.core.utils.final_rendering.run import FinalRun



@renderer(
    "contact_sheet",
    "Tile PNGs the run already exported (renders_train/renders_test/...) into a grid.",
    options={
        "source": Option("str", "renders_train", "subfolder of final/ to tile"),
        "cols": Option("int", 0, "columns (0 = square-ish)"),
        "cell": Option("int", 256, "cell size in pixels (longest side; aspect preserved)"),
        "limit": Option("int", 64, "max images (0 = all)"),
    },
    needs_gpu=False,
)
def render_contact_sheet(run: FinalRun, ctx: RenderContext) -> None:
    out = ctx.path(f"{ctx.opts['source'].replace('/', '_')}.png")
    if not ctx.wants(out):
        if not ctx.dry_run:
            log(f"    {out.name} exists (pass --overwrite to redo it)")
        return
    src = run.dir / ctx.opts["source"]
    paths = sorted(src.glob("**/*.png"))
    if ctx.opts["limit"]:
        paths = paths[: ctx.opts["limit"]]
    if not paths:
        log(f"    ! no PNGs under {src}, skipping")
        return

    cell = ctx.opts["cell"]
    cols = ctx.opts["cols"] or max(1, math.ceil(math.sqrt(len(paths))))
    rows = math.ceil(len(paths) / cols)
    # Cell box follows the source aspect (a run's renders share a resolution),
    # so nothing is squished; `cell` is the longest side.
    with _runtime.Image.open(paths[0]) as im0:
        w0, h0 = im0.size
    scale = cell / max(w0, h0)
    cw, ch = max(1, round(w0 * scale)), max(1, round(h0 * scale))
    bg = tuple((ctx.bg * 255).astype(np.uint8).tolist())
    sheet = _runtime.Image.new("RGB", (cols * cw, rows * ch), bg)
    for i, p in enumerate(paths):
        with _runtime.Image.open(p) as im:
            tile = _runtime.Image.fromarray(flatten_alpha(np.asarray(im.convert("RGBA")), ctx.bg))
        # Fit within the cell box preserving aspect, centred (letterboxes a
        # stray odd-sized image rather than distorting it).
        tile.thumbnail((cw, ch), _runtime.Image.LANCZOS)
        ox = (i % cols) * cw + (cw - tile.width) // 2
        oy = (i // cols) * ch + (ch - tile.height) // 2
        sheet.paste(tile, (ox, oy))
    out.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(out)
    ctx.written.append(out)
