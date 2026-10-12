# Copyright (c) Meta Platforms, Inc. and affiliates.

"""Render training, synthesized, and held-out views with optional tracks."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from genia.core.utils.final_rendering.cameras import resolve_test_cameras
from genia.core.utils.final_rendering.images import _supersampled_size, _write_cropped, to_rgba
from genia.core.utils.final_rendering.options import (
    _FIGURE_CROP_OPTIONS,
    _TRACK_DRAW_OPTIONS,
    _TRACK_OPTIONS,
    _TRAIN_CAMERA_OPTIONS,
)
from genia.core.utils.final_rendering.projection import render_cloud, render_mesh, scale_intrinsics
from genia.core.utils.final_rendering.registry import Option, renderer
from genia.core.utils.final_rendering.runtime import log
from genia.core.utils.final_rendering.tracks import _track_trails, overlay_trails, track_colors
from genia.core.utils.frame_key import frame_key_stem

if TYPE_CHECKING:
    from genia.core.utils.final_rendering.context import RenderContext
    from genia.core.utils.final_rendering.run import FinalRun



def _render_from_train_cameras(run: FinalRun, ctx: RenderContext, *,
                               tracks: bool) -> None:
    """Re-render the posed scene from each dataset (input/train) camera.

    Shared body of the two renders below.  ``tracks`` is the ONLY difference, and it
    is a property of the render, not a knob on a shared one: the output dir is
    ``viz/<render name>``, so an annotated frame must come from a differently-named
    render or it lands in the clean one's folder.
    """
    if not run.has_renderable_geometry:
        log("    ! run has no gaussians or meshes — nothing to re-render")
        return
    use_mesh = run.prefers_mesh
    space = ctx.opts["space"]
    if use_mesh and space != "world":
        log("    (mesh renders are world-space only; ignoring space=camera)")
        space = "world"

    # A dynamic run whose stored geometry never changes (constant poses, no
    # per-frame asset) would emit identical frames -- the deformation is only in
    # the run's own renders_train/. Say so instead of implying it is static.
    per_frame_asset = run.has_perframe_meshes if use_mesh else run.has_perframe_gaussians
    per_frame = run.poses_vary or per_frame_asset
    if run.is_dynamic and not per_frame:
        log("    ! stored geometry is static across frames — every re-render "
            "is identical; see the method's own output with "
            "`-r contact_sheet -O source=renders_train`")

    # Tracks live in `compose(space="world")` space, so a camera-space render has
    # nothing to project them against -- the same reason `orbit` rebases them.
    # Otherwise draw them only when the run shipped a tracks_2d.npz.  A per-frame
    # prediction with no cross-frame correspondence writes none and gets a bare
    # render -- still written, and cropped identically, so a figure row can mix
    # the two.
    show_tracks = tracks and ctx.opts["tracks"] and space == "world" and run.tracks is not None
    if tracks and ctx.opts["tracks"] and not show_tracks:
        log("    (skipping the 3D track overlay: tracks are world-space)" if run.tracks
            is not None else "    (no 3D tracks on disk — rendering without them)")
    trails = _track_trails(run, ctx, ctx.frame_keys) if show_tracks else None
    colors = track_colors(trails.shape[1]) if trails is not None else []

    # Cropping needs the whole sequence before it can write ANY frame, so frames are
    # held in memory (~10 MB for 16 at 518x294) and `wants` is ignored: a partial
    # re-render would fit the box to a subset and silently reframe the rest.
    crop = ctx.opts.get("crop", False) and not ctx.dry_run
    H, W, draw_opts = _supersampled_size(ctx, trails)
    pending: list = []

    for i, key in enumerate(ctx.frame_keys):
        cam = run.camera(key)
        if cam is None or cam.K is None:
            log(f"    ! no camera with intrinsics for {key}, skipping")
            continue
        out = ctx.path(f"{frame_key_stem(key)}.png")
        if not ctx.wants(out) and not crop:
            continue
        if use_mesh:
            geom = run.compose_mesh(key, device=ctx.device, objects=ctx.objects)
        else:
            geom = run.compose(key, device=ctx.device, space=space,
                               objects=ctx.objects, use_perframe=ctx.opts["perframe"])
        if geom is None:
            continue
        c2w = cam.c2w if space == "world" else np.eye(4, dtype=np.float32)
        K = scale_intrinsics(cam.K, ctx.src_hw, (H, W))
        render = render_mesh if use_mesh else render_cloud
        rgb, alpha, depth = render(geom, c2w, K, H, W, ctx.bg)
        frame = to_rgba(rgb, alpha)
        if trails is not None:
            frame = overlay_trails(frame, trails, i, c2w, K, depth, colors, draw_opts)
        if crop:
            pending.append((out, frame))   # held: the box needs every frame first
        else:
            ctx.write_png(out, frame)

    _write_cropped(ctx, pending, H, W)

    if ctx.opts["video"]:
        # Named for the render, which is also the folder -- so the clean and the
        # annotated clip cannot collide.
        ctx.write_per_view_videos(ctx.out_dir.name, ctx.frame_keys)


@renderer(
    "train_views",
    "Re-render the posed scene from each dataset (input/train) camera. Clean "
    "pixels, nothing drawn on top -- see `track_overlay` for the annotated twin.",
    options=dict(_TRAIN_CAMERA_OPTIONS),
)
def render_train_views(run: FinalRun, ctx: RenderContext) -> None:
    _render_from_train_cameras(run, ctx, tracks=False)


@renderer(
    "track_overlay",
    "The train-camera render with the run's 3D tracks drawn over it, into "
    "viz/track_overlays/train_views/. A FIGURE asset: annotated and supersampled, so "
    "it is deliberately kept out of `train_views` and away from every dir an evaluation "
    "reads. A run with no tracks_2d.npz still renders -- same viewpoint and resolution, "
    "just nothing drawn on top, so the pair can be cropped to one box.",
    options={
        **_TRAIN_CAMERA_OPTIONS,
        **_TRACK_OPTIONS,
        **_FIGURE_CROP_OPTIONS,
    },
    out_sub="track_overlays/train_views",
)
def render_track_overlay(run: FinalRun, ctx: RenderContext) -> None:
    _render_from_train_cameras(run, ctx, tracks=True)


@renderer(
    "synth_nvs",
    "Per-timestamp synthesized novel views: the world-placed scene (root motion "
    "KEPT) seen from FOUR fixed offsets off each frame's train camera (\u00b1azimuth, "
    "\u00b1elevation), at the train intrinsics. Reproduces the pipeline's "
    "renders_synth_nvs/ off disk, filenames included.",
    options={
        "azimuth": Option("float", 30.0, "horizontal magnitude, deg (two views at \u00b1this)"),
        "azimuth_step": Option("float", 0.0, "extra azimuth per timestamp (0 = fixed viewpoint)"),
        "elevation": Option("float", 30.0, "vertical magnitude, deg (two views at \u00b1this)"),
        "video": Option("bool", True, "also encode the per-timestamp frames to mp4"),
        **_TRACK_OPTIONS,
    },
)
def render_synth_nvs(run: FinalRun, ctx: RenderContext) -> None:
    # Unlike `orbit` (one pinned timestamp swept through azimuth), this steps the
    # timestamp and keeps root motion — the object moves through the scene while
    # the viewpoint holds a fixed offset off the train camera, at the train K.
    if not run.has_renderable_geometry:
        log("    ! run has no gaussians or meshes — nothing to re-render")
        return
    from genia.core.utils.eval_assets_export import SYNTH_NVS_OFFSETS, _synth_orbit_c2w

    opts = ctx.opts
    az0, az_step, el = opts["azimuth"], opts["azimuth_step"], opts["elevation"]
    use_mesh = run.prefers_mesh
    render = render_mesh if use_mesh else render_cloud
    keys = ctx.frame_keys

    # Tracks are stored posed (root motion kept), which is exactly this render's
    # space — so no rebasing (ref_key=None), unlike `orbit`. Draw them only when
    # the run actually shipped a tracks_2d.npz: per-frame predictions with no
    # cross-frame correspondence don't write one — their per-frame gaussians are
    # independent shapes, not tracks.
    show_tracks = opts["tracks"] and run.tracks is not None
    trails = _track_trails(run, ctx, keys) if show_tracks else None
    colors = track_colors(trails.shape[1]) if trails is not None else []

    # This render REPRODUCES the pipeline's renders_synth_nvs/ off disk: native
    # resolution, uncropped, all four tags. `track_overlay_nvs` is the figure variant.

    for i, key in enumerate(keys):
        cam = run.camera(key)
        if cam is None or cam.K is None:
            log(f"    ! no camera with intrinsics for {key}, skipping")
            continue
        # `wants` RECORDS a skipped path, so probe each output exactly once —
        # asking twice would double-count it in the run summary, and an `any()`
        # short-circuit would leave later tags unrecorded.
        wanted = {tag: ctx.wants(ctx.path(f"{frame_key_stem(key)}_{tag}.png"))
                  for tag, _, _ in SYNTH_NVS_OFFSETS}
        if not any(wanted.values()):
            continue
        if use_mesh:
            geom = run.compose_mesh(key, device=ctx.device, objects=ctx.objects)
        else:
            geom = run.compose(key, device=ctx.device, space="world",
                               objects=ctx.objects)
        if geom is None:
            continue
        centroid = geom.centroid()
        K = scale_intrinsics(cam.K, ctx.src_hw, (ctx.H, ctx.W))
        # The train camera orbited about the world-placed object centroid at the
        # SAME distance (radius = |eye − centroid|), once per offset: the eye at
        # (0, 0) would be the train eye itself, so the magnitudes set how novel
        # each of the four viewpoints is.
        for tag, az_sign, el_sign in SYNTH_NVS_OFFSETS:
            if not wanted[tag]:
                continue
            c2w = _synth_orbit_c2w(cam.c2w, centroid,
                                   az_sign * az0 + az_step * key.frame,
                                   el_sign * el)
            rgb, alpha, depth = render(geom, c2w, K, ctx.H, ctx.W, ctx.bg)
            frame = overlay_trails(to_rgba(rgb, alpha), trails, i, c2w, K, depth,
                                   colors, opts)
            ctx.write_png(ctx.path(f"{frame_key_stem(key)}_{tag}.png"), frame)

    if opts["video"]:
        # One mp4 per offset — `write_per_view_videos` names PNGs by frame key
        # alone, which cannot see the tag.
        for tag, _, _ in SYNTH_NVS_OFFSETS:
            ctx.write_video(
                ctx.path(f"synth_nvs_{tag}.mp4"),
                [ctx.path(f"{frame_key_stem(k)}_{tag}.png") for k in keys],
            )


@renderer(
    "track_overlay_nvs",
    "ONE synthesized novel view per timestamp with the run's 3D tracks drawn over it, "
    "into viz/track_overlays/synth_nvs/. The figure twin of `track_overlay`, seen from "
    "off-axis instead of from the train camera. A FIGURE asset -- annotated and "
    "supersampled -- which is why it is NOT `synth_nvs`: that one stays a faithful, "
    "native-resolution reproduction of the scored renders_synth_nvs/. A run with no "
    "tracks_2d.npz still renders, just with nothing drawn on top.",
    options={
        # Chosen for legibility, NOT to match one of the four scored offsets: at 40/0
        # this is none of renders_synth_nvs/'s tags (azpos is the nearest, at +30/0).
        "azimuth": Option("float", 40.0, "horizontal offset off the train camera, deg"),
        "azimuth_step": Option("float", 0.0, "extra azimuth per timestamp"),
        "elevation": Option("float", 0.0, "vertical offset off the train camera, deg"),
        "video": Option("bool", True, "also encode the frames to mp4"),
        **_TRACK_DRAW_OPTIONS,
        **_FIGURE_CROP_OPTIONS,
    },
    out_sub="track_overlays/synth_nvs",
)
def render_track_overlay_nvs(run: FinalRun, ctx: RenderContext) -> None:
    """One synthesized view per timestamp, offset from that frame's train camera.

    ONE offset, not the four `synth_nvs` writes: a figure shows a single novel view
    per run, always the same one so the row stays a comparison, and rendering the
    other three would be three quarters of the cost thrown away.  Root motion is kept,
    as in `synth_nvs` -- the object moves through the scene while the viewpoint holds a
    fixed offset -- so the tracks need no rebasing.

    The offset follows each frame's OWN train camera rather than orbiting a fixed one,
    so on a moving-camera sequence the viewpoint moves with it.  That is `synth_nvs`'s
    convention, and it is what lets the tracks be drawn without rebasing.
    """
    if not run.has_renderable_geometry:
        log("    ! run has no gaussians or meshes — nothing to re-render")
        return
    from genia.core.utils.eval_assets_export import _synth_orbit_c2w

    opts = ctx.opts
    az0, az_step, el = opts["azimuth"], opts["azimuth_step"], opts["elevation"]
    use_mesh = run.prefers_mesh
    render = render_mesh if use_mesh else render_cloud
    keys = ctx.frame_keys

    trails = _track_trails(run, ctx, keys)
    colors = track_colors(trails.shape[1]) if trails is not None else []

    crop = opts["crop"] and not ctx.dry_run
    H, W, draw_opts = _supersampled_size(ctx, trails)
    pending: list = []

    for i, key in enumerate(keys):
        cam = run.camera(key)
        if cam is None or cam.K is None:
            log(f"    ! no camera with intrinsics for {key}, skipping")
            continue
        out = ctx.path(f"{frame_key_stem(key)}.png")
        if not ctx.wants(out) and not crop:
            continue
        geom = (run.compose_mesh(key, device=ctx.device, objects=ctx.objects) if use_mesh
                else run.compose(key, device=ctx.device, space="world",
                                 objects=ctx.objects))
        if geom is None:
            continue
        K = scale_intrinsics(cam.K, ctx.src_hw, (H, W))
        c2w = _synth_orbit_c2w(cam.c2w, geom.centroid(),
                               az0 + az_step * key.frame, el)
        rgb, alpha, depth = render(geom, c2w, K, H, W, ctx.bg)
        frame = overlay_trails(to_rgba(rgb, alpha), trails, i, c2w, K, depth,
                               colors, draw_opts)
        if crop:
            pending.append((out, frame))
        else:
            ctx.write_png(out, frame)

    _write_cropped(ctx, pending, H, W)

    if opts["video"]:
        ctx.write_per_view_videos("track_overlay_nvs", ctx.frame_keys)


@renderer(
    "track_overlay_test",
    "The run's 3D tracks drawn over a render from each HELD-OUT TEST camera, into "
    "viz/track_overlays/test_views/. The counterpart of `track_overlay_nvs` for the "
    "datasets that ship a GT test split (gso, oursactionbench, co3d): the novel view "
    "is a real one the method never saw, so there is no reason to synthesize one. "
    "A FIGURE asset -- annotated and supersampled -- which is why it is not "
    "`renders_test/`, the dir the NVS metrics are scored from.",
    options={
        "video": Option("bool", True, "also encode the frames to mp4"),
        **_TRACK_DRAW_OPTIONS,
        **_FIGURE_CROP_OPTIONS,
    },
    out_sub="track_overlays/test_views",
)
def render_track_overlay_test(run: FinalRun, ctx: RenderContext) -> None:
    """One render per held-out test camera, with the tracks drawn on.

    The cameras come from the dataset (see :func:`resolve_test_cameras`), not from
    ``poses.json`` -- a run records only the cameras it reconstructed from.  Root
    motion is kept, as in ``synth_nvs``, so the tracks need no rebasing; on
    OursActionBench each timestamp has its OWN test camera, so the viewpoint steps
    with the sequence exactly as ``renders_test/`` does.
    """
    if not run.has_renderable_geometry:
        log("    ! run has no gaussians or meshes — nothing to re-render")
        return
    # Resolved over the WHOLE run (the per-frame resolvers pair a camera with each
    # of the dataset's timestamps), then filtered to the keys this context renders,
    # so `--frames` narrows the overlay the way it narrows every other render.
    selected = set(ctx.frame_keys)
    views = [v for v in resolve_test_cameras(run, ctx.src_hw) if v.key in selected]
    if not views:
        return

    opts = ctx.opts
    use_mesh = run.prefers_mesh
    render = render_mesh if use_mesh else render_cloud

    # `_track_trails` indexes the trail by position in the keys it was built over,
    # so build it over the keys these cameras look at, in the same order.
    trails = _track_trails(run, ctx, [v.key for v in views])
    colors = track_colors(trails.shape[1]) if trails is not None else []

    crop = opts["crop"] and not ctx.dry_run
    H, W, draw_opts = _supersampled_size(ctx, trails)
    pending: list = []

    for i, view in enumerate(views):
        out = ctx.path(f"{view.stem}.png")
        if not ctx.wants(out) and not crop:
            continue
        key = view.key
        geom = (run.compose_mesh(key, device=ctx.device, objects=ctx.objects) if use_mesh
                else run.compose(key, device=ctx.device, space="world",
                                 objects=ctx.objects))
        if geom is None:
            continue
        K = scale_intrinsics(np.asarray(view.K, dtype=np.float32), ctx.src_hw, (H, W))
        c2w = np.asarray(view.c2w, dtype=np.float32)
        rgb, alpha, depth = render(geom, c2w, K, H, W, ctx.bg)
        frame = overlay_trails(to_rgba(rgb, alpha), trails, i, c2w, K, depth,
                               colors, draw_opts)
        if crop:
            pending.append((out, frame))
        else:
            ctx.write_png(out, frame)

    _write_cropped(ctx, pending, H, W)

    if opts["video"]:
        # Named by the render, and ordered by the views rather than by FrameKey:
        # a static split's "frames" are viewpoints, not timestamps.
        ctx.write_video(ctx.path("track_overlay_test.mp4"),
                        [ctx.path(f"{v.stem}.png") for v in views])
