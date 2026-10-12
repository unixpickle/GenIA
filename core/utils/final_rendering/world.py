# Copyright (c) Meta Platforms, Inc. and affiliates.

"""Render world-space overviews, background geometry, and pose axes."""

from __future__ import annotations

from typing import TYPE_CHECKING, Collection

import numpy as np
from genia.core.utils.final_rendering import runtime as _runtime
from genia.core.utils.final_rendering.geometry import (
    GaussianCloud,
    concat_clouds,
    load_gaussian_ply,
)
from genia.core.utils.final_rendering.images import to_rgba
from genia.core.utils.final_rendering.projection import (
    bounding_points,
    bounds_aabb,
    centred_intrinsics,
    fit_focal,
    project_r3,
    render_cloud,
    render_mesh,
    trim_outliers,
)
from genia.core.utils.final_rendering.registry import Option, renderer
from genia.core.utils.final_rendering.runtime import log
from genia.core.utils.frame_key import FrameKey, group_by_view

if TYPE_CHECKING:
    from genia.core.utils.final_rendering.context import RenderContext
    from genia.core.utils.final_rendering.run import FinalRun



_AXIS_COLORS = ((255, 60, 60), (60, 220, 60), (60, 120, 255))


def load_background_cloud(run: FinalRun, device: str) -> "GaussianCloud | None":
    """``final/gaussians/background[.compressed].ply``, or None if the run wrote none.

    Already R3 WORLD space (every frame's background lifted by its own ``c2w`` and
    unioned, per ``gaussian.aggregate_background_gaussians``), so it composes with
    ``FinalRun.compose(space="world")`` by concatenation and nothing else.  Absent for
    every white-background dataset (GSO, OAB) and for any run whose
    ``output.save_world_assets`` was off -- foreground-only is a normal outcome here,
    not an error.
    """
    for name in ("background.compressed.ply", "background.ply"):
        path = run.dir / "gaussians" / name
        if path.is_file():
            return load_gaussian_ply(path, device)
    return None


def overview_cameras(run: FinalRun, keys_by_view: dict[int, list[FrameKey]],
                     target: np.ndarray, distance: float,
                     elevation: float, azimuth: float) -> dict[int, np.ndarray]:
    """One overview c2w per view, each placed behind THAT view's own input camera.

    A single shared camera cannot tell the views of one timestamp apart: an MV run that
    honours ``pipeline.mv_shared_world_pose`` puts every object at the same WORLD point
    from every view, so V renders from one viewpoint are V copies of one picture.  Giving each view the
    viewpoint of the camera that actually observed it is what makes the set worth
    rendering -- the scene layout seen from each side it was captured from.

    Each view keeps its own ELEVATION, which is the half of its vantage that
    ``overview_look_at`` would otherwise throw away: that function flattens the reference
    forward onto the horizontal plane before applying ``elevation``, so two cameras
    differing mainly in height collapse onto the same viewpoint.  ``azimuth`` needs no such treatment -- it is already relative, rotating a ``back_dir``
    that carries the view's own azimuth.

    Elevations are CENTRED ON THEIR MEAN rather than used absolutely, so ``elevation``
    keeps meaning "the centre of the fan sits this far above horizontal".  That also
    makes a mono run a bit-exact no-op: with one view ``own - mean`` is ``x - x``, exactly
    0.0, so ``overview_look_at`` sees identical arguments; with absolute elevations
    every mono run with a tilted input camera would move.

    ``target`` and ``distance`` are SHARED, and the caller fits one focal over all the
    returned cameras, so the views stay directly comparable in position and scale.

    NOTE the world up is R3 ``-Y`` throughout, which is only KNOWN to be up on a run that
    kept the dataset's cameras (GSO + ``camera_poses_source=gt``); a predicted-pose run
    sits in an arbitrary frame, the caveat ``viz_orbit_up=auto`` encodes for the orbit.
    The per-view SEPARATION is unaffected -- relative geometry between views holds
    whichever axis is up -- only the absolute tilt and the image roll rest on the guess.
    """
    # Local, like the orbit's `_synth_orbit_c2w`: `genia.core.utils.evaluation` imports torch,
    # and this file defers that to `_lazy_imports` so --list/--dry-run stay fast.
    from genia.core.utils.evaluation import overview_look_at

    # No camera in poses.json -> look down +Z, the identity camera such a run's poses are
    # already expressed against.
    forwards = {}
    for view, vkeys in keys_by_view.items():
        cam = run.camera(vkeys[0])
        forwards[view] = (cam.c2w[:3, 2] if cam is not None
                          else np.array([0.0, 0.0, 1.0]))

    # Up is -Y in R3 and the camera sits along -forward from the target, so this view's
    # own elevation above horizontal is asin(forward . -up) = asin(forward[1]).
    own = {view: float(np.degrees(np.arcsin(np.clip(
              (f / max(np.linalg.norm(f), 1e-12))[1], -1.0, 1.0))))
           for view, f in forwards.items()}
    mean_own = float(np.mean(list(own.values())))

    # No clamp on the total: a straight-down camera gives own = 90, and overview_look_at
    # at 110 still returns a finite, orthonormal c2w at the right distance (checked).
    # `own - mean` FIRST, then add: grouped the other way, `(elevation + own) - mean`
    # loses the low bits and a mono run drifts by a float32 LSB instead of being the
    # exact no-op the whole centring exists to guarantee.
    return {view: overview_look_at(target, f, distance,
                                   elevation + (own[view] - mean_own), azimuth)
            for view, f in forwards.items()}


def draw_pose_axes(rgba: np.ndarray, run: FinalRun, key: FrameKey, c2w: np.ndarray,
                   K: np.ndarray, objects: Collection[int], axis_length: float,
                   width: int = 2) -> np.ndarray:
    """Draw each object's Sim(3) axes at ``key`` onto an RGBA frame.

    The object's local origin and three axis tips go through ``_place_points`` -- the
    same object-local-P3D -> posed-R3 chain the Gaussians take -- so the axes land on
    the geometry by construction rather than by a re-derivation that could drift from
    it.  ``axis_length`` is in object-local units and is therefore scaled by the pose,
    matching ``interpolation.compute_pose_axes``.

    Drawn on top, with no depth test: the axes originate at the object's own centre,
    so testing them against the rendered surface would bury every one of them.
    """
    from PIL import ImageDraw

    local = np.vstack([np.zeros((1, 3)), np.eye(3) * float(axis_length)])
    img = _runtime.Image.fromarray(rgba).convert("RGBA")
    draw = ImageDraw.Draw(img)
    for obj_idx in sorted(set(run.objects) & set(objects)):
        pose = run.resolve_transform(run.objects[obj_idx], key)
        if pose is None:
            continue
        uv = project_r3(run._place_points(local, pose), c2w, K)
        # All four points strictly in front, or skip the object: a vertex approaching
        # the image plane sends `x / z` to infinity, and PIL would try to rasterize it.
        if not (uv[:, 2] > 1e-6).all():
            continue
        ox, oy = uv[0, :2]
        for axis in range(3):
            draw.line([(ox, oy), tuple(uv[axis + 1, :2])],
                      fill=(*_AXIS_COLORS[axis], 255), width=width)
        draw.ellipse([ox - width, oy - width, ox + width, oy + width],
                     fill=(255, 255, 255, 255))
    return np.array(img)


def view_tag(view: int, multi: bool) -> str:
    """``_vNN`` when more than one view is being rendered, else empty.

    Deliberately NOT ``frame_key_stem``, which leaves view 0 unsuffixed: that reads
    fine where views are separate folders or separate assets, but here the views of one
    timestamp are a SERIES of the same picture from different sides, and a bare
    ``000.png`` beside ``000_v01.png`` does not look like a member of it.  A mono run
    still gets the plain ``{frame:03d}`` every other per-frame writer uses.
    """
    return f"_v{view:02d}" if multi else ""


@renderer(
    "world_space",
    "Every object posed into world space, seen from a fixed overview camera PER VIEW -- "
    "each placed behind the input camera that observed that view, so an MV run shows its "
    "layout from every side it was captured from (one shared camera would render V "
    "copies of one picture). Framed on the FOREGROUND -- the camera aims at the objects' "
    "box centre and ONE focal is fitted to them across every view and frame, so the "
    "objects fill the frame, the camera stays static as they move, and the views stay "
    "comparable in scale. Root motion is KEPT (unlike `orbit`), so a dynamic run shows "
    "its objects travelling through the scene, and a multi-object run shows their real "
    "relative layout. Foreground only by default; add "
    "-O world_space.background=true for the run's background point cloud, "
    "-O world_space.axes=true for each object's Sim(3) pose axes.",
    options={
        "elevation": Option("float", 20.0, "degrees above horizontal for the CENTRE of "
                            "the views' fan — each view keeps its own camera's elevation "
                            "relative to that, so an MV run's views stay distinct"),
        "azimuth": Option("float", 10.0, "camera swing around the world up, degrees "
                          "off straight-behind-the-train-camera (0 = straight behind)"),
        "distance_scale": Option("float", 2.5, "camera distance as a multiple of the "
                                 "foreground's bounding radius"),
        "margin": Option("float", 0.08, "padding kept around the foreground when "
                                        "fitting the focal (fraction of the frame)"),
        "background": Option("bool", False, "include gaussians/background*.ply when the "
                             "run wrote one. OFF by default: it is an unprojected depth "
                             "map, so it arrives full of holes and occluding fragments "
                             "that bury the reconstruction this render exists to show"),
        "axes": Option("bool", False, "draw each object's Sim(3) pose axes. OFF by "
                       "default: this render exists to show the scene, and the axes "
                       "are a pose DEBUG overlay that clutters it"),
        "axis_length": Option("float", 0.15, "axis arm length, in object-local units "
                                             "(so the pose scales it)"),
        "perframe": Option("bool", True, "use per-frame warped PLYs when present"),
        "video": Option("bool", True, "also encode one mp4 per view, when that view has "
                        "more than one frame"),
    },
)
def render_world_space(run: FinalRun, ctx: RenderContext) -> None:
    if not run.has_renderable_geometry:
        log("    ! run has no gaussians or meshes — nothing to render")
        return
    opts = ctx.opts

    # One render per (frame, view), each view from its OWN overview camera -- see
    # `overview_cameras` for why a shared one makes the views indistinguishable.
    keys = list(ctx.frame_keys)
    if not keys:
        return

    use_mesh = run.prefers_mesh
    by_view = group_by_view(keys)
    multi = len(by_view) > 1        # from the FILTERED keys, so `--views 0` reads as mono
    frame_paths = [ctx.path(f"{k.frame:03d}{view_tag(k.view, multi)}.png") for k in keys]
    # A video per VIEW, gated on THAT view having more than one frame: mono-dynamic gets
    # one clip, MV-static gets V stills and no clip (there is no time axis to play), and
    # MV-dynamic gets V clips.  Per-view rather than a global max, so a view with a lone
    # frame is not handed to the encoder on a run whose views are unequal in length.
    videos = {v: ctx.path(f"world_space{view_tag(v, multi)}.mp4")
              for v, vkeys in by_view.items() if opts["video"] and len(vkeys) > 1}
    if ctx.dry_run:
        ctx.written.extend(frame_paths)
        ctx.written.extend(videos.values())
        return
    # One artifact: the camera is fitted over EVERY key, so re-rendering a subset would
    # have to redo that fit anyway -- there is no cheap partial resume to offer.
    primary = next(iter(videos.values()), frame_paths[0])
    if not ctx.wants(primary):
        log(f"    {primary.name} exists (pass --overwrite to redo it)")
        return

    def foreground(key: FrameKey):
        if use_mesh:
            return run.compose_mesh(key, device=ctx.device, objects=ctx.objects)
        return run.compose(key, device=ctx.device, space="world",
                           objects=ctx.objects, use_perframe=opts["perframe"])

    # Bounds over EVERY key, unioned -- what makes the camera STATIC: the objects never
    # leave the frame as they move, and never breathe as their silhouettes narrow and
    # widen.  Measured on the TRIMMED cloud (one stray splat collapses an `x / z` fit)
    # while the render below keeps every splat: a scene view should not silently delete
    # geometry, it should show what the run produced.  A separate pass from the render,
    # like the orbit's, so only one composed cloud is ever held at a time.
    bounds = []
    for key in keys:
        geom = foreground(key)
        if geom is not None:
            bounds.append(bounding_points(
                geom if use_mesh else trim_outliers(geom)[0]))
    if not bounds:
        log("    ! nothing posed at any frame key, skipping")
        return

    pts = np.concatenate([b[0] for b in bounds])
    radii = np.concatenate([b[1] for b in bounds])
    box = bounds_aabb(pts, radii)
    if box is None:
        log("    ! no foreground extent to frame, skipping")
        return
    lo, hi = box
    # The box CENTRE, not a mass centroid: a mass centroid follows
    # wherever the splats are dense and puts an asymmetric scene off-centre.  Distance
    # comes from the foreground's own size, so perspective is scaled to the objects
    # rather than to how far apart the input cameras happened to be.
    target = (lo + hi) * 0.5
    radius = max(float(np.linalg.norm(hi - lo)) * 0.5, 1e-3)

    cams = overview_cameras(run, by_view, target, radius * opts["distance_scale"],
                            opts["elevation"], opts["azimuth"])
    # ONE focal over EVERY view, so the views are comparable in scale -- `fit_focal`
    # already takes a list of cameras and returns the worst case, which is exactly that.
    # Fitted to the foreground ALONE: the background is rendered but does not vote, and
    # letting it in is what leaves the objects a quarter of the frame.
    focal = fit_focal(bounds, list(cams.values()), ctx.W, ctx.H, opts["margin"])
    if focal is None:
        log("    ! nothing in front of the overview camera, skipping")
        return
    K = centred_intrinsics(focal, ctx.W, ctx.H)

    # How much of the frame the foreground won in its TIGHTEST view -- the one number
    # that says whether one shared fit was good enough for all of them, so it belongs in
    # the log rather than only in the PNGs.
    covers = []
    for c2w in cams.values():
        uv = project_r3(pts, c2w, K)
        seen = uv[uv[:, 2] > 0, :2]
        covers.append(float(np.prod(seen.max(0) - seen.min(0))) / (ctx.W * ctx.H)
                      if len(seen) else 0.0)
    log(f"    framing: {ctx.W}x{ctx.H}, focal {focal:.1f}px fitted to the foreground "
        f"over {len(bounds)} frame key(s) from {len(cams)} view(s) "
        f"({opts['margin']:.0%} margin) — its box covers {min(covers):.0%} of the frame")

    bg_cloud = None
    if opts["background"] and not use_mesh:
        bg_cloud = load_background_cloud(run, ctx.device)
        log(f"    + background point cloud ({len(bg_cloud)} splats)"
            if bg_cloud is not None else
            "    (no gaussians/background*.ply — foreground only)")

    render = render_mesh if use_mesh else render_cloud
    paths = dict(zip(keys, frame_paths))
    for key in keys:
        geom = foreground(key)
        if geom is None:
            continue
        if bg_cloud is not None:
            geom = concat_clouds([geom, bg_cloud])
        c2w = cams[key.view]
        rgb, alpha, _ = render(geom, c2w, K, ctx.H, ctx.W, ctx.bg)
        frame = to_rgba(rgb, alpha)
        if opts["axes"]:
            frame = draw_pose_axes(frame, run, key, c2w, K, ctx.objects,
                                   opts["axis_length"])
        ctx.write_png(paths[key], frame)

    # Not `ctx.write_per_view_videos`: that owns two naming decisions this render makes
    # differently (frame_key_stem for the PNGs, an unsuffixed view 0 for the mp4), and
    # three other renders depend on it meaning what it means.  Streaming the PNGs back
    # off disk is its one behaviour worth keeping, and `write_video` already does that.
    for view, out in videos.items():
        ctx.write_video(out, [paths[k] for k in by_view[view]])
