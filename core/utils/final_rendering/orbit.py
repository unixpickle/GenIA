# Copyright (c) Meta Platforms, Inc. and affiliates.

"""Sample deformation time and render canonical-object turntable sequences."""

from __future__ import annotations

from functools import lru_cache
from typing import TYPE_CHECKING, Any

import numpy as np
from genia.core.utils.final_rendering.configuration import resolve_up
from genia.core.utils.final_rendering.deformation import DeformationSource
from genia.core.utils.final_rendering.geometry import blend_geometry, rows_correspond
from genia.core.utils.final_rendering.images import to_rgba
from genia.core.utils.final_rendering.options import _TRACK_OPTIONS
from genia.core.utils.final_rendering.projection import (
    bounding_points,
    bounds_aabb,
    fit_square_focal,
    render_cloud,
    render_mesh,
    square_intrinsics,
    trim_outliers,
)
from genia.core.utils.final_rendering.registry import Option, renderer
from genia.core.utils.final_rendering.runtime import log
from genia.core.utils.final_rendering.tracks import _track_trails, overlay_trails, track_colors

if TYPE_CHECKING:
    from genia.core.utils.final_rendering.context import RenderContext
    from genia.core.utils.final_rendering.run import FinalRun



def motion_sample(m: int, slowmo: int, n_keys: int) -> tuple[int, int, float]:
    """Orbit step ``m`` -> the two frames it sits between, and the blend weight.

    ``m / slowmo`` is a continuous position in the sequence, so the frames land on
    the steps that are multiples of ``slowmo`` and the steps between them ask for a
    blend.  ``slowmo=1`` is the plain one-frame-per-step mapping, weight 0.

    The last interval CLAMPS: the final frame is HELD through the loop's seam
    rather than morphing back into the first.  The seam stays the hard cut it
    already was, and no motion is invented that the sequence does not contain.
    """
    pos = m / slowmo
    t0 = min(int(pos), n_keys - 1)
    return t0, min(t0 + 1, n_keys - 1), pos - t0


def sample_of(m: int, slowmo: int, n_keys: int,
              interpolate: bool) -> tuple[int, int, float]:
    """Which motion sample step ``m`` SHOWS -- :func:`motion_sample`, collapsed onto
    the held frame when the geometry cannot be interpolated.

    Holding the FLOOR frame (not the nearest) keeps the held clip a strict
    sub-sampling of the interpolated one: step ``t * slowmo`` shows frame ``t`` in
    both, so the two sit side by side in a gallery without a phase shift.

    Two steps with the same sample show the same picture, which is why this tuple --
    not the step index -- is what the trim, the bounds and the framing are keyed on.
    """
    t0, t1, w = motion_sample(m, slowmo, n_keys)
    return (t0, t1, w) if interpolate else (t0, t0, 0.0)


@renderer(
    "orbit",
    "Square turntable. ONE camera path per run: the target and FoV are fitted over "
    "every pose at once, so a deforming object does not make the camera chase it. "
    "Gaussians when the run has them, else the mesh. "
    "Dynamic runs deform in place (root motion removed), with their 3D tracks overlaid.",
    options={
        "n_frames": Option("int", 60, "orbit steps; on an animated FULL turn a "
                           "MINIMUM, rounded up to whole sequence loops "
                           "(0 = one per frame key)"),
        "azimuth_start": Option("float", 0.0, "first azimuth, degrees"),
        "azimuth_span": Option("float", 360.0, "total azimuth swept, degrees"),
        "elevation": Option("float", 0.0, "elevation offset, degrees"),
        "up": Option("str", "auto", "the vertical the turntable spins about: "
                     "auto (the dataset's world up where one is known, else "
                     "camera) | camera (the train camera's own up) | world "
                     "(R3 -Y) | 'x,y,z'"),
        "margin": Option("float", 0.04, "padding kept around the object when "
                                        "fitting the FoV (fraction of the frame)"),
        "frame": Option("int?", None, "freeze on this timestamp (implies animate=false)"),
        "animate": Option("bool", True, "step the timestamp along the orbit "
                                        "(dynamic runs; no-op when static)"),
        "slowmo": Option("int", 10, "play the motion this many times slower: each "
                         "1-frame interval is drawn over N steps with the "
                         "deformation LERPed between the two frames (needs "
                         "per-frame geometry, else it renders at 1)"),
        "source": Option("str", "auto", "geometry source: auto (final.pt when it "
                         "can serve the run, else the exported PLYs) | final_pt | plys"),
        "video": Option("bool", True, "encode the orbit to mp4"),
        "keep_frames": Option("bool", True, "also write the individual PNGs"),
        "distance_scale": Option("float", 2.5, "FALLBACK camera distance as a "
                                 "multiple of the framed box's half-diagonal, used "
                                 "only when the self-calibrated radius (the training "
                                 "camera's own distance to the centroid) would put "
                                 "the orbit camera inside the box -- e.g. a "
                                 "whole-scene reconstruction, whose 'object' is the "
                                 "whole scene rather than what one training camera "
                                 "was aimed at. Same convention as world_space's "
                                 "distance_scale."),
        **_TRACK_OPTIONS,
    },
)
def render_orbit(run: FinalRun, ctx: RenderContext) -> None:
    if not ctx.frame_keys:
        return
    opts = ctx.opts
    # Resolved up front so a bad `up` is rejected by --dry-run too, not first
    # discovered after the geometry has been loaded.
    up = resolve_up(opts["up"], run.dir)
    keys = ctx.frame_keys
    if opts["frame"] is not None:
        keys = [k for k in keys if k.frame == opts["frame"]] or keys
    # Step through the sequence unless pinned to one timestamp, taking each
    # frame's *deformation* but keeping the reference frame's Sim(3): a
    # turntable should show the object changing shape in place, not flying
    # across the frame under its root motion. No-op on a static run (one key).
    animate = bool(opts["animate"]) and opts["frame"] is None and len(keys) > 1
    # Slow motion lengthens the clip on EVERY animated run, whether or not its
    # geometry can be interpolated: a held clip keeps the same duration and the same
    # camera path as an interpolated one, so the two sit side by side in a gallery.
    # `interpolate` below decides only whether the object moves between frames.
    #
    # It slows TIME, so it keys on distinct timestamps, not on `keys` -- those are
    # (frame, view) pairs, and on an MV STATIC run they are all one timestamp seen
    # from several sides.  Multiplying by the VIEW count there would make an MV clip
    # longer than the mono clip of the same scene, breaking the side-by-side
    # comparison the length is supposed to protect.
    slowmo = (max(1, int(opts["slowmo"]))
              if animate and len({k.frame for k in keys}) > 1 else 1)
    ref_key = keys[0]
    ref_cam = run.camera(ref_key)
    if ref_cam is None or ref_cam.K is None:
        log("    ! no reference camera with intrinsics, skipping orbit")
        return

    use_mesh = run.prefers_mesh

    # The geometry SOURCE, decided once.  `final.pt` carries the canonical gaussians
    # and the deformation field, so re-warping here beats reading the exported
    # per-frame PLYs on both correctness (they are Morton-sorted, see
    # `rows_correspond`) and speed.  Mesh runs keep the PLY path: their per-frame
    # meshes are written by a plain writer and already correspond.
    want_source = opts["source"] if opts["source"] in ("auto", "final_pt") else None
    # A PER-FRAME RECONSTRUCTION is recognised here rather than discovered object by
    # object inside `open`: it reconstructs each timestamp independently and never
    # holds a canonical asset, so the cache has no canonical gaussians to serve and
    # there is nothing to interpolate between its frames.  Skipping it up front saves
    # loading a large final.pt to learn that.
    per_frame_only = run.has_perframe_gaussians and not run.has_gaussians
    if per_frame_only and want_source:
        log("    (per-frame reconstruction: no canonical asset, so the geometry "
            "comes from the exported per-frame PLYs and the frames cannot be "
            "interpolated)")
    source = None
    if want_source and not use_mesh and not ctx.dry_run and not per_frame_only:
        source = DeformationSource.open(
            run.dir, device=ctx.device, objects=sorted(
                set(run.objects) if ctx.objects is None else set(ctx.objects)),
            frames={k.frame for k in keys},
            required=opts["source"] == "final_pt")
    provider = None if source is None else source.provider()

    def compose_frame(t: int):
        return (run.compose_mesh(keys[t], device=ctx.device, objects=ctx.objects,
                                 pose_from=ref_key) if use_mesh else
                run.compose(keys[t], device=ctx.device, objects=ctx.objects,
                            pose_from=ref_key, geometry=provider))

    # Whether the frames can be interpolated at all, settled once on the first pair
    # rather than per sample.  It does NOT change the clip's length: a run that
    # cannot interpolate holds each frame for `slowmo` steps, so its orbit stays
    # comparable, frame for frame and camera for camera, with one that can.
    interpolate = slowmo > 1
    if interpolate and not ctx.dry_run and source is None:
        # Only the PLY path can be non-corresponding: `DeformationSource` warps one
        # canonical asset, so its frames correspond by construction.
        first, second = compose_frame(0), compose_frame(1)
        if first is None or second is None or not rows_correspond(first, second):
            log(f"    ! the per-frame assets do not correspond row for row — a "
                f"per-frame reconstruction, or (usually) gaussians written by the "
                f"compressed PLY writer, which Morton-sorts each frame into its own "
                f"order. Holding each frame for {slowmo} steps instead of "
                f"interpolating: the clip keeps its length and its camera path, so "
                f"it still sits beside an interpolated one — the motion steps rather "
                f"than flows. Keep final/final.pt, or re-export with "
                f"output.save_compressed_ply=false, to interpolate")
            interpolate = False
        del first, second

    # One pass through the sequence takes `loop` steps: one per frame, or `slowmo`
    # per frame when the motion is slowed.  Everything below indexes MOTION SAMPLES
    # in [0, loop), not frames -- the two coincide at slowmo=1.
    loop = len(keys) * slowmo
    n = max(1, opts["n_frames"] or loop)
    closed = opts["azimuth_span"] % 360.0 == 0.0
    if animate and closed and n % loop:
        # `n_frames` is a MINIMUM on a dynamic run: a turntable that stops
        # part-way through the sequence jumps when the clip loops, and the
        # motion is the point of animating it.  Round up to whole loops.
        n += loop - n % loop
        log(f"    {opts['n_frames']} steps is not a whole number of "
            f"{loop}-step loops — rounding up to {n}")
    if slowmo > 1:
        # "where they correspond", not "lerped": a run can pass the first-pair check
        # and still hold later pairs, and the per-sample note below says when it did.
        log(f"    slow motion x{slowmo}: {len(keys)} frame(s) over {loop} steps, "
            + ("the deformation interpolated where they correspond" if interpolate
               else f"each HELD for {slowmo} steps (nothing to interpolate)"))
    out_video = ctx.path("orbit.mp4") if opts["video"] else None
    frame_paths = [ctx.path("frames", f"{i:03d}.png") for i in range(n)]
    if ctx.dry_run:
        if opts["keep_frames"]:
            ctx.written.extend(frame_paths)
        if out_video is not None:
            ctx.written.append(out_video)
        return
    # The orbit is one artifact -- an existing video means there is nothing to do.
    if out_video is not None and not ctx.wants(out_video):
        log(f"    {out_video.name} exists (pass --overwrite to redo it)")
        return

    dropped: dict[tuple, int] = {}
    held: set[tuple] = set()    # samples whose blend fell back to holding a frame

    # UNtrimmed geometry per frame.  A blend reads two frames and re-reads them for
    # every step between, so those two are worth keeping.  Nothing else re-reads:
    # at slowmo=1, and when holding (every sample is one frame, visited once), each
    # frame is composed exactly once and `maxsize=0` keeps the orbit holding a
    # single cloud at a time.
    compose_at = lru_cache(maxsize=2 if interpolate else 0)(compose_frame)

    def geometry(sample: tuple[int, int, float]):
        """Renderable geometry for one motion sample ``(t0, t1, w)``.

        Every frame is composed with the REFERENCE pose, so blending two of them
        blends their deformations alone -- the Sim(3) they share is affine in the
        points, which makes lerping after posing the same as lerping before it.

        Trimming comes AFTER the blend, deliberately: it drops a different set of
        splats in each frame, and the blend needs the two clouds to still
        correspond row for row.
        """
        t0, t1, w = sample
        geom = compose_at(t0)
        if w > 0 and t1 != t0 and geom is not None:
            other = compose_at(t1)
            blended = geom if other is None else blend_geometry(geom, other, w)
            if blended is geom:
                # `blend_geometry` held t0 rather than interpolating: the frames do
                # not correspond.  Print it once -- the clip is a stutter, not a
                # slow motion, and nothing else in the output would say why.
                if not held:
                    log("    ! two frames stopped corresponding part-way through "
                        "— holding the earlier one for those steps")
                held.add(sample)
            geom = blended
        if geom is None or use_mesh:
            return geom
        # A turntable is the one render that gets close to a stray splat behind
        # the object, so it is the one render that has to reject them.
        geom, n_out = trim_outliers(geom)
        dropped[sample] = n_out
        return geom

    ref_sample = sample_of(0, slowmo, len(keys), interpolate)
    ref_geom = geometry(ref_sample)
    if ref_geom is None:
        log("    ! no gaussians or meshes posed at the reference frame, skipping")
        return
    if dropped.get(ref_sample):
        log(f"    dropped {dropped[ref_sample]} outlier splat(s) outside the "
            f"object's bulk ({len(ref_geom)} kept)")
    if use_mesh:
        log(f"    rendering the mesh ({len(ref_geom)} verts) — run has no gaussians")

    from genia.core.utils.eval_assets_export import _synth_orbit_c2w

    # Which steps show which pose, defined ONCE: the bounds below, the per-pose
    # framing and the render loop all key off it, so they cannot disagree.
    # Keyed by the SAMPLE, not the step: steps showing the same picture collapse
    # into one group, so a held clip trims, bounds and frames each distinct pose
    # once instead of `slowmo` times.  (The renders cannot collapse -- every step
    # is a different azimuth, which is the whole point of keeping the length.)
    steps_of: dict[tuple, list[int]] = {}
    for i in range(n):
        steps_of.setdefault(
            sample_of(i % loop if animate else 0, slowmo, len(keys), interpolate),
            []).append(i)

    # Bounds of every pose, gathered BEFORE the cameras: they decide both where the
    # orbit looks and how large the object comes out.
    pose_bounds = {}
    for t in steps_of:
        g = ref_geom if t == ref_sample else geometry(t)
        if g is not None:
            pose_bounds[t] = bounding_points(g)
    if not pose_bounds:
        log("    ! no geometry to frame, skipping")
        return

    # Trails only mean something against geometry that actually deforms. With a
    # canonical-only asset (a gaussian/mesh run whose motion is pure Sim(3)) the
    # render is static once the pose is pinned, so moving trails would imply
    # motion the picture does not show -- require a per-frame deforming asset.
    show_tracks = (opts["tracks"] and animate
                   and (run.has_perframe_gaussians or run.has_perframe_meshes))
    if opts["tracks"] and not show_tracks and run.tracks is not None:
        log("    (skipping the 3D track overlay: no per-frame geometry to match it)")
    # `pose_bounds` is keyed by motion SAMPLE; the anchor filter indexes each
    # trajectory by FRAME, so collapse it (keeping the exact frame, w=0, which
    # sorts first) before handing it over.
    frame_bounds: dict[int, Any] = {}
    for (t0, _t1, _w), b in pose_bounds.items():
        frame_bounds.setdefault(t0, b)
    trails = (_track_trails(run, ctx, keys, ref_key, frame_bounds)
              if show_tracks else None)
    trail_opts = opts
    if trails is not None and slowmo > 1:
        # Resampled through the SAME mapping the geometry is, so a trail head stays
        # on the splat it belongs to; `track_trail` is scaled with it, so its unit
        # stays FRAMES rather than silently becoming steps.
        trails = np.stack([trails[t0] * (1.0 - w) + trails[t1] * w
                           for t0, t1, w in
                           (sample_of(m, slowmo, len(keys), interpolate)
                            for m in range(loop))])
        trail_opts = {**opts, "track_trail": opts["track_trail"] * slowmo}
    colors = track_colors(trails.shape[1]) if trails is not None else []

    # (0, 0) reproduces the reference camera, so a full turn must not repeat it.
    # Always square: a turntable sweeps the object through every azimuth, so a
    # frame shaped like the source sequence would clip it side-on.
    side = max(ctx.H, ctx.W)
    azimuths = [
        opts["azimuth_start"] + opts["azimuth_span"] * (
            i / n if closed else i / max(1, n - 1))
        for i in range(n)
    ]
    log(f"    spinning about {'the train camera up' if up is None else np.round(up, 3)}"
        f" (up={opts['up']})")

    # ONE framing for the whole clip: the target is the centre of the box covering
    # EVERY pose, and the focal is fitted over every pose from every azimuth (which
    # is what `fit_square_focal` takes a LIST of bounds for).  So the camera path is
    # a plain circle and nothing about it depends on the timestamp.
    #
    # Framing each pose on its OWN extent instead would make the camera chase the
    # object's axis-aligned box, and that box moves every frame on anything that
    # deforms: the object would jitter, and the per-pose focal would make it breathe.
    #
    # The cost: the object fills somewhat less of the frame than a per-pose fit
    # would give it, and a deformation's drift and growth stay visible rather than
    # being normalized away.  For a SHAPE viewer that is the honest picture, and runs
    # stay comparable in a gallery since each is still fitted to its own extent.
    boxes = [box for box in (bounds_aabb(*b) for b in pose_bounds.values())
             if box is not None]
    if not boxes:
        log("    ! no geometry to frame, skipping")
        return
    lo = np.min([b[0] for b in boxes], axis=0)
    hi = np.max([b[1] for b in boxes], axis=0)
    centroid = (lo + hi) * 0.5
    # The self-calibrated radius (`_synth_orbit_c2w`'s default: the training camera's
    # own distance to the centroid) is only a sane orbit radius when the geometry is
    # roughly the size of what that camera was aimed at. On a whole-scene
    # reconstruction the "object" is the whole scene, whose box can dwarf that
    # distance -- the camera would then sit INSIDE the box, and fitting a focal that
    # keeps everything in frame from in there collapses toward zero (a huge FoV that
    # renders the object as impossibly small and far away). Falling back to a
    # world_space-style radius, fitted to the box itself, is the same self-fit
    # `world_space` uses.
    eye0 = np.asarray(ref_cam.c2w[:3, 3], dtype=np.float64)
    self_radius = float(np.linalg.norm(centroid - eye0))
    box_half_diag = float(np.linalg.norm(hi - lo)) * 0.5
    radius = None
    if self_radius < box_half_diag:
        radius = box_half_diag * opts["distance_scale"]
        log(f"    training camera sits inside the framed box (self-calibrated "
            f"radius {self_radius:.2f} < half-diagonal {box_half_diag:.2f}) — "
            f"using a self-fitted radius {radius:.2f} instead")
    cams = [_synth_orbit_c2w(ref_cam.c2w, centroid, az, opts["elevation"], up=up,
                             radius=radius) for az in azimuths]
    focal = fit_square_focal(list(pose_bounds.values()), cams, side,
                             margin=max(0.0, opts["margin"]))
    if focal is None:
        log("    ! nothing in front of the orbit cameras, skipping")
        return
    Ks = [square_intrinsics(focal, side)] * n
    log(f"    framing: {side}x{side}, ONE camera path — target and focal "
        f"({focal:.1f}px) fitted over all {len(pose_bounds)} pose(s) from "
        f"{len(azimuths)} view(s), {opts['margin']:.0%} margin")

    render = render_mesh if use_mesh else render_cloud
    # Frame-major, not step-major: per-frame clouds are deliberately not cached,
    # so stepping i in order would re-decode each one on every loop.  The steps
    # sharing a motion sample are rendered together instead, which decodes each
    # cloud once and still holds only one.
    frames: list[np.ndarray | None] = [None] * n
    for t, idxs in steps_of.items():                      # every step, grouped
        geom = ref_geom if t == ref_sample else geometry(t)
        if geom is None:
            continue
        for i in idxs:
            c2w, K = cams[i], Ks[i]
            if c2w is None:
                continue
            rgb, alpha, depth = render(geom, c2w, K, side, side, ctx.bg)
            # The group key is a sample, not an index; the trails carry one row
            # per step of the loop, so the step's own position indexes them.
            frames[i] = overlay_trails(to_rgba(rgb, alpha), trails,
                                       i % loop if animate else 0, c2w, K,
                                       depth, colors, trail_opts)
            if opts["keep_frames"]:
                ctx.write_png(frame_paths[i], frames[i])

    if out_video is not None:
        ctx.write_video(out_video, [f for f in frames if f is not None])
