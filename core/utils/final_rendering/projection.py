# Copyright (c) Meta Platforms, Inc. and affiliates.

"""Render geometry, project points, and fit camera intrinsics to scene bounds."""

from __future__ import annotations

import numpy as np
from genia.core.utils.final_rendering import runtime as _runtime
from genia.core.utils.final_rendering.geometry import GaussianCloud, MeshData

_GLCTX = None


def render_mesh(mesh: MeshData, c2w: np.ndarray, K: np.ndarray, H: int, W: int,
                bg: np.ndarray):
    """nvdiffrast render -> ``(rgb, alpha, depth)``, matching :func:`render_cloud`.

    Reuses the pipeline's own ``render_rgba_and_depth``, so the projection and
    the y-flip to image convention are the ones Stage-2 mesh guidance uses.
    """
    global _GLCTX
    import nvdiffrast.torch as dr
    from genia.core.utils.mesh_rendering import render_rgba_and_depth

    if _GLCTX is None:
        _GLCTX = dr.RasterizeCudaContext()
    c2w = np.asarray(c2w, dtype=np.float64)
    R = _runtime.torch.as_tensor(c2w[:3, :3], dtype=_runtime.torch.float32, device=mesh.verts.device)
    t = _runtime.torch.as_tensor(c2w[:3, 3], dtype=_runtime.torch.float32, device=mesh.verts.device)
    verts_cam = (mesh.verts - t) @ R          # world -> R3 camera (row-vector c2w)

    rgb, alpha, depth = render_rgba_and_depth(
        verts_cam, mesh.faces, mesh.colors, _GLCTX,
        float(K[0, 0]), float(K[1, 1]), float(K[0, 2]), float(K[1, 2]), H, W,
    )
    # render_rgba_and_depth leaves the background at zero; composite on `bg` so
    # the output matches the Gaussian path (foreground on bg, coverage in alpha).
    bg_t = _runtime.torch.as_tensor(bg, dtype=_runtime.torch.float32, device=rgb.device)
    a = alpha.clamp(0, 1)[..., None]
    return (rgb.clamp(0, 1) * a + bg_t * (1 - a)), alpha.clamp(0, 1), depth


def render_cloud(cloud: GaussianCloud, c2w: np.ndarray, K: np.ndarray,
                 H: int, W: int, bg: np.ndarray):
    """gsplat render -> ``(rgb (H,W,3), alpha (H,W), depth (H,W))`` tensors."""
    bg_t = _runtime.torch.as_tensor(bg, dtype=_runtime.torch.float32, device=cloud.means.device)
    rgb, alpha, depth = _runtime.render_gaussian_params(
        means=cloud.means, quats=cloud.quats, scales=cloud.scales,
        opacities=cloud.opacities, features=cloud.sh,
        c2w=np.asarray(c2w, dtype=np.float32), K_matrix=np.asarray(K, dtype=np.float32),
        W=W, H=H, bg_color=bg_t,
    )
    return rgb.clamp(0, 1), alpha.clamp(0, 1), depth


def project_r3(pts: np.ndarray, c2w: np.ndarray, K: np.ndarray) -> np.ndarray:
    """R3 world points -> ``(N, 3)`` of ``(u, v, z_cam)`` for a pinhole camera."""
    w2c = np.linalg.inv(np.asarray(c2w, np.float64))
    cam = np.asarray(pts, np.float64) @ w2c[:3, :3].T + w2c[:3, 3]
    z = cam[:, 2]
    safe = np.where(np.abs(z) < 1e-6, 1e-6, z)
    u = K[0, 0] * cam[:, 0] / safe + K[0, 2]
    v = K[1, 1] * cam[:, 1] / safe + K[1, 2]
    return np.stack([u, v, z], axis=-1)


def scale_intrinsics(K: np.ndarray, src_hw: tuple[int, int],
                     dst_hw: tuple[int, int]) -> np.ndarray:
    """Rescale a 3x3 K between resolutions (pixel-center convention)."""
    if src_hw == dst_hw:
        return K
    sy, sx = dst_hw[0] / src_hw[0], dst_hw[1] / src_hw[1]
    out = np.asarray(K, dtype=np.float32).copy()
    out[0, 0] *= sx
    out[1, 1] *= sy
    out[0, 2] = (out[0, 2] + 0.5) * sx - 0.5
    out[1, 2] = (out[1, 2] + 0.5) * sy - 0.5
    return out


def centred_intrinsics(focal: float, W: int, H: int) -> np.ndarray:
    """K for a ``W x H`` canvas: one focal on both axes (square pixels), principal
    point at the centre.  Sole owner of that centring convention, so the fitted
    renders cannot disagree with :func:`fit_focal` about where the axis is."""
    return np.array([[focal, 0.0, (W - 1) * 0.5],
                     [0.0, focal, (H - 1) * 0.5],
                     [0.0, 0.0, 1.0]], np.float32)


def square_intrinsics(focal: float, side: int) -> np.ndarray:
    """:func:`centred_intrinsics` on a square canvas, so a turntable shows the
    object the same way at every azimuth."""
    return centred_intrinsics(focal, side, side)


def bounding_points(geom: "GaussianCloud | MeshData", max_points: int = 20000):
    """``(P, 3)`` positions + ``(P,)`` radii bounding the geometry's visible extent.

    Subsampled by stride for the framing fit -- the extremes that decide the
    focal survive subsampling, and the fit's safety margin absorbs the rest.
    For Gaussians the radius is 3 sigma of the widest axis, i.e. roughly where
    gsplat stops drawing a splat, so the fit bounds *rendered* pixels rather
    than centres; mesh vertices are exact and need no radius.
    """
    step = max(1, len(geom) // max_points)
    if isinstance(geom, MeshData):
        # Gaussians reach here already trimmed (`trim_outliers`); a mesh cannot be,
        # since dropping a vertex would tear its faces.  So the same radius cut is
        # applied to the FRAMING only -- a stray vertex still renders, it just does
        # not decide where the camera looks or how far away it sits.
        verts = geom.verts[::step]
        verts = verts[_runtime.bulk_keep_mask(verts)].detach().cpu().numpy()
        return verts, np.zeros(len(verts), np.float32)
    means = geom.means[::step].detach().cpu().numpy()
    radii = 3.0 * geom.scales[::step].max(dim=1).values.detach().cpu().numpy()
    return means, radii


def trim_outliers(geom: "GaussianCloud",
                  factor: float = 1.5) -> tuple["GaussianCloud", int]:
    """Drop stray splats outside the cloud's bulk; returns ``(cloud, dropped)``.

    Four cuts, one per kind of stray a depth-unprojected cloud can carry, and all
    four rules live in ``core/utils/rendering.py`` so the turntable's framing and
    its render cannot desync.  They are independent -- each catches strays the
    others do not, which is why all four run:

    * ``bulk_keep_mask`` -- radius about the median, which catches a second
      CLUSTER stranded behind the object.
    * ``visible_bulk_mask`` -- the opacity-weighted per-axis box, which catches a
      diffuse HAZE the radius rule is blind to: a haze drags the radius percentile
      out with it, so the radius cut alone can drop nothing while the object's box
      is inflated and its centre pulled off the object.
    * ``oversize_splat_mask`` -- splats that are a large fraction of the object
      themselves.  Neither of the others can see these: they sit at the object's
      own depth and are opaque, so they are neither far nor faint, just far too
      big.
    * ``connected_bulk_mask`` -- compact CLUMPS stranded off the body, whose splats
      are individually unremarkable on all three counts above and only objectionable
      together.  A coherent reconstruction is ONE component, so this cannot fire on
      one.

    Why a turntable in particular needs this: it swings the camera right past the
    stranded splats, where they collapse the framing fit (``x / z`` with
    ``z -> 0``), pull the framed box centre off the object, and smear the frame
    with splats the size of the object.
    """
    keep = (_runtime.bulk_keep_mask(geom.means, factor)
            & _runtime.visible_bulk_mask(geom.means, geom.opacities, geom.scales))
    # The last two run on the SURVIVORS, because both measure against the object's
    # extent: a haze still in the cloud inflates that extent, raising the size bar
    # and coarsening the voxel grid the components are labelled on.
    idx = _runtime.torch.nonzero(keep, as_tuple=True)[0]
    inner = (_runtime.oversize_splat_mask(geom.means[idx], geom.scales[idx])
             & _runtime.connected_bulk_mask(geom.means[idx], geom.opacities[idx],
                                   geom.scales[idx]))
    keep[idx[~inner]] = False
    dropped = int(len(geom) - keep.sum())
    if not dropped:
        return geom, 0
    return GaussianCloud(geom.means[keep], geom.quats[keep], geom.scales[keep],
                         geom.opacities[keep], geom.sh[keep]), dropped


def bounds_aabb(pts, radii):
    """``(lo, hi)`` of the axis-aligned box around one ``bounding_points`` entry,
    radii included so it bounds rendered pixels rather than centres, or None."""
    pts = np.asarray(pts, np.float64)
    if not len(pts):
        return None
    r = np.asarray(radii, np.float64).reshape(-1, 1)
    return (pts - r).min(axis=0), (pts + r).max(axis=0)


def fit_focal(bounds, cameras, W: int, H: int, margin: float = 0.04) -> float | None:
    """Largest focal that keeps every point inside a ``W x H`` frame, for every camera.

    Projection offset from the principal point is ``f * x / z``, i.e. linear in
    ``f``, so the tightest framing is just the available half-span divided by
    the worst ``(|x| + r) / z`` over all points and all cameras.  ``bounds`` is
    a list of ``(points, radii)`` -- one entry per distinct geometry the render
    will show, so a deforming object is framed for its whole motion and never
    changes apparent size mid-turn.  Returns None when nothing is in front of
    the camera.

    The two axes are fitted SEPARATELY and the tighter one wins: square pixels
    mean one focal serves both, but a non-square frame has a different half-span
    on each, so folding them into a single worst ratio (what a square-only fit
    can do) would overflow the short side.
    """
    half_x = (W * 0.5) * (1.0 - margin)
    half_y = (H * 0.5) * (1.0 - margin)
    worst_x = worst_y = 0.0
    for c2w in cameras:
        w2c = np.linalg.inv(np.asarray(c2w, np.float64))
        for pts, radii in bounds:
            cam = np.asarray(pts, np.float64) @ w2c[:3, :3].T + w2c[:3, 3]
            front = cam[:, 2] > 1e-6
            if not front.any():
                continue
            z, r = cam[front, 2], np.asarray(radii, np.float64)[front]
            worst_x = max(worst_x, float(((np.abs(cam[front, 0]) + r) / z).max()))
            worst_y = max(worst_y, float(((np.abs(cam[front, 1]) + r) / z).max()))
    fits = [h / w for h, w in ((half_x, worst_x), (half_y, worst_y)) if w > 0]
    return min(fits) if fits else None


def fit_square_focal(bounds, cameras, side: int, margin: float = 0.04) -> float | None:
    """:func:`fit_focal` on a square canvas -- what a turntable always renders into."""
    return fit_focal(bounds, cameras, side, side, margin)
