# Copyright (c) Meta Platforms, Inc. and affiliates.

"""Load, transform, blend, and combine Gaussian clouds and triangle meshes."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
from genia.core.utils.final_rendering import runtime as _runtime
from genia.core.utils.final_rendering.assets import Camera

if TYPE_CHECKING:
    from genia.core.utils.final_rendering.assets import ObjectAssets, Transform
    


@dataclass
class GaussianCloud:
    """Raw gsplat-ready Gaussian parameters (activations already applied)."""

    means: Any       # (N, 3)
    quats: Any       # (N, 4) wxyz, normalized
    scales: Any      # (N, 3)
    opacities: Any   # (N,)
    sh: Any          # (N, K, 3), K = (degree + 1)^2

    def __len__(self) -> int:
        return int(self.means.shape[0])

    def clone(self) -> "GaussianCloud":
        return GaussianCloud(self.means.clone(), self.quats.clone(), self.scales.clone(),
                             self.opacities.clone(), self.sh.clone())

    def centroid(self) -> np.ndarray:
        """Outlier-robust center (median of means)."""
        return self.means.median(dim=0).values.detach().cpu().numpy()


def load_gaussian_ply(path: Path, device: str | None = None) -> GaussianCloud:
    """Read ``.ply`` / ``.compressed.ply`` into activated Gaussian parameters.

    Inverts what ``core/utils/io_utils.py::save_gaussian_ply`` wrote: ``exp``
    on log-scales, ``sigmoid`` on inverse-sigmoid opacities, normalize on the
    biased quaternion.  Both formats carry the full SH (``.compressed.ply``
    round-trips ``f_rest_*`` through gsconverter's quantized ``sh`` element).
    """
    import torch
    from genia.core.device import get_device
    device = str(get_device(device))
    path = Path(path)
    if path.name.endswith(".compressed.ply"):
        from gsconverter.formats.compressed_ply import CompressedPlyFormat
        data = CompressedPlyFormat().read(str(path))
    else:
        from plyfile import PlyData
        data = PlyData.read(str(path))["vertex"].data

    n = len(data)
    xyz = np.stack([data["x"], data["y"], data["z"]], -1).astype(np.float32)
    rot = np.stack([data[f"rot_{i}"] for i in range(4)], -1).astype(np.float32)
    scale = np.stack([data[f"scale_{i}"] for i in range(3)], -1).astype(np.float32)
    opacity = np.asarray(data["opacity"], dtype=np.float32)

    sh = np.stack([data[f"f_dc_{i}"] for i in range(3)], -1)[:, None, :].astype(np.float32)
    n_rest = sum(1 for k in data.dtype.names if k.startswith("f_rest_"))
    if n_rest:
        rest = np.stack([data[f"f_rest_{i}"] for i in range(n_rest)], -1).astype(np.float32)
        # save_gaussian_ply flattens channel-major: (N, 3, K-1) -> (N, 3*(K-1)).
        rest = rest.reshape(n, 3, n_rest // 3).transpose(0, 2, 1)
        sh = np.concatenate([sh, rest], axis=1)

    t = torch.from_numpy
    quats = t(rot).to(device)
    quats = quats / quats.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    return GaussianCloud(
        means=t(xyz).to(device),
        quats=quats,
        scales=t(scale).to(device).exp(),
        opacities=torch.sigmoid(t(opacity).to(device)),
        sh=t(np.ascontiguousarray(sh)).to(device),
    )


@dataclass
class MeshData:
    """A triangle mesh with per-vertex colour, ready to rasterize."""

    verts: Any    # (V, 3)
    faces: Any    # (F, 3) int32
    colors: Any   # (V, 3) in [0, 1]

    def __len__(self) -> int:
        return int(self.verts.shape[0])

    def centroid(self) -> np.ndarray:
        """Outlier-robust center, matching :meth:`GaussianCloud.centroid`."""
        return self.verts.median(dim=0).values.detach().cpu().numpy()


_MESH_CACHE: dict[tuple[Path, str], MeshData] = {}


def load_mesh(path: Path, device: str | None = None) -> MeshData:
    """Read a GLB/PLY/OBJ into object-local vertices, faces and vertex colours.

    Vertex colours are what ``save_per_object_mesh`` embeds in the GLB (the SLAT
    mesh decoder's ``vertex_attrs[:, :3]``); meshes without them render neutral
    grey rather than failing.
    """
    import torch
    from genia.core.device import get_device
    device = str(get_device(device))
    path = Path(path)
    cached = _MESH_CACHE.get((path, device))
    if cached is not None:
        return cached
    import trimesh

    m = trimesh.load(str(path), force="mesh", process=False)
    verts = torch.as_tensor(np.asarray(m.vertices), dtype=torch.float32, device=device)
    faces = torch.as_tensor(np.asarray(m.faces), dtype=torch.int32, device=device)
    rgb = getattr(getattr(m, "visual", None), "vertex_colors", None)
    if rgb is not None and len(rgb) == len(verts):
        colors = torch.as_tensor(
            np.asarray(rgb)[:, :3] / 255.0, dtype=torch.float32, device=device)
    else:
        colors = torch.full((len(verts), 3), 0.75, dtype=torch.float32, device=device)
    out = MeshData(verts=verts, faces=faces, colors=colors)
    _MESH_CACHE[(path, device)] = out
    return out


def merge_meshes(parts: list[MeshData]) -> MeshData:
    """Concatenate meshes into one, offsetting each part's face indices."""
    if len(parts) == 1:
        return parts[0]
    faces, offset = [], 0
    for p in parts:
        faces.append(p.faces + offset)
        offset += len(p)
    return MeshData(
        verts=_runtime.torch.cat([p.verts for p in parts]),
        faces=_runtime.torch.cat(faces).contiguous(),
        colors=_runtime.torch.cat([p.colors for p in parts]),
    )


def rows_correspond(a, b) -> bool:
    """Do two per-frame geometries index the same splats (vertices) row for row?

    **Equal length does not settle it.**  ``save_compressed_ply`` MORTON-SORTS what
    it writes (gsconverter's ``_sort_morton_order``: splats are reordered by a
    spatial curve so neighbours compress together), so every frame of a deformation
    lands in ITS OWN order and row `i` is a different splat in each file -- same
    count, no correspondence.  Blending those rows would mix unrelated Gaussians.

    So test a quantity the deformation does NOT touch.  ``warp_gaussians_high_res``
    warps means and quats only, leaving opacity and scale alone -- so a per-splat
    value that no longer lines up row for row IS the reorder.  For a mesh it is the
    face array: one canonical mesh warped keeps its tessellation, and a per-frame
    reconstruction does not.

    A CONSTANT witness proves nothing: a cloud that writes one opacity for every
    splat passes the opacity comparison whatever the order is.  Fall through to the
    scales, and refuse outright when neither varies: correspondence cannot be
    established, and interpolating on an unverified one would blend unrelated splats.

    A per-frame pipeline that legitimately varies opacity per frame is refused too;
    that is the conservative direction, and the caller says so.
    """
    if len(a) != len(b):
        return False
    if isinstance(a, MeshData):
        return a.faces.shape == b.faces.shape and bool(_runtime.torch.equal(a.faces, b.faces))
    for witness in (a.opacities, a.scales):
        if float(witness.std()) < 1e-6:      # carries no order information
            continue
        other = b.opacities if witness is a.opacities else b.scales
        return bool(_runtime.torch.allclose(witness, other, atol=1e-4))
    return False


def blend_geometry(a, b, w: float):
    """Linearly blend two per-frame geometries that share a correspondence.

    What a deformation field warps per frame is ONE canonical asset, so row `i` is
    the same splat -- the same vertex -- in every frame.  Blending two of them
    therefore samples the motion field BETWEEN the timestamps it was written at,
    which is what ``orbit.slowmo`` renders.

    Falls back to ``a`` (holding the earlier frame) whenever :func:`rows_correspond`
    says the rows do not line up -- a per-frame reconstruction, or the Morton-sorted
    output of the compressed PLY writer.  Row `i` means nothing across those, so
    there is nothing to interpolate.
    """
    if not rows_correspond(a, b):
        return a
    if isinstance(a, MeshData):
        return MeshData(verts=_runtime.torch.lerp(a.verts, b.verts, w), faces=a.faces,
                        colors=_runtime.torch.lerp(a.colors, b.colors, w))
    # Rotations nlerp'd on the shorter arc (hence the sign align): consecutive
    # frames of a deformation are a small rotation apart, where nlerp and slerp
    # differ by far less than a pixel.
    quats = _runtime.torch.where((a.quats * b.quats).sum(-1, keepdim=True) < 0,
                        -b.quats, b.quats)
    quats = _runtime.torch.lerp(a.quats, quats, w)
    return GaussianCloud(
        means=_runtime.torch.lerp(a.means, b.means, w),
        quats=quats / quats.norm(dim=-1, keepdim=True).clamp_min(1e-12),
        scales=_runtime.torch.lerp(a.scales, b.scales, w),
        opacities=_runtime.torch.lerp(a.opacities, b.opacities, w),
        sh=_runtime.torch.lerp(a.sh, b.sh, w),
    )


def apply_color_shift(cloud: GaussianCloud, obj: ObjectAssets,
                      xform: Transform) -> GaussianCloud:
    """Fold this frame's ``dc_offset`` / ``sh_rest`` into the cloud's SH."""
    if obj.dc_offsets is not None:
        delta = _runtime.torch.from_numpy(obj.dc_offsets[xform.index]).to(cloud.sh.device)
        cloud.sh[:, 0, :] = cloud.sh[:, 0, :] + delta               # (N, 3)
    if obj.sh_rest is not None:
        rest = _runtime.torch.from_numpy(obj.sh_rest[xform.index]).to(cloud.sh.device)
        cloud.sh = _runtime.torch.cat([cloud.sh[:, :1], rest], dim=1)        # (N, K-1, 3)
    return cloud


def _sim3_matrix(rotation: np.ndarray) -> np.ndarray:
    """Row-vector rotation matrix R for a wxyz quaternion (P3D convention)."""
    w, x, y, z = np.asarray(rotation, dtype=np.float64) / (
        np.linalg.norm(rotation) + 1e-12)
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y + w * z), 2 * (x * z - w * y)],
        [2 * (x * y - w * z), 1 - 2 * (x * x + z * z), 2 * (y * z + w * x)],
        [2 * (x * z + w * y), 2 * (y * z - w * x), 1 - 2 * (x * x + y * y)],
    ], dtype=np.float64).T  # quaternion_to_matrix is column-vector; row-vector = its T


def apply_sim3_points(pts: np.ndarray, rotation, translation, scale) -> np.ndarray:
    """Object-local -> camera-space Sim(3) on plain points: ``x * s @ R + t``."""
    R = _sim3_matrix(rotation)
    return (np.asarray(pts, np.float64) * np.asarray(scale, np.float64)) @ R \
        + np.asarray(translation, np.float64)


def unapply_sim3_points(pts: np.ndarray, rotation, translation, scale) -> np.ndarray:
    """Inverse of :func:`apply_sim3_points`: ``((p - t) @ R.T) / s``."""
    R = _sim3_matrix(rotation)
    return ((np.asarray(pts, np.float64) - np.asarray(translation, np.float64)) @ R.T) \
        / np.asarray(scale, np.float64)


def r3_to_p3d_points(pts: np.ndarray) -> np.ndarray:
    """R3 <-> P3D positions (negate X and Y -- the flip is self-inverse)."""
    out = np.asarray(pts, np.float64).copy()
    out[..., :2] *= -1
    return out


def _world_to_cam_points(pts: np.ndarray, cam: "Camera | None") -> np.ndarray:
    if cam is None or np.allclose(cam.c2w, np.eye(4)):
        return np.asarray(pts, np.float64)
    R, t = cam.c2w[:3, :3], cam.c2w[:3, 3]
    return (np.asarray(pts, np.float64) - t) @ R


def _cam_to_world_points(pts: np.ndarray, cam: "Camera | None") -> np.ndarray:
    if cam is None or np.allclose(cam.c2w, np.eye(4)):
        return np.asarray(pts, np.float64)
    R, t = cam.c2w[:3, :3], cam.c2w[:3, 3]
    return np.asarray(pts, np.float64) @ R.T + t


def apply_sim3(cloud: GaussianCloud, rotation: np.ndarray, translation: np.ndarray,
               scale: np.ndarray) -> GaussianCloud:
    """Object-local -> camera-space Sim(3), matching ``apply_pose_to_gaussian``.

    Row-vector convention: ``x * s @ R + t`` (the torch twin of
    :func:`apply_sim3_points`); the per-Gaussian orientation is pre-multiplied
    by the *inverse* object quaternion (the ``make_scene`` convention), and
    scales pick up the object scale element-wise.
    """
    device = cloud.means.device
    q = _runtime.torch.as_tensor(rotation, dtype=_runtime.torch.float32, device=device)
    q = q / q.norm().clamp_min(1e-12)
    t = _runtime.torch.as_tensor(translation, dtype=_runtime.torch.float32, device=device)
    s = _runtime.torch.as_tensor(scale, dtype=_runtime.torch.float32, device=device)

    R = _runtime.quaternion_to_matrix(q.unsqueeze(0)).squeeze(0)         # (3, 3)
    cloud.means = _runtime.torch.mm(cloud.means * s, R) + t              # @ R, not @ R.T
    q_inv = _runtime.quaternion_invert(q.unsqueeze(0)).squeeze(0)
    cloud.quats = _runtime.quaternion_multiply(
        q_inv.unsqueeze(0).expand(cloud.quats.shape[0], -1), cloud.quats
    )
    cloud.scales = cloud.scales * s
    return cloud


def p3d_to_r3(cloud: GaussianCloud) -> GaussianCloud:
    """PyTorch3D (X-left/Y-up) -> R3 (X-right/Y-down) camera space."""
    cloud.means = _runtime.p3d_to_r3_positions(cloud.means)
    cloud.quats = _runtime.p3d_to_r3_quaternions(cloud.quats)
    return cloud


def cam_to_world(cloud: GaussianCloud, c2w: np.ndarray) -> GaussianCloud:
    """R3 camera space -> R3 world space (no-op for an identity ``c2w``)."""
    from genia.core.utils.rendering import transform_gaussian_params_cam_to_world
    cloud.means, cloud.quats = transform_gaussian_params_cam_to_world(
        cloud.means, cloud.quats, np.asarray(c2w, dtype=np.float32)
    )
    return cloud


def concat_clouds(clouds: list[GaussianCloud]) -> GaussianCloud:
    """Merge object clouds into one scene cloud (SH bands zero-padded to max K)."""
    if len(clouds) == 1:
        return clouds[0]
    k = max(c.sh.shape[1] for c in clouds)
    shs = []
    for c in clouds:
        if c.sh.shape[1] < k:
            pad = _runtime.torch.zeros(c.sh.shape[0], k - c.sh.shape[1], 3, device=c.sh.device)
            shs.append(_runtime.torch.cat([c.sh, pad], dim=1))
        else:
            shs.append(c.sh)
    return GaussianCloud(
        means=_runtime.torch.cat([c.means for c in clouds]),
        quats=_runtime.torch.cat([c.quats for c in clouds]),
        scales=_runtime.torch.cat([c.scales for c in clouds]),
        opacities=_runtime.torch.cat([c.opacities for c in clouds]),
        sh=_runtime.torch.cat(shs),
    )
