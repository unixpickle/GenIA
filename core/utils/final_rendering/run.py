# Copyright (c) Meta Platforms, Inc. and affiliates.

"""Read a final directory and compose its per-frame scene geometry."""

from __future__ import annotations

import json
from functools import cached_property
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Iterable

import numpy as np
from genia.core.utils.final_rendering import runtime as _runtime
from genia.core.utils.final_rendering.assets import (
    Camera,
    ObjectAssets,
    Tracks,
    Transform,
    _obj_xyz_members,
)
from genia.core.utils.final_rendering.discovery import parse_run_path
from genia.core.utils.final_rendering.geometry import (
    MeshData,
    _cam_to_world_points,
    _world_to_cam_points,
    apply_color_shift,
    apply_sim3,
    apply_sim3_points,
    cam_to_world,
    concat_clouds,
    load_gaussian_ply,
    load_mesh,
    merge_meshes,
    p3d_to_r3,
    r3_to_p3d_points,
    unapply_sim3_points,
)
from genia.core.utils.final_rendering.runtime import _lazy_imports, log
from genia.core.utils.frame_key import FrameKey, frame_key_sort_key

if TYPE_CHECKING:
    from genia.core.utils.final_rendering.geometry import GaussianCloud



class FinalRun:
    """A ``final/`` directory, loaded from disk.

    Owns the parsed ``poses.json`` (objects + cameras), the discovered asset
    paths, and the render-space composition helpers that turn them into a posed
    Gaussian cloud for any frame key.
    """

    def __init__(self, final_dir: Path):
        self.path = parse_run_path(Path(final_dir))
        self.dir = self.path.final_dir
        poses_path = self.dir / "poses.json"
        if not poses_path.is_file():
            raise FileNotFoundError(f"no poses.json in {self.dir}")
        with open(poses_path) as f:
            raw = json.load(f)

        self.objects: dict[int, ObjectAssets] = {}
        for entry in raw.get("objects", []):
            obj = self._parse_object(entry)
            self.objects[obj.obj_idx] = obj

        self.cameras: dict[FrameKey, Camera] = {}
        for entry in raw.get("cameras", []):
            key = FrameKey(int(entry.get("frame", entry.get("frame_idx", 0))),
                           int(entry.get("view", 0)))
            K = entry.get("K")
            self.cameras[key] = Camera(
                c2w=np.asarray(entry["c2w"], dtype=np.float32),
                K=None if K is None else np.asarray(K, dtype=np.float32),
            )
        # MV runs may key cameras on one view only; first camera per timestamp.
        self._camera_by_frame: dict[int, Camera] = {}
        for key in sorted(self.cameras, key=frame_key_sort_key):
            self._camera_by_frame.setdefault(key.frame, self.cameras[key])

        self._cloud_cache: dict[tuple[Path, str], Any] = {}

    # -- construction helpers ------------------------------------------------

    def _parse_object(self, entry: dict) -> ObjectAssets:
        obj_idx = int(entry["obj_idx"])
        transforms: dict[FrameKey, Transform] = {}
        for i, x in enumerate(entry.get("transforms", [])):
            scale = np.asarray(x["scale"], dtype=np.float32).reshape(-1)
            if scale.size == 1:  # isotropic runs write a 1-vector
                scale = np.repeat(scale, 3)
            pf = x.get("ply_perframe")
            t = Transform(
                key=FrameKey(int(x.get("frame", 0)), int(x.get("view", 0))),
                translation=np.asarray(x["translation"], dtype=np.float32).reshape(3),
                rotation=np.asarray(x["rotation"], dtype=np.float32).reshape(4),
                scale=scale,
                ply_perframe=(self.dir / pf) if pf else None,
                index=i,
            )
            transforms[t.key] = t

        # Per-frame deformed meshes follow the eval convention on disk
        # (`meshes/{obj:03d}/{frame:03d}.ply`, written by the per-frame mesh
        # export), matched to each transform by timestamp — poses.json need not
        # name them, mirroring the canonical-GLB fallback below.
        pf_mesh_dir = self.dir / "meshes" / f"{obj_idx:03d}"
        if pf_mesh_dir.is_dir():
            for t in transforms.values():
                cand = pf_mesh_dir / f"{t.key.frame:03d}.ply"
                if cand.is_file():
                    t.mesh_perframe = cand

        def _rel(key: str) -> Path | None:
            """Resolve an asset reference, ignoring stale or non-file entries.

            ``is_file`` matters: ``save_perframe_poses_json`` writes a
            *directory* under ``"ply"`` for per-frame-only pipelines, and a
            mesh-only run can name a GLB it never wrote.
            """
            v = entry.get(key)
            p = self.dir / v if v else None
            return p if p is not None and p.is_file() else None

        # `save_per_object_mesh` writes meshes/{obj:03d}.glb whether or not
        # poses.json names it, so fall back to the conventional path.
        mesh = _rel("mesh")
        if mesh is None:
            conventional = self.dir / "meshes" / f"{obj_idx:03d}.glb"
            mesh = conventional if conventional.is_file() else None

        obj = ObjectAssets(
            obj_idx=obj_idx,
            ply=_rel("ply"),
            mesh=mesh,
            transforms=transforms,
        )
        obj.dc_offsets = self._load_bin(entry, "dc_offsets", len(transforms))
        obj.sh_rest = self._load_bin(entry, "sh_rest", len(transforms))
        return obj

    def _load_bin(self, entry: dict, key: str, n_transforms: int) -> np.ndarray | None:
        """Read a flat Float32 side-car (``dc_offsets`` / ``sh_rest``).

        Rows are positional -- the writer co-iterates them with ``transforms``
        -- so a row count that disagrees means the two are no longer aligned
        and the offsets cannot be attributed to a frame.
        """
        rel, shape = entry.get(f"{key}_file"), entry.get(f"{key}_shape")
        if not rel or not shape:
            return None
        path = self.dir / rel
        if not path.is_file():
            return None
        arr = np.fromfile(path, dtype=np.float32).reshape(*shape)
        if arr.shape[0] != n_transforms:
            log(f"  ! {path.name}: {arr.shape[0]} rows vs {n_transforms} "
                f"transforms -- ignoring (cannot align to frames)")
            return None
        return arr

    # -- identity + shape ----------------------------------------------------

    @cached_property
    def frame_keys(self) -> list[FrameKey]:
        """All ``(frame, view)`` keys, sorted by ``(view, frame)``."""
        keys = set(self.cameras)
        for obj in self.objects.values():
            keys |= set(obj.transforms)
        return sorted(keys, key=frame_key_sort_key)

    @cached_property
    def frames(self) -> list[int]:
        return sorted({k.frame for k in self.frame_keys})

    @cached_property
    def views(self) -> list[int]:
        return sorted({k.view for k in self.frame_keys})

    @property
    def is_dynamic(self) -> bool:
        return len(self.frames) > 1

    @property
    def is_mv(self) -> bool:
        return len(self.views) > 1

    @property
    def has_gaussians(self) -> bool:
        return any(o.ply is not None for o in self.objects.values())

    @property
    def has_perframe_gaussians(self) -> bool:
        return any(
            t.ply_perframe is not None
            for o in self.objects.values()
            for t in o.transforms.values()
        )

    @property
    def has_meshes(self) -> bool:
        return any(o.mesh is not None for o in self.objects.values())

    @property
    def has_perframe_meshes(self) -> bool:
        return any(
            t.mesh_perframe is not None
            for o in self.objects.values()
            for t in o.transforms.values()
        )

    @property
    def has_renderable_geometry(self) -> bool:
        """Anything a GPU render can pose — gaussians (canonical or per-frame) or
        a mesh.  When false, `train_views`/`orbit`/`synth_nvs` have nothing to do."""
        return self.has_gaussians or self.has_perframe_gaussians or self.has_meshes

    @property
    def prefers_mesh(self) -> bool:
        """Render the mesh only when there are no gaussians of either kind —
        gaussians carry appearance the mesh does not, so they always win."""
        return (not (self.has_gaussians or self.has_perframe_gaussians)
                and self.has_meshes)

    @property
    def poses_vary(self) -> bool:
        """Does any object's Sim(3) actually change across its frame keys?

        Distinguishes a genuinely moving object from a run that is *labelled*
        dynamic but stored one static pose per frame (a run whose deformation
        lives only in its pre-rendered ``renders_train/``).
        """
        for obj in self.objects.values():
            xs = list(obj.transforms.values())
            for x in xs[1:]:
                if not (np.allclose(x.translation, xs[0].translation)
                        and np.allclose(x.rotation, xs[0].rotation)
                        and np.allclose(x.scale, xs[0].scale)):
                    return True
        return False

    @cached_property
    def resolution(self) -> tuple[int, int]:
        """(H, W) of the run's renders -- from an exported PNG, else from K.

        ``poses.json`` carries ``K`` but no image size, so an exported PNG is
        the only record of the resolution the run actually rendered at.
        """
        for sub in ("renders_train", "renders_test"):
            hits = sorted((self.dir / sub).glob("**/*.png"))
            if hits:
                _lazy_imports(gpu=False)
                with _runtime.Image.open(hits[0]) as im:
                    return im.size[1], im.size[0]
        for cam in self.cameras.values():
            if cam.K is not None:
                return int(round(cam.K[1, 2] * 2)), int(round(cam.K[0, 2] * 2))
        return 512, 512

    def camera(self, key: FrameKey) -> Camera | None:
        """That key's camera, or the timestamp's when views are keyed on one."""
        return self.cameras.get(key) or self._camera_by_frame.get(key.frame)

    @cached_property
    def tracks(self) -> Tracks | None:
        """The run's 3D tracks: ``final/tracks_2d.npz``'s ``obj_*_xyz``, falling
        back to ``final/tracks_3d.npz`` for the objects it does not carry.

        Two artifacts because a run whose ``tracks_2d.npz`` carries its own
        PIXEL correspondence has its ``obj_*_xyz`` dropped from that file on
        purpose: its ``uv`` anchors are a different, denser set, and pairing two
        anchor counts under one object is the schema hazard
        ``interpolation.write_tapvid_tracks`` exists to prevent.  Such runs still
        write ``tracks_3d.npz`` out of the SAME world-space
        ``compute_object_tracks`` call, so it is the 3D source here.
        """
        # tracks_3d.npz carries no frame axis of its own -- it is written from the
        # same `all_frame_indices` tracks_2d.npz records, so those rows win when
        # both exist, and the run's own timestamps stand in when neither does.
        frames, xyz, vis = self.frames, {}, {}
        dense = self.dir / "tracks_2d.npz"
        if dense.is_file():
            with np.load(dense, allow_pickle=False) as data:
                frames = [int(f) for f in data["frames"]]
                xyz = _obj_xyz_members(data)
                vis = {obj: np.asarray(data[f"obj_{obj}_vis"], dtype=bool)
                       for obj in xyz if f"obj_{obj}_vis" in data.files}
        sparse = self.dir / "tracks_3d.npz"
        if sparse.is_file():
            with np.load(sparse, allow_pickle=False) as data:
                extra = _obj_xyz_members(data)
            for obj, pts in extra.items():
                if obj in xyz:
                    continue
                if pts.shape[0] != len(frames):
                    log(f"  ! tracks_3d.npz obj {obj}: {pts.shape[0]} rows vs "
                        f"{len(frames)} frames -- ignoring")
                    continue
                xyz[obj] = pts
        return Tracks(frames, xyz, vis) if xyz else None

    def summary(self) -> str:
        kind = "dynamic" if self.is_dynamic else "static"
        kind += "/mv" if self.is_mv else "/mono"
        H, W = self.resolution
        bits = [f"{len(self.objects)} object(s)", f"{len(self.frames)} frame(s)",
                f"{len(self.views)} view(s)", kind, f"{W}x{H}"]
        if self.has_perframe_gaussians:
            bits.append("per-frame warped gaussians")
        if not self.has_gaussians:
            bits.append("no canonical gaussians")
        return ", ".join(bits)

    # -- scene composition ---------------------------------------------------

    def _cloud(self, path: Path, device: str, cache: bool) -> "GaussianCloud":
        """Load a PLY, caching only assets that get reused across frames.

        Canonical clouds are posed once per frame, so caching pays for itself;
        a ``ply_perframe`` asset is used exactly once, so caching it would just
        pin every frame's splats in VRAM.
        """
        cached = self._cloud_cache.get((path, device))
        if cached is None:
            cached = load_gaussian_ply(path, device)
            if cache:
                self._cloud_cache[(path, device)] = cached
        return cached

    def resolve_transform(self, obj: ObjectAssets, key: FrameKey) -> Transform | None:
        """The pose to place ``obj`` with at ``key`` -- exact, or its anchor.

        Poses are camera-space, so an object keyed at a different frame key is
        placed via *that* key's ``c2w`` and still lands in the same world spot.
        Two fallbacks: the same timestamp under another view (MV runs key the
        pose on one view), then -- on a static run only -- a lone transform (an
        object reconstructed once but observed by several cameras, e.g. CO3D).
        """
        exact = obj.transforms.get(key)
        if exact is not None:
            return exact
        same_frame = sorted(k for k in obj.transforms if k.frame == key.frame)
        if same_frame:
            return obj.transforms[same_frame[0]]
        if not self.is_dynamic and len(obj.transforms) == 1:
            return next(iter(obj.transforms.values()))
        return None

    def compose(
        self,
        key: FrameKey,
        *,
        device: str | None = None,
        space: str = "world",
        objects: Iterable[int] | None = None,
        use_perframe: bool = True,
        pose_from: FrameKey | None = None,
        geometry: "Callable[[int, FrameKey], GaussianCloud | None] | None" = None,
    ) -> "GaussianCloud | None":
        """Posed Gaussian cloud for one frame key, or None if nothing is posed.

        ``space``: ``object`` (no pose), ``camera`` (Sim(3) + P3D->R3) or
        ``world`` (+ the ``c2w`` of the key each object's pose is anchored on).
        ``camera`` space is only meaningful when every object has a pose at
        ``key`` itself -- otherwise the anchors disagree on what "camera" means.

        ``pose_from`` splits *which frame's shape* from *where it is placed*:
        geometry and colour still come from ``key`` (so the per-frame warped
        deformation is kept) while the Sim(3) + ``c2w`` come from that other
        key.  Pinning it to one frame strips the object's root motion, leaving
        only the deformation -- what a turntable of a moving object wants.

        ``geometry`` overrides where the object-local cloud COMES from, without
        touching where it is put: a callback returning None for an object falls
        through to the exported PLYs, and everything after -- the colour shift, the
        Sim(3), P3D->R3 and the ``c2w`` -- runs identically either way, so a
        provider cloud lands in exactly the same world spot as the PLY it replaces.
        :class:`DeformationSource` is the one caller.
        """
        from genia.core.device import get_device
        device = str(get_device(device))
        wanted = set(self.objects if objects is None else objects)
        clouds: list[GaussianCloud] = []
        for obj_idx in sorted(set(self.objects) & wanted):
            obj = self.objects[obj_idx]
            xform = self.resolve_transform(obj, key)
            if xform is None:
                continue
            pose = xform if pose_from is None else (
                self.resolve_transform(obj, pose_from) or xform
            )
            cloud = None if geometry is None else geometry(obj_idx, key)
            if cloud is None:
                perframe = xform.ply_perframe if use_perframe else None
                src = perframe or obj.ply
                if src is None:
                    continue
                cloud = self._cloud(src, device, cache=perframe is None)
            # Cloned before the colour shift writes into `sh` -- the cache above and
            # a provider both hand out tensors they still own.
            cloud = apply_color_shift(cloud.clone(), obj, xform)
            if space == "object":
                clouds.append(cloud)
                continue
            cloud = p3d_to_r3(
                apply_sim3(cloud, pose.rotation, pose.translation, pose.scale)
            )
            if space == "world":
                cam = self.camera(pose.key)
                if cam is not None:
                    cloud = cam_to_world(cloud, cam.c2w)
            clouds.append(cloud)

        return concat_clouds(clouds) if clouds else None

    def compose_mesh(
        self,
        key: FrameKey,
        *,
        device: str | None = None,
        objects: Iterable[int] | None = None,
        pose_from: FrameKey | None = None,
    ) -> "MeshData | None":
        """Posed world-space mesh for one frame key -- the mesh twin of :meth:`compose`.

        The per-frame deformed mesh (``mesh_perframe``) when present, else the
        canonical GLB -- both live in the same object-local PyTorch3D space as
        the canonical Gaussians, so they go through the identical Sim(3) ->
        P3D->R3 -> ``c2w`` chain, and ``pose_from`` means the same thing: keep
        ``key``'s deformed geometry while borrowing another frame's Sim(3).
        """
        from genia.core.device import get_device
        device = str(get_device(device))
        wanted = set(self.objects if objects is None else objects)
        parts: list[MeshData] = []
        for obj_idx in sorted(set(self.objects) & wanted):
            obj = self.objects[obj_idx]
            xform = self.resolve_transform(obj, key)
            if xform is None:
                continue
            src = xform.mesh_perframe or obj.mesh   # per-frame deformation wins
            if src is None:
                continue
            pose = xform if pose_from is None else (
                self.resolve_transform(obj, pose_from) or xform
            )
            mesh = load_mesh(src, device)
            verts = self._place_points(mesh.verts.detach().cpu().numpy(), pose)
            parts.append(MeshData(
                verts=_runtime.torch.as_tensor(verts, dtype=_runtime.torch.float32, device=device),
                faces=mesh.faces, colors=mesh.colors,
            ))
        return merge_meshes(parts) if parts else None

    def _place_points(self, local: np.ndarray, pose: Transform) -> np.ndarray:
        """Object-local P3D points -> posed R3 world, the plain-point chain."""
        pts = apply_sim3_points(local, pose.rotation, pose.translation, pose.scale)
        return _cam_to_world_points(r3_to_p3d_points(pts), self.camera(pose.key))

    def _unplace_points(self, world: np.ndarray, pose: Transform) -> np.ndarray:
        """Inverse of :meth:`_place_points`."""
        pts = r3_to_p3d_points(_world_to_cam_points(world, self.camera(pose.key)))
        return unapply_sim3_points(pts, pose.rotation, pose.translation, pose.scale)

    def rebase_points(self, obj: ObjectAssets, pts: np.ndarray,
                      src_key: FrameKey, dst_key: FrameKey) -> np.ndarray:
        """Move posed R3 points from ``src_key``'s placement to ``dst_key``'s.

        The inverse of the Gaussian chain followed by the forward one: undo
        ``src``'s c2w + P3D->R3 + Sim(3) back to object-local, then re-apply
        ``dst``'s.  Used to strip root motion from ``final/tracks_2d.npz``
        tracks so they line up with a pose-pinned render.
        """
        src = self.resolve_transform(obj, src_key)
        dst = self.resolve_transform(obj, dst_key)
        if src is None or dst is None:
            return pts
        return self._place_points(self._unplace_points(pts, src), dst)
