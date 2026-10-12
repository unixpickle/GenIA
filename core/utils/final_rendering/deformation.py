# Copyright (c) Meta Platforms, Inc. and affiliates.

"""Recover saved deformation fields and warp exported canonical geometry."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from genia.core.utils.final_rendering import runtime as _runtime
from genia.core.utils.final_rendering.configuration import load_run_config
from genia.core.utils.final_rendering.geometry import GaussianCloud
from genia.core.utils.final_rendering.runtime import log


@dataclass
class _WarpShim:
    """The two attributes ``warp_gaussians_high_res`` reads off a ``Gaussian``."""

    get_xyz: Any
    get_rotation: Any


class DeformationSource:
    """Canonical Gaussians + the deformation field, read from ``final/final.pt``.

    The turntable's geometry source, in place of ``gaussians/{obj}/{frame}.ply``.
    Those exported clouds are this same warp, but ``save_compressed_ply`` (the
    default) MORTON-SORTS every file it writes, so row `i` is a different splat in
    each frame and they cannot be interpolated -- see :func:`rows_correspond`.  The
    cache is order-free and unquantized, so re-running the warp here restores the
    correspondence.  It is also faster: one load, instead of one
    :func:`load_gaussian_ply` per frame per pass.

    Serves a run with no deformation field too, returning the canonical cloud
    unwarped -- static runs, where it is simply the better-conditioned read.
    """

    def __init__(self, canonical, field, warp_kwargs, device):
        self._canonical = canonical      # {obj: GaussianCloud}, object-local
        self._field = field              # the four state-shaped dicts
        self._warp = warp_kwargs
        self._device = device
        self._failed = False

    # -- opening --------------------------------------------------------------

    @classmethod
    def open(cls, final_dir: Path, *, device: str, objects, frames,
             required: bool = False) -> "DeformationSource | None":
        """Load the source, or None (with ONE logged reason) if it cannot serve.

        ALL-OR-NOTHING per run: a partially served scene would compose provider rows
        beside PLY rows, and the correspondence this exists for would be gone again.
        Declines on a missing/broken ``final.pt``, absent canonical Gaussians, an
        object it does not carry, or a field that does not key every requested
        object at every requested frame.  Never raises -- the orbit is cosmetic
        output closing a run that may have cost hours.
        """
        try:
            return cls._open(final_dir, device, objects, frames)
        except Exception as exc:   # noqa: BLE001 — any failure means "use the PLYs"
            log(f"    (final.pt: {type(exc).__name__}: {exc} — falling back to the "
                f"exported per-frame PLYs)")
            if required:
                raise
            return None

    @classmethod
    def _open(cls, final_dir, device, objects, frames):
        from genia.core.utils.deformation import _lookup_per_frame_deformation
        from genia.core.utils.pipeline_state import (
            gaussian_blob_to_render_tensors,
            load_pipeline_cache_raw,
        )

        path = Path(final_dir) / "final.pt"
        if not path.is_file():
            log("    (no final.pt — reading geometry from the exported PLYs)")
            return None
        blob = load_pipeline_cache_raw(str(path))
        raw = blob.get("canonical_gaussians") or {}
        # Keep the four fields we need and drop the rest NOW: the slats, tokens and
        # voxel correspondences in there are most of the file.
        field = {k: blob.get(k) for k in (
            "canonical_mesh_verts", "canonical_mesh_faces",
            "canonical_mesh_per_frame_verts", "canonical_mesh_per_frame_rotations")}
        del blob

        canonical = {}
        for obj_idx in sorted(objects):
            g = raw.get(obj_idx)
            if g is None:
                log(f"    (final.pt carries no canonical gaussian for object "
                    f"{obj_idx} — reading geometry from the exported PLYs)")
                return None   # a canonical run missing ONE object; per-frame-only
                              # runs are caught before the load
            canonical[obj_idx] = GaussianCloud(
                **gaussian_blob_to_render_tensors(g, device=device))

        src = cls(canonical, field, _warp_kwargs(final_dir), device)
        # Every (object, frame) or none: `_lookup_per_frame_deformation` is the
        # export's own "is the field keyed here?" test, so the orbit and the PLY
        # writer agree by construction rather than by convention.
        # ... on the CPU the tensors already sit on: this asks only whether the
        # field is KEYED, and the lookup moves what it resolves, which would copy
        # every frame's tensors to the device to answer a question about dict keys.
        covered = [_lookup_per_frame_deformation(
            field["canonical_mesh_verts"], field["canonical_mesh_per_frame_verts"],
            field["canonical_mesh_per_frame_rotations"], o, f,
            _runtime.torch.device("cpu"), field["canonical_mesh_faces"]) is not None
            for o in canonical for f in sorted(set(frames))]
        if any(covered) and not all(covered):
            log("    (final.pt's deformation field does not cover every frame — "
                "reading geometry from the exported PLYs)")
            return None
        if any(covered):
            # Force the pytorch3d import and its kernel while there is still a clean
            # fallback, rather than mid-clip on the first real warp.
            obj = next(iter(canonical))
            probe = canonical[obj]
            src._warp_one(_WarpShim(probe.means[:8], probe.quats[:8]), obj,
                          sorted(set(frames))[0])
            log(f"    geometry from final.pt: canonical gaussians + the deformation "
                f"field ({len(canonical)} object(s), no Morton-sorted PLYs)")
        else:
            log("    geometry from final.pt: canonical gaussians (no deformation "
                "field — nothing to interpolate)")
        return src

    # -- serving --------------------------------------------------------------

    def cloud(self, obj_idx: int, frame: int) -> "GaussianCloud | None":
        """Object-local Gaussians for one exported frame — what
        ``load_gaussian_ply(gaussians/{obj}/{frame}.ply)`` would give before the
        Morton sort.  The canonical cloud when the object has no field."""
        canon = self._canonical.get(obj_idx)
        if canon is None or self._failed:
            return None
        try:
            warped = self._warp_one(_WarpShim(canon.means, canon.quats),
                                    obj_idx, frame)
        except Exception as exc:   # noqa: BLE001
            log(f"    ! warp failed ({type(exc).__name__}: {exc}) — falling back to "
                f"the exported PLYs for the rest of this render")
            self._failed = True
            return None
        if warped is None:
            return canon
        means, quats = warped
        # scales/opacities/SH are the canonical tensors, SHARED not copied: the warp
        # does not touch them.  That is what makes two frames of this source
        # correspond row for row by construction (`compose` clones before mutating).
        return GaussianCloud(means=means, quats=quats, scales=canon.scales,
                             opacities=canon.opacities, sh=canon.sh)

    def _warp_one(self, shim, obj_idx: int, frame: int):
        from genia.core.utils.deformation import (
            _lookup_per_frame_deformation,
            warp_gaussians_high_res,
        )

        resolved = _lookup_per_frame_deformation(
            self._field["canonical_mesh_verts"],
            self._field["canonical_mesh_per_frame_verts"],
            self._field["canonical_mesh_per_frame_rotations"],
            obj_idx, int(frame), _runtime.torch.device(self._device),
            self._field["canonical_mesh_faces"])
        if resolved is None:
            return None
        canon_verts, frame_verts, frame_rots, faces = resolved
        with _runtime.torch.no_grad():
            return warp_gaussians_high_res(shim, canon_verts, frame_verts,
                                           frame_rots, faces=faces, **self._warp)

    def provider(self):
        """The callback :meth:`FinalRun.compose` takes as ``geometry``."""
        return lambda obj_idx, key: self.cloud(obj_idx, key.frame)


def _warp_kwargs(final_dir: Path) -> dict:
    """The run's OWN ``deformation_warp`` knobs, so re-warping here reproduces the
    warp its per-frame PLYs were exported with.  Defaults mirror
    ``core/configs/deformation_warp/default.yaml``."""
    cfg = (load_run_config(Path(final_dir)) or {}).get("deformation_warp") or {}
    return {
        "K": int(cfg.get("knn_k", 4)),
        "eps": float(cfg.get("knn_eps", 1e-8)),
        "chunk_size": int(cfg.get("knn_chunk_size", 131072)),
    }
