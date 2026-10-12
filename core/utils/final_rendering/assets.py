# Copyright (c) Meta Platforms, Inc. and affiliates.

"""Metadata types for exported objects, poses, cameras, and tracks."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from genia.core.utils.frame_key import FrameKey


@dataclass
class Transform:
    """One object's Sim(3) at one frame key, plus its per-frame asset."""

    key: FrameKey
    translation: np.ndarray  # (3,)
    rotation: np.ndarray     # (4,) wxyz
    scale: np.ndarray        # (3,)
    ply_perframe: Path | None = None
    mesh_perframe: Path | None = None   # per-frame deformed mesh, object-local P3D
    index: int = 0           # row into the object's dc_offsets / sh_rest arrays


@dataclass
class ObjectAssets:
    """Everything ``final/`` holds for one reconstructed object."""

    obj_idx: int
    ply: Path | None                       # canonical Gaussian, object-local P3D
    mesh: Path | None                      # canonical GLB, object-local P3D
    transforms: dict[FrameKey, Transform]
    dc_offsets: np.ndarray | None = None   # (T, N, 3)
    sh_rest: np.ndarray | None = None      # (T, N, K, 3)


@dataclass
class Camera:
    c2w: np.ndarray            # (4, 4)
    K: np.ndarray | None       # (3, 3)


@dataclass
class Tracks:
    """3D object tracks, from ``final/tracks_2d.npz`` or ``final/tracks_3d.npz``.

    ``xyz[obj]`` is ``(T, P, 3)`` in the same R3 space ``compose(space="world")``
    produces: canonical anchors warped by the per-frame deformation, then placed
    by that frame's Sim(3).  ``frames[t]`` is the temporal index of row ``t``.
    """

    frames: list[int]
    xyz: dict[int, np.ndarray]
    vis: dict[int, np.ndarray]

    def row_for(self, frame: int) -> int | None:
        try:
            return self.frames.index(frame)
        except ValueError:
            return None


def _obj_xyz_members(data) -> dict[int, np.ndarray]:
    """The ``obj_{i}_xyz`` ``(T, P, 3)`` members of an open npz, by object index.

    The reader half of ``interpolation._obj_xyz_members``, which is the single
    writer of that key convention across BOTH track artifacts -- so the two
    readers here (dense ``tracks_2d.npz``, sparse ``tracks_3d.npz``) cannot drift
    apart from each other or from it.
    """
    return {int(m.group(1)): np.asarray(data[name], dtype=np.float32)
            for name in data.files
            if (m := re.fullmatch(r"obj_(\d+)_xyz", name))}
