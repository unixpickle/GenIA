# Copyright (c) Meta Platforms, Inc. and affiliates.

"""Deferred image/GPU dependencies shared by exported-result renderers."""

from __future__ import annotations

import os
import sys

from genia.core.paths import SAM3D_OBJECTS_ROOT

os.environ.setdefault("LIDRA_SKIP_INIT", "1")
if str(SAM3D_OBJECTS_ROOT) not in sys.path:
    sys.path.append(str(SAM3D_OBJECTS_ROOT))

torch = None


Image = None


render_gaussian_params = None


p3d_to_r3_positions = None


p3d_to_r3_quaternions = None


quaternion_to_matrix = None


quaternion_invert = None


quaternion_multiply = None


def _lazy_imports(gpu: bool = True) -> None:
    """Import PIL (+ torch, the pipeline render and convention helpers on GPU)."""
    global torch, Image
    global render_gaussian_params, bulk_keep_mask, visible_bulk_mask
    global oversize_splat_mask, connected_bulk_mask
    global p3d_to_r3_positions, p3d_to_r3_quaternions
    global quaternion_to_matrix, quaternion_invert, quaternion_multiply

    if Image is None:
        from PIL import Image as _Image
        Image = _Image
    if not gpu or torch is not None:
        return

    import torch as _torch
    from genia.core.utils.quaternion_ops import p3d_to_r3_positions as _p2r_pos
    from genia.core.utils.quaternion_ops import p3d_to_r3_quaternions as _p2r_quat
    from genia.core.utils.quaternion_ops import quaternion_invert as _qinv
    from genia.core.utils.quaternion_ops import quaternion_multiply as _qmul
    from genia.core.utils.quaternion_ops import quaternion_to_matrix as _q2m
    from genia.core.utils.rendering import bulk_keep_mask as _bulk_keep
    from genia.core.utils.rendering import connected_bulk_mask as _connected
    from genia.core.utils.rendering import oversize_splat_mask as _oversize
    from genia.core.utils.rendering import render_gaussian_params as _render
    from genia.core.utils.rendering import visible_bulk_mask as _visible_bulk

    torch = _torch
    quaternion_invert, quaternion_multiply, quaternion_to_matrix = _qinv, _qmul, _q2m
    p3d_to_r3_positions, p3d_to_r3_quaternions = _p2r_pos, _p2r_quat
    render_gaussian_params = _render
    bulk_keep_mask = _bulk_keep
    visible_bulk_mask = _visible_bulk
    oversize_splat_mask = _oversize
    connected_bulk_mask = _connected


def log(msg: str) -> None:
    print(msg, flush=True)
