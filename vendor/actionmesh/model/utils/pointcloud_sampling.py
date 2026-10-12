# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from enum import Enum
from typing import Optional

import torch
from actionmesh.model.utils.tensor_ops import merge_batch_time, split_batch_time
from pytorch3d.ops import sample_farthest_points
from pytorch3d.ops.utils import masked_gather


class SamplingType(str, Enum):
    """Supported point cloud sampling strategies."""

    RANDOM = "random"
    FPS = "fps"
    FPS_FULL = "fps_full"


# ---------------------------------------------------------------------------
# Low-level helpers
# ---------------------------------------------------------------------------


def _farthest_point_sample(
    points: torch.Tensor,
    n_samples: int,
    random_start_point: bool = True,
    sampling_type: SamplingType = SamplingType.FPS,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run farthest-point sampling on a (B, N, D) point cloud.

    Uses local exact farthest-point sampling on Metal or native PyTorch.
    Both paths support batched inputs and preserve gather gradients.

    Args:
        points: (B, N, D) input point cloud.
        n_samples: number of points to select.
        random_start_point: randomise the FPS seed point.
        sampling_type: ``FPS`` uses only XYZ for distance computation,
            ``FPS_FULL`` uses all D channels.

    Returns:
        sampled_points: (B, n_samples, D)
        indices: (B, n_samples) or (1, n_samples)
    """
    if points.ndim != 3:
        raise ValueError("Expected 3-D tensor (B, N, D), " f"got {points.ndim}-D")

    distance_input = points[..., :3] if sampling_type == SamplingType.FPS else points
    _, indices = sample_farthest_points(
        distance_input, K=n_samples, random_start_point=random_start_point,
    )
    sampled_points = masked_gather(points, indices)

    return sampled_points, indices


def sample_from_indices(
    points: torch.Tensor,
    indices: torch.Tensor,
) -> torch.Tensor:
    """Gather points by pre-computed indices.

    Args:
        points: (B, N_PTS, D)
        indices: (B, M) or (1, M)

    Returns:
        (B, M, D) gathered points.
    """
    if points.ndim != 3:
        raise ValueError(f"Expected 3-D points, got {points.ndim}-D")
    if indices.ndim != 2:
        raise ValueError(f"Expected 2-D indices, got {indices.ndim}-D")
    if indices.shape[0] == 1:
        indices = indices.expand(points.shape[0], -1)
    if indices.shape[0] != points.shape[0]:
        raise ValueError(
            "Batch size mismatch: "
            f"points {points.shape[0]} vs "
            f"indices {indices.shape[0]}"
        )
    return masked_gather(points, indices)


# ---------------------------------------------------------------------------
# Strategy helpers (one per sampling path)
# ---------------------------------------------------------------------------


def _sample_identity(
    points: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Not enough points to downsample — return everything unchanged."""
    indices = torch.arange(
        points.shape[1],
        device=points.device,
    ).reshape(1, -1)
    return points, indices


def _sample_random(
    points: torch.Tensor,
    n_samples: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Independent uniform random sampling per batch element."""
    batch_size = points.shape[0]
    n_pts = points.shape[1]
    indices = torch.stack(
        [torch.randperm(n_pts)[:n_samples] for _ in range(batch_size)],
    ).to(points.device)
    sampled_points = sample_from_indices(points, indices)
    return sampled_points, indices


def _sample_fps(
    points: torch.Tensor,
    n_samples: int,
    sampling_type: SamplingType,
    fps_max_points: Optional[int],
    fps_random: bool,
    fps_chunks: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Farthest-point sampling with optional random pre-sampling and chunking.

    When *fps_max_points* is set the input is first randomly reduced to at most
    that many points (but never fewer than *n_samples*) before FPS is applied,

    *fps_chunks* splits the (possibly pre-sampled) point cloud along the point
    axis and runs independent FPS on each chunk, concatenating the results.
    This can help when N is very large.
    """
    batch_size = points.shape[0]

    # Optional random pre-sampling to cap the FPS input size
    if fps_max_points is not None:
        n_pts = points.shape[1]
        n_pre = min(n_pts, max(fps_max_points, n_samples))
        pre_indices = torch.stack(
            [torch.randperm(n_pts)[:n_pre] for _ in range(batch_size)],
        ).to(points.device)
        points_pre = sample_from_indices(points, pre_indices)
    else:
        n_pre = points.shape[1]
        points_pre = points

    # If pre-sampling already gave us <= n_samples points, nothing left to do
    if n_pre <= n_samples:
        indices = pre_indices if fps_max_points is not None else torch.arange(
            points_pre.shape[1],
            device=points.device,
        ).reshape(1, -1)
        return points_pre, indices

    # Chunked FPS
    chunk_size = n_samples // fps_chunks
    points_list: list[torch.Tensor] = []
    indices_list: list[torch.Tensor] = []
    offset = 0
    for chunk in points_pre.tensor_split(fps_chunks, dim=1):
        chunk_out, chunk_indices = _farthest_point_sample(
            chunk,
            n_samples=chunk_size,
            random_start_point=fps_random,
            sampling_type=sampling_type,
        )
        points_list.append(chunk_out)
        indices_list.append(chunk_indices + offset)
        offset += chunk.shape[1]

    indices = torch.cat(indices_list, dim=1)
    if fps_max_points is not None:
        indices = pre_indices.gather(1, indices)
    sampled_points = torch.cat(points_list, dim=1)
    return sampled_points, indices


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def sample_pc(
    points: torch.Tensor,
    n_samples: int,
    sampling_type: SamplingType | str = SamplingType.RANDOM,
    fps_max_points: Optional[int] = None,
    fps_random: bool = True,
    fps_chunks: int = 1,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Sample *n_samples* points from a batched point cloud.

    Args:
        points: (B, N_PTS, D) input point cloud.
        n_samples: number of points to keep.
        sampling_type: strategy — ``RANDOM``, ``FPS``
            or ``FPS_FULL``.
        fps_max_points: if set, randomly pre-sample at
            most this many points before running FPS.
        fps_random: randomise the FPS starting point.
        fps_chunks: split the point cloud into this many
            chunks and run FPS independently on each.

    Returns:
        sampled_points: (B, n_samples, D)
        indices: (B, n_samples) or (1, n_samples)
    """
    if not isinstance(points, torch.Tensor):
        raise TypeError(f"Expected torch.Tensor, got {type(points)}")
    if points.ndim != 3:
        raise ValueError("Expected 3-D (B, N_PTS, D), " f"got {points.ndim}-D")
    if n_samples % fps_chunks != 0:
        raise ValueError(
            f"n_samples ({n_samples}) must be "
            f"divisible by fps_chunks ({fps_chunks})"
        )

    # Accept plain strings for backward compatibility
    if isinstance(sampling_type, str):
        sampling_type = SamplingType(sampling_type)

    if points.shape[1] <= n_samples:
        return _sample_identity(points)

    if sampling_type is SamplingType.RANDOM:
        return _sample_random(points, n_samples)

    if sampling_type.value.startswith("fps"):
        return _sample_fps(
            points,
            n_samples,
            sampling_type,
            fps_max_points,
            fps_random,
            fps_chunks,
        )

    raise ValueError(f"Unsupported sampling type: {sampling_type}")


def sample_pc_grouped(
    points: torch.Tensor,
    n_samples: int,
    n_grouped_frames: int,
    sampling_type: SamplingType | str = SamplingType.FPS,
    fps_max_points: Optional[int] = None,
    fps_random: bool = True,
    fps_chunks: int = 1,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Sample using the first frame, then broadcast indices
    across all frames.

    Treats the batch dimension as ``(B * n_grouped_frames)``
    and ensures every frame of each batch element shares the
    same sampled point indices.

    Args:
        points: (B*T, N_PTS, D) input point cloud where
            B*T = batch_size * n_grouped_frames.
        n_samples: number of points to keep per frame.
        n_grouped_frames: number of frames per batch element.
        sampling_type: forwarded to :func:`sample_pc`.
        fps_max_points: forwarded to :func:`sample_pc`.
        fps_random: forwarded to :func:`sample_pc`.
        fps_chunks: forwarded to :func:`sample_pc`.

    Returns:
        sampled_points: (B*T, n_samples, D)
        indices: (B*T, n_samples)
    """
    # Accept plain strings for backward compatibility
    if isinstance(sampling_type, str):
        sampling_type = SamplingType(sampling_type)

    # (B*T, N_PTS, D) -> (B, T, N_PTS, D)
    points_batched = split_batch_time(
        points,
        n_grouped_frames,
    )
    # Sample on first timestep only
    _, indices = sample_pc(
        points=points_batched[:, 0],
        n_samples=n_samples,
        sampling_type=sampling_type,
        fps_max_points=fps_max_points,
        fps_random=fps_random,
        fps_chunks=fps_chunks,
    )
    # Broadcast indices to all timesteps
    indices = indices.unsqueeze(1).repeat(
        1,
        n_grouped_frames,
        1,
    )
    indices = merge_batch_time(indices)
    sampled_points = masked_gather(points, indices)
    return sampled_points, indices
