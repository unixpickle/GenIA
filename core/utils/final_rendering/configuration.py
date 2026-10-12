# Copyright (c) Meta Platforms, Inc. and affiliates.

"""Read saved run configuration and resolve world-up conventions."""

from __future__ import annotations

from pathlib import Path

import numpy as np

_WORLD_UP = np.array([0.0, -1.0, 0.0])


_WORLD_UP_DATASETS = {"gso"}


def load_run_config(final_dir: Path) -> dict:
    """The Hydra config snapshot beside ``final/``, or ``{}``.

    A run is otherwise rebuilt from ``final/`` alone; this is the one thing on
    disk that records what the run was OF -- its dataset, and whether it kept
    that dataset's cameras.  A hand-assembled ``final/`` with no snapshot
    answers ``{}``, and every caller has to work from there.
    """
    import yaml

    cfg_path = Path(final_dir).parent / "config.yaml"
    if not cfg_path.is_file():
        return {}
    try:
        return yaml.safe_load(cfg_path.read_text()) or {}
    except yaml.YAMLError:
        return {}


def has_dataset_world_up(final_dir: Path) -> bool:
    """Does THIS run's world frame have a known vertical?

    Two conditions, both recorded in the run's own ``config.yaml`` beside
    ``final/``: the dataset's world frame has a known up, and the run kept that
    dataset's cameras — predicted poses (map-anything) put the object in an
    arbitrary frame, where the dataset's up means nothing.  A run with no
    config (a hand-assembled ``final/``) answers False, the safe direction.
    """
    cfg = load_run_config(final_dir)
    return (cfg.get("dataset", {}).get("name") in _WORLD_UP_DATASETS
            and cfg.get("processing", {}).get("camera_poses_source") == "gt")


def resolve_up(spec: str, final_dir: Path) -> np.ndarray | None:
    """``-O orbit.up=...`` -> the orbit's vertical, or None for the camera's own."""
    spec = (spec or "auto").strip().lower()
    if spec == "camera":
        return None
    if spec == "auto":
        return _WORLD_UP if has_dataset_world_up(final_dir) else None
    if spec == "world":
        return _WORLD_UP
    # ValueError, not SystemExit: this also runs inside FINAL, where killing the
    # process would skip the timing summary at the end of a multi-hour run.
    parts = spec.split(",")
    try:
        if len(parts) != 3:
            raise ValueError
        return np.array([float(p) for p in parts], dtype=np.float64)
    except ValueError:
        raise ValueError(
            f"orbit.up wants auto|camera|world|'x,y,z', got {spec!r}") from None
