# Copyright (c) Meta Platforms, Inc. and affiliates.

"""Resolve held-out cameras and rebase them into the exported run’s frame."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, NamedTuple

import numpy as np
from genia.core.utils.final_rendering.configuration import load_run_config
from genia.core.utils.final_rendering.runtime import log
from genia.core.utils.frame_key import FrameKey, frame_key_stem

if TYPE_CHECKING:
    from genia.core.utils.final_rendering.run import FinalRun



@dataclass
class _CameraSequence:
    """The three fields the test-camera resolvers read off a ``Sequence``.

    They want a live pipeline object; a finished run has only ``final/`` plus its
    config snapshot.  Everything they need from it is the render size and the
    frame keys, both of which the run knows -- so a shim is enough, and the
    resolvers stay the single definition of where each dataset's test cameras
    come from rather than being re-derived here.
    """

    H: int
    W: int
    frame_keys: list[FrameKey]


def _rebase_c2w(run: FinalRun, key: FrameKey, input_c2w, test_c2w) -> np.ndarray:
    """A dataset-world camera, expressed in the space ``compose(space="world")`` builds.

    The two are the same frame for every dataset but OursActionBench, whose
    ``FrameData.c2w`` is identity: its fitted input camera is baked into the
    per-frame OBJECT pose instead, so a run's "world" is that input camera's own
    space while ``camera.json``'s test cameras live in the OAB world.  Composing
    the run's own c2w with the input camera's inverse carries one to the other --
    which is the read-only half of what ``export_oursactionbench_eval_assets``
    does when it lifts the geometry the other way before rendering.
    """
    cam = run.camera(key)
    run_c2w = (np.eye(4) if cam is None else np.asarray(cam.c2w)).astype(np.float64)
    return (run_c2w @ np.linalg.inv(np.asarray(input_c2w, np.float64))
            @ np.asarray(test_c2w, np.float64)).astype(np.float32)


def _src_frame(src: str) -> int:
    """The timestamp a resolver's ``renders_{train,test}/{f}.png`` names.

    The resolvers pad it differently per dataset (OAB test is 2-digit, train
    3-digit), which is exactly why the pairing below goes through this rather
    than through position in the returned list.
    """
    return int(Path(src).stem)


class TestView(NamedTuple):
    """One held-out test camera, ready to render: where to write it, which
    timestamp's geometry it sees, and its (K, c2w) in the run's own world."""

    stem: str
    key: FrameKey
    K: np.ndarray
    c2w: np.ndarray


def resolve_test_cameras(run: FinalRun, hw: tuple[int, int]) -> list[TestView]:
    """This run's held-out test views, with intrinsics scaled to ``hw``.

    Empty for a dataset with no GT test split, when the run has no config snapshot
    to name that dataset, or when the dataset itself is not on disk (the
    cameras live beside the data, not in ``final/``) -- every one of which is a
    reason to render nothing rather than to guess a viewpoint.
    """
    if not run.frame_keys:
        return []
    from genia.core.utils import eval_assets_export as eae

    # The three datasets whose FINAL writes a SCORED `renders_test/`
    # (`genia.core.utils.config.GT_TEST_VIEW_DATASETS`), through the same resolvers
    # `core/utils/colmap_export.py` uses -- so the overlay looks through exactly
    # the cameras the NVS metrics were computed from.  The flag says whether the
    # dataset gives ONE test camera per timestamp (OAB) or a fixed set of views
    # for the whole run (GSO / CO3D).
    sources = {
        "gso": (eae.gso_test_cameras, False),
        "co3d": (eae.co3d_test_cameras, False),
        "oursactionbench": (eae.oursactionbench_test_cameras, True),
    }
    cfg_raw = load_run_config(run.dir)
    name = (cfg_raw.get("dataset") or {}).get("name")
    if name not in sources:
        log(f"    ! {name or 'this run'} has no GT test split — no test cameras")
        return []
    resolver, per_frame = sources[name]
    if name == "co3d" and (cfg_raw.get("processing") or {}).get(
            "camera_poses_source") != "gt":
        # The predicted-pose branch reads the target cameras back off a live
        # Sequence, which a finished run cannot rebuild.
        log("    ! CO3D with predicted cameras — test cameras need the live run")
        return []
    from omegaconf import OmegaConf

    cfg = OmegaConf.create(cfg_raw)
    shim = _CameraSequence(*hw, run.frame_keys)
    try:
        cams = resolver(cfg, shim)
    except (FileNotFoundError, OSError, KeyError) as exc:
        log(f"    ! could not read {name} test cameras ({exc}) — skipping")
        return []
    if not cams:
        log(f"    ! {name} exposes no test cameras for this scene")
        return []
    if not per_frame:
        # A static split: every view sees the run's single timestamp, so the stem
        # comes from the render each camera belongs to (`renders_test/010.png`).
        key = run.frame_keys[0]
        return [TestView(Path(src).stem, key, K, c2w) for K, c2w, src, _ in cams]
    # Per-frame: pair by TIMESTAMP, not by position in the two lists. Both
    # resolvers silently skip a frame camera.json does not cover, and a positional
    # zip would then hand one frame's test camera another frame's input camera.
    keys = {k.frame: k for k in run.frame_keys if k.view == run.views[0]}
    inputs = {_src_frame(src): c2w
              for _, c2w, src, _ in eae.oursactionbench_input_cameras(cfg, shim)}
    out = []
    for K, c2w, src, _ in cams:
        frame = _src_frame(src)
        key, input_c2w = keys.get(frame), inputs.get(frame)
        if key is None or input_c2w is None:
            continue
        out.append(TestView(frame_key_stem(key), key, K,
                            _rebase_c2w(run, key, input_c2w, c2w)))
    return out
