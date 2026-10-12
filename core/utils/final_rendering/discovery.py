# Copyright (c) Meta Platforms, Inc. and affiliates.

"""Discover exported runs and parse their experiment, scene, and timestamp."""

from __future__ import annotations

import glob as _glob
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from genia.core.utils.final_rendering.runtime import log

_TIMESTAMP_RE = re.compile(r"^\d{8}_\d{6}$")


_UNKNOWN = "unknown"


@dataclass(frozen=True)
class RunPath:
    """Identity of a run, parsed from its ``final/`` path."""

    final_dir: Path
    experiment: str | None
    dataset: str
    scene: str
    timestamp: str

    @property
    def label(self) -> str:
        parts = [self.experiment, self.dataset, self.scene, self.timestamp]
        return "/".join(p for p in parts if p)


def parse_run_path(final_dir: Path) -> RunPath:
    """Split a ``final/`` path into its ``results/`` identity components.

    Anchors on the ``results`` component when present (the ``{experiment}``
    segment is optional, so counting from the right alone is ambiguous); falls
    back to positional parsing for results trees rooted elsewhere.  Identity is
    metadata: it labels the log lines and names the ``--out`` subdirs.
    """
    final_dir = final_dir.resolve()
    parts = list(final_dir.parts)

    anchors = [i for i, p in enumerate(parts) if p == "results"]
    if anchors:
        rel = parts[anchors[-1] + 1 : -1]  # drop the trailing "final"
        if len(rel) == 4:
            experiment, dataset, scene, timestamp = rel
            return RunPath(final_dir, experiment, dataset, scene, timestamp)
        if len(rel) == 3:
            dataset, scene, timestamp = rel
            return RunPath(final_dir, None, dataset, scene, timestamp)

    # Fallback: read what we can off the tail, tolerate a shallow tree.
    dataset, scene, timestamp = ([_UNKNOWN] * 3 + parts[:-1])[-3:]
    return RunPath(final_dir, None, dataset, scene,
                   timestamp if _TIMESTAMP_RE.match(timestamp) else _UNKNOWN)


def discover_final_dirs(patterns: Iterable[str]) -> list[Path]:
    """Expand CLI paths (literal, glob, or a run dir holding ``final/``)."""
    out: list[Path] = []
    for pattern in patterns:
        hits = [Path(p) for p in sorted(_glob.glob(pattern))] or [Path(pattern)]
        for hit in hits:
            if (hit / "poses.json").is_file():
                out.append(hit)
            elif (hit / "final" / "poses.json").is_file():
                out.append(hit / "final")
            else:
                log(f"  ! skipping {hit}: no poses.json (or final/poses.json)")
    return list(dict.fromkeys(p.resolve() for p in out))
