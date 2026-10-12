# Copyright (c) Meta Platforms, Inc. and affiliates.

"""Output paths, overwrite policy, image writing, and video encoding."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Iterator

import numpy as np
from genia.core.utils.final_rendering import runtime as _runtime
from genia.core.utils.final_rendering.images import flatten_alpha
from genia.core.utils.frame_key import FrameKey, frame_key_stem, group_by_view


@dataclass
class RenderContext:
    """Everything a renderer needs besides the run itself, plus its output sinks."""

    out_dir: Path                 # {viz_root}/{render.out_sub}
    device: str
    frame_keys: list[FrameKey]    # after --frames / --views filtering
    objects: list[int]            # after --objects filtering
    H: int
    W: int
    src_hw: tuple[int, int]       # the run's own resolution, for scale_intrinsics
    bg: np.ndarray                # (3,) float RGB the foreground composites on
    fps: int
    overwrite: bool
    dry_run: bool
    opts: dict[str, Any]
    written: list[Path] = field(default_factory=list)

    def path(self, *parts: str) -> Path:
        return self.out_dir.joinpath(*parts)

    def wants(self, path: Path) -> bool:
        """False when this output is already done, or we are only planning.

        Either way the path counts as an output, so ``--dry-run`` reports the
        full set and a re-run reports what is already there.
        """
        if self.dry_run or (path.exists() and not self.overwrite):
            self.written.append(path)
            return False
        return True

    def write_png(self, path: Path, arr: np.ndarray) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        _runtime.Image.fromarray(arr).save(path)
        self.written.append(path)
        return path

    def write_video(self, path: Path, frames: Iterable[np.ndarray | Path]) -> Path | None:
        """Encode RGBA frames (arrays, or PNGs to stream off disk) to H.264 mp4.

        Frames are flattened onto ``ctx.bg`` and fed to the encoder one at a
        time, so a long orbit at high resolution never holds the whole clip.
        """
        if not self.wants(path):
            return None
        import imageio.v2 as imageio

        stream = self._rgb_frames(frames)
        first = next(stream, None)
        if first is None:
            return None
        h, w = first.shape[0] - first.shape[0] % 2, first.shape[1] - first.shape[1] % 2
        path.parent.mkdir(parents=True, exist_ok=True)
        writer = imageio.get_writer(str(path), fps=self.fps, codec="libx264",
                                    quality=8, macro_block_size=None)
        n = 0
        try:
            for frame in (first, *stream):
                if frame.shape[:2] != first.shape[:2]:
                    continue  # a stale PNG at another resolution
                writer.append_data(frame[:h, :w])
                n += 1
        finally:
            writer.close()
        if n < 2:  # a 1-frame clip is not worth a video file
            path.unlink(missing_ok=True)
            return None
        self.written.append(path)
        return path

    def _rgb_frames(self, frames: Iterable[np.ndarray | Path]) -> Iterator[np.ndarray]:
        for frame in frames:
            if isinstance(frame, Path):
                if not frame.is_file():
                    continue
                with _runtime.Image.open(frame) as im:
                    frame = np.asarray(im.convert("RGBA"))
            yield flatten_alpha(frame, self.bg)

    def write_per_view_videos(self, prefix: str, keys: list[FrameKey]) -> None:
        """One mp4 per view from the per-frame PNGs a render just wrote:
        ``{prefix}.mp4`` for view 0, ``{prefix}_vNN.mp4`` for the rest.  PNGs are
        streamed back off disk so frames skipped this run still make it in."""
        for view, vkeys in group_by_view(keys).items():
            suffix = "" if view == 0 else f"_v{view:02d}"
            self.write_video(self.path(f"{prefix}{suffix}.mp4"),
                             [self.path(f"{frame_key_stem(k)}.png") for k in vkeys])
