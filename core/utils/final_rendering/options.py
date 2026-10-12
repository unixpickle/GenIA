# Copyright (c) Meta Platforms, Inc. and affiliates.

"""Shared option declarations for track overlays and figure crops."""

from __future__ import annotations

from genia.core.utils.final_rendering.registry import Option

_TRACK_DRAW_OPTIONS = {
    "track_points": Option("int", 40, "cap on tracked points drawn"),
    "track_subsample": Option("float", 0.5, "fraction of the capped points to "
                              "keep, spread evenly through space (1 = all)"),
    "track_trail": Option("int", 0, "trail length in frames (0 = whole history)"),
    "track_width": Option("int", 2, "trail line width in px"),
    "track_opacity": Option("float", 1.0, "overlay opacity (1 = opaque)"),
}


_TRACK_OPTIONS = {
    "tracks": Option("bool", True, "overlay final/tracks_2d.npz 3D tracks "
                     "when the run has them"),
    **_TRACK_DRAW_OPTIONS,
}


_FIGURE_CROP_OPTIONS = {
    "crop": Option("bool", False, "trim every frame to ONE box covering the object "
                   "across the whole sequence, at the frame's aspect ratio. OFF by "
                   "default: figures are cropped later, to a box shared by the "
                   "whole row; a per-run box here would pre-trim the pixels that "
                   "box needs"),
    "crop_pad": Option("float", 0.05, "padding around that box, as a fraction of "
                       "its longer side"),
    "supersample": Option("float", 2.0, "render at this multiple of the run's "
                          "resolution, so whoever crops has the pixels to do it with"),
}


_TRAIN_CAMERA_OPTIONS = {
    "video": Option("bool", True, "also encode the frames to mp4"),
    "space": Option("str", "world", "pose space: world | camera"),
    "perframe": Option("bool", True, "use per-frame warped PLYs when present"),
}
