# Copyright (c) Meta Platforms, Inc. and affiliates.

"""Render visualizations from a pipeline ``final/`` directory.

FINAL renders its orbit, world-space and track-overlay views through
:func:`process_run`; :func:`main` is the command-line entry point.

Point this at any run's ``final/`` folder -- ``results/[{experiment}/]{dataset}/
{scene}/{timestamp}/final`` -- and it re-renders the exported assets
(``poses.json`` + ``gaussians/`` + ``meshes/``) into ``final/viz/``.  Nothing
from the pipeline runtime is needed: the run is reconstructed purely from the
files ``FINAL`` wrote, so any run with a ``final/`` folder can be rendered.
Renders are plug-ins registered with :func:`renderer` and selected on the CLI;
run with ``--list`` to see them."""

from __future__ import annotations

import argparse
import traceback
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from genia.core.utils.final_rendering.builtins import (
    register_builtin_renderers as _register_builtin_renderers,  # noqa: F401 (plugin registration)
)
from genia.core.utils.final_rendering.context import RenderContext
from genia.core.utils.final_rendering.discovery import discover_final_dirs, parse_run_path
from genia.core.utils.final_rendering.registry import (
    RENDERERS,
    check_options_are_claimed,
    resolve_options,
)
from genia.core.utils.final_rendering.run import FinalRun
from genia.core.utils.final_rendering.runtime import _lazy_imports, log

if TYPE_CHECKING:
    from genia.core.utils.final_rendering.registry import Renderer



def parse_index_spec(spec: str, available: list[int]) -> list[int]:
    """``"0,4"`` / ``"0-15"`` / ``"0-30:5"`` / ``"all"`` -> a list of indices."""
    if spec.strip().lower() in ("", "all"):
        return list(available)
    picked: list[int] = []
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        step = 1
        if ":" in part:
            part, step_s = part.split(":", 1)
            step = int(step_s)
        if "-" in part:
            lo, hi = part.split("-", 1)
            picked.extend(range(int(lo), int(hi) + 1, step))
        else:
            picked.append(int(part))
    keep = set(available)
    return [i for i in picked if i in keep]


def parse_bg(spec: str) -> np.ndarray:
    named = {"white": (1.0, 1.0, 1.0), "black": (0.0, 0.0, 0.0), "gray": (0.5, 0.5, 0.5)}
    if spec in named:
        return np.asarray(named[spec], dtype=np.float32)
    vals = [float(v) for v in spec.split(",")]
    if len(vals) != 3:
        raise SystemExit(f"--bg wants a name or 'R,G,B', got {spec!r}")
    scale = 255.0 if max(vals) > 1.0 else 1.0
    return np.asarray([v / scale for v in vals], dtype=np.float32)


def parse_resolution(spec: str | None, native: tuple[int, int]) -> tuple[int, int]:
    if not spec:
        return native
    if "x" in spec.lower():
        w, h = spec.lower().split("x", 1)
        return int(h), int(w)
    return int(spec), int(spec)


def render_menu(show_options: bool = False) -> str:
    """The registered renders as text: a ``name — help`` menu, plus each
    render's ``-O`` options when ``show_options`` (what ``--list`` prints; the
    compact form is the ``--help`` epilog)."""
    lines = ["available renders (select with -r NAME, repeatable; omit -r or "
             "pass -r all for every one):", ""]
    for name in sorted(RENDERERS):
        r = RENDERERS[name]
        gpu = "" if r.needs_gpu else "  [no GPU needed]"
        lines.append(f"  {name}{gpu}\n      {r.help}")
        if show_options:
            for key, opt in r.options.items():
                assign = f"-O {name}.{key}={opt.default!r}"
                lines.append(f"      {assign:<44} ({opt.type}) {opt.help}")
        lines.append("")
    return "\n".join(lines)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=__doc__.split("\n\n")[0],
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=render_menu(),
    )
    p.add_argument("final_dirs", nargs="*",
                   help="one or more final/ dirs (globs and run dirs also work)")
    p.add_argument("-r", "--render", action="append", default=[], dest="renders",
                   help="render name (repeatable); omit or 'all' runs every render")
    p.add_argument("-O", "--opt", action="append", default=[], dest="opts",
                   metavar="KEY=VALUE",
                   help="render option; scope it as RENDER.KEY=VALUE")
    p.add_argument("--list", action="store_true", help="list renders and exit")
    p.add_argument("--out", default=None,
                   help="output root (default: {final}/viz); several runs get "
                        "one subdir each under it")
    p.add_argument("--frames", default="all",
                   help="frame filter: 'all', '0,4,8', '0-15', '0-30:5'")
    p.add_argument("--views", default="all", help="view filter, same syntax")
    p.add_argument("--objects", default="all", help="object-index filter, same syntax")
    p.add_argument("--resolution", default=None,
                   help="render size: 'N' or 'WxH' (default: the run's own)")
    p.add_argument("--bg", default="white",
                   help="background the foreground composites on (white/black/gray/R,G,B)")
    p.add_argument("--fps", type=int, default=24, help="video frame rate")
    p.add_argument("--device", default=None, help="torch device (default: CUDA, Metal, or CPU)")
    p.add_argument("--overwrite", action="store_true", help="re-render existing files")
    p.add_argument("--dry-run", action="store_true",
                   help="print what would be written, render nothing")
    return p


def viz_roots(final_dirs: list[Path], out: str | None) -> dict[Path, Path]:
    """Where each run writes: its own ``{final}/viz``, or a subdir of ``--out``.

    A shared ``--out`` over several runs gets one subdir per run, so their
    identically-named outputs (``orbit.mp4``, ...) cannot overwrite each other.
    """
    if out is None:
        return {d: d / "viz" for d in final_dirs}
    root = Path(out)
    if len(final_dirs) == 1:
        return {final_dirs[0]: root}
    runs = {d: parse_run_path(d) for d in final_dirs}
    scenes = [r.scene for r in runs.values()]
    unique = len(set(scenes)) == len(scenes)
    return {d: root / (r.scene if unique else f"{r.scene}_{r.timestamp}")
            for d, r in runs.items()}


def process_run(final_dir: Path, selected: list[Renderer], args,
                viz_root: Path) -> None:
    run = FinalRun(final_dir)
    log(f"\n=== {run.path.label}")
    log(f"    {run.summary()}")

    frames = set(parse_index_spec(args.frames, run.frames))
    views = set(parse_index_spec(args.views, run.views))
    keys = [k for k in run.frame_keys if k.frame in frames and k.view in views]
    objects = parse_index_spec(args.objects, sorted(run.objects))
    src_hw = run.resolution
    H, W = parse_resolution(args.resolution, src_hw)
    bg = parse_bg(args.bg)

    for rdr in selected:
        ctx = RenderContext(
            out_dir=viz_root / rdr.out_sub,
            device=args.device,
            frame_keys=keys,
            objects=objects,
            H=H, W=W, src_hw=src_hw,
            bg=bg,
            fps=args.fps,
            overwrite=args.overwrite,
            dry_run=args.dry_run,
            opts=resolve_options(rdr, args.opts),
        )
        log(f"  [{rdr.name}] -> {ctx.out_dir}")
        rdr.fn(run, ctx)
        verb = "would write" if args.dry_run else "wrote/kept"
        log(f"  [{rdr.name}] {verb} {len(ctx.written)} file(s)")


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    if args.list:
        log(render_menu(show_options=True))
        return 0
    if not args.final_dirs:
        parser.print_usage()
        log("error: no final/ directory given (use --list to see the renders)")
        return 2

    # No -r means "everything", same as -r all -- the useful default for a
    # first look at a run.
    names = (sorted(RENDERERS) if not args.renders or "all" in args.renders
             else args.renders)
    unknown = [n for n in names if n not in RENDERERS]
    if unknown:
        log(f"error: unknown render(s) {unknown}; available: {sorted(RENDERERS)}")
        return 2
    selected = [RENDERERS[n] for n in names]
    check_options_are_claimed(selected, args.opts)

    final_dirs = discover_final_dirs(args.final_dirs)
    if not final_dirs:
        log("error: no final/ directory matched")
        return 2

    needs_gpu = any(r.needs_gpu for r in selected) and not args.dry_run
    _lazy_imports(gpu=needs_gpu)
    if needs_gpu:
        from genia.core.device import get_device
        args.device = str(get_device(args.device))

    roots = viz_roots(final_dirs, args.out)
    failures = 0
    for final_dir in final_dirs:
        try:
            process_run(final_dir, selected, args, roots[final_dir])
        except Exception as exc:
            failures += 1
            log(f"  ! {final_dir}: {type(exc).__name__}: {exc}")
            if len(final_dirs) == 1:  # nothing else to salvage -- show where
                traceback.print_exc()
    if failures:
        log(f"\n{failures}/{len(final_dirs)} run(s) failed")
    return 1 if failures else 0
