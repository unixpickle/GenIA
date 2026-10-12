# Copyright (c) Meta Platforms, Inc. and affiliates.

"""Register render plugins and validate their named options."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Callable

if TYPE_CHECKING:
    from genia.core.utils.final_rendering.context import RenderContext
    from genia.core.utils.final_rendering.run import FinalRun



@dataclass
class Option:
    """One renderer-specific knob, settable via ``-O [render.]key=value``."""

    type: str          # "int" | "float" | "bool" | "str" | "int?"
    default: Any
    help: str = ""


@dataclass
class Renderer:
    name: str
    help: str
    fn: Callable[[FinalRun, "RenderContext"], None]
    options: dict[str, Option]
    needs_gpu: bool
    # Where under {final}/viz/ this render writes. Defaults to its name; the
    # track-overlay family sets a nested path so the annotated variants group
    # under one folder ("track_overlays/train_views") instead of scattering
    # top-level names that read like the clean renders they are derived from.
    out_sub: str = ""


RENDERERS: dict[str, Renderer] = {}


def renderer(name: str, help: str, options: dict[str, Option] | None = None,
             needs_gpu: bool = True, out_sub: str = ""):
    """Register a render under ``name``; the function gets ``(run, ctx)``.

    Outputs go through ``ctx.write_png`` / ``ctx.write_video``, which own the
    ``--dry-run`` / ``--overwrite`` protocol and the tally.  Options declared
    here are exposed as ``-O <name>.<key>=<value>`` and shown by ``--list``.
    """
    def deco(fn):
        RENDERERS[name] = Renderer(name, help, fn, dict(options or {}), needs_gpu,
                                   out_sub or name)
        return fn
    return deco


def _coerce(value: str, kind: str) -> Any:
    if kind.endswith("?") and value.lower() in ("none", "null", ""):
        return None
    base = kind.rstrip("?")
    if base == "bool":
        if value.lower() in ("1", "true", "yes", "on"):
            return True
        if value.lower() in ("0", "false", "no", "off"):
            return False
        raise ValueError(f"expected a boolean, got {value!r}")
    return {"int": int, "float": float, "str": str}[base](value)


def _split_opt(item: str) -> tuple[str, str, str]:
    """``"[render.]key=value"`` -> ``(render_or_empty, key, value)``."""
    key, sep, value = item.partition("=")
    if not sep:
        raise SystemExit(f"--opt needs KEY=VALUE, got {item!r}")
    target, _, name = key.rpartition(".")
    return target, name, value


def resolve_options(rdr: Renderer, raw: list[str]) -> dict[str, Any]:
    """Merge ``-O`` assignments over the defaults; scoped beats global."""
    opts = {k: o.default for k, o in rdr.options.items()}
    scoped: dict[str, Any] = {}
    for item in raw:
        target, name, value = _split_opt(item)
        if target and target != rdr.name:
            continue                              # another render's knob
        if name not in rdr.options:
            if not target:
                continue                          # a global knob owned elsewhere
            raise SystemExit(f"unknown option {name!r} for render {rdr.name!r}; "
                             f"available: {sorted(rdr.options) or 'none'}")
        (scoped if target else opts)[name] = _coerce(value, rdr.options[name].type)
    return {**opts, **scoped}


def check_options_are_claimed(selected: list[Renderer], raw: list[str]) -> None:
    """Reject a ``-O`` no selected render owns, instead of silently dropping it."""
    names = {r.name for r in selected}
    for item in raw:
        target, name, _value = _split_opt(item)
        if target and target not in names:
            raise SystemExit(f"-O {item}: no selected render named {target!r} "
                             f"(selected: {sorted(names)})")
        if not target and not any(name in r.options for r in selected):
            raise SystemExit(f"-O {item}: no selected render has an option "
                             f"{name!r} (see --list)")
