# Copyright (c) Meta Platforms, Inc. and affiliates.

"""Load built-in renderers in their established registration order."""

from importlib import import_module


def register_builtin_renderers() -> None:
    """Import each plugin once; Python's module cache makes this idempotent."""
    for module in ("views", "orbit", "world", "contact_sheet"):
        import_module(f"{__package__}.{module}")


register_builtin_renderers()
