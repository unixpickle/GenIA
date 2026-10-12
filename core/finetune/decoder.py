# Copyright (c) Meta Platforms, Inc. and affiliates.

"""Decoder precision, activation checkpointing, and shared LoRA setup."""

from __future__ import annotations

import torch
from genia.core.utils.slat_decode import _get_or_build_wrapped_decoder


def _ft_lora_init(ft_config) -> str:
    return str(getattr(ft_config, "lora_init", "default"))


def _ft_use_dora(ft_config) -> bool:
    return bool(getattr(ft_config, "use_dora", False))


def _ft_rs_scaling(ft_config) -> bool:
    return bool(getattr(ft_config, "lora_rs_scaling", False))


def _ft_lora_targets(ft_config) -> str:
    return str(getattr(ft_config, "lora_targets", "all"))


def _ft_random_bg_seed(ft_config):
    return getattr(ft_config, "random_background_seed", None)


def _ft_decoder_autocast_bf16(ft_config, device) -> bool:
    configured = getattr(ft_config, "decoder_autocast_bf16", None)
    if configured is not None:
        return bool(configured)
    return torch.device(device).type != "mps"


def _prepare_finetuning_decoder(decoder, ft_config, device) -> bool:
    """Resolve training precision before constructing the shared adapter copy."""
    autocast = _ft_decoder_autocast_bf16(ft_config, device)
    if torch.device(device).type == "mps" and not autocast:
        # Disabling autocast alone leaves SAM's torso weights/activations FP16.
        # The paired pretrained-pillow experiment retained geometry only with
        # full FP32 decoding; checkpointing bounds its activation memory.
        decoder.float()
        decoder.dtype = torch.float32
    return autocast


def _ft_decoder_checkpoint(ft_config) -> bool:
    return bool(getattr(ft_config, "decoder_checkpoint", False))


def _ft_decoder_checkpoint_min_voxels(ft_config, device) -> int:
    configured = getattr(ft_config, "decoder_checkpoint_min_voxels", None)
    if configured is not None:
        return int(configured)
    # Spatial-window occupancy affects retained activations as well as voxel
    # count: even a 19,650-voxel object reached 38.6 GB on a 32 GB Mac.
    # Checkpoint every Metal object by default; explicit thresholds still win.
    return 0 if torch.device(device).type == "mps" else 30_000


def _set_decoder_checkpoint(decoder: "torch.nn.Module | None", enabled: bool) -> None:
    """Set ``use_checkpoint`` on every submodule that exposes it.

    The transformer-torso blocks captured ``use_checkpoint`` at construction
    and read their own copy live in forward, so a top-level flag won't reach
    them — set it on every submodule.  Frees the torso's retained activation
    graph during the under-grad FINETUNE decode (the OOM term).  no_grad
    decodes (initial/best/export) are ~unaffected (checkpoint is a near-noop
    without grad)."""
    if decoder is None:
        return
    for _m in decoder.modules():
        if hasattr(_m, "use_checkpoint"):
            _m.use_checkpoint = enabled


def setup_shared_lora_decoder(
    base_decoder: torch.nn.Module,
    rank: int,
    alpha: float,
    init: str = "default",
    use_dora: bool = False,
    rs_scaling: bool = False,
    lora_targets: str = "all",
) -> torch.nn.Module:
    """Deep-copy + freeze + LoRA-wrap a decoder **once** (cache-aware).

    Returns the wrapped decoder, which hosts :class:`LoRALayer` wrappers
    around every matching ``nn.Linear``.  Every Parameter from the
    original layers is frozen.  Per-object :class:`LoRAAdapter` instances
    allocate their own ``(A, B)`` (and optional DoRA ``magnitude``)
    Parameters and bind them via :meth:`LoRAAdapter.activate` — so a
    multi-object run holds **one** decoder copy + N small adapters
    instead of N full decoder copies.

    ``use_dora=True`` enables the DoRA magnitude/direction decomposition
    on every wrapped layer (Liu et al. 2024, arXiv:2402.09353).
    ``rs_scaling=True`` switches per-layer scaling from ``alpha/rank``
    to rsLoRA's ``alpha/√rank`` (Kalajdzievski 2023).

    Within a single process, repeated calls with the same configuration
    reuse a cached wrapped decoder (see ``_get_or_build_wrapped_decoder``).
    Safe to share because ``LoRAAdapter.activate`` rebinds the layers'
    ``lora_A/B/magnitude`` attributes — the shared decoder's "default"
    Parameters from ``apply_lora_to_decoder`` are never read after that.
    """
    return _get_or_build_wrapped_decoder(
        base_decoder, rank, alpha, init, use_dora, rs_scaling, lora_targets,
    )
