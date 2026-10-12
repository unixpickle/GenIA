"""Explicit device selection shared by GenIA and its local model implementations."""
import os

import torch


def get_device(device=None):
    """Prefer CUDA, then Metal, then CPU; GENIA_DEVICE can override the default."""
    choice = device or os.environ.get("GENIA_DEVICE")
    if choice is None:
        choice = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    result = torch.device(choice)
    if result.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    if result.type == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("Metal was requested but is unavailable")
    return result


def empty_cache(device=None):
    device = get_device(device)
    if device.type == "cuda":
        torch.cuda.empty_cache()
    elif device.type == "mps":
        torch.mps.empty_cache()


def synchronize(device=None):
    device = get_device(device)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "mps":
        torch.mps.synchronize()
