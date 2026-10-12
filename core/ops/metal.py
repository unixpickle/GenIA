"""Lazy Metal compilation; importing the package also works without MPS."""

from functools import lru_cache
from pathlib import Path

import torch


@lru_cache(None)
def library(name: str):
    if not torch.backends.mps.is_available():
        raise RuntimeError("Metal operators require an Apple GPU and an MPS-enabled PyTorch")
    if not hasattr(torch.mps, "compile_shader"):
        raise RuntimeError("Metal operators require PyTorch >= 2.7")
    return torch.mps.compile_shader(Path(__file__).with_name(name + ".metal").read_text())
