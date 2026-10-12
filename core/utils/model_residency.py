"""Release Metal allocations for sampling networks while fitting decoded assets."""
from contextlib import contextmanager

import torch

from genia.core.device import empty_cache, synchronize


@contextmanager
def offload_sampling_models(pipeline, device):
    """Keep decoders on Metal, moving unused generators/conditioners to CPU.

    Fine-tuning only needs the decoders. Its sampling networks otherwise pin
    several GB of unified memory throughout every optimization step. Restore
    their original devices on exit so cached backbones remain reusable.
    CPU/CUDA callers retain their existing residency behavior.
    """
    device = torch.device(device)
    if device.type != "mps":
        yield
        return
    candidates = [pipeline.models[name] for name in ("ss_generator", "slat_generator")
                  if name in pipeline.models]
    candidates.extend(getattr(pipeline, "condition_embedders", {}).values())
    moved, seen = [], set()
    try:
        synchronize(device)
        for module in candidates:
            if not isinstance(module, torch.nn.Module) or id(module) in seen:
                continue
            seen.add(id(module))
            tensors = list(module.parameters()) + list(module.buffers())
            devices = {tensor.device for tensor in tensors}
            del tensors
            if len(devices) != 1 or next(iter(devices)).type != "mps":
                continue
            original_device = next(iter(devices))
            moved.append((module, original_device))
            module.to("cpu")
            empty_cache(device)
        yield
    finally:
        for module, original_device in moved:
            module.to(original_device)
        if moved:
            empty_cache(device)
