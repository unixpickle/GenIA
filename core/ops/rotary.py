"""Real-valued rotary position embeddings, including on MPS without complex ops."""
import torch


def rotary_embedding(x, angles):
    """Rotate adjacent feature pairs by broadcastable angles in radians."""
    if x.shape[-1] % 2:
        raise ValueError("rotary embeddings need an even feature dimension")
    pairs = x.float().reshape(*x.shape[:-1], -1, 2)
    real, imag = pairs.unbind(-1)
    c, s = angles.cos(), angles.sin()
    return torch.stack((real * c - imag * s, real * s + imag * c), -1).flatten(-2).to(x.dtype)
