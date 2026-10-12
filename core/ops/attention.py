"""Exact query-chunked attention for large Metal model sequences.

PyTorch 2.7's MPS math path materializes attention scores. ActionMesh joins
16 frames of 2,048 tokens, whose full score tensor exceeds a 32 GB Mac's RAM.
Splitting queries preserves the complete key-axis softmax. Checkpointing each
chunk also bounds saved score activations when differentiating the operation.
"""
import math
from collections import defaultdict

import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint


def _mps_autocast_inputs(*tensors):
    if not torch.is_autocast_enabled("mps"):
        return tensors
    dtype = torch.get_autocast_dtype("mps")
    return tuple(
        tensor.to(dtype) if tensor is not None
        and tensor.device.type == "mps" and tensor.is_floating_point()
        and tensor.dtype != torch.float64 else tensor
        for tensor in tensors
    )


def _attention(query, key, value, attn_mask=None, dropout_p=0.0, is_causal=False, *, scale=None):
    if (query.device.type == "mps" and torch.is_grad_enabled()
            and attn_mask is not None and attn_mask.requires_grad
            and not any(t.requires_grad for t in (query, key, value))):
        # PyTorch 2.7's MPS SDPA dispatch overlooks mask-only gradients and
        # selects an operation without backward support. Its generic math
        # implementation supports the floating additive mask's derivative.
        query, key, value, attn_mask = _mps_autocast_inputs(query, key, value, attn_mask)
        # Match SDPA's autocast boundary: its internal softmax/matmuls retain
        # the math implementation's FP32 accumulation, not nested autocast.
        with torch.autocast("mps", enabled=False):
            result = torch.ops.aten._scaled_dot_product_attention_math(
                query, key, value, attn_mask, dropout_p, is_causal, scale=scale,
            )[0]
    else:
        result = F.scaled_dot_product_attention(
            query, key, value, attn_mask=attn_mask, dropout_p=dropout_p,
            is_causal=is_causal, scale=scale,
        )
    if query.device.type == "mps" and attn_mask is not None:
        # The MPS inference path returns NaNs for wholly masked rows; CPU and
        # the differentiable math path return zeros for the same inputs.
        valid = (attn_mask if attn_mask.dtype == torch.bool else attn_mask != -torch.inf)
        result = torch.where(valid.any(dim=-1, keepdim=True), result, 0)
    return result


def scaled_dot_product_attention(
    query, key, value, attn_mask=None, dropout_p=0.0, is_causal=False,
    *, scale=None, max_score_elements=None,
):
    """SDPA with bounded query chunks on MPS; CPU/CUDA use native dispatch.

    An explicit ``max_score_elements`` also enables chunking on CPU for tests.
    The budget covers batch/head/query/key score elements, not model inputs or
    total process memory. Inference grows the default budget with available MPS
    memory; training retains 16 million elements. Dropout uses native dispatch
    to retain RNG behavior.
    """
    kwargs = dict(attn_mask=attn_mask, dropout_p=dropout_p,
                  is_causal=is_causal, scale=scale)
    if max_score_elements is None and query.device.type != "mps":
        return F.scaled_dot_product_attention(query, key, value, **kwargs)
    needs_grad = torch.is_grad_enabled() and any(
        tensor is not None and tensor.requires_grad
        for tensor in (query, key, value, attn_mask)
    )
    budget = max_score_elements
    if budget is None:
        budget = 16 * 1024 * 1024
        if not needs_grad:
            # Tiny query tiles underutilize the MPS matmul kernels. Allow up to
            # 1 GiB of FP32 scores, capped at 1/16 of currently available working
            # memory; leave the smaller recomputation budget for training.
            available = max(0, torch.mps.recommended_max_memory()
                            - torch.mps.current_allocated_memory())
            budget = max(budget, min(256 * 1024 * 1024, available // 64))
    if budget < 1:
        raise ValueError("max_score_elements must be positive")
    if dropout_p or (is_causal and attn_mask is not None):
        return F.scaled_dot_product_attention(query, key, value, **kwargs)
    batch_heads = math.prod(torch.broadcast_shapes(query.shape[:-2], key.shape[:-2], value.shape[:-2]))
    length = query.shape[-2]
    chunk_size = max(1, budget // max(1, batch_heads * key.shape[-2]))
    if length <= chunk_size:
        return _attention(query, key, value, **kwargs)

    def attend(q, k, v, mask):
        return _attention(q, k, v, attn_mask=mask, scale=scale)

    outputs = []
    if not needs_grad and query.device.type == "mps" and torch.is_autocast_enabled("mps"):
        # Native SDPA autocasts all floating inputs, including the mask. Casting
        # long FP32 keys inside every query chunk can dominate inference time.
        # Hoist those conversions only without gradients: in training, casting
        # keys separately preserves FP32 accumulation of their chunk gradients.
        query, key, value, attn_mask = _mps_autocast_inputs(query, key, value, attn_mask)
    for start in range(0, length, chunk_size):
        stop = min(start + chunk_size, length)
        mask = attn_mask
        if is_causal:
            mask = (torch.arange(start, stop, device=query.device)[:, None]
                    >= torch.arange(key.shape[-2], device=query.device)[None, :])
        elif mask is not None and mask.ndim >= 2 and mask.shape[-2] != 1:
            mask = mask[..., start:stop, :]
        q = query[..., start:stop, :]
        if needs_grad:
            outputs.append(checkpoint(attend, q, key, value, mask,
                                      use_reentrant=False, preserve_rng_state=False))
        else:
            outputs.append(attend(q, key, value, mask))
    return torch.cat(outputs, dim=-2)


def packed_self_attention(qkv, lengths, *, max_score_elements=None):
    """Batch ragged self-attention windows using masked power-of-two buckets.

    ``qkv`` is (sum(lengths), 3, heads, channels). Padding stays within each
    window and is excluded from the key softmax; output rows retain input order.
    Buckets avoid padding every small spatial window to the largest window size.
    """
    if qkv.ndim != 4 or qkv.shape[1] != 3:
        raise ValueError("qkv must have shape (tokens, 3, heads, channels)")
    if any(length < 0 for length in lengths) or sum(lengths) != qkv.shape[0]:
        raise ValueError("window lengths do not match qkv")
    if not qkv.shape[0]:
        return qkv[:, 0]
    groups = defaultdict(list)
    offset = 0
    for length in lengths:
        if length:
            groups[1 << (length - 1).bit_length()].append((offset, length))
        offset += length

    outputs, packed_order = [], []
    for width, windows in groups.items():
        # Construct maps on CPU; their values depend only on cached spatial
        # partition lengths, never on model activations or trainable tensors.
        padded_indices, valid_indices, order = [], [], []
        for row, (start, length) in enumerate(windows):
            order.extend(range(start, start + length))
            padded_indices.extend(range(start, start + length))
            padded_indices.extend([start] * (width - length))
            valid_indices.extend(range(row * width, row * width + length))
        index = torch.tensor(padded_indices, device=qkv.device, dtype=torch.long)
        padded = qkv.index_select(0, index).reshape(len(windows), width, *qkv.shape[1:])
        q, k, v = [x.transpose(1, 2) for x in padded.unbind(2)]
        sizes = torch.tensor([length for _, length in windows], device=qkv.device)
        mask = torch.arange(width, device=qkv.device)[None, :] < sizes[:, None]
        output = scaled_dot_product_attention(q, k, v, attn_mask=mask[:, None, None, :],
                                               max_score_elements=max_score_elements)
        output = output.transpose(1, 2).reshape(-1, *qkv.shape[2:])
        valid = torch.tensor(valid_indices, device=qkv.device, dtype=torch.long)
        outputs.append(output.index_select(0, valid))
        packed_order.extend(order)
    inverse = [0] * qkv.shape[0]
    for row, original in enumerate(packed_order):
        inverse[original] = row
    inverse = torch.tensor(inverse, device=qkv.device, dtype=torch.long)
    return torch.cat(outputs, dim=0).index_select(0, inverse)
