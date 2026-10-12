# Copyright (c) Meta Platforms, Inc. and affiliates.
import torch
import torch.nn.functional as F


def block_diag_attn_mask(q_seqlens, kv_seqlens, device=None, dtype=torch.float32):
    """
    Create an additive attention mask for block-diagonal attention.
    The result is shape [sum_q, sum_kv], with 0.0 in the valid
    region(s) and -inf elsewhere.
    """
    total_q = sum(q_seqlens)
    total_kv = sum(kv_seqlens)

    # Start with everything "masked out"
    attn_mask = torch.full(
        (total_q, total_kv), float("-inf"), device=device, dtype=dtype
    )

    q_start = 0
    kv_start = 0
    for q_len, kv_len in zip(q_seqlens, kv_seqlens):
        attn_mask[q_start : q_start + q_len, kv_start : kv_start + kv_len] = 0
        q_start += q_len
        kv_start += kv_len

    return attn_mask


def masked_sdpa(q, k, v, q_seqlen, kv_seqlen, attn_bias=None, *, query_chunk_size=None):
    """Packed block attention with optional additive visibility bias.

    Bias storage follows the original xformers path below. Query chunks bound
    Metal's temporary score allocation without changing attention's key axis.
    """
    if len(q_seqlen) != len(kv_seqlen) or sum(q_seqlen) != q.shape[1] or sum(kv_seqlen) != k.shape[1]:
        raise ValueError("packed attention lengths do not match input shapes")
    if query_chunk_size is not None and query_chunk_size < 1:
        raise ValueError('query_chunk_size must be positive')
    stacked = False
    if attn_bias is not None:
        if attn_bias.ndim != 4 or attn_bias.shape[0] not in (1, len(q_seqlen)):
            raise ValueError('attention bias must be packed or stacked per view')
        if attn_bias.shape[1] not in (1, q.shape[-2]):
            raise ValueError('attention bias head count must be 1 or match queries')
        stacked = _is_stacked_bias(attn_bias, len(q_seqlen))
    outputs = []
    qs = ks = 0
    for i, (nq, nk) in enumerate(zip(q_seqlen, kv_seqlen)):
        query = q[:, qs:qs + nq].transpose(1, 2)
        key = k[:, ks:ks + nk].transpose(1, 2)
        value = v[:, ks:ks + nk].transpose(1, 2)
        bias = None
        if attn_bias is not None:
            bias = _slice_view_bias(attn_bias, i, qs, ks, nq, nk, stacked=stacked)
            if bias.shape[-2:] != (nq, nk):
                raise ValueError('attention bias does not cover this view')
        chunk = query_chunk_size
        if chunk is None:
            chunk = max(1, (16 * 1024 * 1024) // max(1, q.shape[-2] * nk)) if q.device.type == 'mps' and bias is not None else max(nq, 1)
        chunks = []
        for start in range(0, nq, chunk):
            mask = bias[..., start:start+chunk, :].to(q.dtype) if bias is not None else None
            chunks.append(F.scaled_dot_product_attention(query[..., start:start+chunk, :], key, value,
                                                         attn_mask=mask).transpose(1, 2))
        outputs.append(torch.cat(chunks, dim=1) if chunks else value[:, :, :0].transpose(1, 2))
        qs += nq
        ks += nk
    if not outputs:
        return q[0] * 0
    return torch.cat(outputs, dim=1)[0]


def _is_stacked_bias(attn_bias: torch.Tensor, N: int) -> bool:
    """Detect ``(N, H, L_per_view, P_per_view)`` stacked layout vs the
    legacy ``(1, H, L_or_sum_q, P_or_sum_kv)`` packed/template layout.

    The two shapes are indistinguishable when ``N == 1`` (both have
    ``shape[0] == 1``); we conservatively treat that case as packed,
    matching the original behaviour. Callers that build single-view
    stacked biases get the same result either way (one block).
    """
    return attn_bias.shape[0] == N and attn_bias.shape[0] != 1


def _slice_view_bias(
    attn_bias: torch.Tensor,
    i: int,
    q_ofs: int,
    kv_ofs: int,
    q_len: int,
    kv_len: int,
    *,
    stacked: bool,
) -> torch.Tensor:
    """Return the bias slice corresponding to view ``i`` in the per-view
    loop. ``stacked`` selects between the two storage layouts; see
    :func:`_is_stacked_bias`."""
    if stacked:
        return attn_bias[i:i + 1, :, :q_len, :kv_len]
    return attn_bias[:, :, q_ofs:q_ofs + q_len, kv_ofs:kv_ofs + kv_len]


def xops_blockdiag_with_bias(q, k, v, q_seqlen, kv_seqlen, attn_bias):
    """xformers MEA per view-block with additive bias.

    Streams biased block-diagonal attention through the fused cutlass kernel — never materializes
    the dense (sum_q, sum_kv) score / mask / bias tensors that drove the
    APPEARANCE_INIT_2 OOM. Each view's bias slice is at most
    ``L_per_view * P_per_view * dtype_bytes`` and only one is live at a time.

    Args:
        q, k, v: ``(1, sum_L, H, D)`` packed tensors. q is along queries,
                 k/v along keys; same layout the existing xformers
                 ``BlockDiagonalMask`` path expects.
        q_seqlen / kv_seqlen: per-view query / key lengths (must align with
                 the bias's block-diagonal layout — see invariant note).
        attn_bias: additive bias in either of two forms:
            - ``(N, 1 or H, L_per_view, P_per_view)``: stacked per-view
              storage (preferred — N× smaller than packed dense form).
            - ``(1, 1 or H, sum_q, sum_kv)``: legacy packed dense form.
            Detected by ``attn_bias.shape[0] == len(q_seqlen)`` for the
            stacked case.

    Returns:
        ``(sum_q, H, D)`` — same layout as ``masked_sdpa``.

    Invariant: ``q_seqlen`` / ``kv_seqlen`` partition the packed tensors in
    the same view order ``SparseVisibilityBiasHook._build_bias`` uses.

    Errors from the underlying ``xops.memory_efficient_attention`` propagate
    unmodified.
    """

    import xformers.ops as xops

    target_dtype = q.dtype
    H = q.shape[-2]
    N = len(q_seqlen)
    stacked = _is_stacked_bias(attn_bias, N)
    outs = []
    q_ofs = 0
    kv_ofs = 0
    for i, (q_len, kv_len) in enumerate(zip(q_seqlen, kv_seqlen)):
        qi = q[:, q_ofs:q_ofs + q_len].contiguous()
        ki = k[:, kv_ofs:kv_ofs + kv_len].contiguous()
        vi = v[:, kv_ofs:kv_ofs + kv_len].contiguous()
        bi = _slice_view_bias(
            attn_bias, i, q_ofs, kv_ofs, q_len, kv_len, stacked=stacked,
        ).to(target_dtype)
        # cutlass MEA requires the kv-dim to be a multiple of 8 in fp16/bf16
        pad = (-kv_len) % 8
        if pad:
            ki = F.pad(ki, (0, 0, 0, 0, 0, pad))
            vi = F.pad(vi, (0, 0, 0, 0, 0, pad))
            bi = F.pad(bi, (0, pad), value=float("-inf"))
        # cutlass MEA requires (1, H, Q, K) — no broadcasting over the head
        # dim. Expand and materialize (stride-0 expand passes shape check but
        # not all kernel paths honor it).
        if bi.shape[1] == 1 and H > 1:
            bi = bi.expand(-1, H, -1, -1)
        bi = bi.contiguous()
        out_i = xops.memory_efficient_attention(qi, ki, vi, attn_bias=bi)[0]
        outs.append(out_i)
        q_ofs += q_len
        kv_ofs += kv_len
    return torch.cat(outs, dim=0)
