# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Graph-safe Keye RoPE and BF16 paged indexer; reuse SGLang's GQA decode."""

from functools import lru_cache

import torch
import triton
import triton.language as tl
from triton.language.extra.cuda import libdevice


@lru_cache(maxsize=16)
def _frequencies(dim: int, theta: float, device: torch.device) -> torch.Tensor:
    return 1.0 / (theta ** (torch.arange(0, dim, 2, device=device, dtype=torch.float32) / dim))


@triton.jit
def _rope(X, P, F, Y, H: tl.constexpr, D: tl.constexpr, PS: tl.constexpr, AXES: tl.constexpr):
    row = tl.program_id(0)
    d = tl.arange(0, D // 2)
    token = row // H
    axis = tl.where(d < D // 8, 0, tl.where(d < D // 8 + D * 3 // 16, 1, 2))
    position = tl.load(P + token * PS + (axis if AXES else 0)).to(tl.float32)
    angle = position * tl.load(F + d)
    c = libdevice.cos(angle).to(X.dtype.element_ty)
    s = libdevice.sin(angle).to(X.dtype.element_ty)
    a = tl.load(X + row * D + d)
    b = tl.load(X + row * D + d + D // 2)
    # Match the BF16 reference's separately rounded products (no fused FMA).
    ac = (a.to(tl.float32) * c.to(tl.float32)).to(a.dtype).to(tl.float32)
    bs = ((-b).to(tl.float32) * s.to(tl.float32)).to(a.dtype).to(tl.float32)
    bc = (b.to(tl.float32) * c.to(tl.float32)).to(a.dtype).to(tl.float32)
    ass = (a.to(tl.float32) * s.to(tl.float32)).to(a.dtype).to(tl.float32)
    tl.store(Y + row * D + d, ac + bs)
    tl.store(Y + row * D + d + D // 2, bc + ass)


def rope(x: torch.Tensor, positions: torch.Tensor, theta: float) -> torch.Tensor:
    x = x.contiguous()
    positions = positions.contiguous()
    output = torch.empty_like(x)
    _rope[(x.shape[0] * x.shape[1],)](
        x,
        positions,
        _frequencies(x.shape[-1], theta, x.device),
        output,
        x.shape[1],
        x.shape[2],
        positions.stride(0),
        positions.ndim == 2,
        enable_fp_fusion=False,
    )
    return output


@triton.jit
def _index_scores(
    Q,
    K,
    W,
    TABLE,
    REQUESTS,
    LENGTHS,
    OUT,
    QS: tl.constexpr,
    WS: tl.constexpr,
    TS: tl.constexpr,
    WIDTH: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row, tile = tl.program_id(0), tl.program_id(1)
    n = tile * BLOCK + tl.arange(0, BLOCK)
    length = tl.load(LENGTHS + row)
    result = tl.full((BLOCK,), float("-inf"), tl.float32)
    if tile * BLOCK < length:
        request = tl.load(REQUESTS + row)
        slots = tl.load(TABLE + request * TS + n, mask=(n < length) & (n < WIDTH), other=0)
        h = tl.arange(0, 16)
        d = tl.arange(0, 64)
        q = tl.load(Q + row * QS + h[:, None] * 64 + d[None, :])
        k = tl.load(K + slots[None, :] * 64 + d[:, None], mask=n[None, :] < length, other=0)
        # Keep every BF16 rounding boundary of indexer_scores, including
        # matmul -> scale -> head weight -> reduction, to preserve top-k.
        dots = tl.dot(q, k).to(q.dtype).to(tl.float32)
        scores = (dots * 0.125).to(q.dtype).to(tl.float32)
        weights = tl.load(W + row * WS + h)
        weighted = (tl.maximum(scores, 0) * weights[:, None].to(tl.float32)).to(q.dtype)
        result = tl.sum(weighted.to(tl.float32), axis=0)
        result = tl.where(n < length, result, float("-inf"))
    tl.store(OUT + row * WIDTH + n, result, mask=n < WIDTH)


def paged_scores(
    query: torch.Tensor,
    cache: torch.Tensor,
    weights: torch.Tensor,
    table: torch.Tensor,
    requests: torch.Tensor,
    lengths: torch.Tensor,
    width: int,
) -> torch.Tensor:
    scores = torch.empty((query.shape[0], width), device=query.device, dtype=query.dtype)
    _index_scores[(query.shape[0], triton.cdiv(width, 128))](
        query,
        cache,
        weights,
        table,
        requests,
        lengths,
        scores,
        query.stride(0),
        weights.stride(0),
        table.stride(0),
        width,
        128,
        enable_fp_fusion=False,
    )
    return scores


@triton.jit
def _selected_slots(
    SELECTED, TABLE, REQUESTS, LENGTHS, INDPTR, INDICES, TOPK: tl.constexpr, TS: tl.constexpr, BLOCK: tl.constexpr
):
    row = tl.program_id(0)
    n = tl.arange(0, BLOCK)
    count = tl.minimum(tl.load(LENGTHS + row), TOPK)
    request = tl.load(REQUESTS + row)
    index = tl.load(SELECTED + row * TOPK + n, mask=n < count, other=0)
    slot = tl.load(TABLE + request * TS + index, mask=n < count, other=0)
    offset = tl.load(INDPTR + row)
    tl.store(INDICES + offset + n, slot, mask=n < count)


def paged_decode(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    index_query: torch.Tensor,
    index_cache: torch.Tensor,
    weights: torch.Tensor,
    table: torch.Tensor,
    requests: torch.Tensor,
    lengths: torch.Tensor,
    width: int,
) -> torch.Tensor:
    from sglang.kernels.ops.attention.decode_attention import decode_attention_fwd

    scores = paged_scores(index_query, index_cache, weights, table, requests, lengths, width)
    topk = min(2048, width)
    # Sorted scores keep valid selections before the -inf padding for short
    # requests. CSR stores only those valid slots; no padded KV is ever read.
    selected = scores.topk(topk, dim=-1).indices
    counts = lengths.clamp(max=topk)
    indptr = torch.cat((torch.zeros(1, device=query.device, dtype=torch.int32), counts.cumsum(0).int()))
    indices = torch.empty(query.shape[0] * topk, device=query.device, dtype=torch.int32)
    _selected_slots[(query.shape[0],)](
        selected, table, requests, lengths, indptr, indices, topk, table.stride(0), triton.next_power_of_2(topk)
    )
    output = torch.empty_like(query)
    splits = 4
    partial = torch.empty((*query.shape[:2], splits, value.shape[-1]), device=query.device, dtype=torch.float32)
    lse = torch.empty((*query.shape[:2], splits), device=query.device, dtype=torch.float32)
    num_splits = torch.full((query.shape[0],), splits, device=query.device, dtype=torch.int32)
    decode_attention_fwd(
        query, key, value, output, indptr, indices, partial, lse, num_splits, splits, query.shape[-1] ** -0.5, 1.0, 1.0
    )
    return output.flatten(1)
