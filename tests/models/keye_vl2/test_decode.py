# Copyright (c) 2026 Relax Authors. All Rights Reserved.

import pytest
import torch
import torch.nn.functional as F


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Keye BF16 decode requires CUDA")
@pytest.mark.parametrize("width", [73, 2113])
def test_batched_decode_matches_separate_requests_with_poisoned_padding(width):
    from relax.models.keye_vl2.indexer import indexer_scores
    from relax.models.keye_vl2.sglang.model import decode_attention

    torch.manual_seed(27)
    lengths = [width - 15, width - 7, width]
    q = torch.randn(3, 4, 128, device="cuda", dtype=torch.bfloat16)
    k = torch.randn(3, width, 1, 128, device="cuda", dtype=torch.bfloat16)
    v = torch.randn_like(k)
    iq = torch.randn(3, 16, 64, device="cuda", dtype=torch.bfloat16)
    ik = torch.randn(3, width, 64, device="cuda", dtype=torch.bfloat16)
    weights = torch.randn(3, 16, device="cuda", dtype=torch.bfloat16)
    expected = []
    for row, length in enumerate(lengths):
        scores = indexer_scores(iq[row : row + 1], ik[row, :length, None], weights[row : row + 1])
        support = torch.zeros_like(scores, dtype=torch.bool).scatter_(
            1, scores.topk(min(2048, length), -1).indices, True
        )
        result = F.scaled_dot_product_attention(
            q[row : row + 1, :, None],
            k[row : row + 1, :length].repeat_interleave(4, 2).transpose(1, 2),
            v[row : row + 1, :length].repeat_interleave(4, 2).transpose(1, 2),
            attn_mask=support[:, None],
        )
        expected.append(result.flatten(1))
        k[row, length:] = float("nan")
        v[row, length:] = float("nan")
        ik[row, length:] = float("nan")
    actual = decode_attention(q, k, v, iq, ik, weights, torch.tensor(lengths, device="cuda"))
    torch.testing.assert_close(actual, torch.cat(expected), atol=0.02, rtol=0.02)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Keye paged decode requires CUDA")
@pytest.mark.parametrize("dim", [64, 128])
def test_fused_rope_matches_text_and_image_axes(dim):
    from relax.models.keye_vl2.indexer import text_rope
    from relax.models.keye_vl2.sglang.ops import rope

    x = torch.randn(4, 16, dim, device="cuda", dtype=torch.bfloat16)
    for positions in [
        torch.tensor([0, 73, 8001, 32767], device="cuda"),
        torch.tensor([[0, 0, 0], [3, 7, 11], [71, 9, 0], [8001, 8002, 8003]], device="cuda"),
    ]:
        torch.testing.assert_close(rope(x, positions, 1e7), text_rope(x, positions, 1e7), atol=0, rtol=0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Keye paged decode requires CUDA")
def test_paged_decode_preserves_bf16_topk_and_replays_changed_lengths():
    from relax.models.keye_vl2.indexer import indexer_scores
    from relax.models.keye_vl2.sglang.model import decode_attention
    from relax.models.keye_vl2.sglang.ops import paged_decode, paged_scores

    torch.manual_seed(91)
    batch, width = 3, 2176
    q = torch.randn(batch, 8, 128, device="cuda", dtype=torch.bfloat16)
    k = torch.randn(batch * width, 2, 128, device="cuda", dtype=torch.bfloat16)
    v = torch.randn_like(k)
    iq = torch.randn(batch, 16, 64, device="cuda", dtype=torch.bfloat16)
    ik = torch.randn(batch * width, 64, device="cuda", dtype=torch.bfloat16)
    weights = torch.randn(batch, 16, device="cuda", dtype=torch.bfloat16)
    table = torch.randperm(batch * width, device="cuda").int().reshape(batch, width)
    requests = torch.tensor([2, 0, 1], device="cuda", dtype=torch.int32)
    lengths = torch.tensor([73, 2100, width], device="cuda", dtype=torch.int32)

    def run():
        return paged_decode(q, k, v, iq, ik, weights, table, requests, lengths, width)

    for _ in range(3):
        run()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        actual = run()
    for values in [[73, 2100, width], [width, 125, 2051]]:
        lengths.copy_(torch.tensor(values, device="cuda", dtype=torch.int32))
        requests.copy_(requests.roll(1))
        graph.replay()
        slots = table[requests.long()].long()
        scores = paged_scores(iq, ik, weights, table, requests, lengths, width)
        for row, n in enumerate(values):
            expected_scores = indexer_scores(iq[row : row + 1], ik[slots[row, :n], None], weights[row : row + 1])[0]
            torch.testing.assert_close(scores[row, :n], expected_scores, atol=0, rtol=0)
        expected = decode_attention(q, k[slots], v[slots], iq, ik[slots], weights, lengths)
        torch.testing.assert_close(actual, expected, atol=0.02, rtol=0.02)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="SGLang model imports require the CUDA runtime")
@pytest.mark.parametrize("multimodal", [False, True])
def test_language_decode_ignores_unused_graph_mrope_buffer(monkeypatch, multimodal):
    from types import SimpleNamespace

    from relax.models.keye_vl2.sglang import model

    attention = model.KeyeAttention.__new__(model.KeyeAttention)
    torch.nn.Module.__init__(attention)
    attention.qkv_proj = lambda hidden: (hidden, None)
    attention.q_size, attention.kv_size = 256, 128
    attention.num_heads, attention.num_kv_heads, attention.head_dim = 2, 1, 128
    attention.q_norm, attention.k_norm = torch.nn.Identity(), torch.nn.Identity()
    attention.rope_theta = 1e7
    attention.keye_multimodal_enabled = multimodal
    positions = torch.tensor([83, 129, 261])
    batch = SimpleNamespace(mrope_positions=torch.zeros(3, 3, dtype=torch.int64))

    class CapturedPosition(Exception):
        pass

    def capture(x, actual_positions, theta):
        expected = batch.mrope_positions.transpose(0, 1) if multimodal else positions
        torch.testing.assert_close(actual_positions, expected)
        raise CapturedPosition

    monkeypatch.setattr(model, "rope", capture)
    with pytest.raises(CapturedPosition):
        attention(positions, torch.randn(3, 512), batch)
