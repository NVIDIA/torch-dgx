# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
import math

import numpy as np
import pytest
import torch
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel


def _philox_uniform_cpu(shape, seed, offset):
    """Philox4x32-10 draws, with offset measured in individual 32-bit words."""
    count = math.prod(shape)
    offset %= 1 << 64
    start = offset % 4
    blocks = np.arange((start + count + 3) // 4, dtype=np.uint64) + np.uint64(
        offset // 4
    )
    mask = 0xFFFFFFFF
    c0, c1 = blocks & mask, blocks >> 32
    c2, c3 = np.zeros_like(blocks), np.zeros_like(blocks)
    k0, k1 = seed & mask, (seed >> 32) & mask
    # Philox's fixed ten rounds, matching ATen/core/PhiloxRNGEngine.h.
    # Subsequence zero makes the stream independent of GPU/thread partitioning.
    for _ in range(10):
        p0, p1 = c0 * 0xD2511F53, c2 * 0xCD9E8D57
        c0, c1, c2, c3 = (
            (p1 >> 32) ^ c1 ^ k0,
            p1 & mask,
            (p0 >> 32) ^ c3 ^ k1,
            p0 & mask,
        )
        k0, k1 = (k0 + 0x9E3779B9) & mask, (k1 + 0xBB67AE85) & mask
    words = np.stack((c0, c1, c2, c3), axis=1).reshape(-1)[start : start + count]
    # Match curand_uniform4's uint32 -> float32 conversion and (0, 1] range.
    uniform = words.astype(np.float32) * np.float32(2.0**-32) + np.float32(2.0**-33)
    return torch.from_numpy(uniform.reshape(shape))


def _dropout_reference(
    inputs, seed, offset, probability, bias=None, causal=False, scale=None
):
    query, key, value = inputs
    scale = query.size(-1) ** -0.5 if scale is None else scale
    scores = (query @ key.transpose(-2, -1)) * scale
    if bias is not None:
        scores = scores + bias
    if causal:
        keep = torch.ones(scores.shape[-2:], dtype=torch.bool).tril()
        scores = scores.masked_fill(~keep, -torch.inf)
    weights = torch.ops.aten._safe_softmax(scores, -1)
    uniform = _philox_uniform_cpu(scores.shape, seed.item(), offset.item())
    keep = uniform > probability
    return (weights * keep / (1.0 - probability)) @ value


@pytest.mark.parametrize(
    "shape",
    [
        (1, 1, 1, 7, 4, 4),
        (3, 3, 17, 37, 32, 16),
        (1, 5, 65, 137, 96, 32),
        (3, 3, 129, 67, 132, 256),
        (4, 2, 33, 65, 32, 96),
        (1, 3, 257, 137, 32, 96),
        (3, 1, 257, 137, 96, 32),
        (1, 1, 321, 193, 132, 256),
        (1, 1, 3, 5, 65536, 4),
        (1, 1, 3, 5, 4, 65536),
    ],
)
@pytest.mark.parametrize("probability", [0.25, 0.75])
@pytest.mark.parametrize("mask_kind", ["none", "causal", "additive", "boolean"])
def test_sdpa_dropout_forward_backward(
    dgx_setup_teardown, shape, probability, mask_kind
):
    batch, heads, queries, keys, query_dim, value_dim = shape
    shapes = [
        (batch, heads, queries, query_dim),
        (batch, heads, keys, query_dim),
        (batch, heads, keys, value_dim),
    ]
    inputs_cpu = [torch.randn(shape).double().requires_grad_() for shape in shapes]
    # Stepped channel storage also exercises attention's local packing.
    inputs = []
    for reference in inputs_cpu:
        storage = torch.empty(*reference.shape[:-1], reference.size(-1) * 2)
        storage[..., ::2] = reference.float()
        inputs.append(storage.to("dgx")[..., ::2].detach().requires_grad_())
    bias_cpu = bias = None
    if mask_kind in ("additive", "boolean"):
        base = torch.randn(batch, heads, queries, keys)
        if mask_kind == "boolean":
            base = torch.zeros_like(base).masked_fill(base < 0, -torch.inf)
        if queries > 1:
            base[..., 0, :] = -torch.inf
        bias_cpu = base.double().requires_grad_(mask_kind == "additive")
        bias = base.to("dgx").requires_grad_(mask_kind == "additive")
    causal = mask_kind == "causal"
    scale = -0.25 if causal else None
    initial_state = torch.dgx.get_rng_state()
    actual, lse, seed, offset = torch.ops.aten._scaled_dot_product_efficient_attention(
        *inputs, bias, True, probability, causal, scale=scale
    )
    expected = _dropout_reference(
        inputs_cpu, seed, offset, probability, bias_cpu, causal, scale
    )
    torch.testing.assert_close(actual.cpu(), expected.float(), atol=2e-5, rtol=5e-5)

    # Exercise public dispatch too, including ATen's Boolean mask conversion.
    torch.dgx.set_rng_state(initial_state)
    public_mask = bias != -torch.inf if mask_kind == "boolean" else bias
    kwargs = {
        "dropout_p": probability,
        "is_causal": causal,
        "attn_mask": public_mask,
        "scale": scale,
    }
    with sdpa_kernel(SDPBackend.EFFICIENT_ATTENTION):
        assert (
            torch.ops.aten._fused_sdp_choice(*inputs, **kwargs)
            == SDPBackend.EFFICIENT_ATTENTION.value
        )
        public = F.scaled_dot_product_attention(*inputs, **kwargs)
    torch.testing.assert_close(public.cpu(), actual.cpu(), atol=0, rtol=0)
    assert (
        type(public.grad_fn).__name__ == "ScaledDotProductEfficientAttentionBackward0"
    )

    # Change RNG state before backward to ensure it uses forward's saved state.
    torch.rand(257, device="dgx")
    state_before_backward = torch.dgx.get_rng_state()
    # Keep the channel reduction's variance bounded even for 65536-wide V.
    grad = (torch.randn(batch, queries, heads, value_dim) / value_dim**0.5).transpose(
        1, 2
    )
    differentiable_cpu = inputs_cpu + ([bias_cpu] if mask_kind == "additive" else [])
    differentiable = inputs + ([bias] if mask_kind == "additive" else [])
    expected_grads = torch.autograd.grad(expected, differentiable_cpu, grad.double())
    actual_grads = torch.autograd.grad(public, differentiable, grad.to("dgx"))
    assert torch.equal(torch.dgx.get_rng_state(), state_before_backward)
    for gradient, reference in zip(actual_grads, expected_grads):
        torch.testing.assert_close(
            gradient.cpu(), reference.float(), atol=4e-5, rtol=1e-4
        )

    # Dropout must not change the normalization saved for backward.
    _, expected_lse, _, _ = torch.ops.aten._scaled_dot_product_efficient_attention(
        *inputs, bias, True, 0.0, causal, scale=scale
    )
    torch.testing.assert_close(lse.cpu(), expected_lse.cpu(), atol=0, rtol=0)


@pytest.mark.parametrize("queries", [1, 65, 257])
@pytest.mark.parametrize("dim", [4, 96, 132])
@pytest.mark.parametrize("heads", [3, 5])
@pytest.mark.parametrize("mask_kind", ["none", "causal", "additive"])
def test_sdpa_dropout_broadcast_inference(
    dgx_setup_teardown, queries, dim, heads, mask_kind
):
    query = torch.randn(1, heads, queries, dim)
    key, value = [torch.randn(1, 1, 137, dim) for _ in range(2)]
    inputs_cpu = [
        query.double(),
        *(x.double().expand(1, heads, 137, dim) for x in (key, value)),
    ]
    inputs = [
        query.to("dgx"),
        *(x.to("dgx").expand(1, heads, 137, dim) for x in (key, value)),
    ]
    bias = torch.randn(1, heads, queries, 137) if mask_kind == "additive" else None
    with torch.no_grad():
        actual, lse, seed, offset = (
            torch.ops.aten._scaled_dot_product_efficient_attention(
                *inputs,
                None if bias is None else bias.to("dgx"),
                False,
                0.25,
                mask_kind == "causal",
            )
        )
        expected = _dropout_reference(
            inputs_cpu,
            seed,
            offset,
            0.25,
            None if bias is None else bias.double(),
            mask_kind == "causal",
        )
    assert lse.numel() == 0
    torch.testing.assert_close(actual.cpu(), expected.float(), atol=2e-5, rtol=5e-5)


@pytest.mark.parametrize("seed_value", [1234, (1 << 63) + 17, (1 << 64) - 1])
@pytest.mark.parametrize("initial_offset", [0, (1 << 32) + 12, (1 << 63) + 4])
def test_sdpa_dropout_rng_replay(dgx_setup_teardown, seed_value, initial_offset):
    inputs = [torch.randn(1, 3, 257, 96).to("dgx").requires_grad_() for _ in range(3)]
    torch.dgx.manual_seed(seed_value)
    generator = torch.dgx.default_generators[0]
    generator.set_offset(initial_offset)
    initial_state = torch.dgx.get_rng_state()
    deterministic = torch.are_deterministic_algorithms_enabled()
    warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
    try:
        torch.use_deterministic_algorithms(True)
        first, _, seed, offset = torch.ops.aten._scaled_dot_product_efficient_attention(
            *inputs, None, True, 0.25
        )
        advanced_state = torch.dgx.get_rng_state()
        assert seed.device.type == offset.device.type == "cpu"
        assert seed.item() % (1 << 64) == seed_value
        assert offset.item() % (1 << 64) == initial_offset
        expected = _dropout_reference(
            [x.detach().cpu().double() for x in inputs], seed, offset, 0.25
        )
        torch.testing.assert_close(first.cpu(), expected.float(), atol=2e-5, rtol=5e-5)
        second, _, _, second_offset = (
            torch.ops.aten._scaled_dot_product_efficient_attention(
                *inputs, None, True, 0.25
            )
        )
        assert second_offset.item() % (1 << 64) >= initial_offset + 3 * 257 * 257
        assert not torch.equal(first.cpu(), second.cpu())
        torch.dgx.set_rng_state(initial_state)
        with sdpa_kernel(SDPBackend.EFFICIENT_ATTENTION):
            replay = F.scaled_dot_product_attention(*inputs, dropout_p=0.25)
        assert torch.equal(torch.dgx.get_rng_state(), advanced_state)
        torch.testing.assert_close(first.cpu(), replay.cpu(), atol=0, rtol=0)
        grad = torch.randn(1, 3, 257, 1).to("dgx").expand_as(first)
        first_grads = torch.autograd.grad(first, inputs, grad)
        torch.dgx.manual_seed(5678)
        replay_grads = torch.autograd.grad(replay, inputs, grad)
        for first_grad, replay_grad in zip(first_grads, replay_grads):
            torch.testing.assert_close(
                first_grad.cpu(), replay_grad.cpu(), atol=0, rtol=0
            )
    finally:
        torch.use_deterministic_algorithms(deterministic, warn_only=warn_only)


@pytest.mark.parametrize("probability", [0.0, 1.0, 1.0 - 1e-10])
@pytest.mark.parametrize("bias_only", [False, True])
def test_sdpa_dropout_endpoints(dgx_setup_teardown, probability, bias_only):
    inputs = [
        torch.randn(3, 3, 17, 32).to("dgx").requires_grad_(not bias_only)
        for _ in range(3)
    ]
    bias = torch.randn(3, 3, 17, 17, device="dgx", requires_grad=True)
    initial_state = torch.dgx.get_rng_state()
    with sdpa_kernel(SDPBackend.EFFICIENT_ATTENTION):
        output = F.scaled_dot_product_attention(
            *inputs, attn_mask=bias, dropout_p=probability
        )
    if probability:
        torch.testing.assert_close(
            output.cpu(), torch.zeros(output.shape), atol=0, rtol=0
        )
        output.sum().backward()
        for tensor in [bias] if bias_only else [*inputs, bias]:
            torch.testing.assert_close(
                tensor.grad.cpu(), torch.zeros(tensor.shape), atol=0, rtol=0
            )
    assert torch.equal(torch.dgx.get_rng_state(), initial_state)


@pytest.mark.parametrize("probability", [-0.25, 1.25, float("nan"), float("inf")])
def test_sdpa_invalid_dropout(dgx_setup_teardown, probability):
    inputs = [torch.randn(1, 1, 3, 4).to("dgx") for _ in range(3)]
    initial_state = torch.dgx.get_rng_state()
    with pytest.raises(
        RuntimeError, match="Unsupported DGX efficient attention configuration"
    ):
        torch.ops.aten._scaled_dot_product_efficient_attention(
            *inputs, None, False, probability
        )
    assert torch.equal(torch.dgx.get_rng_state(), initial_state)


@pytest.mark.parametrize(
    "shape,supported_32bit,supported_64bit",
    [
        ((1, 1, 2, (1 << 30) - 1), True, True),  # INT32_MAX - 1 elements
        ((1, 1, 2, 1 << 30), False, True),  # INT32_MAX + 1 elements
        ((1, 1, 65537, 65537), False, True),  # Forward query-row overflow.
        ((1, 2, 32768, 32768), False, True),  # Include the head extent.
        ((2, 1, 32768, 32768), False, True),  # Include batches across GPUs.
        ((2, 2, 16383, 32768), True, True),
        ((65535, 65535, 65535, 65535), False, False),  # Above INT64_MAX.
    ],
)
@pytest.mark.parametrize("probability", [0.0, 0.25, 1.0])
@pytest.mark.parametrize("causal", [False, True])
def test_sdpa_dropout_rng_index_limit(
    dgx_setup_teardown, shape, supported_32bit, supported_64bit, probability, causal
):
    batch, heads, queries, keys = shape
    # Broadcast tiny allocations to test dispatch at the indexing boundary
    # without allocating outputs or a quadratic attention matrix.
    inputs = [
        torch.empty(1, 1, 1, 4, device="dgx").expand(batch, heads, length, 4)
        for length in (queries, keys, keys)
    ]
    initial_state = torch.dgx.get_rng_state()
    torch_version = tuple(int(part) for part in torch.__version__.split(".")[:2])
    within_limit = supported_64bit if torch_version >= (2, 14) else supported_32bit
    supported = within_limit or probability in (0.0, 1.0)
    expected_backend = SDPBackend.EFFICIENT_ATTENTION if supported else SDPBackend.MATH
    kwargs = {"dropout_p": probability, "is_causal": causal}
    with sdpa_kernel([SDPBackend.EFFICIENT_ATTENTION, SDPBackend.MATH]):
        assert (
            torch.ops.aten._fused_sdp_choice(*inputs, **kwargs)
            == expected_backend.value
        )
    if not supported:
        with (
            sdpa_kernel(SDPBackend.EFFICIENT_ATTENTION),
            pytest.raises(RuntimeError, match="No available DGX SDPA kernel"),
        ):
            F.scaled_dot_product_attention(*inputs, **kwargs)
        with pytest.raises(
            RuntimeError, match="Unsupported DGX efficient attention configuration"
        ):
            torch.ops.aten._scaled_dot_product_efficient_attention(
                *inputs, None, False, probability, causal
            )
        # The direct backward entry point must reject before reading saved
        # output/LSE or launching a kernel, so tiny placeholders suffice here.
        placeholder = torch.empty(0, device="dgx")
        rng_placeholder = torch.zeros((), dtype=torch.int64)
        with pytest.raises(
            RuntimeError,
            match="Unsupported DGX efficient attention backward configuration",
        ):
            torch.ops.aten._scaled_dot_product_efficient_attention_backward(
                placeholder,
                *inputs,
                None,
                placeholder,
                placeholder,
                rng_placeholder,
                rng_placeholder,
                probability,
                [True, True, True, False],
                causal,
            )
    assert torch.equal(torch.dgx.get_rng_state(), initial_state)
