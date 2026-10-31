# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
from contextlib import ExitStack

import pytest
import torch
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel


@pytest.mark.parametrize(
    "shape,grad_layout,scale",
    [
        (shape, grad_layout, scale)
        for shape in [
            (1, 2, 1, 4),
            (3, 2, 17, 8),
            (3, 3, 65, 32),
            (4, 2, 129, 64),
            (1, 2, 1, 68),
            (1, 8, 512, 96),
            (1, 5, 129, 96),
            (2, 1, 65, 96),
            (3, 2, 129, 128),
            (3, 2, 65, 132),
        ]
        for grad_layout in ["contiguous", "sequence-major", "zero-stride", "unaligned"]
        for scale in [None, -0.25, 0.0]
        # Cross layouts and scales once per head-dimension specialization;
        # retain every boundary and sharding shape with default settings.
        if shape in [(3, 3, 65, 32), (1, 5, 129, 96), (3, 2, 65, 132)]
        or (grad_layout == "contiguous" and scale is None)
    ],
)
def test_sdpa_packed_forward_backward(dgx_setup_teardown, shape, grad_layout, scale):
    batch, heads, sequence, dim = shape
    base_cpu = torch.randn(
        batch + 2, sequence + 2, 3 * heads * dim + 4, requires_grad=True
    )
    base = base_cpu.detach().to("dgx").requires_grad_()

    def split(tensor):
        packed = tensor[1:-1, 1:-1, : 3 * heads * dim]
        return [
            x.reshape(batch, sequence, heads, dim).transpose(1, 2)
            for x in packed.chunk(3, -1)
        ]

    with sdpa_kernel(SDPBackend.MATH):
        expected = F.scaled_dot_product_attention(*split(base_cpu), scale=scale)
    assert (
        torch.ops.aten._fused_sdp_choice(*split(base), scale=scale)
        == SDPBackend.EFFICIENT_ATTENTION.value
    )
    with sdpa_kernel(SDPBackend.EFFICIENT_ATTENTION):
        actual = F.scaled_dot_product_attention(*split(base), scale=scale)
    assert (
        type(actual.grad_fn).__name__ == "ScaledDotProductEfficientAttentionBackward0"
    )
    torch.testing.assert_close(actual.cpu(), expected, atol=1e-5, rtol=3e-5)

    if grad_layout == "zero-stride":
        grad_base = torch.randn(batch, heads, sequence, 1)
        grad_cpu = grad_base.expand_as(expected)
        grad = grad_base.to("dgx").expand_as(actual)
    elif grad_layout == "sequence-major":
        grad_base = torch.randn(batch, sequence, heads, dim)
        grad_cpu = grad_base.transpose(1, 2)
        grad = grad_base.to("dgx").transpose(1, 2)
    elif grad_layout == "unaligned":
        grad_base = torch.randn(batch, heads, sequence, dim + 1)
        grad_cpu = grad_base[..., 1:]
        grad = grad_base.to("dgx")[..., 1:]
    else:
        grad_cpu = torch.randn_like(expected)
        grad = grad_cpu.to("dgx")
    expected.backward(grad_cpu)
    actual.backward(grad)
    torch.testing.assert_close(base.grad.cpu(), base_cpu.grad, atol=2e-5, rtol=5e-5)


@pytest.mark.parametrize("requires_grad", [(True, False, False), (False, True, True)])
@pytest.mark.parametrize("dims", [(32, 16), (32, 96), (96, 32), (128, 256), (256, 128)])
@pytest.mark.parametrize("key_length", [37, 137])
def test_sdpa_cross_attention(dgx_setup_teardown, requires_grad, dims, key_length):
    query_dim, value_dim = dims
    shapes = [
        (3, 2, 19, query_dim),
        (3, 2, key_length, query_dim),
        (3, 2, key_length, value_dim),
    ]
    inputs_cpu = [
        torch.randn(shape, requires_grad=grad)
        for shape, grad in zip(shapes, requires_grad)
    ]
    inputs = [
        value.detach().to("dgx").requires_grad_(grad)
        for value, grad in zip(inputs_cpu, requires_grad)
    ]
    with sdpa_kernel(SDPBackend.MATH):
        expected = F.scaled_dot_product_attention(*inputs_cpu)
    with sdpa_kernel(SDPBackend.EFFICIENT_ATTENTION):
        actual = F.scaled_dot_product_attention(*inputs)
    grad = torch.randn_like(expected)
    expected.backward(grad)
    actual.backward(grad.to("dgx"))
    torch.testing.assert_close(actual.cpu(), expected, atol=1e-5, rtol=3e-5)
    for value, reference, needs_grad in zip(inputs, inputs_cpu, requires_grad):
        if needs_grad:
            torch.testing.assert_close(
                value.grad.cpu(), reference.grad, atol=2e-5, rtol=5e-5
            )
        else:
            assert value.grad is None


@pytest.mark.parametrize(
    "batch,heads,queries,keys,dim,value_dim",
    [
        (1, 1, 256, 137, 32, 16),
        (1, 3, 257, 193, 96, 32),
        (3, 1, 321, 137, 32, 96),
        (1, 3, 511, 257, 132, 256),
        (1, 3, 17, 513, 32, 16),
        (3, 1, 65, 257, 96, 32),
        (1, 1, 1, 512, 132, 256),
        (1, 3, 257, 512, 96, 32),
    ],
)
@pytest.mark.parametrize(
    "mask_kind,grad_kind",
    [
        (mask_kind, grad_kind)
        for mask_kind in ["none", "additive", "broadcast"]
        for grad_kind in ["all", "query", "key_value", "bias"]
        if mask_kind != "none" or grad_kind != "bias"
    ],
)
def test_sdpa_sequence_partition(
    dgx_setup_teardown,
    batch,
    heads,
    queries,
    keys,
    dim,
    value_dim,
    mask_kind,
    grad_kind,
):
    shapes = [
        (batch, heads, queries, dim),
        (batch, heads, keys, dim),
        (batch, heads, keys, value_dim),
    ]
    needs_grad = [
        grad_kind in ("all", "query"),
        grad_kind in ("all", "key_value"),
        grad_kind in ("all", "key_value"),
    ]
    inputs_cpu = [
        torch.randn(shape).double().requires_grad_(required)
        for shape, required in zip(shapes, needs_grad)
    ]
    # Offset channel views exercise copies at nonzero query or key origins.
    inputs = []
    for reference, required in zip(inputs_cpu, needs_grad):
        storage = torch.empty(*reference.shape[:-1], reference.size(-1) + 1)
        storage[..., 1:] = reference.float()
        inputs.append(storage.to("dgx")[..., 1:].detach().requires_grad_(required))
    bias_cpu = bias = None
    if mask_kind != "none":
        mask_shape = (
            (queries, keys)
            if mask_kind == "broadcast"
            else (batch, heads, queries, keys)
        )
        base = torch.randn(mask_shape)
        # Include fully masked rows on both sides of a possible shard boundary.
        base[..., ::64, :] = -torch.inf
        bias_cpu = base.double().requires_grad_(grad_kind in ("all", "bias"))
        bias = base.to("dgx").requires_grad_(bias_cpu.requires_grad)
    with sdpa_kernel(SDPBackend.MATH):
        expected = F.scaled_dot_product_attention(*inputs_cpu, attn_mask=bias_cpu)
    with sdpa_kernel(SDPBackend.EFFICIENT_ATTENTION):
        actual = F.scaled_dot_product_attention(*inputs, attn_mask=bias)
    grad_base = torch.randn(batch, heads, 1, value_dim)
    grad_cpu = grad_base.expand(expected.shape)
    grad = grad_base.to("dgx").expand(actual.shape)
    expected.backward(grad_cpu.double())
    actual.backward(grad)
    torch.testing.assert_close(actual.cpu(), expected.float(), atol=1e-5, rtol=3e-5)
    for tensor, reference in zip(inputs + [bias], inputs_cpu + [bias_cpu]):
        if reference is not None and reference.requires_grad:
            torch.testing.assert_close(
                tensor.grad.cpu(), reference.grad.float(), atol=4e-5, rtol=8e-5
            )


@pytest.mark.parametrize("visible", ["first", "last", "both", "none"])
@pytest.mark.parametrize("scale", [None, 0.0, -0.25])
def test_sdpa_key_partition_masked_shards(dgx_setup_teardown, visible, scale):
    shapes = [(1, 3, 65, 32), (1, 3, 513, 32), (1, 3, 513, 96)]
    references = [torch.randn(shape).double().requires_grad_() for shape in shapes]
    inputs = [x.detach().float().to("dgx").requires_grad_() for x in references]
    # Stepped bias columns check that each shard uses the global key origin.
    storage = torch.full((1, 3, 65, 1026), -torch.inf)
    if visible in ("first", "both"):
        storage[..., 0] = 0.0
    if visible in ("last", "both"):
        storage[..., -2] = 0.0
    bias_cpu = storage[..., ::2].double().requires_grad_()
    bias = storage.to("dgx")[..., ::2].detach().requires_grad_()
    with sdpa_kernel(SDPBackend.MATH):
        expected = F.scaled_dot_product_attention(
            *references, attn_mask=bias_cpu, scale=scale
        )
    actual, lse, _, _ = torch.ops.aten._scaled_dot_product_efficient_attention(
        *inputs, bias, True, scale=scale
    )
    torch.testing.assert_close(actual.cpu(), expected.float(), atol=1e-5, rtol=3e-5)
    grad = torch.randn_like(expected).float()
    expected_grads = torch.autograd.grad(
        expected, [*references, bias_cpu], grad.double()
    )
    actual_grads = torch.autograd.grad(actual, [*inputs, bias], grad.to("dgx"))
    for actual_grad, reference in zip(actual_grads, expected_grads):
        torch.testing.assert_close(
            actual_grad.cpu(), reference.float(), atol=4e-5, rtol=8e-5
        )
    scores = (references[0] @ references[1].transpose(-1, -2)) * (
        32**-0.5 if scale is None else scale
    )
    expected_lse = torch.logsumexp(scores + bias_cpu, -1)
    # Match CUTLASS's zero convention for fully masked rows and +inf padding.
    expected_lse = expected_lse.masked_fill(torch.isneginf(expected_lse), 0.0)
    torch.testing.assert_close(
        lse.cpu()[..., :65], expected_lse.float(), atol=1e-5, rtol=3e-5
    )
    assert torch.isposinf(lse.cpu()[..., 65:]).all()
    with torch.no_grad():
        inference, no_lse, _, _ = (
            torch.ops.aten._scaled_dot_product_efficient_attention(
                *inputs, bias.detach(), False, scale=scale
            )
        )
    assert no_lse.shape == (1, 3, 0)
    torch.testing.assert_close(inference.cpu(), expected.float(), atol=1e-5, rtol=3e-5)


@pytest.mark.parametrize("offset", [-1000.0, 0.0, 1000.0])
def test_sdpa_key_partition_stable_merge(dgx_setup_teardown, offset):
    shapes = [(1, 3, 17, 32), (1, 3, 512, 32), (1, 3, 512, 16)]
    references = [torch.randn(shape).double().requires_grad_() for shape in shapes]
    inputs = [x.detach().float().to("dgx").requires_grad_() for x in references]
    # The two key shards have very different normalization constants. A plain
    # average of their outputs or an unshifted exp(LSE) merge is incorrect.
    bias_cpu = torch.full((17, 512), offset, dtype=torch.float64)
    bias_cpu[:, 256:] += 80.0
    with sdpa_kernel(SDPBackend.MATH):
        expected = F.scaled_dot_product_attention(*references, attn_mask=bias_cpu)
    with sdpa_kernel(SDPBackend.EFFICIENT_ATTENTION):
        actual = F.scaled_dot_product_attention(
            *inputs, attn_mask=bias_cpu.float().to("dgx")
        )
    grad = torch.randn_like(expected).float()
    expected_grads = torch.autograd.grad(expected, references, grad.double())
    actual_grads = torch.autograd.grad(actual, inputs, grad.to("dgx"))
    torch.testing.assert_close(actual.cpu(), expected.float(), atol=3e-5, rtol=1e-4)
    for actual_grad, reference in zip(actual_grads, expected_grads):
        torch.testing.assert_close(
            actual_grad.cpu(), reference.float(), atol=4e-5, rtol=1e-4
        )


@pytest.mark.parametrize("queries", [1, 65])
@pytest.mark.parametrize("masked", [False, True])
def test_sdpa_key_partition_broadcast_inference(dgx_setup_teardown, queries, masked):
    query = torch.randn(1, 3, queries, 96)
    key, value = [torch.randn(1, 1, 513, 96) for _ in range(2)]
    references = [query, *(x.expand(1, 3, 513, 96) for x in (key, value))]
    inputs = [
        query.to("dgx"),
        *(x.to("dgx").expand(1, 3, 513, 96) for x in (key, value)),
    ]
    mask = torch.arange(513).expand(queries, -1) > 400 if masked else None
    with torch.no_grad(), sdpa_kernel(SDPBackend.MATH):
        expected = F.scaled_dot_product_attention(*references, attn_mask=mask)
    with torch.no_grad(), sdpa_kernel(SDPBackend.EFFICIENT_ATTENTION):
        actual = F.scaled_dot_product_attention(
            *inputs, attn_mask=None if mask is None else mask.to("dgx")
        )
    torch.testing.assert_close(actual.cpu(), expected, atol=1e-5, rtol=3e-5)


@pytest.mark.parametrize(
    "heads,queries,keys", [(1, 1, 257), (3, 65, 257), (3, 257, 137)]
)
@pytest.mark.parametrize("saved_offset", [0, 1])
def test_sdpa_sequence_partition_backward_streams(
    dgx_setup_teardown, heads, queries, keys, saved_offset
):
    shapes = [(1, heads, queries, 32), (1, heads, keys, 32), (1, heads, keys, 96)]
    references = [torch.randn(shape).double().requires_grad_() for shape in shapes]
    grad_cpu = torch.randn(1, heads, queries, 96)
    with sdpa_kernel(SDPBackend.MATH):
        expected = F.scaled_dot_product_attention(*references)
    expected_grads = torch.autograd.grad(expected, references, grad_cpu.double())

    with ExitStack() as stack:
        for device in range(torch.cuda.device_count()):
            stack.enter_context(torch.cuda.stream(torch.cuda.Stream(device=device)))
        inputs = [x.detach().float().to("dgx") for x in references]
        grad = grad_cpu.to("dgx")
        results = []
        # Odd numbers of head rows require reduce-scatter padding. Repeated
        # backward calls also exercise stream-ordered lifetime of local dQ.
        for repeat in range(3):
            with torch.cuda.device(torch.cuda.device_count() - 1):
                torch.cuda._sleep(1_000_000)
            actual, lse, seed, offset = (
                torch.ops.aten._scaled_dot_product_efficient_attention(
                    *inputs, None, True
                )
            )
            if saved_offset:
                storage = torch.empty(actual.numel() + saved_offset, device="dgx")
                shifted = (
                    storage[saved_offset:]
                    .reshape(1, queries, heads, 96)
                    .transpose(1, 2)
                )
                shifted.copy_(actual)
                actual = shifted
            gradients = torch.ops.aten._scaled_dot_product_efficient_attention_backward(
                grad,
                *inputs,
                None,
                actual,
                lse,
                seed,
                offset,
                0.0,
                [True, True, True, False],
                False,
            )[:3]
            # Consume the managed result immediately, before synchronizing.
            results.append([g * (repeat + 1) for g in gradients])
        for repeat, gradients in enumerate(results):
            for actual_grad, reference in zip(gradients, expected_grads):
                torch.testing.assert_close(
                    actual_grad.cpu(),
                    (reference * (repeat + 1)).float(),
                    atol=4e-5,
                    rtol=8e-5,
                )


@pytest.mark.parametrize(
    "shape,layout",
    [
        (shape, layout)
        for shape in [(1, 5, 65, 96), (3, 3, 17, 32), (1, 1, 17, 16)]
        for layout in [
            "head-major",
            "sequence-major",
            "sliced",
            "unaligned",
            "odd-strides",
            "channel-stride",
            "channel-transpose",
            "batch-broadcast",
            "head-broadcast",
            "sequence-broadcast",
            "channel-broadcast",
        ]
        # Exercise every layout where all broadcast axes are non-singleton.
        if shape == (3, 3, 17, 32) or layout in ["head-major", "sequence-major"]
    ],
)
@pytest.mark.parametrize("grad_layout", ["contiguous", "zero-stride"])
def test_sdpa_input_layouts(dgx_setup_teardown, shape, layout, grad_layout):
    batch, heads, sequence, dim = shape
    storage_shape = list(shape)
    if layout == "sequence-major":
        storage_shape = [batch, sequence, heads, dim]
    elif layout == "sliced":
        storage_shape = [batch + 2, heads * 2, sequence * 2, dim + 4]
    elif layout == "unaligned":
        storage_shape[-1] += 4
    elif layout == "odd-strides":
        storage_shape[-1] += 1
    elif layout == "channel-stride":
        storage_shape[-1] *= 2
    elif layout == "channel-transpose":
        storage_shape = [batch, heads, dim, sequence]
    elif layout.endswith("broadcast"):
        axis = {
            "batch-broadcast": 0,
            "head-broadcast": 1,
            "sequence-broadcast": 2,
            "channel-broadcast": 3,
        }[layout]
        storage_shape[axis] = 1
    bases_cpu = [torch.randn(storage_shape, requires_grad=True) for _ in range(3)]
    bases = [x.detach().to("dgx").requires_grad_() for x in bases_cpu]

    def view(x):
        if layout == "sequence-major":
            return x.transpose(1, 2)
        if layout == "sliced":
            return x[1:-1, ::2, ::2, 4:]
        if layout == "unaligned":
            return x[..., 1 : dim + 1]
        if layout == "odd-strides":
            return x[..., :dim]
        if layout == "channel-stride":
            return x[..., ::2]
        if layout == "channel-transpose":
            return x.transpose(2, 3)
        return x.expand(shape)

    # Test attention's logical gradients directly; ExpandBackward's reductions
    # are covered separately.
    # Accumulate the reference in FP64: repeated channel values can amplify
    # FP32 rounding in the reference's dot products and softmax.
    inputs_cpu = [view(x.double()).detach().requires_grad_() for x in bases_cpu]
    inputs = [view(x).detach().requires_grad_() for x in bases]
    with sdpa_kernel(SDPBackend.MATH):
        expected = F.scaled_dot_product_attention(*inputs_cpu)
    with sdpa_kernel(SDPBackend.EFFICIENT_ATTENTION):
        assert (
            torch.ops.aten._fused_sdp_choice(*inputs)
            == SDPBackend.EFFICIENT_ATTENTION.value
        )
        actual = F.scaled_dot_product_attention(*inputs)
    # Broadcast entire query rows as well as channels, testing direct dO packing.
    grad_shape = shape if grad_layout == "contiguous" else (1, heads, 1, 1)
    grad_base = torch.randn(grad_shape)
    expected_grads = torch.autograd.grad(
        expected, inputs_cpu, grad_base.double().expand(shape)
    )
    actual_grads = torch.autograd.grad(
        actual, inputs, grad_base.to("dgx").expand(shape)
    )
    torch.testing.assert_close(actual.cpu(), expected.float(), atol=1e-5, rtol=3e-5)
    for gradient, reference in zip(actual_grads, expected_grads):
        torch.testing.assert_close(
            gradient.cpu(), reference.float(), atol=2e-5, rtol=5e-5
        )


@pytest.mark.parametrize("axis", [0, 1, 2])
def test_sdpa_large_singleton_stride(dgx_setup_teardown, axis):
    # Singleton axes can have strides beyond int32 without a large allocation.
    # The source strides never reach CUTLASS after attention packs local inputs.
    shape = [3, 3, 17, 32]
    shape[axis] = 1
    bases_cpu = [torch.randn(shape) for _ in range(3)]
    bases = [x.to("dgx") for x in bases_cpu]

    def view(x):
        strides = list(x.stride())
        strides[axis] = 2**33
        return x.as_strided(shape, strides).detach().requires_grad_()

    inputs_cpu = [view(x) for x in bases_cpu]
    inputs = [view(x) for x in bases]
    with sdpa_kernel(SDPBackend.MATH):
        expected = F.scaled_dot_product_attention(*inputs_cpu)
    with sdpa_kernel(SDPBackend.EFFICIENT_ATTENTION):
        actual = F.scaled_dot_product_attention(*inputs)
    grad = torch.randn(shape)
    expected_grads = torch.autograd.grad(expected, inputs_cpu, grad)
    actual_grads = torch.autograd.grad(actual, inputs, grad.to("dgx"))
    torch.testing.assert_close(actual.cpu(), expected, atol=1e-5, rtol=3e-5)
    for gradient, reference in zip(actual_grads, expected_grads):
        torch.testing.assert_close(gradient.cpu(), reference, atol=2e-5, rtol=5e-5)


@pytest.mark.parametrize("queries", [17, 257])
@pytest.mark.parametrize("mask_kind", ["none", "causal", "additive"])
def test_sdpa_saved_logsumexp(dgx_setup_teardown, mask_kind, queries):
    query = torch.randn(1, 3, queries, 32)
    key, value = [torch.randn(1, 3, 37, 32) for _ in range(2)]
    bias = torch.randn(1, 3, queries, 37) if mask_kind == "additive" else None
    _, lse, _, _ = torch.ops.aten._scaled_dot_product_efficient_attention(
        *(x.to("dgx") for x in (query, key, value)),
        bias.to("dgx") if bias is not None else None,
        True,
        is_causal=mask_kind == "causal",
    )
    scores = (query @ key.transpose(-2, -1)) * (32**-0.5)
    if bias is not None:
        scores += bias
    if mask_kind == "causal":
        scores.masked_fill_(
            ~torch.ones(queries, 37, dtype=torch.bool).tril(), -torch.inf
        )
    expected = torch.logsumexp(scores, -1)
    actual = lse.cpu()
    assert actual.shape == (1, 3, (queries + 31) // 32 * 32)
    torch.testing.assert_close(actual[..., :queries], expected)
    assert torch.isposinf(actual[..., queries:]).all()


@pytest.mark.parametrize("dim", [16, 96, 256, 512])
def test_sdpa_inference(dgx_setup_teardown, dim):
    inputs_cpu = [torch.randn(3, 2, 17, dim) for _ in range(3)]
    inputs = [value.to("dgx") for value in inputs_cpu]
    with torch.no_grad(), sdpa_kernel(SDPBackend.EFFICIENT_ATTENTION):
        actual = F.scaled_dot_product_attention(*inputs)
        output, lse, seed, offset = (
            torch.ops.aten._scaled_dot_product_efficient_attention(*inputs, None, False)
        )
    assert lse.shape == (3, 2, 0)
    assert seed.item() == offset.item() == 0
    torch.testing.assert_close(
        actual.cpu(), F.scaled_dot_product_attention(*inputs_cpu), atol=1e-5, rtol=3e-5
    )
    torch.testing.assert_close(output.cpu(), actual.cpu())


@pytest.mark.parametrize("dim", [4, 96, 132])
@pytest.mark.parametrize("heads", [3, 5])
@pytest.mark.parametrize("query_length", [1, 65, 257])
def test_sdpa_broadcast_head_inference(dgx_setup_teardown, dim, heads, query_length):
    # CUTLASS packs heads into the single-query tail tile when K/V are broadcast.
    # An odd number of heads checks that this stays within each GPU's shard.
    query = torch.randn(1, heads, query_length, dim)
    key, value = [torch.randn(1, 1, 17, dim) for _ in range(2)]
    inputs_cpu = [query, key.expand(1, heads, 17, dim), value.expand(1, heads, 17, dim)]
    inputs = [
        query.to("dgx"),
        *(x.to("dgx").expand(1, heads, 17, dim) for x in [key, value]),
    ]
    with torch.no_grad(), sdpa_kernel(SDPBackend.MATH):
        expected = F.scaled_dot_product_attention(*inputs_cpu)
    with torch.no_grad(), sdpa_kernel(SDPBackend.EFFICIENT_ATTENTION):
        actual = F.scaled_dot_product_attention(*inputs)
    torch.testing.assert_close(actual.cpu(), expected, atol=1e-5, rtol=3e-5)


@pytest.mark.parametrize("dim", [96, 256])
@pytest.mark.parametrize("queries", [129, 257])
@pytest.mark.parametrize("mask_kind", ["none", "causal", "additive"])
def test_sdpa_deterministic_backward(dgx_setup_teardown, dim, queries, mask_kind):
    inputs_cpu = [torch.randn(1, 3, queries, dim, requires_grad=True) for _ in range(3)]
    inputs = [x.detach().to("dgx").requires_grad_() for x in inputs_cpu]
    grad = torch.randn(1, 3, queries, dim)
    kwargs_cpu = {"is_causal": mask_kind == "causal"}
    kwargs = dict(kwargs_cpu)
    if mask_kind == "additive":
        kwargs_cpu["attn_mask"] = torch.randn(queries, queries)
        kwargs["attn_mask"] = kwargs_cpu["attn_mask"].to("dgx")
    with sdpa_kernel(SDPBackend.MATH):
        expected = F.scaled_dot_product_attention(*inputs_cpu, **kwargs_cpu)
        expected.backward(grad)
    deterministic = torch.are_deterministic_algorithms_enabled()
    warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
    try:
        torch.use_deterministic_algorithms(True)
        first_gradients = None
        for _ in range(3):
            with sdpa_kernel(SDPBackend.EFFICIENT_ATTENTION):
                actual = F.scaled_dot_product_attention(*inputs, **kwargs)
            gradients = torch.autograd.grad(actual, inputs, grad.to("dgx"))
            gradients = [x.cpu() for x in gradients]
            for actual_grad, reference in zip(gradients, inputs_cpu):
                torch.testing.assert_close(
                    actual_grad, reference.grad, atol=2e-5, rtol=5e-5
                )
            if first_gradients is None:
                first_gradients = gradients
            else:
                for actual_grad, first_grad in zip(gradients, first_gradients):
                    torch.testing.assert_close(actual_grad, first_grad, atol=0, rtol=0)
    finally:
        torch.use_deterministic_algorithms(deterministic, warn_only=warn_only)


@pytest.mark.parametrize("dims", [(512, 512), (65536, 4), (4, 65536)])
def test_sdpa_large_head_forward_backward(dgx_setup_teardown, dims):
    # CUDA's general CUTLASS specialization caps both Q/K and V at 2**16.
    # Keep sequence lengths small to exercise that boundary with little memory.
    query_dim, value_dim = dims
    shapes = [(1, 1, 3, query_dim), (1, 1, 5, query_dim), (1, 1, 5, value_dim)]
    # Accumulate the long dot products in FP64 for an accurate reference, while
    # keeping the sampled inputs and upstream gradient representable in FP32.
    inputs_cpu = [torch.randn(shape).double().requires_grad_() for shape in shapes]
    inputs = [value.detach().float().to("dgx").requires_grad_() for value in inputs_cpu]
    with sdpa_kernel(SDPBackend.MATH):
        expected = F.scaled_dot_product_attention(*inputs_cpu)
    with sdpa_kernel(SDPBackend.EFFICIENT_ATTENTION):
        actual = F.scaled_dot_product_attention(*inputs)
    grad = torch.randn(expected.shape)
    expected.backward(grad.double())
    actual.backward(grad.to("dgx"))
    torch.testing.assert_close(actual.cpu(), expected.float(), atol=1e-5, rtol=3e-5)
    for value, reference in zip(inputs, inputs_cpu):
        torch.testing.assert_close(
            value.grad.cpu(), reference.grad.float(), atol=2e-5, rtol=5e-5
        )


@pytest.mark.parametrize("dims", [(65540, 4), (4, 65540)])
def test_sdpa_head_dim_limit(dgx_setup_teardown, dims):
    query_dim, value_dim = dims
    inputs = [
        torch.randn(1, 1, 2, dim).to("dgx") for dim in (query_dim, query_dim, value_dim)
    ]
    with sdpa_kernel([SDPBackend.EFFICIENT_ATTENTION, SDPBackend.MATH]):
        assert torch.ops.aten._fused_sdp_choice(*inputs) == SDPBackend.MATH.value
    with (
        sdpa_kernel(SDPBackend.EFFICIENT_ATTENTION),
        pytest.raises(RuntimeError, match="No available DGX SDPA kernel"),
    ):
        F.scaled_dot_product_attention(*inputs)
    with pytest.raises(
        RuntimeError, match="Unsupported DGX efficient attention configuration"
    ):
        torch.ops.aten._scaled_dot_product_efficient_attention(*inputs, None, False)


def test_sdpa_output_stride_limit(dgx_setup_teardown):
    # The inputs fit in memory, but a sequence-major output would have an
    # overflowing int32 row stride (32768 heads * 65536 channels).
    inputs = [
        torch.randn(1, 1, 1, 65536).to("dgx").expand(1, 32768, 1, 65536)
        for _ in range(3)
    ]
    with sdpa_kernel([SDPBackend.EFFICIENT_ATTENTION, SDPBackend.MATH]):
        assert torch.ops.aten._fused_sdp_choice(*inputs) == SDPBackend.MATH.value
    with (
        sdpa_kernel(SDPBackend.EFFICIENT_ATTENTION),
        pytest.raises(RuntimeError, match="No available DGX SDPA kernel"),
    ):
        F.scaled_dot_product_attention(*inputs)


@pytest.mark.parametrize("case", ["double", "head-dim", "gqa"])
def test_sdpa_math_fallback_selection(dgx_setup_teardown, case):
    dtype = torch.float64 if case == "double" else torch.float32
    dim = 37 if case == "head-dim" else 16
    inputs = [torch.randn(3, 2, 17, dim, dtype=dtype).to("dgx") for _ in range(3)]
    kwargs = {}
    if case == "gqa":
        kwargs["enable_gqa"] = True
    with sdpa_kernel([SDPBackend.EFFICIENT_ATTENTION, SDPBackend.MATH]):
        assert (
            torch.ops.aten._fused_sdp_choice(*inputs, **kwargs) == SDPBackend.MATH.value
        )
        actual = F.scaled_dot_product_attention(*inputs, **kwargs)
        expected = F.scaled_dot_product_attention(
            *(value.cpu() for value in inputs), **kwargs
        )
        torch.testing.assert_close(actual.cpu(), expected, atol=1e-5, rtol=3e-5)
    with (
        sdpa_kernel(SDPBackend.EFFICIENT_ATTENTION),
        pytest.raises(RuntimeError, match="No available DGX SDPA kernel"),
    ):
        F.scaled_dot_product_attention(*inputs, **kwargs)


def test_sdpa_math_backend_override(dgx_setup_teardown):
    inputs = [
        torch.randn(3, 2, 17, 16, device="dgx", requires_grad=True) for _ in range(3)
    ]
    with sdpa_kernel(SDPBackend.MATH):
        assert torch.ops.aten._fused_sdp_choice(*inputs) == SDPBackend.MATH.value
        expected = F.scaled_dot_product_attention(*inputs)
    with sdpa_kernel(SDPBackend.EFFICIENT_ATTENTION):
        actual = F.scaled_dot_product_attention(*inputs)
    torch.testing.assert_close(actual.cpu(), expected.cpu(), atol=1e-5, rtol=3e-5)


@pytest.mark.parametrize(
    "lengths,scale",
    [
        (lengths, scale)
        for lengths in [(1, 137), (17, 137), (137, 19), (129, 129)]
        for scale in [None, -0.25, 0.0]
        # Cover every causal geometry; vary scale on one multi-tile shape.
        if scale is None or lengths == (129, 129)
    ],
)
@pytest.mark.parametrize("dims", [(32, 16), (96, 96), (132, 256)])
@pytest.mark.parametrize("requires_grad", [False, True])
def test_sdpa_causal(dgx_setup_teardown, lengths, dims, scale, requires_grad):
    queries, keys = lengths
    query_dim, value_dim = dims
    shapes = [
        (3, 3, queries, query_dim),
        (3, 3, keys, query_dim),
        (3, 3, keys, value_dim),
    ]
    inputs_cpu = [
        torch.randn(shape).double().requires_grad_(requires_grad) for shape in shapes
    ]
    inputs = [
        x.detach().float().to("dgx").requires_grad_(requires_grad) for x in inputs_cpu
    ]
    kwargs = {"is_causal": True, "scale": scale}
    with sdpa_kernel(SDPBackend.MATH):
        expected = F.scaled_dot_product_attention(*inputs_cpu, **kwargs)
    with sdpa_kernel(SDPBackend.EFFICIENT_ATTENTION):
        assert (
            torch.ops.aten._fused_sdp_choice(*inputs, **kwargs)
            == SDPBackend.EFFICIENT_ATTENTION.value
        )
        actual = F.scaled_dot_product_attention(*inputs, **kwargs)
    torch.testing.assert_close(actual.cpu(), expected.float(), atol=1e-5, rtol=3e-5)
    if requires_grad:
        grad = torch.randn(actual.shape)
        expected_grads = torch.autograd.grad(expected, inputs_cpu, grad.double())
        actual_grads = torch.autograd.grad(actual, inputs, grad.to("dgx"))
        for gradient, reference in zip(actual_grads, expected_grads):
            torch.testing.assert_close(
                gradient.cpu(), reference.float(), atol=2e-5, rtol=5e-5
            )
        if keys > queries:
            # Top-left causality leaves these keys completely unused.
            assert torch.count_nonzero(actual_grads[1].cpu()[..., queries:, :]) == 0
            assert torch.count_nonzero(actual_grads[2].cpu()[..., queries:, :]) == 0


@pytest.mark.parametrize("dim", [32, 96, 132])
@pytest.mark.parametrize("boolean", [False, True])
@pytest.mark.parametrize(
    "layout",
    [
        "2d",
        "3d",
        "full",
        "padding",
        "query",
        "key",
        "transposed",
        "stepped",
        "unaligned",
    ],
)
def test_sdpa_masks(dgx_setup_teardown, dim, boolean, layout):
    batch, heads, queries, keys = 3, 3, 65, 137
    shapes = [
        (batch, heads, queries, dim),
        (batch, heads, keys, dim),
        (batch, heads, keys, dim),
    ]
    inputs_cpu = [torch.randn(shape).double().requires_grad_() for shape in shapes]
    inputs = [x.detach().float().to("dgx").requires_grad_() for x in inputs_cpu]
    mask_shape = {
        "2d": (queries, keys),
        "3d": (heads, queries, keys),
        "full": (batch, heads, queries, keys),
        "padding": (batch, 1, 1, keys),
        "query": (1, heads, queries, 1),
        "key": (1, 1, 1, keys),
        "transposed": (keys, queries),
        "stepped": (queries, 2 * keys),
        "unaligned": (queries, keys + 1),
    }[layout]
    base = torch.rand(mask_shape) > 0.25 if boolean else torch.randn(mask_shape)

    def view(x):
        if layout == "transposed":
            return x.transpose(-2, -1)
        if layout == "stepped":
            return x[..., ::2]
        if layout == "unaligned":
            return x[..., 1:]
        return x

    mask_cpu = view(base)
    mask_cpu[..., 0, :] = False if boolean else -torch.inf
    if layout == "full":
        # Exercise online softmax when the first key tiles are entirely masked
        # and only the final partial tile contributes to the remaining rows.
        mask_cpu[..., 1:, :128] = False if boolean else -torch.inf
    mask = view(base.to("dgx"))
    with sdpa_kernel(SDPBackend.MATH):
        expected = F.scaled_dot_product_attention(
            *inputs_cpu, attn_mask=mask_cpu if boolean else mask_cpu.double()
        )
    with sdpa_kernel(SDPBackend.EFFICIENT_ATTENTION):
        assert (
            torch.ops.aten._fused_sdp_choice(*inputs, attn_mask=mask)
            == SDPBackend.EFFICIENT_ATTENTION.value
        )
        actual = F.scaled_dot_product_attention(*inputs, attn_mask=mask)
    torch.testing.assert_close(actual.cpu(), expected.float(), atol=1e-5, rtol=3e-5)
    assert torch.count_nonzero(actual.cpu()[..., 0, :]) == 0
    grad = torch.randn(actual.shape)
    expected_grads = torch.autograd.grad(expected, inputs_cpu, grad.double())
    actual_grads = torch.autograd.grad(actual, inputs, grad.to("dgx"))
    for gradient, reference in zip(actual_grads, expected_grads):
        torch.testing.assert_close(
            gradient.cpu(), reference.float(), atol=2e-5, rtol=5e-5
        )


@pytest.mark.parametrize(
    "mask_shape",
    [(65, 137), (3, 65, 137), (3, 1, 1, 137), (1, 3, 65, 1), (3, 3, 65, 137)],
)
@pytest.mark.parametrize("qkv_grad", [False, True])
def test_sdpa_mask_gradients(dgx_setup_teardown, mask_shape, qkv_grad):
    shapes = [(3, 3, 65, 96), (3, 3, 137, 96), (3, 3, 137, 32), mask_shape]
    inputs_cpu = [torch.randn(shape).double() for shape in shapes]
    inputs_cpu[-1][..., 0, :] = -torch.inf
    needs_grad = [qkv_grad] * 3 + [True]
    inputs_cpu = [x.requires_grad_(g) for x, g in zip(inputs_cpu, needs_grad)]
    inputs = [
        x.detach().float().to("dgx").requires_grad_(g)
        for x, g in zip(inputs_cpu, needs_grad)
    ]
    with sdpa_kernel(SDPBackend.MATH):
        expected = F.scaled_dot_product_attention(*inputs_cpu, scale=-0.25)
    with sdpa_kernel(SDPBackend.EFFICIENT_ATTENTION):
        actual = F.scaled_dot_product_attention(*inputs, scale=-0.25)
    grad = torch.randn(actual.shape)
    expected.backward(grad.double())
    actual.backward(grad.to("dgx"))
    torch.testing.assert_close(actual.cpu(), expected.float(), atol=1e-5, rtol=3e-5)
    for value, reference, required in zip(inputs, inputs_cpu, needs_grad):
        if required:
            torch.testing.assert_close(
                value.grad.cpu(), reference.grad.float(), atol=2e-5, rtol=5e-5
            )
        else:
            assert value.grad is None


@pytest.mark.parametrize("queries", [1, 65])
@pytest.mark.parametrize("mask_kind", ["causal", "additive"])
def test_sdpa_masked_broadcast_head_inference(dgx_setup_teardown, queries, mask_kind):
    query = torch.randn(1, 5, queries, 32)
    key, value = [torch.randn(1, 1, 137, 32) for _ in range(2)]
    inputs_cpu = [query, *(x.expand(1, 5, 137, 32) for x in (key, value))]
    inputs = [
        query.to("dgx"),
        *(x.to("dgx").expand(1, 5, 137, 32) for x in (key, value)),
    ]
    kwargs_cpu = {"is_causal": mask_kind == "causal"}
    kwargs = dict(kwargs_cpu)
    if mask_kind == "additive":
        kwargs_cpu["attn_mask"] = torch.randn(1, 5, queries, 137)
        kwargs["attn_mask"] = kwargs_cpu["attn_mask"].to("dgx")
    with sdpa_kernel(SDPBackend.MATH):
        expected = F.scaled_dot_product_attention(*inputs_cpu, **kwargs_cpu)
    with sdpa_kernel(SDPBackend.EFFICIENT_ATTENTION):
        actual = F.scaled_dot_product_attention(*inputs, **kwargs)
    torch.testing.assert_close(actual.cpu(), expected, atol=1e-5, rtol=3e-5)


@pytest.mark.parametrize("layout", ["unaligned", "transposed", "stepped", "broadcast"])
def test_sdpa_direct_bias_layouts(dgx_setup_teardown, layout):
    # Bypass ATen's mask preprocessing to exercise local packing and gradient
    # copies for masks whose pointers, row strides, or key strides are unaligned.
    shapes = [(3, 3, 17, 32), (3, 3, 37, 32), (3, 3, 37, 16)]
    inputs_cpu = [torch.randn(shape).double().requires_grad_() for shape in shapes]
    inputs = [x.detach().float().to("dgx").requires_grad_() for x in inputs_cpu]
    bias_shape = {
        "unaligned": (3, 3, 17, 38),
        "transposed": (3, 3, 37, 17),
        "stepped": (3, 3, 17, 74),
        "broadcast": (1, 1, 1, 37),
    }[layout]
    base = torch.randn(bias_shape)

    def view(x):
        if layout == "unaligned":
            return x[..., 1:]
        if layout == "transposed":
            return x.transpose(-2, -1)
        if layout == "stepped":
            return x[..., ::2]
        return x.expand(3, 3, 17, 37)

    bias_cpu = view(base.double()).detach().requires_grad_()
    bias = view(base.to("dgx")).detach().requires_grad_()
    with sdpa_kernel(SDPBackend.MATH):
        expected = F.scaled_dot_product_attention(*inputs_cpu, attn_mask=bias_cpu)
    actual, _, _, _ = torch.ops.aten._scaled_dot_product_efficient_attention(
        *inputs, bias, True
    )
    grad = torch.randn(actual.shape)
    expected_grads = torch.autograd.grad(
        expected, [*inputs_cpu, bias_cpu], grad.double()
    )
    actual_grads = torch.autograd.grad(actual, [*inputs, bias], grad.to("dgx"))
    torch.testing.assert_close(actual.cpu(), expected.float(), atol=1e-5, rtol=3e-5)
    for gradient, reference in zip(actual_grads, expected_grads):
        torch.testing.assert_close(
            gradient.cpu(), reference.float(), atol=2e-5, rtol=5e-5
        )


@pytest.mark.parametrize("case", ["shape", "dtype", "rank", "device", "causal"])
def test_sdpa_unsupported_mask_selection(dgx_setup_teardown, case):
    inputs = [torch.randn(3, 2, 17, 32).to("dgx") for _ in range(3)]
    mask = torch.zeros(17, 17, device="dgx")
    if case == "shape":
        mask = mask[:, :5]
    elif case == "dtype":
        mask = mask.double()
    elif case == "rank":
        mask = mask.unsqueeze(0).unsqueeze(0).unsqueeze(0)
    elif case == "device":
        mask = mask.cpu()
    kwargs = {"attn_mask": mask, "is_causal": case == "causal"}
    with sdpa_kernel([SDPBackend.EFFICIENT_ATTENTION, SDPBackend.MATH]):
        assert (
            torch.ops.aten._fused_sdp_choice(*inputs, **kwargs) == SDPBackend.MATH.value
        )
    with (
        sdpa_kernel(SDPBackend.EFFICIENT_ATTENTION),
        pytest.raises(RuntimeError, match="No available DGX SDPA kernel"),
    ):
        torch.ops.aten._fused_sdp_choice(*inputs, **kwargs)
