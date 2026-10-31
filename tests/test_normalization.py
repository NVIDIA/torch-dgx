# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
import pytest
import torch
import torch.nn.functional as F
from torch.utils._python_dispatch import TorchDispatchMode


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64], ids=str)
def test_native_batch_norm_channel_paths(dgx_setup_teardown, dtype):
    # Cover a small shard, uneven sharding for 2--32 devices, and at least 32
    # channels per device at the backend's 32-device limit.
    channel_counts = [
        3,
        67,
        1024,
    ]

    for channels in channel_counts:
        cpu_input = torch.normal(0.1, 1.2, (1, channels, 257), dtype=dtype)
        dgx_input = cpu_input.to("dgx")
        args = (None, None, None, None, True, 0.1, 1e-5)

        cpu_output = torch.ops.aten.native_batch_norm.default(cpu_input, *args)
        dgx_output = torch.ops.aten.native_batch_norm.default(dgx_input, *args)

        for actual, expected in zip(dgx_output, cpu_output):
            torch.testing.assert_close(actual.cpu(), expected, rtol=1e-5, atol=1e-7)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64], ids=str)
def test_group_norm(dgx_setup_teardown, dtype):
    cpu_input = torch.normal(0.1, 1.2, (1, 32, 26), dtype=dtype)
    dgx_input = cpu_input.to("dgx")
    module = torch.nn.GroupNorm(8, 32).to(dtype)
    cpu_output = module(cpu_input)
    module.to("dgx")
    dgx_output = module(dgx_input)
    torch.testing.assert_close(dgx_output, cpu_output.to("dgx"))


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64], ids=str)
@pytest.mark.parametrize(
    "input_shape,normalized_shape",
    [((4, 9, 23), (23,)), ((3, 11, 5, 7), (5, 7)), ((17, 768), (768,))],
)
@pytest.mark.parametrize("elementwise_affine", [False, True])
def test_layer_norm_forward_backward(
    dgx_setup_teardown,
    dtype,
    input_shape,
    normalized_shape,
    elementwise_affine,
):
    input_cpu = torch.randn(input_shape, dtype=dtype, requires_grad=True)
    input = input_cpu.detach().to("dgx").requires_grad_()
    grad_cpu = torch.randn(input_shape, dtype=dtype)
    module_cpu = torch.nn.LayerNorm(
        normalized_shape,
        elementwise_affine=elementwise_affine,
        dtype=dtype,
    )
    module = torch.nn.LayerNorm(
        normalized_shape,
        elementwise_affine=elementwise_affine,
        dtype=dtype,
    )
    module.load_state_dict(module_cpu.state_dict())
    module.to("dgx")
    atol = 1e-5 if dtype == torch.float32 else 1e-7

    output_cpu = module_cpu(input_cpu)
    output = module(input)

    torch.testing.assert_close(output.cpu(), output_cpu.detach(), rtol=1e-5, atol=atol)

    output_cpu.backward(grad_cpu)
    output.backward(grad_cpu.to("dgx"))

    torch.testing.assert_close(input.grad.cpu(), input_cpu.grad, rtol=1e-5, atol=atol)
    if elementwise_affine:
        torch.testing.assert_close(
            module.weight.grad.cpu(), module_cpu.weight.grad, rtol=1e-5, atol=atol
        )
        torch.testing.assert_close(
            module.bias.grad.cpu(), module_cpu.bias.grad, rtol=1e-5, atol=atol
        )


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64], ids=str)
@pytest.mark.parametrize(
    "case,rows",
    [
        ("empty", 0),
        ("empty_shards", 1),
        ("small", 128),
        # Cover reduction boundaries with one device and up to 32 devices.
        ("threshold", 129),
        ("threshold", 4097),
        ("row_tail", 257),
        ("row_tail", 8193),
        ("large", 32769),
    ],
)
@pytest.mark.parametrize(
    "mask",
    [
        (True, True, True),
        (False, True, False),
        (False, False, True),
        (True, False, False),
        (False, False, False),
    ],
)
def test_layer_norm_backward_parameter_reduction(
    dgx_setup_teardown, dtype, case, rows, mask
):
    normalized_shape = (192,) if case == "large" else (5, 7)
    shape = (rows, *normalized_shape)
    if case == "row_tail":
        # Uneven row shards, a partial feature tile, a sliced input, and a
        # broadcast output gradient exercise both reductions' indexing.
        base_cpu = torch.randn((*shape, 2), dtype=dtype)
        input_cpu = base_cpu[..., 1]
        input = base_cpu.to("dgx")[..., 1]
        grad_base = torch.randn((*shape[:-1], 1), dtype=dtype)
        grad_cpu = grad_base.expand(shape)
        grad = grad_base.to("dgx").expand(shape)
    else:
        input_cpu = torch.randn(shape, dtype=dtype)
        input = input_cpu.to("dgx")
        grad_cpu = torch.randn(shape, dtype=dtype)
        grad = grad_cpu.to("dgx")
    weight_cpu = torch.randn(normalized_shape, dtype=dtype)
    bias_cpu = torch.randn(normalized_shape, dtype=dtype)
    _, mean, rstd = torch.ops.aten.native_layer_norm(
        input_cpu, normalized_shape, weight_cpu, bias_cpu, 1e-5
    )
    # Share saved statistics to isolate backward from forward's rounding.
    expected = torch.ops.aten.native_layer_norm_backward(
        grad_cpu,
        input_cpu,
        normalized_shape,
        mean,
        rstd,
        weight_cpu,
        bias_cpu,
        mask,
    )
    actual = torch.ops.aten.native_layer_norm_backward(
        grad,
        input,
        normalized_shape,
        mean.to("dgx"),
        rstd.to("dgx"),
        weight_cpu.to("dgx"),
        bias_cpu.to("dgx"),
        mask,
    )
    for index, (enabled, got, want) in enumerate(zip(mask, actual, expected)):
        if not enabled:
            assert got is None
            continue
        if dtype == torch.float64:
            rtol, atol = 1e-10, 1e-10
        elif index == 0:
            rtol, atol = 1e-5, 1e-5
        else:
            # Parameter sums over many rows have a different reduction order.
            rtol, atol = 1e-4, 1e-3
        torch.testing.assert_close(got.cpu(), want, rtol=rtol, atol=atol)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64], ids=str)
@pytest.mark.parametrize(
    "shape,normalized_shape",
    [
        ((7,), (7,)),
        ((1, 1), (1,)),
        ((11, 96), (96,)),
        ((9, 1025), (1025,)),
        ((2, 3, 5, 7), (5, 7)),
        ((3, 5, 7, 9, 11), (7, 9, 11)),
        ((0, 3, 7), (7,)),
        ((2, 0), (0,)),
    ],
)
@pytest.mark.parametrize("affine", [False, True])
@pytest.mark.parametrize("grad_layout", ["contiguous", "zero-stride"])
def test_rms_norm_forward_backward(
    dgx_setup_teardown, dtype, shape, normalized_shape, affine, grad_layout
):
    x_cpu = torch.randn(shape, dtype=dtype, requires_grad=True)
    x = x_cpu.detach().to("dgx").requires_grad_()
    w_cpu = (
        torch.randn(normalized_shape, dtype=dtype, requires_grad=True)
        if affine
        else None
    )
    w = w_cpu.detach().to("dgx").requires_grad_() if affine else None
    if grad_layout == "zero-stride":
        grad_cpu = torch.randn((*shape[:-1], 1), dtype=dtype)
        grad = grad_cpu.to("dgx").expand(shape)
        grad_cpu = grad_cpu.expand(shape)
    else:
        grad_cpu = torch.randn(shape, dtype=dtype)
        grad = grad_cpu.to("dgx")
    expected = F.rms_norm(x_cpu, normalized_shape, w_cpu, eps=1e-6)
    actual = F.rms_norm(x, normalized_shape, w, eps=1e-6)
    torch.testing.assert_close(actual.cpu(), expected)
    expected.backward(grad_cpu)
    actual.backward(grad)
    torch.testing.assert_close(x.grad.cpu(), x_cpu.grad)
    if affine:
        torch.testing.assert_close(w.grad.cpu(), w_cpu.grad)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64], ids=str)
@pytest.mark.parametrize("eps", [None, 0.0, 1e-6, 0.1])
@pytest.mark.parametrize("zero_input", [False, True])
def test_fused_rms_norm_saved_statistics(dgx_setup_teardown, dtype, eps, zero_input):
    x_cpu = (
        torch.zeros(7, 33, dtype=dtype)
        if zero_input
        else torch.randn(7, 33, dtype=dtype)
    )
    expected = torch.ops.aten._fused_rms_norm(x_cpu, [33], None, eps)
    actual = torch.ops.aten._fused_rms_norm(x_cpu.to("dgx"), [33], None, eps)
    for got, want in zip(actual, expected):
        torch.testing.assert_close(got.cpu(), want, equal_nan=True)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64], ids=str)
@pytest.mark.parametrize("layout", ["packed", "transposed", "sliced", "expanded"])
@pytest.mark.parametrize("grad_layout", ["contiguous", "zero-stride", "sliced"])
def test_rms_norm_strides(dgx_setup_teardown, dtype, layout, grad_layout):
    base_cpu = torch.randn(3, 7, 3, 5, 33, dtype=dtype)

    def view(base):
        if layout == "packed":
            return base[1:, 1:-1, 1].transpose(1, 2), (33,)
        if layout == "transposed":
            return base[1, :, 0].transpose(1, 2), (33, 5)
        if layout == "sliced":
            return base[1, :, 1, :, 1::2], (16,)
        return base[1:2, :, 1].expand(3, -1, -1, -1), (33,)

    x_cpu, normalized_shape = view(base_cpu)
    x, _ = view(base_cpu.to("dgx"))
    x_cpu = x_cpu.detach().requires_grad_()
    x = x.detach().requires_grad_()
    w_base = torch.randn((*normalized_shape, 2), dtype=dtype)
    w_cpu = w_base[..., 1].detach().requires_grad_()
    w = w_base.to("dgx")[..., 1].detach().requires_grad_()
    if grad_layout == "zero-stride":
        base_grad = torch.randn((*x.shape[:-1], 1), dtype=dtype)
        grad_cpu = base_grad.expand(x.shape)
        grad = base_grad.to("dgx").expand(x.shape)
    elif grad_layout == "sliced":
        base_grad = torch.randn((*x.shape, 2), dtype=dtype)
        grad_cpu = base_grad[..., 1]
        grad = base_grad.to("dgx")[..., 1]
    else:
        grad_cpu = torch.randn(x.shape, dtype=dtype)
        grad = grad_cpu.to("dgx")
    expected = F.rms_norm(x_cpu, normalized_shape, w_cpu, eps=1e-6)
    actual = F.rms_norm(x, normalized_shape, w, eps=1e-6)
    torch.testing.assert_close(actual.cpu(), expected)
    expected.backward(grad_cpu)
    actual.backward(grad)
    torch.testing.assert_close(x.grad.cpu(), x_cpu.grad)
    torch.testing.assert_close(w.grad.cpu(), w_cpu.grad)


@pytest.mark.parametrize(
    "mask", [(True, False), (False, True), (True, True), (False, False)]
)
@pytest.mark.parametrize("affine", [False, True])
def test_fused_rms_norm_backward_mask(dgx_setup_teardown, mask, affine):
    x_cpu = torch.randn(5, 17, requires_grad=True)
    w_cpu = torch.randn(17, requires_grad=True) if affine else None
    grad_cpu = torch.randn_like(x_cpu)
    expected = F.rms_norm(x_cpu, (17,), w_cpu, eps=1e-6)
    expected.backward(grad_cpu)
    x = x_cpu.detach().to("dgx")
    w = w_cpu.detach().to("dgx") if affine else None
    _, stats = torch.ops.aten._fused_rms_norm(x, [17], w, 1e-6)
    grads = torch.ops.aten._fused_rms_norm_backward(
        grad_cpu.to("dgx"), x, [17], stats, w, mask
    )
    for enabled, actual, expected in zip(
        (mask[0], mask[1] and affine),
        grads,
        (x_cpu.grad, w_cpu.grad if affine else None),
    ):
        if enabled:
            torch.testing.assert_close(actual.cpu(), expected)
        else:
            assert actual is None


class _RMSNormDispatchTrace(TorchDispatchMode):
    def __init__(self):
        self.ops = []

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        self.ops.append(func)
        return func(*args, **(kwargs or {}))


@pytest.mark.parametrize("batch", [1, 3])
@pytest.mark.parametrize("grad_layout", ["contiguous", "zero-stride"])
def test_rms_norm_attention_integration(dgx_setup_teardown, batch, grad_layout):
    # Compare the entire projection-to-attention gradient chain to explicit CPU RMS arithmetic.
    heads, dim, sequence = 3, 12, 17
    hidden = heads * dim
    inputs_cpu = [
        torch.randn(batch, sequence, hidden, requires_grad=True),
        (torch.randn(3 * hidden, hidden) * 0.1).requires_grad_(),
        torch.randn(3 * hidden, requires_grad=True),
    ]
    inputs = [t.detach().to("dgx").requires_grad_() for t in inputs_cpu]

    def attention(tensors, fused):
        packed = F.linear(*tensors).view(batch, sequence, 3, heads, dim)
        q, k, v = packed.permute(2, 0, 3, 1, 4).unbind(0)
        if fused:
            q, k = (F.rms_norm(t, (dim,), eps=1e-6) for t in (q, k))
        else:
            q, k = (
                t * torch.rsqrt(t.square().mean(-1, keepdim=True) + 1e-6)
                for t in (q, k)
            )
        return F.scaled_dot_product_attention(q, k, v)

    expected = attention(inputs_cpu, False)
    with _RMSNormDispatchTrace() as trace:
        actual = attention(inputs, True)
    assert torch.ops.aten._scaled_dot_product_efficient_attention.default in trace.ops
    grad_cpu = torch.randn(
        batch, heads, sequence, 1 if grad_layout == "zero-stride" else dim
    )
    grad = grad_cpu.to("dgx").expand_as(actual)
    expected.backward(grad_cpu.expand_as(expected))
    actual.backward(grad)
    assert torch.allclose(actual, expected.to("dgx"), rtol=5e-4, atol=5e-5)
    for actual_input, expected_input in zip(inputs, inputs_cpu):
        assert torch.allclose(
            actual_input.grad, expected_input.grad.to("dgx"), rtol=5e-4, atol=5e-5
        )


@pytest.mark.parametrize(
    "normalized_shape,weight_shape",
    [((), None), ((8,), None), ((5, 7), (35,)), ((2, 5, 7), None)],
)
def test_fused_rms_norm_invalid_shape(
    dgx_setup_teardown, normalized_shape, weight_shape
):
    x = torch.randn(5, 7, device="dgx")
    w = torch.ones(weight_shape, device="dgx") if weight_shape else None
    with pytest.raises(RuntimeError):
        torch.ops.aten._fused_rms_norm(x, normalized_shape, w, 1e-6)
