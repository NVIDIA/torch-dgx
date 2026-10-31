# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
import pytest
import torch
import torch.nn.functional as F


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64], ids=str)
@pytest.mark.parametrize("dim", [-1, 0, 1])
def test_softmax_2d(dgx_setup_teardown, dtype, dim):
    samples = torch.randn(17, 23, dtype=dtype)
    result = torch.softmax(samples.to("dgx"), dim)
    torch.testing.assert_close(result.to("cpu"), torch.softmax(samples, dim))


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64], ids=str)
@pytest.mark.parametrize("dim", [-1, 0, 1, 2])
def test_softmax_3d(dgx_setup_teardown, dtype, dim):
    samples = torch.randn(5, 11, 13, dtype=dtype)
    result = torch.softmax(samples.to("dgx"), dim)
    torch.testing.assert_close(result.to("cpu"), torch.softmax(samples, dim))


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64], ids=str)
def test_softmax_large_values(dgx_setup_teardown, dtype):
    samples = torch.randn(8, 16, dtype=dtype) * 100
    result = torch.softmax(samples.to("dgx"), -1)
    torch.testing.assert_close(result.to("cpu"), torch.softmax(samples, -1))


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64], ids=str)
@pytest.mark.parametrize(
    "shape,dim",
    [
        pytest.param((7, 11), -1, id="last-dim"),
        pytest.param((7, 11), 0, id="first-dim"),
        pytest.param((7, 11), 1, id="last-dim-positive"),
        pytest.param((17, 1024), -1, id="large-last-dim"),
    ],
)
@pytest.mark.parametrize("grad_output_layout", ["contiguous", "zero-stride"])
def test_softmax_backward(dgx_setup_teardown, dtype, shape, dim, grad_output_layout):
    input_cpu = torch.randn(shape, dtype=dtype, requires_grad=True)
    input = input_cpu.detach().to("dgx").requires_grad_()

    if grad_output_layout == "zero-stride":
        base_shape = (shape[0], 1) if dim == 0 else (1, shape[1])
        grad_output_base_cpu = torch.randn(base_shape, dtype=dtype)
        grad_output_cpu = grad_output_base_cpu.expand_as(input_cpu)
        grad_output = grad_output_base_cpu.to("dgx").expand_as(input)
        assert 0 in grad_output.stride()
    else:
        grad_output_cpu = torch.randn_like(input_cpu)
        grad_output = grad_output_cpu.to("dgx")

    output_cpu = torch.softmax(input_cpu, dim)
    output = torch.softmax(input, dim)
    torch.testing.assert_close(output.cpu(), output_cpu.detach())
    output_cpu.backward(grad_output_cpu)
    output.backward(grad_output)

    torch.testing.assert_close(input.grad.cpu(), input_cpu.grad)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64], ids=str)
@pytest.mark.parametrize("dim", [0, 1, 2])
def test_softmax_backward_noncontiguous_grad_input(dgx_setup_teardown, dtype, dim):
    output_cpu = torch.softmax(torch.randn(3, 5, 7, dtype=dtype), dim)
    grad_output_cpu = torch.randn_like(output_cpu)
    expected = torch.ops.aten._softmax_backward_data(
        grad_output_cpu, output_cpu, dim, dtype
    )
    output = output_cpu.to("dgx")
    grad_output = grad_output_cpu.to("dgx")
    backing = torch.full((3, 5, 14), -1, dtype=dtype, device="dgx")
    grad_input = backing[..., ::2]

    returned = torch.ops.aten._softmax_backward_data.out(
        grad_output, output, dim, dtype, grad_input=grad_input
    )

    assert returned is grad_input
    assert not grad_input.is_contiguous()
    torch.testing.assert_close(grad_input.cpu(), expected)
    assert torch.equal(backing[..., 1::2].cpu(), torch.full((3, 5, 7), -1, dtype=dtype))


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64], ids=str)
@pytest.mark.parametrize(
    "shape", [(17, 1), (19, 23), (2, 3, 513), (2053, 512)], ids=str
)
@pytest.mark.parametrize("broadcast_grad", [False, True])
def test_safe_softmax_forward_backward(
    dgx_setup_teardown, dtype, shape, broadcast_grad
):
    samples = torch.randn(shape, dtype=dtype) * 3
    rows = samples.view(-1, shape[-1])
    rows[[0, -1]] = -torch.inf
    rows[1, ::2] = -torch.inf
    rows[2] = 1000
    samples.requires_grad_()
    input = samples.detach().to("dgx").requires_grad_()
    expected = torch.ops.aten._safe_softmax(samples, -1)
    output = torch.ops.aten._safe_softmax(input, -1)
    output_cpu = output.cpu()
    torch.testing.assert_close(output_cpu, expected.detach())

    grad_shape = (*shape[:-1], 1) if broadcast_grad else shape
    grad_base = torch.randn(grad_shape, dtype=dtype)
    grad_cpu = grad_base.expand(shape)
    grad = grad_base.to("dgx").expand(shape)
    expected.backward(grad_cpu)
    output.backward(grad)
    grad_input_cpu = input.grad.cpu()
    torch.testing.assert_close(grad_input_cpu, samples.grad)
    assert torch.count_nonzero(output_cpu.view_as(rows)[[0, -1]]) == 0
    assert torch.count_nonzero(grad_input_cpu.view_as(rows)[[0, -1]]) == 0


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64], ids=str)
@pytest.mark.parametrize("width", [7, 513])
def test_safe_softmax_nonfinite_rows(dgx_setup_teardown, dtype, width):
    samples = torch.full((10, width), -torch.inf, dtype=dtype)
    samples[1, 0] = torch.nan
    samples[2, -1] = torch.nan
    samples[3] = torch.nan
    samples[4, 0] = torch.inf
    samples[5] = torch.inf
    samples[6] = 1
    samples[6, width // 2] = torch.nan
    samples[7] = 1
    samples[7, width // 2] = torch.inf
    samples[8, -1] = -torch.finfo(dtype).max
    samples[9] = -torch.finfo(dtype).max
    samples.requires_grad_()
    input = samples.detach().to("dgx").requires_grad_()
    expected = torch.ops.aten._safe_softmax(samples, 1)
    output = torch.ops.aten._safe_softmax(input, 1)
    torch.testing.assert_close(output.cpu(), expected.detach(), equal_nan=True)
    # The ordinary softmax specialization must retain NaNs for fully masked rows.
    torch.testing.assert_close(
        torch.softmax(input, 1).cpu(), torch.softmax(samples, 1), equal_nan=True
    )

    grad = torch.randn_like(samples)
    expected.backward(grad)
    output.backward(grad.to("dgx"))
    torch.testing.assert_close(input.grad.cpu(), samples.grad, equal_nan=True)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64], ids=str)
@pytest.mark.parametrize("case", ["transpose", "slice", "expand", "inner_dim"])
def test_safe_softmax_layout_fallback(dgx_setup_teardown, dtype, case):
    samples = torch.randn(3, 5, 7, dtype=dtype)
    samples[0] = -torch.inf
    base = samples.to("dgx")
    dim = -1
    if case == "transpose":
        samples, base = samples.transpose(1, 2), base.transpose(1, 2)
    elif case == "slice":
        samples, base = samples[..., ::2], base[..., ::2]
    elif case == "expand":
        samples = samples[:, :1, :].expand(3, 5, 7)
        base = base[:, :1, :].expand(3, 5, 7)
    elif case == "inner_dim":
        dim = 1
    samples = samples.detach().requires_grad_()
    input = base.detach().requires_grad_()
    expected = torch.ops.aten._safe_softmax(samples, dim, dtype)
    output = torch.ops.aten._safe_softmax(input, dim, dtype)
    torch.testing.assert_close(output.cpu(), expected.detach(), equal_nan=True)
    grad = torch.randn_like(expected)
    expected.backward(grad)
    output.backward(grad.to("dgx"))
    torch.testing.assert_close(input.grad.cpu(), samples.grad, equal_nan=True)


@pytest.mark.parametrize(
    "input_dtype", [torch.float64, torch.float16, torch.bfloat16], ids=str
)
def test_safe_softmax_dtype_conversion_forward(dgx_setup_teardown, input_dtype):
    samples = torch.randn(3, 5, 7, dtype=input_dtype)
    samples[0] = -torch.inf
    if input_dtype == torch.float64:
        # Casting creates additional infinities; masking must keep the composite semantics.
        samples[1] = -1e300
    expected = torch.ops.aten._safe_softmax(samples, -1, torch.float32)
    output = torch.ops.aten._safe_softmax(samples.to("dgx"), -1, torch.float32)
    torch.testing.assert_close(output.cpu(), expected, equal_nan=True)


@pytest.mark.parametrize("shape", [(), (0, 7), (3, 0), (2, 0, 5)])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64], ids=str)
def test_safe_softmax_scalar_and_empty(dgx_setup_teardown, shape, dtype):
    samples = torch.full(shape, -torch.inf, dtype=dtype)
    expected = torch.ops.aten._safe_softmax(samples, -1, dtype)
    output = torch.ops.aten._safe_softmax(samples.to("dgx"), -1, dtype)
    torch.testing.assert_close(output.cpu(), expected)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64], ids=str)
def test_safe_softmax_attention_masked_rows(dgx_setup_teardown, dtype):
    # Match DiT's transposed head views and exercise the existing SDPA autograd path.
    bases = [
        torch.randn(3, 17, 2, 8, dtype=dtype, requires_grad=True) for _ in range(3)
    ]
    inputs = [base.detach().to("dgx").requires_grad_() for base in bases]
    mask = torch.zeros(17, 17, dtype=dtype)
    mask[0] = -torch.inf
    mask[1, ::2] = -torch.inf
    expected = F.scaled_dot_product_attention(
        *(base.transpose(1, 2) for base in bases), attn_mask=mask
    )
    output = F.scaled_dot_product_attention(
        *(input.transpose(1, 2) for input in inputs), attn_mask=mask.to("dgx")
    )
    torch.testing.assert_close(output.cpu(), expected.detach())
    grad = torch.randn_like(expected)
    expected.backward(grad)
    output.backward(grad.to("dgx"))
    for input, base in zip(inputs, bases):
        torch.testing.assert_close(input.grad.cpu(), base.grad)
