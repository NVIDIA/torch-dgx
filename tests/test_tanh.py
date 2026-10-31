# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
import pytest
import torch


pytestmark = pytest.mark.filterwarnings("error:.*fall back to run on the CPU.*")
DTYPES = [torch.float32, torch.float64, torch.float16, torch.bfloat16]


@pytest.mark.parametrize("dtype", DTYPES, ids=str)
@pytest.mark.parametrize(
    "layout", ["contiguous", "sliced", "transposed", "expanded", "empty", "scalar"]
)
def test_tanh_forward_backward(dgx_setup_teardown, dtype, layout):
    samples = torch.randn(5, 7, 18, dtype=dtype)
    input = samples.to("dgx")
    reference = samples.cpu()
    if layout == "sliced":
        input, reference = input[..., ::2], reference[..., ::2]
    elif layout == "transposed":
        input, reference = input.transpose(0, 2), reference.transpose(0, 2)
    elif layout == "expanded":
        input, reference = input[:1].expand(5, -1, -1), reference[:1].expand(5, -1, -1)
    elif layout == "empty":
        input, reference = input[:0], reference[:0]
    elif layout == "scalar":
        input, reference = input[0, 0, 0], reference[0, 0, 0]
    input = input.detach().requires_grad_()
    reference = reference.detach().requires_grad_()
    actual = torch.tanh(input)
    expected = torch.tanh(reference)
    torch.testing.assert_close(actual.cpu(), expected.detach().cpu())
    # A broadcast upstream gradient catches accidental contiguous-only backward kernels.
    actual.backward(torch.tensor(0.75, dtype=dtype, device="dgx").expand_as(actual))
    expected.backward(torch.tensor(0.75, dtype=dtype, device="cpu").expand_as(expected))
    torch.testing.assert_close(input.grad.cpu(), reference.grad.cpu())


@pytest.mark.parametrize("dtype", DTYPES, ids=str)
def test_tanh_out_and_inplace(dgx_setup_teardown, dtype):
    samples = torch.randn(7, 13, dtype=dtype)
    input = samples.to("dgx")
    backing = torch.full((7, 26), 42, dtype=dtype).to("dgx")
    out = backing[:, ::2]
    assert torch.tanh(input, out=out) is out
    torch.testing.assert_close(out.cpu(), samples.tanh())
    torch.testing.assert_close(
        backing[:, 1::2].cpu(), torch.full((7, 13), 42, dtype=dtype)
    )
    grad = torch.randn_like(samples)
    grad_backing = torch.full((7, 26), 42, dtype=dtype).to("dgx")
    grad_input = grad_backing[:, ::2]
    torch.ops.aten.tanh_backward.grad_input(grad.to("dgx"), out, grad_input=grad_input)
    expected = torch.ops.aten.tanh_backward(grad, samples.tanh())
    torch.testing.assert_close(grad_input.cpu(), expected)
    torch.testing.assert_close(
        grad_backing[:, 1::2].cpu(), torch.full((7, 13), 42, dtype=dtype)
    )
    assert input.tanh_() is input
    torch.testing.assert_close(input.cpu(), samples.tanh())


def test_tanh_nonfinite_and_integer(dgx_setup_teardown):
    samples = torch.tensor([-torch.inf, -100, -0.0, 0.0, 100, torch.inf, float("nan")])
    result = torch.tanh(samples.to("dgx")).cpu()
    torch.testing.assert_close(result, samples.tanh(), equal_nan=True)
    assert torch.signbit(result[2]) and not torch.signbit(result[3])
    integers = torch.arange(-5, 6, dtype=torch.int64)
    torch.testing.assert_close(torch.tanh(integers.to("dgx")).cpu(), integers.tanh())


@pytest.mark.parametrize(
    "input_dtype,out_dtype",
    [(torch.float16, torch.float32), (torch.float32, torch.float64)],
)
def test_tanh_out_dtype_conversion(dgx_setup_teardown, input_dtype, out_dtype):
    samples = torch.randn(7, 13, dtype=input_dtype)
    expected = torch.empty_like(samples, dtype=out_dtype)
    torch.tanh(samples, out=expected)
    out = torch.empty(samples.shape, dtype=out_dtype, device="dgx")
    torch.tanh(samples.to("dgx"), out=out)
    torch.testing.assert_close(out.cpu(), expected)

    grad = torch.randn_like(samples)
    saved = samples.tanh()
    expected_grad = torch.empty_like(samples, dtype=out_dtype)
    torch.ops.aten.tanh_backward.grad_input(grad, saved, grad_input=expected_grad)
    grad_input = torch.empty_like(out)
    torch.ops.aten.tanh_backward.grad_input(
        grad.to("dgx"), saved.to("dgx"), grad_input=grad_input
    )
    torch.testing.assert_close(grad_input.cpu(), expected_grad)
