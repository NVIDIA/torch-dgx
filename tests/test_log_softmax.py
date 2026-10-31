# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
import pytest
import torch
import torch.nn.functional as F


pytestmark = pytest.mark.filterwarnings("error:.*fall back to run on the CPU.*")
DTYPES = [torch.float32, torch.float64, torch.float16, torch.bfloat16]


def cpu_log_softmax_backward(grad, output, dim, input_dtype):
    # Use the CPU operator with a double accumulator: native CPU half/bfloat16
    # backward rounds intermediate sums, and long float32 reductions can also
    # lose precision. Supply the same saved output to isolate backward accuracy.
    return torch.ops.aten._log_softmax_backward_data(
        grad.double(), output.double(), dim, torch.float64
    ).to(input_dtype)


@pytest.mark.parametrize("dtype", DTYPES, ids=str)
@pytest.mark.parametrize(
    "shape,dim",
    [
        ((), 0),
        ((7,), -1),
        ((0, 7), -1),
        ((3, 0), 1),
        ((17, 513), -1),
        ((2, 3, 11), 0),
        ((2, 3, 11), 1),
        ((2, 3, 11), -1),
        ((2, 65537), -1),
    ],
)
@pytest.mark.parametrize("broadcast_grad", [False, True])
def test_log_softmax_forward_backward(
    dgx_setup_teardown, dtype, shape, dim, broadcast_grad
):
    samples = (torch.randn(shape) * 5 + 1000).to(dtype)
    # Use CPU double precision for the reference: native float32 reductions can
    # lose accuracy on long rows (especially with AVX2), and half/bfloat16 can
    # round small normalization terms to zero.
    input = samples.to("dgx").requires_grad_()
    expected = F.log_softmax(samples.double(), dim).to(dtype)
    actual = F.log_softmax(input, dim)
    torch.testing.assert_close(actual.cpu(), expected.detach().cpu(), equal_nan=True)

    grad = (
        torch.randn((), dtype=dtype)
        if broadcast_grad
        else torch.randn(shape, dtype=dtype)
    )
    expected_grad = cpu_log_softmax_backward(
        grad.expand(shape), actual.detach().cpu(), dim, dtype
    )
    actual.backward(grad.to("dgx").expand(shape))
    tolerances = {"rtol": 1e-5, "atol": 1e-5} if dtype == torch.float32 else {}
    torch.testing.assert_close(
        input.grad.cpu(), expected_grad, equal_nan=True, **tolerances
    )


@pytest.mark.parametrize("dtype", DTYPES, ids=str)
@pytest.mark.parametrize("dim", [0, 1, -1])
def test_log_softmax_strided_out(dgx_setup_teardown, dtype, dim):
    samples = torch.randn(3, 5, 14, dtype=dtype)
    input = samples.to("dgx")[..., ::2].transpose(0, 1)
    reference = samples[..., ::2].transpose(0, 1)
    backing = torch.full((5, 3, 14), 42, dtype=dtype).to("dgx")
    out = backing[..., ::2]
    returned = torch.ops.aten._log_softmax.out(input, dim, False, out=out)
    expected = F.log_softmax(reference, dim)
    assert returned is out
    torch.testing.assert_close(out.cpu(), expected.cpu())
    torch.testing.assert_close(
        backing[..., 1::2].cpu(), torch.full((5, 3, 7), 42, dtype=dtype)
    )

    grad_base = torch.randn(3, 5, 14, dtype=dtype)
    grad = grad_base.to("dgx")[..., ::2].transpose(0, 1)
    ref_grad = grad_base[..., ::2].transpose(0, 1)
    grad_backing = torch.full((5, 3, 14), 42, dtype=dtype).to("dgx")
    grad_input = grad_backing[..., ::2]
    returned = torch.ops.aten._log_softmax_backward_data.out(
        grad, out, dim, dtype, out=grad_input
    )
    expected_grad = cpu_log_softmax_backward(ref_grad, out.cpu(), dim, dtype)
    assert returned is grad_input
    torch.testing.assert_close(grad_input.cpu(), expected_grad.cpu())
    torch.testing.assert_close(
        grad_backing[..., 1::2].cpu(), torch.full((5, 3, 7), 42, dtype=dtype)
    )


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16], ids=str)
def test_log_softmax_dtype_promotion(dgx_setup_teardown, dtype):
    samples = torch.randn(7, 23, dtype=dtype)
    input = samples.to("dgx").requires_grad_()
    reference = samples.detach().requires_grad_()
    actual = F.log_softmax(input, -1, dtype=torch.float32)
    expected = F.log_softmax(reference, -1, dtype=torch.float32)
    assert actual.dtype == torch.float32
    torch.testing.assert_close(actual.cpu(), expected.detach().cpu())
    grad = torch.randn_like(samples, dtype=torch.float32)
    actual.backward(grad.to("dgx"))
    expected.backward(grad)
    torch.testing.assert_close(input.grad.cpu(), reference.grad)


def test_log_softmax_half_to_float_dispatch(dgx_setup_teardown):
    samples = torch.randn(7, 23, dtype=torch.float16)
    actual = torch.ops.aten._log_softmax(samples.to("dgx"), -1, True)
    expected = F.log_softmax(samples.float(), -1)
    assert actual.dtype == torch.float32
    torch.testing.assert_close(actual.cpu(), expected.cpu())
    grad = torch.randn(samples.shape)
    actual_grad = torch.ops.aten._log_softmax_backward_data(
        grad.to("dgx"), actual, -1, torch.float16
    )
    expected_grad = torch.ops.aten._log_softmax_backward_data(
        grad, expected, -1, torch.float32
    ).half()
    assert actual_grad.dtype == torch.float16
    torch.testing.assert_close(actual_grad.cpu(), expected_grad.cpu())


def test_log_softmax_nonfinite_and_extreme(dgx_setup_teardown):
    samples = torch.tensor(
        [
            [10000.0, -10000.0, 0.0],
            [-torch.inf, -torch.inf, -torch.inf],
            [torch.inf, 0.0, -torch.inf],
            [float("nan"), -torch.inf, 0.0],
            [-torch.inf, 0.0, 1.0],
        ]
    )
    expected = F.log_softmax(samples, -1)
    actual = F.log_softmax(samples.to("dgx"), -1)
    torch.testing.assert_close(actual.cpu(), expected, equal_nan=True)
    assert actual.cpu()[0, 1].item() == -20000.0
    grad = torch.randn_like(samples)
    expected_grad = torch.ops.aten._log_softmax_backward_data(
        grad, expected, -1, samples.dtype
    )
    actual_grad = torch.ops.aten._log_softmax_backward_data(
        grad.to("dgx"), actual, -1, samples.dtype
    )
    torch.testing.assert_close(actual_grad.cpu(), expected_grad, equal_nan=True)


def test_log_softmax_resizes_out(dgx_setup_teardown):
    samples = torch.randn(7, 13)
    out = torch.empty(0, device="dgx")
    torch.ops.aten._log_softmax.out(samples.to("dgx"), -1, False, out=out)
    torch.testing.assert_close(out.cpu(), F.log_softmax(samples, -1))
    grad_input = torch.empty(0, device="dgx")
    grad = torch.ones_like(samples)
    torch.ops.aten._log_softmax_backward_data.out(
        grad.to("dgx"), out, -1, torch.float32, out=grad_input
    )
    expected = torch.ops.aten._log_softmax_backward_data(
        grad, F.log_softmax(samples, -1), -1, torch.float32
    )
    torch.testing.assert_close(grad_input.cpu(), expected)
