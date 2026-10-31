# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
import pytest
import torch
import torch.nn.functional as F
from torch.library import opcheck

datatypes = [torch.float32, torch.float64]


def test_neg_noncontiguous(dgx_setup_teardown):
    input_cpu = torch.arange(-15, 15, dtype=torch.float32).reshape(5, 6)
    input = input_cpu.to("dgx").t()
    assert not input.is_contiguous()

    torch.testing.assert_close(torch.neg(input).cpu(), torch.neg(input_cpu.t()))


@pytest.mark.parametrize("dtype", datatypes, ids=str)
def test_abs(dgx_setup_teardown, dtype):
    tensor = torch.normal(0, 1, (11, 3), dtype=dtype).to("dgx")
    opcheck(torch.ops.aten.abs.default, (tensor,))
    abs_tensor = torch.abs(tensor)
    torch.testing.assert_close(abs_tensor.to("cpu"), torch.abs(tensor.to("cpu")))


@pytest.mark.parametrize("dtype", datatypes, ids=str)
def test_ceil(dgx_setup_teardown, dtype):
    tensor = torch.normal(0, 1, (9, 2), dtype=dtype).to("dgx")
    opcheck(torch.ops.aten.ceil.default, (tensor,))
    ceil_tensor = torch.ceil(tensor)
    torch.testing.assert_close(ceil_tensor.to("cpu"), torch.ceil(tensor.to("cpu")))


@pytest.mark.parametrize("dtype", datatypes, ids=str)
def test_exp(dgx_setup_teardown, dtype):
    tensor = torch.normal(0, 1, (7, 4), dtype=dtype).to("dgx")
    opcheck(torch.ops.aten.exp.default, (tensor,))
    exp_tensor = torch.exp(tensor)
    torch.testing.assert_close(exp_tensor.to("cpu"), torch.exp(tensor.to("cpu")))


@pytest.mark.parametrize("dtype", datatypes, ids=str)
def test_log(dgx_setup_teardown, dtype):
    tensor = torch.add(torch.rand(2, 6, dtype=dtype).to("dgx"), 0.1)
    opcheck(torch.ops.aten.log.default, (tensor,))
    log_tensor = torch.log(tensor)
    torch.testing.assert_close(log_tensor.to("cpu"), torch.log(tensor.to("cpu")))


@pytest.mark.parametrize("dtype", datatypes, ids=str)
def test_log10(dgx_setup_teardown, dtype):
    tensor = torch.add(torch.rand(3, 5, 2, dtype=dtype).to("dgx"), 0.1)
    opcheck(torch.ops.aten.log10.default, (tensor,))
    log10_tensor = torch.log10(tensor)
    torch.testing.assert_close(log10_tensor.to("cpu"), torch.log10(tensor.to("cpu")))


@pytest.mark.parametrize("dtype", datatypes, ids=str)
def test_isnan_non_contiguous(dgx_setup_teardown, dtype):
    tensor = torch.tensor(
        [[0.0, float("nan")], [float("inf"), -1.0]], dtype=dtype, device="dgx"
    ).t()
    assert not tensor.is_contiguous()

    opcheck(torch.ops.aten.isnan.default, (tensor,))
    assert torch.equal(torch.isnan(tensor).to("cpu"), torch.isnan(tensor.to("cpu")))


@pytest.mark.parametrize("dtype", datatypes, ids=str)
def test_logit(dgx_setup_teardown, dtype):
    tensor = torch.mul(torch.rand(3, 4, dtype=dtype).to("dgx"), 0.7)
    opcheck(torch.ops.aten.logit.default, (tensor, None))
    logit_tensor = torch.logit(tensor)
    torch.testing.assert_close(logit_tensor.to("cpu"), torch.logit(tensor.to("cpu")))


@pytest.mark.parametrize("dtype", datatypes, ids=str)
def test_floor(dgx_setup_teardown, dtype):
    tensor = torch.normal(0, 1, (5, 4), dtype=dtype).to("dgx")
    opcheck(torch.ops.aten.floor.default, (tensor,))
    floor_tensor = torch.floor(tensor)
    torch.testing.assert_close(floor_tensor.to("cpu"), torch.floor(tensor.to("cpu")))


@pytest.mark.parametrize("dtype", datatypes, ids=str)
def test_relu(dgx_setup_teardown, dtype):
    sample_inputs = [(torch.normal(0, 1, (41, 3), dtype=dtype).to("dgx"),)]

    for args in sample_inputs:
        opcheck(torch.ops.aten.relu.default, args)
        output = torch.relu(*args)
        reference = torch.relu(*(arg.to("cpu") for arg in args))
        torch.testing.assert_close(output, reference.to("dgx"))


@pytest.mark.parametrize("dtype", datatypes, ids=str)
@pytest.mark.parametrize("grad_output_layout", ["contiguous", "zero-stride"])
def test_threshold_backward_out(dgx_setup_teardown, dtype, grad_output_layout):
    threshold = 0.25
    input_cpu = torch.linspace(-1, 1, 30, dtype=dtype).reshape(2, 3, 5)
    input = input_cpu.to("dgx")

    if grad_output_layout == "zero-stride":
        grad_output_cpu = torch.tensor(0.75, dtype=dtype).expand_as(input_cpu)
        grad_output = torch.tensor(0.75, dtype=dtype, device="dgx").expand_as(input)
        assert grad_output.stride() == (0, 0, 0)
    else:
        grad_output_cpu = torch.linspace(-2, 2, 30, dtype=dtype).reshape_as(input_cpu)
        grad_output = grad_output_cpu.to("dgx")

    expected = torch.ops.aten.threshold_backward.default(
        grad_output_cpu, input_cpu, threshold
    )
    grad_input = torch.empty(0, dtype=dtype, device="dgx")
    returned = torch.ops.aten.threshold_backward.grad_input(
        grad_output, input, threshold, grad_input=grad_input
    )

    assert returned is grad_input
    assert torch.equal(grad_input.cpu(), expected)


@pytest.mark.parametrize("dtype", datatypes, ids=str)
@pytest.mark.parametrize("grad_output_layout", ["contiguous", "zero-stride"])
def test_relu_backward(dgx_setup_teardown, dtype, grad_output_layout):
    input_cpu = torch.randn(2, 3, 5, dtype=dtype, requires_grad=True)
    input = input_cpu.detach().to("dgx").requires_grad_()

    if grad_output_layout == "zero-stride":
        grad_output_cpu = torch.tensor(0.75, dtype=dtype).expand_as(input_cpu)
        grad_output = torch.tensor(0.75, dtype=dtype, device="dgx").expand_as(input)
        assert grad_output.stride() == (0, 0, 0)
    else:
        grad_output_cpu = torch.randn_like(input_cpu)
        grad_output = grad_output_cpu.to("dgx")

    torch.relu(input_cpu).backward(grad_output_cpu)
    torch.relu(input).backward(grad_output)

    assert torch.equal(input.grad.cpu(), input_cpu.grad)


@pytest.mark.parametrize("dtype", datatypes, ids=str)
def test_silu_out(dgx_setup_teardown, dtype):
    input_cpu = torch.linspace(-5, 5, 33, dtype=dtype).reshape(3, 11)
    input = input_cpu.to("dgx")
    out = torch.empty(0, dtype=dtype, device="dgx")

    returned = torch.ops.aten.silu.out(input, out=out)

    assert returned is out
    torch.testing.assert_close(out.cpu(), F.silu(input_cpu))


@pytest.mark.parametrize("approximate", ["none", "tanh"])
@pytest.mark.parametrize("dtype", datatypes, ids=str)
def test_gelu_forward(dgx_setup_teardown, dtype, approximate):
    input_cpu = torch.linspace(-5, 5, 33, dtype=dtype).reshape(3, 11)
    input = input_cpu.to("dgx")

    reference = F.gelu(input_cpu, approximate=approximate)
    output = F.gelu(input, approximate=approximate)
    torch.testing.assert_close(output.cpu(), reference)


@pytest.mark.parametrize("approximate", ["none", "tanh"])
@pytest.mark.parametrize("dtype", datatypes, ids=str)
@pytest.mark.parametrize("grad_output_layout", ["contiguous", "zero-stride"])
def test_gelu_backward(dgx_setup_teardown, dtype, approximate, grad_output_layout):
    input_cpu = torch.randn(2, 3, 5, dtype=dtype, requires_grad=True)
    input = input_cpu.detach().to("dgx").requires_grad_()

    if grad_output_layout == "zero-stride":
        grad_output_cpu = torch.tensor(0.75, dtype=dtype).expand_as(input_cpu)
        grad_output = torch.tensor(0.75, dtype=dtype, device="dgx").expand_as(input)
        assert grad_output.stride() == (0, 0, 0)
    else:
        grad_output_cpu = torch.randn_like(input_cpu)
        grad_output = grad_output_cpu.to("dgx")

    F.gelu(input_cpu, approximate=approximate).backward(grad_output_cpu)
    F.gelu(input, approximate=approximate).backward(grad_output)

    torch.testing.assert_close(input.grad.cpu(), input_cpu.grad)


@pytest.mark.parametrize("dtype", datatypes, ids=str)
def test_clamp_min(dgx_setup_teardown, dtype):
    sample_inputs = [(torch.normal(0, 1, (19, 3), dtype=dtype).to("dgx"), 0.1)]

    for args in sample_inputs:
        opcheck(torch.ops.aten.clamp_min.default, args)
        output = torch.clamp_min(*args)
        reference = torch.clamp_min(
            *(arg.to("cpu") if isinstance(arg, torch.Tensor) else arg for arg in args)
        )
        torch.testing.assert_close(output, reference.to("dgx"))


@pytest.mark.parametrize("dtype", datatypes, ids=str)
def test_clamp_max(dgx_setup_teardown, dtype):
    sample_inputs = [(torch.normal(0, 1, (19, 3), dtype=dtype).to("dgx"), -0.2)]

    for args in sample_inputs:
        opcheck(torch.ops.aten.clamp_max.default, args)
        output = torch.clamp_max(*args)
        reference = torch.clamp_max(
            *(arg.to("cpu") if isinstance(arg, torch.Tensor) else arg for arg in args)
        )
        torch.testing.assert_close(output, reference.to("dgx"))


@pytest.mark.parametrize("dtype", datatypes, ids=str)
def test_clamp(dgx_setup_teardown, dtype):
    sample_inputs = [
        (torch.normal(0, 1, (19, 3), dtype=dtype).to("dgx"), -0.2, 0.3),
        (torch.normal(0, 1, (15, 2, 2), dtype=dtype).to("dgx"), 0.4, 0.05),
    ]

    for args in sample_inputs:
        opcheck(torch.ops.aten.clamp.default, args)
        output = torch.clamp(*args)
        reference = torch.clamp(
            *(arg.to("cpu") if isinstance(arg, torch.Tensor) else arg for arg in args)
        )
        torch.testing.assert_close(output, reference.to("dgx"))


@pytest.mark.parametrize("dtype", datatypes, ids=str)
def test_reciprocal_out(dgx_setup_teardown, dtype):
    tensor = torch.tensor([-4.0, -0.5, 0.25, 2.0, 8.0], dtype=dtype, device="dgx")
    opcheck(torch.ops.aten.reciprocal.default, (tensor,))

    output = torch.empty(0, dtype=dtype, device="dgx")
    returned = torch.reciprocal(tensor, out=output)

    assert returned is output
    assert output.shape == tensor.shape
    torch.testing.assert_close(output.to("cpu"), torch.reciprocal(tensor.to("cpu")))


@pytest.mark.parametrize("dtype", datatypes, ids=str)
def test_round(dgx_setup_teardown, dtype):
    tensor = torch.normal(0, 1, (19, 3), dtype=dtype).to("dgx")
    opcheck(torch.ops.aten.round.default, (tensor,))
    round_tensor = torch.round(tensor)
    torch.testing.assert_close(round_tensor.to("cpu"), torch.round(tensor.to("cpu")))

    ties = torch.tensor([-2.5, -1.5, -0.5, 0.5, 1.5, 2.5], dtype=dtype).to("dgx")
    expected = torch.tensor([-2.0, -2.0, 0.0, 0.0, 2.0, 2.0], dtype=dtype)
    torch.testing.assert_close(torch.round(ties).cpu(), expected, rtol=0, atol=0)


@pytest.mark.parametrize("dtype", datatypes, ids=str)
def test_rsqrt_out(dgx_setup_teardown, dtype):
    tensor = torch.rand(4, 5, dtype=dtype).add_(0.1).to("dgx")
    opcheck(torch.ops.aten.rsqrt.default, (tensor,))

    output = torch.empty(0, dtype=dtype, device="dgx")
    returned = torch.rsqrt(tensor, out=output)

    assert returned is output
    assert output.shape == tensor.shape
    torch.testing.assert_close(output.to("cpu"), torch.rsqrt(tensor.to("cpu")))


@pytest.mark.parametrize("dtype", datatypes, ids=str)
def test_sigmoid(dgx_setup_teardown, dtype):
    tensor = torch.normal(0, 1, (8, 14), dtype=dtype).to("dgx")
    opcheck(torch.ops.aten.sigmoid.default, (tensor,))
    sigmoid_tensor = torch.sigmoid(tensor)
    torch.testing.assert_close(
        sigmoid_tensor.to("cpu"), torch.sigmoid(tensor.to("cpu"))
    )


@pytest.mark.parametrize("dtype", datatypes, ids=str)
def test_sqrt(dgx_setup_teardown, dtype):
    tensor = torch.rand(4, 5, dtype=dtype).to("dgx")
    opcheck(torch.ops.aten.sqrt.default, (tensor,))
    sqrt_tensor = torch.sqrt(tensor)
    torch.testing.assert_close(sqrt_tensor.to("cpu"), torch.sqrt(tensor.to("cpu")))


@pytest.mark.parametrize("dtype", datatypes, ids=str)
def test_sin(dgx_setup_teardown, dtype):
    tensor = torch.rand(4, 5, dtype=dtype).to("dgx")
    opcheck(torch.ops.aten.sin.default, (tensor,))
    sin_tensor = torch.sin(tensor)
    torch.testing.assert_close(sin_tensor.to("cpu"), torch.sin(tensor.to("cpu")))


@pytest.mark.parametrize("dtype", datatypes, ids=str)
def test_cos(dgx_setup_teardown, dtype):
    tensor = torch.rand(4, 5, dtype=dtype).to("dgx")
    opcheck(torch.ops.aten.cos.default, (tensor,))
    cos_tensor = torch.cos(tensor)
    torch.testing.assert_close(cos_tensor.to("cpu"), torch.cos(tensor.to("cpu")))


def test_bitwise_not(dgx_setup_teardown):
    tensor = torch.rand(102).round().to(torch.bool).to("dgx")
    opcheck(torch.ops.aten.bitwise_not.default, (tensor,))
    bitwise_not_tensor = torch.bitwise_not(tensor)
    assert torch.equal(
        bitwise_not_tensor.to("cpu"), torch.bitwise_not(tensor.to("cpu"))
    )
