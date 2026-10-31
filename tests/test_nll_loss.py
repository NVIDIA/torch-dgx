# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
import pytest
import torch
import torch.nn.functional as F


pytestmark = pytest.mark.filterwarnings("error:.*fall back to run on the CPU.*")
DTYPES = [torch.float32, torch.float64, torch.float16, torch.bfloat16]


@pytest.mark.parametrize("dtype", DTYPES, ids=str)
@pytest.mark.parametrize(
    "shape", [(7,), (17, 7), (2, 7, 11), (3, 7, 5, 9), (2, 7, 3, 5, 2)]
)
@pytest.mark.parametrize("reduction", ["none", "sum", "mean"])
@pytest.mark.parametrize("weighted", [False, True])
def test_nll_loss_forward_backward(
    dgx_setup_teardown, dtype, shape, reduction, weighted
):
    samples = torch.randn(shape, dtype=dtype)
    classes = shape[0] if len(shape) == 1 else shape[1]
    target_shape = () if len(shape) == 1 else (shape[0], *shape[2:])
    targets = torch.randint(classes, target_shape)
    if targets.numel() > 1:
        targets.reshape(-1)[::5] = -100
    weight = torch.linspace(0.5, 2, classes, dtype=dtype) if weighted else None
    input = samples.to("dgx").requires_grad_()
    # Accumulate the CPU reference in double for reduced-precision inputs;
    # CPU NLL otherwise rounds each partial sum in the input dtype.
    reference_dtype = (
        torch.float64 if dtype in (torch.float16, torch.bfloat16) else dtype
    )
    reference = samples.detach().to(reference_dtype).requires_grad_()
    actual = F.nll_loss(
        input,
        targets.to("dgx"),
        weight=None if weight is None else weight.to("dgx"),
        reduction=reduction,
    )
    expected = F.nll_loss(
        reference,
        targets,
        weight=None if weight is None else weight.to(reference_dtype),
        reduction=reduction,
    )
    torch.testing.assert_close(
        actual.cpu(), expected.detach().to(dtype), equal_nan=True
    )
    grad = torch.randn((), dtype=dtype)
    actual.backward(grad.to("dgx").expand_as(actual))
    expected.backward(grad.to(reference_dtype).expand_as(expected))
    torch.testing.assert_close(
        input.grad.cpu(), reference.grad.to(dtype), equal_nan=True
    )


@pytest.mark.parametrize(
    "shape", [(0, 7), (0, 7, 3, 5), (2, 7, 0, 5), (7,), (5, 7), (2, 7, 3, 5)]
)
@pytest.mark.parametrize("reduction", ["none", "sum", "mean"])
@pytest.mark.parametrize("zero_weight", [False, True])
def test_nll_loss_empty_ignored_and_zero_weight(
    dgx_setup_teardown, shape, reduction, zero_weight
):
    samples = torch.randn(shape)
    target_shape = () if len(shape) == 1 else (shape[0], *shape[2:])
    targets = torch.full(target_shape, 1 if zero_weight else -100, dtype=torch.int64)
    weight = torch.zeros(7) if zero_weight else None
    input = samples.to("dgx").requires_grad_()
    reference = samples.detach().requires_grad_()
    actual = F.nll_loss(
        input,
        targets.to("dgx"),
        weight=None if weight is None else weight.to("dgx"),
        reduction=reduction,
    )
    expected = F.nll_loss(
        reference,
        targets.cpu(),
        weight=None if weight is None else weight.cpu(),
        reduction=reduction,
    )
    torch.testing.assert_close(actual.cpu(), expected.detach().cpu(), equal_nan=True)
    actual.backward(torch.ones_like(actual))
    expected.backward(torch.ones_like(expected))
    torch.testing.assert_close(input.grad.cpu(), reference.grad.cpu(), equal_nan=True)


@pytest.mark.parametrize("spatial", [False, True])
@pytest.mark.parametrize("reduction", [0, 1, 2])
@pytest.mark.parametrize("dtype", DTYPES, ids=str)
def test_nll_loss_strides_and_out(dgx_setup_teardown, spatial, reduction, dtype):
    shape = (3, 7, 5, 18) if spatial else (17, 14)
    samples = torch.randn(shape, dtype=dtype)
    input = samples.to("dgx")[..., ::2]
    reference_dtype = (
        torch.float64 if dtype in (torch.float16, torch.bfloat16) else dtype
    )
    reference = samples.to(reference_dtype)[..., ::2]
    target_shape = (3, 5, 18) if spatial else (34,)
    targets = torch.randint(7, target_shape)
    targets.reshape(-1)[::7] = -100
    target = targets.to("dgx")[..., ::2]
    ref_target = targets.cpu()[..., ::2]
    weights = torch.linspace(0.5, 2, 14, dtype=dtype)
    weight = weights.to("dgx")[::2]
    ref_weight = weights.to(reference_dtype)[::2]
    forward = (
        torch.ops.aten.nll_loss2d_forward
        if spatial
        else torch.ops.aten.nll_loss_forward
    )
    backward = (
        torch.ops.aten.nll_loss2d_backward
        if spatial
        else torch.ops.aten.nll_loss_backward
    )
    expected, expected_weight = forward(
        reference, ref_target, ref_weight, reduction, -100
    )
    out_shape = list(expected.shape)
    if out_shape:
        out_shape[-1] *= 2
    output_backing = torch.full(out_shape, 42, dtype=dtype).to("dgx")
    output = output_backing[..., ::2] if out_shape else output_backing
    total_weight = torch.empty(0, device="dgx", dtype=dtype)
    returned = forward.output(
        input, target, weight, reduction, -100, output=output, total_weight=total_weight
    )
    assert returned[0] is output and returned[1] is total_weight
    torch.testing.assert_close(output.cpu(), expected.to(dtype), equal_nan=True)
    torch.testing.assert_close(total_weight.cpu(), expected_weight.to(dtype))
    if out_shape:
        torch.testing.assert_close(
            output_backing[..., 1::2].cpu(), torch.full(expected.shape, 42, dtype=dtype)
        )

    grad = torch.tensor(0.75, dtype=dtype, device="dgx").expand_as(output)
    ref_grad = torch.tensor(0.75, dtype=reference_dtype).expand_as(expected)
    grad_backing = torch.full(shape, 42, dtype=dtype).to("dgx")
    grad_input = grad_backing[..., ::2]
    returned = backward.grad_input(
        grad,
        input,
        target,
        weight,
        reduction,
        -100,
        total_weight,
        grad_input=grad_input,
    )
    assert returned is grad_input
    expected_grad = backward(
        ref_grad,
        reference,
        ref_target,
        ref_weight,
        reduction,
        -100,
        expected_weight.to(dtype).to(reference_dtype),
    )
    torch.testing.assert_close(grad_input.cpu(), expected_grad.to(dtype))
    torch.testing.assert_close(
        grad_backing[..., 1::2].cpu(), torch.full(input.shape, 42, dtype=dtype)
    )


@pytest.mark.parametrize("dtype", DTYPES, ids=str)
@pytest.mark.parametrize("shape", [(17, 23), (3, 23, 5, 7)])
def test_cross_entropy_forward_backward(dgx_setup_teardown, dtype, shape):
    samples = (torch.randn(shape) * 3).to(dtype)
    target_shape = (shape[0], *shape[2:])
    targets = torch.randint(shape[1], target_shape)
    targets.reshape(-1)[::7] = -100
    weights = torch.linspace(0.5, 2, shape[1], dtype=dtype)
    input = samples.to("dgx").requires_grad_()
    actual = F.cross_entropy(input, targets.to("dgx"), weight=weights.to("dgx"))
    # Keep the saved log probabilities and total weight in the tested dtype,
    # while using CPU double kernels as the reduction accuracy reference.
    log_probs = F.log_softmax(samples.double(), 1).to(dtype)
    forward = (
        torch.ops.aten.nll_loss_forward
        if len(shape) == 2
        else torch.ops.aten.nll_loss2d_forward
    )
    backward = (
        torch.ops.aten.nll_loss_backward
        if len(shape) == 2
        else torch.ops.aten.nll_loss2d_backward
    )
    expected, total_weight = forward(
        log_probs.double(), targets, weights.double(), 1, -100
    )
    torch.testing.assert_close(actual.cpu(), expected.to(dtype))
    actual.backward()
    loss_grad = backward(
        torch.tensor(1.0, dtype=torch.float64),
        log_probs.double(),
        targets,
        weights.double(),
        1,
        -100,
        total_weight.to(dtype).double(),
    ).to(dtype)
    expected_grad = torch.ops.aten._log_softmax_backward_data(
        loss_grad.double(), log_probs.double(), 1, torch.float64
    ).to(dtype)
    torch.testing.assert_close(input.grad.cpu(), expected_grad)


def test_nll_loss_byte_targets_and_positive_ignore_index(dgx_setup_teardown):
    samples = torch.randn(17, 7)
    target = torch.randint(7, (17,), dtype=torch.uint8)
    actual = F.nll_loss(samples.to("dgx"), target.to("dgx"), ignore_index=2)
    expected = F.nll_loss(samples, target, ignore_index=2)
    torch.testing.assert_close(actual.cpu(), expected)


@pytest.mark.parametrize(
    "case", ["target_dtype", "target_shape", "weight_shape", "weight_dtype"]
)
def test_nll_loss_invalid_metadata(dgx_setup_teardown, case):
    input = torch.randn(5, 7, device="dgx")
    target = torch.zeros(5, dtype=torch.int64, device="dgx")
    weight = torch.ones(7, device="dgx")
    if case == "target_dtype":
        target = target.float()
    elif case == "target_shape":
        target = target[:3]
    elif case == "weight_shape":
        weight = weight[:3]
    else:
        weight = weight.double()
    with pytest.raises((RuntimeError, ValueError)):
        F.nll_loss(input, target, weight=weight)
