# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
import torch
import pytest


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64], ids=str)
def test_mse_loss(dgx_setup_teardown, dtype):
    loss = torch.nn.MSELoss()
    input = torch.rand(3, 5, dtype=dtype).to("dgx")
    target = torch.rand(3, 5, dtype=dtype).to("dgx")
    output = loss(input, target)
    expected = loss(input.to("cpu"), target.to("cpu")).to("dgx")
    torch.testing.assert_close(output, expected)


@pytest.mark.parametrize(
    "input_shape,target_shape,permutation,dtype",
    [
        pytest.param(
            (4, 3, 5),
            (1, 3, 1),
            (2, 1, 0),
            torch.float32,
            id="broadcast-noncontiguous-float32",
        ),
        pytest.param(
            (4, 3, 5),
            (1, 3, 1),
            (2, 1, 0),
            torch.float64,
            id="broadcast-noncontiguous-float64",
        ),
        pytest.param((0, 3), (0, 3), None, torch.float32, id="empty"),
    ],
)
@pytest.mark.parametrize("reduction", ["sum", "mean"])
def test_mse_loss_reduction(
    dgx_setup_teardown,
    input_shape,
    target_shape,
    permutation,
    dtype,
    reduction,
):
    input = torch.randn(input_shape, dtype=dtype)
    target = torch.randn(target_shape, dtype=dtype)
    if permutation is not None:
        input = input.permute(permutation)

    if input.shape != target.shape:
        warning = "Using a target size .* that is different to the input size"
        with pytest.warns(UserWarning, match=warning):
            actual = torch.nn.functional.mse_loss(
                input.to("dgx"), target.to("dgx"), reduction=reduction
            )
        with pytest.warns(UserWarning, match=warning):
            expected = torch.nn.functional.mse_loss(input, target, reduction=reduction)
    else:
        actual = torch.nn.functional.mse_loss(
            input.to("dgx"), target.to("dgx"), reduction=reduction
        )
        expected = torch.nn.functional.mse_loss(input, target, reduction=reduction)

    torch.testing.assert_close(actual.cpu(), expected, equal_nan=True)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64], ids=str)
@pytest.mark.parametrize("reduction", ["none", "sum", "mean"])
@pytest.mark.parametrize("layout", ["contiguous", "slice", "transpose", "broadcast", "empty"])
def test_mse_loss_backward_layouts(dgx_setup_teardown, dtype, reduction, layout):
    input_data = torch.randn(2, 7, 8, dtype=dtype)
    target_data = torch.randn_like(input_data)
    cpu_input = input_data.clone().requires_grad_()
    cpu_target = target_data.clone().requires_grad_()
    input = input_data.to("dgx").requires_grad_()
    target = target_data.to("dgx").requires_grad_()

    def views(x, y):
        if layout == "slice":
            # Channel fields of a decoder output have gaps and nonzero offsets.
            return x[:, 1:6, 1:4], y[:, 1:6, 4:7]
        if layout == "transpose":
            return x.transpose(0, 2), y.transpose(1, 2).contiguous().transpose(1, 2).transpose(0, 2)
        if layout == "broadcast":
            return x[:1, :, 1:2].expand_as(y), y
        if layout == "empty":
            return x[:, :0], y[:, :0]
        return x, y

    expected = torch.nn.functional.mse_loss(*views(cpu_input, cpu_target), reduction=reduction)
    actual = torch.nn.functional.mse_loss(*views(input, target), reduction=reduction)
    torch.testing.assert_close(actual.cpu(), expected, equal_nan=True)

    # A varying, strided upstream gradient detects treating grad_output as scalar.
    grad_data = torch.randn(*expected.shape, 2, dtype=dtype)
    expected.backward(grad_data[..., 1])
    actual.backward(grad_data.to("dgx")[..., 1])
    torch.testing.assert_close(input.grad.cpu(), cpu_input.grad)
    torch.testing.assert_close(target.grad.cpu(), cpu_target.grad)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64], ids=str)
@pytest.mark.parametrize("reduction", [0, 1, 2])
def test_mse_loss_strided_out(dgx_setup_teardown, dtype, reduction):
    input_data = torch.randn(4, 7, dtype=dtype)
    target_data = torch.randn(1, 7, dtype=dtype)
    cpu_input = input_data[:, 1::2]
    cpu_target = target_data[:, 1::2].expand_as(cpu_input)
    input = input_data.to("dgx")[:, 1::2]
    target = target_data.to("dgx")[:, 1::2].expand_as(input)

    expected = torch.ops.aten.mse_loss(cpu_input, cpu_target, reduction)
    if reduction == 0:
        storage = torch.full((4, 7), -99.0, dtype=dtype, device="dgx")
        out = storage[:, 1::2]
        result = torch.ops.aten.mse_loss.out(input, target, reduction, out=out)
        assert result is out
        torch.testing.assert_close(result.cpu(), expected)
        torch.testing.assert_close(storage.cpu()[:, ::2], torch.full((4, 4), -99.0, dtype=dtype))

    grad_data = torch.randn(*expected.shape, 2, dtype=dtype)
    expected_grad = torch.ops.aten.mse_loss_backward(grad_data[..., 1], cpu_input, cpu_target, reduction)
    grad_storage = torch.full((4, 7), -99.0, dtype=dtype, device="dgx")
    grad_input = grad_storage[:, 1::2]
    result = torch.ops.aten.mse_loss_backward.grad_input(
        grad_data.to("dgx")[..., 1], input, target, reduction, grad_input=grad_input
    )
    assert result is grad_input
    torch.testing.assert_close(result.cpu(), expected_grad)
    torch.testing.assert_close(grad_storage.cpu()[:, ::2], torch.full((4, 4), -99.0, dtype=dtype))
