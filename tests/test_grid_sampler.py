# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
import pytest
import torch
import torch.nn.functional as F


@pytest.mark.parametrize("mode", ["bilinear", "nearest", "bicubic"])
@pytest.mark.parametrize("padding_mode", ["zeros", "border", "reflection"])
@pytest.mark.parametrize("align_corners", [False, True])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64], ids=str)
@pytest.mark.parametrize("batch,noncontiguous", [(1, False), (3, True)])
def test_grid_sample_forward_backward(
    dgx_setup_teardown, mode, padding_mode, align_corners, dtype, batch, noncontiguous
):
    input = torch.randn(batch, 3, 5, 7, dtype=dtype)
    # Include out-of-bounds coordinates to exercise each padding mode.
    grid = torch.rand(batch, 7, 9, 2, dtype=dtype) * 6 - 3
    grad_output = torch.randn(batch, 3, 7, 9, dtype=dtype)

    def run(device):
        x = input.to(device)
        g = grid.to(device)
        grad = grad_output.to(device)
        if noncontiguous:
            x = x.transpose(2, 3)
            g = g.transpose(1, 2)
            grad = grad.transpose(2, 3)
        x.requires_grad_()
        g.requires_grad_()
        output = F.grid_sample(
            x, g, mode=mode, padding_mode=padding_mode, align_corners=align_corners
        )
        gradients = torch.autograd.grad(output, (x, g), grad)
        return output.detach().cpu(), *(value.cpu() for value in gradients)

    reference = run("cpu")
    actual = run("dgx")
    for name, result, expected in zip(
        ("output", "grad_input", "grad_grid"), actual, reference
    ):
        # Bicubic border gradients can nearly cancel in float32.
        atol = 1e-5
        if (
            dtype == torch.float32
            and mode == "bicubic"
            and padding_mode == "border"
            and name == "grad_grid"
        ):
            atol = 3e-5
        torch.testing.assert_close(
            result,
            expected,
            rtol=1e-4,
            atol=atol,
            msg=lambda message: f"{name}: {message}",
        )


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16], ids=str)
@pytest.mark.parametrize("mode", ["bilinear", "nearest", "bicubic"])
def test_grid_sample_low_precision(dgx_setup_teardown, dtype, mode):
    input = torch.randn(3, 2, 5, 7, dtype=dtype)
    grid = torch.rand(3, 7, 9, 2, dtype=dtype) * 4 - 2
    outputs = []
    # Use float32 on CPU for the low-precision reference kernels.
    for device in ("cpu", "dgx"):
        run_dtype = torch.float32 if device == "cpu" else dtype
        x = input.to(device=device, dtype=run_dtype).requires_grad_()
        g = grid.to(device=device, dtype=run_dtype).requires_grad_()
        result = F.grid_sample(
            x, g, mode=mode, padding_mode="reflection", align_corners=False
        )
        gradients = torch.autograd.grad(result, (x, g), torch.ones_like(result))
        outputs.append((result.detach().cpu(), *(value.cpu() for value in gradients)))
    for actual, expected in zip(outputs[1], outputs[0]):
        torch.testing.assert_close(actual, expected.to(dtype), rtol=0.05, atol=0.05)


@pytest.mark.parametrize("output_mask", [(True, False), (False, True)])
def test_grid_sample_backward_output_mask(dgx_setup_teardown, output_mask):
    input = torch.randn(3, 2, 5, 7)
    grid = torch.rand(3, 7, 9, 2) * 4 - 2
    grad = torch.randn(3, 2, 7, 9)
    outputs = []
    for device in ("cpu", "dgx"):
        outputs.append(
            torch.ops.aten.grid_sampler_2d_backward.default(
                grad.to(device),
                input.to(device),
                grid.to(device),
                0,
                0,
                False,
                output_mask,
            )
        )
    for index, requested in enumerate(output_mask):
        if requested:
            torch.testing.assert_close(
                outputs[1][index].cpu(), outputs[0][index].cpu(), rtol=1e-4, atol=1e-5
            )


@pytest.mark.parametrize(
    "batch,channels,height,width",
    [(0, 2, 7, 9), (1, 0, 7, 9), (1, 2, 0, 9), (1, 2, 7, 0)],
)
def test_grid_sample_empty(dgx_setup_teardown, batch, channels, height, width):
    input = torch.randn(batch, channels, 5, 7, device="dgx", requires_grad=True)
    grid = torch.empty(batch, height, width, 2, device="dgx", requires_grad=True)
    output = F.grid_sample(input, grid, align_corners=False)
    assert output.shape == (batch, channels, height, width)
    grad_input, grad_grid = torch.autograd.grad(
        output, (input, grid), torch.empty_like(output)
    )
    torch.testing.assert_close(grad_input.cpu(), torch.zeros_like(grad_input.cpu()))
    torch.testing.assert_close(grad_grid.cpu(), torch.zeros_like(grad_grid.cpu()))


def test_grid_sample_mismatched_dtype(dgx_setup_teardown):
    input = torch.randn(1, 2, 5, 7, device="dgx")
    grid = torch.randn(1, 7, 9, 2, device="dgx", dtype=torch.float64)
    with pytest.raises(RuntimeError, match="same dtype"):
        F.grid_sample(input, grid, align_corners=False)
