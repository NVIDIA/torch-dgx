# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
import pytest
import torch
import torch.nn.functional as F


@pytest.mark.parametrize(
    "shape,dtype,kernel_size",
    [
        ((3, 9, 11), torch.float64, (3, 2)),
        ((1, 1, 1), torch.float32, 1),
        ((0, 3, 5, 7), torch.float32, 2),
    ],
    ids=["unbatched-fp64", "single-value", "empty-batch"],
)
def test_avg_pool2d(dgx_setup_teardown, shape, dtype, kernel_size):
    source = torch.randn(shape, dtype=dtype)
    output = F.avg_pool2d(source.to("dgx"), kernel_size)
    torch.testing.assert_close(output.cpu(), F.avg_pool2d(source, kernel_size))


@pytest.mark.parametrize(
    "padding,count_include_pad,divisor_override",
    [(1, True, None), (1, False, None), (0, True, None), (1, False, -5)],
    ids=["include-pad", "exclude-pad", "no-pad", "divisor"],
)
def test_avg_pool2d_ceil_mode(
    dgx_setup_teardown, padding, count_include_pad, divisor_override
):
    source = torch.randn(2, 3, 6, 8)
    kwargs = dict(
        kernel_size=3,
        stride=(2, 3),
        padding=padding,
        ceil_mode=True,
        count_include_pad=count_include_pad,
        divisor_override=divisor_override,
    )
    output = F.avg_pool2d(source.to("dgx"), **kwargs)
    torch.testing.assert_close(output.cpu(), F.avg_pool2d(source, **kwargs))


@pytest.mark.parametrize(
    "layout,dtype", [("channels_last", torch.float32), ("strided", torch.float16)]
)
def test_avg_pool2d_layout(dgx_setup_teardown, layout, dtype):
    source = torch.randn(2, 7, 9, 11, dtype=dtype)
    tensor = source.to("dgx")
    if layout == "channels_last":
        source = source.contiguous(memory_format=torch.channels_last)
        tensor = tensor.contiguous(memory_format=torch.channels_last)
    else:
        source = source.transpose(-1, -2)[..., ::2]
        tensor = tensor.transpose(-1, -2)[..., ::2]

    output = F.avg_pool2d(tensor, 3, stride=2, padding=1)
    torch.testing.assert_close(
        output.cpu(), F.avg_pool2d(source, 3, stride=2, padding=1)
    )
    if layout == "channels_last":
        assert output.is_contiguous(memory_format=torch.channels_last)


@pytest.mark.parametrize("layout", ["resize", "strided", "alias"])
def test_avg_pool2d_out(dgx_setup_teardown, layout):
    source = torch.randn(1, 1, 257, 259)
    expected = F.avg_pool2d(source + 1, 2)
    expected_next = F.avg_pool2d(expected.transpose(-1, -2), 3, stride=1)
    tensor = source.to("dgx") + 1
    if layout == "resize":
        out = torch.empty(0, device="dgx")
    elif layout == "strided":
        storage = torch.full((*expected.shape, 2), -123.0, device="dgx")
        out = storage[..., 0]
    else:
        out = tensor[..., : expected.shape[-2], : expected.shape[-1]]

    result = torch.ops.aten.avg_pool2d.out(tensor, [2], out=out)
    assert result is out
    # Change GPU ownership through a transpose before the first host read.
    consumed = F.avg_pool2d(result.transpose(-1, -2), 3, stride=1)
    torch.testing.assert_close(consumed.cpu(), expected_next)
    torch.testing.assert_close(out.cpu(), expected)
    if layout == "strided":
        assert torch.all(storage[..., 1].cpu() == -123.0)


@pytest.mark.parametrize(
    "kwargs,match", [({"stride": [0]}, "stride"), ({"divisor_override": 0}, "divisor")]
)
def test_avg_pool2d_invalid_input(dgx_setup_teardown, kwargs, match):
    tensor = torch.empty(1, 3, 5, 7, device="dgx")
    out = torch.empty(0, device="dgx")
    with pytest.raises(RuntimeError, match=match):
        torch.ops.aten.avg_pool2d.out(tensor, [2], out=out, **kwargs)


@pytest.mark.parametrize(
    "shape,kernel,stride,padding,ceil_mode,include_pad,divisor,dtype",
    [
        ((3, 9, 11), (3, 2), (), (1, 0), False, True, None, torch.float64),
        ((1, 3, 8, 9), (3, 3), (2, 2), (1, 1), True, True, None, torch.float16),
        ((2, 3, 8, 9), (3, 3), (2, 3), (1, 1), True, False, None, torch.bfloat16),
        ((1, 2, 9, 11), (2, 2), (4, 3), (0, 0), True, False, None, torch.float32),
        ((2, 3, 5, 7), (3, 3), (1, 2), (1, 1), True, False, -5, torch.float64),
        ((0, 3, 5, 7), (2, 2), (), (0, 0), False, True, None, torch.float32),
        ((1, 1, 1), (1, 1), (), (0, 0), False, True, 7, torch.bfloat16),
    ],
    ids=[
        "unbatched",
        "ceil-pad",
        "ceil-exclude-pad",
        "skipped-pixels",
        "divisor",
        "empty",
        "single",
    ],
)
def test_avg_pool2d_backward(
    dgx_setup_teardown,
    dtype,
    shape,
    kernel,
    stride,
    padding,
    ceil_mode,
    include_pad,
    divisor,
):
    source = torch.randn(shape, dtype=dtype)
    # Use FP32 CPU accumulation as the oracle for quantized low-precision gradients.
    reference = (
        source.float() if dtype in (torch.float16, torch.bfloat16) else source.clone()
    ).requires_grad_()
    tensor = source.to("dgx").requires_grad_()
    args = (kernel, stride, padding, ceil_mode, include_pad, divisor)
    expected = F.avg_pool2d(reference, *args)
    actual = F.avg_pool2d(tensor, *args)
    gradient = torch.randn(expected.shape, dtype=dtype)
    expected.backward(gradient.to(reference.dtype))
    actual.backward(gradient.to("dgx"))
    torch.testing.assert_close(tensor.grad.cpu(), reference.grad.to(dtype))


@pytest.mark.parametrize(
    "dtype", [torch.float16, torch.bfloat16, torch.float32, torch.float64], ids=str
)
@pytest.mark.parametrize("layout", ["channels_last", "sliced", "expanded"])
def test_avg_pool2d_backward_layout(dgx_setup_teardown, dtype, layout):
    source = torch.randn(2, 7, 13, 17, dtype=dtype)
    reference = (
        source.float() if dtype in (torch.float16, torch.bfloat16) else source.clone()
    )
    tensor = source.to("dgx")
    if layout == "channels_last":
        reference = reference.contiguous(memory_format=torch.channels_last)
        tensor = tensor.contiguous(memory_format=torch.channels_last)
    elif layout == "sliced":
        reference = reference.transpose(-1, -2)[..., 1::2]
        tensor = tensor.transpose(-1, -2)[..., 1::2]
    reference.requires_grad_()
    tensor.requires_grad_()
    args = ((3, 2), (2, 1), (1, 0), True, False, None)
    expected, actual = F.avg_pool2d(reference, *args), F.avg_pool2d(tensor, *args)
    if layout == "expanded":
        storage = torch.randn(1, dtype=dtype)
        gradient = storage.expand(expected.shape)
        actual_gradient = storage.to("dgx").expand(expected.shape)
    else:
        storage = torch.randn((*expected.shape, 2), dtype=dtype)
        gradient = storage[..., 1]
        actual_gradient = storage.to("dgx")[..., 1]
    expected.backward(gradient.to(reference.dtype))
    actual.backward(actual_gradient)
    torch.testing.assert_close(tensor.grad.cpu(), reference.grad.to(dtype))
    if layout == "channels_last":
        assert tensor.grad.is_contiguous(memory_format=torch.channels_last)


@pytest.mark.parametrize(
    "layout", ["resize", "strided", "alias-gradient", "alias-input"]
)
def test_avg_pool2d_backward_out(dgx_setup_teardown, layout):
    source = torch.randn(1, 3, 129, 131)
    args = ([3], [2], [1], True, False, None)
    output_shape = F.avg_pool2d(source, *args).shape
    gradient = torch.randn(output_shape)
    expected = torch.ops.aten.avg_pool2d_backward.default(gradient, source, *args)
    tensor, actual_gradient = source.to("dgx"), gradient.to("dgx")
    if layout == "resize":
        out = torch.empty(0, device="dgx")
    elif layout == "strided":
        storage = torch.full((*source.shape, 2), -123.0, device="dgx")
        out = storage[..., 1]
    elif layout == "alias-input":
        out = tensor
    else:
        out = torch.empty_like(tensor)
        actual_gradient = out[..., : output_shape[-2], : output_shape[-1]]
        actual_gradient.copy_(gradient)
    result = torch.ops.aten.avg_pool2d_backward.grad_input(
        actual_gradient, tensor, *args, grad_input=out
    )
    assert result is out
    # Consume on DGX before any host read, with a different partition layout.
    consumed = F.avg_pool2d(result.transpose(-1, -2), 2)
    torch.testing.assert_close(
        consumed.cpu(), F.avg_pool2d(expected.transpose(-1, -2), 2)
    )
    torch.testing.assert_close(out.cpu(), expected)
    if layout == "strided":
        assert torch.all(storage[..., 0].cpu() == -123.0)


@pytest.mark.parametrize(
    "case,match",
    [
        ("batch", "batch"),
        ("shape", "size|shape"),
        ("dtype", "dtype"),
        ("overlap", "single memory location"),
    ],
)
def test_avg_pool2d_backward_invalid(dgx_setup_teardown, case, match):
    tensor = torch.randn(2, 3, 5, 7, device="dgx")
    grad = torch.randn(2, 3, 2, 3, device="dgx")
    out = torch.empty_like(tensor)
    if case == "batch":
        grad = grad[:1]
    elif case == "shape":
        grad = grad[..., :1]
    elif case == "dtype":
        grad = grad.double()
    else:
        out = torch.empty(1, device="dgx").expand(tensor.shape)
    with pytest.raises(RuntimeError, match=match):
        torch.ops.aten.avg_pool2d_backward.grad_input(
            grad, tensor, [2], [], [0], False, True, None, grad_input=out
        )
