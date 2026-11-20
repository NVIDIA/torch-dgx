# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
import pytest
import torch
import torch.nn.functional as F


@pytest.fixture(autouse=True)
def full_precision_convolution():
    with torch.backends.cudnn.flags(allow_tf32=False):
        yield


@pytest.mark.parametrize(
    "dtype,memory_format",
    [
        (torch.float32, torch.contiguous_format),
        (torch.float64, torch.contiguous_format),
        (torch.float32, torch.channels_last),
    ],
    ids=["float32", "float64", "channels-last"],
)
@pytest.mark.parametrize(
    "shape,weight_shape,stride,padding,dilation",
    [
        ((1, 3, 13, 17), (5, 3, 3, 2), (2, 1), (1, 0), (1, 1)),
        ((5, 2, 11, 13), (7, 2, 3, 3), (1, 2), (2, 1), (2, 1)),
        ((1, 2, 1, 5), (1, 2, 1, 1), (1, 1), (12, 2), (1, 1)),
        ((0, 3, 7, 9), (5, 3, 3, 3), (1, 1), (1, 1), (1, 1)),
        ((1, 2, 35, 19), (3, 2, 3, 3), (2, 1), (2, 1), (2, 1)),
        ((1, 2, 10, 7), (3, 2, 10, 3), (1, 1), (0, 1), (1, 1)),
        ((1, 2, 37, 11), (3, 2, 3, 3), (4, 2), (1, 1), (1, 1)),
        ((1, 2, 5, 9), (3, 2, 3, 3), (1, 1), (6, 1), (1, 1)),
        ((1, 3, 65, 67), (5, 3, 3, 3), (1, 1), (1, 1), (1, 1)),
    ],
    ids=[
        "spatial-shards",
        "batch-shards",
        "all-padding",
        "empty-batch",
        "dilated-uneven",
        "one-output-row",
        "stride-gaps",
        "large-halo",
        "unaligned-pages",
    ],
)
def test_convolution_backward(
    dgx_setup_teardown,
    dtype,
    memory_format,
    shape,
    weight_shape,
    stride,
    padding,
    dilation,
):
    mask = (True, True, True)
    source = torch.randn(shape, dtype=dtype).contiguous(memory_format=memory_format)
    weights = torch.randn(weight_shape, dtype=dtype).contiguous(
        memory_format=memory_format
    )
    x, w = source.to("dgx"), weights.to("dgx")
    bias = torch.randn(weight_shape[0], dtype=dtype)
    reference_output = F.conv2d(
        source, weights, bias, stride=stride, padding=padding, dilation=dilation
    )
    output = F.conv2d(
        x, w, bias.to("dgx"), stride=stride, padding=padding, dilation=dilation
    )
    if source.numel():
        assert output.is_contiguous(memory_format=memory_format)
    torch.testing.assert_close(
        output.cpu(),
        reference_output,
        rtol=1e-4 if dtype == torch.float32 else 1e-10,
        atol=2e-5 if dtype == torch.float32 else 1e-10,
    )
    grad = torch.randn(reference_output.shape, dtype=dtype).contiguous(
        memory_format=memory_format
    )
    args = ([weight_shape[0]], stride, padding, dilation, False, [0, 0], 1, mask)
    expected = torch.ops.aten.convolution_backward.default(grad, source, weights, *args)
    actual = torch.ops.aten.convolution_backward.default(grad.to("dgx"), x, w, *args)
    for result, reference in zip(actual, expected):
        assert result.device.type == "dgx"
        if source.numel():
            assert result.is_contiguous(
                memory_format=memory_format
                if result.ndim == 4
                else torch.contiguous_format
            )
        torch.testing.assert_close(
            result.cpu(),
            reference,
            rtol=1e-4 if dtype == torch.float32 else 1e-10,
            atol=2e-5 if dtype == torch.float32 else 1e-10,
        )


@pytest.mark.parametrize("batch", [1, 5])
def test_convolution_forward_precision_cache(dgx_setup_teardown, batch):
    source = torch.randn(batch, 2, 35, 19).contiguous(memory_format=torch.channels_last)
    weights = torch.randn(3, 2, 3, 3).contiguous(memory_format=torch.channels_last)
    expected = F.conv2d(source, weights, stride=(2, 1), padding=(2, 1), dilation=(2, 1))
    x, w = source.to("dgx"), weights.to("dgx")
    # Revisit both settings to exercise cached plans after a precision change.
    for allow_tf32 in (True, False, True, False):
        with torch.backends.cudnn.flags(allow_tf32=allow_tf32):
            result = F.conv2d(
                x, w, stride=(2, 1), padding=(2, 1), dilation=(2, 1)
            ).cpu()
        torch.testing.assert_close(
            result,
            expected,
            rtol=1e-2 if allow_tf32 else 1e-4,
            atol=1e-2 if allow_tf32 else 2e-5,
        )


@pytest.mark.parametrize("batch", [1, 5])
@pytest.mark.parametrize(
    "memory_format", [torch.contiguous_format, torch.channels_last], ids=str
)
@pytest.mark.parametrize(
    "mask",
    [
        (False, False, False),
        (True, False, False),
        (False, True, False),
        (False, False, True),
    ],
    ids=["none", "input", "weight", "bias"],
)
def test_convolution_backward_masks(dgx_setup_teardown, batch, memory_format, mask):
    source = torch.randn(batch, 2, 7, 9).contiguous(memory_format=memory_format)
    weights = torch.randn(5, 2, 3, 3).contiguous(memory_format=memory_format)
    grad = torch.randn(batch, 5, 7, 9)
    args = ([5], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, mask)
    expected = torch.ops.aten.convolution_backward.default(grad, source, weights, *args)
    actual = torch.ops.aten.convolution_backward.default(
        grad.to("dgx"), source.to("dgx"), weights.to("dgx"), *args
    )
    for enabled, result, reference in zip(mask, actual, expected):
        if enabled:
            torch.testing.assert_close(result.cpu(), reference, rtol=1e-4, atol=2e-5)
        else:
            assert result is None


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64], ids=str)
@pytest.mark.parametrize(
    "layout",
    [
        "channels_last",
        "channels_last_input",
        "channels_last_weight",
        "sliced",
        "expanded-gradient",
    ],
)
@pytest.mark.parametrize("batch", [1, 5])
def test_convolution_backward_layout(dgx_setup_teardown, dtype, layout, batch):
    source = torch.randn(batch, 2, 9, 11, dtype=dtype)
    weights = torch.randn(5, 2, 3, 3, dtype=dtype)
    inputs = [source, source.to("dgx")]
    filters = [weights, weights.to("dgx")]
    if layout in ("channels_last", "channels_last_input"):
        inputs = [x.contiguous(memory_format=torch.channels_last) for x in inputs]
    if layout in ("channels_last", "channels_last_weight"):
        filters = [w.contiguous(memory_format=torch.channels_last) for w in filters]
    elif layout == "sliced":
        inputs = [x.transpose(-1, -2)[..., 1::2] for x in inputs]
        filters = [w.transpose(-1, -2) for w in filters]
    reference = F.conv2d(inputs[0], filters[0], padding=1)
    output = F.conv2d(inputs[1], filters[1], padding=1)
    torch.testing.assert_close(
        output.cpu(),
        reference,
        rtol=1e-4 if dtype == torch.float32 else 1e-10,
        atol=2e-5 if dtype == torch.float32 else 1e-10,
    )
    memory_format = (
        torch.channels_last
        if layout.startswith("channels_last") and dtype != torch.float64
        else torch.contiguous_format
    )
    assert output.is_contiguous(memory_format=memory_format)
    output_shape = reference.shape
    storage = torch.randn((*output_shape, 2), dtype=dtype)
    gradients = [storage[..., 1], storage.to("dgx")[..., 1]]
    if layout == "expanded-gradient":
        base = torch.randn(1, dtype=dtype)
        gradients = [base.expand(output_shape), base.to("dgx").expand(output_shape)]
    results = [
        torch.ops.aten.convolution_backward.default(
            g, x, w, [5], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, True]
        )
        for g, x, w in zip(gradients, inputs, filters)
    ]
    for actual, expected in zip(results[1], results[0]):
        torch.testing.assert_close(
            actual.cpu(),
            expected,
            rtol=1e-4 if dtype == torch.float32 else 1e-10,
            atol=2e-5 if dtype == torch.float32 else 1e-10,
        )


@pytest.mark.parametrize("batch", [1, 5])
@pytest.mark.parametrize("bias", [False, True])
@pytest.mark.parametrize(
    "memory_format", [torch.contiguous_format, torch.channels_last], ids=str
)
def test_conv2d_avg_pool2d_training(dgx_setup_teardown, batch, bias, memory_format):
    source = torch.randn(batch, 2, 13, 17).contiguous(memory_format=memory_format)
    weights = torch.randn(5, 2, 3, 3).contiguous(memory_format=memory_format)
    biases = torch.randn(5) if bias else None
    gradient = torch.randn(batch, 5, 6, 8)

    def run(device):
        x = source.to(device).detach().requires_grad_()
        w = weights.to(device).detach().requires_grad_()
        b = biases.to(device).detach().requires_grad_() if bias else None
        conv = F.conv2d(x, w, b, padding=1)
        assert conv.is_contiguous(memory_format=memory_format)
        y = F.avg_pool2d(conv, 3, stride=2)
        y.backward(gradient.to(device))
        return [t.grad.cpu() for t in (x, w, b) if t is not None]

    expected, actual = run("cpu"), run("dgx")
    for result, reference in zip(actual, expected):
        torch.testing.assert_close(result, reference, rtol=1e-4, atol=2e-5)


@pytest.mark.parametrize(
    "case,match",
    [("groups", "group"), ("transposed", "Transposed"), ("shape", "shape")],
)
def test_convolution_backward_overrideable_invalid(dgx_setup_teardown, case, match):
    source = torch.randn(1, 2, 7, 9, device="dgx")
    weights = torch.randn(5, 2, 3, 3, device="dgx")
    grad = torch.randn(1, 5, 5, 7, device="dgx")
    if case == "shape":
        grad = grad[..., :1]
    with pytest.raises(RuntimeError, match=match):
        torch.ops.aten.convolution_backward_overrideable.default(
            grad,
            source,
            weights,
            [1, 1],
            [0, 0],
            [1, 1],
            case == "transposed",
            [0, 0],
            2 if case == "groups" else 1,
            [True, True, True],
        )
