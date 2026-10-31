# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
import itertools
import copy

import pytest
import torch
import torch.nn.functional as F
from torch.utils._python_dispatch import TorchDispatchMode


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize(
    "mask", [mask for mask in itertools.product([False, True], repeat=3) if any(mask)]
)
@pytest.mark.parametrize(
    "shape,weight_shape",
    [
        ((17, 13), (7, 13)),  # Token sharding, uneven rows and collective padding.
        ((1, 1), (1, 1)),  # Token sharding with GPUs owning no input rows.
        (
            (2, 257),
            (13, 257),
        ),  # Feature sharding with fewer output than input features.
        ((3, 17), (9, 17)),  # Uneven feature shards and complete-row dX padding.
        ((3, 9), (17, 9)),  # Feature sharding with more output than input features.
        (
            (11, 8),
            (24, 8),
        ),  # QKV-like shape: requesting dX changes the preferred partition.
        ((10, 32), (8, 32)),  # Contracting MLP: full X replication favors token shards.
        ((2, 2), (4, 2)),  # Bias reduction breaks a dW-only communication tie.
        ((3, 5, 11), (7, 11)),
        ((7,), (5, 7)),
        ((7,), (7,)),
        ((2, 5, 7), (7,)),
        ((0, 7), (5, 7)),
        ((2, 0, 7), (5, 7)),
        ((2, 3, 0), (5, 0)),
        ((2, 3, 7), (0, 7)),
    ],
)
def test_linear_backward_masks(dgx_setup_teardown, dtype, mask, shape, weight_shape):
    x = torch.randn(shape, dtype=dtype, requires_grad=True)
    w = torch.randn(weight_shape, dtype=dtype, requires_grad=True)
    bias_shape = () if len(weight_shape) == 1 else (weight_shape[0],)
    b = torch.randn(bias_shape, dtype=dtype, requires_grad=True)
    y = F.linear(x, w, b)
    g = torch.randn_like(y)
    reference = torch.autograd.grad(y, (x, w, b), g)
    actual = torch.ops.aten.linear_backward(
        x.detach().to("dgx"), g.to("dgx"), w.detach().to("dgx"), mask
    )
    for result, expected, requested in zip(actual, reference, mask):
        if requested:
            torch.testing.assert_close(result.cpu(), expected)
        else:
            assert result is None


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_linear_backward_no_requested_gradients(dgx_setup_teardown, dtype):
    # An empty mask returns before sharding, so one shape covers every dtype.
    x = torch.randn(17, 13, dtype=dtype, device="dgx")
    w = torch.randn(7, 13, dtype=dtype, device="dgx")
    g = torch.randn(17, 7, dtype=dtype, device="dgx")
    actual = torch.ops.aten.linear_backward(x, g, w, (False, False, False))
    assert actual == (None, None, None)


def matrix(rows, cols, layout):
    if layout == "transpose":
        return torch.randn(cols, rows, dtype=torch.float64, device="dgx").t()
    if layout == "slice":
        return torch.randn(rows, cols * 2, dtype=torch.float64, device="dgx")[:, ::2]
    if layout == "broadcast":
        return torch.randn(1, cols, dtype=torch.float64, device="dgx").expand(
            rows, cols
        )
    return torch.randn(rows * cols + 1, dtype=torch.float64, device="dgx")[1:].view(
        rows, cols
    )


@pytest.mark.parametrize(
    "shape",
    [(33, 17, 9), (3, 17, 9), (3, 9, 17)],
    ids=["tokens", "features-wide-input", "features-wide-output"],
)
@pytest.mark.parametrize(
    "layouts",
    [
        ("offset", "offset", "offset"),
        ("transpose", "offset", "offset"),
        ("offset", "transpose", "offset"),
        ("offset", "offset", "transpose"),
        ("transpose", "transpose", "transpose"),
        ("slice", "slice", "slice"),
        ("offset", "broadcast", "offset"),
    ],
)
def test_linear_backward_layouts(dgx_setup_teardown, shape, layouts):
    rows, features, outputs = shape
    x = matrix(rows, features, layouts[0])
    g = matrix(rows, outputs, layouts[1])
    w = matrix(outputs, features, layouts[2])
    cpu_x, cpu_w = [t.cpu().requires_grad_() for t in (x, w)]
    cpu_b = torch.randn(outputs, dtype=torch.float64, requires_grad=True)
    reference = torch.autograd.grad(
        F.linear(cpu_x, cpu_w, cpu_b), (cpu_x, cpu_w, cpu_b), g.cpu()
    )
    actual = torch.ops.aten.linear_backward(x, g, w, [True, True, True])
    for result, expected in zip(actual, reference):
        torch.testing.assert_close(result.cpu(), expected)


class BackwardCalls(TorchDispatchMode):
    def __init__(self):
        super().__init__()
        self.calls = 0

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        if func == torch.ops.aten.linear_backward.default:
            self.calls += 1
        return func(*args, **(kwargs or {}))


@pytest.mark.parametrize(
    "shape,outputs,bias",
    [
        ((17, 13), 7, True),
        ((3, 65, 19), 5, False),
        ((1, 512, 768), 2304, True),
        ((4, 512, 768), 3072, True),
        ((1, 16384, 192), 768, True),
    ],
)
def test_linear_autograd_dispatch(dgx_setup_teardown, shape, outputs, bias):
    x = torch.randn(shape, requires_grad=True)
    w = torch.randn(outputs, shape[-1], requires_grad=True)
    b = torch.randn(outputs, requires_grad=True) if bias else None
    dgx_inputs = [
        t.detach().to("dgx").requires_grad_() if t is not None else None
        for t in (x, w, b)
    ]
    expected = F.linear(x, w, b)
    g = torch.randn_like(expected)
    expected.backward(g)
    with BackwardCalls() as trace:
        result = F.linear(*dgx_inputs)
        result.backward(g.to("dgx"))
    assert trace.calls == 1
    torch.testing.assert_close(result.cpu(), expected, atol=3e-4, rtol=3e-4)
    for actual, reference in zip(dgx_inputs, (x, w, b)):
        if actual is not None:
            torch.testing.assert_close(
                actual.grad.cpu(), reference.grad, atol=5e-4, rtol=3e-4
            )


@pytest.mark.parametrize("shape", [(17, 7), (3, 5, 7), (1, 7), (1, 1, 7)])
@pytest.mark.parametrize(
    "mask", [(True, True, True), (True, False, True), (False, True, False)]
)
def test_linear_double_backward(dgx_setup_teardown, shape, mask):
    inputs = [
        torch.randn(s, dtype=torch.float64, requires_grad=True)
        for s in (shape, (5, 7), (5,))
    ]
    dgx_inputs = [t.detach().to("dgx").requires_grad_() for t in inputs]

    def derivatives(tensors):
        y = F.linear(*tensors)
        selected = [t for t, requested in zip(tensors, mask) if requested]
        first = torch.autograd.grad(y.square().sum(), selected, create_graph=True)
        second = torch.autograd.grad(sum(g.square().sum() for g in first), tensors)
        return first + second

    for actual, expected in zip(derivatives(dgx_inputs), derivatives(inputs)):
        torch.testing.assert_close(actual.cpu(), expected, atol=1e-7, rtol=1e-9)


def test_linear_backward_noncontiguous_leading_dimensions(dgx_setup_teardown):
    x = (
        torch.randn(3, 5, 7, dtype=torch.float64, device="dgx")
        .transpose(0, 1)
        .requires_grad_()
    )
    w = torch.randn(11, 7, dtype=torch.float64, device="dgx", requires_grad=True)
    g = torch.randn(3, 5, 11, dtype=torch.float64, device="dgx").transpose(0, 1)
    reference = [t.detach().cpu().requires_grad_() for t in (x, w)]
    expected = F.linear(*reference)
    expected.backward(g.cpu())
    result = F.linear(x, w)
    result.backward(g)
    torch.testing.assert_close(result.cpu(), expected)
    for actual, target in zip((x, w), reference):
        torch.testing.assert_close(actual.grad.cpu(), target.grad)


@pytest.mark.parametrize("batch,tokens", [(1, 17), (4, 512)])
def test_linear_attention_training(dgx_setup_teardown, batch, tokens):
    class Attention(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.qkv = torch.nn.Linear(96, 288)
            self.proj = torch.nn.Linear(96, 96)

        def forward(self, x):
            q, k, v = (
                self.qkv(x)
                .reshape(batch, tokens, 3, 3, 32)
                .permute(2, 0, 3, 1, 4)
                .unbind(0)
            )
            q, k = [F.rms_norm(t, (32,), eps=1e-6) for t in (q, k)]
            y = F.scaled_dot_product_attention(q, k, v)
            return x + self.proj(y.transpose(1, 2).reshape_as(x))

    reference = Attention()
    candidate = copy.deepcopy(reference).to("dgx")
    x = torch.randn(batch, tokens, 96, requires_grad=True)
    dx = x.detach().to("dgx").requires_grad_()
    g = torch.randn_like(x)
    for _ in range(3):
        reference.zero_grad(set_to_none=True)
        candidate.zero_grad(set_to_none=True)
        x.grad, dx.grad = None, None
        expected, actual = reference(x), candidate(dx)
        expected.backward(g)
        actual.backward(g.to("dgx"))
        torch.testing.assert_close(actual.cpu(), expected, atol=2e-4, rtol=2e-4)
        torch.testing.assert_close(dx.grad.cpu(), x.grad, atol=2e-4, rtol=2e-4)
        for p, q in zip(candidate.parameters(), reference.parameters()):
            torch.testing.assert_close(p.grad.cpu(), q.grad, atol=5e-4, rtol=3e-4)
            with torch.no_grad():
                p.add_(p.grad, alpha=-1e-4)
                q.add_(q.grad, alpha=-1e-4)


@pytest.mark.parametrize(
    "shape,weight_shape,grad_shape,message",
    [
        ((), (3, 1), (3,), "at least one dimension"),
        ((2, 3), (2, 4, 3), (2, 4), "1D or 2D weight"),
        ((2, 3), (4, 5), (2, 4), "features must match"),
        ((2, 3), (4, 3), (3, 4), "output shape"),
    ],
)
def test_linear_backward_invalid_shapes(
    dgx_setup_teardown, shape, weight_shape, grad_shape, message
):
    x, w, g = [torch.empty(s, device="dgx") for s in (shape, weight_shape, grad_shape)]
    with pytest.raises(RuntimeError, match=message):
        torch.ops.aten.linear_backward(x, g, w, [True, True, True])
