# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
import pytest
import torch
from torch.library import opcheck

datatypes = [torch.float32, torch.float64]


@pytest.mark.parametrize("dtype", datatypes, ids=str)
def test_hypot(dgx_setup_teardown, dtype):
    lhs = torch.randn(2, 11, dtype=dtype).to("dgx")
    rhs = torch.randn(2, 11, dtype=dtype).to("dgx")
    expected = torch.hypot(lhs.cpu(), rhs.cpu())
    torch.testing.assert_close(torch.hypot(lhs, rhs).cpu(), expected)
    opcheck(torch.ops.aten.hypot.default, (lhs, rhs))

    out = torch.empty_like(lhs)
    assert torch.hypot(lhs, rhs, out=out) is out
    torch.testing.assert_close(out.cpu(), expected)


@pytest.mark.parametrize("dtype", [torch.int32, torch.int64, torch.bool])
@pytest.mark.parametrize("transpose", [False, True])
@pytest.mark.parametrize("scalar_other", [False, True])
def test_true_divide_integer_promotes(
    dgx_setup_teardown, dtype, transpose, scalar_other
):
    reference = torch.arange(-10, 11).reshape(3, 7).to(dtype)
    if transpose:
        reference = reference.t()
    other = 2 if scalar_other else torch.full_like(reference, 2)
    actual = reference.to("dgx")
    dgx_other = other if scalar_other else other.to("dgx")
    torch.testing.assert_close(
        torch.div(actual, dgx_other).cpu(), torch.div(reference, other)
    )


def _transposed_key(tensor, heads, head_dim):
    return tensor.reshape(*tensor.shape[:2], heads, head_dim).permute(0, 2, 3, 1)


@pytest.mark.parametrize("dtype", datatypes, ids=str)
def test_add(dgx_setup_teardown, dtype):
    a = torch.rand(2, 11, dtype=dtype).to("dgx")
    b = torch.rand(2, 11, dtype=dtype).to("dgx")
    c = torch.add(a, 47)
    ref = torch.add(a.to("cpu"), 47).to("dgx")
    torch.testing.assert_close(c, ref)
    opcheck(torch.ops.aten.add.Scalar, (a, 47))

    c = torch.add(a, b, alpha=4)
    ref = torch.add(a.to("cpu"), b.to("cpu"), alpha=4).to("dgx")
    torch.testing.assert_close(c, ref)
    opcheck(torch.ops.aten.add.Tensor, (a, b), {"alpha": 4})

    c = torch.add(a, b.to(torch.int32), alpha=4)
    ref = torch.add(a.to("cpu"), b.to(torch.int32).to("cpu"), alpha=4).to("dgx")
    torch.testing.assert_close(c, ref)
    opcheck(torch.ops.aten.add.Tensor, (a, b.to(torch.int32)), {"alpha": 4})

    a = torch.rand(1, 14, 3, dtype=dtype).to("dgx")
    b = torch.rand(3, 14, 1, dtype=dtype).to("dgx")
    c = torch.add(a, 2)
    ref = torch.add(a.to("cpu"), 2).to("dgx")
    torch.testing.assert_close(c, ref)
    opcheck(torch.ops.aten.add.Scalar, (a, 2))


@pytest.mark.parametrize("dtype", datatypes, ids=str)
def test_sub(dgx_setup_teardown, dtype):
    a = torch.rand(2, 11, dtype=dtype).to("dgx")
    b = torch.rand(2, 11, dtype=dtype).to("dgx")
    c = torch.sub(a, 31)
    ref = torch.sub(a.to("cpu"), 31).to("dgx")
    torch.testing.assert_close(c, ref)
    opcheck(torch.ops.aten.sub.Scalar, (a, 31))

    c = torch.sub(a, b, alpha=3)
    ref = torch.sub(a.to("cpu"), b.to("cpu"), alpha=3).to("dgx")
    torch.testing.assert_close(c, ref)
    opcheck(torch.ops.aten.sub.Tensor, (a, b), {"alpha": 3})

    c = torch.sub(a, b.to(torch.int32), alpha=3)
    ref = torch.sub(a.to("cpu"), b.to(torch.int32).to("cpu"), alpha=3).to("dgx")
    torch.testing.assert_close(c, ref)
    opcheck(torch.ops.aten.sub.Tensor, (a, b.to(torch.int32)), {"alpha": 3})


@pytest.mark.parametrize("dtype", datatypes, ids=str)
def test_mul(dgx_setup_teardown, dtype):
    a = torch.rand(52, dtype=dtype).to("dgx")
    b = torch.rand(52, dtype=dtype).to("dgx")
    c = torch.mul(a, 5)
    ref = torch.mul(a.to("cpu"), 5).to("dgx")
    torch.testing.assert_close(c, ref)
    opcheck(torch.ops.aten.mul.Scalar, (a, 5))

    c = torch.mul(a, b)
    ref = torch.mul(a.to("cpu"), b.to("cpu")).to("dgx")
    torch.testing.assert_close(c, ref)
    opcheck(torch.ops.aten.mul.Tensor, (a, b))

    c = torch.mul(a.to(torch.int32), b)
    ref = torch.mul(a.to(torch.int32).to("cpu"), b.to("cpu")).to("dgx")
    torch.testing.assert_close(c, ref)
    opcheck(torch.ops.aten.mul.Tensor, (a.to(torch.int32), b))


@pytest.mark.parametrize(
    "op,tensor_dtype,scalar_dtype",
    [
        (op, tensor_dtype, scalar_dtype)
        for op in [torch.add, torch.sub, torch.mul, torch.div, torch.remainder]
        for tensor_dtype, scalar_dtype in [
            (torch.float32, torch.float64),
            (torch.float64, torch.float32),
            (torch.float32, torch.int64),
            (torch.int64, torch.int32),
        ]
        if op is not torch.div or tensor_dtype != torch.int64
    ]
    + [(torch.mul, torch.complex64, torch.float64)],
    ids=lambda value: value.__name__ if callable(value) else str(value),
)
@pytest.mark.parametrize("scalar_source", ["reduction", "view"])
def test_binary_mixed_dtype_device_scalar(
    dgx_setup_teardown, op, tensor_dtype, scalar_dtype, scalar_source
):
    reference = torch.arange(32771, dtype=torch.float32).to(tensor_dtype)
    actual = reference.to("dgx")
    scalar_backing = torch.tensor([19, 1.5, 37], dtype=scalar_dtype)
    backing = scalar_backing.to("dgx")
    if scalar_source == "reduction":
        # Integer reductions produce int64; convert on device when testing an
        # int32 scalar so the consumer still has no intervening host read.
        expected_scalar = scalar_backing.sum().to(scalar_dtype)
        scalar = backing.sum().to(scalar_dtype)
    else:
        # A GPU-produced scalar with a storage offset and a nonzero neighbor
        # also detects reading more bytes than the scalar's storage dtype.
        expected_scalar = (scalar_backing * 2)[1]
        scalar = (backing * 2)[1]
    kwargs = {"alpha": -2} if op in (torch.add, torch.sub) else {}

    expected = op(reference, expected_scalar, **kwargs)
    result = op(actual, scalar, **kwargs)

    torch.testing.assert_close(result.cpu(), expected)


@pytest.mark.parametrize("shape", [(2, 3, 17, 29), (3, 3, 37, 1733)], ids=str)
@pytest.mark.parametrize(
    "op,dtype",
    [
        (op, dtype)
        for op in [torch.add, torch.sub, torch.mul, torch.div, torch.remainder]
        for dtype in [torch.float32, torch.float64, torch.int64]
        if op is not torch.div or dtype != torch.int64
    ]
    + [
        (torch.mul, torch.complex64),
        (torch.logical_and, torch.bool),
        (torch.logical_or, torch.bool),
    ],
    ids=lambda value: value.__name__ if callable(value) else str(value),
)
@pytest.mark.parametrize("scalar_device", ["cpu", "dgx"])
def test_binary_transposed_packed_key(
    dgx_setup_teardown, shape, op, dtype, scalar_device
):
    b, h, d, s = shape
    # Include a storage offset and gaps in both the sequence and batch strides.
    packed_cpu = torch.randint(-8, 9, (b + 2, s + 2, 3 * h * d + 7)).to(dtype)

    def key_view(packed):
        return _transposed_key(packed[1:-1, 1:-1, h * d : 2 * h * d], h, d)

    key_cpu = key_view(packed_cpu)
    packed = packed_cpu.to("dgx")
    key = key_view(packed)
    scalar_value = 3 if dtype == torch.int64 else 0.25
    if op is torch.remainder:
        scalar_value = -3 if dtype == torch.int64 else -3.5
    if dtype == torch.bool:
        scalar_value = True
    if dtype == torch.complex64:
        scalar_value = 0.25 + 0.5j
    kwargs = {"alpha": -2} if op in (torch.add, torch.sub) else {}
    scalar_cpu = torch.tensor(scalar_value, dtype=dtype)
    scalar = scalar_cpu.to(scalar_device)
    if dtype == torch.complex64:
        # Preserve lazy conjugate bits on both the reference and device scalar.
        scalar_cpu, scalar = scalar_cpu.conj(), scalar.conj()
    expected = op(key_cpu, scalar_cpu, **kwargs)

    result = op(key, scalar, **kwargs)
    assert result.stride() == expected.stride()
    torch.testing.assert_close(result.cpu(), expected)
    if dtype == torch.complex64:
        torch.testing.assert_close(
            (key.conj() * scalar).cpu(), key_cpu.conj() * scalar_cpu
        )

    # Explicit gapped output and in-place operations must leave Q/V and
    # padding untouched, including when a device shard ends within a head.
    output_cpu = torch.full_like(packed_cpu, False if dtype == torch.bool else -99)
    output = output_cpu.to("dgx")
    out = key_view(output)
    strides = out.stride()
    assert op(key, scalar, out=out, **kwargs) is out
    key_view(output_cpu).copy_(expected)
    assert out.stride() == strides
    torch.testing.assert_close(output.cpu(), output_cpu)

    getattr(key, f"{op.__name__}_")(scalar, **kwargs)
    key_view(packed_cpu).copy_(expected)
    torch.testing.assert_close(packed.cpu(), packed_cpu)


@pytest.mark.parametrize(
    "op", [torch.add, torch.sub, torch.mul, torch.div], ids=lambda op: op.__name__
)
@pytest.mark.parametrize("grad_layout", ["contiguous", "zero-stride"])
def test_binary_transposed_key_backward(dgx_setup_teardown, op, grad_layout):
    backing_cpu = torch.randn(3, 41, 3 * 5 * 37, requires_grad=True)
    backing = backing_cpu.detach().to("dgx").requires_grad_()
    kwargs = {"alpha": -2} if op in (torch.add, torch.sub) else {}

    def operation(value):
        key = _transposed_key(value.chunk(3, dim=-1)[1], 5, 37)
        return op(key, 0.25, **kwargs).reshape(15, 37, 41)

    expected = operation(backing_cpu)
    actual = operation(backing)
    if grad_layout == "zero-stride":
        grad_base_cpu = torch.randn(15, 37, 1)
        grad_cpu = grad_base_cpu.expand_as(expected)
        grad = grad_base_cpu.to("dgx").expand_as(actual)
    else:
        grad_cpu = torch.randn_like(expected)
        grad = grad_cpu.to("dgx")
    expected.backward(grad_cpu)
    actual.backward(grad)
    torch.testing.assert_close(actual.cpu(), expected)
    torch.testing.assert_close(backing.grad.cpu(), backing_cpu.grad)


@pytest.mark.parametrize(
    "op",
    [torch.add, torch.sub, torch.mul, torch.div, torch.remainder],
    ids=lambda op: op.__name__,
)
@pytest.mark.parametrize("scalar_device", ["python", "cpu", "dgx"])
def test_binary_transposed_key_strided_channels(dgx_setup_teardown, op, scalar_device):
    backing_cpu = torch.randn(2, 41, 3, 74)
    backing = backing_cpu.to("dgx")
    key_cpu = backing_cpu[..., ::2].permute(0, 2, 3, 1)
    key = backing[..., ::2].permute(0, 2, 3, 1)
    # Exercise channel stride 2 with a contiguous output.
    out = torch.empty(key.shape, device="dgx")
    kwargs = {"alpha": -2} if op in (torch.add, torch.sub) else {}
    scalar = (
        0.25
        if scalar_device == "python"
        else torch.tensor(0.25, dtype=key.dtype, device=scalar_device)
    )
    op(key, scalar, out=out, **kwargs)
    torch.testing.assert_close(out.cpu(), op(key_cpu, 0.25, **kwargs))


@pytest.mark.parametrize("frequency_layout", ["contiguous", "broadcast"])
def test_complex_float_mul_rope(dgx_setup_teardown, frequency_layout):
    query_components = torch.randn(2, 4, 5, 6, 2, dtype=torch.float32)
    query_cpu = torch.view_as_complex(query_components).requires_grad_()
    frequency_shape = (2, 5, 6) if frequency_layout == "broadcast" else query_cpu.shape
    magnitude_cpu = torch.ones(frequency_shape, dtype=torch.float32)
    angle_cpu = torch.randn(frequency_shape, dtype=torch.float32)
    frequencies_cpu = torch.polar(magnitude_cpu, angle_cpu)

    query = query_cpu.detach().to("dgx").requires_grad_()
    frequencies = torch.polar(magnitude_cpu.to("dgx"), angle_cpu.to("dgx"))
    if frequency_layout == "broadcast":
        frequencies_cpu = frequencies_cpu[:, None]
        frequencies = frequencies[:, None]

    reference = query_cpu * frequencies_cpu
    result = query * frequencies
    torch.testing.assert_close(result.cpu(), reference.detach())

    grad_output_cpu = torch.randn_like(reference)
    reference.backward(grad_output_cpu)
    result.backward(grad_output_cpu.to("dgx"))
    torch.testing.assert_close(query.grad.cpu(), query_cpu.grad)


@pytest.mark.parametrize("input_layout", ["contiguous", "zero-stride"])
def test_polar_out_complex_float(dgx_setup_teardown, input_layout):
    if input_layout == "zero-stride":
        magnitude_base_cpu = torch.linspace(0, 2, 5, dtype=torch.float32).reshape(5, 1)
        magnitude_cpu = magnitude_base_cpu.expand(5, 7)
        magnitude = magnitude_base_cpu.to("dgx").expand(5, 7)
        assert 0 in magnitude.stride()
    else:
        magnitude_cpu = torch.linspace(0, 2, 35, dtype=torch.float32).reshape(5, 7)
        magnitude = magnitude_cpu.to("dgx")

    angle_cpu = torch.linspace(-torch.pi, torch.pi, 35, dtype=torch.float32).reshape(
        5, 7
    )
    angle = angle_cpu.to("dgx")
    out = torch.empty(0, dtype=torch.complex64, device="dgx")

    returned = torch.polar(magnitude, angle, out=out)

    assert returned is out
    torch.testing.assert_close(out.cpu(), torch.polar(magnitude_cpu, angle_cpu))


@pytest.mark.parametrize("dtype", datatypes, ids=str)
def test_div(dgx_setup_teardown, dtype):
    a = torch.rand(17, dtype=dtype).to("dgx")
    b = torch.rand(17, dtype=dtype).to("dgx")
    c = torch.div(a, 7)
    ref = torch.div(a.to("cpu"), 7).to("dgx")
    torch.testing.assert_close(c, ref)
    opcheck(torch.ops.aten.div.Scalar, (a, 7))

    c = torch.div(a, b)
    ref = torch.div(a.to("cpu"), b.to("cpu")).to("dgx")
    torch.testing.assert_close(c, ref)
    opcheck(torch.ops.aten.div.Tensor, (a, b))

    c = torch.div(a.to(torch.int32), b)
    ref = torch.div(a.to(torch.int32).to("cpu"), b.to("cpu")).to("dgx")
    torch.testing.assert_close(c, ref)
    opcheck(torch.ops.aten.div.Tensor, (a.to(torch.int32), b))

    numerator = torch.tensor(
        [[1.0, 0.0, 1.0], [0.0, 1.0, 1.0]], dtype=dtype, device="dgx"
    )
    denominator = torch.tensor([[2], [3]], dtype=torch.int64, device="dgx")
    actual = torch.div(numerator, denominator)
    expected = torch.div(numerator.cpu(), denominator.cpu())
    torch.testing.assert_close(actual.cpu(), expected)


@pytest.mark.parametrize(
    "dtype", [torch.float32, torch.float64, torch.int32, torch.int64], ids=str
)
def test_remainder_tensor_out(dgx_setup_teardown, dtype):
    dividend_cpu = torch.tensor([[-7, -6, -1, 0, 1, 6, 7]], dtype=dtype)
    divisor_cpu = torch.tensor([[3], [-3]], dtype=dtype)
    dividend = dividend_cpu.to("dgx")
    divisor = divisor_cpu.to("dgx")

    actual = torch.remainder(dividend, divisor)
    expected = torch.remainder(dividend_cpu, divisor_cpu)
    torch.testing.assert_close(actual.cpu(), expected)
    opcheck(torch.ops.aten.remainder.Tensor, (dividend, divisor))

    out = torch.empty((7, 2), dtype=dtype, device="dgx").T
    returned = torch.remainder(dividend, divisor, out=out)
    assert returned is out
    assert not out.is_contiguous()
    torch.testing.assert_close(out.cpu(), expected)


@pytest.mark.parametrize("dtype", [torch.bool], ids=str)
def test_logical_and(dgx_setup_teardown, dtype):
    a = (torch.normal(0, 1, (21,), dtype=torch.float32) > 0.1).to(dtype=dtype).to("dgx")
    b = (torch.normal(0, 1, (21,), dtype=torch.float32) > 0.1).to(dtype=dtype).to("dgx")
    c = torch.logical_and(a, b)
    ref = torch.logical_and(a.to("cpu"), b.to("cpu")).to("dgx")
    assert torch.equal(c, ref)
    opcheck(torch.ops.aten.logical_and.default, (a, b))


@pytest.mark.parametrize("dtype", [torch.bool], ids=str)
def test_logical_or(dgx_setup_teardown, dtype):
    a = (torch.normal(0, 1, (21,), dtype=torch.float32) > 0.1).to(dtype=dtype).to("dgx")
    b = (torch.normal(0, 1, (21,), dtype=torch.float32) > 0.1).to(dtype=dtype).to("dgx")
    c = torch.logical_or(a, b)
    ref = torch.logical_or(a.to("cpu"), b.to("cpu")).to("dgx")
    assert torch.equal(c, ref)
    opcheck(torch.ops.aten.logical_or.default, (a, b))


@pytest.mark.parametrize("dtype", datatypes, ids=str)
@pytest.mark.parametrize("shape", [(2, 11), (2, 3, 4, 5, 6)], ids=["2d", "5d"])
def test_pow(dgx_setup_teardown, dtype, shape):
    a = torch.rand(shape, dtype=dtype).to("dgx")
    c = torch.pow(a, 1.5)
    ref = torch.pow(a.to("cpu"), 1.5).to("dgx")
    torch.testing.assert_close(c, ref)
    opcheck(torch.ops.aten.pow.Tensor_Scalar, (a, 1.5))
