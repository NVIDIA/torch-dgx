# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
import pytest
import torch
from torch.library import opcheck

datatypes = [torch.float32, torch.float64]


@pytest.mark.parametrize("dtype", datatypes, ids=str)
def test_elementwise_equal(dgx_setup_teardown, dtype):
    a = torch.rand(2, 11, dtype=dtype).to("dgx")
    b = torch.rand(2, 11, dtype=dtype).to("dgx")
    c = torch.eq(a, b)
    ref = torch.eq(a.to("cpu"), b.to("cpu"))
    assert torch.equal(c.to("cpu"), ref)
    opcheck(torch.ops.aten.eq.Tensor, (a, b))

    c = torch.eq(a, 0.47)
    ref = torch.eq(a.to("cpu"), 0.47)
    assert torch.equal(c.to("cpu"), ref)
    opcheck(torch.ops.aten.eq.Scalar, (a, 0.47))


@pytest.mark.parametrize("dtype", datatypes, ids=str)
def test_not_equal(dgx_setup_teardown, dtype):
    a = torch.rand(2, 11, dtype=dtype).to("dgx")
    b = torch.rand(2, 11, dtype=dtype).to("dgx")
    c = torch.ne(a, b)
    ref = torch.ne(a.to("cpu"), b.to("cpu"))
    assert torch.equal(c.to("cpu"), ref)
    opcheck(torch.ops.aten.ne.Tensor, (a, b))

    c = torch.ne(a, 0.47)
    ref = torch.ne(a.to("cpu"), 0.47)
    assert torch.equal(c.to("cpu"), ref)
    opcheck(torch.ops.aten.ne.Scalar, (a, 0.47))


@pytest.mark.parametrize("dtype", datatypes, ids=str)
def test_gt(dgx_setup_teardown, dtype):
    a = torch.normal(0, 1, (52,), dtype=dtype).to("dgx")
    b = torch.normal(0, 1, (52,), dtype=dtype).to("dgx")
    c = torch.gt(a, b)
    ref = torch.gt(a.to("cpu"), b.to("cpu"))
    assert torch.equal(c.to("cpu"), ref)
    opcheck(torch.ops.aten.gt.Tensor, (a, b))

    c = torch.gt(a, 0)
    ref = torch.gt(a.to("cpu"), 0)
    assert torch.equal(c.to("cpu"), ref)
    opcheck(torch.ops.aten.gt.Scalar, (a, 0))


@pytest.mark.parametrize("dtype", datatypes, ids=str)
def test_lt(dgx_setup_teardown, dtype):
    a = torch.normal(0, 1, (52,), dtype=dtype).to("dgx")
    b = torch.normal(0, 1, (52,), dtype=dtype).to("dgx")
    c = torch.lt(a, b)
    ref = torch.lt(a.to("cpu"), b.to("cpu"))
    assert torch.equal(c.to("cpu"), ref)
    opcheck(torch.ops.aten.lt.Tensor, (a, b))

    c = torch.lt(a, 0)
    ref = torch.lt(a.to("cpu"), 0)
    assert torch.equal(c.to("cpu"), ref)
    opcheck(torch.ops.aten.lt.Scalar, (a, 0))


@pytest.mark.parametrize("dtype", datatypes, ids=str)
def test_ge(dgx_setup_teardown, dtype):
    a = torch.normal(0, 1, (52,), dtype=dtype).to("dgx")
    b = torch.normal(0, 1, (52,), dtype=dtype).to("dgx")
    c = torch.ge(a, b)
    ref = torch.ge(a.to("cpu"), b.to("cpu"))
    assert torch.equal(c.to("cpu"), ref)
    opcheck(torch.ops.aten.ge.Tensor, (a, b))

    c = torch.ge(a, 0)
    ref = torch.ge(a.to("cpu"), 0)
    assert torch.equal(c.to("cpu"), ref)
    opcheck(torch.ops.aten.ge.Scalar, (a, 0))


@pytest.mark.parametrize("dtype", datatypes, ids=str)
def test_le(dgx_setup_teardown, dtype):
    a = torch.normal(0, 1, (52,), dtype=dtype).to("dgx")
    b = torch.normal(0, 1, (52,), dtype=dtype).to("dgx")
    c = torch.le(a, b)
    ref = torch.le(a.to("cpu"), b.to("cpu"))
    assert torch.equal(c.to("cpu"), ref)
    opcheck(torch.ops.aten.le.Tensor, (a, b))

    c = torch.le(a, 0)
    ref = torch.le(a.to("cpu"), 0)
    assert torch.equal(c.to("cpu"), ref)
    opcheck(torch.ops.aten.le.Scalar, (a, 0))


@pytest.mark.parametrize("dtype", datatypes, ids=str)
def test_where(dgx_setup_teardown, dtype):
    condition = (torch.normal(0, 1, (3, 5), dtype=dtype) > 0.5).to("dgx")
    a = torch.normal(0, 1, (3, 5), dtype=dtype).to("dgx")
    b = torch.rand(3, 5, dtype=dtype).to("dgx")

    c = torch.where(condition, a, b)
    ref = torch.where(condition.to("cpu"), a.to("cpu"), b.to("cpu"))
    assert torch.equal(c.to("cpu"), ref)
    opcheck(torch.ops.aten.where.self, (condition, a, b))

    out = torch.empty(0, dtype=dtype, device="dgx")
    returned = torch.where(condition, a, b, out=out)
    assert returned is out
    assert torch.equal(out.to("cpu"), ref)

    backing = torch.full((3, 10), -1, dtype=dtype, device="dgx")
    strided_out = backing[:, ::2]
    torch.where(condition, a, b, out=strided_out)
    assert torch.equal(strided_out.cpu(), ref)
    assert torch.equal(backing[:, 1::2].cpu(), torch.full((3, 5), -1, dtype=dtype))

    b = torch.tensor(0.3).to("dgx").to(dtype)

    c = torch.where(condition, a, b)
    ref = torch.where(condition.to("cpu"), a.to("cpu"), b.to("cpu"))
    assert torch.equal(c.to("cpu"), ref)
    opcheck(torch.ops.aten.where.self, (condition, a, b))


@pytest.mark.parametrize("dtype", datatypes, ids=str)
def test_isneginf_out(dgx_setup_teardown, dtype):
    tensor = torch.tensor(
        [float("-inf"), -1.0, 0.0, float("inf"), float("nan")],
        dtype=dtype,
        device="dgx",
    )
    out = torch.empty(0, dtype=torch.bool, device="dgx")

    returned = torch.isneginf(tensor, out=out)

    assert returned is out
    assert torch.equal(out.cpu(), torch.isneginf(tensor.cpu()))


@pytest.mark.parametrize(
    ("features_dtype", "background_dtype"),
    [(torch.float32, torch.float32), (torch.float32, torch.float64)],
    ids=lambda dtype: str(dtype),
)
def test_where_broadcast_rank4(dgx_setup_teardown, features_dtype, background_dtype):
    condition = torch.zeros((1, 16, 16, 1), dtype=torch.bool).to("dgx")
    features = torch.randn((1, 16, 16, 3), dtype=features_dtype).to("dgx")
    background_cpu = torch.tensor([[[[0.1, -0.2, 0.3]]]], dtype=background_dtype)
    background = background_cpu.to("dgx")

    result = torch.where(condition, features, background)
    expected = background_cpu.expand(1, 16, 16, 3)

    assert result.shape == (1, 16, 16, 3)
    assert result.dtype == expected.dtype
    assert torch.equal(result.to("cpu"), expected)


@pytest.mark.parametrize(
    ("condition_shape", "output_shape"),
    [
        ((1, 1, 4, 5), (1, 3, 4, 5)),
        ((2, 1, 4, 5), (2, 3, 4, 5)),
        ((2, 3, 4, 1), (2, 3, 4, 5)),
    ],
)
def test_where_single_broadcast_dimension(
    dgx_setup_teardown, condition_shape, output_shape
):
    condition_cpu = torch.rand(condition_shape) > 0.5
    self_cpu = torch.randn(output_shape)
    other_cpu = torch.randn(output_shape)

    result = torch.where(
        condition_cpu.to("dgx"), self_cpu.to("dgx"), other_cpu.to("dgx")
    )
    expected = torch.where(condition_cpu, self_cpu, other_cpu)

    assert torch.equal(result.to("cpu"), expected)


@pytest.mark.parametrize(
    "dtype", datatypes + [torch.uint16, torch.uint32, torch.uint64], ids=str
)
def test_equal(dgx_setup_teardown, dtype):
    a = torch.ones((21,), dtype=dtype).to("dgx")
    b = torch.zeros((21,), dtype=dtype).to("dgx")
    c = torch.zeros((42,), dtype=dtype).to("dgx")

    # opcheck fails on the equal operator due to a PyTorch bug
    # opcheck(torch.ops.aten.equal.default, (a, a))
    # opcheck(torch.ops.aten.equal.default, (a, b))

    assert torch.equal(a, a)
    assert not torch.equal(a, b)
    assert not torch.equal(a, c)


@pytest.mark.parametrize(
    "dtype", datatypes + [torch.uint16, torch.uint32, torch.uint64], ids=str
)
def test_equal_expanded_tensor(dgx_setup_teardown, dtype):
    base = torch.arange(6).to(dtype).reshape(2, 1, 1, 3)
    contiguous = base.expand(2, 4, 5, 3).contiguous().to("dgx")
    expanded = base.to("dgx").expand(2, 4, 5, 3)

    assert expanded.stride() == (3, 0, 0, 1)
    assert torch.equal(contiguous, expanded)
    assert torch.equal(expanded, contiguous)


@pytest.mark.parametrize("dtype", datatypes, ids=str)
def test_allclose(dgx_setup_teardown, dtype):
    actual = torch.tensor(
        [1.0, 2.0, float("inf"), -float("inf"), float("nan")],
        dtype=dtype,
        device="dgx",
    )
    close = torch.tensor(
        [1.0 + 1e-6, 2.0 - 1e-6, float("inf"), -float("inf"), float("nan")],
        dtype=dtype,
        device="dgx",
    )
    far = close.clone()
    far[1] = 2.1

    assert torch.allclose(actual, close, equal_nan=True)
    assert not torch.allclose(actual, close)
    assert not torch.allclose(actual, far, equal_nan=True)


@pytest.mark.parametrize("dtype", datatypes, ids=str)
def test_elementwise_equal_noncontiguous_rank4(dgx_setup_teardown, dtype):
    base = torch.rand((1, 16, 16, 4), dtype=dtype).to("dgx")
    actual = base[..., :-1]
    expected = actual.contiguous()

    assert actual.stride() == (1024, 64, 4, 1)
    assert not actual.is_contiguous()
    assert torch.eq(actual, expected).all()
    torch.testing.assert_close(actual, expected)

    out_storage = torch.empty((1, 16, 16, 4), dtype=torch.bool).to("dgx")
    out = out_storage[..., :-1]
    torch.eq(actual, expected, out=out)
    assert out.all().item()

    scalar_expected = torch.eq(actual.cpu(), 0.5)
    torch.eq(actual, 0.5, out=out)
    assert torch.equal(out.cpu(), scalar_expected)
