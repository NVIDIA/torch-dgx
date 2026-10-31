# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
import torch
import pytest
from torch.library import opcheck

datatypes = [torch.float32, torch.float64]

# tril and triu share a kernel and differ only in which side of the diagonal is kept.
triangles = pytest.mark.parametrize("op", ["tril", "triu"])


@triangles
@pytest.mark.parametrize(
    "dtype",
    [
        torch.bool,
        torch.uint8,
        torch.int8,
        torch.int16,
        torch.int32,
        torch.int64,
        torch.float16,
        torch.bfloat16,
        torch.float32,
        torch.float64,
        torch.complex64,
    ],
    ids=str,
)
def test_triangle_out(dgx_setup_teardown, dtype, op):
    reference = getattr(torch, op)
    for shape in [(5, 7), (7, 5), (2, 1, 3, 5, 7), (0, 3), (3, 0), (2, 0, 4, 5)]:
        sample = torch.randint(0, 10, shape).to(dtype)
        if dtype.is_complex:
            sample += 2j * sample
        input = sample.to("dgx")
        for diagonal in [-(2**63), -9, -2, 0, 3, 9, 2**63 - 1]:
            expected = reference(sample, diagonal)
            out = torch.empty(0, dtype=dtype, device="dgx")
            actual = getattr(torch.ops.aten, op).out(input, diagonal, out=out)
            assert actual is out
            torch.testing.assert_close(out.cpu(), expected, rtol=0, atol=0)


@triangles
@pytest.mark.parametrize("layout", ["transpose", "slice", "permute", "expand"])
@pytest.mark.parametrize("diagonal", [-2, 0, 3])
def test_triangle_strided(dgx_setup_teardown, layout, diagonal, op):
    reference = getattr(torch, op)
    def view(tensor):
        if layout == "transpose":
            return tensor.transpose(-1, -2)
        if layout == "slice":
            return tensor[..., 1::2, 1::2]
        if layout == "permute":
            return tensor.permute(2, 0, 3, 1)
        return tensor[:1, :1].expand(3, 4, 7, 10)

    base = torch.randn(2, 3, 7, 10)
    sample = view(base)
    input = view(base.to("dgx"))
    expected = reference(sample, diagonal)
    torch.testing.assert_close(reference(input, diagonal).cpu(), expected)

    # The output has different strides and a storage offset, with untouched padding.
    shape = sample.shape
    output_base = torch.full((*shape[:-2], shape[-1] + 2, shape[-2] + 2), -123.0)
    expected_base = output_base.clone()
    output_base = output_base.to("dgx")
    out = output_base[..., 1:-1, 1:-1].transpose(-1, -2)
    expected_base[..., 1:-1, 1:-1].transpose(-1, -2).copy_(expected)
    strides = out.stride()
    actual = reference(input, diagonal, out=out)
    assert actual is out
    assert out.stride() == strides
    torch.testing.assert_close(output_base.cpu(), expected_base)


@triangles
@pytest.mark.parametrize("inplace", [False, True])
@pytest.mark.parametrize("strided", [False, True])
def test_triangle_alias(dgx_setup_teardown, inplace, strided, op):
    reference = getattr(torch, op)
    for diagonal in [-2, 0, 3]:
        sample = torch.randn(2, 5, 7)
        input = sample.to("dgx")
        if strided:
            sample = sample.transpose(-1, -2)
            input = input.transpose(-1, -2)
        expected = reference(sample, diagonal)
        if inplace:
            actual = getattr(input, f"{op}_")(diagonal)
        else:
            actual = reference(input, diagonal, out=input)
        assert actual is input
        torch.testing.assert_close(input.cpu(), expected)


@triangles
def test_triangle_sharding(dgx_setup_teardown, op):
    reference = getattr(torch, op)
    sample = torch.randn(3, 1025, 513)
    input = sample.to("dgx")
    for diagonal in [-513, 0, 257]:
        actual = reference(input, diagonal)
        torch.testing.assert_close(actual.cpu(), reference(sample, diagonal))


@triangles
def test_triangle_nonfinite(dgx_setup_teardown, op):
    reference = getattr(torch, op)
    sample = torch.tensor(
        [[float("nan"), float("inf"), -float("inf")], [1.0, -0.0, float("nan")]]
    )
    actual = reference(sample.to("dgx"))
    torch.testing.assert_close(
        actual.cpu(), reference(sample), rtol=0, atol=0, equal_nan=True
    )


@triangles
def test_triangle_out_validation(dgx_setup_teardown, op):
    reference = getattr(torch, op)
    for shape in [(), (3,)]:
        input = torch.empty(shape, device="dgx")
        with pytest.raises(RuntimeError, match="at least 2 dimensions"):
            reference(input, out=torch.empty_like(input))

    input = torch.empty(3, 4, device="dgx")
    with pytest.raises(RuntimeError, match="dtype"):
        reference(input, out=torch.empty(0, dtype=torch.float64, device="dgx"))
    with pytest.raises(RuntimeError, match="same device"):
        reference(input, out=torch.empty(3, 4))


@pytest.mark.parametrize("dtype", datatypes, ids=str)
def test_linalg_vector_norm(dgx_setup_teardown, dtype):
    sample_inputs = [
        (torch.rand((2, 3), dtype=dtype, device="dgx"),),
        (torch.rand((1, 3), dtype=dtype, device="dgx"),),
        (torch.rand((5, 3, 2), dtype=dtype, device="dgx"),),
        (torch.rand((5, 3, 2), dtype=dtype, device="dgx") * -1.0,),
    ]

    for args in sample_inputs:
        kwargs = {"dim": -1, "keepdim": True}
        opcheck(torch.ops.aten.linalg_vector_norm.default, args, kwargs)
        out = torch.linalg.vector_norm(*args, **kwargs)
        ref = torch.linalg.vector_norm(*(arg.to("cpu") for arg in args), **kwargs)
        torch.testing.assert_close(out, ref.to("dgx"))

    for args in sample_inputs:
        kwargs = {"ord": 1.5, "dim": -1, "keepdim": True}
        opcheck(torch.ops.aten.linalg_vector_norm.default, args, kwargs)
        out = torch.linalg.vector_norm(*args, **kwargs)
        ref = torch.linalg.vector_norm(*(arg.to("cpu") for arg in args), **kwargs)
        torch.testing.assert_close(out, ref.to("dgx"))

    for args in sample_inputs:
        kwargs = {"dim": -1, "keepdim": False}
        opcheck(torch.ops.aten.linalg_vector_norm.default, args, kwargs)
        out = torch.linalg.vector_norm(*args, **kwargs)
        ref = torch.linalg.vector_norm(*(arg.to("cpu") for arg in args), **kwargs)
        torch.testing.assert_close(out, ref.to("dgx"))

    for args in sample_inputs:
        kwargs = {"ord": 1.5, "dim": -1, "keepdim": False}
        opcheck(torch.ops.aten.linalg_vector_norm.default, args, kwargs)
        out = torch.linalg.vector_norm(*args, **kwargs)
        ref = torch.linalg.vector_norm(*(arg.to("cpu") for arg in args), **kwargs)
        torch.testing.assert_close(out, ref.to("dgx"))


@pytest.mark.parametrize("keepdim", [False, True], ids=["dropdim", "keepdim"])
def test_linalg_vector_norm_last_dim_arbitrary_rank(dgx_setup_teardown, keepdim):
    samples = [
        torch.randn(17),
        torch.randn(2, 3, 33),
        torch.randn(2, 1, 3, 5, 257),
        torch.randn(2, 65, 3, 4).permute(0, 2, 3, 1),
    ]
    for sample in samples:
        for ord in (0.0, 1.5, 2.0, -1.0, float("inf"), -float("inf")):
            expected = torch.linalg.vector_norm(
                sample, ord=ord, dim=-1, keepdim=keepdim
            )
            actual = torch.linalg.vector_norm(
                sample.to("dgx"), ord=ord, dim=-1, keepdim=keepdim
            )
            torch.testing.assert_close(actual.cpu(), expected)


def test_linalg_vector_norm_row_reduction_boundaries(dgx_setup_teardown):
    widths = [
        0,
        1,
        2,
        4,
        5,
        8,
        9,
        16,
        17,
        32,
        33,
        64,
        65,
        128,
        129,
        256,
        257,
        512,
        513,
        1024,
        1025,
        2048,
        2049,
        4096,
    ]
    for width in widths:
        samples = torch.randn(3, width)
        expected = torch.linalg.vector_norm(samples, dim=-1)
        actual = torch.linalg.vector_norm(samples.to("dgx"), dim=-1)
        torch.testing.assert_close(actual.cpu(), expected)


@pytest.mark.parametrize("ord", [-1.0, float("inf"), -float("inf")], ids=str)
def test_linalg_vector_norm_empty_last_dim_errors(dgx_setup_teardown, ord):
    samples = torch.empty(3, 0, device="dgx")
    with pytest.raises(RuntimeError, match="does not have an identity"):
        torch.linalg.vector_norm(samples, ord=ord, dim=-1)


@pytest.mark.parametrize("dtype", datatypes, ids=str)
@pytest.mark.parametrize("ord", [0.0, 1.5, 2.0, float("inf"), -float("inf")], ids=str)
def test_linalg_vector_norm_all_dimensions(dgx_setup_teardown, dtype, ord):
    sample_inputs = [
        torch.rand((7,), dtype=dtype, device="dgx"),
        torch.rand((2, 3), dtype=dtype, device="dgx"),
        torch.rand((2, 3, 4), dtype=dtype, device="dgx"),
        torch.rand((3, 2), dtype=dtype, device="dgx").t(),
    ]

    for input in sample_inputs:
        kwargs = {"ord": ord}
        opcheck(torch.ops.aten.linalg_vector_norm.default, (input,), kwargs)
        out = torch.linalg.vector_norm(input, **kwargs)
        ref = torch.linalg.vector_norm(input.to("cpu"), **kwargs)
        torch.testing.assert_close(out, ref.to("dgx"))


@pytest.mark.parametrize("dtype", datatypes, ids=str)
def test_foreach_norm(dgx_setup_teardown, dtype):
    inputs = [
        torch.rand((7,), dtype=dtype, device="dgx"),
        torch.rand((2, 3), dtype=dtype, device="dgx"),
        torch.rand((2, 3, 4), dtype=dtype, device="dgx") * -1.0,
    ]

    out = torch._foreach_norm(inputs, 2.0)
    ref = [torch.linalg.vector_norm(input.to("cpu"), 2.0) for input in inputs]

    assert len(out) == len(ref)
    for actual, expected in zip(out, ref):
        torch.testing.assert_close(actual, expected.to("dgx"))


@pytest.mark.parametrize("dtype", datatypes, ids=str)
def test_linalg_cross(dgx_setup_teardown, dtype):
    sample_inputs = [
        (
            torch.rand((2, 3), dtype=dtype, device="dgx"),
            torch.rand((2, 3), dtype=dtype, device="dgx"),
        ),
        (
            torch.rand((1, 3), dtype=dtype, device="dgx"),
            torch.rand((1, 3), dtype=dtype, device="dgx"),
        ),
    ]

    for args in sample_inputs:
        opcheck(torch.ops.aten.linalg_cross.default, args)
        out = torch.linalg.cross(*args)
        ref = torch.linalg.cross(*(arg.to("cpu") for arg in args))
        torch.testing.assert_close(out, ref.to("dgx"))
