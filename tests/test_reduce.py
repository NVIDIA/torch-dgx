# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
import torch
import pytest


def test_sum_dim0_dtype_conversion(dgx_setup_teardown):
    samples = torch.randn(3, 65)
    dgx_samples = samples.to("dgx")

    actual = dgx_samples.sum(dim=0, dtype=torch.float64)

    torch.testing.assert_close(actual.cpu(), samples.sum(dim=0, dtype=torch.float64))


@pytest.mark.parametrize("cameras,count", [(1, 65), (3, 1), (3, 65)])
@pytest.mark.parametrize("dim", [0, -2])
def test_sum_dim0_out_dtype(dgx_setup_teardown, cameras, count, dim):
    samples = torch.randn(cameras, count)
    dgx_samples = samples.to("dgx")
    expected = torch.empty(count, dtype=torch.float64)
    output = torch.empty(count, dtype=torch.float64, device="dgx")

    torch.sum(samples, dim=dim, out=expected)
    actual = torch.sum(dgx_samples, dim=dim, out=output)

    assert actual is output
    torch.testing.assert_close(output.cpu(), expected)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64], ids=str)
def test_min(dgx_setup_teardown, dtype):
    samples = torch.rand(73, dtype=dtype)
    result = torch.min(samples.to("dgx"))
    torch.testing.assert_close(result.to("cpu"), torch.min(samples))


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64], ids=str)
def test_max(dgx_setup_teardown, dtype):
    samples = torch.rand(73, dtype=dtype)
    result = torch.max(samples.to("dgx"))
    torch.testing.assert_close(result.to("cpu"), torch.max(samples))


@pytest.mark.parametrize("op", ["any", "all", "equal"])
def test_empty_scalar_reduction(dgx_setup_teardown, op):
    samples = torch.empty(0, device="dgx")
    if op == "equal":
        assert torch.equal(samples, samples)
    else:
        reduction = getattr(torch, op)
        assert reduction(samples).item() == reduction(torch.empty(0)).item()


@pytest.mark.parametrize("op", [torch.min, torch.max], ids=["min", "max"])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64], ids=str)
def test_minmax_out(dgx_setup_teardown, dtype, op):
    samples = torch.randn(45, 3, dtype=dtype)
    for dim in (-1, 0):
        for keepdim in (False, True):
            result = op(samples.to("dgx"), dim, keepdim)
            ref = op(samples, dim, keepdim)
            torch.testing.assert_close(result.values.to("cpu"), ref.values)
            assert torch.equal(result.indices.to("cpu"), ref.indices)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64], ids=str)
def test_minmax_dim0_tall(dgx_setup_teardown, dtype):
    # Tall, narrow tensor (rows >> cols): the dim-0 argmin/argmax path.
    samples = torch.randn(5175801, 3, dtype=dtype)
    for op in (torch.min, torch.max):
        result = op(samples.to("dgx"), 0)
        ref = op(samples, 0)
        torch.testing.assert_close(result.values.to("cpu"), ref.values)
        assert torch.equal(result.indices.to("cpu"), ref.indices)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64], ids=str)
def test_minmax_dim0_wide(dgx_setup_teardown, dtype):
    # Wide tensor (cols > kDim0ArgReduceMaxCols): the column-parallel fallback path.
    samples = torch.randn(45, 4096, dtype=dtype)
    for op in (torch.min, torch.max):
        for keepdim in (False, True):
            result = op(samples.to("dgx"), 0, keepdim)
            ref = op(samples, 0, keepdim)
            torch.testing.assert_close(result.values.to("cpu"), ref.values)
            assert torch.equal(result.indices.to("cpu"), ref.indices)


@pytest.mark.parametrize(
    "dtype", [torch.float32, torch.float64, torch.int32, torch.int64], ids=str
)
def test_minmax_dim_extreme_values(dgx_setup_teardown, dtype):
    info = torch.finfo(dtype) if dtype.is_floating_point else torch.iinfo(dtype)
    for op, value in ((torch.min, info.max), (torch.max, info.min)):
        samples = torch.full((3, 65), value, dtype=dtype)
        for dim in (0, 1):
            for keepdim in (False, True):
                result = op(samples.to("dgx"), dim, keepdim)
                ref = op(samples, dim, keepdim)
                assert torch.equal(result.values.to("cpu"), ref.values)
                assert torch.equal(result.indices.to("cpu"), ref.indices)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64], ids=str)
def test_minmax_dim_infinities(dgx_setup_teardown, dtype):
    for op, value in ((torch.min, float("inf")), (torch.max, float("-inf"))):
        samples = torch.full((3, 65), value, dtype=dtype)
        for dim in (0, 1):
            for keepdim in (False, True):
                result = op(samples.to("dgx"), dim, keepdim)
                ref = op(samples, dim, keepdim)
                assert torch.equal(result.values.to("cpu"), ref.values)
                assert torch.equal(result.indices.to("cpu"), ref.indices)


@pytest.mark.parametrize(
    "shape,dim",
    [((1024, 3), 0), ((2, 3, 65), -1)],
    ids=["dim0", "last-dim"],
)
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64], ids=str)
def test_minmax_strided_out(dgx_setup_teardown, dtype, shape, dim):
    samples = torch.randn(shape, dtype=dtype)
    dgx_samples = samples.to("dgx")
    for op in (torch.min, torch.max):
        expected = op(samples, dim)
        storage_shape = (
            *expected.values.shape[:-1],
            expected.values.shape[-1] * 2,
        )
        values = torch.empty(storage_shape, dtype=dtype, device="dgx")[..., ::2]
        indices = torch.empty(storage_shape, dtype=torch.int64, device="dgx")[..., ::2]

        op(dgx_samples, dim, out=(values, indices))

        assert not values.is_contiguous()
        assert not indices.is_contiguous()
        torch.testing.assert_close(values.cpu(), expected.values)
        torch.testing.assert_close(indices.cpu(), expected.indices)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64], ids=str)
def test_minmax_dim0_transposed(dgx_setup_teardown, dtype):
    # Non-contiguous (transposed) input must respect strides, not assume row-major.
    base = torch.randn(3, 4096, dtype=dtype)
    samples = base.t()  # shape [4096, 3], non-contiguous
    for op in (torch.min, torch.max):
        result = op(samples.to("dgx"), 0)
        ref = op(samples, 0)
        torch.testing.assert_close(result.values.to("cpu"), ref.values)
        assert torch.equal(result.indices.to("cpu"), ref.indices)


@pytest.mark.parametrize("keepdim", [False, True], ids=["dropdim", "keepdim"])
def test_minmax_last_dim_arbitrary_rank(dgx_setup_teardown, keepdim):
    samples = [
        torch.randn(17),
        torch.randn(2, 3, 33),
        torch.randn(2, 1, 3, 5, 257),
        torch.randn(2, 65, 3, 4).permute(0, 2, 3, 1),
    ]
    for sample in samples:
        for op in (torch.min, torch.max):
            expected = op(sample, dim=-1, keepdim=keepdim)
            actual = op(sample.to("dgx"), dim=-1, keepdim=keepdim)
            torch.testing.assert_close(actual.values.cpu(), expected.values)
            torch.testing.assert_close(actual.indices.cpu(), expected.indices)


def test_minmax_last_dim_row_reduction_boundaries(dgx_setup_teardown):
    widths = [
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
        for op in (torch.min, torch.max):
            expected = op(samples, dim=-1)
            actual = op(samples.to("dgx"), dim=-1)
            torch.testing.assert_close(actual.values.cpu(), expected.values)
            torch.testing.assert_close(actual.indices.cpu(), expected.indices)


def test_minmax_last_dim_nan_tie_and_signed_zero_semantics(dgx_setup_teardown):
    samples = torch.zeros(3, 65)
    samples[0, 3] = -7
    samples[0, 19] = -7
    samples[0, 5] = 9
    samples[0, 37] = 9
    samples[1, 7] = float("nan")
    samples[1, 41] = float("nan")
    samples[2, 0] = -0.0
    samples[2, 1] = 0.0

    for op in (torch.min, torch.max):
        expected = op(samples, dim=-1)
        actual = op(samples.to("dgx"), dim=-1)
        values = actual.values.cpu()
        torch.testing.assert_close(values, expected.values, equal_nan=True)
        assert torch.equal(torch.signbit(values), torch.signbit(expected.values))
        assert torch.equal(actual.indices.cpu(), expected.indices)


argmax_datatypes = [
    torch.float32,
    torch.float64,
    torch.float16,
    torch.bfloat16,
    torch.int32,
    torch.int64,
]

# argmax and argmin share a kernel and differ only in the comparison.
extrema = pytest.mark.parametrize("op", ["argmax", "argmin"])


@extrema
@pytest.mark.parametrize("dtype", argmax_datatypes, ids=str)
@pytest.mark.parametrize("keepdim", [False, True], ids=["dropdim", "keepdim"])
def test_arg_reduce_dims(dgx_setup_teardown, dtype, keepdim, op):
    reference = getattr(torch, op)
    # Reductions over a trailing dim are contiguous; the others exercise the strided path.
    for shape, dim in [
        ((17,), 0),
        ((17,), -1),
        ((5, 65), -1),
        ((5, 65), 0),
        ((2, 3, 33), 1),
        ((2, 1, 3, 5, 257), -1),
    ]:
        if dtype.is_floating_point:
            samples = torch.randn(shape).to(dtype)
        else:
            samples = torch.randint(-1000, 1000, shape, dtype=dtype)
        expected = reference(samples, dim=dim, keepdim=keepdim)
        actual = reference(samples.to("dgx"), dim=dim, keepdim=keepdim)
        assert actual.shape == expected.shape
        assert torch.equal(actual.cpu(), expected)


@extrema
@pytest.mark.parametrize("keepdim", [False, True], ids=["dropdim", "keepdim"])
def test_arg_reduce_flattened(dgx_setup_teardown, keepdim, op):
    reference = getattr(torch, op)
    # Without a dim the reduction covers the flattened input, and keepdim leaves all-singleton dims.
    for shape in [(), (17,), (3, 65), (2, 3, 4, 9)]:
        samples = torch.randn(shape)
        expected = reference(samples, keepdim=keepdim)
        actual = reference(samples.to("dgx"), keepdim=keepdim)
        assert actual.shape == expected.shape
        assert torch.equal(actual.cpu(), expected)


@extrema
def test_arg_reduce_flattened_strided(dgx_setup_teardown, op):
    reference = getattr(torch, op)
    # A non-contiguous input has to be flattened by value, not by walking the storage.
    storage = torch.randn(4, 65)
    samples = storage.t()
    dgx_samples = storage.to("dgx").t()
    assert not dgx_samples.is_contiguous()

    actual = reference(dgx_samples)
    assert torch.equal(actual.cpu(), reference(samples))


@extrema
def test_arg_reduce_ties_and_nan(dgx_setup_teardown, op):
    reference = getattr(torch, op)
    samples = torch.zeros(3, 65)
    samples[0, 5] = 9
    samples[0, 37] = 9  # Ties resolve to the lowest index.
    samples[1, 7] = float("nan")
    samples[1, 41] = float("nan")  # NaN outranks every value.
    samples[2, 0] = -0.0
    samples[2, 1] = 0.0

    expected = reference(samples, dim=-1)
    actual = reference(samples.to("dgx"), dim=-1)
    assert torch.equal(actual.cpu(), expected)


@extrema
@pytest.mark.parametrize("dtype", [torch.float32, torch.int64], ids=str)
def test_arg_reduce_extreme_values(dgx_setup_teardown, dtype, op):
    reference = getattr(torch, op)
    info = torch.finfo(dtype) if dtype.is_floating_point else torch.iinfo(dtype)
    # Fill with the value the reduction's identity holds, so the identity must lose every comparison.
    samples = torch.full((3, 65), info.min if op == "argmax" else info.max, dtype=dtype)
    for dim in (0, 1):
        expected = reference(samples, dim=dim)
        actual = reference(samples.to("dgx"), dim=dim)
        assert torch.equal(actual.cpu(), expected)


@extrema
def test_arg_reduce_row_reduction_boundaries(dgx_setup_teardown, op):
    reference = getattr(torch, op)
    for width in [1, 2, 4, 5, 8, 9, 16, 17, 32, 33, 64, 65, 128, 129, 1024, 1025, 4096]:
        samples = torch.randn(3, width)
        expected = reference(samples, dim=-1)
        actual = reference(samples.to("dgx"), dim=-1)
        assert torch.equal(actual.cpu(), expected)


@extrema
def test_arg_reduce_sharding(dgx_setup_teardown, op):
    reference = getattr(torch, op)
    samples = torch.randn(1025, 513)
    expected = reference(samples, dim=-1)
    actual = reference(samples.to("dgx"), dim=-1)
    assert torch.equal(actual.cpu(), expected)


@extrema
@pytest.mark.parametrize("keepdim", [False, True], ids=["dropdim", "keepdim"])
def test_arg_reduce_strided_input_and_output(dgx_setup_teardown, keepdim, op):
    reference = getattr(torch, op)
    storage = torch.randn(2, 4, 3, 5, 7)
    samples = storage.permute(0, 2, 3, 4, 1)
    dgx_samples = storage.to("dgx").permute(0, 2, 3, 4, 1)
    assert not dgx_samples.is_contiguous()

    expected = reference(samples, dim=-1, keepdim=keepdim)
    output_storage = torch.empty(
        (*expected.shape[:-1], expected.shape[-1] * 2), dtype=torch.int64, device="dgx"
    )
    output = output_storage[..., ::2]
    assert output.shape == expected.shape
    assert not output.is_contiguous()

    actual = reference(dgx_samples, dim=-1, keepdim=keepdim, out=output)
    assert actual is output
    assert torch.equal(output.cpu(), expected)


@extrema
def test_arg_reduce_validation(dgx_setup_teardown, op):
    reference = getattr(torch, op)
    with pytest.raises((IndexError, RuntimeError), match="non-zero size"):
        reference(torch.empty(3, 0, device="dgx"), dim=1)
    with pytest.raises((IndexError, RuntimeError)):
        reference(torch.empty(0, device="dgx"))


@pytest.mark.parametrize("dtype", [torch.int64], ids=str)
def test_any(dgx_setup_teardown, dtype):
    samples = torch.Tensor([1, 0, 1, 0, 1]).to("dgx")
    result = torch.any(samples)
    assert result.to("cpu") == torch.any(samples.to("cpu"))

    samples = torch.tensor(
        [
            [0.0, float("inf"), 1.0],
            [0.0, 2.0, 3.0],
            [float("-inf"), 2.0, 0.0],
        ],
        dtype=torch.float32,
    )
    result = torch.isinf(samples.to("dgx")).any(dim=1)
    assert torch.equal(result.to("cpu"), torch.isinf(samples).any(dim=1))

    mask = torch.tensor(
        [
            [[False, False], [True, False], [False, True]],
            [[False, False], [False, False], [True, True]],
        ],
        device="dgx",
    )
    result = mask.any(dim=-1, keepdim=True)
    assert torch.equal(result.to("cpu"), mask.to("cpu").any(dim=-1, keepdim=True))

    out = torch.empty((2, 3), dtype=torch.bool, device="dgx")
    torch.any(mask, dim=-1, out=out)
    assert torch.equal(out.to("cpu"), mask.to("cpu").any(dim=-1))

    storage = torch.ones((1, 16, 16, 4), dtype=torch.bool)
    storage[..., :-1] = False
    samples = storage.to("dgx")[..., :-1]
    assert samples.stride() == (1024, 64, 4, 1)
    assert not samples.is_contiguous()
    assert not samples.any().item()


def test_truth_last_dim_row_reduction_boundaries(dgx_setup_teardown):
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
        samples = torch.arange(3 * width).reshape(3, width).remainder(3) != 0
        for reduction in (torch.all, torch.any):
            expected = reduction(samples, dim=-1)
            actual = reduction(samples.to("dgx"), dim=-1)
            assert torch.equal(actual.cpu(), expected)


@pytest.mark.parametrize(
    "input_dtype",
    [
        torch.bool,
        torch.uint8,
        torch.int8,
        torch.int16,
        torch.int32,
        torch.int64,
        torch.float32,
        torch.float64,
    ],
    ids=str,
)
@pytest.mark.parametrize("output_dtype", [torch.bool, torch.uint8], ids=str)
@pytest.mark.parametrize("keepdim", [False, True], ids=["dropdim", "keepdim"])
def test_any_dim_output_dtypes(
    dgx_setup_teardown, input_dtype, output_dtype, keepdim
):
    # Exercise every contiguous row-reduction variant with an offset, strided
    # output. Untouched bytes catch stores that overwrite neighboring elements.
    widths = [0, 1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 2049]
    rows = 9
    for width in widths:
        samples = torch.zeros((rows, width), dtype=input_dtype)
        if width:
            samples[1, 0] = 1
            samples[2, -1] = 1
            samples[4, :] = 1
            if input_dtype.is_floating_point:
                samples[5, width // 2] = float("nan")
                samples[7, -1] = float("inf")
            if input_dtype not in (torch.bool, torch.uint8):
                samples[6, 0] = -1
            elif input_dtype == torch.uint8:
                samples[6, 0] = 255

        expected = torch.any(samples, dim=-1, keepdim=keepdim).to(output_dtype)
        sentinel = True if output_dtype == torch.bool else 127
        expected_storage = torch.full((rows * 2 + 1,), sentinel, dtype=output_dtype)
        output_storage = expected_storage.to("dgx")
        output = output_storage[1::2]
        if keepdim:
            output = output.unsqueeze(-1)
        expected_storage[1::2] = expected.reshape(-1)

        actual = torch.any(samples.to("dgx"), dim=-1, keepdim=keepdim, out=output)

        assert actual is output
        assert torch.equal(output.cpu(), expected)
        # Check raw bytes as well as values, including normalized bool storage.
        assert torch.equal(
            output_storage.cpu().view(torch.uint8),
            expected_storage.view(torch.uint8),
        )


@pytest.mark.parametrize(
    "reduction,input_dtype,output_dtype,input_storage_shape,permutation",
    [
        (torch.any, torch.int8, torch.bool, (2, 4, 3, 5, 7), (0, 2, 3, 4, 1)),
        (torch.any, torch.int8, torch.uint8, (2, 4, 3, 5, 7), (0, 2, 3, 4, 1)),
        (torch.all, torch.bool, torch.bool, (2, 4, 3, 5), (0, 2, 3, 1)),
    ],
    ids=["any-rank5-bool", "any-rank5-byte", "all-rank4-bool"],
)
@pytest.mark.parametrize("keepdim", [False, True], ids=["dropdim", "keepdim"])
def test_truth_reduction_strided_input_and_output(
    dgx_setup_teardown,
    reduction,
    input_dtype,
    output_dtype,
    input_storage_shape,
    permutation,
    keepdim,
):
    if input_dtype == torch.bool:
        storage = torch.randint(0, 2, input_storage_shape, dtype=input_dtype)
    else:
        storage = torch.randint(-1, 2, input_storage_shape, dtype=input_dtype)
    samples = storage.permute(permutation)
    dgx_samples = storage.to("dgx").permute(permutation)
    assert not dgx_samples.is_contiguous()

    expected = reduction(samples, dim=-1, keepdim=keepdim).to(output_dtype)
    output_storage_shape = (*expected.shape[:-1], expected.shape[-1] * 2)
    output_storage = torch.empty(output_storage_shape, dtype=output_dtype, device="dgx")
    output = output_storage[..., ::2]
    assert output.shape == expected.shape
    assert not output.is_contiguous()

    reduction(dgx_samples, dim=-1, keepdim=keepdim, out=output)

    assert torch.equal(output.cpu(), expected)


@pytest.mark.parametrize("dtype", [torch.int64], ids=str)
def test_all(dgx_setup_teardown, dtype):
    samples = torch.Tensor([1, 0, 1, 0, 1]).to("dgx")
    result = torch.all(samples)
    assert result.to("cpu") == torch.all(samples.to("cpu"))

    samples = torch.rand(1, 53, 2, dtype=torch.float32, device="dgx")
    result = (samples > 0.5).all(dim=-1)
    assert torch.equal(result.to("cpu"), (samples.to("cpu") > 0.5).all(dim=-1))

    samples = torch.rand(4, 27, dtype=torch.float32, device="dgx")
    result = (samples > 0.5).all(dim=-1)
    assert torch.equal(result.to("cpu"), (samples.to("cpu") > 0.5).all(dim=-1))

    storage = torch.zeros((1, 16, 16, 4), dtype=torch.bool)
    storage[..., :-1] = True
    samples = storage.to("dgx")[..., :-1]
    assert samples.stride() == (1024, 64, 4, 1)
    assert not samples.is_contiguous()
    assert samples.all().item()
