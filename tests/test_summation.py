# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
import pytest
import torch


def _assert_reduction(reduction, samples, dim=None, keepdim=False, dgx_samples=None):
    if dgx_samples is None:
        dgx_samples = samples.to("dgx")

    if dim is None:
        actual = reduction(dgx_samples)
        expected = reduction(samples)
    else:
        actual = reduction(dgx_samples, dim=dim, keepdim=keepdim)
        expected = reduction(samples, dim=dim, keepdim=keepdim)

    assert actual.device.type == "dgx"
    if expected.dtype.is_floating_point:
        torch.testing.assert_close(actual.cpu(), expected, rtol=1e-5, atol=1e-8)
    else:
        torch.testing.assert_close(actual.cpu(), expected, rtol=0, atol=0)


@pytest.mark.parametrize(
    "dtype",
    [torch.float32, torch.float64, torch.int32, torch.int64, torch.bool],
    ids=str,
)
@pytest.mark.parametrize("numel", [0, 1, 5, 65, 1025])
def test_cumsum(dgx_setup_teardown, dtype, numel):
    samples = torch.ones(numel, dtype=dtype)
    actual = torch.cumsum(samples.to("dgx"), 0)
    expected = torch.cumsum(samples, 0)

    torch.testing.assert_close(actual.cpu(), expected, rtol=0, atol=0)


@pytest.mark.parametrize(
    "shape,dim,dtype",
    [
        ((3, 1025), -1, torch.float32),
        ((2, 5, 7), 1, torch.float64),
        ((2, 5, 7), -3, torch.float32),
        ((3, 5), 0, torch.float16),
        ((3, 5), 1, torch.bfloat16),
        ((3, 5), 0, torch.int32),
        ((3, 5), 1, torch.bool),
    ],
)
def test_cumsum_multidimensional(dgx_setup_teardown, shape, dim, dtype):
    samples = torch.randint(0, 2, shape).to(dtype)
    actual = torch.cumsum(samples.to("dgx"), dim)
    expected = torch.cumsum(samples, dim)

    torch.testing.assert_close(actual.cpu(), expected, rtol=0, atol=0)


@pytest.mark.parametrize("dim", [0, 1, -1])
def test_cumsum_noncontiguous_forward_backward(dgx_setup_teardown, dim):
    base = torch.randn(3, 7, 10, dtype=torch.float64)
    samples = base.transpose(0, 1)[..., 1::2].requires_grad_()
    dgx_samples = base.to("dgx").transpose(0, 1)[..., 1::2].requires_grad_()
    output = torch.empty(7, 3, 10, dtype=torch.float64, device="dgx")[..., ::2]

    actual = torch.cumsum(dgx_samples, dim)
    expected = torch.cumsum(samples, dim)
    torch.cumsum(dgx_samples.detach(), dim, out=output)

    torch.testing.assert_close(actual.cpu(), expected)
    torch.testing.assert_close(output.cpu(), expected)

    grad_output = torch.randn_like(samples)
    expected.backward(grad_output)
    actual.backward(grad_output.to("dgx"))
    torch.testing.assert_close(dgx_samples.grad.cpu(), samples.grad)


@pytest.mark.parametrize(
    "shape,dim", [((), -1), ((0, 3), 0), ((2, 0, 3), 1), ((5,), -1)]
)
def test_cumsum_scalar_empty_and_negative_dim(dgx_setup_teardown, shape, dim):
    samples = torch.randn(shape)
    actual = torch.cumsum(samples.to("dgx"), dim)
    expected = torch.cumsum(samples, dim)

    torch.testing.assert_close(actual.cpu(), expected)


def test_cumsum_output_dtype(dgx_setup_teardown):
    samples = torch.arange(15, dtype=torch.int32).reshape(3, 5)
    output = torch.empty(3, 5, dtype=torch.float64, device="dgx")

    torch.cumsum(samples.to("dgx"), 1, out=output)

    torch.testing.assert_close(
        output.cpu(), torch.cumsum(samples, 1, dtype=torch.float64)
    )


@pytest.mark.parametrize(
    "shape, dim, dtype",
    [
        ((), 0, torch.float32),
        ((3, 0), -1, torch.float32),
        ((513,), 0, torch.float64),
        ((3, 513), -1, torch.float32),
        ((513, 3), 0, torch.float16),
        ((3, 17), -1, torch.bfloat16),
        ((3, 17), 0, torch.int32),
        ((17,), -1, torch.bool),
        ((3, 17), 1, torch.complex64),
    ],
)
def test_cumprod(dgx_setup_teardown, shape, dim, dtype):
    source = torch.randint(-1, 2, shape).to(dtype)
    if dtype.is_floating_point or dtype.is_complex:
        source = source * 0.01 + 0.99
    if len(shape) == 2:
        source = source.t().contiguous().t()
    tensor = source.to("dgx")
    expected = source.cumprod(dim)
    torch.testing.assert_close(tensor.cumprod(dim).cpu(), expected)
    storage = torch.empty((*shape, 2), dtype=expected.dtype, device="dgx")
    out = storage[..., 0]
    torch.cumprod(tensor, dim, dtype=expected.dtype, out=out)
    torch.testing.assert_close(out.cpu(), expected)
    tensor = tensor.to(expected.dtype)
    torch.cumprod(tensor, dim, out=tensor)
    torch.testing.assert_close(tensor.cpu(), expected)


def test_cumprod_backward(dgx_setup_teardown):
    # No zeros, one zero, and multiple zeros exercise distinct backward paths.
    source = torch.tensor(
        [[0.9, 0.8, 0.7, 0.6], [0.9, 0.0, 0.7, 0.6], [0.0, 0.8, 0.0, 0.6]],
        requires_grad=True,
    )
    tensor = source.detach().to("dgx").requires_grad_()
    weights = torch.arange(12, dtype=source.dtype).reshape_as(source)
    (source.cumprod(-1) * weights).sum().backward()
    (tensor.cumprod(-1) * weights.to("dgx")).sum().backward()
    torch.testing.assert_close(tensor.grad.cpu(), source.grad)


@pytest.mark.parametrize("dims", [(0,), (1,), (0, 1), (1, 2), (-1, 0)])
@pytest.mark.parametrize("keepdim", [False, True])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_sum_attention_bias_broadcast_axes(dgx_setup_teardown, dims, keepdim, dtype):
    # Bias gradients can reduce several nonadjacent axes of a strided BHQK tensor.
    base = torch.randn(3, 17, 5, 39, dtype=dtype)
    samples = base.transpose(1, 2)[..., 1::2]
    dgx_samples = base.to("dgx").transpose(1, 2)[..., 1::2]
    expected = samples.sum(dims, keepdim=keepdim)
    storage_shape = (*expected.shape[:-1], 2 * expected.shape[-1])
    output = torch.empty(storage_shape, dtype=dtype, device="dgx")[..., ::2]
    torch.sum(dgx_samples, dims, keepdim=keepdim, out=output)
    torch.testing.assert_close(output.cpu(), expected, atol=1e-5, rtol=1e-5)


@pytest.mark.parametrize("shape", [(3, 0, 5, 7), (0, 3, 5, 7), (3, 1, 5, 7)])
def test_sum_broadcast_axes_empty_and_singleton(dgx_setup_teardown, shape):
    base = torch.randn(shape)
    sample = base.transpose(2, 3)
    dgx_sample = base.to("dgx").transpose(2, 3)
    for dims in [(1,), (0, 1)]:
        torch.testing.assert_close(dgx_sample.sum(dims).cpu(), sample.sum(dims))


@pytest.mark.parametrize("keepdim", [False, True], ids=["dropdim", "keepdim"])
def test_last_dim_sum_mean_row_reduction_boundaries(dgx_setup_teardown, keepdim):
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
        samples = torch.randn((3, width), dtype=torch.float32)
        for reduction in (torch.sum, torch.mean):
            expected = reduction(samples, dim=-1, keepdim=keepdim)
            actual = reduction(samples.to("dgx"), dim=-1, keepdim=keepdim)
            torch.testing.assert_close(
                actual.cpu(), expected, rtol=1e-5, atol=1e-8, equal_nan=True
            )


def test_last_dim_sum_mean_seed_identity_signed_zero(dgx_setup_teardown):
    exact_widths = [2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048]
    for width in exact_widths:
        samples = torch.full((3, width), -0.0)
        for reduction in (torch.sum, torch.mean):
            expected = reduction(samples, dim=-1)
            actual = reduction(samples.to("dgx"), dim=-1).cpu()
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
            assert torch.equal(torch.signbit(actual), torch.signbit(expected))


@pytest.mark.parametrize(
    "dtype,shape,dim,keepdim",
    [
        (torch.float32, (51,), None, False),
        (torch.float64, (51,), -1, True),
        (torch.float32, (23, 1), -1, True),
        (torch.float64, (1, 17, 5), 0, False),
        (torch.float32, (17, 1, 5), -2, True),
        (torch.float64, (101, 3), -1, False),
        (torch.float32, (3, 36), -1, False),
        (torch.float64, (8, 3, 16), -1, False),
        (torch.float32, (1, 3, 32, 32), (1, 2, 3), False),
    ],
    ids=[
        "full-float32",
        "full-explicit-keepdim-float64",
        "singleton-dim-float32",
        "singleton-first-dim-float64",
        "singleton-middle-dim-float32",
        "last-dim-many-rows-float64",
        "last-dim-long-rows-float32",
        "last-dim-rank3-float64",
        "multi-dim-full-float32",
    ],
)
def test_mean_paths(dgx_setup_teardown, dtype, shape, dim, keepdim):
    samples = torch.rand(shape, dtype=dtype)
    _assert_reduction(torch.mean, samples, dim=dim, keepdim=keepdim)


@pytest.mark.parametrize("input_dtype", [torch.int32, torch.int64, torch.bool], ids=str)
@pytest.mark.parametrize("output_dtype", [torch.float32, torch.float64], ids=str)
@pytest.mark.parametrize("dim", [None, -1], ids=["full", "last-dim"])
def test_mean_integer_input_with_floating_dtype(
    dgx_setup_teardown, input_dtype, output_dtype, dim
):
    if input_dtype == torch.bool:
        samples = torch.tensor([[True, False, True], [False, True, True]])
    else:
        samples = torch.arange(6, dtype=input_dtype).reshape(2, 3)

    kwargs = {"dtype": output_dtype}
    if dim is not None:
        kwargs["dim"] = dim

    actual = torch.mean(samples.to("dgx"), **kwargs)
    expected = torch.mean(samples, **kwargs)

    assert actual.dtype == output_dtype
    torch.testing.assert_close(actual.cpu(), expected)


@pytest.mark.parametrize(
    "reduction,dtype,view",
    [
        (torch.sum, torch.float32, "channel"),
        (torch.mean, torch.float64, "channel"),
        (torch.sum, torch.float64, "transposed-slice"),
        (torch.mean, torch.float32, "transposed-slice"),
    ],
    ids=[
        "channel-sum-float32",
        "channel-mean-float64",
        "transposed-slice-sum-float64",
        "transposed-slice-mean-float32",
    ],
)
def test_full_reduction_noncontiguous(dgx_setup_teardown, dtype, reduction, view):
    if view == "channel":
        storage = torch.arange(5 * 32 * 4, dtype=dtype).reshape(5, 32, 4)
        dgx_storage = storage.to("dgx")
        samples = storage[..., -1]
        dgx_samples = dgx_storage[..., -1]
        assert dgx_samples.stride() == (128, 4)
    else:
        storage = torch.arange(9 * 13, dtype=dtype).reshape(9, 13)
        dgx_storage = storage.to("dgx")
        samples = storage[1:8, 2:11].transpose(0, 1)
        dgx_samples = dgx_storage[1:8, 2:11].transpose(0, 1)

    assert not dgx_samples.is_contiguous()
    assert dgx_samples.storage_offset() > 0
    _assert_reduction(reduction, samples, dgx_samples=dgx_samples)


@pytest.mark.parametrize(
    "dtype,shape,dim,keepdim",
    [
        (torch.float32, (73,), None, False),
        (torch.float64, (73,), -1, True),
        (torch.float32, (23, 1), -1, True),
        (torch.float32, (100, 64), 0, True),
        (torch.float64, (100, 64), 0, False),
        (torch.float64, (2, 17, 3), 0, True),
        (torch.float32, (2, 17, 3), 0, False),
        (torch.float32, (100, 64), -1, True),
        (torch.float64, (8, 3, 16), -1, True),
        (torch.float32, (8, 3, 16), 1, True),
        (torch.float64, (8, 3, 16), 1, False),
        (torch.float64, (1, 256, 3), 1, True),
        (torch.float32, (1, 256, 3), -2, False),
        (torch.float32, (16, 256, 768), 1, True),
    ],
    ids=[
        "full-float32",
        "full-explicit-keepdim-float64",
        "singleton-dim-float32",
        "dim0-rank2-keepdim-float32",
        "dim0-rank2-float64",
        "dim0-rank3-keepdim-float64",
        "dim0-rank3-float32",
        "last-dim-rank2-float32",
        "last-dim-rank3-float64",
        "middle-dim-small-keepdim-float32",
        "middle-dim-small-float64",
        "middle-dim-large-keepdim-float64",
        "middle-dim-large-float32",
        "middle-dim-adaln-keepdim-float32",
    ],
)
def test_sum_paths(dgx_setup_teardown, dtype, shape, dim, keepdim):
    samples = torch.rand(shape, dtype=dtype)
    _assert_reduction(torch.sum, samples, dim=dim, keepdim=keepdim)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64], ids=str)
@pytest.mark.parametrize(
    "shape,layout",
    [
        ((1, 0, 3), "contiguous"),
        ((3, 17, 5), "contiguous"),
        ((1, 257, 3), "contiguous"),
        ((1, 256, 32), "contiguous"),
        ((1, 257, 33), "contiguous"),
        ((3, 513, 65), "contiguous"),
        ((3, 257, 33), "sequence-slice"),
        ((3, 257, 33), "expanded"),
        ((3, 257, 33), "feature-slice"),
    ],
)
@pytest.mark.parametrize("keepdim", [False, True])
def test_middle_dim_sum_tile_boundaries(
    dgx_setup_teardown, dtype, shape, layout, keepdim
):
    batch, sequence, features = shape
    storage_sequence = 1 if layout == "expanded" else sequence
    if layout == "sequence-slice":
        storage_sequence *= 2
    storage_features = features * (2 if layout == "feature-slice" else 1)
    storage_shape = (batch, storage_sequence, storage_features)
    # Exact binary fractions make this a check of indexing and shard boundaries
    # independent of the order used to accumulate positive and negative values.
    storage = torch.randint(-64, 65, storage_shape).to(dtype) / 64
    dgx_storage = storage.to("dgx")

    def view(tensor):
        if layout == "sequence-slice":
            return tensor[:, 1::2, :]
        if layout == "feature-slice":
            return tensor[:, :, 1::2]
        return tensor.expand(shape)

    samples, dgx_samples = view(storage), view(dgx_storage)
    expected = samples.sum(dim=1, keepdim=keepdim)
    # Exercise the contiguous temporary and final copy, including an offset
    # output view whose surrounding storage must not be overwritten.
    output_storage = torch.full(
        (batch, 2 * features + 1), -999.0, dtype=dtype, device="dgx"
    )
    output = output_storage[:, 1::2]
    if keepdim:
        output = output.unsqueeze(1)
    returned = torch.sum(dgx_samples, dim=-2, keepdim=keepdim, out=output)
    assert returned is output
    torch.testing.assert_close(output.cpu(), expected, rtol=0, atol=0)
    torch.testing.assert_close(
        output_storage.cpu()[:, ::2],
        torch.full((batch, features + 1), -999.0, dtype=dtype),
        rtol=0,
        atol=0,
    )


@pytest.mark.parametrize(
    "reduction,dtype,dim,input_storage_shape,permutation",
    [
        (torch.sum, torch.float64, 0, (4, 17, 5), None),
        (torch.sum, torch.float64, 1, (4, 17, 5), None),
        (torch.sum, torch.float64, 2, (4, 17, 5), None),
        (torch.mean, torch.float64, 2, (4, 17, 5), None),
        (torch.mean, torch.float64, 0, (1, 17, 5), None),
        (torch.mean, torch.float32, -2, (1, 17, 5), (2, 0, 1)),
        (torch.sum, torch.float32, [3], (1, 3, 2392, 1592), (0, 2, 3, 1)),
        (torch.mean, torch.float32, [3], (1, 3, 2392, 1592), (0, 2, 3, 1)),
        (torch.sum, torch.float64, -1, (2, 4, 3, 5, 7), (0, 2, 3, 4, 1)),
        (torch.mean, torch.float64, -1, (2, 4, 3, 5, 7), (0, 2, 3, 4, 1)),
    ],
    ids=[
        "sum-first",
        "sum-middle",
        "sum-last",
        "mean-last",
        "mean-singleton-first",
        "mean-singleton-middle-strided-input",
        "sum-last-rank4-strided-input",
        "mean-last-rank4-strided-input",
        "sum-last-rank5-strided-input",
        "mean-last-rank5-strided-input",
    ],
)
@pytest.mark.parametrize("keepdim", [False, True], ids=["dropdim", "keepdim"])
def test_reduction_noncontiguous_out(
    dgx_setup_teardown,
    reduction,
    dtype,
    dim,
    input_storage_shape,
    permutation,
    keepdim,
):
    storage = torch.rand(input_storage_shape, dtype=dtype)
    dgx_storage = storage.to("dgx")
    samples = storage if permutation is None else storage.permute(permutation)
    dgx_samples = (
        dgx_storage if permutation is None else dgx_storage.permute(permutation)
    )
    if permutation is not None:
        assert not dgx_samples.is_contiguous()
    expected = reduction(samples, dim=dim, keepdim=keepdim)

    output_storage_shape = (*expected.shape[:-1], expected.shape[-1] * 2)
    output_storage = torch.empty(
        output_storage_shape, dtype=expected.dtype, device="dgx"
    )
    output = output_storage[..., ::2]
    assert output.shape == expected.shape
    assert not output.is_contiguous()

    returned = reduction(dgx_samples, dim=dim, keepdim=keepdim, out=output)

    assert returned.data_ptr() == output.data_ptr()
    torch.testing.assert_close(output.cpu(), expected, rtol=1e-5, atol=1e-8)


@pytest.mark.parametrize("dtype", [torch.int32, torch.int64, torch.bool], ids=str)
def test_sum_nonfloating(dgx_setup_teardown, dtype):
    if dtype == torch.bool:
        samples = torch.tensor([True, False, False, True, False, True])
    else:
        samples = torch.randint(0, 10, (57,), dtype=dtype)

    _assert_reduction(torch.sum, samples)
