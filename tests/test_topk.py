# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
import pytest
import torch


def check_topk(source, values, indices, k, dim=-1, largest=True, sorted=True):
    assert values.device.type == indices.device.type == "dgx"
    assert values.dtype == source.dtype
    assert indices.dtype == torch.int64
    values, indices = values.cpu(), indices.cpu()
    expected = source.topk(k, dim=dim, largest=largest).values
    compared = (
        values
        if sorted or source.ndim == 0
        else values.sort(dim=dim, descending=largest).values
    )
    torch.testing.assert_close(compared, expected, rtol=0, atol=0, equal_nan=True)
    if source.is_floating_point():
        # Gather integer views: CPU half gather can canonicalize NaN payloads.
        bit_dtype = {2: torch.int16, 4: torch.int32, 8: torch.int64}[
            source.element_size()
        ]
        values = values.contiguous().view(bit_dtype)
        source = source.contiguous().view(bit_dtype)
    if source.ndim == 0:
        assert indices.item() == 0
        gathered = source
    else:
        assert torch.all((indices >= 0) & (indices < source.size(dim)))
        ordered_indices = indices.sort(dim=dim).values
        if k > 1:
            assert torch.all(ordered_indices.diff(dim=dim) > 0)
        gathered = source.gather(dim, indices)
    assert torch.equal(values, gathered)


@pytest.mark.parametrize(
    "dtype",
    [
        torch.uint8,
        torch.int8,
        torch.int16,
        torch.int32,
        torch.int64,
        torch.float16,
        torch.float32,
        torch.float64,
    ],
)
@pytest.mark.parametrize(
    "shape, largest, sorted",
    [((7, 257), False, True), ((65537,), True, False)],
)
def test_topk_dtypes(dgx_setup_teardown, dtype, shape, largest, sorted):
    source = torch.randint(0 if dtype == torch.uint8 else -100, 100, shape).to(dtype)
    tensor = source.to("dgx")
    values, indices = tensor.topk(5, largest=largest, sorted=sorted)
    check_topk(source, values, indices, 5, largest=largest, sorted=sorted)
    assert torch.equal(tensor.cpu(), source)


@pytest.mark.parametrize(
    "shape, k, dim",
    [
        ((), 0, 0),
        ((), 1, -1),
        ((1,), 1, 0),
        ((3,), 2, -1),
        ((7,), 0, 0),
        ((7,), 7, 0),
        ((0, 3), 2, 1),
        ((3, 0), 0, -1),
        ((2, 3, 17), 2, -2),
        ((1, 43, 1), 5, 1),
        ((3, 1, 4), 1, 1),
    ],
)
def test_topk_shapes(dgx_setup_teardown, shape, k, dim):
    source = torch.randn(shape)
    values, indices = source.to("dgx").topk(k, dim)
    check_topk(source, values, indices, k, dim)


@pytest.mark.parametrize(
    "columns, k, dtype, largest, sorted",
    [
        (17, 4, torch.float32, False, True),
        (255, 255, torch.int64, True, True),
        (256, 1, torch.float32, False, False),
        (257, 7, torch.int64, True, False),
        (1023, 1023, torch.float64, False, True),
        (1024, 4, torch.float32, True, True),
        (1025, 4, torch.float32, False, True),
    ],
)
def test_topk_short_rows(dgx_setup_teardown, dtype, columns, k, largest, sorted):
    # Exercise shared-memory reuse when each block processes multiple rows.
    rows = 17 if columns > 1024 else 513
    if columns == 17:
        rows = 4096 * torch.cuda.device_count() + 1
    source = torch.randint(-100, 100, (rows, columns)).to(dtype)
    values, indices = source.to("dgx").topk(k, largest=largest, sorted=sorted)
    check_topk(source, values, indices, k, largest=largest, sorted=sorted)


@pytest.mark.parametrize(
    "dtype, value, largest",
    [
        (torch.int64, -(1 << 63), True),
        (torch.int64, (1 << 63) - 1, False),
        (torch.uint8, 0, True),
        (torch.uint8, 255, False),
        (torch.float16, float("nan"), False),
        (torch.float32, float("nan"), False),
        (torch.float64, float("nan"), False),
    ],
)
def test_topk_short_rows_padding(dgx_setup_teardown, dtype, value, largest):
    # Valid extreme keys can equal the padding key; all returned indices must be valid.
    source = torch.full((19, 257), value, dtype=dtype)
    values, indices = source.to("dgx").topk(257, largest=largest)
    check_topk(source, values, indices, 257, largest=largest)


@pytest.mark.parametrize(
    "rows, columns, k, largest, sorted",
    [
        (1, 65535, 4, False, True),
        (1, 65536, 4, True, False),
        (1, 65537, 4, True, True),
        (2, 131075, 4, False, False),
        (3, 131075, 4, True, True),
        (2, 131075, 8192, False, True),
        (2, 131075, 8193, False, True),
        (17, 4097, 4, False, False),
    ],
)
def test_topk_wide_rows(dgx_setup_teardown, rows, columns, k, largest, sorted):
    source = torch.arange(columns, dtype=torch.float32).repeat(rows, 1)
    source[1::2] = source[1::2].flip(-1)
    # Each row's winners lie in one column shard, alternating sides between rows.
    values, indices = source.to("dgx").topk(k, largest=largest, sorted=sorted)
    check_topk(source, values, indices, k, largest=largest, sorted=sorted)


@pytest.mark.parametrize(
    "dtype, rows", [(torch.int64, 2), (torch.float32, 3), (torch.float64, 2)]
)
def test_topk_wide_rows_strided(dgx_setup_teardown, dtype, rows):
    source_storage = torch.randint(-1000, 1000, (131075, rows * 2)).to(dtype)
    source = source_storage[:, 1::2]
    tensor = source_storage.to("dgx")[:, 1::2]
    values_storage = torch.full((8, rows), -12345, dtype=dtype, device="dgx")
    indices_storage = torch.full((8, rows), -12345, dtype=torch.int64, device="dgx")
    values, indices = values_storage[1::2], indices_storage[1::2]
    torch.topk(tensor, 4, dim=0, largest=False, out=(values, indices))
    check_topk(source, values, indices, 4, dim=0, largest=False)
    for output_storage in (values_storage, indices_storage):
        assert torch.all(output_storage.cpu()[::2] == -12345)
    assert torch.equal(tensor.cpu(), source)


@pytest.mark.parametrize(
    "k, largest, sorted", [(1, False, True), (31, True, False), (65537, False, True)]
)
def test_topk_large_vector(dgx_setup_teardown, k, largest, sorted):
    # Reversing a unique range puts all winners on one side of the device partition.
    source = torch.arange(65537, dtype=torch.int64).flip(0) - 32768
    values, indices = source.to("dgx").topk(k, largest=largest, sorted=sorted)
    check_topk(source, values, indices, k, largest=largest, sorted=sorted)


@pytest.mark.parametrize(
    "dtype, rows, columns, k, largest, sorted",
    [
        (torch.float16, 1, 1032, 1032, False, True),
        (torch.float32, 1, 1032, 17, True, False),
        (torch.float64, 1, 1032, 516, False, True),
        (torch.float32, 1, 131075, 4, True, True),
        (torch.float32, 3, 131075, 4, False, False),
        (torch.float64, 1, 131075, 4, False, True),
        (torch.float64, 3, 131075, 4, True, False),
    ],
)
def test_topk_special_values(
    dgx_setup_teardown, dtype, rows, columns, k, largest, sorted
):
    pattern = torch.tensor(
        [
            float("nan"),
            -float("nan"),
            float("inf"),
            -float("inf"),
            0.0,
            -0.0,
            -3.0,
            3.0,
        ],
        dtype=dtype,
    )
    source = pattern.repeat(rows, (columns + 7) // 8)[:, :columns].contiguous()
    values, indices = source.to("dgx").topk(k, largest=largest, sorted=sorted)
    check_topk(source, values, indices, k, largest=largest, sorted=sorted)


@pytest.mark.parametrize("largest", [False, True])
def test_topk_int64_extremes(dgx_setup_teardown, largest):
    source = torch.tensor([-(1 << 63), (1 << 63) - 1, 0, -1, 1]).repeat(13108)[:65537]
    values, indices = source.to("dgx").topk(137, largest=largest)
    check_topk(source, values, indices, 137, largest=largest)


@pytest.mark.parametrize("dim", [0, 1, -1])
def test_topk_strided_input_and_outputs(dgx_setup_teardown, dim):
    storage = torch.randn(4, 5, 14)
    source = storage[..., 1::2].transpose(0, 2)
    tensor = storage.to("dgx")[..., 1::2].transpose(0, 2)
    shape = list(source.shape)
    shape[dim] = 3
    shape[-1] *= 2
    values_storage = torch.full(shape, -12345, dtype=source.dtype, device="dgx")
    indices_storage = torch.full(shape, -12345, dtype=torch.int64, device="dgx")
    values, indices = values_storage[..., 1::2], indices_storage[..., 1::2]
    strides = values.stride(), indices.stride()
    returned = torch.topk(tensor, 3, dim=dim, largest=False, out=(values, indices))
    assert returned.values is values and returned.indices is indices
    assert (values.stride(), indices.stride()) == strides
    check_topk(source, values, indices, 3, dim, largest=False)
    assert torch.equal(tensor.cpu(), source)
    for output_storage in (values_storage, indices_storage):
        assert torch.all(output_storage.cpu()[..., ::2] == -12345)


@pytest.mark.parametrize("alias", ["values", "indices", "resized_view", "shifted_view"])
def test_topk_out_alias(dgx_setup_teardown, alias):
    source = torch.randint(-1000, 1000, (5, 17))
    tensor = source.to("dgx")
    values = torch.empty(0, dtype=torch.int64, device="dgx")
    indices = torch.empty_like(values)
    if alias == "values":
        values = tensor
    elif alias == "indices":
        indices = tensor
    elif alias == "resized_view":
        values = tensor[-1:]
    else:
        values = tensor[1:, :3]
        tensor, source = tensor[:-1], source[:-1]
    torch.topk(tensor, 3, out=(values, indices))
    check_topk(source, values, indices, 3)


@pytest.mark.parametrize(
    "overlap", ["full", "partial", "values_internal", "indices_internal"]
)
def test_topk_rejects_overlapping_outputs(dgx_setup_teardown, overlap):
    tensor = torch.arange(43, device="dgx", dtype=torch.int64)
    storage = torch.empty(8, device="dgx", dtype=torch.int64)
    values = storage[:7]
    indices = torch.empty(7, device="dgx", dtype=torch.int64)
    if overlap == "full":
        indices = values
    elif overlap == "partial":
        indices = storage[1:]
    elif overlap == "values_internal":
        values = storage[:1].expand(7)
    else:
        indices = storage[:1].expand(7)
    with pytest.raises(RuntimeError, match="single memory location"):
        torch.topk(tensor, 7, out=(values, indices))


@pytest.mark.parametrize("k", [-1, 4])
def test_topk_invalid_k(dgx_setup_teardown, k):
    with pytest.raises(RuntimeError, match="selected index k out of range"):
        torch.empty((2, 3), device="dgx").topk(k)


@pytest.mark.parametrize("shape, dim", [((), 1), ((2, 3), -3), ((0, 3), 2)])
def test_topk_invalid_dim(dgx_setup_teardown, shape, dim):
    with pytest.raises(IndexError, match="Dimension out of range"):
        torch.empty(shape, device="dgx").topk(0, dim=dim)
