# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
import torch
import pytest


def check_sort(source, values, indices, dim=-1, descending=False, stable=False):
    values, indices = values.cpu(), indices.cpu()
    expected = source.sort(dim=dim, descending=descending, stable=stable)
    if stable:
        assert torch.equal(indices, expected.indices)
    torch.testing.assert_close(
        values,
        expected.values,
        rtol=0,
        atol=0,
        equal_nan=True,
    )
    if source.is_floating_point():
        # Gather integer views to preserve NaN payloads and signed zeros.
        bit_dtype = {2: torch.int16, 4: torch.int32, 8: torch.int64}[
            source.element_size()
        ]
        values = values.contiguous().view(bit_dtype)
        source = source.contiguous().view(bit_dtype)
    if source.ndim == 0:
        assert indices.item() == 0
        gathered = source
    else:
        index_shape = [1] * source.ndim
        index_shape[dim] = source.size(dim)
        expected_indices = (
            torch.arange(source.size(dim)).reshape(index_shape).expand_as(source)
        )
        assert torch.equal(indices.sort(dim=dim).values, expected_indices)
        gathered = source.gather(dim, indices)
    assert torch.equal(values, gathered)


@pytest.mark.parametrize(
    "size, dtype, descending",
    [
        (0, torch.int64, False),
        (3, torch.int64, True),
        (65537, torch.float32, True),
        (43, torch.float64, False),
    ],
)
@pytest.mark.parametrize("stable", [False, True])
def test_argsort(dgx_setup_teardown, size, dtype, descending, stable):
    generator = torch.Generator().manual_seed(42)
    source = torch.randint(-421, 421, (size,), generator=generator).to(dtype)
    tensor = source.to("dgx")
    indices = torch.argsort(tensor, descending=descending, stable=stable)
    check_sort(source, tensor[indices], indices, descending=descending, stable=stable)
    assert torch.equal(tensor.cpu(), source)


@pytest.mark.parametrize(
    "pattern, descending",
    [
        ("random", False),
        ("equal", False),
        ("skewed", False),
        ("sorted", False),
        ("reversed", False),
        ("extremes", False),
        ("periodic", False),
        ("equal", True),
        ("extremes", True),
    ],
)
def test_sort_partition_distributions(dgx_setup_teardown, pattern, descending):
    size = 262147
    generator = torch.Generator().manual_seed(2026)
    if pattern == "random":
        source = torch.randint(-(1 << 62), 1 << 62, (size,), generator=generator)
    elif pattern == "equal":
        source = torch.full((size,), -7, dtype=torch.int64)
    elif pattern == "skewed":
        source = torch.zeros(size, dtype=torch.int64)
        source[::97] = torch.randint(
            -(1 << 62), 1 << 62, source[::97].shape, generator=generator
        )
    elif pattern == "extremes":
        source = torch.tensor([-(1 << 63), (1 << 63) - 1, 0, -1, 1]).repeat(
            (size + 4) // 5
        )[:size]
    elif pattern == "periodic":
        source = (torch.arange(size) % 32) - 16
    else:
        source = torch.arange(size) - size // 2
        if pattern == "reversed":
            source = source.flip(0)
    tensor = source.to("dgx")
    values, indices = torch.sort(tensor, dim=0, stable=True, descending=descending)
    check_sort(source, values, indices, descending=descending, stable=True)
    assert torch.equal(tensor.cpu(), source)


@pytest.mark.parametrize(
    "size, dtype",
    [(131075, torch.int64), (43, torch.float32), (131075, torch.float64)],
)
def test_sort_strided_input_and_outputs(dgx_setup_teardown, size, dtype):
    source_storage = torch.randint(-1000000, 1000000, (size * 2 + 3,)).to(dtype)
    source = source_storage[1 : 1 + size * 2 : 2]
    tensor = source_storage.to("dgx")[1 : 1 + size * 2 : 2]
    values_storage = torch.full((size * 2 + 3,), -123456789, device="dgx", dtype=dtype)
    indices_storage = torch.full_like(values_storage, -123456789, dtype=torch.int64)
    values = values_storage[1 : 1 + size * 2 : 2]
    indices = indices_storage[1 : 1 + size * 2 : 2]
    torch.sort(tensor, stable=False, out=(values, indices))
    check_sort(source, values, indices)
    for storage in (values_storage, indices_storage):
        host = storage.cpu()
        assert torch.all(host[::2] == -123456789)
        assert host[-2] == -123456789


@pytest.mark.parametrize(
    "size, layout, alias_indices",
    [
        (0, "contiguous", False),
        (1, "contiguous", True),
        (43, "contiguous", False),
        (43, "contiguous", True),
        (131075, "strided", False),
        (131075, "strided", True),
        (131075, "resized_view", False),
        (131075, "resized_view", True),
    ],
)
def test_sort_out_resize_and_alias(dgx_setup_teardown, size, layout, alias_indices):
    source_size = size * 2 if layout == "strided" else size
    source = torch.randint(-(1 << 62), 1 << 62, (source_size,))
    tensor = source.to("dgx")
    if layout == "strided":
        source, tensor = source[::2], tensor[::2]
    aliased_output = tensor[-1:] if layout == "resized_view" else tensor
    other_output = torch.empty(0, device="dgx", dtype=torch.int64)
    values, indices = aliased_output, other_output
    if alias_indices:
        values, indices = indices, values
    torch.sort(tensor, stable=False, out=(values, indices))
    assert tensor.shape == source.shape
    check_sort(source, values, indices)


@pytest.mark.parametrize(
    "overlap", ["full", "partial", "values_internal", "indices_internal"]
)
def test_sort_rejects_overlapping_outputs(dgx_setup_teardown, overlap):
    tensor = torch.arange(43, device="dgx", dtype=torch.int64)
    storage = torch.empty(44, device="dgx", dtype=torch.int64)
    values = storage[:43]
    indices = torch.empty_like(tensor)
    if overlap == "full":
        indices = values
    elif overlap == "partial":
        indices = storage[1:]
    elif overlap == "values_internal":
        values = storage[:1].expand(43)
    else:
        indices = storage[:1].expand(43)
    with pytest.raises(RuntimeError, match="single memory location"):
        torch.sort(tensor, stable=False, out=(values, indices))


@pytest.mark.parametrize(
    "size, dtype, descending",
    [
        (1, torch.float32, False),
        (65536, torch.float32, False),
        (43, torch.float64, True),
        (262147, torch.float64, False),
    ],
)
def test_sort_floating(dgx_setup_teardown, size, dtype, descending):
    source = torch.randn(size, dtype=dtype)
    tensor = source.to("dgx")
    values, indices = torch.sort(tensor, stable=False, descending=descending)
    check_sort(source, values, indices, descending=descending)
    assert torch.equal(tensor.cpu(), source)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize(
    "pattern, descending",
    [("mixed", False), ("mixed", True), ("nan", False), ("zero", True)],
)
def test_sort_floating_special_values(dgx_setup_teardown, dtype, pattern, descending):
    if dtype == torch.float32:
        bit_dtype, sign_bit = torch.int32, 1 << 31
        infinity, quiet_nan_bit = 0x7F800000, 1 << 22
    else:
        bit_dtype, sign_bit = torch.int64, 1 << 63
        infinity, quiet_nan_bit = 0x7FF0000000000000, 1 << 51
    # Zeros, smallest subnormals, largest finite values, infinities, and
    # signaling/quiet NaNs with distinct payloads, each with both signs.
    magnitudes = [
        0,
        1,
        infinity - 1,
        infinity,
        infinity + 1,
        infinity + quiet_nan_bit + 0x1234,
        infinity + quiet_nan_bit + 0x4321,
    ]
    bits = magnitudes + [value - sign_bit for value in magnitudes]
    source = torch.tensor(bits, dtype=bit_dtype).view(dtype)
    if pattern == "nan":
        source = source[torch.isnan(source)]
    elif pattern == "zero":
        source = source[source == 0]
    size = 65537
    source = source.repeat((size + source.numel() - 1) // source.numel())[:size]
    tensor = source.to("dgx")
    values, indices = torch.sort(tensor, stable=True, descending=descending)
    check_sort(source, values, indices, descending=descending, stable=True)
    assert torch.equal(tensor.cpu().view(bit_dtype), source.view(bit_dtype))


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("alias", ["values", "resized_values", "indices"])
def test_sort_floating_out_alias(dgx_setup_teardown, dtype, alias):
    source = torch.randn(131075, dtype=dtype)
    tensor = source.to("dgx")
    values = torch.empty(0, device="dgx", dtype=dtype)
    indices = torch.empty(0, device="dgx", dtype=torch.int64)
    if alias == "values":
        values = tensor
    elif alias == "resized_values":
        values = tensor[-1:]
    else:
        indices.set_(tensor.untyped_storage(), 0, (0,), (1,))
    torch.sort(tensor, stable=False, out=(values, indices))
    assert tensor.shape == source.shape
    check_sort(source, values, indices)


@pytest.mark.parametrize(
    "shape, dim, descending, dtype",
    [
        ((7, 11), 0, False, torch.float32),
        ((3, 65537), -1, True, torch.int64),
        ((3, 65537), -1, True, torch.float32),
        ((2, 3, 17), -2, True, torch.float64),
        ((1, 43, 1), 1, False, torch.int64),
        ((0, 3, 2), 1, False, torch.float32),
        ((3, 0), -1, True, torch.int64),
        ((3, 1, 4), 1, True, torch.float64),
        ((), 0, False, torch.float32),
    ],
)
def test_sort_slices(dgx_setup_teardown, dtype, shape, dim, descending):
    source = torch.randint(-421, 421, shape).to(dtype)
    if source.is_floating_point():
        source = source / 7
    tensor = source.to("dgx")
    values, indices = torch.sort(tensor, dim=dim, stable=False, descending=descending)
    check_sort(source, values, indices, dim, descending)
    assert torch.equal(tensor.cpu(), source)


@pytest.mark.parametrize("dtype", [torch.int64, torch.float32, torch.float64])
def test_sort_slices_strided_input_and_outputs(dgx_setup_teardown, dtype):
    storage = torch.randint(-421, 421, (4, 5, 14)).to(dtype)
    source = storage[..., 1::2].transpose(0, 2)
    tensor = storage.to("dgx")[..., 1::2].transpose(0, 2)
    output_shape = (*source.shape[:-1], source.size(-1) * 2 + 2)
    values_storage = torch.full(output_shape, -1234567, dtype=dtype, device="dgx")
    indices_storage = torch.full(
        output_shape, -1234567, dtype=torch.int64, device="dgx"
    )
    values, indices = values_storage[..., 1:-1:2], indices_storage[..., 1:-1:2]
    torch.sort(tensor, dim=1, stable=False, descending=True, out=(values, indices))
    check_sort(source, values, indices, dim=1, descending=True)
    assert torch.equal(tensor.cpu(), source)
    for output_storage in (values_storage, indices_storage):
        host = output_storage.cpu()
        assert torch.all(host[..., ::2] == -1234567)
        assert torch.all(host[..., -1] == -1234567)


@pytest.mark.parametrize("dtype", [torch.int64, torch.float32, torch.float64])
def test_sort_slices_transposed_out_alias(dgx_setup_teardown, dtype):
    storage = torch.randint(-421, 421, (5, 7)).to(dtype)
    source = storage.t()
    values = storage.to("dgx").t()
    indices = torch.empty((5, 7), dtype=torch.int64, device="dgx").t()
    torch.sort(values, dim=0, stable=False, descending=True, out=(values, indices))
    check_sort(source, values, indices, dim=0, descending=True)
    assert values.stride() == indices.stride() == (1, 7)


@pytest.mark.parametrize("dtype", [torch.int64, torch.float32, torch.float64])
def test_sort_slices_shifted_out_alias(dgx_setup_teardown, dtype):
    source = torch.randint(-421, 421, (5, 17)).to(dtype)
    storage = source.to("dgx")
    tensor, values = storage[:-1], storage[1:]
    indices = torch.empty_like(values, dtype=torch.int64)
    torch.sort(tensor, descending=True, out=(values, indices))
    check_sort(source[:-1], values, indices, descending=True)


def test_quantile_columns(dgx_setup_teardown):
    source = torch.randn(16384, 3)
    tensor = source.to("dgx")[:, :2]
    quantiles = torch.tensor([0.01, 0.99])
    result = torch.quantile(tensor, quantiles.to("dgx"), dim=0)
    expected = torch.quantile(source[:, :2], quantiles, dim=0)
    torch.testing.assert_close(result.cpu(), expected)


@pytest.mark.parametrize("shape, dim", [((), 1), ((2, 3), -3), ((0, 3), 2)])
def test_sort_invalid_dim(dgx_setup_teardown, shape, dim):
    tensor = torch.empty(shape, dtype=torch.int64, device="dgx")
    with pytest.raises(IndexError, match="Dimension out of range"):
        torch.sort(tensor, dim=dim)


@pytest.mark.parametrize("descending", [False, True])
def test_stable_sort_strided_backward(dgx_setup_teardown, descending):
    source = torch.tensor([[2.0, 1.0, 2.0, 1.0, 2.0], [1.0, 2.0, 1.0, 2.0, 1.0]]).t()
    tensor = source.to("dgx").detach().requires_grad_()
    values, indices = tensor.sort(dim=0, stable=True, descending=descending)
    check_sort(source, values, indices, dim=0, descending=descending, stable=True)
    weights = torch.arange(source.numel(), dtype=source.dtype).reshape(source.shape)
    (values * weights.to("dgx")).sum().backward()
    expected_grad = torch.zeros_like(source).scatter_(0, indices.cpu(), weights)
    torch.testing.assert_close(tensor.grad.cpu(), expected_grad)
