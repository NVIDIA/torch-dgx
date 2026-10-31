# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
import torch
import pytest


def check_sort(source, values, indices):
    values, indices = values.cpu(), indices.cpu()
    assert torch.equal(values, source.sort().values)
    assert torch.equal(indices.sort().values, torch.arange(source.numel()))
    assert torch.equal(values, source[indices])


@pytest.mark.parametrize("size", [0, 1, 3, 43, 65536, 65537, 262147])
def test_argsort(dgx_setup_teardown, size):
    generator = torch.Generator().manual_seed(42)
    source = torch.randint(-421, 421, (size,), generator=generator)
    tensor = source.to("dgx")
    indices = torch.argsort(tensor)
    check_sort(source, tensor[indices], indices)
    assert torch.equal(tensor.cpu(), source)


@pytest.mark.parametrize(
    "pattern",
    ["random", "equal", "skewed", "sorted", "reversed", "extremes", "periodic"],
)
def test_sort_partition_distributions(dgx_setup_teardown, pattern):
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
    values, indices = torch.sort(tensor, dim=0, stable=False)
    check_sort(source, values, indices)
    assert torch.equal(tensor.cpu(), source)


@pytest.mark.parametrize("size", [43, 131075])
def test_sort_strided_input_and_outputs(dgx_setup_teardown, size):
    source_storage = torch.randint(-1000000, 1000000, (size * 2 + 3,))
    source = source_storage[1 : 1 + size * 2 : 2]
    tensor = source_storage.to("dgx")[1 : 1 + size * 2 : 2]
    values_storage = torch.full(
        (size * 2 + 3,), -123456789, device="dgx", dtype=torch.int64
    )
    indices_storage = torch.full_like(values_storage, -123456789)
    values = values_storage[1 : 1 + size * 2 : 2]
    indices = indices_storage[1 : 1 + size * 2 : 2]
    torch.sort(tensor, stable=False, out=(values, indices))
    check_sort(source, values, indices)
    for storage in (values_storage, indices_storage):
        host = storage.cpu()
        assert torch.all(host[::2] == -123456789)
        assert host[-2] == -123456789


def test_sort_out_resize_and_alias(dgx_setup_teardown):
    source = torch.randint(-(1 << 62), 1 << 62, (131075,))
    tensor = source.to("dgx")
    indices = torch.empty(0, device="dgx", dtype=torch.int64)
    torch.sort(tensor, stable=False, out=(tensor, indices))
    check_sort(source, tensor, indices)
