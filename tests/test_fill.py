# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
import torch
import pytest

datatypes = [torch.int32, torch.int64, torch.float16, torch.float32, torch.float64]


@pytest.mark.parametrize("dtype", datatypes, ids=str)
def test_empty(dgx_setup_teardown, dtype):
    tensor = torch.empty(7, dtype=dtype, device="dgx")
    assert tensor.dtype == dtype
    assert tensor.shape == torch.Size([7])


@pytest.mark.parametrize("dtype", datatypes + [torch.bfloat16], ids=str)
@pytest.mark.parametrize(
    "storage_shape,permutation",
    [((10, 2), None), ((2, 4, 3, 5), (0, 2, 3, 1))],
    ids=["contiguous", "noncontiguous-rank4"],
)
def test_full(dgx_setup_teardown, dtype, storage_shape, permutation):
    storage = torch.full(storage_shape, 3.2, dtype=dtype, device="dgx")
    tensor = storage if permutation is None else storage.permute(permutation)
    if permutation is not None:
        assert not tensor.is_contiguous()
    tensor.fill_(-3.7)
    torch.testing.assert_close(
        tensor.to("cpu"),
        torch.full(tensor.shape, -3.7, dtype=dtype, device="cpu"),
    )


@pytest.mark.parametrize("dtype", datatypes, ids=str)
def test_masked_fill_scalar(dgx_setup_teardown, dtype):
    tensor_cpu = torch.arange(12, dtype=dtype).reshape(3, 4)
    mask_cpu = tensor_cpu.remainder(3).eq(0)

    tensor = tensor_cpu.to("dgx")
    mask = mask_cpu.to("dgx")
    result = tensor.masked_fill_(mask, -5)

    assert result.data_ptr() == tensor.data_ptr()
    assert torch.equal(result.to("cpu"), tensor_cpu.masked_fill(mask_cpu, -5))


@pytest.mark.parametrize("dtype", datatypes, ids=str)
@pytest.mark.parametrize(
    "storage_shape,permutation,mask_shape",
    [
        ((2, 3, 4), None, (3, 1)),
        (
            (2, 3, 2, 2, 1, 2, 1, 2, 2),
            (0, 2, 3, 5, 7, 8, 1, 4, 6),
            (1, 2, 1, 2, 1, 2, 1, 1, 1),
        ),
    ],
    ids=["rank3", "noncontiguous-rank9"],
)
def test_masked_fill_scalar_broadcast_mask(
    dgx_setup_teardown, dtype, storage_shape, permutation, mask_shape
):
    storage_cpu = torch.arange(
        torch.tensor(storage_shape).prod().item(), dtype=dtype
    ).reshape(storage_shape)
    tensor_cpu = (
        storage_cpu if permutation is None else storage_cpu.permute(permutation)
    )
    mask_cpu = (
        torch.arange(torch.tensor(mask_shape).prod().item()).reshape(mask_shape) % 2
        == 0
    )

    storage = storage_cpu.to("dgx")
    tensor = storage if permutation is None else storage.permute(permutation)
    mask = mask_cpu.to("dgx")
    if permutation is not None:
        assert not tensor.is_contiguous()
    result = tensor.masked_fill_(mask, 99)

    assert result.data_ptr() == tensor.data_ptr()
    assert torch.equal(result.to("cpu"), tensor_cpu.masked_fill(mask_cpu, 99))


@pytest.mark.parametrize("dtype", datatypes, ids=str)
def test_ones(dgx_setup_teardown, dtype):
    tensor = torch.ones(31, dtype=dtype, device="dgx")
    assert torch.all(
        torch.eq(tensor.to("cpu"), torch.ones(31, dtype=dtype, device="cpu"))
    )
    assert tensor.shape == torch.Size([31])


@pytest.mark.parametrize("dtype", datatypes, ids=str)
def test_zeros(dgx_setup_teardown, dtype):
    tensor = torch.zeros(31, dtype=dtype, device="dgx")
    assert torch.all(
        torch.eq(tensor.to("cpu"), torch.zeros(31, dtype=dtype, device="cpu"))
    )
    assert tensor.shape == torch.Size([31])
