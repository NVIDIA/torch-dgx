# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: BSD-3-Clause
import pytest
import torch


@pytest.mark.parametrize(
    "dtype,value",
    [
        (torch.bool, True),
        (torch.uint8, 7),
        (torch.int8, -7),
        (torch.int16, -17),
        (torch.int32, -123456),
        (torch.int64, 2**40 + 3),
        (torch.float16, 1.5),
        (torch.bfloat16, 1.5),
        (torch.float32, -2.25),
        (torch.float64, 1.125),
        (torch.complex64, 1.5 - 2.25j),
        (torch.complex128, 1.5 - 2.25j),
    ],
)
def test_item_after_copy(dgx_setup_teardown, dtype, value):
    expected = torch.tensor(value, dtype=dtype)
    actual = expected.to("dgx").item()

    assert actual == expected.item()
    assert type(actual) is type(expected.item())


@pytest.mark.parametrize("reduction", ["sum", "mean", "min", "max", "any", "all"])
def test_item_after_reduction(dgx_setup_teardown, reduction):
    tensor = torch.ones(65_537, device="dgx")
    expected = 65_537 if reduction == "sum" else 1

    assert getattr(tensor, reduction)().item() == expected
