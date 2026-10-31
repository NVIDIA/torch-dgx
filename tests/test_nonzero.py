# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
import pytest
import torch


@pytest.mark.parametrize(
    "dtype",
    [torch.float32, torch.float64, torch.bool, torch.int32, torch.int64],
    ids=str,
)
def test_nonzero(dgx_setup_teardown, dtype):
    shapes = [(7,), (2, 7), (2, 3, 4), (2, 3, 4, 2), (1, 2, 1, 2, 1, 2, 2)]
    for shape in shapes:
        samples_cpu = (
            ((torch.arange(torch.Size(shape).numel()) + 1) % 3).reshape(shape).to(dtype)
        )
        samples = samples_cpu.to("dgx")

        result = torch.nonzero(samples)
        expected = torch.nonzero(samples_cpu)
        assert torch.equal(result.cpu(), expected)

        result_tuple = torch.nonzero(samples, as_tuple=True)
        expected_tuple = torch.nonzero(samples_cpu, as_tuple=True)
        assert all(
            torch.equal(actual.cpu(), expected)
            for actual, expected in zip(result_tuple, expected_tuple, strict=True)
        )
