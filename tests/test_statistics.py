# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
import warnings

import pytest
import torch


def var_mean_with_warnings(*args, **kwargs):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = torch.var_mean(*args, **kwargs)
    return result, caught


def normalized_warnings(caught):
    return [
        (
            warning.category,
            str(warning.message).partition(" (Triggered internally")[0],
        )
        for warning in caught
    ]


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64], ids=str)
@pytest.mark.parametrize(
    "dim,keepdim",
    [
        (None, False),
        ([], True),
        ([-1], False),
        ([1], True),
        ([0, 2], False),
    ],
)
def test_var_mean_correction(dgx_setup_teardown, dtype, dim, keepdim):
    samples = torch.randn((5, 7, 3), dtype=dtype)

    actual_variance, actual_mean = torch.var_mean(
        samples.to("dgx"), dim=dim, correction=1.5, keepdim=keepdim
    )
    expected_variance, expected_mean = torch.var_mean(
        samples, dim=dim, correction=1.5, keepdim=keepdim
    )

    torch.testing.assert_close(
        actual_variance.to("cpu"), expected_variance, rtol=1e-3, atol=1e-3
    )
    torch.testing.assert_close(
        actual_mean.to("cpu"), expected_mean, rtol=1e-3, atol=1e-3
    )


def test_var_mean_correction_default_and_noncontiguous(dgx_setup_teardown):
    samples = torch.randn((4, 5, 6), dtype=torch.float64).transpose(0, 2)

    actual = torch.var_mean(samples.to("dgx"), dim=(0, -1))
    expected = torch.var_mean(samples, dim=(0, -1))

    torch.testing.assert_close(actual[0].to("cpu"), expected[0])
    torch.testing.assert_close(actual[1].to("cpu"), expected[1])


@pytest.mark.parametrize("correction", [0, 1, -1, 10])
def test_var_mean_correction_edge_cases(dgx_setup_teardown, correction):
    samples = torch.tensor([[1.0], [3.0]], dtype=torch.float32)

    actual, actual_warnings = var_mean_with_warnings(
        samples.to("dgx"), dim=1, correction=correction
    )
    expected, expected_warnings = var_mean_with_warnings(
        samples, dim=1, correction=correction
    )

    torch.testing.assert_close(actual[0].to("cpu"), expected[0], equal_nan=True)
    torch.testing.assert_close(actual[1].to("cpu"), expected[1], equal_nan=True)
    assert normalized_warnings(actual_warnings) == normalized_warnings(
        expected_warnings
    )


@pytest.mark.parametrize("shape", [(2, 0, 3), (0, 3)])
def test_var_mean_empty_reduction(dgx_setup_teardown, shape):
    samples = torch.empty(shape, dtype=torch.float32)

    actual, actual_warnings = var_mean_with_warnings(
        samples.to("dgx"), dim=1, correction=0
    )
    expected, expected_warnings = var_mean_with_warnings(samples, dim=1, correction=0)

    torch.testing.assert_close(actual[0].to("cpu"), expected[0], equal_nan=True)
    torch.testing.assert_close(actual[1].to("cpu"), expected[1], equal_nan=True)
    assert normalized_warnings(actual_warnings) == normalized_warnings(
        expected_warnings
    )


def test_var_mean_more_than_25_dimensions(dgx_setup_teardown):
    samples = torch.randn((1,) * 26, dtype=torch.float32)

    actual = torch.var_mean(samples.to("dgx"), dim=(0, -1), correction=0)
    expected = torch.var_mean(samples, dim=(0, -1), correction=0)

    assert torch.equal(actual[0].to("cpu"), expected[0])
    assert torch.equal(actual[1].to("cpu"), expected[1])
