# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
import pytest
import torch


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64], ids=str)
def test_multinomial_with_replacement(dgx_setup_teardown, dtype):
    weights = torch.tensor([0.0, 0.0, 4.0, 0.0], dtype=dtype, device="dgx")
    samples = torch.multinomial(weights, 97, replacement=True)

    assert samples.device.type == "dgx"
    assert samples.dtype == torch.int64
    assert torch.equal(samples.to("cpu"), torch.full((97,), 2, dtype=torch.int64))


def test_multinomial_with_replacement_distribution(dgx_setup_teardown):
    weights = torch.tensor([1.0, 3.0, 6.0], device="dgx")
    samples = torch.multinomial(weights, 20_000, replacement=True)
    samples_cpu = samples.cpu()
    frequencies = samples_cpu.bincount(minlength=3) / samples_cpu.numel()

    torch.testing.assert_close(
        frequencies, torch.tensor([0.1, 0.3, 0.6]), atol=0.02, rtol=0
    )


def test_multinomial_without_replacement_compatibility_path(dgx_setup_teardown):
    weights = torch.tensor([1.0, 3.0, 6.0], device="dgx")
    samples = torch.multinomial(weights, 3, replacement=False)

    assert samples.device.type == "dgx"
    assert samples.to("cpu").unique().numel() == 3


@pytest.mark.parametrize(
    "num_categories,replacement",
    [
        pytest.param(0, True, id="zero-categories-with-replacement"),
        pytest.param(3, True, id="with-replacement"),
        pytest.param(3, False, id="without-replacement"),
    ],
)
def test_multinomial_empty_batch_preserves_generator_state(
    dgx_setup_teardown, num_categories, replacement
):
    generator = torch.Generator(device="dgx").manual_seed(2468)
    initial_state = generator.get_state()
    weights = torch.empty((0, num_categories), device="dgx")

    samples = torch.multinomial(
        weights, 2, replacement=replacement, generator=generator
    )

    assert samples.shape == (0, 2)
    assert samples.device.type == "dgx"
    assert torch.equal(generator.get_state(), initial_state)


@pytest.mark.parametrize("dtype", [torch.int32, torch.int64], ids=str)
def test_bincount(dgx_setup_teardown, dtype):
    values = torch.tensor([0, 5, 2, 5, 2, 5, 9, 2], dtype=dtype)
    result = torch.bincount(values.to("dgx"), minlength=13)

    assert result.device.type == "dgx"
    assert torch.equal(result.to("cpu"), torch.bincount(values, minlength=13))


@pytest.mark.parametrize(
    "weights_dtype", [torch.float32, torch.float64, torch.int64], ids=str
)
def test_bincount_weights(dgx_setup_teardown, weights_dtype):
    values = torch.tensor([0, 5, 2, 5, 2, 5, 9, 2], dtype=torch.int64)
    weights = torch.arange(1, values.numel() + 1, dtype=weights_dtype)

    result = torch.bincount(values.to("dgx"), weights=weights.to("dgx"), minlength=13)
    expected = torch.bincount(values, weights=weights, minlength=13)

    assert result.dtype == expected.dtype
    assert torch.equal(result.to("cpu"), expected)


def test_bincount_empty(dgx_setup_teardown):
    result = torch.bincount(
        torch.empty(0, dtype=torch.int64, device="dgx"), minlength=7
    )

    assert torch.equal(result.to("cpu"), torch.zeros(7, dtype=torch.int64))


def test_bincount_dense_hot_bins(dgx_setup_teardown):
    values = torch.arange(20_003, dtype=torch.int64) % 4
    weights = torch.linspace(0.5, 1.5, values.numel(), dtype=torch.float64)

    counts = torch.bincount(values.to("dgx"), minlength=4)
    weighted_counts = torch.bincount(
        values.to("dgx"), weights=weights.to("dgx"), minlength=4
    )

    assert torch.equal(counts.to("cpu"), torch.bincount(values, minlength=4))
    torch.testing.assert_close(
        weighted_counts.to("cpu"),
        torch.bincount(values, weights=weights, minlength=4),
    )


def test_bincount_page_owned_sparse_output(dgx_setup_teardown):
    bins_per_page = (2 * 1024 * 1024) // torch.tensor(
        [], dtype=torch.int64
    ).element_size()
    minlength = 2 * bins_per_page + 11
    values = torch.tensor(
        [
            0,
            bins_per_page - 1,
            bins_per_page,
            2 * bins_per_page - 1,
            2 * bins_per_page,
            minlength - 1,
            minlength - 1,
        ],
        dtype=torch.int64,
    )

    result = torch.bincount(values.to("dgx"), minlength=minlength)

    assert torch.equal(result.to("cpu"), torch.bincount(values, minlength=minlength))


@pytest.mark.parametrize("replacement", [True, False], ids=["native", "compatibility"])
def test_multinomial_generator_is_deterministic(dgx_setup_teardown, replacement):
    weights = torch.ones(257, device="dgx")
    first_generator = torch.Generator(device="dgx").manual_seed(1234)
    second_generator = torch.Generator(device="dgx").manual_seed(1234)
    cpu_state = torch.get_rng_state()

    first_samples = torch.multinomial(
        weights, 129, replacement=replacement, generator=first_generator
    )
    second_samples = torch.multinomial(
        weights, 129, replacement=replacement, generator=second_generator
    )

    assert torch.equal(first_samples.cpu(), second_samples.cpu())
    assert torch.equal(first_generator.get_state(), second_generator.get_state())
    assert torch.equal(torch.get_rng_state(), cpu_state)


@pytest.mark.parametrize("replacement", [True, False], ids=["native", "compatibility"])
def test_multinomial_default_generator_state_replays(dgx_setup_teardown, replacement):
    weights = torch.ones(257, device="dgx")
    torch.dgx.manual_seed(4321)
    initial_state = torch.dgx.get_rng_state()

    first_samples = torch.multinomial(weights, 129, replacement=replacement)
    advanced_state = torch.dgx.get_rng_state()
    torch.dgx.set_rng_state(initial_state)
    replay_samples = torch.multinomial(weights, 129, replacement=replacement)

    assert not torch.equal(advanced_state, initial_state)
    assert torch.equal(first_samples.cpu(), replay_samples.cpu())
