# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
import pytest
import torch


def _fill_with_distribution(tensor, distribution, generator):
    if distribution == "uniform":
        tensor.uniform_(-1.0, 2.0, generator=generator)
    elif distribution == "normal":
        tensor.normal_(0.5, 1.5, generator=generator)
    else:
        tensor.random_(-17, 23, generator=generator)


@pytest.mark.parametrize(
    "dtype", [torch.float16, torch.bfloat16, torch.float32, torch.float64], ids=str
)
def test_uniform_and_deterministic(dgx_setup_teardown, dtype):
    first_generator = torch.Generator(device="dgx").manual_seed(1234)
    second_generator = torch.Generator(device="dgx").manual_seed(1234)

    first = torch.empty(4097, dtype=dtype, device="dgx").uniform_(
        -2.5, 3.75, generator=first_generator
    )
    second = torch.empty(4097, dtype=dtype, device="dgx").uniform_(
        -2.5, 3.75, generator=second_generator
    )

    first_cpu = first.cpu()
    assert torch.equal(first_cpu, second.cpu())
    assert torch.all(first_cpu >= -2.5)
    assert torch.all(first_cpu < 3.75)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64], ids=str)
def test_normal_has_expected_moments(dgx_setup_teardown, dtype):
    samples = torch.empty(100_000, dtype=dtype, device="dgx").normal_(
        mean=1.25, std=2.5
    )

    samples_cpu = samples.cpu()
    assert samples_cpu.mean().item() == pytest.approx(1.25, abs=0.03)
    assert samples_cpu.std().item() == pytest.approx(2.5, abs=0.03)


@pytest.mark.parametrize("distribution", ["uniform", "normal", "random_from"])
def test_random_generator_state_advances(dgx_setup_teardown, distribution):
    generator = torch.Generator(device="dgx").manual_seed(5678)
    first = torch.empty(257, device="dgx")
    second = torch.empty(257, device="dgx")

    initial_state = generator.get_state()

    _fill_with_distribution(first, distribution, generator)
    advanced_state = generator.get_state()
    _fill_with_distribution(second, distribution, generator)

    assert not torch.equal(first.cpu(), second.cpu())

    replay = torch.empty_like(second)
    generator.set_state(advanced_state)
    _fill_with_distribution(replay, distribution, generator)
    assert torch.equal(second.cpu(), replay.cpu())

    generator.set_state(initial_state)
    assert generator.get_offset() == 0


@pytest.mark.parametrize("distribution", ["uniform", "normal", "random_from"])
@pytest.mark.parametrize(
    "logical_shape,storage_shape,permutation",
    [
        ((5, 7), (7, 5), (1, 0)),
        ((2, 3, 5, 7, 4), (2, 4, 3, 5, 7), (0, 2, 3, 4, 1)),
    ],
    ids=["rank2", "rank5"],
)
def test_random_noncontiguous_matches_logical_order(
    dgx_setup_teardown,
    distribution,
    logical_shape,
    storage_shape,
    permutation,
):
    contiguous = torch.empty(logical_shape, device="dgx")
    noncontiguous = torch.empty(storage_shape, device="dgx").permute(permutation)
    assert not noncontiguous.is_contiguous()
    first_generator = torch.Generator(device="dgx").manual_seed(91011)
    second_generator = torch.Generator(device="dgx").manual_seed(91011)

    _fill_with_distribution(contiguous, distribution, first_generator)
    _fill_with_distribution(noncontiguous, distribution, second_generator)

    assert torch.equal(contiguous.cpu(), noncontiguous.cpu())


@pytest.mark.parametrize(
    "dtype,from_,to",
    [
        (torch.bool, 0, 2),
        (torch.uint8, 3, 251),
        (torch.int8, -101, 101),
        (torch.int16, -1000, 1001),
        (torch.int32, -(1 << 30), 1 << 30),
        (torch.int64, -(1 << 40), 1 << 40),
        (torch.float16, -1000, 1001),
        (torch.bfloat16, -(1 << 8), (1 << 8) + 1),
        (torch.float32, -(1 << 24), (1 << 24) + 1),
        (torch.float64, -(1 << 40), 1 << 40),
    ],
    ids=lambda value: str(value),
)
def test_random_from_to_is_deterministic(dgx_setup_teardown, dtype, from_, to):
    first_generator = torch.Generator(device="dgx").manual_seed(2468)
    second_generator = torch.Generator(device="dgx").manual_seed(2468)

    first = torch.empty(4097, dtype=dtype, device="dgx").random_(
        from_, to, generator=first_generator
    )
    second = torch.empty(4097, dtype=dtype, device="dgx").random_(
        from_, to, generator=second_generator
    )

    first_cpu = first.cpu()
    assert torch.equal(first_cpu, second.cpu())
    comparison_values = first_cpu.float() if dtype == torch.bfloat16 else first_cpu
    assert torch.all(comparison_values >= from_)
    assert torch.all(comparison_values < to)


@pytest.mark.parametrize("dtype", [torch.int32, torch.int64, torch.float32], ids=str)
def test_randperm_is_deterministic(dgx_setup_teardown, dtype):
    first_generator = torch.Generator(device="dgx").manual_seed(97531)
    second_generator = torch.Generator(device="dgx").manual_seed(97531)

    first = torch.randperm(4097, dtype=dtype, device="dgx", generator=first_generator)
    second = torch.randperm(4097, dtype=dtype, device="dgx", generator=second_generator)

    first_cpu = first.cpu()
    assert torch.equal(first_cpu, second.cpu())
    assert torch.equal(first_cpu.sort().values, torch.arange(4097, dtype=dtype))


def test_randperm_bfloat16_precision_limit(dgx_setup_teardown):
    valid = torch.randperm(257, dtype=torch.bfloat16, device="dgx")

    assert torch.equal(
        valid.cpu().sort().values, torch.arange(257, dtype=torch.bfloat16)
    )
    with pytest.raises(RuntimeError, match="greater than 257 for BFloat16"):
        torch.randperm(258, dtype=torch.bfloat16, device="dgx")


def test_random_from_full_int64_range(dgx_setup_teardown):
    generator = torch.Generator(device="dgx").manual_seed(1357)
    initial_state = generator.get_state()

    random_from = getattr(torch.ops.aten.random_, "from")
    first = random_from(
        torch.empty(4097, dtype=torch.int64, device="dgx"),
        torch.iinfo(torch.int64).min,
        None,
        generator=generator,
    )
    generator.set_state(initial_state)
    second = random_from(
        torch.empty_like(first),
        torch.iinfo(torch.int64).min,
        None,
        generator=generator,
    )

    assert torch.equal(first.cpu(), second.cpu())


@pytest.mark.parametrize(
    "distribution,dtype",
    [
        pytest.param("normal", torch.int32, id="normal-int32"),
        pytest.param("normal", torch.int64, id="normal-int64"),
        pytest.param("full-range", torch.int32, id="full-range-int32"),
        pytest.param("full-range", torch.float16, id="full-range-float16"),
    ],
)
def test_random_error_preserves_generator_state(
    dgx_setup_teardown, distribution, dtype
):
    generator = torch.Generator(device="dgx").manual_seed(8642)
    initial_state = generator.get_state()
    output = torch.empty(17, dtype=dtype, device="dgx")

    with pytest.raises(RuntimeError):
        if distribution == "normal":
            output.normal_(generator=generator)
        else:
            random_from = getattr(torch.ops.aten.random_, "from")
            random_from(
                output,
                torch.iinfo(torch.int64).min,
                None,
                generator=generator,
            )

    assert torch.equal(generator.get_state(), initial_state)


def test_dgx_generator_interface_and_cpu_generator_rejection(dgx_setup_teardown):
    generator = torch.Generator(device="dgx").manual_seed(123)

    assert generator.device == torch.device("dgx:0")
    assert generator.initial_seed() == 123
    assert generator.get_offset() == 0
    assert generator.get_state().shape == (16,)

    torch.empty(1, device="dgx").uniform_(generator=generator)
    assert generator.get_offset() == 4

    cloned = generator.clone_state()
    assert cloned.device == generator.device
    assert torch.equal(cloned.get_state(), generator.get_state())

    with pytest.raises(RuntimeError, match="16 bytes"):
        generator.set_state(torch.zeros(15, dtype=torch.uint8))

    with pytest.raises(RuntimeError, match="device type for generator"):
        torch.empty(1, device="dgx").uniform_(generator=torch.Generator())


@pytest.mark.parametrize(
    "make_state,error,message",
    [
        pytest.param(
            lambda: torch.empty(0, dtype=torch.uint8),
            RuntimeError,
            "16 bytes",
            id="empty",
        ),
        pytest.param(
            lambda: torch.empty(15, dtype=torch.uint8),
            RuntimeError,
            "16 bytes",
            id="undersized",
        ),
        pytest.param(
            lambda: torch.empty(17, dtype=torch.uint8),
            RuntimeError,
            "16 bytes",
            id="oversized",
        ),
        pytest.param(
            lambda: torch.zeros(1, dtype=torch.uint8).expand(1 << 30),
            RuntimeError,
            "16 bytes",
            id="large-expanded-view",
        ),
        pytest.param(
            lambda: torch.zeros(16, dtype=torch.float32),
            TypeError,
            "torch.ByteTensor",
            id="wrong-dtype",
        ),
        pytest.param(
            lambda: torch.empty(16, dtype=torch.uint8, device="meta"),
            TypeError,
            "torch.ByteTensor",
            id="non-cpu",
        ),
        pytest.param(
            lambda: torch.zeros(16, dtype=torch.uint8).to_sparse(),
            TypeError,
            "torch.ByteTensor",
            id="sparse",
        ),
        pytest.param(
            lambda: bytes(16),
            TypeError,
            "torch.ByteTensor",
            id="not-a-tensor",
        ),
    ],
)
def test_set_rng_state_rejects_invalid_before_clone(
    dgx_setup_teardown, monkeypatch, make_state, error, message
):
    initial_state = torch.dgx.get_rng_state()
    invalid_state = make_state()

    def unexpected_clone(*args, **kwargs):
        pytest.fail("Invalid RNG state was cloned before validation")

    monkeypatch.setattr(torch.Tensor, "clone", unexpected_clone)
    with pytest.raises(error, match=message):
        torch.dgx.set_rng_state(invalid_state)

    assert torch.equal(torch.dgx.get_rng_state(), initial_state)


def test_set_rng_state_accepts_noncontiguous_state(dgx_setup_teardown):
    initial_state = torch.dgx.get_rng_state()
    storage = torch.zeros(initial_state.numel() * 2, dtype=torch.uint8)
    state = storage[::2]
    state.copy_(initial_state)
    assert not state.is_contiguous()

    torch.dgx.manual_seed(123)
    torch.dgx.set_rng_state(state)

    assert torch.equal(torch.dgx.get_rng_state(), initial_state)


def test_default_generator_state_is_independent_from_cpu(dgx_setup_teardown):
    torch.manual_seed(2024)
    dgx_state = torch.dgx.get_rng_state()
    cpu_state = torch.get_rng_state()
    first = torch.empty(257, device="dgx").normal_()

    assert not torch.equal(torch.dgx.get_rng_state(), dgx_state)
    assert torch.equal(torch.get_rng_state(), cpu_state)

    torch.dgx.set_rng_state(dgx_state)
    replay = torch.empty(257, device="dgx").normal_()
    assert torch.equal(first.cpu(), replay.cpu())
    assert torch.dgx.default_generators[0].initial_seed() == 2024

    state_before_fork = torch.dgx.get_rng_state()
    with torch.random.fork_rng(devices=[0], device_type="dgx"):
        torch.dgx.manual_seed(99)
        torch.empty(1, device="dgx").uniform_()
        assert not torch.equal(torch.dgx.get_rng_state(), state_before_fork)
    assert torch.equal(torch.dgx.get_rng_state(), state_before_fork)
