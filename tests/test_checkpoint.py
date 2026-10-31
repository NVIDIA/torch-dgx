# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
import pytest
import torch
from torch.utils.checkpoint import checkpoint


@pytest.mark.parametrize("use_reentrant", [False, True])
@pytest.mark.parametrize("preserve_rng_state", [True, False])
def test_checkpoint_rng_state(dgx_setup_teardown, use_reentrant, preserve_rng_state):
    input = torch.ones(257, device="dgx", requires_grad=True)
    initial_state = torch.dgx.get_rng_state()
    entry_states = []
    samples = []

    def random_scale(x):
        entry_states.append(torch.dgx.get_rng_state())
        sample = torch.rand_like(x)
        samples.append(sample)
        # Backward needs the random multiplier, forcing checkpoint recomputation.
        return x * sample

    # Exercise the default preservation behavior without passing True.
    kwargs = {} if preserve_rng_state else {"preserve_rng_state": False}
    output = checkpoint(random_scale, input, use_reentrant=use_reentrant, **kwargs)
    assert len(samples) == 1
    assert torch.equal(entry_states[0], initial_state)
    assert not torch.equal(torch.dgx.get_rng_state(), initial_state)

    # Simulate unrelated RNG use between forward and backward. Record the next
    # draw and rewind so we can check that recomputation preserves the sequence.
    torch.rand(113, device="dgx")
    before_backward = torch.dgx.get_rng_state()
    expected_next = torch.rand_like(input)
    torch.dgx.set_rng_state(before_backward)

    output.sum().backward()
    after_backward = torch.dgx.get_rng_state()
    actual_next = torch.rand_like(input)
    assert len(samples) == 2
    assert torch.equal(output.detach().cpu(), samples[0].cpu())
    assert torch.equal(input.grad.cpu(), samples[1].cpu())

    if preserve_rng_state:
        assert torch.equal(entry_states[1], initial_state)
        assert torch.equal(samples[0].cpu(), samples[1].cpu())
        assert torch.equal(after_backward, before_backward)
        assert torch.equal(actual_next.cpu(), expected_next.cpu())
    else:
        assert torch.equal(entry_states[1], before_backward)
        assert not torch.equal(samples[0].cpu(), samples[1].cpu())
        assert not torch.equal(after_backward, before_backward)
        assert not torch.equal(actual_next.cpu(), expected_next.cpu())
