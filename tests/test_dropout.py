# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
import pytest
import torch


@pytest.mark.parametrize(
    "dtype", [torch.float16, torch.bfloat16, torch.float32, torch.float64], ids=str
)
def test_native_dropout_forward_backward(dgx_setup_teardown, dtype):
    probability = 0.25
    input = torch.randn(31, 47, dtype=dtype, device="dgx", requires_grad=True)
    grad_output = torch.randn_like(input)

    output, mask = torch.ops.aten.native_dropout(input, probability, True)
    output.backward(grad_output)

    scale = 1.0 / (1.0 - probability)
    mask_cpu = mask.cpu()
    expected_output = input.detach().cpu() * mask_cpu * scale
    expected_grad = grad_output.cpu() * mask_cpu * scale
    assert mask.dtype == torch.bool
    assert mask_cpu.any()
    assert (~mask_cpu).any()
    torch.testing.assert_close(output.cpu(), expected_output)
    torch.testing.assert_close(input.grad.cpu(), expected_grad)


def test_native_dropout_eval(dgx_setup_teardown):
    input = torch.randn(7, 11, device="dgx")

    output, mask = torch.ops.aten.native_dropout(input, 0.5, False)

    assert torch.equal(output.cpu(), input.cpu())
    assert torch.all(mask.cpu())


def test_native_dropout_replays_default_generator_state(dgx_setup_teardown):
    input = torch.ones(257, device="dgx")
    torch.dgx.manual_seed(1234)
    initial_state = torch.dgx.get_rng_state()

    first_output, first_mask = torch.ops.aten.native_dropout(input, 0.25, True)
    advanced_state = torch.dgx.get_rng_state()
    torch.dgx.set_rng_state(initial_state)
    replay_output, replay_mask = torch.ops.aten.native_dropout(input, 0.25, True)

    assert not torch.equal(advanced_state, initial_state)
    assert torch.equal(first_output.cpu(), replay_output.cpu())
    assert torch.equal(first_mask.cpu(), replay_mask.cpu())
