# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
import pytest
import torch


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64], ids=str)
@pytest.mark.parametrize(
    "name,weight_decay", [("adam", 0.0), ("adamw", 0.0), ("adamw", 0.1)]
)
@pytest.mark.parametrize(
    "lr,betas,eps",
    [(0.003, (0.9, 0.999), 1e-8), (0.1, (0.5, 0.9), 1e-4), (0.0, (0.9, 0.999), 1e-8)],
)
def test_fused_adam_matches_cpu(dgx_setup_teardown, dtype, name, weight_decay, lr, betas, eps):
    # Include empty tensors, scalar/vector tails, and a tensor spanning GPU shards.
    shapes = [(0,), (), (3,), (11, 7), (2**20 + 3,)]
    initial = [
        [torch.randn(shape, dtype=dtype) * 0.1 for shape in shapes]
        for _ in range(3)
    ]
    initial.append([torch.rand(shape, dtype=dtype) * 0.01 for shape in shapes])
    for grad in initial[1]:
        grad.reshape(-1)[::17] = 0
    initial_steps = [0, 1, 10, 1000, 65536]
    operands = {
        device: [[tensor.to(device) for tensor in tensors] for tensors in initial]
        for device in ("cpu", "dgx")
    }
    steps = {
        device: [torch.tensor(float(step), device=device) for step in initial_steps]
        for device in operands
    }
    op = getattr(torch, f"_fused_{name}_")
    assert torch._C._dispatch_has_kernel_for_dispatch_key(
        f"aten::_fused_{name}_", "PrivateUse1"
    )
    # CPU rounds 1 - beta before the moment update; DGX uses the rounded beta
    # in fused multiply-adds. Allow the resulting FP32 rounding differences.
    tolerances = {"rtol": 1e-5, "atol": 1e-5} if dtype == torch.float32 else {}

    for iteration in range(3):
        for device, (params, grads, exp_avgs, exp_avg_sqs) in operands.items():
            for step, grad in zip(steps[device], grads):
                step.add_(1)
                grad.mul_(-0.5)
            op(
                params, grads, exp_avgs, exp_avg_sqs, [], steps[device],
                lr=lr, beta1=betas[0], beta2=betas[1], weight_decay=weight_decay,
                eps=eps, amsgrad=False, maximize=False,
            )

        for actual_list, expected_list in zip(operands["dgx"], operands["cpu"]):
            for actual, expected in zip(actual_list, expected_list):
                torch.testing.assert_close(actual.cpu(), expected, **tolerances)
        for step, initial_step in zip(steps["dgx"], initial_steps):
            assert step.cpu().item() == initial_step + iteration + 1


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64], ids=str)
@pytest.mark.parametrize("lr", [0.0, 0.1])
def test_fused_adamw_zero_grad_weight_decay(dgx_setup_teardown, dtype, lr):
    param = torch.tensor([1.0, -2.0, 3.0], dtype=dtype, device="dgx")
    grad = torch.zeros_like(param)
    exp_avg = torch.zeros_like(param)
    exp_avg_sq = torch.zeros_like(param)
    expected = param.cpu()

    for step in range(1, 4):
        torch._fused_adamw_(
            [param], [grad], [exp_avg], [exp_avg_sq], [],
            [torch.tensor(float(step), device="dgx")],
            lr=lr, beta1=0.9, beta2=0.999, weight_decay=0.2,
            eps=1e-8, amsgrad=False, maximize=False,
        )
        expected.mul_(1 - lr * 0.2)
        torch.testing.assert_close(param.cpu(), expected)
        torch.testing.assert_close(exp_avg.cpu(), torch.zeros_like(expected))
        torch.testing.assert_close(exp_avg_sq.cpu(), torch.zeros_like(expected))
        torch.testing.assert_close(grad.cpu(), torch.zeros_like(expected))


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64], ids=str)
def test_fused_adamw_optimizer(dgx_setup_teardown, dtype):
    initial = [torch.randn(shape, dtype=dtype) for shape in [(17,), (3, 5), (1,)]]
    params = {
        device: [torch.nn.Parameter(tensor.to(device)) for tensor in initial]
        for device in ("cpu", "dgx")
    }
    optimizers = {
        device: torch.optim.AdamW(
            [{"params": tensors[:2]}, {"params": tensors[2:], "lr": 0.02, "weight_decay": 0.0}],
            fused=device == "dgx", foreach=False,
        )
        for device, tensors in params.items()
    }

    for iteration in range(4):
        grads = [torch.randn_like(tensor) for tensor in initial]
        for device, optimizer in optimizers.items():
            for index, (param, grad) in enumerate(zip(params[device], grads)):
                param.grad = None if index == 1 and iteration == 1 else grad.to(device)
            optimizer.step()

        for actual, expected in zip(params["dgx"], params["cpu"]):
            torch.testing.assert_close(actual.cpu(), expected.cpu())
            actual_state = optimizers["dgx"].state[actual]
            expected_state = optimizers["cpu"].state[expected]
            for key in ("step", "exp_avg", "exp_avg_sq"):
                torch.testing.assert_close(actual_state[key].cpu(), expected_state[key].cpu())


@pytest.mark.parametrize("name", ["adam", "adamw"])
def test_fused_adam_mixed_dtypes(dgx_setup_teardown, name):
    # Reuse per-device step scratch across dtype changes and empty shards.
    specs = [
        (17, torch.float32),
        (19, torch.float64),
        (0, torch.float32),
        (2**20 + 3, torch.float64),
        (3, torch.float32),
    ]
    initial = [
        [torch.randn(size, dtype=dtype) * 0.1 for size, dtype in specs]
        for _ in range(3)
    ]
    initial.append([torch.rand(size, dtype=dtype) * 0.01 for size, dtype in specs])
    operands = {
        device: [[tensor.to(device) for tensor in tensors] for tensors in initial]
        for device in ("cpu", "dgx")
    }
    steps = {
        device: [
            torch.tensor(float(step), device=device) for step in (0, 1, 10, 1000, 65536)
        ]
        for device in operands
    }
    op = getattr(torch, f"_fused_{name}_")
    kwargs = dict(
        lr=0.003,
        beta1=0.9,
        beta2=0.999,
        weight_decay=0.1 if name == "adamw" else 0.0,
        eps=1e-8,
        amsgrad=False,
        maximize=False,
    )

    for _ in range(3):
        for device in operands:
            torch._foreach_add_(steps[device], 1)
        # The CPU oracle takes homogeneous lists; DGX receives the mixed list.
        for i in range(len(specs)):
            op(
                *[[tensors[i]] for tensors in operands["cpu"]],
                [],
                [steps["cpu"][i]],
                **kwargs,
            )
        op(*operands["dgx"], [], steps["dgx"], **kwargs)

        for actual_list, expected_list in zip(operands["dgx"], operands["cpu"]):
            for actual, expected in zip(actual_list, expected_list):
                tolerance = (
                    dict(rtol=1e-5, atol=1e-5) if actual.dtype == torch.float32 else {}
                )
                torch.testing.assert_close(actual.cpu(), expected, **tolerance)
        for actual, expected in zip(steps["dgx"], steps["cpu"]):
            torch.testing.assert_close(actual.cpu(), expected)
