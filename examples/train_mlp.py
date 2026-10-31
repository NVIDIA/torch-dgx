# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
import tyro
import torch
import pytest
import numpy


@torch._dynamo.disable
def main(device="dgx"):
    numpy.random.seed(42)
    torch.random.manual_seed(42)

    if device == "dgx":
        import torch_dgx

    model = torch.nn.Sequential(
        torch.nn.Linear(1, 64), torch.nn.ReLU(), torch.nn.Linear(64, 1)
    ).to(device)

    n_samples = 100
    x_train = numpy.random.uniform(-3, 3, size=(n_samples, 1)).astype("float32")
    y_train = x_train**2
    x_train = torch.tensor(x_train, device=device)
    y_train = torch.tensor(y_train, device=device)

    torch.cuda.profiler.cudart().cudaProfilerStart()
    loss_fn = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.1, fused=True)
    for i in range(10):
        torch.cuda.nvtx.range_push(f"Iteration {i}")
        outputs = model(x_train)
        loss = loss_fn(outputs, y_train)
        # for device_id in range(torch.cuda.device_count()):
        #     torch.cuda.synchronize(device_id)
        if i == 9:
            print(f"Iteration {i} Loss = {loss.item()}")
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # for device_id in range(torch.cuda.device_count()):
        #     torch.cuda.synchronize(device_id)
        torch.cuda.nvtx.range_pop()
    torch.cuda.profiler.cudart().cudaProfilerStop()


if __name__ == "__main__":
    tyro.cli(main)
