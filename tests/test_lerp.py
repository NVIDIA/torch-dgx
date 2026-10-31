# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
import torch
import pytest


datatypes = [torch.float32, torch.float64]


@pytest.mark.parametrize("dtype", datatypes, ids=str)
def test_lerp(dgx_setup_teardown, dtype):
    start = torch.arange(1.0, 5.0, device="dgx")
    end = torch.full((4,), 10.0, device="dgx")
    torch.testing.assert_close(
        torch.lerp(start, end, 0.5).to("cpu"),
        torch.lerp(start.to("cpu"), end.to("cpu"), 0.5),
    )

    weights = torch.full((4,), 0.3, device="dgx")
    torch.testing.assert_close(
        torch.lerp(start, end, weights).to("cpu"),
        torch.lerp(start.to("cpu"), end.to("cpu"), weights.to("cpu")),
    )
