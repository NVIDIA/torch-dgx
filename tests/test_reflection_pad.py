# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
import torch
import pytest

datatypes = [torch.int64, torch.float32, torch.float64]


@pytest.mark.parametrize("dtype", datatypes, ids=str)
def test_reflection_pad2d(dgx_setup_teardown, dtype):
    module = torch.nn.ReflectionPad2d(2).to("dgx")
    input = torch.arange(9, dtype=dtype, device="dgx").reshape(1, 1, 3, 3)
    output = module(input)
    reference = module.to("cpu")(input.to("cpu"))
    torch.testing.assert_close(output, reference.to("dgx"))

    module = torch.nn.ReflectionPad2d((1, 3, 2, 0)).to("dgx")
    input = torch.arange(16, dtype=dtype, device="dgx").reshape(1, 4, 4)
    output = module(input)
    reference = module.to("cpu")(input.to("cpu"))
    torch.testing.assert_close(output, reference.to("dgx"))
