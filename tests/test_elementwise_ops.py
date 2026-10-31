# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
import torch
import pytest
from torch.library import opcheck

datatypes = [torch.float32, torch.float64]


@pytest.mark.parametrize("dtype", datatypes, ids=str)
def test_addcmul(dgx_setup_teardown, dtype):
    a = torch.rand(2, 11, dtype=dtype).to("dgx")
    b = torch.rand(2, 11, dtype=dtype).to("dgx")
    c = torch.rand(2, 11, dtype=dtype).to("dgx")
    d = torch.addcmul(a, b, c, value=0.3)
    ref = torch.addcmul(a.to("cpu"), b.to("cpu"), c.to("cpu"), value=0.3).to("dgx")
    torch.testing.assert_close(d, ref)
    opcheck(torch.ops.aten.addcmul.default, (a, b, c), {"value": 0.3})


@pytest.mark.parametrize("dtype", datatypes, ids=str)
def test_addcdiv(dgx_setup_teardown, dtype):
    a = torch.rand(2, 11, dtype=dtype).to("dgx")
    b = torch.rand(2, 11, dtype=dtype).to("dgx")
    c = torch.abs(torch.rand(2, 11, dtype=dtype)).to("dgx") + 0.1
    d = torch.addcdiv(a, b, c, value=0.3)
    ref = torch.addcdiv(a.to("cpu"), b.to("cpu"), c.to("cpu"), value=0.3).to("dgx")
    torch.testing.assert_close(d, ref)
    opcheck(torch.ops.aten.addcdiv.default, (a, b, c), {"value": 0.3})
