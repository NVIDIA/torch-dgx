# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
import pytest
import torch
import torch_dgx


@pytest.fixture(scope="function")
def dgx_setup_teardown():
    torch.random.manual_seed(42)

    # Check that the dgx device has been registered correctly
    torch.device("dgx")
    yield
