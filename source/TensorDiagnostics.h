/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
#pragma once

#include <ATen/core/Tensor.h>

#include <iostream>

namespace at::native
{

namespace dgx
{

inline void printTensorDiagnostics(const std::string& name, const Tensor& tensor)
{
    std::cout << name << ": numel = " << tensor.numel() << ", sizes = " << tensor.sizes()
              << ", strides = " << tensor.strides() << ", dim = " << tensor.dim() << ", dtype = " << tensor.scalar_type()
              << ", device = " << tensor.device() << ", contiguous = " << tensor.is_contiguous()
              << ", nbytes = " << tensor.storage().nbytes() << ", layout = " << tensor.layout() << std::endl;
}

} // namespace dgx

} // namespace at::native
