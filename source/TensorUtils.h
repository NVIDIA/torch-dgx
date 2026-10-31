/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
#pragma once

#include "CompatibilityUtils.h"

#include <c10/cuda/CUDAFunctions.h>

#include <ATen/EmptyTensor.h>
#include <ATen/core/Tensor.h>

#include <tuple>

namespace at::native
{

namespace dgx
{

static constexpr size_t kPageSize = 1u << 21;

std::tuple<size_t, size_t> aligned_chunk(size_t alignment, size_t size, c10::DeviceIndex device);

std::tuple<size_t, size_t> chunk(size_t size, c10::DeviceIndex device);

} // namespace dgx

} // namespace at::native
