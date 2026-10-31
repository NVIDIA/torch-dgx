/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
#pragma once

#include <c10/core/Device.h>
#include <c10/util/FunctionRef.h>

#include <cuda_runtime_api.h>

#include <cstddef>
#include <cstdint>

namespace at::native::dgx
{

using ScalarReductionFn =
    c10::function_ref<cudaError_t(void*, size_t&, c10::DeviceIndex, int64_t, int64_t, cudaStream_t)>;
using ScalarReductionMergeFn = c10::function_ref<cudaError_t(void*, size_t&, c10::DeviceIndex, cudaStream_t)>;

// Callbacks query CUB storage with a null pointer, then launch with allocated storage.
// They are invoked synchronously and never retained; the GPU work remains asynchronous.
void scalar_reduce(int64_t numel, ScalarReductionFn reduce, ScalarReductionMergeFn merge, bool broadcast_result = true);

} // namespace at::native::dgx
