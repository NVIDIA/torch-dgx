/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
#pragma once

#include "EventUtils.h"
#include "TensorUtils.h"

#include <c10/cuda/CUDAStream.h>

#include <cuda/cmath>

#include <algorithm>
#include <limits>

namespace at::native
{

namespace dgx
{

constexpr unsigned int kNumThreads = 256;

template <typename Func, typename... Args>
__global__ __launch_bounds__(kNumThreads) void lambda_kernel(int64_t numel, Func func, Args... args)
{
    const int64_t tid = static_cast<int64_t>(threadIdx.x) + static_cast<int64_t>(blockIdx.x) * blockDim.x;
    const int64_t stride = static_cast<int64_t>(blockDim.x) * gridDim.x;
    for (int64_t i = tid; i < numel; i += stride)
    {
        func(i, args...);
    }
}

template <typename Func, typename... Args>
__global__ __launch_bounds__(kNumThreads) void offset_lambda_kernel(int64_t numel, int64_t offset, Func func, Args... args)
{
    const int64_t tid = static_cast<int64_t>(threadIdx.x) + static_cast<int64_t>(blockIdx.x) * blockDim.x;
    const int64_t stride = static_cast<int64_t>(blockDim.x) * gridDim.x;
    for (int64_t i = tid; i < numel; i += stride)
    {
        func(i + offset, args...);
    }
}

template <typename Func, typename... Args>
void parallel_for_each(Func func, Args... args)
{
    int current_device = cudaInvalidDeviceId;
    C10_CUDA_CHECK(cudaGetDevice(&current_device));

    for (const auto device_id : c10::irange(c10::cuda::device_count()))
    {
        auto stream = c10::cuda::getCurrentCUDAStream(device_id);
        C10_CUDA_CHECK(cudaSetDevice(device_id));
        func(device_id, stream, args...);
    }

    merge_streams();

    C10_CUDA_CHECK(cudaSetDevice(current_device));
}

template <typename Func, typename... Args>
void parallel_for_each_async(Func func, Args... args)
{
    int current_device = cudaInvalidDeviceId;
    C10_CUDA_CHECK(cudaGetDevice(&current_device));

    for (const auto device_id : c10::irange(c10::cuda::device_count()))
    {
        auto stream = c10::cuda::getCurrentCUDAStream(device_id);
        C10_CUDA_CHECK(cudaSetDevice(device_id));
        func(device_id, stream, args...);
    }

    C10_CUDA_CHECK(cudaSetDevice(current_device));
}

template <typename Func, typename... Args>
void mgpu_launch_func(int64_t alignment, int64_t numel, Func func, Args... args)
{
    parallel_for_each([&](int device, cudaStream_t stream) {
        int64_t device_offset, device_numel;
        std::tie(device_offset, device_numel) = aligned_chunk(alignment, numel, device);
        const int64_t required_num_blocks = ::cuda::ceil_div(device_numel, kNumThreads);
        const unsigned int device_num_blocks =
            static_cast<unsigned int>(std::min<int64_t>(required_num_blocks, std::numeric_limits<int>::max()));
        if (device_num_blocks)
        {
            offset_lambda_kernel<<<device_num_blocks, kNumThreads, 0, stream>>>(
                device_numel, device_offset, func, args...);
            C10_CUDA_KERNEL_LAUNCH_CHECK();
        }
    });
}

template <typename Func, typename... Args>
void mgpu_launch_func(int64_t numel, Func func, Args... args)
{
    // Elementwise ownership follows the balanced logical partition of row-wise
    // producers. Rounding these ranges to managed-memory pages changes ownership
    // when neighboring operations use different feature widths.
    mgpu_launch_func(1, numel, func, args...);
}

} // namespace dgx

} // namespace at::native
