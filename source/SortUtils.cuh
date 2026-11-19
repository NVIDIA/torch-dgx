/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
#pragma once

#include "CompatibilityUtils.h"
#include "KernelLaunchUtils.cuh"

#include <c10/util/BFloat16.h>
#include <c10/util/Half.h>
#include <c10/util/ScopeExit.h>

#include <cub/device/device_radix_sort.cuh>

#include <algorithm>
#include <cstdint>

namespace at::native::dgx
{

// Floating-point keys encode ascending order, with NaNs last and both zeros equal.
// Gather original values after sorting to retain NaN payloads and zero signs.
template <typename scalar_t>
struct SortKeyTraits
{
    using key_t = scalar_t;

    __device__ static key_t encode(scalar_t value)
    {
        return value;
    }
};

template <>
struct SortKeyTraits<float>
{
    using key_t = uint32_t;

    __device__ static key_t encode(float value)
    {
        constexpr key_t sign_bit = UINT32_C(1) << 31;
        if (value != value)
            return UINT32_MAX;
        if (value == 0.0f)
            return sign_bit;
        const key_t bits = __float_as_uint(value);
        return bits & sign_bit ? ~bits : bits ^ sign_bit;
    }
};

template <>
struct SortKeyTraits<double>
{
    using key_t = uint64_t;

    __device__ static key_t encode(double value)
    {
        constexpr key_t sign_bit = UINT64_C(1) << 63;
        if (value != value)
            return UINT64_MAX;
        if (value == 0.0)
            return sign_bit;
        const key_t bits = static_cast<key_t>(__double_as_longlong(value));
        return bits & sign_bit ? ~bits : bits ^ sign_bit;
    }
};

template <>
struct SortKeyTraits<c10::Half> : SortKeyTraits<float>
{
};

template <>
struct SortKeyTraits<c10::BFloat16> : SortKeyTraits<float>
{
};

inline int sort_grid_size(int64_t count)
{
    return static_cast<int>(std::min<int64_t>(::cuda::ceil_div(count, int64_t(kNumThreads) * 4), 4096));
}

template <typename key_t>
void radix_sort_pairs(const key_t* keys_in,
                      key_t* keys_out,
                      const int64_t* indices_in,
                      int64_t* indices_out,
                      int64_t count,
                      bool descending,
                      cudaStream_t stream)
{
    size_t scratch_size = 0;
    void* scratch_data = nullptr;
    auto sort = [&] {
        if (descending)
            return cub::DeviceRadixSort::SortPairsDescending(scratch_data, scratch_size, keys_in, keys_out, indices_in,
                                                             indices_out, count, 0, sizeof(key_t) * 8, stream);
        return cub::DeviceRadixSort::SortPairs(scratch_data, scratch_size, keys_in, keys_out, indices_in, indices_out,
                                               count, 0, sizeof(key_t) * 8, stream);
    };
    C10_CUDA_CHECK(sort());
    C10_CUDA_CHECK(cudaMallocAsync(&scratch_data, scratch_size, stream));
    auto release = c10::make_scope_exit([&] { C10_CUDA_CHECK_WARN(cudaFreeAsync(scratch_data, stream)); });
    C10_CUDA_CHECK(sort());
}

template <typename key_t>
void prefetch_sort_outputs(key_t* keys, int64_t* indices, int64_t count, int device, cudaStream_t stream)
{
    const cudaMemLocation location{ cudaMemLocationTypeDevice, device };
    C10_CUDA_CHECK(memPrefetchAsync(keys, count * sizeof(key_t), location, 0, stream));
    C10_CUDA_CHECK(memPrefetchAsync(indices, count * sizeof(int64_t), location, 0, stream));
}

} // namespace at::native::dgx
