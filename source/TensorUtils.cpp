/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
#include "TensorUtils.h"

#include <c10/cuda/CUDAStream.h>

#include <cuda/cmath>

#include <algorithm>

namespace at::native
{

namespace dgx
{

std::tuple<size_t, size_t> aligned_chunk(size_t alignment, size_t size, c10::DeviceIndex device)
{
    // Divide in alignment units to avoid overflowing alignment * device_count.
    const auto chunk_units = ::cuda::ceil_div(::cuda::ceil_div(size, alignment), c10::cuda::device_count());
    const auto chunk_size = chunk_units * alignment;
    // Clamp empty shards before multiplying, and avoid computing an end offset.
    if (!chunk_size || static_cast<size_t>(device) > size / chunk_size)
        return std::make_tuple(size, size_t(0));
    const auto chunk_offset = chunk_size * device;
    return std::make_tuple(chunk_offset, std::min(chunk_size, size - chunk_offset));
}

std::tuple<size_t, size_t> chunk(size_t size, c10::DeviceIndex device)
{
    return aligned_chunk(1u, size, device);
}

} // namespace dgx

} // namespace at::native
