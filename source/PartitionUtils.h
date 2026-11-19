/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */

#pragma once

#include <c10/cuda/CUDAFunctions.h>
#include <c10/macros/Macros.h>
#include <c10/util/Exception.h>

#include <cstddef>
#include <tuple>

namespace at::native::dgx
{
inline constexpr size_t kPageSize = 1u << 21;

namespace partition
{
// Keep the partition arithmetic identical in gsplat and torch-dgx.
// Take alignment before size, matching aligned_range and aligned_alloc.
// Alignment and device count must be positive; ranks must be less than devices.
C10_HOST_DEVICE inline size_t units(size_t alignment, size_t size)
{
    return size / alignment + (size % alignment != 0);
}

C10_HOST_DEVICE inline size_t slot_units(size_t alignment, size_t size, size_t devices)
{
    const auto num_units = units(alignment, size);
    return num_units / devices + (num_units % devices != 0);
}

// Logical ranges balance alignment units, assigning one extra unit to early ranks.
C10_HOST_DEVICE inline size_t offset(size_t alignment, size_t size, size_t devices, size_t rank)
{
    const auto num_units = units(alignment, size);
    const auto extra = num_units % devices;
    const auto first = rank * (num_units / devices) + (rank < extra ? rank : extra);
    // Clamp empty ranks before multiplying, including a partial final unit.
    return first == num_units ? size : first * alignment;
}

C10_HOST_DEVICE inline size_t count(size_t alignment, size_t size, size_t devices, size_t rank)
{
    const auto num_units = units(alignment, size);
    const auto count_units = num_units / devices + (rank < num_units % devices);
    const auto available = size - offset(alignment, size, devices, rank);
    return count_units > available / alignment ? available : count_units * alignment;
}

// Fixed-size slots are used only for NCCL. Callers must check these products for overflow.
C10_HOST_DEVICE inline size_t slot_size(size_t alignment, size_t size, size_t devices)
{
    return slot_units(alignment, size, devices) * alignment;
}

C10_HOST_DEVICE inline size_t padded_size(size_t alignment, size_t size, size_t devices)
{
    return devices * slot_size(alignment, size, devices);
}

C10_HOST_DEVICE inline size_t packed_offset(size_t alignment, size_t size, size_t devices, size_t rank)
{
    return rank * slot_size(alignment, size, devices);
}

C10_HOST_DEVICE inline bool has_internal_padding(size_t alignment, size_t size, size_t devices)
{
    const auto num_units = units(alignment, size);
    const auto extra = num_units % devices;
    // Gaps occur only before another nonempty small rank. A single smaller
    // final rank (or an empty tail) needs only trailing padding.
    return num_units / devices > 0 && extra > 0 && extra + 1 < devices;
}

// Map a valid logical index to its rank's slot, preserving order within each rank.
C10_HOST_DEVICE inline size_t packed_index(size_t alignment, size_t size, size_t devices, size_t index)
{
    if (!has_internal_padding(alignment, size, devices))
        return index;
    const auto num_units = units(alignment, size);
    const auto base = num_units / devices;
    const auto extra = num_units % devices;
    const auto unit = index / alignment;
    const auto large_units = extra * (base + 1);
    const auto rank = unit < large_units ? unit / (base + 1) : extra + (unit - large_units) / base;
    return packed_offset(alignment, size, devices, rank) + (index - offset(alignment, size, devices, rank));
}

// Padding maps to size, which is outside the valid logical range.
C10_HOST_DEVICE inline size_t logical_index(size_t alignment, size_t size, size_t devices, size_t index)
{
    if (!has_internal_padding(alignment, size, devices))
        return index < size ? index : size;
    const auto slot_elements = slot_size(alignment, size, devices);
    const auto rank = index / slot_elements;
    const auto local_index = index % slot_elements;
    return local_index < count(alignment, size, devices, rank) ? offset(alignment, size, devices, rank) + local_index :
                                                                 size;
}

// Balance ceil(size / alignment) units across devices, assigning the remainder
// to the first ranks. Clip the final unit and return {size, 0} for empty ranks.
inline std::tuple<size_t, size_t> aligned_range(size_t alignment, size_t size, c10::DeviceIndex device)
{
    TORCH_CHECK(alignment > 0 && c10::cuda::device_count() > 0 && device >= 0 && device < c10::cuda::device_count(),
                "aligned_range requires positive alignment and a valid CUDA device");
    return { offset(alignment, size, c10::cuda::device_count(), device),
             count(alignment, size, c10::cuda::device_count(), device) };
}

inline std::tuple<size_t, size_t> range(size_t size, c10::DeviceIndex device)
{
    return aligned_range(1u, size, device);
}
} // namespace partition
} // namespace at::native::dgx
