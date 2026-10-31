/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
#pragma once

#include "StridedTensorUtils.cuh"

#include <array>
#include <cstdint>

namespace at::native::dgx
{

template <size_t num_tensors>
struct ReductionLayout
{
    int64_t num_segments;
    int64_t reduction_size;
    int64_t reduction_stride;
    StridedOffsetCalculator<num_tensors> offsets;
};

// Builds a flattened segment layout for reducing any one input dimension. The first stride list is
// the input. Output stride lists may either retain the reduced singleton dimension (keepdim=true) or
// omit it. StridedOffsetCalculator copies all sizes and strides, so the temporary vectors are safe.
template <size_t num_tensors>
ReductionLayout<num_tensors> make_reduction_layout(IntArrayRef input_sizes,
                                                   int64_t reduction_dim,
                                                   const std::array<IntArrayRef, num_tensors>& tensor_strides);

// Host builders are compiled once in ReductionLayout.cu for the supported tensor counts.
extern template ReductionLayout<2> make_reduction_layout<2>(IntArrayRef, int64_t, const std::array<IntArrayRef, 2>&);
extern template ReductionLayout<3> make_reduction_layout<3>(IntArrayRef, int64_t, const std::array<IntArrayRef, 3>&);

} // namespace at::native::dgx
