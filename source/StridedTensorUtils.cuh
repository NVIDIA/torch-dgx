/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
#pragma once

#include <c10/util/ArrayRef.h>

#include <ATen/TensorIterator.h>

#include <cuda/std/array>

#include <array>
#include <cstddef>
#include <cstdint>

namespace at::native::dgx
{

constexpr size_t kMaxTensorDimensions = TensorIteratorBase::DimMask{}.size();

template <size_t num_tensors>
struct StridedOffsetCalculator
{
    static_assert(num_tensors > 0);

    using offset_type = ::cuda::std::array<int64_t, num_tensors>;

    int64_t num_dims = 0;
    int64_t sizes[kMaxTensorDimensions] = {};
    int64_t strides[kMaxTensorDimensions][num_tensors] = {};

    __host__ __device__ offset_type get(int64_t index) const
    {
        offset_type offsets{};
        for (int64_t dim = num_dims - 1; dim >= 0; --dim)
        {
            const auto coordinate = index % sizes[dim];
            index /= sizes[dim];
#pragma unroll
            for (size_t tensor = 0; tensor < num_tensors; ++tensor)
            {
                offsets[tensor] += coordinate * strides[dim][tensor];
            }
        }
        return offsets;
    }
};

template <size_t num_tensors>
StridedOffsetCalculator<num_tensors> make_strided_offset_calculator(IntArrayRef sizes,
                                                                    const std::array<IntArrayRef, num_tensors>& strides);

// Host builders are compiled once in StridedTensorUtils.cu for the supported tensor counts.
extern template StridedOffsetCalculator<1> make_strided_offset_calculator<1>(IntArrayRef,
                                                                             const std::array<IntArrayRef, 1>&);
extern template StridedOffsetCalculator<2> make_strided_offset_calculator<2>(IntArrayRef,
                                                                             const std::array<IntArrayRef, 2>&);
extern template StridedOffsetCalculator<3> make_strided_offset_calculator<3>(IntArrayRef,
                                                                             const std::array<IntArrayRef, 3>&);
extern template StridedOffsetCalculator<4> make_strided_offset_calculator<4>(IntArrayRef,
                                                                             const std::array<IntArrayRef, 4>&);

template <typename input_t, typename output_t = input_t>
struct StridedInputOp
{
    const input_t* input;
    StridedOffsetCalculator<1> offsets;

    __device__ output_t operator()(int64_t index) const
    {
        return static_cast<output_t>(input[offsets.get(index)[0]]);
    }
};

} // namespace at::native::dgx
