/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
#include "StridedTensorUtils.cuh"

#include <c10/util/Exception.h>
#include <c10/util/irange.h>

namespace at::native::dgx
{

template <size_t num_tensors>
StridedOffsetCalculator<num_tensors> make_strided_offset_calculator(IntArrayRef sizes,
                                                                    const std::array<IntArrayRef, num_tensors>& strides)
{
    TORCH_CHECK(sizes.size() <= kMaxTensorDimensions, "Tensors with more than ", kMaxTensorDimensions,
                " dimensions are not supported");

    StridedOffsetCalculator<num_tensors> calculator{};
    calculator.num_dims = sizes.size();
    for (size_t tensor = 0; tensor < num_tensors; ++tensor)
    {
        TORCH_CHECK(strides[tensor].size() == sizes.size(), "Expected ", sizes.size(), " strides for tensor ", tensor,
                    " but got ", strides[tensor].size());
    }

    for (const auto dim : c10::irange(sizes.size()))
    {
        calculator.sizes[dim] = sizes[dim];
        for (size_t tensor = 0; tensor < num_tensors; ++tensor)
        {
            calculator.strides[dim][tensor] = strides[tensor][dim];
        }
    }
    return calculator;
}

template StridedOffsetCalculator<1> make_strided_offset_calculator<1>(IntArrayRef, const std::array<IntArrayRef, 1>&);
template StridedOffsetCalculator<2> make_strided_offset_calculator<2>(IntArrayRef, const std::array<IntArrayRef, 2>&);
template StridedOffsetCalculator<3> make_strided_offset_calculator<3>(IntArrayRef, const std::array<IntArrayRef, 3>&);
template StridedOffsetCalculator<4> make_strided_offset_calculator<4>(IntArrayRef, const std::array<IntArrayRef, 4>&);

} // namespace at::native::dgx
