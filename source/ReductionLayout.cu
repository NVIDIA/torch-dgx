/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
#include "ReductionLayout.cuh"

#include <c10/util/Exception.h>
#include <c10/util/irange.h>

#include <vector>

namespace at::native::dgx
{

template <size_t num_tensors>
ReductionLayout<num_tensors> make_reduction_layout(IntArrayRef input_sizes,
                                                   int64_t reduction_dim,
                                                   const std::array<IntArrayRef, num_tensors>& tensor_strides)
{
    static_assert(num_tensors > 1);
    TORCH_CHECK(!input_sizes.empty(), "Reduction layout requires a non-scalar input");
    TORCH_CHECK(reduction_dim >= 0 && reduction_dim < input_sizes.size(), "Invalid reduction dimension");
    TORCH_CHECK(tensor_strides[0].size() == input_sizes.size(), "Input sizes and strides must have the same rank");

    std::vector<int64_t> segment_sizes;
    segment_sizes.reserve(input_sizes.size() - 1);
    std::array<std::vector<int64_t>, num_tensors> segment_strides;
    for (auto& strides : segment_strides)
    {
        strides.reserve(input_sizes.size() - 1);
    }

    int64_t num_segments = 1;
    for (const auto dim : c10::irange(input_sizes.size()))
    {
        if (dim == reduction_dim)
        {
            continue;
        }
        segment_sizes.push_back(input_sizes[dim]);
        num_segments *= input_sizes[dim];
        segment_strides[0].push_back(tensor_strides[0][dim]);
    }

    for (size_t tensor = 1; tensor < num_tensors; ++tensor)
    {
        const auto strides = tensor_strides[tensor];
        if (strides.size() == input_sizes.size())
        {
            for (const auto dim : c10::irange(strides.size()))
            {
                if (dim != reduction_dim)
                {
                    segment_strides[tensor].push_back(strides[dim]);
                }
            }
        }
        else
        {
            TORCH_CHECK(strides.size() + 1 == input_sizes.size(), "Output rank does not match reduction layout");
            segment_strides[tensor].assign(strides.begin(), strides.end());
        }
    }

    std::array<IntArrayRef, num_tensors> stride_refs;
    for (size_t tensor = 0; tensor < num_tensors; ++tensor)
    {
        stride_refs[tensor] = segment_strides[tensor];
    }

    return { num_segments, input_sizes[reduction_dim], tensor_strides[0][reduction_dim],
             make_strided_offset_calculator<num_tensors>(segment_sizes, stride_refs) };
}

template ReductionLayout<2> make_reduction_layout<2>(IntArrayRef, int64_t, const std::array<IntArrayRef, 2>&);
template ReductionLayout<3> make_reduction_layout<3>(IntArrayRef, int64_t, const std::array<IntArrayRef, 3>&);

} // namespace at::native::dgx
