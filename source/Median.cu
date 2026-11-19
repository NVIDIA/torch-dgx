/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
#include "ReductionLayout.cuh"
#include "SortUtils.cuh"

#include <c10/core/WrapDimMinimal.h>

#include <ATen/Dispatch_v2.h>
#include <ATen/MemoryOverlap.h>
#include <ATen/native/ReduceOpsUtils.h>
#include <ATen/native/Resize.h>
#include <ATen/ops/empty.h>

#include <torch/library.h>

#include <cub/block/block_reduce.cuh>
#include <cuda/functional>

#include <algorithm>
#include <array>
#include <cstdint>
#include <type_traits>

namespace at::native
{

namespace dgx
{

namespace
{

constexpr int kMedianThreadSize = 32;

template <typename scalar_t>
using median_key_t = std::make_unsigned_t<typename SortKeyTraits<scalar_t>::key_t>;

template <typename scalar_t>
__device__ median_key_t<scalar_t> median_key(scalar_t value)
{
    using key_t = median_key_t<scalar_t>;
    key_t key = static_cast<key_t>(SortKeyTraits<scalar_t>::encode(value));
    if constexpr (std::is_signed_v<typename SortKeyTraits<scalar_t>::key_t>)
        key ^= key_t(1) << (sizeof(key_t) * 8 - 1);
    return key;
}

template <typename scalar_t>
struct MedianSmallFunctor
{
    __device__ void operator()(
        int64_t segment, const scalar_t* input, scalar_t* values, int64_t* indices, ReductionLayout<3> layout) const
    {
        const auto offsets = layout.offsets.get(segment);
        const auto* slice = input + offsets[0];
        median_key_t<scalar_t> keys[kMedianThreadSize];
        int positions[kMedianThreadSize];
        for (int i = 0; i < layout.reduction_size; ++i)
        {
            const scalar_t value = slice[i * layout.reduction_stride];
            if (value != value)
            {
                values[offsets[1]] = value;
                indices[offsets[2]] = i;
                return;
            }
            const auto key = median_key(value);
            int position = i;
            while (position > 0 && key < keys[position - 1])
            {
                keys[position] = keys[position - 1];
                positions[position] = positions[position - 1];
                --position;
            }
            keys[position] = key;
            positions[position] = i;
        }
        int rank = (layout.reduction_size - 1) / 2;
        while (rank > 0 && keys[rank - 1] == keys[rank])
            --rank;
        const int64_t index = positions[rank];
        values[offsets[1]] = slice[index * layout.reduction_stride];
        indices[offsets[2]] = index;
    }
};

template <typename key_t>
struct MedianBounds
{
    key_t lower;
    key_t upper;
    int64_t nan_index;
};

template <typename key_t>
struct MedianBoundsOp
{
    __device__ MedianBounds<key_t> operator()(MedianBounds<key_t> a, MedianBounds<key_t> b) const
    {
        return { a.lower < b.lower ? a.lower : b.lower, a.upper > b.upper ? a.upper : b.upper,
                 a.nan_index < b.nan_index ? a.nan_index : b.nan_index };
    }
};

template <typename scalar_t>
__global__ __launch_bounds__(kNumThreads) void median_block_kernel(int64_t segment_offset,
                                                                   int64_t segment_count,
                                                                   const scalar_t* input,
                                                                   scalar_t* values,
                                                                   int64_t* indices,
                                                                   ReductionLayout<3> layout)
{
    using key_t = median_key_t<scalar_t>;
    using BoundsReduce = cub::BlockReduce<MedianBounds<key_t>, kNumThreads>;
    using CountReduce = cub::BlockReduce<int64_t, kNumThreads>;
    __shared__ typename BoundsReduce::TempStorage bounds_storage;
    __shared__ typename CountReduce::TempStorage count_storage;
    __shared__ MedianBounds<key_t> bounds;
    __shared__ int64_t lower_count;

    for (int64_t segment = blockIdx.x; segment < segment_count; segment += gridDim.x)
    {
        const auto offsets = layout.offsets.get(segment_offset + segment);
        const auto* slice = input + offsets[0];
        MedianBounds<key_t> local{ key_t(~key_t(0)), 0, layout.reduction_size };
        for (int64_t i = threadIdx.x; i < layout.reduction_size; i += blockDim.x)
        {
            const scalar_t value = slice[i * layout.reduction_stride];
            const auto key = median_key(value);
            local = MedianBoundsOp<key_t>()(local, { key, key, value != value ? i : layout.reduction_size });
        }
        const auto aggregate = BoundsReduce(bounds_storage).Reduce(local, MedianBoundsOp<key_t>());
        if (!threadIdx.x)
            bounds = aggregate;
        __syncthreads();

        if (bounds.nan_index < layout.reduction_size)
        {
            if (!threadIdx.x)
            {
                values[offsets[1]] = slice[bounds.nan_index * layout.reduction_stride];
                indices[offsets[2]] = bounds.nan_index;
            }
            __syncthreads();
            continue;
        }

        const key_t different = bounds.lower ^ bounds.upper;
        key_t bit = different ? key_t(1) << (63 - __clzll(static_cast<unsigned long long>(different))) : 0;
        key_t mask = bit ? key_t(~(bit | (bit - 1))) : key_t(~key_t(0));
        key_t prefix = bounds.lower & mask;
        int64_t rank = (layout.reduction_size - 1) / 2;
        int64_t candidates = layout.reduction_size;
        // Refine only the prefix containing the lower median, stopping when its index is unique.
        while (bit && candidates > 1)
        {
            int64_t count = 0;
            for (int64_t i = threadIdx.x; i < layout.reduction_size; i += blockDim.x)
            {
                const auto key = median_key(slice[i * layout.reduction_stride]);
                count += (key & mask) == prefix && !(key & bit);
            }
            const int64_t total = CountReduce(count_storage).Sum(count);
            if (!threadIdx.x)
                lower_count = total;
            __syncthreads();
            if (rank < lower_count)
            {
                candidates = lower_count;
            }
            else
            {
                rank -= lower_count;
                candidates -= lower_count;
                prefix |= bit;
            }
            mask |= bit;
            bit >>= 1;
        }

        int64_t first = layout.reduction_size;
        for (int64_t i = threadIdx.x; i < layout.reduction_size; i += blockDim.x)
        {
            if ((median_key(slice[i * layout.reduction_stride]) & mask) == prefix)
            {
                first = i;
                break;
            }
        }
        const int64_t index = CountReduce(count_storage).Reduce(first, ::cuda::minimum<int64_t>());
        if (!threadIdx.x)
        {
            values[offsets[1]] = slice[index * layout.reduction_stride];
            indices[offsets[2]] = index;
        }
        __syncthreads();
    }
}

} // namespace

std::tuple<Tensor&, Tensor&> median_dim_out(const Tensor& self, int64_t dim, bool keepdim, Tensor& values, Tensor& indices)
{
    dim = c10::maybe_wrap_dim(dim, self.dim());
    zero_numel_check_dims(self, dim, "median()");
    TORCH_CHECK(values.scalar_type() == self.scalar_type(), "DGX median values must have the input dtype");
    TORCH_CHECK(indices.scalar_type() == kLong, "DGX median indices must have dtype int64");
    TORCH_CHECK(values.device() == self.device() && indices.device() == self.device(),
                "DGX median outputs must be on the input device");

    auto output_sizes = self.sizes().vec();
    if (self.dim())
    {
        if (keepdim)
            output_sizes[dim] = 1;
        else
            output_sizes.erase(output_sizes.begin() + dim);
    }
    // Snapshot shared storage before resizing outputs or launching overlapping reductions.
    auto input = self.is_alias_of(values) || self.is_alias_of(indices) ? self.clone() : self;
    resize_output(values, output_sizes);
    resize_output(indices, output_sizes);
    assert_no_internal_overlap(values);
    assert_no_internal_overlap(indices);
    assert_no_overlap(values, indices);
    if (!values.numel())
        return std::forward_as_tuple(values, indices);
    if (!input.dim())
        input = input.view({ 1 });

    const auto layout = make_reduction_layout<3>(
        input.sizes(), dim, std::array<IntArrayRef, 3>{ input.strides(), values.strides(), indices.strides() });
    AT_DISPATCH_V2(input.scalar_type(), "median_dim_out", AT_WRAP([&] {
                       if (layout.reduction_size <= kMedianThreadSize)
                       {
                           // Short view axes keep adjacent pixels in adjacent threads, without transposing the input.
                           mgpu_launch_func(layout.num_segments, MedianSmallFunctor<scalar_t>(),
                                            input.const_data_ptr<scalar_t>(), values.data_ptr<scalar_t>(),
                                            indices.data_ptr<int64_t>(), layout);
                       }
                       else
                       {
                           parallel_for_each([&](int device, cudaStream_t stream) {
                               size_t offset, count;
                               std::tie(offset, count) = partition::range(layout.num_segments, device);
                               if (!count)
                                   return;
                               const auto blocks = static_cast<unsigned int>(std::min<int64_t>(count, 4096));
                               median_block_kernel<<<blocks, kNumThreads, 0, stream>>>(
                                   offset, count, input.const_data_ptr<scalar_t>(), values.data_ptr<scalar_t>(),
                                   indices.data_ptr<int64_t>(), layout);
                               C10_CUDA_KERNEL_LAUNCH_CHECK();
                           });
                       }
                   }),
                   AT_EXPAND(AT_ALL_TYPES), kHalf);
    return std::forward_as_tuple(values, indices);
}

TORCH_LIBRARY_IMPL(aten, PrivateUse1, m)
{
    m.impl(TORCH_SELECTIVE_NAME("aten::median.dim_values"), TORCH_FN(median_dim_out));
}

} // namespace dgx

} // namespace at::native
