/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
#pragma once

#include "KernelLaunchUtils.cuh"
#include "ReductionLayout.cuh"

#include <cub/block/block_reduce.cuh>
#include <cub/thread/thread_reduce.cuh>
#include <cub/warp/warp_reduce.cuh>

#include <algorithm>
#include <cstdint>
#include <limits>
#include <tuple>

namespace at::native::dgx
{

namespace detail
{

template <int items_per_thread, typename input_t, typename acc_t, size_t num_tensors, typename TransformOp, typename ReductionOp, typename OutputOp>
__global__ void row_reduce_thread_kernel(int64_t num_segments,
                                         int64_t segment_offset,
                                         const input_t* input,
                                         ReductionLayout<num_tensors> layout,
                                         TransformOp transform_op,
                                         ReductionOp reduction_op,
                                         acc_t identity,
                                         OutputOp output_op)
{
    const int64_t tid = static_cast<int64_t>(threadIdx.x) + static_cast<int64_t>(blockIdx.x) * blockDim.x;
    const int64_t step = static_cast<int64_t>(blockDim.x) * gridDim.x;
    for (int64_t segment = tid; segment < num_segments; segment += step)
    {
        const auto offsets = layout.offsets.get(segment_offset + segment);
        acc_t items[items_per_thread];
#pragma unroll
        for (int item = 0; item < items_per_thread; ++item)
        {
            items[item] = item < layout.reduction_size ?
                              transform_op(input[offsets[0] + item * layout.reduction_stride], item) :
                              identity;
        }
        output_op(offsets, reduction_op(identity, cub::ThreadReduce(items, reduction_op)));
    }
}

template <typename input_t, typename acc_t, size_t num_tensors, typename TransformOp, typename ReductionOp, typename OutputOp>
__global__ void row_reduce_strided_kernel(int64_t num_segments,
                                          int64_t segment_offset,
                                          const input_t* input,
                                          ReductionLayout<num_tensors> layout,
                                          TransformOp transform_op,
                                          ReductionOp reduction_op,
                                          acc_t identity,
                                          OutputOp output_op)
{
    const int64_t tid = static_cast<int64_t>(threadIdx.x) + static_cast<int64_t>(blockIdx.x) * blockDim.x;
    const int64_t step = static_cast<int64_t>(blockDim.x) * gridDim.x;
    for (int64_t segment = tid; segment < num_segments; segment += step)
    {
        const auto offsets = layout.offsets.get(segment_offset + segment);
        acc_t aggregate = identity;
        for (int64_t item = 0; item < layout.reduction_size; ++item)
        {
            aggregate = reduction_op(aggregate, transform_op(input[offsets[0] + item * layout.reduction_stride], item));
        }
        output_op(offsets, aggregate);
    }
}

template <int logical_warp_threads,
          int items_per_thread,
          typename input_t,
          typename acc_t,
          size_t num_tensors,
          typename TransformOp,
          typename ReductionOp,
          typename OutputOp>
__global__ void row_reduce_warp_kernel(int64_t num_segments,
                                       int64_t segment_offset,
                                       const input_t* input,
                                       ReductionLayout<num_tensors> layout,
                                       TransformOp transform_op,
                                       ReductionOp reduction_op,
                                       acc_t identity,
                                       OutputOp output_op)
{
    using WarpReduce = cub::WarpReduce<acc_t, logical_warp_threads>;
    constexpr int warps_per_block = kNumThreads / logical_warp_threads;
    __shared__ typename WarpReduce::TempStorage temp_storage[warps_per_block];

    const int warp = threadIdx.x / logical_warp_threads;
    const int lane = threadIdx.x % logical_warp_threads;
    unsigned int logical_warp_mask;
    if constexpr (logical_warp_threads == 32)
    {
        logical_warp_mask = 0xffffffffu;
    }
    else
    {
        const int logical_warp_in_physical_warp = (threadIdx.x % 32) / logical_warp_threads;
        logical_warp_mask = ((1u << logical_warp_threads) - 1) << (logical_warp_in_physical_warp * logical_warp_threads);
    }
    const int64_t step = static_cast<int64_t>(gridDim.x) * warps_per_block;
    for (int64_t base = static_cast<int64_t>(blockIdx.x) * warps_per_block; base < num_segments; base += step)
    {
        const int64_t segment = base + warp;
        typename StridedOffsetCalculator<num_tensors>::offset_type offsets{};
        acc_t items[items_per_thread];
#pragma unroll
        for (int item = 0; item < items_per_thread; ++item)
        {
            items[item] = identity;
        }

        if (segment < num_segments)
        {
            offsets = layout.offsets.get(segment_offset + segment);
#pragma unroll
            for (int item = 0; item < items_per_thread; ++item)
            {
                const int64_t reduction_index = lane + item * logical_warp_threads;
                if (reduction_index < layout.reduction_size)
                {
                    items[item] =
                        transform_op(input[offsets[0] + reduction_index * layout.reduction_stride], reduction_index);
                }
            }
        }

        const acc_t thread_aggregate = cub::ThreadReduce(items, reduction_op);
        const acc_t aggregate = WarpReduce(temp_storage[warp]).Reduce(thread_aggregate, reduction_op);
        if (segment < num_segments && lane == 0)
        {
            output_op(offsets, reduction_op(identity, aggregate));
        }
        __syncwarp(logical_warp_mask);
    }
}

template <int items_per_thread, typename input_t, typename acc_t, size_t num_tensors, typename TransformOp, typename ReductionOp, typename OutputOp>
__global__ void row_reduce_block_kernel(int64_t num_segments,
                                        int64_t segment_offset,
                                        const input_t* input,
                                        ReductionLayout<num_tensors> layout,
                                        TransformOp transform_op,
                                        ReductionOp reduction_op,
                                        acc_t identity,
                                        OutputOp output_op)
{
    using BlockReduce = cub::BlockReduce<acc_t, kNumThreads>;
    __shared__ typename BlockReduce::TempStorage temp_storage;

    for (int64_t segment = blockIdx.x; segment < num_segments; segment += gridDim.x)
    {
        const auto offsets = layout.offsets.get(segment_offset + segment);
        acc_t items[items_per_thread];
#pragma unroll
        for (int item = 0; item < items_per_thread; ++item)
        {
            const int64_t reduction_index = threadIdx.x + item * kNumThreads;
            items[item] =
                reduction_index < layout.reduction_size ?
                    transform_op(input[offsets[0] + reduction_index * layout.reduction_stride], reduction_index) :
                    identity;
        }

        const acc_t thread_aggregate = cub::ThreadReduce(items, reduction_op);
        const acc_t aggregate = BlockReduce(temp_storage).Reduce(thread_aggregate, reduction_op);
        if (threadIdx.x == 0)
        {
            output_op(offsets, reduction_op(identity, aggregate));
        }
        __syncthreads();
    }
}

template <typename input_t, typename acc_t, size_t num_tensors, typename TransformOp, typename ReductionOp, typename OutputOp>
__global__ void row_reduce_dynamic_block_kernel(int64_t num_segments,
                                                int64_t segment_offset,
                                                const input_t* input,
                                                ReductionLayout<num_tensors> layout,
                                                TransformOp transform_op,
                                                ReductionOp reduction_op,
                                                acc_t identity,
                                                OutputOp output_op)
{
    using BlockReduce = cub::BlockReduce<acc_t, kNumThreads>;
    __shared__ typename BlockReduce::TempStorage temp_storage;

    for (int64_t segment = blockIdx.x; segment < num_segments; segment += gridDim.x)
    {
        const auto offsets = layout.offsets.get(segment_offset + segment);
        acc_t thread_aggregate = identity;
        for (int64_t reduction_index = threadIdx.x; reduction_index < layout.reduction_size;
             reduction_index += kNumThreads)
        {
            thread_aggregate = reduction_op(
                thread_aggregate,
                transform_op(input[offsets[0] + reduction_index * layout.reduction_stride], reduction_index));
        }

        const acc_t aggregate = BlockReduce(temp_storage).Reduce(thread_aggregate, reduction_op);
        if (threadIdx.x == 0)
        {
            output_op(offsets, aggregate);
        }
        __syncthreads();
    }
}

inline unsigned int reduction_grid_size(int64_t work_items)
{
    return static_cast<unsigned int>(std::min<int64_t>(work_items, std::numeric_limits<int>::max()));
}

} // namespace detail

template <typename input_t, typename acc_t, size_t num_tensors, typename TransformOp, typename ReductionOp, typename OutputOp>
void row_reduce(const input_t* input,
                const ReductionLayout<num_tensors>& layout,
                TransformOp transform_op,
                ReductionOp reduction_op,
                acc_t identity,
                OutputOp output_op)
{
    parallel_for_each([&](int device, cudaStream_t stream) {
        int64_t segment_offset, num_segments;
        std::tie(segment_offset, num_segments) = chunk(layout.num_segments, device);
        if (!num_segments)
        {
            return;
        }

        if (layout.reduction_stride != 1)
        {
            const auto blocks = detail::reduction_grid_size(::cuda::ceil_div(num_segments, kNumThreads));
            detail::row_reduce_strided_kernel<<<blocks, kNumThreads, 0, stream>>>(
                num_segments, segment_offset, input, layout, transform_op, reduction_op, identity, output_op);
        }
        else if (layout.reduction_size <= 1)
        {
            const auto blocks = detail::reduction_grid_size(::cuda::ceil_div(num_segments, kNumThreads));
            detail::row_reduce_thread_kernel<1><<<blocks, kNumThreads, 0, stream>>>(
                num_segments, segment_offset, input, layout, transform_op, reduction_op, identity, output_op);
        }
        else if (layout.reduction_size <= 2)
        {
            const auto blocks = detail::reduction_grid_size(::cuda::ceil_div(num_segments, kNumThreads));
            detail::row_reduce_thread_kernel<2><<<blocks, kNumThreads, 0, stream>>>(
                num_segments, segment_offset, input, layout, transform_op, reduction_op, identity, output_op);
        }
        else if (layout.reduction_size <= 4)
        {
            const auto blocks = detail::reduction_grid_size(::cuda::ceil_div(num_segments, kNumThreads));
            detail::row_reduce_thread_kernel<4><<<blocks, kNumThreads, 0, stream>>>(
                num_segments, segment_offset, input, layout, transform_op, reduction_op, identity, output_op);
        }
        else if (layout.reduction_size <= 8)
        {
            constexpr int rows_per_block = kNumThreads / 8;
            const auto blocks = detail::reduction_grid_size(::cuda::ceil_div(num_segments, rows_per_block));
            detail::row_reduce_warp_kernel<8, 1><<<blocks, kNumThreads, 0, stream>>>(
                num_segments, segment_offset, input, layout, transform_op, reduction_op, identity, output_op);
        }
        else if (layout.reduction_size <= 16)
        {
            constexpr int rows_per_block = kNumThreads / 16;
            const auto blocks = detail::reduction_grid_size(::cuda::ceil_div(num_segments, rows_per_block));
            detail::row_reduce_warp_kernel<16, 1><<<blocks, kNumThreads, 0, stream>>>(
                num_segments, segment_offset, input, layout, transform_op, reduction_op, identity, output_op);
        }
        else if (layout.reduction_size <= 32)
        {
            constexpr int rows_per_block = kNumThreads / 32;
            const auto blocks = detail::reduction_grid_size(::cuda::ceil_div(num_segments, rows_per_block));
            detail::row_reduce_warp_kernel<32, 1><<<blocks, kNumThreads, 0, stream>>>(
                num_segments, segment_offset, input, layout, transform_op, reduction_op, identity, output_op);
        }
        else if (layout.reduction_size <= 64)
        {
            constexpr int rows_per_block = kNumThreads / 32;
            const auto blocks = detail::reduction_grid_size(::cuda::ceil_div(num_segments, rows_per_block));
            detail::row_reduce_warp_kernel<32, 2><<<blocks, kNumThreads, 0, stream>>>(
                num_segments, segment_offset, input, layout, transform_op, reduction_op, identity, output_op);
        }
        else if (layout.reduction_size <= 128)
        {
            constexpr int rows_per_block = kNumThreads / 32;
            const auto blocks = detail::reduction_grid_size(::cuda::ceil_div(num_segments, rows_per_block));
            detail::row_reduce_warp_kernel<32, 4><<<blocks, kNumThreads, 0, stream>>>(
                num_segments, segment_offset, input, layout, transform_op, reduction_op, identity, output_op);
        }
        else if (layout.reduction_size <= 256)
        {
            constexpr int rows_per_block = kNumThreads / 32;
            const auto blocks = detail::reduction_grid_size(::cuda::ceil_div(num_segments, rows_per_block));
            detail::row_reduce_warp_kernel<32, 8><<<blocks, kNumThreads, 0, stream>>>(
                num_segments, segment_offset, input, layout, transform_op, reduction_op, identity, output_op);
        }
        else if (layout.reduction_size <= 512)
        {
            const auto blocks = detail::reduction_grid_size(num_segments);
            detail::row_reduce_block_kernel<2><<<blocks, kNumThreads, 0, stream>>>(
                num_segments, segment_offset, input, layout, transform_op, reduction_op, identity, output_op);
        }
        else if (layout.reduction_size <= 1024)
        {
            const auto blocks = detail::reduction_grid_size(num_segments);
            detail::row_reduce_block_kernel<4><<<blocks, kNumThreads, 0, stream>>>(
                num_segments, segment_offset, input, layout, transform_op, reduction_op, identity, output_op);
        }
        else if (layout.reduction_size <= 2048)
        {
            const auto blocks = detail::reduction_grid_size(num_segments);
            detail::row_reduce_block_kernel<8><<<blocks, kNumThreads, 0, stream>>>(
                num_segments, segment_offset, input, layout, transform_op, reduction_op, identity, output_op);
        }
        else
        {
            const auto blocks = detail::reduction_grid_size(num_segments);
            detail::row_reduce_dynamic_block_kernel<<<blocks, kNumThreads, 0, stream>>>(
                num_segments, segment_offset, input, layout, transform_op, reduction_op, identity, output_op);
        }
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    });
}

} // namespace at::native::dgx
