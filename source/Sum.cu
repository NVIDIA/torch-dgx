/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
#include "KernelLaunchUtils.cuh"
#include "NCCLUtils.h"
#include "RowReduction.cuh"
#include "StridedTensorUtils.cuh"

#include <c10/cuda/CUDAStream.h>

#include <ATen/AccumulateType.h>
#include <ATen/Dispatch_v2.h>
#include <ATen/ops/empty.h>

#include <torch/library.h>

#include <cub/block/block_reduce.cuh>
#include <cub/device/device_reduce.cuh>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/transform_iterator.h>

namespace at::native
{

namespace dgx
{

namespace
{

// PyTorch keeps non-fastest-dimension reductions mapped across outputs until
// each output has enough input work to justify splitting the reduction.
constexpr int64_t kDim1BlockReduceMinReductionSize = kNumThreads;
constexpr int kDim1TileColumns = 32;
constexpr int kDim1TileRows = kNumThreads / kDim1TileColumns;

template <typename InputIteratorT, typename accscalar_t>
void sum_async(int64_t numel, InputIteratorT in, accscalar_t* out)
{
    const ::cuda::std::plus<accscalar_t> reduction_op;

    std::vector<accscalar_t*> partials(c10::cuda::device_count());
    for (const auto device_id : c10::irange(c10::cuda::device_count()))
    {
        C10_CUDA_CHECK(cudaSetDevice(device_id));
        C10_CUDA_CHECK(
            cudaMallocAsync(&partials[device_id], sizeof(accscalar_t), c10::cuda::getCurrentCUDAStream(device_id)));
    }

    for (const auto device_id : c10::irange(c10::cuda::device_count()))
    {
        C10_CUDA_CHECK(cudaSetDevice(device_id));
        auto stream = c10::cuda::getCurrentCUDAStream(device_id);

        int64_t device_offset, device_numel;
        std::tie(device_offset, device_numel) = chunk(numel, device_id);
        auto device_in = in + device_offset;

        size_t scratch_size = 0;
        void* scratch_data = nullptr;
        C10_CUDA_CHECK(cub::DeviceReduce::Reduce(scratch_data, scratch_size, device_in, partials[device_id],
                                                 device_numel, reduction_op, accscalar_t(0), stream));
        C10_CUDA_CHECK(cudaMallocAsync(&scratch_data, scratch_size, stream));
        C10_CUDA_CHECK(cub::DeviceReduce::Reduce(scratch_data, scratch_size, device_in, partials[device_id],
                                                 device_numel, reduction_op, accscalar_t(0), stream));
        C10_CUDA_CHECK(cudaFreeAsync(scratch_data, stream));
    }

    nccl_reduce_partials(partials, out, 1);

    for (const auto device_id : c10::irange(c10::cuda::device_count()))
    {
        C10_CUDA_CHECK(cudaSetDevice(device_id));
        C10_CUDA_CHECK(cudaFreeAsync(partials[device_id], c10::cuda::getCurrentCUDAStream(device_id)));
    }

    // Broadcast GPU 0's NCCL completion to all device streams so callers can safely read out.
    constexpr int merge_device_id = 0;
    C10_CUDA_CHECK(cudaSetDevice(merge_device_id));
    cudaEvent_t merge_event;
    C10_CUDA_CHECK(cudaEventCreateWithFlags(&merge_event, cudaEventDisableTiming));
    C10_CUDA_CHECK(cudaEventRecord(merge_event, c10::cuda::getCurrentCUDAStream(merge_device_id)));
    for (const auto device_id : c10::irange(1, c10::cuda::device_count()))
    {
        C10_CUDA_CHECK(cudaSetDevice(device_id));
        C10_CUDA_CHECK(cudaStreamWaitEvent(c10::cuda::getCurrentCUDAStream(device_id), merge_event));
    }
    C10_CUDA_CHECK(cudaEventDestroy(merge_event));
}

template <typename T>
struct ScalarDivideOp
{
    ScalarDivideOp(T denominator) : denominator_(denominator)
    {
    }

    __host__ __device__ T operator()(const T& x) const
    {
        return x / denominator_;
    }

private:
    const T denominator_;
};

template <typename scalar_t, typename InputIteratorT, typename accscalar_t>
void mean_async(int64_t numel, InputIteratorT in, accscalar_t* out)
{
    const ::cuda::std::plus<accscalar_t> reduction_op;
    const ScalarDivideOp<scalar_t> transform_op(numel);

    std::vector<accscalar_t*> partials(c10::cuda::device_count());
    for (const auto device_id : c10::irange(c10::cuda::device_count()))
    {
        C10_CUDA_CHECK(cudaSetDevice(device_id));
        C10_CUDA_CHECK(
            cudaMallocAsync(&partials[device_id], sizeof(accscalar_t), c10::cuda::getCurrentCUDAStream(device_id)));
    }

    for (const auto device_id : c10::irange(c10::cuda::device_count()))
    {
        C10_CUDA_CHECK(cudaSetDevice(device_id));
        auto stream = c10::cuda::getCurrentCUDAStream(device_id);

        int64_t device_offset, device_numel;
        std::tie(device_offset, device_numel) = chunk(numel, device_id);
        auto device_in = in + device_offset;

        size_t scratch_size = 0;
        void* scratch_data = nullptr;
        C10_CUDA_CHECK(cub::DeviceReduce::TransformReduce(scratch_data, scratch_size, device_in, partials[device_id],
                                                          device_numel, reduction_op, transform_op, accscalar_t(0),
                                                          stream));
        C10_CUDA_CHECK(cudaMallocAsync(&scratch_data, scratch_size, stream));
        C10_CUDA_CHECK(cub::DeviceReduce::TransformReduce(scratch_data, scratch_size, device_in, partials[device_id],
                                                          device_numel, reduction_op, transform_op, accscalar_t(0),
                                                          stream));
        C10_CUDA_CHECK(cudaFreeAsync(scratch_data, stream));
    }

    nccl_reduce_partials(partials, out, 1);

    for (const auto device_id : c10::irange(c10::cuda::device_count()))
    {
        C10_CUDA_CHECK(cudaSetDevice(device_id));
        C10_CUDA_CHECK(cudaFreeAsync(partials[device_id], c10::cuda::getCurrentCUDAStream(device_id)));
    }

    constexpr int merge_device_id = 0;
    C10_CUDA_CHECK(cudaSetDevice(merge_device_id));
    cudaEvent_t merge_event;
    C10_CUDA_CHECK(cudaEventCreateWithFlags(&merge_event, cudaEventDisableTiming));
    C10_CUDA_CHECK(cudaEventRecord(merge_event, c10::cuda::getCurrentCUDAStream(merge_device_id)));
    for (const auto device_id : c10::irange(1, c10::cuda::device_count()))
    {
        C10_CUDA_CHECK(cudaSetDevice(device_id));
        C10_CUDA_CHECK(cudaStreamWaitEvent(c10::cuda::getCurrentCUDAStream(device_id), merge_event));
    }
    C10_CUDA_CHECK(cudaEventDestroy(merge_event));
}

template <typename input_t, typename acc_t>
struct RowSumTransform
{
    __device__ acc_t operator()(input_t value, int64_t) const
    {
        return static_cast<acc_t>(value);
    }
};

template <typename output_t, typename acc_t>
struct RowSumOutput
{
    output_t* output;

    template <typename Offsets>
    __device__ void operator()(const Offsets& offsets, acc_t aggregate) const
    {
        output[offsets[1]] = static_cast<output_t>(aggregate);
    }
};

template <typename output_t, typename acc_t>
struct RowMeanOutput
{
    output_t* output;
    int64_t reduction_size;

    template <typename Offsets>
    __device__ void operator()(const Offsets& offsets, acc_t aggregate) const
    {
        output[offsets[1]] = static_cast<output_t>(aggregate / static_cast<acc_t>(reduction_size));
    }
};

template <typename scalar_t, typename accscalar_t>
__global__ void sum_kernel_2d(int64_t count,
                              int64_t row_offset,
                              int64_t row_count,
                              const PackedTensorAccessor64<scalar_t, 2, RestrictPtrTraits> self,
                              accscalar_t* out)
{
    auto col = threadIdx.x + blockIdx.x * blockDim.x;
    if (col < count)
    {
        accscalar_t acc = 0;
        for (auto row = row_offset; row < row_offset + row_count; ++row)
            acc += static_cast<accscalar_t>(self[row][col]);
        out[col] = acc;
    }
}

template <typename accscalar_t>
__device__ accscalar_t block_sum(accscalar_t value)
{
    using BlockReduce = cub::BlockReduce<accscalar_t, kNumThreads>;
    __shared__ typename BlockReduce::TempStorage temp_storage;
    return BlockReduce(temp_storage).Sum(value);
}

template <typename scalar_t, typename accscalar_t>
__global__ void sum_kernel_3d_dim0(int64_t count,
                                   int64_t offset,
                                   const PackedTensorAccessor64<scalar_t, 3, RestrictPtrTraits> self,
                                   accscalar_t* out,
                                   StridedOffsetCalculator<1> output_offsets)
{
    auto output_index = threadIdx.x + blockIdx.x * blockDim.x;
    if (output_index < count)
    {
        auto linear_index = output_index + offset;
        auto j = linear_index / self.size(2);
        auto k = linear_index % self.size(2);

        accscalar_t acc = 0;
#pragma unroll
        for (auto i = 0; i < self.size(0); ++i)
        {
            acc += static_cast<accscalar_t>(self[i][j][k]);
        }
        out[output_offsets.get(linear_index)[0]] = acc;
    }
}

template <bool block_reduce = true, typename scalar_t, typename accscalar_t>
__global__ void sum_kernel_3d_dim1(int64_t count,
                                   int64_t offset,
                                   const PackedTensorAccessor64<scalar_t, 3, RestrictPtrTraits> self,
                                   accscalar_t* out)
{
    if constexpr (block_reduce)
    {
        auto output_index = blockIdx.x;
        if (output_index < count)
        {
            auto linear_index = output_index + offset;
            auto i = linear_index / self.size(2);
            auto k = linear_index % self.size(2);

            accscalar_t acc = 0;
            for (auto j = threadIdx.x; j < self.size(1); j += blockDim.x)
            {
                acc += static_cast<accscalar_t>(self[i][j][k]);
            }
            acc = block_sum(acc);
            if (threadIdx.x == 0)
            {
                out[output_index] = acc;
            }
        }
    }
    else
    {
        auto output_index = threadIdx.x + blockIdx.x * blockDim.x;
        if (output_index < count)
        {
            auto linear_index = output_index + offset;
            auto i = linear_index / self.size(2);
            auto k = linear_index % self.size(2);

            accscalar_t acc = 0;
            for (auto j = 0; j < self.size(1); ++j)
            {
                acc += static_cast<accscalar_t>(self[i][j][k]);
            }
            out[output_index] = acc;
        }
    }
}

template <typename scalar_t, typename accscalar_t>
__global__ __launch_bounds__(kNumThreads) void sum_kernel_3d_dim1_tiled(
    int64_t count, int64_t offset, const PackedTensorAccessor64<scalar_t, 3, RestrictPtrTraits> self, accscalar_t* out)
{
    // A warp reads adjacent features; the eight warps sum separate sequence rows.
    // Tile the existing flattened (batch, feature) shard, including partial tiles
    // at device and batch boundaries, without changing which GPU owns an output.
    __shared__ accscalar_t partials[kDim1TileRows][kDim1TileColumns];
    const auto column = threadIdx.x % kDim1TileColumns;
    const auto row = threadIdx.x / kDim1TileColumns;
    const auto output_index = static_cast<int64_t>(blockIdx.x) * kDim1TileColumns + column;
    const auto linear_index = offset + output_index;
    accscalar_t acc = 0;
    if (output_index < count)
    {
        const auto tid_x = linear_index / self.size(2);
        const auto tid_z = linear_index % self.size(2);
        for (int64_t tid_y = row; tid_y < self.size(1); tid_y += kDim1TileRows)
        {
            acc += static_cast<accscalar_t>(self[tid_x][tid_y][tid_z]);
        }
    }
    partials[row][column] = acc;
    __syncthreads();
    if (row == 0 && output_index < count)
    {
#pragma unroll
        for (int r = 1; r < kDim1TileRows; ++r)
        {
            acc += partials[r][column];
        }
        out[output_index] = acc;
    }
}

} // namespace

void sum_dimension(const Tensor& input, int64_t dim, Tensor& output)
{
    const auto layout =
        make_reduction_layout<2>(input.sizes(), dim, std::array<IntArrayRef, 2>{ input.strides(), output.strides() });
    AT_DISPATCH_V2(input.scalar_type(), "sum_out", AT_WRAP([&] {
                       using accscalar_t = at::acc_type<scalar_t, true>;
                       row_reduce<scalar_t, accscalar_t>(
                           input.const_data_ptr<scalar_t>(), layout, RowSumTransform<scalar_t, accscalar_t>(),
                           ::cuda::std::plus<accscalar_t>(), accscalar_t(0),
                           RowSumOutput<scalar_t, accscalar_t>{ output.data_ptr<scalar_t>() });
                   }),
                   AT_EXPAND(AT_ALL_TYPES));
}

Tensor& sum_out(const Tensor& self, OptionalIntArrayRef dim, bool keepdim, std::optional<ScalarType> dtype, Tensor& out)
{
    int device = cudaInvalidDeviceId;
    C10_CUDA_CHECK(cudaGetDevice(&device));

    // Floating sums use the output dtype; integer/bool sums can accumulate without promoting the input.
    const auto input_dtype = dtype.value_or(self.is_floating_point() ? out.scalar_type() : self.scalar_type());
    auto in = self.to(input_dtype);

    if (!out.numel())
    {
        C10_CUDA_CHECK(cudaSetDevice(device));
        return out;
    }

    if (!dim || (*dim).empty() || out.numel() == 1)
    {
        if (in.scalar_type() == kBool)
        {
            using scalar_t = bool;
            using accscalar_t = int64_t;
            auto output = out.data_ptr<accscalar_t>();
            if (in.is_contiguous())
            {
                sum_async(in.numel(), in.const_data_ptr<scalar_t>(), output);
            }
            else
            {
                auto logical_indices = thrust::make_counting_iterator<int64_t>(0);
                const auto offsets =
                    make_strided_offset_calculator<1>(in.sizes(), std::array<IntArrayRef, 1>{ in.strides() });
                StridedInputOp<scalar_t> transform_op{ in.const_data_ptr<scalar_t>(), offsets };
                auto input = thrust::make_transform_iterator(logical_indices, transform_op);
                sum_async(in.numel(), input, output);
            }
        }
        else
        {
            AT_DISPATCH_V2(in.scalar_type(), "sum_out", AT_WRAP([&] {
                               using accscalar_t = at::acc_type<scalar_t, true>;
                               auto output = out.data_ptr<accscalar_t>();
                               if (in.is_contiguous())
                               {
                                   sum_async(in.numel(), in.const_data_ptr<scalar_t>(), output);
                               }
                               else
                               {
                                   auto logical_indices = thrust::make_counting_iterator<int64_t>(0);
                                   const auto offsets = make_strided_offset_calculator<1>(
                                       in.sizes(), std::array<IntArrayRef, 1>{ in.strides() });
                                   StridedInputOp<scalar_t> transform_op{ in.const_data_ptr<scalar_t>(), offsets };
                                   auto input = thrust::make_transform_iterator(logical_indices, transform_op);
                                   sum_async(in.numel(), input, output);
                               }
                           }),
                           AT_EXPAND(AT_ALL_TYPES));
        }
    }
    else if ((*dim).size() == 1 && in.size((*dim)[0]) == 1 && in.is_contiguous() && out.is_contiguous())
    {
        parallel_for_each([&](int device, cudaStream_t stream) {
            AT_DISPATCH_V2(
                out.scalar_type(), "sum_out", AT_WRAP([&] {
                    int64_t device_offset, device_numel;
                    std::tie(device_offset, device_numel) = chunk(in.numel(), device);
                    const auto input = in.const_data_ptr<scalar_t>();
                    auto output = out.data_ptr<scalar_t>();
                    C10_CUDA_CHECK(cudaMemcpyAsync(output + device_offset, input + device_offset,
                                                   device_numel * sizeof(scalar_t), cudaMemcpyDefault, stream));
                }),
                AT_EXPAND(AT_ALL_TYPES));
        });
    }
    else if ((*dim).size() == 1 && (*dim)[0] == 0 && (in.dim() == 2 || in.dim() == 3))
    {
        if (in.dim() == 2)
        {
            AT_DISPATCH_V2(
                in.scalar_type(), "sum_out", AT_WRAP([&] {
                    using accscalar_t = at::acc_type<scalar_t, true>;
                    // Bias gradients reduce rows. Compute each GPU's local-row partial and combine the
                    // partial vectors instead of making every GPU scan the full managed tensor.
                    const auto num_cols = in.size(1);
                    const auto padded_numel = nccl_reduce_scatter_padded_numel(num_cols);

                    std::vector<accscalar_t*> partials(c10::cuda::device_count());

                    parallel_for_each_async([&](int device, cudaStream_t stream) {
                        int64_t row_offset, row_count;
                        std::tie(row_offset, row_count) = chunk(in.size(0), device);
                        C10_CUDA_CHECK(cudaMallocAsync(&partials[device], padded_numel * sizeof(accscalar_t), stream));
                        if (padded_numel > num_cols)
                            C10_CUDA_CHECK(cudaMemsetAsync(partials[device] + num_cols, 0,
                                                           (padded_numel - num_cols) * sizeof(accscalar_t), stream));
                        const auto num_blocks = static_cast<unsigned int>(::cuda::ceil_div(num_cols, kNumThreads));
                        sum_kernel_2d<<<num_blocks, kNumThreads, 0, stream>>>(
                            num_cols, row_offset, row_count, in.packed_accessor64<scalar_t, 2, RestrictPtrTraits>(),
                            partials[device]);
                        C10_CUDA_KERNEL_LAUNCH_CHECK();
                    });

                    auto result = out.is_contiguous() ? out : at::empty(out.sizes(), out.options());
                    nccl_reduce_scatter_partials(partials, result.data_ptr<accscalar_t>(), num_cols);

                    parallel_for_each_async([&](int device, cudaStream_t stream) {
                        C10_CUDA_CHECK(cudaFreeAsync(partials[device], stream));
                    });
                    merge_streams();
                    if (!out.is_same(result))
                        out.copy_(result);
                }),
                AT_EXPAND(AT_FLOATING_TYPES));
        }
        else if (in.dim() == 3)
        {
            parallel_for_each([&](int device, cudaStream_t stream) {
                int64_t device_offset, device_numel;
                std::tie(device_offset, device_numel) = chunk(in.size(1) * in.size(2), device);
                const unsigned int device_num_blocks = ::cuda::ceil_div(device_numel, kNumThreads);

                AT_DISPATCH_V2(
                    in.scalar_type(), "sum_out", AT_WRAP([&] {
                        using accscalar_t = at::acc_type<scalar_t, true>;
                        auto output_offsets =
                            make_strided_offset_calculator<1>(out.sizes(), std::array<IntArrayRef, 1>{ out.strides() });
                        sum_kernel_3d_dim0<<<device_num_blocks, kNumThreads, 0, stream>>>(
                            device_numel, device_offset, in.packed_accessor64<scalar_t, 3, RestrictPtrTraits>(),
                            out.data_ptr<accscalar_t>(), output_offsets);
                    }),
                    AT_EXPAND(AT_FLOATING_TYPES));
                C10_CUDA_KERNEL_LAUNCH_CHECK();
            });
        }
        else
        {
            TORCH_CHECK(false, "Unsupported configuration in sum_out");
        }
    }
    else if ((*dim).size() == 1 && in.dim() == 3 && ((*dim)[0] == 1 || (*dim)[0] == -2))
    {
        AT_DISPATCH_V2(
            in.scalar_type(), "sum_out", AT_WRAP([&] {
                using accscalar_t = at::acc_type<scalar_t, true>;
                const auto output_numel = in.size(0) * in.size(2);
                if (!output_numel)
                {
                    return;
                }
                const bool use_block_reduce = in.size(1) >= kDim1BlockReduceMinReductionSize;
                const bool use_tiled_reduce = use_block_reduce && in.stride(2) == 1 && in.size(2) >= kDim1TileColumns;
                // Each output belongs to one GPU and already includes the full sequence reduction.
                // Keep that ownership through a compact local buffer and its managed-output copy.
                auto result = out.is_contiguous() ? out : at::empty(out.sizes(), out.options());

                parallel_for_each([&](int device, cudaStream_t stream) {
                    int64_t device_offset, device_numel;
                    std::tie(device_offset, device_numel) = chunk(output_numel, device);
                    if (!device_numel)
                    {
                        return;
                    }
                    accscalar_t* device_output = nullptr;
                    C10_CUDA_CHECK(cudaMallocAsync(&device_output, device_numel * sizeof(accscalar_t), stream));

                    if (use_tiled_reduce)
                    {
                        const auto num_blocks = ::cuda::ceil_div(device_numel, kDim1TileColumns);
                        sum_kernel_3d_dim1_tiled<<<num_blocks, kNumThreads, 0, stream>>>(
                            device_numel, device_offset, in.packed_accessor64<scalar_t, 3, RestrictPtrTraits>(),
                            device_output);
                    }
                    else if (use_block_reduce)
                    {
                        const auto num_blocks = static_cast<unsigned int>(device_numel);
                        sum_kernel_3d_dim1<<<num_blocks, kNumThreads, 0, stream>>>(
                            device_numel, device_offset, in.packed_accessor64<scalar_t, 3, RestrictPtrTraits>(),
                            device_output);
                    }
                    else
                    {
                        const auto num_blocks = ::cuda::ceil_div(device_numel, kNumThreads);
                        sum_kernel_3d_dim1<false><<<num_blocks, kNumThreads, 0, stream>>>(
                            device_numel, device_offset, in.packed_accessor64<scalar_t, 3, RestrictPtrTraits>(),
                            device_output);
                    }
                    C10_CUDA_KERNEL_LAUNCH_CHECK();
                    C10_CUDA_CHECK(cudaMemcpyAsync(result.data_ptr<accscalar_t>() + device_offset, device_output,
                                                   device_numel * sizeof(accscalar_t), cudaMemcpyDefault, stream));
                    C10_CUDA_CHECK(cudaFreeAsync(device_output, stream));
                });
                if (!out.is_same(result))
                {
                    out.copy_(result);
                }
            }),
            AT_EXPAND(AT_FLOATING_TYPES));
    }
    else if ((*dim).size() == 1)
    {
        sum_dimension(in, c10::maybe_wrap_dim((*dim)[0], in.dim()), out);
    }
    else
    {
        // ExpandBackward for attention biases reduces broadcast batch/head/query
        // axes together. Retain singleton axes between passes so indices remain
        // stable, and write the final pass directly into the requested output.
        auto current = in;
        for (const auto index : c10::irange((*dim).size()))
        {
            const auto axis = c10::maybe_wrap_dim((*dim)[index], in.dim());
            auto shape = current.sizes().vec();
            shape[axis] = 1;
            auto next = index + 1 == (*dim).size() ? out.view(shape) : at::empty(shape, out.options());
            sum_dimension(current, axis, next);
            current = std::move(next);
        }
    }

    C10_CUDA_CHECK(cudaSetDevice(device));

    return out;
}

Tensor& mean_out(const Tensor& self, OptionalIntArrayRef dim, bool keepdim, std::optional<ScalarType> dtype, Tensor& out)
{
    int device = cudaInvalidDeviceId;
    C10_CUDA_CHECK(cudaGetDevice(&device));

    auto in = dtype ? self.to(dtype.value()) : self;

    if (!dim || (*dim).empty() || out.numel() == 1)
    {
        AT_DISPATCH_V2(in.scalar_type(), "mean_out", AT_WRAP([&] {
                           auto output = out.data_ptr<scalar_t>();
                           if (in.is_contiguous())
                           {
                               mean_async<scalar_t>(in.numel(), in.const_data_ptr<scalar_t>(), output);
                           }
                           else
                           {
                               auto logical_indices = thrust::make_counting_iterator<int64_t>(0);
                               const auto offsets = make_strided_offset_calculator<1>(
                                   in.sizes(), std::array<IntArrayRef, 1>{ in.strides() });
                               StridedInputOp<scalar_t> transform_op{ in.const_data_ptr<scalar_t>(), offsets };
                               auto input = thrust::make_transform_iterator(logical_indices, transform_op);
                               mean_async<scalar_t>(in.numel(), input, output);
                           }
                       }),
                       AT_EXPAND(AT_FLOATING_TYPES));
    }
    else if ((*dim).size() == 1 && in.size((*dim)[0]) == 1)
    {
        parallel_for_each([&](int device, cudaStream_t stream) {
            AT_DISPATCH_V2(
                out.scalar_type(), "mean_out", AT_WRAP([&] {
                    int64_t device_offset, device_numel;
                    std::tie(device_offset, device_numel) = chunk(in.numel(), device);
                    const auto input = in.const_data_ptr<scalar_t>();
                    auto output = out.data_ptr<scalar_t>();
                    C10_CUDA_CHECK(cudaMemcpyAsync(output + device_offset, input + device_offset,
                                                   device_numel * sizeof(scalar_t), cudaMemcpyDefault, stream));
                }),
                AT_EXPAND(AT_FLOATING_TYPES));
        });
    }
    else if ((*dim).size() == 1 && ((*dim)[0] == -1 || (*dim)[0] == (in.dim() - 1)))
    {
        const auto layout = make_reduction_layout<2>(
            in.sizes(), in.dim() - 1, std::array<IntArrayRef, 2>{ in.strides(), out.strides() });
        AT_DISPATCH_V2(in.scalar_type(), "mean_out", AT_WRAP([&] {
                           using accscalar_t = at::acc_type<scalar_t, true>;
                           row_reduce<scalar_t, accscalar_t>(
                               in.const_data_ptr<scalar_t>(), layout, RowSumTransform<scalar_t, accscalar_t>(),
                               ::cuda::std::plus<accscalar_t>(), accscalar_t(0),
                               RowMeanOutput<scalar_t, accscalar_t>{ out.data_ptr<scalar_t>(), in.size(-1) });
                       }),
                       AT_EXPAND(AT_FLOATING_TYPES));
    }
    else
    {
        TORCH_CHECK(false, "Unsupported configuration in mean_out");
    }

    C10_CUDA_CHECK(cudaSetDevice(device));

    return out;
}

TORCH_LIBRARY_IMPL(aten, PrivateUse1, m)
{
    m.impl(TORCH_SELECTIVE_NAME("aten::sum.IntList_out"), TORCH_FN(sum_out));
    m.impl(TORCH_SELECTIVE_NAME("aten::mean.out"), TORCH_FN(mean_out));
}

} // namespace dgx

} // namespace at::native
