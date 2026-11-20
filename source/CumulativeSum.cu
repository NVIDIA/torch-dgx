/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
#include "KernelLaunchUtils.cuh"

#include <c10/cuda/CUDAStream.h>

#include <ATen/AccumulateType.h>
#include <ATen/Dispatch_v2.h>
#include <ATen/MemoryOverlap.h>
#include <ATen/WrapDimUtils.h>
#include <ATen/native/Resize.h>
#include <ATen/ops/empty.h>

#include <torch/library.h>

#include <cub/block/block_scan.cuh>
#include <cub/device/device_reduce.cuh>
#include <cub/device/device_scan.cuh>
#include <cub/util_type.cuh>
#include <cub/warp/warp_scan.cuh>
#include <cuda/std/functional>

// Define utility macro used to call CUB functions that use dynamic temporary storage
#ifndef CUB_LAUNCH
#    ifdef _WIN32
#        define CUB_LAUNCH(func, stream, ...)                                                                          \
            size_t scratch_size = 0;                                                                                   \
            void* scratch_data = nullptr;                                                                              \
            C10_CUDA_CHECK(cub::func(scratch_data, scratch_size, __VA_ARGS__, stream));                                \
            C10_CUDA_CHECK(cudaMallocAsync(&scratch_data, scratch_size, stream));                                      \
            C10_CUDA_CHECK(cub::func(scratch_data, scratch_size, __VA_ARGS__, stream));                                \
            C10_CUDA_CHECK(cudaFreeAsync(scratch_data, stream));
#    else
#        define CUB_LAUNCH(func, stream, args...)                                                                      \
            size_t scratch_size = 0;                                                                                   \
            void* scratch_data = nullptr;                                                                              \
            C10_CUDA_CHECK(cub::func(scratch_data, scratch_size, args, stream));                                       \
            C10_CUDA_CHECK(cudaMallocAsync(&scratch_data, scratch_size, stream));                                      \
            C10_CUDA_CHECK(cub::func(scratch_data, scratch_size, args, stream));                                       \
            C10_CUDA_CHECK(cudaFreeAsync(scratch_data, stream));
#    endif
#endif

namespace at::native
{

namespace dgx
{

namespace
{

constexpr int kWarpSize = 32;

template <typename scalar_t>
__global__ void cumprod_rows_kernel(
    const scalar_t* input, scalar_t* output, int64_t size, int64_t row_offset, int64_t row_count)
{
    using acc_t = acc_type<scalar_t, true>;
    using BlockScan = cub::BlockScan<acc_t, kNumThreads>;
    __shared__ typename BlockScan::TempStorage temp_storage;
    for (int64_t row = blockIdx.x; row < row_count; row += gridDim.x)
    {
        acc_t carry = acc_t(1);
        const int64_t offset = (row_offset + row) * size;
        for (int64_t start = 0; start < size; start += blockDim.x)
        {
            const int64_t col = start + threadIdx.x;
            acc_t value = col < size ? static_cast<acc_t>(input[offset + col]) : acc_t(1);
            acc_t total;
            BlockScan(temp_storage).InclusiveScan(value, value, ::cuda::std::multiplies<acc_t>{}, total);
            if (col < size)
            {
                output[offset + col] = static_cast<scalar_t>(carry * value);
            }
            carry *= total;
            __syncthreads();
        }
    }
}

template <typename scalar_t>
__global__ void cumsum_dim_kernel(
    int64_t row_offset, int64_t num_rows, int64_t dim_size, int64_t inner, const scalar_t* input, scalar_t* output)
{
    using acc_t = acc_type<scalar_t, true>;
    using BlockScan = cub::BlockScan<acc_t, kNumThreads>;
    __shared__ typename BlockScan::TempStorage temp_storage;
    __shared__ acc_t carry;

    for (int64_t row = blockIdx.x; row < num_rows; row += gridDim.x)
    {
        const auto index = row_offset + row;
        const auto base = (index / inner) * dim_size * inner + index % inner;
        if (threadIdx.x == 0)
        {
            carry = acc_t(0);
        }
        __syncthreads();

        for (int64_t start = 0; start < dim_size; start += kNumThreads)
        {
            const auto column = start + threadIdx.x;
            const acc_t value = column < dim_size ? static_cast<acc_t>(input[base + column * inner]) : acc_t(0);
            acc_t prefix, total;
            BlockScan(temp_storage).InclusiveSum(value, prefix, total);
            if (column < dim_size)
            {
                output[base + column * inner] = static_cast<scalar_t>(carry + prefix);
            }
            // Finish reading carry and scan storage before updating either.
            __syncthreads();
            if (threadIdx.x == 0)
            {
                carry += total;
            }
            __syncthreads();
        }
    }
}

template <typename T>
__global__ void exclusive_warp_sum_kernel(const T* input, T* output, int64_t count)
{
    using WarpScan = cub::WarpScan<T, kWarpSize>;
    __shared__ typename WarpScan::TempStorage temp_storage;

    const int64_t lane = threadIdx.x;
    const T value = lane < count ? input[lane] : T(0);
    T exclusive_sum;
    WarpScan(temp_storage).ExclusiveSum(value, exclusive_sum);
    if (lane < count)
    {
        output[lane] = exclusive_sum;
    }
}

template <typename T>
void inclusive_sum_async(const T* in, T* out, int64_t* counts, T* block_totals, T* block_offsets)
{
    constexpr int merge_device_id = 0;
    std::vector<cudaEvent_t> events(c10::cuda::device_count());
    std::vector<const T*> device_ins(c10::cuda::device_count());
    std::vector<T*> device_outs(c10::cuda::device_count());
    {
        const T* device_in = in;
        T* device_out = out;
        for (const auto device_id : c10::irange(c10::cuda::device_count()))
        {
            device_ins[device_id] = device_in;
            device_outs[device_id] = device_out;
            device_in += counts[device_id];
            device_out += counts[device_id];
        }
    }

    // Each device writes its total, including zero for an empty shard, before recording readiness.
    parallel_for_each_async([&](int device_id, cudaStream_t stream) {
        if (counts[device_id] > 0)
        {
            CUB_LAUNCH(DeviceReduce::Sum, stream, device_ins[device_id], block_totals + device_id, counts[device_id]);
        }
        else
        {
            C10_CUDA_CHECK(cudaMemsetAsync(block_totals + device_id, 0, sizeof(T), stream));
        }
    });

    for (const auto device_id : c10::irange(1, c10::cuda::device_count()))
    {
        C10_CUDA_CHECK(cudaSetDevice(device_id));
        auto stream = c10::cuda::getCurrentCUDAStream(device_id);
        C10_CUDA_CHECK(cudaEventCreateWithFlags(&events[device_id], cudaEventDisableTiming));
        C10_CUDA_CHECK(cudaEventRecord(events[device_id], stream));
    }

    // Run a small exclusive sum on device 0
    C10_CUDA_CHECK(cudaSetDevice(merge_device_id));
    auto merge_stream = c10::cuda::getCurrentCUDAStream(merge_device_id);
    for (const auto device_id : c10::irange(1, c10::cuda::device_count()))
    {
        C10_CUDA_CHECK(cudaStreamWaitEvent(merge_stream, events[device_id]));
    }
    exclusive_warp_sum_kernel<<<1, kWarpSize, 0, merge_stream>>>(block_totals, block_offsets, c10::cuda::device_count());
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    cudaEvent_t merge_event = nullptr;
    C10_CUDA_CHECK(cudaEventCreateWithFlags(&merge_event, cudaEventDisableTiming));
    C10_CUDA_CHECK(cudaEventRecord(merge_event, merge_stream));

    for (const auto device_id : c10::irange(1, c10::cuda::device_count()))
    {
        if (counts[device_id] == 0)
            continue;
        C10_CUDA_CHECK(cudaSetDevice(device_id));
        auto stream = c10::cuda::getCurrentCUDAStream(device_id);
        C10_CUDA_CHECK(cudaStreamWaitEvent(stream, merge_event));
    }

    // Use the results of the exclusive sum as the init in a per-device InclusiveScan
    parallel_for_each_async([&](int device_id, cudaStream_t stream) {
        if (counts[device_id] == 0)
            return;
        CUB_LAUNCH(DeviceScan::InclusiveScanInit, stream, device_ins[device_id], device_outs[device_id],
                   ::cuda::std::plus<T>{}, cub::FutureValue<T>(&block_offsets[device_id]), counts[device_id]);
    });

    C10_CUDA_CHECK(cudaEventDestroy(merge_event));
    for (const auto device_id : c10::irange(1, c10::cuda::device_count()))
    {
        C10_CUDA_CHECK(cudaEventDestroy(events[device_id]));
    }

    merge_streams();
}

} // namespace

Tensor& cumsum_out(const Tensor& self, int64_t dim, std::optional<ScalarType> dtype, Tensor& out)
{
    dim = at::maybe_wrap_dim(dim, self.dim());
    TORCH_CHECK(out.device() == self.device(), "cumsum: output must be on the input device");
    TORCH_CHECK(!dtype || *dtype == out.scalar_type(), "cumsum: output must have the requested dtype");
    at::native::resize_output(out, self.sizes());
    at::assert_no_internal_overlap(out);
    at::assert_no_partial_overlap(out, self);
    if (self.numel() == 0)
    {
        return out;
    }

    auto in = self.to(out.scalar_type()).contiguous();
    auto result = out.is_contiguous() ? out : at::empty(self.sizes(), out.options());

    if (self.dim() != 1 || out.scalar_type() == kHalf || out.scalar_type() == kBFloat16 || out.is_complex())
    {
        const auto dim_size = self.dim() == 0 ? 1 : self.size(dim);
        int64_t inner = 1;
        for (int64_t axis = dim + 1; axis < self.dim(); ++axis)
        {
            inner *= self.size(axis);
        }
        const auto num_rows = self.numel() / dim_size;

        AT_DISPATCH_V2(out.scalar_type(), "cumsum_dim_out", AT_WRAP([&] {
                           parallel_for_each([&](int device, cudaStream_t stream) {
                               size_t offset, count;
                               std::tie(offset, count) = partition::range(num_rows, device);
                               if (count == 0)
                               {
                                   return;
                               }
                               const auto num_blocks = static_cast<unsigned int>(std::min<int64_t>(count, 65535));
                               cumsum_dim_kernel<<<num_blocks, kNumThreads, 0, stream>>>(offset, count, dim_size, inner,
                                                                                         in.const_data_ptr<scalar_t>(),
                                                                                         result.data_ptr<scalar_t>());
                               C10_CUDA_KERNEL_LAUNCH_CHECK();
                           });
                       }),
                       AT_EXPAND(AT_ALL_TYPES), kHalf, kBFloat16, kComplexFloat, kComplexDouble);

        if (!out.is_contiguous())
        {
            out.copy_(result);
        }
        return out;
    }

    TORCH_CHECK(c10::cuda::device_count() <= kWarpSize, "dgx cumsum supports at most ", kWarpSize, " CUDA devices");

    std::vector<int64_t> device_numels(c10::cuda::device_count());
    for (const auto device_id : c10::irange(c10::cuda::device_count()))
    {
        std::tie(std::ignore, device_numels[device_id]) = partition::range(self.numel(), device_id);
    }

    C10_CUDA_CHECK(cudaSetDevice(0));
    auto block_totals = at::empty({ c10::cuda::device_count() }, out.options());
    auto block_offsets = at::empty({ c10::cuda::device_count() }, out.options());

    AT_DISPATCH_V2(out.scalar_type(), "cumsum_out", AT_WRAP([&] {
                       const auto input = in.const_data_ptr<scalar_t>();
                       auto output = result.data_ptr<scalar_t>();
                       inclusive_sum_async(input, output, device_numels.data(), block_totals.data_ptr<scalar_t>(),
                                           block_offsets.data_ptr<scalar_t>());
                   }),
                   AT_EXPAND(AT_ALL_TYPES));

    if (!out.is_contiguous())
    {
        out.copy_(result);
    }
    return out;
}

Tensor& cumprod_out(const Tensor& self, int64_t dim, std::optional<ScalarType> dtype, Tensor& out)
{
    dim = maybe_wrap_dim(dim, self.dim());
    TORCH_CHECK(out.device() == self.device(), "cumprod: output must be on the input device");
    TORCH_CHECK(!dtype || *dtype == out.scalar_type(), "cumprod dtype must match the output dtype");
    out.resize_(self.sizes());
    assert_no_internal_overlap(out);
    if (!self.numel())
    {
        return out;
    }
    if (!self.dim())
    {
        return out.copy_(self);
    }

    // Scan independent contiguous rows, keeping complete rows on each GPU.
    // A separate result also makes out= aliases safe and supports strided outputs.
    const auto input = self.to(out.scalar_type()).movedim(dim, -1).contiguous();
    auto result = at::empty(input.sizes(), out.options());
    const int64_t size = input.size(-1);
    const int64_t rows = input.numel() / size;
    AT_DISPATCH_V2(out.scalar_type(), "cumprod_out", AT_WRAP([&] {
                       parallel_for_each([&](int device, cudaStream_t stream) {
                           size_t offset, count;
                           std::tie(offset, count) = partition::range(rows, device);
                           if (!count)
                           {
                               return;
                           }
                           const unsigned int blocks = static_cast<unsigned int>(std::min<int64_t>(count, 4096));
                           cumprod_rows_kernel<<<blocks, kNumThreads, 0, stream>>>(
                               input.const_data_ptr<scalar_t>(), result.data_ptr<scalar_t>(), size, offset, count);
                           C10_CUDA_KERNEL_LAUNCH_CHECK();
                       });
                   }),
                   AT_EXPAND(AT_ALL_TYPES), kBool, kHalf, kBFloat16, kComplexFloat, kComplexDouble);
    return out.copy_(result.movedim(-1, dim));
}

TORCH_LIBRARY_IMPL(aten, PrivateUse1, m)
{
    m.impl(TORCH_SELECTIVE_NAME("aten::cumsum.out"), TORCH_FN(cumsum_out));
    m.impl(TORCH_SELECTIVE_NAME("aten::cumprod.out"), TORCH_FN(cumprod_out));
}

} // namespace dgx

} // namespace at::native
