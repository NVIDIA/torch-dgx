/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
#include "KernelLaunchUtils.cuh"

#include <c10/cuda/CUDAStream.h>

#include <ATen/Dispatch_v2.h>
#include <ATen/native/ReduceOpsUtils.h>
#include <ATen/ops/empty.h>

#include <torch/library.h>

#include <cub/device/device_reduce.cuh>
#include <cub/device/device_scan.cuh>
#include <cub/util_type.cuh>
#include <cub/warp/warp_scan.cuh>

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
    int device = cudaInvalidDeviceId;
    C10_CUDA_CHECK(cudaGetDevice(&device));

    TORCH_CHECK(self.dim() == 1, "dgx cumsum only supports 1D inputs")
    TORCH_CHECK(dim == 0, "dgx cumsum only supports cumulative summation along the zeroth dimension");
    TORCH_CHECK(c10::cuda::device_count() <= kWarpSize, "dgx cumsum supports at most ", kWarpSize, " CUDA devices");

    std::vector<int64_t> device_numels(c10::cuda::device_count());
    for (const auto device_id : c10::irange(c10::cuda::device_count()))
    {
        std::tie(std::ignore, device_numels[device_id]) = chunk(self.numel(), device_id);
    }

    auto in = self.to(get_dtype_from_self(self, dtype, true));

    C10_CUDA_CHECK(cudaSetDevice(0));
    auto block_totals = at::empty({ c10::cuda::device_count() }, out.options());
    auto block_offsets = at::empty({ c10::cuda::device_count() }, out.options());

    AT_DISPATCH_V2(out.scalar_type(), "cumsum_out", AT_WRAP([&] {
                       const auto input = in.const_data_ptr<scalar_t>();
                       auto output = out.data_ptr<scalar_t>();
                       inclusive_sum_async(input, output, device_numels.data(), block_totals.data_ptr<scalar_t>(),
                                           block_offsets.data_ptr<scalar_t>());
                   }),
                   AT_EXPAND(AT_ALL_TYPES));

    C10_CUDA_CHECK(cudaSetDevice(device));

    return out;
}

TORCH_LIBRARY_IMPL(aten, PrivateUse1, m)
{
    m.impl(TORCH_SELECTIVE_NAME("aten::cumsum.out"), TORCH_FN(cumsum_out));
}

} // namespace dgx

} // namespace at::native
