/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
#include "ScalarReductionUtils.h"
#include "TensorUtils.h"

#include <c10/cuda/CUDAStream.h>

#include <vector>

namespace at::native::dgx
{

namespace
{

constexpr c10::DeviceIndex kMergeDevice = 0;

template <typename Reduce>
void run_reduce(Reduce reduce, cudaStream_t stream)
{
    size_t temp_storage_bytes = 0;
    C10_CUDA_CHECK(reduce(nullptr, temp_storage_bytes));
    void* temp_storage = nullptr;
    C10_CUDA_CHECK(cudaMallocAsync(&temp_storage, temp_storage_bytes, stream));
    C10_CUDA_CHECK(reduce(temp_storage, temp_storage_bytes));
    C10_CUDA_CHECK(cudaFreeAsync(temp_storage, stream));
}

} // namespace

void scalar_reduce(int64_t numel, ScalarReductionFn reduce, ScalarReductionMergeFn merge, bool broadcast_result)
{
    // The operator restores the device after releasing its temporary tensors.
    std::vector<cudaEvent_t> events(c10::cuda::device_count(), nullptr);
    cudaEvent_t merge_event = nullptr;

    for (const auto device : c10::irange(c10::cuda::device_count()))
    {
        C10_CUDA_CHECK(cudaSetDevice(device));
        auto stream = c10::cuda::getCurrentCUDAStream(device);
        int64_t offset, count;
        std::tie(offset, count) = chunk(numel, device);
        run_reduce(
            [&](void* temp_storage, size_t& temp_storage_bytes) {
                return reduce(temp_storage, temp_storage_bytes, device, offset, count, stream);
            },
            stream);
    }

    for (const auto device : c10::irange(1, c10::cuda::device_count()))
    {
        C10_CUDA_CHECK(cudaSetDevice(device));
        auto stream = c10::cuda::getCurrentCUDAStream(device);
        C10_CUDA_CHECK(cudaEventCreateWithFlags(&events[device], cudaEventDisableTiming));
        C10_CUDA_CHECK(cudaEventRecord(events[device], stream));
    }

    C10_CUDA_CHECK(cudaSetDevice(kMergeDevice));
    if (broadcast_result)
        C10_CUDA_CHECK(cudaEventCreateWithFlags(&merge_event, cudaEventDisableTiming));
    auto stream = c10::cuda::getCurrentCUDAStream(kMergeDevice);
    // GPU 0's local reduction is already ordered on this stream.
    for (const auto device : c10::irange(1, c10::cuda::device_count()))
    {
        C10_CUDA_CHECK(cudaStreamWaitEvent(stream, events[device]));
    }
    run_reduce(
        [&](void* temp_storage, size_t& temp_storage_bytes) {
            return merge(temp_storage, temp_storage_bytes, c10::cuda::device_count(), stream);
        },
        stream);

    if (merge_event)
        C10_CUDA_CHECK(cudaEventRecord(merge_event, stream));
    for (const auto device : c10::irange(1, c10::cuda::device_count()))
    {
        C10_CUDA_CHECK(cudaSetDevice(device));
        if (merge_event)
        {
            auto stream = c10::cuda::getCurrentCUDAStream(device);
            C10_CUDA_CHECK(cudaStreamWaitEvent(stream, merge_event));
        }
        C10_CUDA_CHECK(cudaEventDestroy(events[device]));
    }
    if (merge_event)
    {
        C10_CUDA_CHECK(cudaSetDevice(kMergeDevice));
        C10_CUDA_CHECK(cudaEventDestroy(merge_event));
    }
}

} // namespace at::native::dgx
