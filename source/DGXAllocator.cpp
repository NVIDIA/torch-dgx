/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
#include "DGXAllocator.h"

#include "CompatibilityUtils.h"

#include <c10/cuda/CUDAStream.h>
#include <c10/util/CallOnce.h>

#include <cuda_runtime_api.h>

#include <iostream>

// CUDA 13.0 adds support for managed memory pools
#if (CUDART_VERSION >= 13000)
static c10::once_flag gMemPoolInitFlag;

cudaMemLocation gMemPoolLocation = { cudaMemLocationTypeNone, 0 };
cudaMemPool_t gMemPool = nullptr;

constexpr int gMemPoolMergeDeviceId = 0;
cudaEvent_t gMemPoolMergeEvent = nullptr;
std::vector<cudaEvent_t> gMemPoolEvents;

std::mutex gMemPoolMutex;

static void initMemPool()
{
    C10_CUDA_CHECK(cudaMemGetDefaultMemPool(&gMemPool, &gMemPoolLocation, cudaMemAllocationTypeManaged));
    cuuint64_t releaseThreshold = static_cast<cuuint64_t>(-1);
    C10_CUDA_CHECK(cudaMemPoolSetAttribute(gMemPool, cudaMemPoolAttrReleaseThreshold, &releaseThreshold));

    std::vector<cudaMemAccessDesc> descList(c10::cuda::device_count());
    for (const auto device_id : c10::irange(c10::cuda::device_count()))
    {
        descList[device_id].location = { cudaMemLocationTypeDevice, device_id };
        descList[device_id].flags = cudaMemAccessFlagsProtReadWrite;
    }
    // descList[c10::cuda::device_count()].location = { cudaMemLocationTypeHost, cudaCpuDeviceId };
    // descList[c10::cuda::device_count()].flags = cudaMemAccessFlagsProtReadWrite;
    C10_CUDA_CHECK(cudaMemPoolSetAccess(gMemPool, descList.data(), descList.size()));

    C10_CUDA_CHECK(cudaSetDevice(gMemPoolMergeDeviceId));
    C10_CUDA_CHECK(cudaEventCreateWithFlags(&gMemPoolMergeEvent, cudaEventDisableTiming));

    gMemPoolEvents.resize(c10::cuda::device_count());
    for (const auto device_id : c10::irange(1, c10::cuda::device_count()))
    {
        C10_CUDA_CHECK(cudaSetDevice(device_id));
        C10_CUDA_CHECK(cudaEventCreateWithFlags(&gMemPoolEvents[device_id], cudaEventDisableTiming));
    }

    // In addition to initializing the managed memory pool shared across devices, we also enable peer access
    // and set the release threshold on the per-device default memory pool so that cudaMallocAsync and
    // cudaFreeAsync don't block when allocating and deallocating scratch memory for CUB calls.
    for (const auto device_id : c10::irange(c10::cuda::device_count()))
    {
        C10_CUDA_CHECK(cudaSetDevice(device_id));
        cudaMemPool_t memPool = nullptr;
        cudaMemLocation location = { cudaMemLocationTypeDevice, device_id };
        C10_CUDA_CHECK(cudaMemGetDefaultMemPool(&memPool, &location, cudaMemAllocationTypePinned));
        C10_CUDA_CHECK(cudaMemPoolSetAttribute(memPool, cudaMemPoolAttrReleaseThreshold, &releaseThreshold));
        C10_CUDA_CHECK(cudaMemPoolSetAccess(memPool, descList.data(), descList.size()));
    }
}

c10::DataPtr DGXAllocator::allocate(size_t nbytes)
{
    c10::call_once(gMemPoolInitFlag, initMemPool);

    // std::lock_guard<std::mutex> memPoolLockGuard(gMemPoolMutex);

    const c10::Device device(c10::DeviceType::PrivateUse1, 0);

    if (nbytes > 0)
    {
        void* data = nullptr;

        // Allocate memory on the merge stream
        C10_CUDA_CHECK(cudaSetDevice(gMemPoolMergeDeviceId));
        auto merge_stream = c10::cuda::getCurrentCUDAStream(gMemPoolMergeDeviceId);
        C10_CUDA_CHECK(cudaMallocFromPoolAsync(&data, nbytes, gMemPool, merge_stream));
        // C10_CUDA_CHECK(cudaMemPrefetchAsync(data, nbytes, {cudaMemLocationTypeDevice, 0}, 0, merge_stream));
        C10_CUDA_CHECK(cudaEventRecord(gMemPoolMergeEvent, merge_stream));

        // Only peer streams need to wait; the merge stream is already ordered.
        for (const auto device_id : c10::irange(1, c10::cuda::device_count()))
        {
            C10_CUDA_CHECK(cudaSetDevice(device_id));
            auto stream = c10::cuda::getCurrentCUDAStream(device_id);
            C10_CUDA_CHECK(cudaStreamWaitEvent(stream, gMemPoolMergeEvent));
        }

        return { data, data, &Delete, device };
    }
    else
    {
        return c10::DataPtr(nullptr, device);
    }
}

void DGXAllocator::Delete(void* ptr)
{
    // std::lock_guard<std::mutex> memPoolLockGuard(gMemPoolMutex);

    // Record peer events; the merge device's accesses are already ordered before its free.
    for (const auto device_id : c10::irange(1, c10::cuda::device_count()))
    {
        C10_CUDA_CHECK(cudaSetDevice(device_id));
        auto stream = c10::cuda::getCurrentCUDAStream(device_id);
        C10_CUDA_CHECK(cudaEventRecord(gMemPoolEvents[device_id], stream));
    }

    // On the merge stream, wait until the peer events have completed.
    C10_CUDA_CHECK(cudaSetDevice(gMemPoolMergeDeviceId));
    auto merge_stream = c10::cuda::getCurrentCUDAStream(gMemPoolMergeDeviceId);
    for (const auto device_id : c10::irange(1, c10::cuda::device_count()))
    {
        C10_CUDA_CHECK(cudaStreamWaitEvent(merge_stream, gMemPoolEvents[device_id]));
    }

    // Free memory on the merge stream
    C10_CUDA_CHECK(cudaFreeAsync(ptr, merge_stream));
}
#else
c10::DataPtr DGXAllocator::allocate(size_t nbytes)
{
    constexpr size_t kPageSize = 2048 * 1024;
    const c10::Device device(c10::DeviceType::PrivateUse1, 0);

    if (nbytes > 0)
    {
        void* data = nullptr;
        C10_CUDA_CHECK(cudaMallocManaged(&data, nbytes));

        const size_t npages = nbytes / kPageSize;
        for (const auto device_id : c10::irange(c10::cuda::device_count()))
        {
            auto device_npages = (npages + c10::cuda::device_count() - 1) / c10::cuda::device_count();
            auto device_page_offset = device_npages * device_id;

            size_t device_nbytes = device_npages * kPageSize;
            size_t device_offset = device_page_offset * kPageSize;

            if (device_offset + device_nbytes > nbytes)
            {
                device_offset = std::min(device_offset, nbytes);
                device_nbytes = std::min(device_nbytes, nbytes - device_offset);
            }

            if (device_nbytes)
            {
                C10_CUDA_CHECK(cudaMemAdvise_v2(static_cast<uint8_t*>(data) + device_offset, device_nbytes,
                                                cudaMemAdviseSetPreferredLocation,
                                                { cudaMemLocationTypeDevice, device_id }));
            }
        }
        return { data, data, &Delete, device };
    }
    else
    {
        return c10::DataPtr(nullptr, device);
    }
}

void DGXAllocator::Delete(void* ptr)
{
    C10_CUDA_CHECK(cudaFree(ptr));
}
#endif

c10::DeleterFnPtr DGXAllocator::raw_deleter() const
{
    return &Delete;
}

void DGXAllocator::copy_data(void* dest, const void* src, std::size_t count) const
{
    C10_CUDA_CHECK(cudaMemcpy(dest, src, count, cudaMemcpyDefault));
}

// Register our dummy allocator
static DGXAllocator global_dgx_alloc;
REGISTER_ALLOCATOR(c10::DeviceType::PrivateUse1, &global_dgx_alloc);
