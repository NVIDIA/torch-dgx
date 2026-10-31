/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
#include <c10/cuda/CUDAStream.h>
#include <c10/util/CallOnce.h>

namespace at::native
{

namespace dgx
{

void merge_streams()
{
    static c10::once_flag gInitFlag;

    static const int gMergeDeviceId = 0;
    static cudaEvent_t gMergeEvent = nullptr;
    static std::vector<cudaEvent_t> gEvents(c10::cuda::device_count());

    c10::call_once(gInitFlag, []() {
        C10_CUDA_CHECK(cudaSetDevice(gMergeDeviceId));
        C10_CUDA_CHECK(cudaEventCreateWithFlags(&gMergeEvent, cudaEventDisableTiming));

        for (const auto device_id : c10::irange(1, c10::cuda::device_count()))
        {
            C10_CUDA_CHECK(cudaSetDevice(device_id));
            C10_CUDA_CHECK(cudaEventCreateWithFlags(&gEvents[device_id], cudaEventDisableTiming));
        }
    });

    // Record peer events; the merge device is already ordered by its stream.
    for (const auto device_id : c10::irange(1, c10::cuda::device_count()))
    {
        C10_CUDA_CHECK(cudaSetDevice(device_id));
        auto stream = c10::cuda::getCurrentCUDAStream(device_id);
        C10_CUDA_CHECK(cudaEventRecord(gEvents[device_id], stream));
    }

    // Wait on the merge device
    C10_CUDA_CHECK(cudaSetDevice(gMergeDeviceId));
    auto merge_stream = c10::cuda::getCurrentCUDAStream(gMergeDeviceId);
    // Wait for the peer streams before recording the merge event.
    for (const auto device_id : c10::irange(1, c10::cuda::device_count()))
    {
        C10_CUDA_CHECK(cudaStreamWaitEvent(merge_stream, gEvents[device_id]));
    }
    // Record an event on the merge stream to signify that the per-device events have been merged
    C10_CUDA_CHECK(cudaEventRecord(gMergeEvent, merge_stream));

    // Make the peer streams wait for the merge; the merge stream is already ordered.
    for (const auto device_id : c10::irange(1, c10::cuda::device_count()))
    {
        C10_CUDA_CHECK(cudaSetDevice(device_id));
        auto stream = c10::cuda::getCurrentCUDAStream(device_id);
        C10_CUDA_CHECK(cudaStreamWaitEvent(stream, gMergeEvent));
    }
}

} // namespace dgx

} // namespace at::native
