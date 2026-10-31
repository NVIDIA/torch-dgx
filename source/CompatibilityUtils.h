/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
#pragma once

#include <cuda_runtime_api.h>

namespace at::native
{

namespace dgx
{

// CUDA 13.0 changes cudaMemPrefetchAsync and cudaMemPrefetch to use a cudaMemLocation as an argument as
// opposed to an integer device id. In earlier versions of CUDA, the equivalent function was defined
// alongside the legacy version as cudaMemPrefetchAsync_v2 and cudaMemAdvise_v2. This function provides
// a wrapper to maintain compatibility across CUDA versions.
#if (CUDART_VERSION < 13000)
inline cudaError_t memAdvise(const void* devPtr, size_t count, cudaMemoryAdvise advice, cudaMemLocation location)
{
    return cudaMemAdvise_v2(devPtr, count, advice, location);
}

inline cudaError_t memPrefetchAsync(
    const void* devPtr, size_t count, cudaMemLocation location, unsigned int flags, cudaStream_t stream = 0)
{
    return cudaMemPrefetchAsync_v2(devPtr, count, location, flags, stream);
}
#else
inline cudaError_t memAdvise(const void* devPtr, size_t count, cudaMemoryAdvise advice, cudaMemLocation location)
{
    return cudaMemAdvise(devPtr, count, advice, location);
}

inline cudaError_t memPrefetchAsync(
    const void* devPtr, size_t count, cudaMemLocation location, unsigned int flags, cudaStream_t stream = 0)
{
    return cudaMemPrefetchAsync(devPtr, count, location, flags, stream);
}
#endif

} // namespace dgx

} // namespace at::native
