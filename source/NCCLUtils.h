/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
#pragma once

#include "KernelLaunchUtils.cuh"
#include "PartitionUtils.h"

#include <c10/util/ScopeExit.h>
#include <c10/util/irange.h>

#include <ATen/ops/from_blob.h>

#include <torch/csrc/cuda/nccl.h>

#include <vector>

namespace at::native::dgx
{

// Submit a reduction to GPU 0 on the current per-device streams. The caller owns
// the buffers and must order their release and synchronize consumers of output.
template <typename scalar_t>
void nccl_reduce_partials(const std::vector<scalar_t*>& partials, scalar_t* output, int64_t numel, int64_t source_offset = 0)
{
    const auto options = at::TensorOptions().dtype<scalar_t>();
    std::vector<Tensor> inputs;
    inputs.reserve(partials.size());
    for (const auto device : c10::irange(partials.size()))
    {
        inputs.push_back(
            at::from_blob(partials[device] + source_offset, { numel }, options.device(c10::DeviceType::CUDA, device)));
    }
    auto result = at::from_blob(output, { numel }, options.device(c10::DeviceType::CUDA, 0));
    torch::cuda::nccl::reduce(inputs, result);
}

inline int64_t nccl_reduce_scatter_padded_numel(int64_t numel, int64_t alignment = 1)
{
    TORCH_CHECK(numel >= 0 && alignment > 0, "reduce-scatter requires nonnegative size and positive alignment");
    TORCH_CHECK(c10::cuda::device_count() > 0, "reduce-scatter requires at least one CUDA device");
    TORCH_CHECK(partition::slot_units(alignment, numel, c10::cuda::device_count()) <=
                    std::numeric_limits<int64_t>::max() / c10::cuda::device_count() / alignment,
                "reduce-scatter padded output size overflow");
    return partition::padded_size(alignment, numel, c10::cuda::device_count());
}

// Each partial contains equal-sized rank slots. Slot r holds the balanced
// logical range for r followed by zero padding, including between rank ranges.
// Reduce into device-local slots, then let each GPU write its managed range.
// Alignment can preserve complete token rows when publishing input gradients.
// The caller orders scratch release and merges streams after these copies.
// A writeback callback can join destination placement after the local reduction,
// allowing independent work and NCCL to overlap that placement.
template <typename scalar_t, typename BeforeWriteback>
void nccl_reduce_scatter_packed_partials(const std::vector<scalar_t*>& partials,
                                         scalar_t* output,
                                         int64_t numel,
                                         int64_t alignment,
                                         BeforeWriteback before_writeback)
{
    if (!numel)
        return;
    const auto padded_numel = nccl_reduce_scatter_padded_numel(numel, alignment);
    TORCH_CHECK(partials.size() == c10::cuda::device_count(), "reduce-scatter requires one partial per CUDA device");
    const int64_t shard_numel = padded_numel / c10::cuda::device_count();
    const auto options = at::TensorOptions().dtype<scalar_t>();
    std::vector<Tensor> inputs;
    std::vector<Tensor> outputs;
    inputs.reserve(partials.size());
    outputs.reserve(partials.size());
    for (const auto device : c10::irange(partials.size()))
    {
        inputs.push_back(at::from_blob(partials[device], { padded_numel }, options.device(c10::DeviceType::CUDA, device)));
        outputs.push_back(inputs.back().narrow(0, device * shard_numel, shard_numel));
    }
    torch::cuda::nccl::reduce_scatter(inputs, outputs);
    parallel_for_each_async([&](int device, cudaStream_t stream) {
        const auto offset = partition::offset(alignment, numel, c10::cuda::device_count(), device);
        const auto count = partition::count(alignment, numel, c10::cuda::device_count(), device);
        if (count)
        {
            before_writeback(device, stream);
            C10_CUDA_CHECK(cudaMemcpyAsync(output + offset, partials[device] + device * shard_numel,
                                           count * sizeof(scalar_t), cudaMemcpyDefault, stream));
        }
    });
}

template <typename scalar_t>
void nccl_reduce_scatter_packed_partials(const std::vector<scalar_t*>& partials,
                                         scalar_t* output,
                                         int64_t numel,
                                         int64_t alignment = 1)
{
    nccl_reduce_scatter_packed_partials(partials, output, numel, alignment, [](int, cudaStream_t) {});
}

template <typename scalar_t>
__global__ void pack_reduce_scatter_partials_kernel(
    int64_t numel, int64_t alignment, int devices, int64_t padded_numel, const scalar_t* dense, scalar_t* packed)
{
    const auto stride = static_cast<size_t>(blockDim.x) * gridDim.x;
    for (auto i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x; i < padded_numel; i += stride)
    {
        const auto logical_index = partition::logical_index(alignment, numel, devices, i);
        packed[i] = logical_index < numel ? dense[logical_index] : scalar_t(0);
    }
}

// Dense producers (e.g. cuBLAS) write logical order plus zero trailing padding.
// Pack locally only when the balanced ranges introduce padding between ranks.
template <typename scalar_t, typename BeforeWriteback>
void nccl_reduce_scatter_dense_partials(const std::vector<scalar_t*>& partials,
                                        scalar_t* output,
                                        int64_t numel,
                                        int64_t alignment,
                                        BeforeWriteback before_writeback)
{
    if (!numel)
        return;
    const auto padded_numel = nccl_reduce_scatter_padded_numel(numel, alignment);
    TORCH_CHECK(partials.size() == c10::cuda::device_count(), "reduce-scatter requires one partial per CUDA device");
    if (!partition::has_internal_padding(alignment, numel, c10::cuda::device_count()))
    {
        nccl_reduce_scatter_packed_partials(partials, output, numel, alignment, before_writeback);
        return;
    }

    TORCH_CHECK(padded_numel <= std::numeric_limits<size_t>::max() / sizeof(scalar_t),
                "reduce-scatter packed buffer size overflow");
    std::vector<scalar_t*> packed(partials.size(), nullptr);
    auto release = c10::make_scope_exit([&] {
        parallel_for_each_async([&](int device, cudaStream_t stream) {
            if (packed[device])
                C10_CUDA_CHECK_WARN(cudaFreeAsync(packed[device], stream));
        });
    });
    parallel_for_each_async([&](int device, cudaStream_t stream) {
        C10_CUDA_CHECK(cudaMallocAsync(&packed[device], padded_numel * sizeof(scalar_t), stream));
        const auto blocks =
            static_cast<unsigned int>(std::min<int64_t>(::cuda::ceil_div(padded_numel, int64_t(kNumThreads)), 65535));
        pack_reduce_scatter_partials_kernel<<<blocks, kNumThreads, 0, stream>>>(
            numel, alignment, c10::cuda::device_count(), padded_numel, partials[device], packed[device]);
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    });
    nccl_reduce_scatter_packed_partials(packed, output, numel, alignment, before_writeback);
}

template <typename scalar_t>
void nccl_reduce_scatter_dense_partials(const std::vector<scalar_t*>& partials,
                                        scalar_t* output,
                                        int64_t numel,
                                        int64_t alignment = 1)
{
    nccl_reduce_scatter_dense_partials(partials, output, numel, alignment, [](int, cudaStream_t) {});
}

} // namespace at::native::dgx
