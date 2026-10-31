/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
#pragma once

#include "KernelLaunchUtils.cuh"

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
    const auto shard_units = ::cuda::ceil_div(::cuda::ceil_div(numel, alignment), c10::cuda::device_count());
    TORCH_CHECK(shard_units <= std::numeric_limits<int64_t>::max() / c10::cuda::device_count() / alignment,
                "reduce-scatter padded output size overflow");
    return shard_units * c10::cuda::device_count() * alignment;
}

// Each partial must hold padded_numel elements, with zeroes in its padding.
// Reduce into device-local slices, then let each GPU write its managed chunk.
// Alignment can preserve complete token rows when publishing input gradients.
// The caller orders scratch release and merges streams after these copies.
// A writeback callback can join destination placement after the local reduction,
// allowing independent work and NCCL to overlap that placement.
template <typename scalar_t, typename BeforeWriteback>
void nccl_reduce_scatter_partials(const std::vector<scalar_t*>& partials,
                                  scalar_t* output,
                                  int64_t numel,
                                  int64_t alignment,
                                  BeforeWriteback before_writeback)
{
    if (!numel)
        return;
    const auto padded_numel = nccl_reduce_scatter_padded_numel(numel, alignment);
    const int64_t shard_numel = padded_numel / partials.size();
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
        // Publish the same slice NCCL reduced, excluding trailing padding.
        const auto offset = device * shard_numel;
        if (offset < numel)
        {
            const auto count = std::min(shard_numel, numel - offset);
            before_writeback(device, stream);
            C10_CUDA_CHECK(cudaMemcpyAsync(
                output + offset, partials[device] + offset, count * sizeof(scalar_t), cudaMemcpyDefault, stream));
        }
    });
}

template <typename scalar_t>
void nccl_reduce_scatter_partials(const std::vector<scalar_t*>& partials,
                                  scalar_t* output,
                                  int64_t numel,
                                  int64_t alignment = 1)
{
    nccl_reduce_scatter_partials(partials, output, numel, alignment, [](int, cudaStream_t) {});
}

} // namespace at::native::dgx
