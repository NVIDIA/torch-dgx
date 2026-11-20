/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
#include "KernelLaunchUtils.cuh"
#include "NCCLUtils.h"

#include <ATen/Dispatch_v2.h>
#include <ATen/ops/empty.h>

#include <torch/library.h>

#include <limits>
#include <vector>

namespace at::native
{

namespace dgx
{

template <typename scalar_t, typename index_t>
__global__ void embedding_dense_backward_kernel(int64_t num_indices,
                                                int64_t grad_output_offset,
                                                const index_t* indices,
                                                const PackedTensorAccessor64<scalar_t, 2, RestrictPtrTraits> grad_output,
                                                scalar_t* partial_grad_weight,
                                                int64_t weight_numel,
                                                int devices)
{
    const auto i = threadIdx.x + blockIdx.x * blockDim.x;
    const auto j = threadIdx.y + blockIdx.y * blockDim.y;

    if (i < num_indices && j < grad_output.size(1))
    {
        const auto row = indices[i];
        atomicAdd(partial_grad_weight + partition::packed_index(
                                            grad_output.size(1), weight_numel, devices, row * grad_output.size(1) + j),
                  grad_output[grad_output_offset + i][j]);
    }
}

Tensor embedding_dense_backward(
    const Tensor& grad_output, const Tensor& indices, int64_t num_weights, int64_t padding_idx, bool scale_grad_by_freq)
{
    TORCH_CHECK(indices.is_contiguous(), "Unsupported configuration in embedding_dense_backward");
    TORCH_CHECK(!scale_grad_by_freq, "Unsupported configuration in embedding_dense_backward");
    TORCH_CHECK(padding_idx < 0, "Unsupported configuration in embedding_dense_backward");
    TORCH_CHECK(num_weights >= 0, "num_weights must be non-negative");

    auto grad = grad_output.contiguous().view({ indices.numel(), grad_output.size(-1) });
    const auto embedding_dim = grad.size(1);
    if (!num_weights || !embedding_dim)
        return at::empty({ num_weights, embedding_dim }, grad_output.options());

    TORCH_CHECK(embedding_dim <= std::numeric_limits<int64_t>::max() / num_weights,
                "embedding_dense_backward output size overflow");
    // Each NCCL slot contains its rank's balanced complete rows, then zero padding.
    const auto weight_numel = num_weights * embedding_dim;
    const auto padded_numel = nccl_reduce_scatter_padded_numel(weight_numel, embedding_dim);
    auto grad_weight = at::empty({ num_weights, embedding_dim }, grad_output.options());

    using index_t = int64_t;
    constexpr dim3 num_threads(4, kNumThreads / 4);

    AT_DISPATCH_V2(
        grad_output.scalar_type(), "embedding_dense_backward", AT_WRAP([&] {
            std::vector<scalar_t*> partial_grad_weights(c10::cuda::device_count());
            TORCH_CHECK(padded_numel <= std::numeric_limits<size_t>::max() / sizeof(scalar_t),
                        "embedding_dense_backward partial buffer size overflow");
            const auto partial_grad_weight_bytes = padded_numel * sizeof(scalar_t);

            parallel_for_each_async([&](int device, cudaStream_t stream) {
                C10_CUDA_CHECK(cudaMallocAsync(&partial_grad_weights[device], partial_grad_weight_bytes, stream));
                C10_CUDA_CHECK(cudaMemsetAsync(partial_grad_weights[device], 0, partial_grad_weight_bytes, stream));

                int64_t indices_offset, indices_count;
                std::tie(indices_offset, indices_count) = partition::range(indices.numel(), device);

                const dim3 num_blocks(
                    ::cuda::ceil_div(indices_count, num_threads.x), ::cuda::ceil_div(grad.size(1), num_threads.y));

                if (num_blocks.x > 0 && num_blocks.y > 0)
                {
                    embedding_dense_backward_kernel<scalar_t, index_t><<<num_blocks, num_threads, 0, stream>>>(
                        indices_count, indices_offset, indices.const_data_ptr<index_t>() + indices_offset,
                        grad.packed_accessor64<scalar_t, 2, RestrictPtrTraits>(), partial_grad_weights[device],
                        weight_numel, c10::cuda::device_count());
                    C10_CUDA_KERNEL_LAUNCH_CHECK();
                }
            });

            nccl_reduce_scatter_packed_partials(
                partial_grad_weights, grad_weight.data_ptr<scalar_t>(), grad_weight.numel(), embedding_dim);

            parallel_for_each([&](int device, cudaStream_t stream) {
                C10_CUDA_CHECK(cudaFreeAsync(partial_grad_weights[device], stream));
            });
        }),
        AT_EXPAND(AT_FLOATING_TYPES));

    return grad_weight;
}

TORCH_LIBRARY_IMPL(aten, PrivateUse1, m)
{
    m.impl(TORCH_SELECTIVE_NAME("aten::embedding_dense_backward"), TORCH_FN(embedding_dense_backward));
}

} // namespace dgx

} // namespace at::native
