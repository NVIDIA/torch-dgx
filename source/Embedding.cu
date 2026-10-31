/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
#include "KernelLaunchUtils.cuh"

#include <ATen/Dispatch_v2.h>
#include <ATen/ops/empty.h>

#include <torch/csrc/cuda/nccl.h>
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
                                                scalar_t* partial_grad_weight)
{
    const auto i = threadIdx.x + blockIdx.x * blockDim.x;
    const auto j = threadIdx.y + blockIdx.y * blockDim.y;

    if (i < num_indices && j < grad_output.size(1))
    {
        const auto row = indices[i];
        atomicAdd(partial_grad_weight + row * grad_output.size(1) + j, grad_output[grad_output_offset + i][j]);
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

    int current_device = cudaInvalidDeviceId;
    C10_CUDA_CHECK(cudaGetDevice(&current_device));

    // Pad by complete rows so every reduce-scatter output is an equally sized, contiguous row shard.
    const auto rows_per_device = num_weights / c10::cuda::device_count() + (num_weights % c10::cuda::device_count() != 0);
    TORCH_CHECK(rows_per_device <= std::numeric_limits<int64_t>::max() / c10::cuda::device_count(),
                "embedding_dense_backward padded row count overflow");
    const auto padded_num_weights = rows_per_device * c10::cuda::device_count();
    TORCH_CHECK(embedding_dim <= std::numeric_limits<int64_t>::max() / padded_num_weights,
                "embedding_dense_backward padded output size overflow");
    const auto padded_numel = padded_num_weights * embedding_dim;
    const auto shard_numel = rows_per_device * embedding_dim;

    auto grad_weight_storage = at::empty({ padded_numel }, grad_output.options());
    auto grad_weight = grad_weight_storage.narrow(0, 0, num_weights * embedding_dim).view({ num_weights, embedding_dim });

    using index_t = int64_t;
    constexpr dim3 num_threads(4, kNumThreads / 4);

    AT_DISPATCH_V2(
        grad_output.scalar_type(), "embedding_dense_backward", AT_WRAP([&] {
            std::vector<scalar_t*> partial_grad_weights(c10::cuda::device_count());
            const auto partial_grad_weight_bytes = grad_weight_storage.nbytes();

            parallel_for_each_async([&](int device, cudaStream_t stream) {
                C10_CUDA_CHECK(cudaMallocAsync(&partial_grad_weights[device], partial_grad_weight_bytes, stream));
                C10_CUDA_CHECK(cudaMemsetAsync(partial_grad_weights[device], 0, partial_grad_weight_bytes, stream));

                int64_t indices_offset, indices_count;
                std::tie(indices_offset, indices_count) = chunk(indices.numel(), device);

                const dim3 num_blocks(
                    ::cuda::ceil_div(indices_count, num_threads.x), ::cuda::ceil_div(grad.size(1), num_threads.y));

                if (num_blocks.x > 0 && num_blocks.y > 0)
                {
                    embedding_dense_backward_kernel<scalar_t, index_t><<<num_blocks, num_threads, 0, stream>>>(
                        indices_count, indices_offset, indices.const_data_ptr<index_t>() + indices_offset,
                        grad.packed_accessor64<scalar_t, 2, RestrictPtrTraits>(), partial_grad_weights[device]);
                    C10_CUDA_KERNEL_LAUNCH_CHECK();
                }
            });

            std::vector<Tensor> partial_grad_weight_tensors;
            std::vector<Tensor> grad_weight_shards;
            partial_grad_weight_tensors.reserve(c10::cuda::device_count());
            grad_weight_shards.reserve(c10::cuda::device_count());
            for (const auto device : c10::irange(c10::cuda::device_count()))
            {
                partial_grad_weight_tensors.push_back(
                    at::from_blob(partial_grad_weights[device], { padded_numel },
                                  grad_weight.options().device(c10::DeviceType::CUDA, device)));
                grad_weight_shards.push_back(
                    partial_grad_weight_tensors.back().narrow(0, device * shard_numel, shard_numel));
            }
            torch::cuda::nccl::reduce_scatter(partial_grad_weight_tensors, grad_weight_shards);

            parallel_for_each([&](int device, cudaStream_t stream) {
                C10_CUDA_CHECK(cudaMemcpyAsync(grad_weight_storage.data_ptr<scalar_t>() + device * shard_numel,
                                               partial_grad_weights[device] + device * shard_numel,
                                               shard_numel * sizeof(scalar_t), cudaMemcpyDefault, stream));
                C10_CUDA_CHECK(cudaFreeAsync(partial_grad_weights[device], stream));
            });
        }),
        AT_EXPAND(AT_FLOATING_TYPES));

    C10_CUDA_CHECK(cudaSetDevice(current_device));
    return grad_weight;
}

TORCH_LIBRARY_IMPL(aten, PrivateUse1, m)
{
    m.impl(TORCH_SELECTIVE_NAME("aten::embedding_dense_backward"), TORCH_FN(embedding_dense_backward));
}

} // namespace dgx

} // namespace at::native
