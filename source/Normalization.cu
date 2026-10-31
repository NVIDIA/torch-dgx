/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
#include "KernelLaunchUtils.cuh"
#include "NCCLUtils.h"

#include <c10/cuda/CUDAMathCompat.h>
#include <c10/util/ScopeExit.h>

#include <ATen/AccumulateType.h>
#include <ATen/Dispatch_v2.h>
#include <ATen/native/layer_norm.h>
#include <ATen/ops/empty.h>
#include <ATen/ops/empty_like.h>

#include <torch/library.h>

#include <cub/cub.cuh>
#include <cuda/warp>

#include <algorithm>
#include <array>
#include <limits>

namespace at::native
{

namespace dgx
{

namespace
{

constexpr int kBatchNormItemsPerThread = 4;
constexpr int64_t kBatchNormTileSize = kNumThreads * kBatchNormItemsPerThread;

// Separate row and feature indexing so packed Q/K views need no contiguous copy.
// Outputs are dense in logical order; a complete head stays on one GPU.
struct RMSNormLayout
{
    static constexpr int kMaxDims = 32;
    int dims, axis;
    int64_t sizes[kMaxDims]{}, strides[kMaxDims]{};

    RMSNormLayout(const Tensor& tensor, int normalized_dims) : dims(tensor.dim()), axis(dims - normalized_dims)
    {
        TORCH_CHECK(dims <= kMaxDims, "DGX RMSNorm supports at most ", kMaxDims, " dimensions");
        for (int dim = 0; dim < dims; ++dim)
        {
            sizes[dim] = tensor.size(dim);
            strides[dim] = tensor.stride(dim);
        }
    }

    __device__ int64_t row_offset(int64_t row) const
    {
        int64_t offset = 0;
        for (int dim = axis - 1; dim >= 0; --dim)
        {
            offset += (row % sizes[dim]) * strides[dim];
            row /= sizes[dim];
        }
        return offset;
    }

    __device__ int64_t feature_offset(int64_t feature) const
    {
        int64_t offset = 0;
        for (int dim = dims - 1; dim > axis; --dim)
        {
            offset += (feature % sizes[dim]) * strides[dim];
            feature /= sizes[dim];
        }
        return offset + feature * strides[axis];
    }
};

std::pair<int64_t, int64_t> rms_norm_shape(const Tensor& input,
                                           IntArrayRef normalized_shape,
                                           const std::optional<Tensor>& weight)
{
    TORCH_CHECK(input.device().type() == kPrivateUse1 && input.layout() == kStrided && !input.is_nested(),
                "DGX RMSNorm requires a strided DGX input");
    TORCH_CHECK(input.scalar_type() == kFloat || input.scalar_type() == kDouble,
                "DGX fused RMSNorm supports float32 and float64");
    const auto shape = _check_layer_norm_inputs(input, normalized_shape, weight.value_or(Tensor()), Tensor());
    if (weight && weight->defined())
    {
        TORCH_CHECK(weight->device() == input.device() && weight->scalar_type() == input.scalar_type() &&
                        weight->layout() == kStrided,
                    "Expected RMSNorm weight to have the input device, dtype and strided layout");
    }
    return shape;
}

std::pair<int64_t, int64_t> rms_norm_rows(const Tensor& input, int normalized_dims, int64_t rows, int device)
{
    const int64_t devices = c10::cuda::device_count();
    const int64_t rows_per_group = input.dim() == 4 && normalized_dims == 1 ? input.size(2) : 1;
    if (!rows || !rows_per_group)
        return { 0, 0 };
    const int64_t groups = rows / rows_per_group;
    const int64_t count = groups / devices;
    const int64_t extra = groups % devices;
    return { (device * count + std::min<int64_t>(device, extra)) * rows_per_group,
             (count + (device < extra)) * rows_per_group };
}

template <typename scalar_t, int ThreadsPerRow>
__device__ scalar_t rms_norm_row_sum(scalar_t sum)
{
    if constexpr (ThreadsPerRow == 32)
    {
        for (int offset = 16; offset; offset /= 2)
            sum += ::cuda::device::warp_shuffle_down(sum, offset).data;
        return ::cuda::device::warp_shuffle_idx(sum, 0).data;
    }
    else
    {
        using Reduce = cub::BlockReduce<scalar_t, kNumThreads>;
        __shared__ typename Reduce::TempStorage storage;
        __shared__ scalar_t result;
        const auto total = Reduce(storage).Sum(sum);
        if (!threadIdx.x)
            result = total;
        __syncthreads();
        return result;
    }
}

template <typename scalar_t, int ThreadsPerRow>
__global__ __launch_bounds__(kNumThreads) void rms_norm_forward_kernel(int64_t begin,
                                                                       int64_t count,
                                                                       int64_t width,
                                                                       RMSNormLayout layout,
                                                                       const scalar_t* input,
                                                                       const scalar_t* weight,
                                                                       scalar_t epsilon,
                                                                       scalar_t* output,
                                                                       scalar_t* rstd)
{
    constexpr int rows_per_block = kNumThreads / ThreadsPerRow;
    const int lane = threadIdx.x % ThreadsPerRow;
    for (int64_t local_row = int64_t(blockIdx.x) * rows_per_block + threadIdx.x / ThreadsPerRow; local_row < count;
         local_row += int64_t(gridDim.x) * rows_per_block)
    {
        const int64_t row = begin + local_row;
        const int64_t base = layout.row_offset(row);
        scalar_t sum = 0;
        for (int64_t feature = lane; feature < width; feature += ThreadsPerRow)
        {
            const auto x = input[base + layout.feature_offset(feature)];
            sum += x * x;
        }
        sum = rms_norm_row_sum<scalar_t, ThreadsPerRow>(sum);
        const auto inv_rms = c10::cuda::compat::rsqrt(sum / scalar_t(width) + epsilon);
        if (!lane)
            rstd[row] = inv_rms;
        for (int64_t feature = lane; feature < width; feature += ThreadsPerRow)
        {
            auto y = input[base + layout.feature_offset(feature)] * inv_rms;
            if (weight)
                y *= weight[feature];
            output[row * width + feature] = y;
        }
        if constexpr (ThreadsPerRow != 32)
            __syncthreads();
    }
}

template <typename scalar_t, int ThreadsPerRow>
__global__ __launch_bounds__(kNumThreads) void rms_norm_backward_kernel(int64_t begin,
                                                                        int64_t count,
                                                                        int64_t width,
                                                                        RMSNormLayout layout,
                                                                        RMSNormLayout grad_layout,
                                                                        const scalar_t* input,
                                                                        const scalar_t* grad,
                                                                        const scalar_t* weight,
                                                                        const scalar_t* rstd,
                                                                        scalar_t* output)
{
    constexpr int rows_per_block = kNumThreads / ThreadsPerRow;
    const int lane = threadIdx.x % ThreadsPerRow;
    for (int64_t local_row = int64_t(blockIdx.x) * rows_per_block + threadIdx.x / ThreadsPerRow; local_row < count;
         local_row += int64_t(gridDim.x) * rows_per_block)
    {
        const int64_t row = begin + local_row;
        const auto base = layout.row_offset(row);
        const auto grad_base = grad_layout.row_offset(row);
        const auto inv_rms = rstd[row];
        scalar_t sum = 0;
        for (int64_t feature = lane; feature < width; feature += ThreadsPerRow)
        {
            auto g = grad[grad_base + grad_layout.feature_offset(feature)];
            if (weight)
                g *= weight[feature];
            sum += g * input[base + layout.feature_offset(feature)];
        }
        sum = rms_norm_row_sum<scalar_t, ThreadsPerRow>(sum);
        const auto correction = (sum / scalar_t(width)) * inv_rms * inv_rms;
        for (int64_t feature = lane; feature < width; feature += ThreadsPerRow)
        {
            auto g = grad[grad_base + grad_layout.feature_offset(feature)];
            if (weight)
                g *= weight[feature];
            output[row * width + feature] = inv_rms * (g - input[base + layout.feature_offset(feature)] * correction);
        }
        if constexpr (ThreadsPerRow != 32)
            __syncthreads();
    }
}

template <typename scalar_t>
struct RMSNormWeightBackward
{
    __device__ void operator()(int64_t feature,
                               int64_t begin,
                               int64_t count,
                               int64_t width,
                               RMSNormLayout layout,
                               RMSNormLayout grad_layout,
                               const scalar_t* input,
                               const scalar_t* grad,
                               const scalar_t* rstd,
                               scalar_t* partial)
    {
        scalar_t sum = 0;
        if (feature < width)
        {
            const auto x_offset = layout.feature_offset(feature);
            const auto g_offset = grad_layout.feature_offset(feature);
            for (int64_t row = begin; row < begin + count; ++row)
                sum += grad[grad_layout.row_offset(row) + g_offset] *
                       (input[layout.row_offset(row) + x_offset] * rstd[row]);
        }
        partial[feature] = sum;
    }
};

template <typename accscalar_t>
struct LayerNormBackwardSums
{
    accscalar_t grad;
    accscalar_t grad_centered;
};

template <typename accscalar_t>
struct LayerNormBackwardSumCombine
{
    __device__ __forceinline__ LayerNormBackwardSums<accscalar_t> operator()(LayerNormBackwardSums<accscalar_t> lhs,
                                                                             LayerNormBackwardSums<accscalar_t> rhs) const
    {
        return { lhs.grad + rhs.grad, lhs.grad_centered + rhs.grad_centered };
    }
};

template <typename scalar_t, typename accscalar_t>
__global__ __launch_bounds__(kNumThreads) void layer_norm_input_backward_kernel(int64_t row_offset,
                                                                                int64_t normalized_size,
                                                                                const scalar_t* grad_output,
                                                                                const scalar_t* input,
                                                                                const accscalar_t* mean,
                                                                                const accscalar_t* rstd,
                                                                                const scalar_t* weight,
                                                                                scalar_t* grad_input)
{
    using BackwardSums = LayerNormBackwardSums<accscalar_t>;
    using BlockReduce = cub::BlockReduce<BackwardSums, kNumThreads>;
    __shared__ typename BlockReduce::TempStorage temp_storage;
    __shared__ BackwardSums row_sums;

    const auto row = row_offset + static_cast<int64_t>(blockIdx.x);
    const auto base_offset = row * normalized_size;
    const auto row_mean = mean[row];
    const auto row_rstd = rstd[row];
    BackwardSums thread_sums{ accscalar_t(0), accscalar_t(0) };

    for (int64_t feature = threadIdx.x; feature < normalized_size; feature += blockDim.x)
    {
        const auto offset = base_offset + feature;
        auto grad = static_cast<accscalar_t>(grad_output[offset]);
        if (weight)
        {
            grad *= static_cast<accscalar_t>(weight[feature]);
        }
        const auto centered = static_cast<accscalar_t>(input[offset]) - row_mean;
        thread_sums.grad += grad;
        thread_sums.grad_centered += grad * centered;
    }

    const auto block_sums = BlockReduce(temp_storage).Reduce(thread_sums, LayerNormBackwardSumCombine<accscalar_t>());
    if (!threadIdx.x)
    {
        row_sums = block_sums;
    }
    __syncthreads();

    const auto inv_normalized_size = accscalar_t(1) / static_cast<accscalar_t>(normalized_size);
    const auto mean_grad = row_sums.grad * inv_normalized_size;
    const auto mean_grad_centered = row_sums.grad_centered * inv_normalized_size;
    const auto rstd_squared = row_rstd * row_rstd;
    for (int64_t feature = threadIdx.x; feature < normalized_size; feature += blockDim.x)
    {
        const auto offset = base_offset + feature;
        auto grad = static_cast<accscalar_t>(grad_output[offset]);
        if (weight)
        {
            grad *= static_cast<accscalar_t>(weight[feature]);
        }
        const auto centered = static_cast<accscalar_t>(input[offset]) - row_mean;
        grad_input[offset] =
            static_cast<scalar_t>((grad - mean_grad - centered * rstd_squared * mean_grad_centered) * row_rstd);
    }
}

template <typename scalar_t, typename accscalar_t>
struct LayerNormParameterBackwardFunctor
{
    __device__ void operator()(size_t feature,
                               int64_t row_offset,
                               int64_t row_count,
                               int64_t normalized_size,
                               const scalar_t* grad_output,
                               const scalar_t* input,
                               const accscalar_t* mean,
                               const accscalar_t* rstd,
                               accscalar_t* grad_weight,
                               accscalar_t* grad_bias)
    {
        accscalar_t weight_sum = 0;
        accscalar_t bias_sum = 0;
        for (int64_t row = row_offset; row < row_offset + row_count; ++row)
        {
            const auto offset = row * normalized_size + feature;
            const auto grad = static_cast<accscalar_t>(grad_output[offset]);
            if (grad_weight)
            {
                weight_sum += grad * (static_cast<accscalar_t>(input[offset]) - mean[row]) * rstd[row];
            }
            if (grad_bias)
            {
                bias_sum += grad;
            }
        }

        if (grad_weight)
        {
            grad_weight[feature] = weight_sum;
        }
        if (grad_bias)
        {
            grad_bias[feature] = bias_sum;
        }
    }
};

constexpr int kLayerNormParameterFeatures = 32;
constexpr int kLayerNormParameterWarps = kNumThreads / kLayerNormParameterFeatures;
constexpr int64_t kLayerNormParameterRows = 256;
constexpr int64_t kLayerNormParameterSmallRows = 128;

// Each warp holds the same feature columns for different rows. Combine those
// row sums in a fixed order; every thread participates, including feature tails.
template <typename accscalar_t>
__device__ void layer_norm_store_parameter_sums(accscalar_t weight_sum,
                                                accscalar_t bias_sum,
                                                accscalar_t* shared_weight,
                                                accscalar_t* shared_bias,
                                                bool valid_feature,
                                                int64_t output_offset,
                                                accscalar_t* grad_weight,
                                                accscalar_t* grad_bias)
{
    const int thread = threadIdx.y * kLayerNormParameterFeatures + threadIdx.x;
    if (grad_weight)
        shared_weight[thread] = weight_sum;
    if (grad_bias)
        shared_bias[thread] = bias_sum;
    __syncthreads();
    if (threadIdx.y == 0 && valid_feature)
    {
        weight_sum = 0;
        bias_sum = 0;
        for (int warp = 0; warp < kLayerNormParameterWarps; ++warp)
        {
            const int index = warp * kLayerNormParameterFeatures + threadIdx.x;
            if (grad_weight)
                weight_sum += shared_weight[index];
            if (grad_bias)
                bias_sum += shared_bias[index];
        }
        if (grad_weight)
            grad_weight[output_offset] = weight_sum;
        if (grad_bias)
            grad_bias[output_offset] = bias_sum;
    }
    // A grid-stride iteration must not overwrite sums still being consumed.
    __syncthreads();
}

template <typename scalar_t, typename accscalar_t>
__global__ __launch_bounds__(kNumThreads) void layer_norm_parameter_partials_kernel(int64_t row_offset,
                                                                                   int64_t row_count,
                                                                                   int64_t normalized_size,
                                                                                   int64_t row_tiles,
                                                                                   int64_t feature_tiles,
                                                                                   const scalar_t* grad_output,
                                                                                   const scalar_t* input,
                                                                                   const accscalar_t* mean,
                                                                                   const accscalar_t* rstd,
                                                                                   accscalar_t* partial_weight,
                                                                                   accscalar_t* partial_bias)
{
    __shared__ accscalar_t shared_weight[kNumThreads];
    __shared__ accscalar_t shared_bias[kNumThreads];
    for (int64_t tile = blockIdx.x; tile < row_tiles * feature_tiles; tile += gridDim.x)
    {
        const int64_t row_tile = tile / feature_tiles;
        const int64_t feature = (tile % feature_tiles) * kLayerNormParameterFeatures + threadIdx.x;
        const int64_t begin = row_tile * kLayerNormParameterRows;
        accscalar_t weight_sum = 0;
        accscalar_t bias_sum = 0;
        if (feature < normalized_size)
        {
            for (int64_t local_row = begin + threadIdx.y;
                 local_row < row_count && local_row < begin + kLayerNormParameterRows;
                 local_row += kLayerNormParameterWarps)
            {
                const auto row = row_offset + local_row;
                const auto offset = row * normalized_size + feature;
                const auto grad = static_cast<accscalar_t>(grad_output[offset]);
                if (partial_weight)
                    weight_sum += grad * (static_cast<accscalar_t>(input[offset]) - mean[row]) * rstd[row];
                if (partial_bias)
                    bias_sum += grad;
            }
        }
        layer_norm_store_parameter_sums(weight_sum, bias_sum, shared_weight, shared_bias, feature < normalized_size,
                                       row_tile * normalized_size + feature, partial_weight, partial_bias);
    }
}

template <typename accscalar_t>
__global__ __launch_bounds__(kNumThreads) void layer_norm_parameter_reduce_kernel(int64_t normalized_size,
                                                                                 int64_t row_tiles,
                                                                                 const accscalar_t* partial_weight,
                                                                                 const accscalar_t* partial_bias,
                                                                                 accscalar_t* grad_weight,
                                                                                 accscalar_t* grad_bias)
{
    __shared__ accscalar_t shared_weight[kNumThreads];
    __shared__ accscalar_t shared_bias[kNumThreads];
    for (int64_t begin = int64_t(blockIdx.x) * kLayerNormParameterFeatures; begin < normalized_size;
         begin += int64_t(gridDim.x) * kLayerNormParameterFeatures)
    {
        const auto feature = begin + threadIdx.x;
        accscalar_t weight_sum = 0;
        accscalar_t bias_sum = 0;
        if (feature < normalized_size)
        {
            for (int64_t tile = threadIdx.y; tile < row_tiles; tile += kLayerNormParameterWarps)
            {
                const auto offset = tile * normalized_size + feature;
                if (grad_weight)
                    weight_sum += partial_weight[offset];
                if (grad_bias)
                    bias_sum += partial_bias[offset];
            }
        }
        layer_norm_store_parameter_sums(weight_sum, bias_sum, shared_weight, shared_bias, feature < normalized_size,
                                       feature, grad_weight, grad_bias);
    }
}

template <typename scalar_t, typename accscalar_t>
void layer_norm_parameter_backward(int64_t row_offset,
                                   int64_t row_count,
                                   int64_t normalized_size,
                                   const scalar_t* grad_output,
                                   const scalar_t* input,
                                   const accscalar_t* mean,
                                   const accscalar_t* rstd,
                                   accscalar_t* grad_weight,
                                   accscalar_t* grad_bias,
                                   cudaStream_t stream)
{
    if (row_count <= kLayerNormParameterSmallRows)
    {
        // Also writes zeroes for empty shards, which must still participate in NCCL.
        const auto blocks = static_cast<unsigned int>(
            std::min<int64_t>(::cuda::ceil_div(normalized_size, int64_t(kNumThreads)), 65535));
        lambda_kernel<<<blocks, kNumThreads, 0, stream>>>(
            normalized_size, LayerNormParameterBackwardFunctor<scalar_t, accscalar_t>(), row_offset, row_count,
            normalized_size, grad_output, input, mean, rstd, grad_weight, grad_bias);
        C10_CUDA_KERNEL_LAUNCH_CHECK();
        return;
    }

    const auto row_tiles = ::cuda::ceil_div(row_count, kLayerNormParameterRows);
    const auto feature_tiles = ::cuda::ceil_div(normalized_size, int64_t(kLayerNormParameterFeatures));
    const auto partial_numel = row_tiles * normalized_size;
    const int64_t num_parameters = int64_t(grad_weight != nullptr) + int64_t(grad_bias != nullptr);
    TORCH_CHECK(partial_numel <= std::numeric_limits<size_t>::max() / sizeof(accscalar_t) / num_parameters,
                "LayerNorm parameter gradient workspace size overflow");
    accscalar_t* partials = nullptr;
    C10_CUDA_CHECK(cudaMallocAsync(&partials, partial_numel * num_parameters * sizeof(accscalar_t), stream));
    auto release = c10::make_scope_exit([&] { C10_CUDA_CHECK_WARN(cudaFreeAsync(partials, stream)); });
    auto* partial_weight = grad_weight ? partials : nullptr;
    auto* partial_bias = grad_bias ? partials + (grad_weight ? partial_numel : 0) : nullptr;
    const dim3 threads(kLayerNormParameterFeatures, kLayerNormParameterWarps);
    const auto partial_blocks = static_cast<unsigned int>(std::min<int64_t>(row_tiles * feature_tiles, 65535));
    layer_norm_parameter_partials_kernel<<<partial_blocks, threads, 0, stream>>>(
        row_offset, row_count, normalized_size, row_tiles, feature_tiles, grad_output, input, mean, rstd,
        partial_weight, partial_bias);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    const auto reduce_blocks = static_cast<unsigned int>(std::min<int64_t>(feature_tiles, 65535));
    layer_norm_parameter_reduce_kernel<<<reduce_blocks, threads, 0, stream>>>(
        normalized_size, row_tiles, partial_weight, partial_bias, grad_weight, grad_bias);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

template <typename accscalar_t>
struct BatchNormWelfordData
{
    accscalar_t mean;
    accscalar_t m2;
    int64_t count;
};

template <typename accscalar_t>
struct BatchNormWelfordCombine
{
    __device__ __forceinline__ BatchNormWelfordData<accscalar_t> operator()(BatchNormWelfordData<accscalar_t> lhs,
                                                                            BatchNormWelfordData<accscalar_t> rhs) const
    {
        if (!lhs.count)
        {
            return rhs;
        }
        if (!rhs.count)
        {
            return lhs;
        }

        const auto count = lhs.count + rhs.count;
        const auto lhs_count = static_cast<accscalar_t>(lhs.count);
        const auto rhs_count = static_cast<accscalar_t>(rhs.count);
        const auto total_count = static_cast<accscalar_t>(count);
        const auto delta = rhs.mean - lhs.mean;
        return { lhs.mean + delta * rhs_count / total_count,
                 lhs.m2 + rhs.m2 + delta * delta * lhs_count * rhs_count / total_count, count };
    }
};

template <typename scalar_t>
__global__ __launch_bounds__(kNumThreads) void batch_norm_welford_kernel(const scalar_t* input,
                                                                         scalar_t* output,
                                                                         acc_type<scalar_t, true>* save_mean,
                                                                         acc_type<scalar_t, true>* save_invstd,
                                                                         int64_t input_channel_stride,
                                                                         int64_t output_channel_stride,
                                                                         int64_t channel_size,
                                                                         double epsilon)
{
    using accscalar_t = acc_type<scalar_t, true>;
    using WelfordData = BatchNormWelfordData<accscalar_t>;
    using BlockLoad = cub::BlockLoad<scalar_t, kNumThreads, kBatchNormItemsPerThread, cub::BLOCK_LOAD_WARP_TRANSPOSE>;
    using BlockStore = cub::BlockStore<scalar_t, kNumThreads, kBatchNormItemsPerThread, cub::BLOCK_STORE_WARP_TRANSPOSE>;
    using BlockReduce = cub::BlockReduce<WelfordData, kNumThreads>;

    struct IoTempStorage
    {
        typename BlockLoad::TempStorage load;
        typename BlockStore::TempStorage store;
    };

    union TempStorage
    {
        IoTempStorage io;
        typename BlockReduce::TempStorage reduce;
    };

    __shared__ TempStorage temp_storage;
    __shared__ accscalar_t channel_mean;
    __shared__ accscalar_t channel_invstd;

    const auto channel = static_cast<int64_t>(blockIdx.x);
    const auto channel_input = input + channel * input_channel_stride;
    auto channel_output = output + channel * output_channel_stride;
    WelfordData thread_data{ accscalar_t(0), accscalar_t(0), 0 };

    for (int64_t tile_offset = 0; tile_offset < channel_size; tile_offset += kBatchNormTileSize)
    {
        scalar_t items[kBatchNormItemsPerThread];
        const auto remaining = channel_size - tile_offset;
        const auto valid_items = static_cast<int>(remaining < kBatchNormTileSize ? remaining : kBatchNormTileSize);
        BlockLoad(temp_storage.io.load).Load(channel_input + tile_offset, items, valid_items);

#pragma unroll
        for (int item = 0; item < kBatchNormItemsPerThread; ++item)
        {
            const auto item_offset = static_cast<int>(threadIdx.x) * kBatchNormItemsPerThread + item;
            if (item_offset < valid_items)
            {
                const auto value = static_cast<accscalar_t>(items[item]);
                ++thread_data.count;
                const auto delta = value - thread_data.mean;
                thread_data.mean += delta / static_cast<accscalar_t>(thread_data.count);
                const auto updated_delta = value - thread_data.mean;
                thread_data.m2 += delta * updated_delta;
            }
        }
        __syncthreads();
    }

    const auto channel_data =
        BlockReduce(temp_storage.reduce).Reduce(thread_data, BatchNormWelfordCombine<accscalar_t>());
    if (!threadIdx.x)
    {
        channel_mean = channel_data.mean;
        channel_invstd = c10::cuda::compat::rsqrt(channel_data.m2 / static_cast<accscalar_t>(channel_data.count) +
                                                  static_cast<accscalar_t>(epsilon));
        save_mean[channel] = channel_mean;
        save_invstd[channel] = channel_invstd;
    }
    __syncthreads();

    for (int64_t tile_offset = 0; tile_offset < channel_size; tile_offset += kBatchNormTileSize)
    {
        scalar_t items[kBatchNormItemsPerThread];
        const auto remaining = channel_size - tile_offset;
        const auto valid_items = static_cast<int>(remaining < kBatchNormTileSize ? remaining : kBatchNormTileSize);
        BlockLoad(temp_storage.io.load).Load(channel_input + tile_offset, items, valid_items);

#pragma unroll
        for (int item = 0; item < kBatchNormItemsPerThread; ++item)
        {
            const auto item_offset = static_cast<int>(threadIdx.x) * kBatchNormItemsPerThread + item;
            if (item_offset < valid_items)
            {
                items[item] =
                    static_cast<scalar_t>((static_cast<accscalar_t>(items[item]) - channel_mean) * channel_invstd);
            }
        }

        BlockStore(temp_storage.io.store).Store(channel_output + tile_offset, items, valid_items);
        __syncthreads();
    }
}

} // namespace

std::tuple<Tensor, Tensor> fused_rms_norm(const Tensor& input,
                                          IntArrayRef normalized_shape,
                                          const std::optional<Tensor>& weight_opt,
                                          std::optional<double> eps)
{
    const auto [rows, width] = rms_norm_shape(input, normalized_shape, weight_opt);
    const RMSNormLayout layout(input, normalized_shape.size());
    const auto weight = weight_opt && weight_opt->defined() ? weight_opt->contiguous() : Tensor();
    auto output = at::empty(input.sizes(), input.options());
    auto stats_shape = input.sizes().vec();
    std::fill(stats_shape.end() - normalized_shape.size(), stats_shape.end(), 1);
    auto rstd = at::empty(stats_shape, input.options());
    AT_DISPATCH_V2(input.scalar_type(), "fused_rms_norm", AT_WRAP([&] {
                       const auto epsilon = scalar_t(eps.value_or(std::numeric_limits<scalar_t>::epsilon()));
                       const auto* weight_ptr = weight.defined() ? weight.const_data_ptr<scalar_t>() : nullptr;
                       parallel_for_each([&](int device, cudaStream_t stream) {
                           const auto [begin, count] = rms_norm_rows(input, normalized_shape.size(), rows, device);
                           if (!count)
                               return;
                           const int rows_per_block = width <= 1024 ? kNumThreads / 32 : 1;
                           const auto blocks = static_cast<unsigned int>(
                               std::min<int64_t>(::cuda::ceil_div(count, int64_t(rows_per_block)), 65535));
                           const auto kernel = width <= 1024 ? rms_norm_forward_kernel<scalar_t, 32> :
                                                               rms_norm_forward_kernel<scalar_t, kNumThreads>;
                           kernel<<<blocks, kNumThreads, 0, stream>>>(
                               begin, count, width, layout, input.const_data_ptr<scalar_t>(), weight_ptr, epsilon,
                               output.data_ptr<scalar_t>(), rstd.data_ptr<scalar_t>());
                           C10_CUDA_KERNEL_LAUNCH_CHECK();
                       });
                   }),
                   AT_EXPAND(AT_FLOATING_TYPES));
    return { output, rstd };
}

std::tuple<Tensor, Tensor> fused_rms_norm_backward(const Tensor& grad_output,
                                                   const Tensor& input,
                                                   IntArrayRef normalized_shape,
                                                   const Tensor& rstd,
                                                   const std::optional<Tensor>& weight_opt,
                                                   std::array<bool, 2> output_mask)
{
    const auto [rows, width] = rms_norm_shape(input, normalized_shape, weight_opt);
    TORCH_CHECK(grad_output.sizes() == input.sizes() && grad_output.device() == input.device() &&
                    grad_output.scalar_type() == input.scalar_type() && grad_output.layout() == kStrided,
                "Expected RMSNorm grad_output to have the input shape, device, dtype and strided layout");
    TORCH_CHECK(rstd.numel() == rows && rstd.device() == input.device() && rstd.scalar_type() == input.scalar_type() &&
                    rstd.layout() == kStrided,
                "Expected RMSNorm rstd to contain one value per row with the input device and dtype");
    const RMSNormLayout layout(input, normalized_shape.size());
    const RMSNormLayout grad_layout(grad_output, normalized_shape.size());
    const auto weight = weight_opt && weight_opt->defined() ? weight_opt->contiguous() : Tensor();
    const auto stats = rstd.contiguous();
    auto grad_input = output_mask[0] ? at::empty(input.sizes(), input.options()) : Tensor();
    auto grad_weight = output_mask[1] && weight.defined() ? at::empty(weight.sizes(), weight.options()) : Tensor();
    AT_DISPATCH_V2(input.scalar_type(), "fused_rms_norm_backward", AT_WRAP([&] {
                       if (grad_input.defined())
                       {
                           const auto* weight_ptr = weight.defined() ? weight.const_data_ptr<scalar_t>() : nullptr;
                           // dX and dW only share read-only inputs; merge once after both paths.
                           parallel_for_each_async([&](int device, cudaStream_t stream) {
                               const auto [begin, count] = rms_norm_rows(input, normalized_shape.size(), rows, device);
                               if (!count || !width)
                                   return;
                               const int rows_per_block = width <= 1024 ? kNumThreads / 32 : 1;
                               const auto blocks = static_cast<unsigned int>(
                                   std::min<int64_t>(::cuda::ceil_div(count, int64_t(rows_per_block)), 65535));
                               const auto kernel = width <= 1024 ? rms_norm_backward_kernel<scalar_t, 32> :
                                                                   rms_norm_backward_kernel<scalar_t, kNumThreads>;
                               kernel<<<blocks, kNumThreads, 0, stream>>>(
                                   begin, count, width, layout, grad_layout, input.const_data_ptr<scalar_t>(),
                                   grad_output.const_data_ptr<scalar_t>(), weight_ptr, stats.const_data_ptr<scalar_t>(),
                                   grad_input.data_ptr<scalar_t>());
                               C10_CUDA_KERNEL_LAUNCH_CHECK();
                           });
                       }
                       if (grad_weight.defined() && width)
                       {
                           const auto padded = nccl_reduce_scatter_padded_numel(width);
                           std::vector<scalar_t*> partials(c10::cuda::device_count());
                           parallel_for_each_async([&](int device, cudaStream_t stream) {
                               const auto [begin, count] = rms_norm_rows(input, normalized_shape.size(), rows, device);
                               C10_CUDA_CHECK(cudaMallocAsync(&partials[device], padded * sizeof(scalar_t), stream));
                               const auto blocks = static_cast<unsigned int>(
                                   std::min<int64_t>(::cuda::ceil_div(padded, int64_t(kNumThreads)), 65535));
                               lambda_kernel<<<blocks, kNumThreads, 0, stream>>>(
                                   padded, RMSNormWeightBackward<scalar_t>(), begin, count, width, layout, grad_layout,
                                   input.const_data_ptr<scalar_t>(), grad_output.const_data_ptr<scalar_t>(),
                                   stats.const_data_ptr<scalar_t>(), partials[device]);
                               C10_CUDA_KERNEL_LAUNCH_CHECK();
                           });
                           nccl_reduce_scatter_partials(partials, grad_weight.data_ptr<scalar_t>(), width);
                           parallel_for_each_async([&](int device, cudaStream_t stream) {
                               C10_CUDA_CHECK(cudaFreeAsync(partials[device], stream));
                           });
                       }
                       if (grad_input.defined() || (grad_weight.defined() && width))
                           merge_streams();
                   }),
                   AT_EXPAND(AT_FLOATING_TYPES));
    return { grad_input, grad_weight };
}

std::tuple<Tensor, Tensor, Tensor> batch_norm(const Tensor& self,
                                              const std::optional<Tensor>& weight_opt,
                                              const std::optional<Tensor>& bias_opt,
                                              const std::optional<Tensor>& running_mean_opt,
                                              const std::optional<Tensor>& running_var_opt,
                                              bool train,
                                              double momentum,
                                              double epsilon)
{
    TORCH_CHECK(!weight_opt);
    TORCH_CHECK(!bias_opt);
    TORCH_CHECK(!running_mean_opt);
    TORCH_CHECK(!running_var_opt);
    TORCH_CHECK(train);

    auto output = at::empty_like(self);
    int64_t n_input = self.size(1);
    auto options = self.options().dtype(at::toAccumulateType(self.scalar_type(), true));
    auto save_mean = at::empty({ n_input }, options);
    auto save_invstd = at::empty({ n_input }, options);

    AT_DISPATCH_V2(self.scalar_type(), "batch_norm", AT_WRAP([&] {
                       using accscalar_t = acc_type<scalar_t, true>;

                       parallel_for_each([&](int device, cudaStream_t stream) {
                           int64_t device_offset, device_channels;
                           std::tie(device_offset, device_channels) = chunk(self.size(1), device);
                           if (device_channels)
                           {
                               batch_norm_welford_kernel<scalar_t>
                                   <<<static_cast<unsigned int>(device_channels), kNumThreads, 0, stream>>>(
                                       self.const_data_ptr<scalar_t>() + device_offset * self.stride(1),
                                       output.data_ptr<scalar_t>() + device_offset * output.stride(1),
                                       save_mean.data_ptr<accscalar_t>() + device_offset,
                                       save_invstd.data_ptr<accscalar_t>() + device_offset, self.stride(1),
                                       output.stride(1), self.size(2), epsilon);
                               C10_CUDA_KERNEL_LAUNCH_CHECK();
                           }
                       });
                   }),
                   AT_EXPAND(AT_FLOATING_TYPES));

    return std::make_tuple(output, save_mean, save_invstd);
}

std::tuple<Tensor, Tensor, Tensor> native_layer_norm_backward(const Tensor& grad_output,
                                                              const Tensor& input,
                                                              c10::SymIntArrayRef normalized_shape,
                                                              const Tensor& mean,
                                                              const Tensor& rstd,
                                                              const std::optional<Tensor>& weight_opt,
                                                              const std::optional<Tensor>& bias_opt,
                                                              std::array<bool, 3> output_mask)
{
    int device = cudaInvalidDeviceId;
    C10_CUDA_CHECK(cudaGetDevice(&device));
    TORCH_CHECK(!normalized_shape.empty(), "Expected normalized_shape to contain at least one element");
    TORCH_CHECK(input.dim() >= normalized_shape.size(), "normalized_shape has more dimensions than the input");
    TORCH_CHECK(grad_output.sizes() == input.sizes(), "Expected grad_output and input to have the same shape");
    TORCH_CHECK(
        grad_output.scalar_type() == input.scalar_type(), "Expected grad_output and input to have the same dtype");

    int64_t normalized_size = 1;
    const auto first_normalized_dim = input.dim() - static_cast<int64_t>(normalized_shape.size());
    for (const auto dim : c10::irange(normalized_shape.size()))
    {
        const auto size = normalized_shape[dim].guard_int(__FILE__, __LINE__);
        TORCH_CHECK(input.size(first_normalized_dim + dim) == size,
                    "Expected normalized_shape to match the trailing input dimensions");
        normalized_size *= size;
    }
    TORCH_CHECK(normalized_size > 0, "LayerNorm backward does not support an empty normalized dimension");
    const auto num_rows = input.numel() / normalized_size;
    TORCH_CHECK(mean.numel() == num_rows && rstd.numel() == num_rows,
                "Expected mean and rstd to contain one value per normalized row");

    const auto expected_accumulation_type = at::toAccumulateType(input.scalar_type(), true);
    TORCH_CHECK(mean.scalar_type() == expected_accumulation_type && rstd.scalar_type() == expected_accumulation_type,
                "Expected mean and rstd to use the input accumulation dtype");

    const auto weight_defined = weight_opt.has_value() && weight_opt->defined();
    const auto bias_defined = bias_opt.has_value() && bias_opt->defined();
    // Bias-only backward needs neither the input nor the saved statistics.
    const bool need_statistics = output_mask[0] || (output_mask[1] && weight_defined);
    const auto input_contiguous = need_statistics ? input.contiguous() : input;
    const auto grad_output_contiguous = grad_output.contiguous();
    const auto mean_contiguous = need_statistics ? mean.contiguous() : mean;
    const auto rstd_contiguous = need_statistics ? rstd.contiguous() : rstd;
    std::optional<Tensor> weight_contiguous;
    if (weight_defined)
    {
        TORCH_CHECK(weight_opt->numel() == normalized_size, "Expected weight to match normalized_shape");
        TORCH_CHECK(weight_opt->scalar_type() == input.scalar_type(), "Expected weight and input to have the same dtype");
        weight_contiguous = weight_opt->contiguous();
    }
    if (bias_defined)
    {
        TORCH_CHECK(bias_opt->numel() == normalized_size, "Expected bias to match normalized_shape");
        TORCH_CHECK(bias_opt->scalar_type() == input.scalar_type(), "Expected bias and input to have the same dtype");
    }

    Tensor grad_input;
    Tensor grad_weight;
    Tensor grad_bias;
    if (output_mask[0])
    {
        grad_input = at::empty(input.sizes(), input.options());
    }
    if (output_mask[1] && weight_defined)
    {
        grad_weight = at::empty(weight_opt->sizes(), weight_opt->options());
    }
    if (output_mask[2] && bias_defined)
    {
        grad_bias = at::empty(bias_opt->sizes(), bias_opt->options());
    }

    AT_DISPATCH_V2(
        input.scalar_type(), "native_layer_norm_backward", AT_WRAP([&] {
            using accscalar_t = acc_type<scalar_t, true>;
            const auto weight = weight_contiguous ? weight_contiguous->const_data_ptr<scalar_t>() : nullptr;

            if (grad_input.defined())
            {
                // dX and the parameter gradients only share read-only inputs.
                // Keep their work stream-ordered and join once at the operator boundary.
                parallel_for_each_async([&](int device, cudaStream_t stream) {
                    int64_t row_offset, row_count;
                    std::tie(row_offset, row_count) = chunk(num_rows, device);
                    if (row_count)
                    {
                        const auto num_blocks = static_cast<unsigned int>(row_count);
                        layer_norm_input_backward_kernel<<<num_blocks, kNumThreads, 0, stream>>>(
                            row_offset, normalized_size, grad_output_contiguous.const_data_ptr<scalar_t>(),
                            input_contiguous.const_data_ptr<scalar_t>(), mean_contiguous.const_data_ptr<accscalar_t>(),
                            rstd_contiguous.const_data_ptr<accscalar_t>(), weight, grad_input.data_ptr<scalar_t>());
                        C10_CUDA_KERNEL_LAUNCH_CHECK();
                    }
                });
            }
            if (grad_weight.defined() || grad_bias.defined())
            {
                // Preserve grad_input's row ownership. Reduce tiles locally before
                // combining the small per-GPU parameter gradients on GPU 0.
                const auto num_parameters =
                    static_cast<int64_t>(grad_weight.defined()) + static_cast<int64_t>(grad_bias.defined());
                const auto partial_numel = num_parameters * normalized_size;
                const auto bias_offset = grad_weight.defined() ? normalized_size : int64_t(0);
                std::vector<accscalar_t*> partial_gradients(c10::cuda::device_count());

                parallel_for_each_async([&](int device, cudaStream_t stream) {
                    int64_t row_offset, row_count;
                    std::tie(row_offset, row_count) = chunk(num_rows, device);
                    C10_CUDA_CHECK(
                        cudaMallocAsync(&partial_gradients[device], partial_numel * sizeof(accscalar_t), stream));
                    auto* partial_grad_weight = grad_weight.defined() ? partial_gradients[device] : nullptr;
                    auto* partial_grad_bias = grad_bias.defined() ? partial_gradients[device] + bias_offset : nullptr;
                    layer_norm_parameter_backward(
                        row_offset, row_count, normalized_size, grad_output_contiguous.const_data_ptr<scalar_t>(),
                        input_contiguous.const_data_ptr<scalar_t>(), mean_contiguous.const_data_ptr<accscalar_t>(),
                        rstd_contiguous.const_data_ptr<accscalar_t>(), partial_grad_weight, partial_grad_bias, stream);
                });

                if (grad_weight.defined())
                {
                    nccl_reduce_partials(partial_gradients, grad_weight.data_ptr<scalar_t>(), normalized_size);
                }
                if (grad_bias.defined())
                {
                    nccl_reduce_partials(partial_gradients, grad_bias.data_ptr<scalar_t>(), normalized_size, bias_offset);
                }

                parallel_for_each_async([&](int device, cudaStream_t stream) {
                    C10_CUDA_CHECK(cudaFreeAsync(partial_gradients[device], stream));
                });
            }
            if (grad_input.defined() || grad_weight.defined() || grad_bias.defined())
                merge_streams();
        }),
        AT_EXPAND(AT_FLOATING_TYPES));

    C10_CUDA_CHECK(cudaSetDevice(device));
    return std::make_tuple(grad_input, grad_weight, grad_bias);
}

TORCH_LIBRARY_IMPL(aten, PrivateUse1, m)
{
    m.impl(TORCH_SELECTIVE_NAME("aten::_fused_rms_norm"), TORCH_FN(fused_rms_norm));
    m.impl(TORCH_SELECTIVE_NAME("aten::_fused_rms_norm_backward"), TORCH_FN(fused_rms_norm_backward));
    m.impl(TORCH_SELECTIVE_NAME("aten::native_batch_norm"), TORCH_FN(batch_norm));
    m.impl(TORCH_SELECTIVE_NAME("aten::native_layer_norm_backward"), TORCH_FN(native_layer_norm_backward));
}

} // namespace dgx

} // namespace at::native
