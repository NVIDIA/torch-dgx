/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
#include "KernelLaunchUtils.cuh"
#include "NCCLUtils.h"

#include <c10/util/irange.h>

#include <ATen/Context.h>
#include <ATen/Dispatch_v2.h>
#include <ATen/OpMathType.h>
#include <ATen/cuda/Exceptions.h>
#include <ATen/cudnn/Descriptors.h>
#include <ATen/cudnn/Handle.h>
#include <ATen/cudnn/Types.h>
#include <ATen/native/ConvUtils.h>
#include <ATen/ops/convolution_backward.h>
#include <ATen/ops/empty.h>
#include <ATen/ops/from_blob.h>
#include <ATen/ops/zeros.h>

#include <torch/library.h>

#include <algorithm>
#include <array>
#include <cstdint>
#include <unordered_map>
#include <vector>

namespace at::native
{

namespace dgx
{

namespace
{

using ConvolutionPlanKey = std::array<int64_t, 38>;

struct ConvolutionPlanKeyHash
{
    size_t operator()(const ConvolutionPlanKey& key) const
    {
        size_t result = 0;
        for (const auto value : key)
        {
            result ^= std::hash<int64_t>{}(value) + 0x9e3779b9 + (result << 6) + (result >> 2);
        }
        return result;
    }
};

struct CachedExecutionPlan
{
    cudnnBackendDescriptor_t desc;
    int64_t workspace_size;
};

using ConvolutionPlanCache = std::unordered_map<ConvolutionPlanKey, CachedExecutionPlan, ConvolutionPlanKeyHash>;

ConvolutionPlanCache& get_convolution_plan_cache()
{
    static thread_local ConvolutionPlanCache cache;
    return cache;
}

ConvolutionPlanKey make_convolution_plan_key(int device_id,
                                             cudnnDataType_t data_type,
                                             const int64_t* input_sizes,
                                             const int64_t* input_strides,
                                             const int64_t* weight_sizes,
                                             const int64_t* weight_strides,
                                             const int64_t* output_sizes,
                                             const int64_t* output_strides,
                                             const int64_t* convolution_stride,
                                             const int64_t* pre_padding,
                                             const int64_t* post_padding,
                                             const int64_t* dilation,
                                             const int64_t* alignments,
                                             bool allow_tf32)
{
    ConvolutionPlanKey key{};
    auto key_it = key.begin();
    *key_it++ = device_id;
    *key_it++ = data_type;
    *key_it++ = allow_tf32;
    for (const auto values : { input_sizes, input_strides, weight_sizes, weight_strides, output_sizes, output_strides })
    {
        key_it = std::copy_n(values, 4, key_it);
    }
    for (const auto values : { convolution_stride, pre_padding, post_padding, dilation })
    {
        key_it = std::copy_n(values, 2, key_it);
    }
    std::copy_n(alignments, 3, key_it);
    return key;
}

int64_t get_alignment(const void* ptr)
{
    int64_t alignment = 1;
    const auto address = reinterpret_cast<uintptr_t>(ptr);
    for (; alignment < 32; alignment *= 2)
    {
        if (address % (alignment * 2))
            return alignment;
    }
    return alignment;
}

void populate_convolution_descriptor(cudnnBackendDescriptor_t convolution_desc,
                                     const Tensor& input,
                                     IntArrayRef stride,
                                     IntArrayRef pre_padding,
                                     IntArrayRef post_padding,
                                     IntArrayRef dilation)
{
    cudnnDataType_t comp_type = input.scalar_type() == kDouble ? CUDNN_DATA_DOUBLE : CUDNN_DATA_FLOAT;
    cudnnConvolutionMode_t mode = CUDNN_CROSS_CORRELATION;
    int64_t spatial_dims = 2;

    AT_CUDNN_CHECK(cudnnBackendSetAttribute(
        convolution_desc, CUDNN_ATTR_CONVOLUTION_COMP_TYPE, CUDNN_TYPE_DATA_TYPE, 1, &comp_type));
    AT_CUDNN_CHECK(cudnnBackendSetAttribute(
        convolution_desc, CUDNN_ATTR_CONVOLUTION_CONV_MODE, CUDNN_TYPE_CONVOLUTION_MODE, 1, &mode));
    AT_CUDNN_CHECK(cudnnBackendSetAttribute(
        convolution_desc, CUDNN_ATTR_CONVOLUTION_DILATIONS, CUDNN_TYPE_INT64, dilation.size(), dilation.data()));
    AT_CUDNN_CHECK(cudnnBackendSetAttribute(
        convolution_desc, CUDNN_ATTR_CONVOLUTION_FILTER_STRIDES, CUDNN_TYPE_INT64, stride.size(), stride.data()));
    AT_CUDNN_CHECK(cudnnBackendSetAttribute(convolution_desc, CUDNN_ATTR_CONVOLUTION_PRE_PADDINGS, CUDNN_TYPE_INT64,
                                            pre_padding.size(), pre_padding.data()));
    AT_CUDNN_CHECK(cudnnBackendSetAttribute(convolution_desc, CUDNN_ATTR_CONVOLUTION_POST_PADDINGS, CUDNN_TYPE_INT64,
                                            post_padding.size(), post_padding.data()));
    AT_CUDNN_CHECK(cudnnBackendSetAttribute(
        convolution_desc, CUDNN_ATTR_CONVOLUTION_SPATIAL_DIMS, CUDNN_TYPE_INT64, 1, &spatial_dims));
    AT_CUDNN_CHECK(cudnnBackendFinalize(convolution_desc));
}

void populate_operation_convolution_forward_descriptor(cudnnBackendDescriptor_t operation_desc,
                                                       cudnnBackendDescriptor_t input_desc,
                                                       cudnnBackendDescriptor_t weight_desc,
                                                       cudnnBackendDescriptor_t output_desc,
                                                       cudnnBackendDescriptor_t convolution_desc,
                                                       cudnnDataType_t data_type)
{
    const float alpha = 1.0f;
    const float beta = 0.0f;
    const double alpha_double = 1.0;
    const double beta_double = 0.0;
    const auto scale_type = data_type == CUDNN_DATA_DOUBLE ? CUDNN_TYPE_DOUBLE : CUDNN_TYPE_FLOAT;
    const void* alpha_ptr = data_type == CUDNN_DATA_DOUBLE ? static_cast<const void*>(&alpha_double) : &alpha;
    const void* beta_ptr = data_type == CUDNN_DATA_DOUBLE ? static_cast<const void*>(&beta_double) : &beta;

    AT_CUDNN_CHECK(cudnnBackendSetAttribute(
        operation_desc, CUDNN_ATTR_OPERATION_CONVOLUTION_FORWARD_X, CUDNN_TYPE_BACKEND_DESCRIPTOR, 1, &input_desc));
    AT_CUDNN_CHECK(cudnnBackendSetAttribute(
        operation_desc, CUDNN_ATTR_OPERATION_CONVOLUTION_FORWARD_W, CUDNN_TYPE_BACKEND_DESCRIPTOR, 1, &weight_desc));
    AT_CUDNN_CHECK(cudnnBackendSetAttribute(
        operation_desc, CUDNN_ATTR_OPERATION_CONVOLUTION_FORWARD_Y, CUDNN_TYPE_BACKEND_DESCRIPTOR, 1, &output_desc));
    AT_CUDNN_CHECK(cudnnBackendSetAttribute(operation_desc, CUDNN_ATTR_OPERATION_CONVOLUTION_FORWARD_CONV_DESC,
                                            CUDNN_TYPE_BACKEND_DESCRIPTOR, 1, &convolution_desc));
    AT_CUDNN_CHECK(cudnnBackendSetAttribute(
        operation_desc, CUDNN_ATTR_OPERATION_CONVOLUTION_FORWARD_ALPHA, scale_type, 1, alpha_ptr));
    AT_CUDNN_CHECK(cudnnBackendSetAttribute(
        operation_desc, CUDNN_ATTR_OPERATION_CONVOLUTION_FORWARD_BETA, scale_type, 1, beta_ptr));
    AT_CUDNN_CHECK(cudnnBackendFinalize(operation_desc));
}

bool finalize_execution_plan(cudnnBackendDescriptor_t execution_plan, int64_t* workspace_size, cudnnStatus_t* plan_status)
{
    cudnnStatus_t status = cudnnBackendFinalize(execution_plan);
    if (status == CUDNN_STATUS_SUCCESS)
    {
        status = cudnnBackendGetAttribute(
            execution_plan, CUDNN_ATTR_EXECUTION_PLAN_WORKSPACE_SIZE, CUDNN_TYPE_INT64, 1, nullptr, workspace_size);
    }

    *plan_status = status;
    return status == CUDNN_STATUS_SUCCESS && *workspace_size >= 0;
}

void* allocate_workspace(int64_t workspace_size, cudaStream_t stream, cudaError_t* allocation_status)
{
    *allocation_status = cudaSuccess;
    if (workspace_size == 0)
        return nullptr;

    void* workspace = nullptr;
    cudaError_t status = cudaMallocAsync(&workspace, workspace_size, stream);
    *allocation_status = status;

    if (status == cudaErrorMemoryAllocation)
        std::ignore = cudaGetLastError(); // Clear the OOM before trying the next plan.
    else if (status != cudaSuccess)
        AT_CUDA_CHECK(status);
    return workspace;
}

cudnnStatus_t execute_plan(cudnnHandle_t handle,
                           const CachedExecutionPlan& execution_plan,
                           void* const* data_ptrs,
                           const int64_t* uids,
                           cudaStream_t stream,
                           cudaError_t* allocation_status)
{
    void* workspace = allocate_workspace(execution_plan.workspace_size, stream, allocation_status);
    if (*allocation_status == cudaErrorMemoryAllocation)
        return CUDNN_STATUS_ALLOC_FAILED;

    cudnnBackendDescriptor_t variant_pack_desc;
    AT_CUDNN_CHECK(cudnnBackendCreateDescriptor(CUDNN_BACKEND_VARIANT_PACK_DESCRIPTOR, &variant_pack_desc));
    AT_CUDNN_CHECK(cudnnBackendSetAttribute(
        variant_pack_desc, CUDNN_ATTR_VARIANT_PACK_DATA_POINTERS, CUDNN_TYPE_VOID_PTR, 3, data_ptrs));
    AT_CUDNN_CHECK(
        cudnnBackendSetAttribute(variant_pack_desc, CUDNN_ATTR_VARIANT_PACK_UNIQUE_IDS, CUDNN_TYPE_INT64, 3, uids));
    AT_CUDNN_CHECK(cudnnBackendSetAttribute(
        variant_pack_desc, CUDNN_ATTR_VARIANT_PACK_WORKSPACE, CUDNN_TYPE_VOID_PTR, 1, &workspace));
    AT_CUDNN_CHECK(cudnnBackendFinalize(variant_pack_desc));

    const auto execution_status = cudnnBackendExecute(handle, execution_plan.desc, variant_pack_desc);
    if (workspace)
        AT_CUDA_CHECK(cudaFreeAsync(workspace, stream));
    cudnnBackendDestroyDescriptor(variant_pack_desc);
    return execution_status;
}

CachedExecutionPlan create_and_execute_convolution_plan(cudnnHandle_t handle,
                                                        ArrayRef<cudnnBackendDescriptor_t> engine_config_descs,
                                                        void* const* data_ptrs,
                                                        const int64_t* uids,
                                                        cudaStream_t stream,
                                                        cudnnDataType_t data_type,
                                                        bool allow_tf32)
{
    cudnnStatus_t plan_status = CUDNN_STATUS_NOT_SUPPORTED;
    cudnnStatus_t execution_status = CUDNN_STATUS_NOT_SUPPORTED;
    cudaError_t allocation_status = cudaSuccess;

    for (auto engine_config_desc : engine_config_descs)
    {
        cudnnBackendDescriptor_t engine_desc;
        AT_CUDNN_CHECK(cudnnBackendCreateDescriptor(CUDNN_BACKEND_ENGINE_DESCRIPTOR, &engine_desc));
        int64_t count = 0;
        AT_CUDNN_CHECK(cudnnBackendGetAttribute(
            engine_config_desc, CUDNN_ATTR_ENGINECFG_ENGINE, CUDNN_TYPE_BACKEND_DESCRIPTOR, 1, &count, &engine_desc));
        std::array<cudnnBackendNumericalNote_t, CUDNN_NUMERICAL_NOTE_TYPE_COUNT> notes;
        AT_CUDNN_CHECK(cudnnBackendGetAttribute(engine_desc, CUDNN_ATTR_ENGINE_NUMERICAL_NOTE,
                                                CUDNN_TYPE_NUMERICAL_NOTE, notes.size(), &count, notes.data()));
        cudnnBackendDestroyDescriptor(engine_desc);
        // Match PyTorch's cuDNN precision policy before trying an engine.
        const bool disallowed = std::any_of(notes.begin(), notes.begin() + count, [&](const auto note) {
            return note == CUDNN_NUMERICAL_NOTE_DOWN_CONVERT_INPUTS ||
                   (data_type == CUDNN_DATA_FLOAT && !allow_tf32 && note == CUDNN_NUMERICAL_NOTE_TENSOR_CORE);
        });
        if (disallowed)
            continue;

        cudnnBackendDescriptor_t candidate_execution_plan_desc;
        AT_CUDNN_CHECK(
            cudnnBackendCreateDescriptor(CUDNN_BACKEND_EXECUTION_PLAN_DESCRIPTOR, &candidate_execution_plan_desc));
        AT_CUDNN_CHECK(cudnnBackendSetAttribute(candidate_execution_plan_desc, CUDNN_ATTR_EXECUTION_PLAN_ENGINE_CONFIG,
                                                CUDNN_TYPE_BACKEND_DESCRIPTOR, 1, &engine_config_desc));

        int64_t workspace_size = 0;
        if (!finalize_execution_plan(candidate_execution_plan_desc, &workspace_size, &plan_status))
        {
            cudnnBackendDestroyDescriptor(candidate_execution_plan_desc);
            continue;
        }

        CachedExecutionPlan candidate{ candidate_execution_plan_desc, workspace_size };
        execution_status = execute_plan(handle, candidate, data_ptrs, uids, stream, &allocation_status);

        if (execution_status == CUDNN_STATUS_SUCCESS)
            return candidate;

        cudnnBackendDestroyDescriptor(candidate_execution_plan_desc);
    }

    TORCH_CHECK(false,
                "cuDNN: no convolution execution plan succeeded; last plan status: ", cudnnGetErrorString(plan_status),
                ", last execution status: ", cudnnGetErrorString(execution_status),
                ", last CUDA status: ", cudaGetErrorString(allocation_status));
}

// Clone a managed tensor view into device-local storage before invoking CUDA
// convolution kernels. Preserve the source strides and offset while copying.
Tensor convolution_local_copy(const Tensor& tensor, int device, MemoryFormat memory_format)
{
    const c10::Device cuda_device(c10::DeviceType::CUDA, device);
    return at::for_blob(tensor.data_ptr(), tensor.sizes())
        .strides(tensor.strides())
        .options(tensor.options().device(cuda_device))
        .target_device(cuda_device)
        .make_tensor()
        .clone(memory_format);
}

struct SpatialConvolutionShard
{
    int64_t output_begin, output_rows;
    int64_t input_begin, input_rows;
    int64_t owned_begin, owned_rows;
    int64_t top_begin, top_rows;
    int64_t bottom_begin, bottom_rows;
};

SpatialConvolutionShard spatial_convolution_shard(
    int64_t input_height, int64_t output_height, int64_t kernel_height, int64_t stride, int64_t padding, int device)
{
    SpatialConvolutionShard shard{};
    std::tie(shard.output_begin, shard.output_rows) = partition::range(output_height, device);
    std::tie(shard.owned_begin, shard.owned_rows) = partition::range(input_height, device);
    shard.input_begin = shard.output_begin * stride - padding;
    if (shard.output_rows)
        shard.input_rows = (shard.output_rows - 1) * stride + kernel_height;
    const auto input_end = shard.input_begin + shard.input_rows;
    shard.top_begin = std::max<int64_t>(shard.input_begin, 0);
    shard.top_rows = std::max<int64_t>(0, std::min({ input_end, shard.owned_begin, input_height }) - shard.top_begin);
    shard.bottom_begin = std::max({ shard.input_begin, shard.owned_begin + shard.owned_rows, int64_t(0) });
    shard.bottom_rows = std::max<int64_t>(0, std::min(input_end, input_height) - shard.bottom_begin);
    return shard;
}

template <typename scalar_t>
struct ConvolutionRowsFunctor
{
    const scalar_t* source;
    scalar_t* destination;
    int64_t source_height, destination_height, row_size;
    int64_t source_begin, destination_begin, rows;
    bool accumulate;
    const scalar_t* bias;
    int64_t channels, bias_stride;
    bool channels_last;

    __device__ void operator()(int64_t index) const
    {
        const auto plane = index / (rows * row_size);
        const auto row = index / row_size % rows;
        const auto column = index % row_size;
        const auto source_row = source_begin + row;
        using opmath_t = at::opmath_type<scalar_t>;
        opmath_t value = 0;
        if (source_row >= 0 && source_row < source_height)
            value = source[(plane * source_height + source_row) * row_size + column];
        if (bias)
            value += static_cast<opmath_t>(bias[(channels_last ? column % channels : plane % channels) * bias_stride]);
        auto& output = destination[(plane * destination_height + destination_begin + row) * row_size + column];
        output = static_cast<scalar_t>(accumulate ? static_cast<opmath_t>(output) + value : value);
    }
};

// Pack halo rows, publish disjoint output/input-gradient strips, or add received
// boundary contributions. Out-of-image rows are explicit zeros in the local slab.
// Channels-last packs all channels into each row; NCHW has one plane per channel.
template <typename scalar_t>
void copy_convolution_rows(const scalar_t* source,
                           scalar_t* destination,
                           int64_t planes,
                           int64_t row_size,
                           int64_t source_height,
                           int64_t destination_height,
                           int64_t source_begin,
                           int64_t destination_begin,
                           int64_t rows,
                           cudaStream_t stream,
                           bool accumulate = false,
                           const scalar_t* bias = nullptr,
                           int64_t channels = 1,
                           int64_t bias_stride = 1,
                           bool channels_last = false)
{
    const auto numel = planes * rows * row_size;
    if (!numel)
        return;
    if (!accumulate && !bias)
    {
        const auto first = std::clamp<int64_t>(-source_begin, 0, rows);
        const auto last = std::clamp<int64_t>(source_height - source_begin, first, rows);
        const auto source_pitch = source_height * row_size * sizeof(scalar_t);
        const auto destination_pitch = destination_height * row_size * sizeof(scalar_t);
        if (first)
            C10_CUDA_CHECK(cudaMemset2DAsync(destination + destination_begin * row_size, destination_pitch, 0,
                                             first * row_size * sizeof(scalar_t), planes, stream));
        if (first < last)
            C10_CUDA_CHECK(cudaMemcpy2DAsync(destination + (destination_begin + first) * row_size, destination_pitch,
                                             source + (source_begin + first) * row_size, source_pitch,
                                             (last - first) * row_size * sizeof(scalar_t), planes, cudaMemcpyDefault,
                                             stream));
        if (last < rows)
            C10_CUDA_CHECK(cudaMemset2DAsync(destination + (destination_begin + last) * row_size, destination_pitch, 0,
                                             (rows - last) * row_size * sizeof(scalar_t), planes, stream));
        return;
    }
    const auto blocks =
        static_cast<unsigned int>(std::min<int64_t>(::cuda::ceil_div(numel, int64_t(kNumThreads)), 65535));
    const ConvolutionRowsFunctor<scalar_t> functor{ source,       destination,  source_height,     destination_height,
                                                    row_size,     source_begin, destination_begin, rows,
                                                    accumulate,   bias,         channels,          bias_stride,
                                                    channels_last };
    lambda_kernel<<<blocks, kNumThreads, 0, stream>>>(numel, functor);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

} // namespace

Tensor convolution_overrideable(const Tensor& input,
                                const Tensor& weight,
                                const std::optional<Tensor>& bias_opt,
                                IntArrayRef stride,
                                IntArrayRef padding,
                                IntArrayRef dilation,
                                bool transposed,
                                IntArrayRef output_padding,
                                int64_t groups)
{
    TORCH_CHECK(input.dim() == 4 && weight.dim() == 4, "Convolution only supports 2D convolution");
    TORCH_CHECK(!transposed, "Transposed convolution is not currently supported");
    TORCH_CHECK(groups == 1, "Convolution does not support more than a single group");
    TORCH_CHECK(stride.size() == 2 && padding.size() == 2 && dilation.size() == 2,
                "Convolution parameters must have two spatial dimensions");
    TORCH_CHECK(output_padding.size() == 2 && !output_padding[0] && !output_padding[1],
                "Nonzero output padding is not supported");

    const auto memory_format = cudnn_conv_suggest_memory_format(input, weight);
    const bool channels_last = memory_format == MemoryFormat::ChannelsLast;
    Tensor input_ = input.contiguous(memory_format);
    Tensor weight_ = weight.contiguous(memory_format);
    auto output =
        at::empty({ input_.size(0), weight_.size(0),
                    (input_.size(2) + 2 * padding[0] - dilation[0] * (weight_.size(2) - 1) - 1) / stride[0] + 1,
                    (input_.size(3) + 2 * padding[1] - dilation[1] * (weight_.size(3) - 1) - 1) / stride[1] + 1 },
                  input_.options(), memory_format);
    if (!output.numel())
        return output;

    const bool allow_tf32 =
        !at::NoTF32Guard::should_disable_tf32() && at::globalContext().allowTF32CuDNN(at::Float32Op::CONV);

    // Small batches use output-height strips with explicit input halos. Keep
    // convolution storage local and copy disjoint output strips back afterward.
    const bool shard_batches = input_.size(0) >= c10::cuda::device_count();
    const auto input_planes = input_.size(0) * (channels_last ? 1 : input_.size(1));
    const auto input_row_size = input_.size(3) * (channels_last ? input_.size(1) : 1);
    const auto output_planes = output.size(0) * (channels_last ? 1 : output.size(1));
    const auto output_row_size = output.size(3) * (channels_last ? output.size(1) : 1);
    std::vector<void*> local_outputs(c10::cuda::device_count(), nullptr);
    parallel_for_each_async([&](int device_id, cudaStream_t stream) {
        int64_t device_input_sizes[] = { input_.size(0), input_.size(1), input_.size(2), input_.size(3) };
        int64_t device_input_strides[] = { input_.stride(0), input_.stride(1), input_.stride(2), input_.stride(3) };
        int64_t device_weight_sizes[] = { weight_.size(0), weight_.size(1), weight_.size(2), weight_.size(3) };
        int64_t device_output_sizes[] = { output.size(0), output.size(1), output.size(2), output.size(3) };
        int64_t device_output_strides[] = { output.stride(0), output.stride(1), output.stride(2), output.stride(3) };
        int64_t device_pre_padding[] = { padding[0], padding[1] };
        int64_t device_post_padding[] = { padding[0], padding[1] };
        int64_t input_offset = 0;
        int64_t weight_offset = 0;
        int64_t output_offset = 0;

        if (shard_batches)
        {
            int64_t device_batch_offset, device_batch_count;
            std::tie(device_batch_offset, device_batch_count) = partition::range(input_.size(0), device_id);
            if (!device_batch_count)
                return;

            device_input_sizes[0] = device_batch_count;
            device_output_sizes[0] = device_batch_count;
            input_offset = device_batch_offset * input_.stride(0);
            output_offset = device_batch_offset * output.stride(0);
        }
        else
        {
            const auto shard =
                spatial_convolution_shard(input_.size(2), output.size(2), dilation[0] * (weight_.size(2) - 1) + 1,
                                          stride[0], padding[0], device_id);
            if (!shard.output_rows)
                return;
            device_input_sizes[2] = shard.input_rows;
            device_input_strides[0] = input_.size(1) * shard.input_rows * input_.size(3);
            device_output_sizes[2] = shard.output_rows;
            device_output_strides[0] = output.size(1) * shard.output_rows * output.size(3);
            if (!channels_last)
            {
                device_input_strides[1] = shard.input_rows * input_.size(3);
                device_output_strides[1] = shard.output_rows * output.size(3);
            }
            device_pre_padding[0] = device_post_padding[0] = 0;
            output_offset = shard.output_begin * output.stride(2);
        }

        void* weight_local = nullptr;
        const auto weight_nbytes = device_weight_sizes[0] * weight_.stride(0) * weight_.element_size();
        C10_CUDA_CHECK(cudaMallocAsync(&weight_local, weight_nbytes, stream));
        C10_CUDA_CHECK(cudaMemcpyAsync(
            weight_local, static_cast<const uint8_t*>(weight_.const_data_ptr()) + weight_offset * weight_.element_size(),
            weight_nbytes, cudaMemcpyDefault, stream));

        void* input_local = nullptr;
        void* output_local = nullptr;
        if (!shard_batches)
        {
            const auto shard =
                spatial_convolution_shard(input_.size(2), output.size(2), dilation[0] * (weight_.size(2) - 1) + 1,
                                          stride[0], padding[0], device_id);
            C10_CUDA_CHECK(cudaMallocAsync(
                &input_local, input_.size(0) * device_input_strides[0] * input_.element_size(), stream));
            C10_CUDA_CHECK(cudaMallocAsync(
                &output_local, output.size(0) * device_output_strides[0] * output.element_size(), stream));
            AT_DISPATCH_V2(input_.scalar_type(), "convolution_pack_halo", AT_WRAP([&] {
                               copy_convolution_rows(input_.const_data_ptr<scalar_t>(),
                                                     static_cast<scalar_t*>(input_local), input_planes, input_row_size,
                                                     input_.size(2), shard.input_rows, shard.input_begin, 0,
                                                     shard.input_rows, stream);
                           }),
                           AT_EXPAND(AT_FLOATING_TYPES), kHalf, kBFloat16);
        }
        auto finish = [&] {
            if (shard_batches && bias_opt && bias_opt->defined())
            {
                AT_DISPATCH_V2(output.scalar_type(), "convolution_bias", AT_WRAP([&] {
                                   auto* data = output.data_ptr<scalar_t>() + output_offset;
                                   copy_convolution_rows(
                                       data, data, device_output_sizes[0] * (channels_last ? 1 : output.size(1)),
                                       output_row_size, device_output_sizes[2], device_output_sizes[2], 0, 0,
                                       device_output_sizes[2], stream, false, bias_opt->const_data_ptr<scalar_t>(),
                                       output.size(1), bias_opt->stride(0), channels_last);
                               }),
                               AT_EXPAND(AT_FLOATING_TYPES), kHalf, kBFloat16);
            }
            local_outputs[device_id] = output_local;
            if (input_local)
                C10_CUDA_CHECK(cudaFreeAsync(input_local, stream));
            C10_CUDA_CHECK(cudaFreeAsync(weight_local, stream));
        };

        void* dev_ptrs[3] = {
            input_local ? input_local : static_cast<uint8_t*>(input_.data_ptr()) + input_offset * input_.element_size(),
            weight_local,
            output_local ? output_local : static_cast<uint8_t*>(output.data_ptr()) + output_offset * output.element_size()
        };

        const int64_t input_alignment = get_alignment(dev_ptrs[0]);
        const int64_t weight_alignment = get_alignment(dev_ptrs[1]);
        const int64_t output_alignment = get_alignment(dev_ptrs[2]);

        int64_t uids[3] = { 'x', 'w', 'y' };

        cudnnHandle_t handle = at::native::getCudnnHandle();
        AT_CUDNN_CHECK(cudnnSetStream(handle, stream));

        cudnnDataType_t input_data_type = at::native::getCudnnDataTypeFromScalarType(input_.scalar_type());
        const int64_t alignments[] = { input_alignment, weight_alignment, output_alignment };
        const auto plan_key = make_convolution_plan_key(
            device_id, input_data_type, device_input_sizes, device_input_strides, device_weight_sizes,
            weight_.strides().data(), device_output_sizes, device_output_strides, stride.data(), device_pre_padding,
            device_post_padding, dilation.data(), alignments, allow_tf32);
        auto& plan_cache = get_convolution_plan_cache();
        if (const auto cached_plan = plan_cache.find(plan_key); cached_plan != plan_cache.end())
        {
            cudaError_t allocation_status;
            const auto status = execute_plan(handle, cached_plan->second, dev_ptrs, uids, stream, &allocation_status);
            if (status == CUDNN_STATUS_SUCCESS)
            {
                finish();
                return;
            }

            cudnnBackendDestroyDescriptor(cached_plan->second.desc);
            plan_cache.erase(cached_plan);
        }

        cudnnBackendDescriptor_t input_desc;
        AT_CUDNN_CHECK(cudnnBackendCreateDescriptor(CUDNN_BACKEND_TENSOR_DESCRIPTOR, &input_desc));
        AT_CUDNN_CHECK(cudnnBackendSetAttribute(
            input_desc, CUDNN_ATTR_TENSOR_DATA_TYPE, CUDNN_TYPE_DATA_TYPE, 1, &input_data_type));
        AT_CUDNN_CHECK(cudnnBackendSetAttribute(
            input_desc, CUDNN_ATTR_TENSOR_DIMENSIONS, CUDNN_TYPE_INT64, input_.sizes().size(), device_input_sizes));
        AT_CUDNN_CHECK(cudnnBackendSetAttribute(
            input_desc, CUDNN_ATTR_TENSOR_STRIDES, CUDNN_TYPE_INT64, input_.strides().size(), device_input_strides));
        AT_CUDNN_CHECK(cudnnBackendSetAttribute(input_desc, CUDNN_ATTR_TENSOR_UNIQUE_ID, CUDNN_TYPE_INT64, 1, uids));
        AT_CUDNN_CHECK(cudnnBackendSetAttribute(
            input_desc, CUDNN_ATTR_TENSOR_BYTE_ALIGNMENT, CUDNN_TYPE_INT64, 1, &input_alignment));
        AT_CUDNN_CHECK(cudnnBackendFinalize(input_desc));

        cudnnBackendDescriptor_t weight_desc;
        AT_CUDNN_CHECK(cudnnBackendCreateDescriptor(CUDNN_BACKEND_TENSOR_DESCRIPTOR, &weight_desc));
        cudnnDataType_t weight_data_type = at::native::getCudnnDataTypeFromScalarType(weight_.scalar_type());
        AT_CUDNN_CHECK(cudnnBackendSetAttribute(
            weight_desc, CUDNN_ATTR_TENSOR_DATA_TYPE, CUDNN_TYPE_DATA_TYPE, 1, &weight_data_type));
        AT_CUDNN_CHECK(cudnnBackendSetAttribute(
            weight_desc, CUDNN_ATTR_TENSOR_DIMENSIONS, CUDNN_TYPE_INT64, weight_.sizes().size(), device_weight_sizes));
        AT_CUDNN_CHECK(cudnnBackendSetAttribute(weight_desc, CUDNN_ATTR_TENSOR_STRIDES, CUDNN_TYPE_INT64,
                                                weight_.strides().size(), weight_.strides().data()));
        AT_CUDNN_CHECK(cudnnBackendSetAttribute(weight_desc, CUDNN_ATTR_TENSOR_UNIQUE_ID, CUDNN_TYPE_INT64, 1, uids + 1));
        AT_CUDNN_CHECK(cudnnBackendSetAttribute(
            weight_desc, CUDNN_ATTR_TENSOR_BYTE_ALIGNMENT, CUDNN_TYPE_INT64, 1, &weight_alignment));
        AT_CUDNN_CHECK(cudnnBackendFinalize(weight_desc));

        cudnnBackendDescriptor_t output_desc;
        AT_CUDNN_CHECK(cudnnBackendCreateDescriptor(CUDNN_BACKEND_TENSOR_DESCRIPTOR, &output_desc));
        cudnnDataType_t output_data_type = at::native::getCudnnDataTypeFromScalarType(output.scalar_type());
        AT_CUDNN_CHECK(cudnnBackendSetAttribute(
            output_desc, CUDNN_ATTR_TENSOR_DATA_TYPE, CUDNN_TYPE_DATA_TYPE, 1, &output_data_type));
        AT_CUDNN_CHECK(cudnnBackendSetAttribute(
            output_desc, CUDNN_ATTR_TENSOR_DIMENSIONS, CUDNN_TYPE_INT64, output.sizes().size(), device_output_sizes));
        AT_CUDNN_CHECK(cudnnBackendSetAttribute(
            output_desc, CUDNN_ATTR_TENSOR_STRIDES, CUDNN_TYPE_INT64, output.strides().size(), device_output_strides));
        AT_CUDNN_CHECK(cudnnBackendSetAttribute(output_desc, CUDNN_ATTR_TENSOR_UNIQUE_ID, CUDNN_TYPE_INT64, 1, uids + 2));
        AT_CUDNN_CHECK(cudnnBackendSetAttribute(
            output_desc, CUDNN_ATTR_TENSOR_BYTE_ALIGNMENT, CUDNN_TYPE_INT64, 1, &output_alignment));
        AT_CUDNN_CHECK(cudnnBackendFinalize(output_desc));

        cudnnBackendDescriptor_t convolution_desc;
        AT_CUDNN_CHECK(cudnnBackendCreateDescriptor(CUDNN_BACKEND_CONVOLUTION_DESCRIPTOR, &convolution_desc));
        populate_convolution_descriptor(
            convolution_desc, input_, stride, device_pre_padding, device_post_padding, dilation);

        cudnnBackendDescriptor_t operation_desc;
        AT_CUDNN_CHECK(
            cudnnBackendCreateDescriptor(CUDNN_BACKEND_OPERATION_CONVOLUTION_FORWARD_DESCRIPTOR, &operation_desc));
        populate_operation_convolution_forward_descriptor(
            operation_desc, input_desc, weight_desc, output_desc, convolution_desc, input_data_type);

        cudnnBackendDescriptor_t operationgraph_desc;
        AT_CUDNN_CHECK(cudnnBackendCreateDescriptor(CUDNN_BACKEND_OPERATIONGRAPH_DESCRIPTOR, &operationgraph_desc));
        AT_CUDNN_CHECK(cudnnBackendSetAttribute(
            operationgraph_desc, CUDNN_ATTR_OPERATIONGRAPH_OPS, CUDNN_TYPE_BACKEND_DESCRIPTOR, 1, &operation_desc));
        AT_CUDNN_CHECK(cudnnBackendSetAttribute(
            operationgraph_desc, CUDNN_ATTR_OPERATIONGRAPH_HANDLE, CUDNN_TYPE_HANDLE, 1, &handle));
        AT_CUDNN_CHECK(cudnnBackendFinalize(operationgraph_desc));

        cudnnBackendDescriptor_t engine_heuristic_desc;
        AT_CUDNN_CHECK(cudnnBackendCreateDescriptor(CUDNN_BACKEND_ENGINEHEUR_DESCRIPTOR, &engine_heuristic_desc));
        cudnnBackendHeurMode_t heuristic_mode = CUDNN_HEUR_MODE_INSTANT;
        AT_CUDNN_CHECK(cudnnBackendSetAttribute(engine_heuristic_desc, CUDNN_ATTR_ENGINEHEUR_OPERATION_GRAPH,
                                                CUDNN_TYPE_BACKEND_DESCRIPTOR, 1, &operationgraph_desc));
        AT_CUDNN_CHECK(cudnnBackendSetAttribute(
            engine_heuristic_desc, CUDNN_ATTR_ENGINEHEUR_MODE, CUDNN_TYPE_HEUR_MODE, 1, &heuristic_mode));
        AT_CUDNN_CHECK(cudnnBackendFinalize(engine_heuristic_desc));

        int64_t available_engine_config_count = 0;
        AT_CUDNN_CHECK(cudnnBackendGetAttribute(engine_heuristic_desc, CUDNN_ATTR_ENGINEHEUR_RESULTS,
                                                CUDNN_TYPE_BACKEND_DESCRIPTOR, 0, &available_engine_config_count,
                                                nullptr));
        TORCH_CHECK(available_engine_config_count > 0, "cuDNN: no convolution engine configurations found");

        std::vector<cudnnBackendDescriptor_t> engine_config_descs(available_engine_config_count);
        for (auto& engine_config_desc : engine_config_descs)
        {
            AT_CUDNN_CHECK(cudnnBackendCreateDescriptor(CUDNN_BACKEND_ENGINECFG_DESCRIPTOR, &engine_config_desc));
        }

        int64_t actual_engine_config_count = 0;
        AT_CUDNN_CHECK(cudnnBackendGetAttribute(engine_heuristic_desc, CUDNN_ATTR_ENGINEHEUR_RESULTS,
                                                CUDNN_TYPE_BACKEND_DESCRIPTOR, available_engine_config_count,
                                                &actual_engine_config_count, engine_config_descs.data()));
        TORCH_CHECK(actual_engine_config_count > 0, "cuDNN: no convolution engine configurations returned");

        auto execution_plan = create_and_execute_convolution_plan(
            handle, ArrayRef<cudnnBackendDescriptor_t>(engine_config_descs.data(), actual_engine_config_count),
            dev_ptrs, uids, stream, input_data_type, allow_tf32);
        constexpr size_t max_cached_plans = 128;
        if (plan_cache.size() == max_cached_plans)
        {
            cudnnBackendDestroyDescriptor(plan_cache.begin()->second.desc);
            plan_cache.erase(plan_cache.begin());
        }
        plan_cache.emplace(plan_key, execution_plan);
        finish();

        for (auto engine_config_desc : engine_config_descs)
            cudnnBackendDestroyDescriptor(engine_config_desc);
        cudnnBackendDestroyDescriptor(engine_heuristic_desc);
        cudnnBackendDestroyDescriptor(operationgraph_desc);
        cudnnBackendDestroyDescriptor(operation_desc);
        cudnnBackendDestroyDescriptor(convolution_desc);
        cudnnBackendDestroyDescriptor(output_desc);
        cudnnBackendDestroyDescriptor(weight_desc);
        cudnnBackendDestroyDescriptor(input_desc);
    });

    if (!shard_batches)
    {
        parallel_for_each_async([&](int device, cudaStream_t stream) {
            if (!local_outputs[device])
                return;
            const auto shard = spatial_convolution_shard(
                input_.size(2), output.size(2), dilation[0] * (weight_.size(2) - 1) + 1, stride[0], padding[0], device);
            AT_DISPATCH_V2(output.scalar_type(), "convolution_publish_strip", AT_WRAP([&] {
                               auto* local = static_cast<scalar_t*>(local_outputs[device]);
                               const auto* bias =
                                   bias_opt && bias_opt->defined() ? bias_opt->const_data_ptr<scalar_t>() : nullptr;
                               // Add bias while publishing the strip to avoid a separate pass over the local output.
                               copy_convolution_rows(local, output.data_ptr<scalar_t>(), output_planes, output_row_size,
                                                     shard.output_rows, output.size(2), 0, shard.output_begin,
                                                     shard.output_rows, stream, false, bias, output.size(1),
                                                     bias ? bias_opt->stride(0) : 1, channels_last);
                           }),
                           AT_EXPAND(AT_FLOATING_TYPES), kHalf, kBFloat16);
            C10_CUDA_CHECK(cudaFreeAsync(local_outputs[device], stream));
        });
    }
    merge_streams();

    return output;
}

std::tuple<Tensor, Tensor, Tensor> convolution_backward_overrideable(const Tensor& grad_output,
                                                                     const Tensor& input,
                                                                     const Tensor& weight,
                                                                     IntArrayRef stride,
                                                                     IntArrayRef padding,
                                                                     IntArrayRef dilation,
                                                                     bool transposed,
                                                                     IntArrayRef output_padding,
                                                                     int64_t groups,
                                                                     std::array<bool, 3> output_mask)
{
    TORCH_CHECK(input.dim() == 4 && weight.dim() == 4, "Convolution backward only supports 2D convolution");
    TORCH_CHECK(!transposed, "Transposed convolution backward is not currently supported");
    TORCH_CHECK(groups == 1, "Convolution backward does not support more than a single group");
    TORCH_CHECK(stride.size() == 2 && padding.size() == 2 && dilation.size() == 2,
                "Convolution backward parameters must have two spatial dimensions");
    TORCH_CHECK(output_padding.size() == 2 && !output_padding[0] && !output_padding[1],
                "Nonzero output padding is not supported");
    TORCH_CHECK(input.device() == weight.device() && input.device() == grad_output.device(),
                "Convolution backward tensors must be on the same device");
    TORCH_CHECK(input.scalar_type() == weight.scalar_type() && input.scalar_type() == grad_output.scalar_type(),
                "Convolution backward tensors must have the same dtype");
    TORCH_CHECK(input.scalar_type() == kFloat || input.scalar_type() == kDouble,
                "Convolution backward supports float32 and float64");
    TORCH_CHECK(input.size(1) == weight.size(1) && input.size(1) > 0 && weight.size(0) > 0,
                "Convolution backward input and weight channels must match and be nonempty");
    for (const auto dim : c10::irange(2))
    {
        TORCH_CHECK(stride[dim] > 0 && dilation[dim] > 0 && padding[dim] >= 0,
                    "Convolution backward requires positive stride/dilation and nonnegative padding");
        TORCH_CHECK(weight.size(dim + 2) > 0 && input.size(dim + 2) > 0 &&
                        input.size(dim + 2) + 2 * padding[dim] >= dilation[dim] * (weight.size(dim + 2) - 1) + 1,
                    "Convolution backward kernel exceeds the padded input size");
    }
    const auto output_sizes = conv_output_size(input.sizes(), weight.sizes(), padding, stride, dilation);
    TORCH_CHECK(grad_output.sizes() == IntArrayRef(output_sizes), "Convolution backward grad_output has incorrect shape");

    const auto memory_format = cudnn_conv_suggest_memory_format(input, weight);
    const bool channels_last = memory_format == MemoryFormat::ChannelsLast;
    std::array<Tensor, 3> result{ output_mask[0] ? at::empty(input.sizes(), input.options(), memory_format) : Tensor(),
                                  output_mask[1] ? at::empty(weight.sizes(), weight.options(), memory_format) : Tensor(),
                                  output_mask[2] ? at::empty({ weight.size(0) }, input.options()) : Tensor() };
    if (!input.numel())
    {
        for (auto& gradient : result)
            if (gradient.defined())
                gradient.zero_();
        return { result[0], result[1], result[2] };
    }
    if (!output_mask[0] && !output_mask[1] && !output_mask[2])
        return { result[0], result[1], result[2] };

    // Batch and spatial shards both reduce only parameter gradients. Spatial
    // input gradients exchange the boundary rows outside their owner's strip.
    const auto devices = c10::cuda::device_count();
    const bool shard_batches = input.size(0) >= devices;
    const std::array<bool, 3> reduce{ false, true, true };
    const auto spatial_input = !shard_batches && output_mask[1] ? input.contiguous(memory_format) : input;
    const auto spatial_output = !shard_batches ? grad_output.contiguous(memory_format) : grad_output;
    const auto input_planes = input.size(0) * (channels_last ? 1 : input.size(1));
    const auto input_row_size = input.size(3) * (channels_last ? input.size(1) : 1);
    const auto output_planes = grad_output.size(0) * (channels_last ? 1 : grad_output.size(1));
    const auto output_row_size = grad_output.size(3) * (channels_last ? grad_output.size(1) : 1);
    std::vector<SpatialConvolutionShard> shards(devices);
    std::vector<Tensor> halos(devices);
    if (!shard_batches)
    {
        for (const auto device : c10::irange(c10::cuda::device_count()))
        {
            auto& shard = shards[device];
            shard = spatial_convolution_shard(input.size(2), grad_output.size(2),
                                              dilation[0] * (weight.size(2) - 1) + 1, stride[0], padding[0], device);
            if (output_mask[0])
                halos[device] =
                    at::empty({ input.size(0), input.size(1), shard.top_rows + shard.bottom_rows, input.size(3) },
                              input.options(), memory_format);
        }
    }
    std::vector<Tensor> spatial_grad_inputs(devices);
    AT_DISPATCH_V2(
        input.scalar_type(), "convolution_backward_dgx", AT_WRAP([&] {
            std::array<std::vector<Tensor>, 3> partials;
            std::array<std::vector<scalar_t*>, 3> partial_ptrs;
            for (const auto gradient : c10::irange(3))
            {
                if (output_mask[gradient] && reduce[gradient])
                {
                    partials[gradient].resize(devices);
                    partial_ptrs[gradient].resize(devices);
                }
            }
            parallel_for_each_async([&](int device, cudaStream_t stream) {
                const auto options = input.options().device(c10::DeviceType::CUDA, device);
                for (const auto gradient : c10::irange(3))
                {
                    if (output_mask[gradient] && reduce[gradient])
                    {
                        const auto padded = nccl_reduce_scatter_padded_numel(result[gradient].numel());
                        partials[gradient][device] = at::zeros({ padded }, options);
                        partial_ptrs[gradient][device] = partials[gradient][device].data_ptr<scalar_t>();
                    }
                }
                int64_t offset, count;
                std::tie(offset, count) = partition::range(input.size(0), device);
                const auto& shard = shards[device];
                if (!shard_batches)
                {
                    offset = shard.output_begin;
                    count = shard.output_rows;
                }
                if (!count)
                    return;
                Tensor local_input, local_output;
                const auto local_weight = convolution_local_copy(weight, device, memory_format);
                std::array<int64_t, 2> local_padding{ padding[0], padding[1] };
                if (shard_batches)
                {
                    local_input = convolution_local_copy(input.narrow(0, offset, count), device, memory_format);
                    local_output = convolution_local_copy(grad_output.narrow(0, offset, count), device, memory_format);
                }
                else
                {
                    local_padding[0] = 0;
                    local_input = at::empty(
                        { input.size(0), input.size(1), shard.input_rows, input.size(3) }, options, memory_format);
                    local_output =
                        at::empty({ grad_output.size(0), grad_output.size(1), shard.output_rows, grad_output.size(3) },
                                  options, memory_format);
                    if (output_mask[1])
                        copy_convolution_rows(spatial_input.const_data_ptr<scalar_t>(), local_input.data_ptr<scalar_t>(),
                                              input_planes, input_row_size, input.size(2), shard.input_rows,
                                              shard.input_begin, 0, shard.input_rows, stream);
                    copy_convolution_rows(spatial_output.const_data_ptr<scalar_t>(), local_output.data_ptr<scalar_t>(),
                                          output_planes, output_row_size, grad_output.size(2), shard.output_rows,
                                          shard.output_begin, 0, shard.output_rows, stream);
                }
                const std::array<int64_t, 1> bias_sizes{ local_output.size(1) };
                const auto [grad_input, grad_weight, grad_bias] =
                    at::convolution_backward(local_output, local_input, local_weight, IntArrayRef(bias_sizes), stride,
                                             local_padding, dilation, false, output_padding, 1, output_mask);
                const std::array<Tensor, 3> local_gradients{ grad_input, grad_weight, grad_bias };
                for (const auto gradient : c10::irange(3))
                {
                    if (!output_mask[gradient])
                        continue;
                    const auto local =
                        local_gradients[gradient].contiguous(gradient == 2 ? MemoryFormat::Contiguous : memory_format);
                    if (!shard_batches && gradient == 0)
                    {
                        spatial_grad_inputs[device] = local;
                        continue;
                    }
                    scalar_t* destination;
                    if (reduce[gradient])
                        destination = partial_ptrs[gradient][device];
                    else
                    {
                        const auto gradient_offset = gradient == 2 ? offset : offset * result[gradient].stride(0);
                        destination = result[gradient].data_ptr<scalar_t>() + gradient_offset;
                    }
                    C10_CUDA_CHECK(cudaMemcpyAsync(
                        destination, local.const_data_ptr(), local.nbytes(), cudaMemcpyDefault, stream));
                }
            });
            for (const auto gradient : c10::irange(3))
                if (output_mask[gradient] && reduce[gradient])
                    nccl_reduce_scatter_dense_partials(
                        partial_ptrs[gradient], result[gradient].data_ptr<scalar_t>(), result[gradient].numel());
            if (!shard_batches && output_mask[0])
            {
                parallel_for_each_async([&](int device, cudaStream_t stream) {
                    const auto& shard = shards[device];
                    const auto* local = spatial_grad_inputs[device].defined() ?
                                            spatial_grad_inputs[device].const_data_ptr<scalar_t>() :
                                            nullptr;
                    const auto local_rows = local ? shard.input_rows : 0;
                    copy_convolution_rows(local, result[0].data_ptr<scalar_t>(), input_planes, input_row_size,
                                          local_rows, input.size(2), shard.owned_begin - shard.input_begin,
                                          shard.owned_begin, shard.owned_rows, stream);
                    const auto halo_rows = shard.top_rows + shard.bottom_rows;
                    copy_convolution_rows(local, halos[device].data_ptr<scalar_t>(), input_planes, input_row_size,
                                          local_rows, halo_rows, shard.top_begin - shard.input_begin, 0, shard.top_rows,
                                          stream);
                    copy_convolution_rows(local, halos[device].data_ptr<scalar_t>(), input_planes, input_row_size,
                                          local_rows, halo_rows, shard.bottom_begin - shard.input_begin, shard.top_rows,
                                          shard.bottom_rows, stream);
                });
                merge_streams();
                parallel_for_each_async([&](int device, cudaStream_t stream) {
                    const auto& owner = shards[device];
                    for (const auto source : c10::irange(c10::cuda::device_count()))
                    {
                        if (source == device)
                            continue;
                        const auto& shard = shards[source];
                        const auto halo_rows = shard.top_rows + shard.bottom_rows;
                        const std::array<int64_t, 2> begins{ shard.top_begin, shard.bottom_begin };
                        const std::array<int64_t, 2> rows{ shard.top_rows, shard.bottom_rows };
                        for (const auto side : c10::irange(2))
                        {
                            const auto begin = std::max(owner.owned_begin, begins[side]);
                            const auto end = std::min(owner.owned_begin + owner.owned_rows, begins[side] + rows[side]);
                            if (begin < end)
                                copy_convolution_rows(halos[source].const_data_ptr<scalar_t>(),
                                                      result[0].data_ptr<scalar_t>(), input_planes, input_row_size,
                                                      halo_rows, input.size(2),
                                                      begin - begins[side] + (side ? shard.top_rows : 0), begin,
                                                      end - begin, stream, true);
                        }
                    }
                });
            }
            merge_streams();
        }),
        AT_EXPAND(AT_FLOATING_TYPES));
    return { result[0], result[1], result[2] };
}

TORCH_LIBRARY_IMPL(aten, PrivateUse1, m)
{
    m.impl(TORCH_SELECTIVE_NAME("aten::convolution_overrideable"), TORCH_FN(convolution_overrideable));
    m.impl(TORCH_SELECTIVE_NAME("aten::convolution_backward_overrideable"), TORCH_FN(convolution_backward_overrideable));
}

} // namespace dgx

} // namespace at::native
