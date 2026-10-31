/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
#include "KernelLaunchUtils.cuh"

#include <ATen/Dispatch_v2.h>
#include <ATen/cuda/Exceptions.h>
#include <ATen/cudnn/Descriptors.h>
#include <ATen/cudnn/Handle.h>
#include <ATen/cudnn/Types.h>
#include <ATen/native/ConvUtils.h>
#include <ATen/ops/empty.h>

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

using ConvolutionPlanKey = std::array<int64_t, 37>;

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
                                             const int64_t* alignments)
{
    ConvolutionPlanKey key{};
    auto key_it = key.begin();
    *key_it++ = device_id;
    *key_it++ = data_type;
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
    cudnnDataType_t comp_type = at::native::getCudnnDataTypeFromScalarType(input.scalar_type());
    cudnnConvolutionMode_t mode = CUDNN_CROSS_CORRELATION;
    int64_t spatial_dims = 2;

    AT_CUDNN_CHECK(cudnnBackendSetAttribute(convolution_desc, CUDNN_ATTR_CONVOLUTION_COMP_TYPE, CUDNN_TYPE_DATA_TYPE, 1, &comp_type));
    AT_CUDNN_CHECK(cudnnBackendSetAttribute(convolution_desc, CUDNN_ATTR_CONVOLUTION_CONV_MODE, CUDNN_TYPE_CONVOLUTION_MODE, 1, &mode));
    AT_CUDNN_CHECK(cudnnBackendSetAttribute(
        convolution_desc, CUDNN_ATTR_CONVOLUTION_DILATIONS, CUDNN_TYPE_INT64, dilation.size(), dilation.data()));
    AT_CUDNN_CHECK(cudnnBackendSetAttribute(
        convolution_desc, CUDNN_ATTR_CONVOLUTION_FILTER_STRIDES, CUDNN_TYPE_INT64, stride.size(), stride.data()));
    AT_CUDNN_CHECK(cudnnBackendSetAttribute(convolution_desc, CUDNN_ATTR_CONVOLUTION_PRE_PADDINGS, CUDNN_TYPE_INT64,
                             pre_padding.size(), pre_padding.data()));
    AT_CUDNN_CHECK(cudnnBackendSetAttribute(convolution_desc, CUDNN_ATTR_CONVOLUTION_POST_PADDINGS, CUDNN_TYPE_INT64,
                             post_padding.size(), post_padding.data()));
    AT_CUDNN_CHECK(cudnnBackendSetAttribute(convolution_desc, CUDNN_ATTR_CONVOLUTION_SPATIAL_DIMS, CUDNN_TYPE_INT64, 1, &spatial_dims));
    AT_CUDNN_CHECK(cudnnBackendFinalize(convolution_desc));
}

void populate_operation_convolution_forward_descriptor(cudnnBackendDescriptor_t operation_desc,
                                                       cudnnBackendDescriptor_t input_desc,
                                                       cudnnBackendDescriptor_t weight_desc,
                                                       cudnnBackendDescriptor_t output_desc,
                                                       cudnnBackendDescriptor_t convolution_desc)
{
    float alpha = 1.0;
    float beta = 0.0;

    AT_CUDNN_CHECK(cudnnBackendSetAttribute(
        operation_desc, CUDNN_ATTR_OPERATION_CONVOLUTION_FORWARD_X, CUDNN_TYPE_BACKEND_DESCRIPTOR, 1, &input_desc));
    AT_CUDNN_CHECK(cudnnBackendSetAttribute(
        operation_desc, CUDNN_ATTR_OPERATION_CONVOLUTION_FORWARD_W, CUDNN_TYPE_BACKEND_DESCRIPTOR, 1, &weight_desc));
    AT_CUDNN_CHECK(cudnnBackendSetAttribute(
        operation_desc, CUDNN_ATTR_OPERATION_CONVOLUTION_FORWARD_Y, CUDNN_TYPE_BACKEND_DESCRIPTOR, 1, &output_desc));
    AT_CUDNN_CHECK(cudnnBackendSetAttribute(operation_desc, CUDNN_ATTR_OPERATION_CONVOLUTION_FORWARD_CONV_DESC,
                             CUDNN_TYPE_BACKEND_DESCRIPTOR, 1, &convolution_desc));
    AT_CUDNN_CHECK(cudnnBackendSetAttribute(operation_desc, CUDNN_ATTR_OPERATION_CONVOLUTION_FORWARD_ALPHA, CUDNN_TYPE_FLOAT, 1, &alpha));
    AT_CUDNN_CHECK(cudnnBackendSetAttribute(operation_desc, CUDNN_ATTR_OPERATION_CONVOLUTION_FORWARD_BETA, CUDNN_TYPE_FLOAT, 1, &beta));
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
                                                        cudaStream_t stream)
{
    cudnnStatus_t plan_status = CUDNN_STATUS_NOT_SUPPORTED;
    cudnnStatus_t execution_status = CUDNN_STATUS_NOT_SUPPORTED;
    cudaError_t allocation_status = cudaSuccess;

    for (auto engine_config_desc : engine_config_descs)
    {
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

    Tensor input_ = input.is_contiguous() ? input : input.contiguous();
    Tensor weight_ = weight.is_contiguous() ? weight : weight.contiguous();
    auto output =
        at::empty({ input_.size(0), weight_.size(0),
                    (input_.size(2) + 2 * padding[0] - dilation[0] * (weight_.size(2) - 1) - 1) / stride[0] + 1,
                    (input_.size(3) + 2 * padding[1] - dilation[1] * (weight_.size(3) - 1) - 1) / stride[1] + 1 },
                  input_.options());

    // Batch shards are contiguous in NCHW and let each device consume a disjoint input region. For small batches,
    // shard output channels instead: filter and output shards remain contiguous, while spatial shards would make
    // every device touch every channel's allocation pages.
    const bool shard_batches = input_.size(0) >= c10::cuda::device_count();
    parallel_for_each([&](int device_id, cudaStream_t stream) {
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
            std::tie(device_batch_offset, device_batch_count) = chunk(input_.size(0), device_id);
            if (!device_batch_count)
                return;

            device_input_sizes[0] = device_batch_count;
            device_output_sizes[0] = device_batch_count;
            input_offset = device_batch_offset * input_.stride(0);
            output_offset = device_batch_offset * output.stride(0);
        }
        else
        {
            int64_t device_channel_offset, device_channel_count;
            std::tie(device_channel_offset, device_channel_count) = chunk(weight_.size(0), device_id);
            if (!device_channel_count)
                return;

            device_weight_sizes[0] = device_channel_count;
            device_output_sizes[1] = device_channel_count;
            weight_offset = device_channel_offset * weight_.stride(0);
            output_offset = device_channel_offset * output.stride(1);
        }

        void* weight_local = nullptr;
        const auto weight_nbytes = device_weight_sizes[0] * weight_.stride(0) * weight_.element_size();
        C10_CUDA_CHECK(cudaMallocAsync(&weight_local, weight_nbytes, stream));
        C10_CUDA_CHECK(cudaMemcpyAsync(
            weight_local, static_cast<const uint8_t*>(weight_.const_data_ptr()) + weight_offset * weight_.element_size(),
            weight_nbytes, cudaMemcpyDefault, stream));

        void* input_local = nullptr;
        if (!shard_batches)
        {
            const auto allocation_status = cudaMallocAsync(&input_local, input_.nbytes(), stream);
            if (allocation_status == cudaErrorMemoryAllocation)
            {
                std::ignore = cudaGetLastError();
            }
            else
            {
                C10_CUDA_CHECK(allocation_status);
                C10_CUDA_CHECK(
                    cudaMemcpyAsync(input_local, input_.const_data_ptr(), input_.nbytes(), cudaMemcpyDefault, stream));
            }
        }

        void* dev_ptrs[3] = { input_local ? input_local :
                                            static_cast<uint8_t*>(input_.data_ptr()) +
                                                input_offset * c10::elementSize(input_.scalar_type()),
                              weight_local,
                              static_cast<uint8_t*>(output.data_ptr()) +
                                  output_offset * c10::elementSize(output.scalar_type()) };
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
            device_post_padding, dilation.data(), alignments);
        auto& plan_cache = get_convolution_plan_cache();
        if (const auto cached_plan = plan_cache.find(plan_key); cached_plan != plan_cache.end())
        {
            cudaError_t allocation_status;
            const auto status = execute_plan(handle, cached_plan->second, dev_ptrs, uids, stream, &allocation_status);
            if (status == CUDNN_STATUS_SUCCESS)
            {
                if (input_local)
                    C10_CUDA_CHECK(cudaFreeAsync(input_local, stream));
                C10_CUDA_CHECK(cudaFreeAsync(weight_local, stream));
                return;
            }

            cudnnBackendDestroyDescriptor(cached_plan->second.desc);
            plan_cache.erase(cached_plan);
        }

        cudnnBackendDescriptor_t input_desc;
        AT_CUDNN_CHECK(cudnnBackendCreateDescriptor(CUDNN_BACKEND_TENSOR_DESCRIPTOR, &input_desc));
        AT_CUDNN_CHECK(cudnnBackendSetAttribute(input_desc, CUDNN_ATTR_TENSOR_DATA_TYPE, CUDNN_TYPE_DATA_TYPE, 1, &input_data_type));
        AT_CUDNN_CHECK(cudnnBackendSetAttribute(
            input_desc, CUDNN_ATTR_TENSOR_DIMENSIONS, CUDNN_TYPE_INT64, input_.sizes().size(), device_input_sizes));
        AT_CUDNN_CHECK(cudnnBackendSetAttribute(
            input_desc, CUDNN_ATTR_TENSOR_STRIDES, CUDNN_TYPE_INT64, input_.strides().size(), device_input_strides));
        AT_CUDNN_CHECK(cudnnBackendSetAttribute(input_desc, CUDNN_ATTR_TENSOR_UNIQUE_ID, CUDNN_TYPE_INT64, 1, uids));
        AT_CUDNN_CHECK(cudnnBackendSetAttribute(input_desc, CUDNN_ATTR_TENSOR_BYTE_ALIGNMENT, CUDNN_TYPE_INT64, 1, &input_alignment));
        AT_CUDNN_CHECK(cudnnBackendFinalize(input_desc));

        cudnnBackendDescriptor_t weight_desc;
        AT_CUDNN_CHECK(cudnnBackendCreateDescriptor(CUDNN_BACKEND_TENSOR_DESCRIPTOR, &weight_desc));
        cudnnDataType_t weight_data_type = at::native::getCudnnDataTypeFromScalarType(weight_.scalar_type());
        AT_CUDNN_CHECK(cudnnBackendSetAttribute(weight_desc, CUDNN_ATTR_TENSOR_DATA_TYPE, CUDNN_TYPE_DATA_TYPE, 1, &weight_data_type));
        AT_CUDNN_CHECK(cudnnBackendSetAttribute(
            weight_desc, CUDNN_ATTR_TENSOR_DIMENSIONS, CUDNN_TYPE_INT64, weight_.sizes().size(), device_weight_sizes));
        AT_CUDNN_CHECK(cudnnBackendSetAttribute(weight_desc, CUDNN_ATTR_TENSOR_STRIDES, CUDNN_TYPE_INT64, weight_.strides().size(),
                                 weight_.strides().data()));
        AT_CUDNN_CHECK(cudnnBackendSetAttribute(weight_desc, CUDNN_ATTR_TENSOR_UNIQUE_ID, CUDNN_TYPE_INT64, 1, uids + 1));
        AT_CUDNN_CHECK(cudnnBackendSetAttribute(weight_desc, CUDNN_ATTR_TENSOR_BYTE_ALIGNMENT, CUDNN_TYPE_INT64, 1, &weight_alignment));
        AT_CUDNN_CHECK(cudnnBackendFinalize(weight_desc));

        cudnnBackendDescriptor_t output_desc;
        AT_CUDNN_CHECK(cudnnBackendCreateDescriptor(CUDNN_BACKEND_TENSOR_DESCRIPTOR, &output_desc));
        cudnnDataType_t output_data_type = at::native::getCudnnDataTypeFromScalarType(output.scalar_type());
        AT_CUDNN_CHECK(cudnnBackendSetAttribute(output_desc, CUDNN_ATTR_TENSOR_DATA_TYPE, CUDNN_TYPE_DATA_TYPE, 1, &output_data_type));
        AT_CUDNN_CHECK(cudnnBackendSetAttribute(
            output_desc, CUDNN_ATTR_TENSOR_DIMENSIONS, CUDNN_TYPE_INT64, output.sizes().size(), device_output_sizes));
        AT_CUDNN_CHECK(cudnnBackendSetAttribute(
            output_desc, CUDNN_ATTR_TENSOR_STRIDES, CUDNN_TYPE_INT64, output.strides().size(), device_output_strides));
        AT_CUDNN_CHECK(cudnnBackendSetAttribute(output_desc, CUDNN_ATTR_TENSOR_UNIQUE_ID, CUDNN_TYPE_INT64, 1, uids + 2));
        AT_CUDNN_CHECK(cudnnBackendSetAttribute(output_desc, CUDNN_ATTR_TENSOR_BYTE_ALIGNMENT, CUDNN_TYPE_INT64, 1, &output_alignment));
        AT_CUDNN_CHECK(cudnnBackendFinalize(output_desc));

        cudnnBackendDescriptor_t convolution_desc;
        AT_CUDNN_CHECK(cudnnBackendCreateDescriptor(CUDNN_BACKEND_CONVOLUTION_DESCRIPTOR, &convolution_desc));
        populate_convolution_descriptor(
            convolution_desc, input_, stride, device_pre_padding, device_post_padding, dilation);

        cudnnBackendDescriptor_t operation_desc;
        AT_CUDNN_CHECK(cudnnBackendCreateDescriptor(CUDNN_BACKEND_OPERATION_CONVOLUTION_FORWARD_DESCRIPTOR, &operation_desc));
        populate_operation_convolution_forward_descriptor(
            operation_desc, input_desc, weight_desc, output_desc, convolution_desc);

        cudnnBackendDescriptor_t operationgraph_desc;
        AT_CUDNN_CHECK(cudnnBackendCreateDescriptor(CUDNN_BACKEND_OPERATIONGRAPH_DESCRIPTOR, &operationgraph_desc));
        AT_CUDNN_CHECK(cudnnBackendSetAttribute(
            operationgraph_desc, CUDNN_ATTR_OPERATIONGRAPH_OPS, CUDNN_TYPE_BACKEND_DESCRIPTOR, 1, &operation_desc));
        AT_CUDNN_CHECK(cudnnBackendSetAttribute(operationgraph_desc, CUDNN_ATTR_OPERATIONGRAPH_HANDLE, CUDNN_TYPE_HANDLE, 1, &handle));
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
            dev_ptrs, uids, stream);
        constexpr size_t max_cached_plans = 128;
        if (plan_cache.size() == max_cached_plans)
        {
            cudnnBackendDestroyDescriptor(plan_cache.begin()->second.desc);
            plan_cache.erase(plan_cache.begin());
        }
        plan_cache.emplace(plan_key, execution_plan);
        if (input_local)
            C10_CUDA_CHECK(cudaFreeAsync(input_local, stream));
        C10_CUDA_CHECK(cudaFreeAsync(weight_local, stream));

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

    if (bias_opt && bias_opt->defined())
    {
        output.add_(reshape_bias(input.dim(), *bias_opt));
    }

    return output;
}

TORCH_LIBRARY_IMPL(aten, PrivateUse1, m)
{
    m.impl(TORCH_SELECTIVE_NAME("aten::convolution_overrideable"), TORCH_FN(convolution_overrideable));
}

} // namespace dgx

} // namespace at::native
