/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
#include "KernelLaunchUtils.cuh"
#include "StridedTensorUtils.cuh"

#include <c10/cuda/CUDAStream.h>

#include <ATen/Dispatch_v2.h>
#include <ATen/ExpandUtils.h>
#include <ATen/TensorIterator.h>
#include <ATen/core/Reduction.h>
#include <ATen/ops/empty.h>
#include <ATen/ops/empty_like.h>
#include <ATen/ops/mse_loss_backward.h>

#include <torch/csrc/cuda/nccl.h>
#include <torch/library.h>

#include <cub/cub.cuh>
#include <thrust/iterator/counting_iterator.h>

#include <limits>
#include <vector>

namespace at::native
{

namespace dgx
{

namespace
{

template <class T>
struct SquaredErrorFunctor
{
    __device__ void operator()(size_t tid, const T* self, const T* other, T* out, StridedOffsetCalculator<3> offsets)
    {
        const auto tensor_offsets = offsets.get(tid);
        const auto difference = self[tensor_offsets[0]] - other[tensor_offsets[1]];
        out[tensor_offsets[2]] = difference * difference;
    }
};

template <class T>
struct SquaredErrorBackwardFunctor
{
    __device__ void operator()(size_t tid,
                               const T* input,
                               const T* target,
                               const T* grad_output,
                               T* grad_input,
                               T alpha,
                               StridedOffsetCalculator<4> offsets)
    {
        const auto tensor_offsets = offsets.get(tid);
        grad_input[tensor_offsets[3]] =
            alpha * (input[tensor_offsets[0]] - target[tensor_offsets[1]]) * grad_output[tensor_offsets[2]];
    }
};

template <class T>
struct SquaredErrorTransformOp
{
    const T* self;
    const T* target;
    StridedOffsetCalculator<2> offsets;
    T normalization;

    __device__ T operator()(int64_t index) const
    {
        const auto tensor_offsets = offsets.get(index);
        const auto difference = self[tensor_offsets[0]] - target[tensor_offsets[1]];
        return difference * difference * normalization;
    }
};

template <class T>
void mse_loss_reduce_async(const Tensor& self, const Tensor& target, T normalization, Tensor& out)
{
    std::vector<T*> partials(c10::cuda::device_count());
    const auto offsets =
        make_strided_offset_calculator<2>(self.sizes(), std::array<IntArrayRef, 2>{ self.strides(), target.strides() });

    for (const auto device : c10::irange(c10::cuda::device_count()))
    {
        C10_CUDA_CHECK(cudaSetDevice(device));
        auto stream = c10::cuda::getCurrentCUDAStream(device);
        C10_CUDA_CHECK(cudaMallocAsync(&partials[device], sizeof(T), stream));

        int64_t device_offset, device_numel;
        std::tie(device_offset, device_numel) = chunk(self.numel(), device);
        if (device_numel)
        {
            auto logical_indices = thrust::make_counting_iterator<int64_t>(device_offset);
            SquaredErrorTransformOp<T> transform_op{ self.const_data_ptr<T>(), target.const_data_ptr<T>(), offsets,
                                                     normalization };
            void* scratch_data = nullptr;
            size_t scratch_size = 0;
            C10_CUDA_CHECK(cub::DeviceReduce::TransformReduce(scratch_data, scratch_size, logical_indices,
                                                              partials[device], device_numel, ::cuda::std::plus<T>(),
                                                              transform_op, T(0), stream));
            C10_CUDA_CHECK(cudaMallocAsync(&scratch_data, scratch_size, stream));
            C10_CUDA_CHECK(cub::DeviceReduce::TransformReduce(scratch_data, scratch_size, logical_indices,
                                                              partials[device], device_numel, ::cuda::std::plus<T>(),
                                                              transform_op, T(0), stream));
            C10_CUDA_CHECK(cudaFreeAsync(scratch_data, stream));
        }
        else
        {
            C10_CUDA_CHECK(cudaMemsetAsync(partials[device], 0, sizeof(T), stream));
        }
    }

    const auto options = at::TensorOptions().dtype<T>();
    std::vector<Tensor> partial_tensors;
    partial_tensors.reserve(c10::cuda::device_count());
    for (const auto device : c10::irange(c10::cuda::device_count()))
        partial_tensors.push_back(at::from_blob(partials[device], { 1 }, options.device(c10::DeviceType::CUDA, device)));
    Tensor nccl_out = at::from_blob(out.data_ptr<T>(), { 1 }, options.device(c10::DeviceType::CUDA, 0));
    torch::cuda::nccl::reduce(partial_tensors, nccl_out);

    for (const auto device : c10::irange(c10::cuda::device_count()))
    {
        C10_CUDA_CHECK(cudaSetDevice(device));
        C10_CUDA_CHECK(cudaFreeAsync(partials[device], c10::cuda::getCurrentCUDAStream(device)));
    }

    constexpr int merge_device = 0;
    C10_CUDA_CHECK(cudaSetDevice(merge_device));
    auto merge_stream = c10::cuda::getCurrentCUDAStream(merge_device);
    cudaEvent_t merge_event = nullptr;
    C10_CUDA_CHECK(cudaEventCreateWithFlags(&merge_event, cudaEventDisableTiming));
    C10_CUDA_CHECK(cudaEventRecord(merge_event, merge_stream));
    for (const auto device : c10::irange(1, c10::cuda::device_count()))
    {
        C10_CUDA_CHECK(cudaSetDevice(device));
        C10_CUDA_CHECK(cudaStreamWaitEvent(c10::cuda::getCurrentCUDAStream(device), merge_event));
    }
    C10_CUDA_CHECK(cudaEventDestroy(merge_event));
}

} // namespace


Tensor& mse_loss_out(const Tensor& self, const Tensor& target, int64_t reduction, Tensor& out)
{
    if (reduction != Reduction::None)
    {
        out = at::empty({}, self.options());
        const auto common_dtype = at::result_type(self, target);
        const auto expanded_size = at::infer_size_dimvector(self.sizes(), target.sizes());
        const Tensor self_expanded = self.to(common_dtype).expand(expanded_size);
        const Tensor target_expanded = target.to(common_dtype).expand(expanded_size);
        Tensor reduction_out = common_dtype == out.scalar_type() ? out : at::empty({}, out.options().dtype(common_dtype));

        if (!self_expanded.numel())
        {
            if (reduction == Reduction::Mean)
                reduction_out.fill_(std::numeric_limits<double>::quiet_NaN());
            else
                reduction_out.zero_();
        }
        else
        {
            int current_device = cudaInvalidDeviceId;
            C10_CUDA_CHECK(cudaGetDevice(&current_device));
            AT_DISPATCH_V2(common_dtype, "mse_loss_out", AT_WRAP([&] {
                               const auto normalization =
                                   reduction == Reduction::Mean ? scalar_t(1) / self_expanded.numel() : scalar_t(1);
                               mse_loss_reduce_async<scalar_t>(
                                   self_expanded, target_expanded, normalization, reduction_out);
                           }),
                           AT_EXPAND(AT_FLOATING_TYPES), kHalf, kBFloat16);
            C10_CUDA_CHECK(cudaSetDevice(current_device));
        }

        if (!out.is_same(reduction_out))
            out.copy_(reduction_out);
    }
    else
    {
        auto iter = TensorIterator::borrowing_binary_op(out, self, target);
        const auto self_expanded = iter.input(0).expand(iter.output().sizes());
        const auto target_expanded = iter.input(1).expand(iter.output().sizes());
        const auto offsets = make_strided_offset_calculator<3>(
            iter.output().sizes(),
            std::array<IntArrayRef, 3>{ self_expanded.strides(), target_expanded.strides(), iter.output().strides() });
        AT_DISPATCH_V2(iter.common_dtype(), "mse_loss_out", AT_WRAP([&] {
                           mgpu_launch_func(
                               iter.numel(), SquaredErrorFunctor<scalar_t>(), self_expanded.const_data_ptr<scalar_t>(),
                               target_expanded.const_data_ptr<scalar_t>(), iter.output().data_ptr<scalar_t>(), offsets);
                       }),
                       AT_EXPAND(AT_FLOATING_TYPES), kHalf, kBFloat16);
        iter.cast_outputs();
    }
    return out;
}

Tensor mse_loss_backward(const Tensor& grad_output, const Tensor& input, const Tensor& target, int64_t reduction)
{
    Tensor grad_input = at::empty_like(input, LEGACY_CONTIGUOUS_MEMORY_FORMAT);
    return at::mse_loss_backward_out(grad_input, grad_output, input, target, reduction);
}

Tensor& mse_loss_backward_out(
    const Tensor& grad_output, const Tensor& input, const Tensor& target, int64_t reduction, Tensor& grad_input)
{
    auto norm = reduction == Reduction::Mean ? 2. / input.numel() : 2.;
    auto iter = at::TensorIteratorConfig()
                    .add_output(grad_input)
                    .add_const_input(input)
                    .add_const_input(target)
                    .add_const_input(grad_output)
                    .build();

    const auto input_expanded = iter.input(0).expand(iter.output().sizes());
    const auto target_expanded = iter.input(1).expand(iter.output().sizes());
    const auto grad_output_expanded = iter.input(2).expand(iter.output().sizes());
    const auto offsets = make_strided_offset_calculator<4>(
        iter.output().sizes(), std::array<IntArrayRef, 4>{ input_expanded.strides(), target_expanded.strides(),
                                                           grad_output_expanded.strides(), iter.output().strides() });
    AT_DISPATCH_V2(iter.common_dtype(), "mse_loss_backward_out", AT_WRAP([&] {
                       mgpu_launch_func(iter.numel(), SquaredErrorBackwardFunctor<scalar_t>(),
                                        input_expanded.const_data_ptr<scalar_t>(),
                                        target_expanded.const_data_ptr<scalar_t>(),
                                        grad_output_expanded.const_data_ptr<scalar_t>(),
                                        iter.output().data_ptr<scalar_t>(), static_cast<scalar_t>(norm), offsets);
                   }),
                   AT_EXPAND(AT_FLOATING_TYPES), kHalf, kBFloat16);

    return grad_input;
}

TORCH_LIBRARY_IMPL(aten, PrivateUse1, m)
{
    m.impl(TORCH_SELECTIVE_NAME("aten::mse_loss.out"), TORCH_FN(mse_loss_out));
    m.impl(TORCH_SELECTIVE_NAME("aten::mse_loss_backward"), TORCH_FN(mse_loss_backward));
    m.impl(TORCH_SELECTIVE_NAME("aten::mse_loss_backward.grad_input"), TORCH_FN(mse_loss_backward_out));
}

} // namespace dgx

} // namespace at::native
