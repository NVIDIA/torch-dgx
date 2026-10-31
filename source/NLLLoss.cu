/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
#include "KernelLaunchUtils.cuh"
#include "NCCLUtils.h"

#include <ATen/AccumulateType.h>
#include <ATen/Dispatch_v2.h>
#include <ATen/MemoryOverlap.h>
#include <ATen/core/Reduction.h>
#include <ATen/native/Resize.h>
#include <ATen/ops/empty.h>

#include <torch/library.h>

#include <cub/cub.cuh>
#include <thrust/iterator/counting_iterator.h>

#include <tuple>
#include <vector>

namespace at::native::dgx
{
namespace
{

template <typename acc_t>
struct NLLSum
{
    acc_t loss;
    acc_t weight;
};

template <typename acc_t>
struct NLLSumOp
{
    __host__ __device__ NLLSum<acc_t> operator()(NLLSum<acc_t> a, NLLSum<acc_t> b) const
    {
        return { a.loss + b.loss, a.weight + b.weight };
    }
};

template <typename scalar_t, typename target_t>
struct NLLTransform
{
    using acc_t = acc_type<scalar_t, true>;
    const scalar_t* input;
    const target_t* target;
    const scalar_t* weight;
    int64_t classes;
    int64_t spatial_size;
    int64_t ignore_index;

    __device__ NLLSum<acc_t> operator()(int64_t index) const
    {
        const int64_t label = target[index];
        if (label == ignore_index)
            return { 0, 0 };
        CUDA_KERNEL_ASSERT(label >= 0 && label < classes && "NLL target out of bounds");
        const auto offset = (index / spatial_size * classes + label) * spatial_size + index % spatial_size;
        const acc_t w = weight ? static_cast<acc_t>(weight[label]) : acc_t(1);
        return { -static_cast<acc_t>(input[offset]) * w, w };
    }

    __device__ void operator()(int64_t index, scalar_t* output) const
    {
        output[index] = static_cast<scalar_t>((*this)(index).loss);
    }
};

template <typename scalar_t, typename acc_t>
__global__ void nll_finalize_kernel(const acc_t* sums, int64_t reduction, scalar_t* output, scalar_t* total_weight)
{
    *total_weight = static_cast<scalar_t>(sums[1]);
    *output = static_cast<scalar_t>(reduction == Reduction::Mean ? sums[0] / sums[1] : sums[0]);
}

template <typename scalar_t, typename target_t>
void nll_reduce(
    int64_t count, NLLTransform<scalar_t, target_t> transform, int64_t reduction, Tensor& output, Tensor& total_weight)
{
    using acc_t = acc_type<scalar_t, true>;
    static_assert(sizeof(NLLSum<acc_t>) == 2 * sizeof(acc_t));
    const Tensor sums = at::empty({ 2 }, output.options().dtype<acc_t>());
    std::vector<acc_t*> partials(c10::cuda::device_count());
    int current_device = cudaInvalidDeviceId;
    C10_CUDA_CHECK(cudaGetDevice(&current_device));
    parallel_for_each_async([&](int device, cudaStream_t stream) {
        C10_CUDA_CHECK(cudaMallocAsync(&partials[device], 2 * sizeof(acc_t), stream));
        const auto [offset, size] = chunk(count, device);
        if (size)
        {
            const auto indices = thrust::make_counting_iterator<int64_t>(offset);
            auto* partial = reinterpret_cast<NLLSum<acc_t>*>(partials[device]);
            void* scratch = nullptr;
            size_t bytes = 0;
            C10_CUDA_CHECK(cub::DeviceReduce::TransformReduce(
                scratch, bytes, indices, partial, size, NLLSumOp<acc_t>(), transform, NLLSum<acc_t>{ 0, 0 }, stream));
            C10_CUDA_CHECK(cudaMallocAsync(&scratch, bytes, stream));
            C10_CUDA_CHECK(cub::DeviceReduce::TransformReduce(
                scratch, bytes, indices, partial, size, NLLSumOp<acc_t>(), transform, NLLSum<acc_t>{ 0, 0 }, stream));
            C10_CUDA_CHECK(cudaFreeAsync(scratch, stream));
        }
        else
        {
            C10_CUDA_CHECK(cudaMemsetAsync(partials[device], 0, 2 * sizeof(acc_t), stream));
        }
    });
    nccl_reduce_partials(partials, sums.data_ptr<acc_t>(), 2);
    C10_CUDA_CHECK(cudaSetDevice(0));
    nll_finalize_kernel<<<1, 1, 0, c10::cuda::getCurrentCUDAStream(0)>>>(
        sums.const_data_ptr<acc_t>(), reduction, output.data_ptr<scalar_t>(), total_weight.data_ptr<scalar_t>());
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    parallel_for_each([&](int device, cudaStream_t stream) { C10_CUDA_CHECK(cudaFreeAsync(partials[device], stream)); });
    C10_CUDA_CHECK(cudaSetDevice(current_device));
}

template <typename scalar_t, typename target_t>
struct NLLBackwardFunctor
{
    const target_t* target;
    const scalar_t* weight;
    const scalar_t* grad_output;
    const scalar_t* total_weight;
    scalar_t* grad_input;
    int64_t classes;
    int64_t spatial_size;
    int64_t ignore_index;
    int64_t reduction;

    __device__ void operator()(int64_t index) const
    {
        using acc_t = acc_type<scalar_t, true>;
        const int64_t channel = index / spatial_size % classes;
        const int64_t sample = index / (classes * spatial_size) * spatial_size + index % spatial_size;
        const int64_t label = target[sample];
        if (label == ignore_index)
        {
            grad_input[index] = scalar_t(0);
            return;
        }
        CUDA_KERNEL_ASSERT(label >= 0 && label < classes && "NLL target out of bounds");
        if (channel != label)
        {
            grad_input[index] = scalar_t(0);
            return;
        }
        acc_t grad = -static_cast<acc_t>(grad_output[reduction == Reduction::None ? sample : 0]);
        if (reduction == Reduction::Mean)
            grad /= static_cast<acc_t>(*total_weight);
        if (weight)
            grad *= static_cast<acc_t>(weight[label]);
        grad_input[index] = static_cast<scalar_t>(grad);
    }
};

void check_nll_args(const Tensor& input, const Tensor& target, const Tensor& weight, int64_t reduction, bool spatial)
{
    TORCH_CHECK(reduction >= Reduction::None && reduction <= Reduction::Sum, "Invalid NLL reduction");
    TORCH_CHECK(spatial ? input.dim() == 4 : (input.dim() == 1 || input.dim() == 2),
                spatial ? "NLL spatial input must be 4D" : "NLL input must be 1D or 2D");
    TORCH_CHECK(input.is_floating_point(), "NLL input must have a floating point dtype");
    TORCH_CHECK(target.scalar_type() == kLong || target.scalar_type() == kByte, "NLL target must be int64 or uint8");
    TORCH_CHECK(target.device() == input.device(), "NLL input and target must be on the same device");
    const auto classes = input.size(input.dim() == 1 ? 0 : 1);
    if (input.dim() == 1)
    {
        TORCH_CHECK(target.dim() <= 1 && target.numel() == 1, "Expected one target for an unbatched NLL input");
    }
    else
    {
        auto expected = input.sizes().vec();
        expected.erase(expected.begin() + 1);
        TORCH_CHECK(target.sizes() == IntArrayRef(expected), "NLL target shape does not match input");
    }
    if (weight.defined())
    {
        TORCH_CHECK(weight.dim() == 1 && weight.numel() == classes, "NLL weight must contain one value per class");
        TORCH_CHECK(weight.device() == input.device() && weight.scalar_type() == input.scalar_type(),
                    "NLL weight must have the input device and dtype");
    }
}

std::vector<int64_t> nll_output_shape(const Tensor& input, const Tensor& target, int64_t reduction)
{
    return reduction == Reduction::None && input.dim() != 1 ? target.sizes().vec() : std::vector<int64_t>{};
}

void check_nll_output(const Tensor& output, const Tensor& input, const Tensor& target, const Tensor& weight)
{
    TORCH_CHECK(output.device() == input.device() && output.scalar_type() == input.scalar_type(),
                "NLL output must have the input device and dtype");
    at::assert_no_internal_overlap(output);
    at::assert_no_overlap(output, input);
    at::assert_no_overlap(output, target);
    if (weight.defined())
        at::assert_no_overlap(output, weight);
}

template <bool spatial>
std::tuple<Tensor&, Tensor&> nll_forward_out(const Tensor& self,
                                             const Tensor& target,
                                             const std::optional<Tensor>& weight_opt,
                                             int64_t reduction,
                                             int64_t ignore_index,
                                             Tensor& output,
                                             Tensor& total_weight)
{
    const auto weight = weight_opt.value_or(Tensor());
    check_nll_args(self, target, weight, reduction, spatial);
    at::native::resize_output(output, nll_output_shape(self, target, reduction));
    at::native::resize_output(total_weight, {});
    check_nll_output(output, self, target, weight);
    check_nll_output(total_weight, self, target, weight);
    at::assert_no_overlap(output, total_weight);
    const auto input = self.contiguous();
    const auto labels = target.contiguous();
    const auto weights = weight.defined() ? weight.contiguous() : weight;
    auto result = output.is_contiguous() ? output : at::empty(output.sizes(), output.options());
    const int64_t classes = self.size(self.dim() == 1 ? 0 : 1);
    const int64_t spatial_size = spatial ? self.size(2) * self.size(3) : 1;
    const bool unreduced = reduction == Reduction::None && self.dim() != 1;

    AT_DISPATCH_V2(self.scalar_type(), "nll_loss_forward", AT_WRAP([&] {
                       using value_t = scalar_t;
                       AT_DISPATCH_V2(target.scalar_type(), "nll_loss_target", AT_WRAP([&] {
                                          const NLLTransform<value_t, scalar_t> transform{
                                              input.const_data_ptr<value_t>(),
                                              labels.const_data_ptr<scalar_t>(),
                                              weights.defined() ? weights.const_data_ptr<value_t>() : nullptr,
                                              classes,
                                              spatial_size,
                                              ignore_index
                                          };
                                          if (unreduced)
                                          {
                                              total_weight.zero_();
                                              mgpu_launch_func(labels.numel(), transform, result.data_ptr<value_t>());
                                          }
                                          else
                                          {
                                              nll_reduce(labels.numel(), transform, reduction, result, total_weight);
                                          }
                                      }),
                                      kLong, kByte);
                   }),
                   AT_EXPAND(AT_FLOATING_TYPES), kHalf, kBFloat16);

    if (!result.is_same(output))
        output.copy_(result);
    return { output, total_weight };
}

template <bool spatial>
std::tuple<Tensor, Tensor> nll_forward(const Tensor& self,
                                       const Tensor& target,
                                       const std::optional<Tensor>& weight,
                                       int64_t reduction,
                                       int64_t ignore_index)
{
    auto output = at::empty({ 0 }, self.options());
    auto total_weight = at::empty({}, self.options());
    nll_forward_out<spatial>(self, target, weight, reduction, ignore_index, output, total_weight);
    return { output, total_weight };
}

template <bool spatial>
Tensor& nll_backward_out(const Tensor& grad_output,
                         const Tensor& self,
                         const Tensor& target,
                         const std::optional<Tensor>& weight_opt,
                         int64_t reduction,
                         int64_t ignore_index,
                         const Tensor& total_weight,
                         Tensor& grad_input)
{
    const auto weight = weight_opt.value_or(Tensor());
    check_nll_args(self, target, weight, reduction, spatial);
    TORCH_CHECK(grad_output.sizes() == IntArrayRef(nll_output_shape(self, target, reduction)),
                "NLL grad_output has an incorrect shape");
    TORCH_CHECK(grad_output.device() == self.device() && grad_output.scalar_type() == self.scalar_type(),
                "NLL grad_output must have the input device and dtype");
    TORCH_CHECK(total_weight.numel() == 1 && total_weight.device() == self.device() &&
                    total_weight.scalar_type() == self.scalar_type(),
                "Invalid NLL total_weight");
    at::native::resize_output(grad_input, self.sizes());
    check_nll_output(grad_input, self, target, weight);
    at::assert_no_overlap(grad_input, grad_output);
    at::assert_no_overlap(grad_input, total_weight);
    const auto labels = target.contiguous();
    const auto weights = weight.defined() ? weight.contiguous() : weight;
    const auto grad = grad_output.contiguous();
    auto result = grad_input.is_contiguous() ? grad_input : at::empty(self.sizes(), grad_input.options());
    const int64_t classes = self.size(self.dim() == 1 ? 0 : 1);
    const int64_t spatial_size = spatial ? self.size(2) * self.size(3) : 1;
    AT_DISPATCH_V2(self.scalar_type(), "nll_loss_backward", AT_WRAP([&] {
                       using value_t = scalar_t;
                       AT_DISPATCH_V2(target.scalar_type(), "nll_loss_backward_target", AT_WRAP([&] {
                                          const NLLBackwardFunctor<value_t, scalar_t> functor{
                                              labels.const_data_ptr<scalar_t>(),
                                              weights.defined() ? weights.const_data_ptr<value_t>() : nullptr,
                                              grad.const_data_ptr<value_t>(),
                                              total_weight.const_data_ptr<value_t>(),
                                              result.data_ptr<value_t>(),
                                              classes,
                                              spatial_size,
                                              ignore_index,
                                              reduction
                                          };
                                          mgpu_launch_func(result.numel(), functor);
                                      }),
                                      kLong, kByte);
                   }),
                   AT_EXPAND(AT_FLOATING_TYPES), kHalf, kBFloat16);
    if (!result.is_same(grad_input))
        grad_input.copy_(result);
    return grad_input;
}

template <bool spatial>
Tensor nll_backward(const Tensor& grad_output,
                    const Tensor& self,
                    const Tensor& target,
                    const std::optional<Tensor>& weight,
                    int64_t reduction,
                    int64_t ignore_index,
                    const Tensor& total_weight)
{
    auto grad_input = at::empty(self.sizes(), self.options());
    nll_backward_out<spatial>(grad_output, self, target, weight, reduction, ignore_index, total_weight, grad_input);
    return grad_input;
}

} // namespace

TORCH_LIBRARY_IMPL(aten, PrivateUse1, m)
{
    m.impl(TORCH_SELECTIVE_NAME("aten::nll_loss_forward.output"), TORCH_FN(nll_forward_out<false>));
    m.impl(TORCH_SELECTIVE_NAME("aten::nll_loss_backward.grad_input"), TORCH_FN(nll_backward_out<false>));
    m.impl(TORCH_SELECTIVE_NAME("aten::nll_loss2d_forward"), TORCH_FN(nll_forward<true>));
    m.impl(TORCH_SELECTIVE_NAME("aten::nll_loss2d_forward.output"), TORCH_FN(nll_forward_out<true>));
    m.impl(TORCH_SELECTIVE_NAME("aten::nll_loss2d_backward"), TORCH_FN(nll_backward<true>));
    m.impl(TORCH_SELECTIVE_NAME("aten::nll_loss2d_backward.grad_input"), TORCH_FN(nll_backward_out<true>));
}

} // namespace at::native::dgx
