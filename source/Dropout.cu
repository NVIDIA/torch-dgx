/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
#include "DGXGenerator.h"
#include "KernelLaunchUtils.cuh"

#include <ATen/AccumulateType.h>
#include <ATen/Dispatch_v2.h>
#include <ATen/ops/empty.h>
#include <ATen/ops/empty_like.h>
#include <ATen/ops/ones_like.h>
#include <ATen/ops/zeros_like.h>

#include <torch/library.h>

#include <curand_kernel.h>

#include <cstdint>
#include <optional>
#include <tuple>
#include <utility>

namespace at::native
{

namespace dgx
{

namespace
{

constexpr uint64_t kPhiloxOffsetPerCall = 4;

template <typename scalar_t>
struct DropoutFunctor
{
    using accscalar_t = acc_type<scalar_t, true>;

    __device__ void operator()(size_t tid,
                               const scalar_t* input,
                               double probability,
                               accscalar_t scale,
                               uint64_t seed,
                               uint64_t offset,
                               scalar_t* output,
                               bool* mask)
    {
        curandStatePhilox4_32_10_t state;
        curand_init(seed, tid, offset, &state);
        const bool keep = curand_uniform(&state) > probability;
        mask[tid] = keep;
        output[tid] = keep ? static_cast<scalar_t>(static_cast<accscalar_t>(input[tid]) * scale) : scalar_t(0);
    }
};

template <typename scalar_t>
struct DropoutBackwardFunctor
{
    using accscalar_t = acc_type<scalar_t, true>;

    __device__ void operator()(
        size_t tid, const scalar_t* grad_output, const bool* mask, accscalar_t scale, scalar_t* grad_input)
    {
        grad_input[tid] =
            mask[tid] ? static_cast<scalar_t>(static_cast<accscalar_t>(grad_output[tid]) * scale) : scalar_t(0);
    }
};

} // namespace

std::tuple<Tensor, Tensor> native_dropout(const Tensor& input, double probability, std::optional<bool> train)
{
    TORCH_CHECK(probability >= 0.0 && probability <= 1.0, "dropout probability has to be between 0 and 1");

    if (train.has_value() && !*train)
    {
        return std::make_tuple(input.clone(), at::ones_like(input, input.options().dtype(kBool)));
    }
    if (probability == 1.0)
    {
        return std::make_tuple(at::zeros_like(input), at::zeros_like(input, input.options().dtype(kBool)));
    }
    if (!input.numel())
    {
        return std::make_tuple(input.clone(), at::empty_like(input, input.options().dtype(kBool)));
    }

    const auto contiguous_input = input.contiguous();
    Tensor output = at::empty(input.sizes(), input.options());
    Tensor mask = at::empty(input.sizes(), input.options().dtype(kBool));
    const auto [seed, offset] = at::dgx::philox_engine_inputs(std::nullopt, kPhiloxOffsetPerCall);
    const double scale = 1.0 / (1.0 - probability);

    AT_DISPATCH_V2(input.scalar_type(), "native_dropout", AT_WRAP([&] {
                       using accscalar_t = acc_type<scalar_t, true>;
                       // Align for both output element sizes so GPUs never share an output or mask page.

                       mgpu_launch_func(input.numel(), DropoutFunctor<scalar_t>(),
                                        contiguous_input.const_data_ptr<scalar_t>(), probability,
                                        static_cast<accscalar_t>(scale), seed, offset, output.data_ptr<scalar_t>(),
                                        mask.data_ptr<bool>());
                   }),
                   AT_EXPAND(AT_FLOATING_TYPES), kHalf, kBFloat16);
    return std::make_tuple(std::move(output), std::move(mask));
}

Tensor native_dropout_backward(const Tensor& grad_output, const Tensor& mask, double scale)
{
    TORCH_CHECK(mask.scalar_type() == kBool, "Mask should have Bool dtype");
    TORCH_CHECK(mask.sizes() == grad_output.sizes(), "Expected mask and grad_output to have the same shape");
    TORCH_CHECK(mask.device() == grad_output.device(), "Expected mask and grad_output to be on the same device");

    const auto contiguous_grad_output = grad_output.contiguous();
    const auto contiguous_mask = mask.contiguous();
    Tensor grad_input = at::empty(grad_output.sizes(), grad_output.options());
    AT_DISPATCH_V2(grad_output.scalar_type(), "native_dropout_backward", AT_WRAP([&] {
                       using accscalar_t = acc_type<scalar_t, true>;

                       mgpu_launch_func(grad_output.numel(), DropoutBackwardFunctor<scalar_t>(),
                                        contiguous_grad_output.const_data_ptr<scalar_t>(),
                                        contiguous_mask.const_data_ptr<bool>(), static_cast<accscalar_t>(scale),
                                        grad_input.data_ptr<scalar_t>());
                   }),
                   AT_EXPAND(AT_FLOATING_TYPES), kHalf, kBFloat16);
    return grad_input;
}

TORCH_LIBRARY_IMPL(aten, PrivateUse1, m)
{
    m.impl(TORCH_SELECTIVE_NAME("aten::native_dropout"), TORCH_FN(native_dropout));
    m.impl(TORCH_SELECTIVE_NAME("aten::native_dropout_backward"), TORCH_FN(native_dropout_backward));
}

} // namespace dgx

} // namespace at::native
