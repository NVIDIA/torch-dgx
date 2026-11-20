/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
#include "KernelLaunchUtils.cuh"

#include <ATen/AccumulateType.h>
#include <ATen/Dispatch_v2.h>
#include <ATen/MemoryOverlap.h>
#include <ATen/native/Pool.h>
#include <ATen/native/Resize.h>

#include <torch/library.h>

#include <cuda/std/algorithm>

#include <array>

namespace at::native
{

namespace dgx
{

namespace
{

std::array<int64_t, 2> pool2d_pair(IntArrayRef values, const char* name)
{
    TORCH_CHECK(values.size() == 1 || values.size() == 2, "avg_pool2d: ", name, " must contain one or two integers");
    return { values[0], values.size() == 1 ? values[0] : values[1] };
}

template <typename scalar_t>
struct AvgPool2dFunctor
{
    using acc_t = acc_type<scalar_t, true>;

    int64_t kernel_h, kernel_w;
    int64_t stride_h, stride_w;
    int64_t pad_h, pad_w;
    int64_t divisor_override;
    bool count_include_pad;
    bool channels_last;

    __device__ void operator()(int64_t tid,
                               PackedTensorAccessor64<const scalar_t, 4, RestrictPtrTraits> input,
                               PackedTensorAccessor64<scalar_t, 4, RestrictPtrTraits> output) const
    {
        const int64_t channels = output.size(1);
        const int64_t height = output.size(2);
        const int64_t width = output.size(3);
        const int64_t n = tid / (channels * height * width);
        const int64_t c = channels_last ? tid % channels : tid / (height * width) % channels;
        const int64_t h = channels_last ? tid / (channels * width) % height : tid / width % height;
        const int64_t w = channels_last ? tid / channels % width : tid % width;

        const int64_t start_h = h * stride_h - pad_h;
        const int64_t start_w = w * stride_w - pad_w;
        const int64_t end_h = ::cuda::std::min(start_h + kernel_h, input.size(2) + pad_h);
        const int64_t end_w = ::cuda::std::min(start_w + kernel_w, input.size(3) + pad_w);
        const int64_t first_h = ::cuda::std::max<int64_t>(start_h, 0);
        const int64_t first_w = ::cuda::std::max<int64_t>(start_w, 0);
        const int64_t last_h = ::cuda::std::min(end_h, input.size(2));
        const int64_t last_w = ::cuda::std::min(end_w, input.size(3));

        if (first_h >= last_h || first_w >= last_w)
        {
            output[n][c][h][w] = scalar_t(0);
            return;
        }

        acc_t sum = 0;
        const auto input_plane = input[n][c];
        for (int64_t ih = first_h; ih < last_h; ++ih)
        {
            for (int64_t iw = first_w; iw < last_w; ++iw)
            {
                sum += static_cast<acc_t>(input_plane[ih][iw]);
            }
        }

        // Ceil-mode windows include explicit padding, but not the extra tail beyond it.
        const int64_t pool_size =
            count_include_pad ? (end_h - start_h) * (end_w - start_w) : (last_h - first_h) * (last_w - first_w);
        const int64_t divisor = divisor_override != 0 ? divisor_override : pool_size;
        output[n][c][h][w] = static_cast<scalar_t>(sum / static_cast<acc_t>(divisor));
    }
};

template <typename scalar_t>
struct AvgPool2dBackwardFunctor
{
    using acc_t = acc_type<scalar_t, true>;

    int64_t kernel_h, kernel_w;
    int64_t stride_h, stride_w;
    int64_t pad_h, pad_w;
    int64_t divisor_override;
    bool count_include_pad;
    bool channels_last;

    __device__ void operator()(int64_t tid,
                               PackedTensorAccessor64<const scalar_t, 4, RestrictPtrTraits> grad_output,
                               PackedTensorAccessor64<scalar_t, 4, RestrictPtrTraits> grad_input) const
    {
        const int64_t channels = grad_input.size(1);
        const int64_t height = grad_input.size(2);
        const int64_t width = grad_input.size(3);
        const int64_t n = tid / (channels * height * width);
        const int64_t c = channels_last ? tid % channels : tid / (height * width) % channels;
        const int64_t h = channels_last ? tid / (channels * width) % height : tid / width % height;
        const int64_t w = channels_last ? tid / channels % width : tid % width;

        const int64_t padded_h = h + pad_h;
        const int64_t padded_w = w + pad_w;
        const int64_t first_h = padded_h < kernel_h ? 0 : (padded_h - kernel_h) / stride_h + 1;
        const int64_t first_w = padded_w < kernel_w ? 0 : (padded_w - kernel_w) / stride_w + 1;
        const int64_t last_h = ::cuda::std::min(padded_h / stride_h + 1, grad_output.size(2));
        const int64_t last_w = ::cuda::std::min(padded_w / stride_w + 1, grad_output.size(3));

        acc_t sum = 0;
        for (int64_t oh = first_h; oh < last_h; ++oh)
        {
            const int64_t start_h = oh * stride_h - pad_h;
            const int64_t end_h = ::cuda::std::min(start_h + kernel_h, height + pad_h);
            for (int64_t ow = first_w; ow < last_w; ++ow)
            {
                const int64_t start_w = ow * stride_w - pad_w;
                const int64_t end_w = ::cuda::std::min(start_w + kernel_w, width + pad_w);
                const int64_t pool_size =
                    count_include_pad ? (end_h - start_h) * (end_w - start_w) :
                                        (::cuda::std::min(end_h, height) - ::cuda::std::max<int64_t>(start_h, 0)) *
                                            (::cuda::std::min(end_w, width) - ::cuda::std::max<int64_t>(start_w, 0));
                const int64_t divisor = divisor_override != 0 ? divisor_override : pool_size;
                sum += static_cast<acc_t>(grad_output[n][c][oh][ow]) / static_cast<acc_t>(divisor);
            }
        }
        // One thread owns each input gradient, including pixels skipped by the
        // pooling windows. Overlapping windows need no atomics or zero-fill.
        grad_input[n][c][h][w] = static_cast<scalar_t>(sum);
    }
};

} // namespace

Tensor& avg_pool2d_out(const Tensor& self,
                       IntArrayRef kernel_size,
                       IntArrayRef stride,
                       IntArrayRef padding,
                       bool ceil_mode,
                       bool count_include_pad,
                       std::optional<int64_t> divisor_override,
                       Tensor& out)
{
    TORCH_CHECK(self.dim() == 3 || self.dim() == 4, "avg_pool2d: expected a 3D or 4D input");
    TORCH_CHECK(out.device() == self.device(), "avg_pool2d: output must be on the input device");
    TORCH_CHECK(out.scalar_type() == self.scalar_type(), "avg_pool2d: output must have the input dtype");
    const auto [kernel_h, kernel_w] = pool2d_pair(kernel_size, "kernel_size");
    const auto [stride_h, stride_w] = pool2d_pair(stride.empty() ? kernel_size : stride, "stride");
    const auto [pad_h, pad_w] = pool2d_pair(padding, "padding");
    TORCH_CHECK(kernel_h > 0 && kernel_w > 0, "avg_pool2d: kernel size must be positive");
    TORCH_CHECK(stride_h > 0 && stride_w > 0, "avg_pool2d: stride must be positive");
    TORCH_CHECK(!divisor_override.has_value() || *divisor_override != 0, "avg_pool2d: divisor must be nonzero");

    const int64_t output_h = pooling_output_shape<int64_t>(self.size(-2), kernel_h, pad_h, stride_h, 1, ceil_mode);
    const int64_t output_w = pooling_output_shape<int64_t>(self.size(-1), kernel_w, pad_w, stride_w, 1, ceil_mode);
    pool2d_shape_check(self, kernel_h, kernel_w, stride_h, stride_w, pad_h, pad_w, 1, 1, self.size(-3), self.size(-2),
                       self.size(-1), output_h, output_w, self.suggest_memory_format());
    auto output_sizes = self.sizes().vec();
    output_sizes[self.dim() - 2] = output_h;
    output_sizes[self.dim() - 1] = output_w;

    // Preserve aliased input before resizing or writing the output.
    auto input = self.is_alias_of(out) ? self.clone() : self;
    resize_output(out, output_sizes);
    assert_no_internal_overlap(out);
    if (!out.numel())
        return out;

    auto output = out;
    if (input.dim() == 3)
    {
        input = input.unsqueeze(0);
        output = output.unsqueeze(0);
    }
    const bool channels_last = output.suggest_memory_format() == MemoryFormat::ChannelsLast;

    // The launch helper joins current streams; the allocator orders release of any input clone after queued work.
    AT_DISPATCH_V2(
        input.scalar_type(), "avg_pool2d_out", AT_WRAP([&] {
            const AvgPool2dFunctor<scalar_t> functor{
                kernel_h,          kernel_w,     stride_h, stride_w, pad_h, pad_w, divisor_override.value_or(0),
                count_include_pad, channels_last
            };
            mgpu_launch_func(out.numel(), functor, input.packed_accessor64<const scalar_t, 4, RestrictPtrTraits>(),
                             output.packed_accessor64<scalar_t, 4, RestrictPtrTraits>());
        }),
        AT_EXPAND(AT_FLOATING_TYPES), kHalf, kBFloat16);
    return out;
}

Tensor& avg_pool2d_backward_out(const Tensor& grad_output,
                                const Tensor& self,
                                IntArrayRef kernel_size,
                                IntArrayRef stride,
                                IntArrayRef padding,
                                bool ceil_mode,
                                bool count_include_pad,
                                std::optional<int64_t> divisor_override,
                                Tensor& grad_input)
{
    TORCH_CHECK(self.dim() == 3 || self.dim() == 4, "avg_pool2d_backward: expected a 3D or 4D input");
    TORCH_CHECK(grad_output.device() == self.device() && grad_input.device() == self.device(),
                "avg_pool2d_backward: gradients must be on the input device");
    TORCH_CHECK(grad_output.scalar_type() == self.scalar_type() && grad_input.scalar_type() == self.scalar_type(),
                "avg_pool2d_backward: gradients must have the input dtype");
    const auto [kernel_h, kernel_w] = pool2d_pair(kernel_size, "kernel_size");
    const auto [stride_h, stride_w] = pool2d_pair(stride.empty() ? kernel_size : stride, "stride");
    const auto [pad_h, pad_w] = pool2d_pair(padding, "padding");
    TORCH_CHECK(kernel_h > 0 && kernel_w > 0, "avg_pool2d_backward: kernel size must be positive");
    TORCH_CHECK(stride_h > 0 && stride_w > 0, "avg_pool2d_backward: stride must be positive");
    TORCH_CHECK(!divisor_override.has_value() || *divisor_override != 0, "avg_pool2d_backward: divisor must be nonzero");
    const int64_t output_h = pooling_output_shape<int64_t>(self.size(-2), kernel_h, pad_h, stride_h, 1, ceil_mode);
    const int64_t output_w = pooling_output_shape<int64_t>(self.size(-1), kernel_w, pad_w, stride_w, 1, ceil_mode);
    pool2d_shape_check(self, kernel_h, kernel_w, stride_h, stride_w, pad_h, pad_w, 1, 1, self.size(-3), self.size(-2),
                       self.size(-1), output_h, output_w, self.suggest_memory_format());
    check_dim_size(grad_output, self.dim(), self.dim() - 3, self.size(-3));
    check_dim_size(grad_output, self.dim(), self.dim() - 2, output_h);
    check_dim_size(grad_output, self.dim(), self.dim() - 1, output_w);
    TORCH_CHECK(self.dim() != 4 || grad_output.size(0) == self.size(0),
                "avg_pool2d_backward: grad_output batch size must match the input");

    auto grad_output_ = grad_output.is_alias_of(grad_input) ? grad_output.clone() : grad_output;
    resize_output(grad_input, self.sizes());
    assert_no_internal_overlap(grad_input);
    if (!grad_input.numel())
        return grad_input;
    auto grad_input_ = grad_input;
    if (self.dim() == 3)
    {
        grad_output_ = grad_output_.unsqueeze(0);
        grad_input_ = grad_input_.unsqueeze(0);
    }
    const bool channels_last = grad_input_.suggest_memory_format() == MemoryFormat::ChannelsLast;
    AT_DISPATCH_V2(
        self.scalar_type(), "avg_pool2d_backward_out", AT_WRAP([&] {
            const AvgPool2dBackwardFunctor<scalar_t> functor{
                kernel_h,          kernel_w,     stride_h, stride_w, pad_h, pad_w, divisor_override.value_or(0),
                count_include_pad, channels_last
            };
            mgpu_launch_func(grad_input.numel(), functor,
                             grad_output_.packed_accessor64<const scalar_t, 4, RestrictPtrTraits>(),
                             grad_input_.packed_accessor64<scalar_t, 4, RestrictPtrTraits>());
        }),
        AT_EXPAND(AT_FLOATING_TYPES), kHalf, kBFloat16);
    return grad_input;
}

TORCH_LIBRARY_IMPL(aten, PrivateUse1, m)
{
    m.impl(TORCH_SELECTIVE_NAME("aten::avg_pool2d.out"), TORCH_FN(avg_pool2d_out));
    m.impl(TORCH_SELECTIVE_NAME("aten::avg_pool2d_backward.grad_input"), TORCH_FN(avg_pool2d_backward_out));
}

} // namespace dgx

} // namespace at::native
