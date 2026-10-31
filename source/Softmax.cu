/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
#include "KernelLaunchUtils.cuh"
#include "ReductionLayout.cuh"

#include <ATen/AccumulateType.h>
#include <ATen/Dispatch_v2.h>
#include <ATen/MemoryOverlap.h>
#include <ATen/WrapDimUtils.h>
#include <ATen/native/Resize.h>
#include <ATen/ops/_safe_softmax_native.h>
#include <ATen/ops/empty.h>

#include <torch/library.h>

#include <cub/block/block_reduce.cuh>
#include <cuda/functional>
#include <cuda/std/cmath>

#include <limits>
#include <type_traits>

namespace at::native
{

namespace dgx
{

namespace
{

template <typename scalar_t>
struct SoftmaxForwardFunctor
{
    __device__ void operator()(size_t tid, int64_t dim_size, int64_t inner, const scalar_t* self, scalar_t* out)
    {
        const size_t outer_index = tid / inner;
        const size_t inner_index = tid % inner;
        const auto base_offset = outer_index * dim_size * inner + inner_index;

        scalar_t max = self[base_offset];
        scalar_t sum = 1;
        for (int64_t dim_index = 1; dim_index < dim_size; ++dim_index)
        {
            const auto value = self[base_offset + dim_index * inner];
            if (value > max)
            {
                sum = sum * ::cuda::std::exp(max - value) + 1;
                max = value;
            }
            else
            {
                sum += ::cuda::std::exp(value - max);
            }
        }

        const scalar_t inv_sum = scalar_t(1) / sum;
        for (int64_t dim_index = 0; dim_index < dim_size; ++dim_index)
        {
            const auto offset = base_offset + dim_index * inner;
            out[offset] = ::cuda::std::exp(self[offset] - max) * inv_sum;
        }
    }
};

template <typename scalar_t>
struct SoftmaxMaxNan
{
    __device__ scalar_t operator()(scalar_t a, scalar_t b) const
    {
        return (::cuda::std::isnan(a) || a > b) ? a : b;
    }
};

template <typename scalar_t, bool safe = false>
__global__ __launch_bounds__(kNumThreads) void softmax_forward_block_kernel(int64_t row_offset,
                                                                            int64_t dim_size,
                                                                            const scalar_t* self,
                                                                            scalar_t* out)
{
    using BlockReduce = cub::BlockReduce<scalar_t, kNumThreads>;
    using MaxOp = std::conditional_t<safe, SoftmaxMaxNan<scalar_t>, ::cuda::maximum<scalar_t>>;
    __shared__ typename BlockReduce::TempStorage temp_storage;
    __shared__ scalar_t row_max;
    __shared__ scalar_t row_inv_sum;

    const auto row = row_offset + static_cast<int64_t>(blockIdx.x);
    const auto base_offset = row * dim_size;
    scalar_t thread_max = -std::numeric_limits<scalar_t>::infinity();
    for (int64_t dim_index = threadIdx.x; dim_index < dim_size; dim_index += blockDim.x)
    {
        const auto value = self[base_offset + dim_index];
        if constexpr (safe)
        {
            thread_max = MaxOp()(thread_max, value);
        }
        else if (value > thread_max)
        {
            thread_max = value;
        }
    }

    const auto block_max = BlockReduce(temp_storage).Reduce(thread_max, MaxOp());
    if (!threadIdx.x)
    {
        row_max = block_max;
    }
    __syncthreads();

    if constexpr (safe)
    {
        // Preserve NaNs in the max reduction: [-inf, NaN] is not fully masked.
        // The uniform branch avoids exp(-inf - -inf) for fully masked rows.
        if (row_max == -std::numeric_limits<scalar_t>::infinity())
        {
            for (int64_t dim_index = threadIdx.x; dim_index < dim_size; dim_index += blockDim.x)
                out[base_offset + dim_index] = scalar_t(0);
            return;
        }
    }

    scalar_t thread_sum = 0;
    for (int64_t dim_index = threadIdx.x; dim_index < dim_size; dim_index += blockDim.x)
    {
        thread_sum += ::cuda::std::exp(self[base_offset + dim_index] - row_max);
    }

    const auto block_sum = BlockReduce(temp_storage).Sum(thread_sum);
    if (!threadIdx.x)
    {
        row_inv_sum = scalar_t(1) / block_sum;
    }
    __syncthreads();

    for (int64_t dim_index = threadIdx.x; dim_index < dim_size; dim_index += blockDim.x)
    {
        const auto offset = base_offset + dim_index;
        out[offset] = ::cuda::std::exp(self[offset] - row_max) * row_inv_sum;
    }
}

template <typename scalar_t, bool safe = false>
void launch_softmax_forward(int64_t rows, int64_t dim_size, const scalar_t* self, scalar_t* out)
{
    parallel_for_each([&](int device, cudaStream_t stream) {
        int64_t row_offset, row_count;
        std::tie(row_offset, row_count) = chunk(rows, device);
        if (!row_count)
            return;
        softmax_forward_block_kernel<scalar_t, safe>
            <<<static_cast<unsigned int>(row_count), kNumThreads, 0, stream>>>(row_offset, dim_size, self, out);
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    });
}

template <typename output_t, typename input_t>
struct SoftmaxBackwardFunctor
{
    __device__ void operator()(size_t tid,
                               int64_t dim_size,
                               int64_t inner,
                               const output_t* grad_output,
                               const output_t* output,
                               input_t* grad_input)
    {
        using accscalar_t = acc_type<output_t, true>;
        const size_t outer_index = tid / inner;
        const size_t inner_index = tid % inner;
        const auto base_offset = outer_index * dim_size * inner + inner_index;

        accscalar_t dot = 0;
        for (int64_t dim_index = 0; dim_index < dim_size; ++dim_index)
        {
            const auto offset = base_offset + dim_index * inner;
            dot += static_cast<accscalar_t>(grad_output[offset]) * static_cast<accscalar_t>(output[offset]);
        }

        for (int64_t dim_index = 0; dim_index < dim_size; ++dim_index)
        {
            const auto offset = base_offset + dim_index * inner;
            grad_input[offset] = static_cast<input_t>(static_cast<accscalar_t>(output[offset]) *
                                                      (static_cast<accscalar_t>(grad_output[offset]) - dot));
        }
    }
};

template <typename output_t, typename input_t>
__global__ __launch_bounds__(kNumThreads) void softmax_backward_block_kernel(
    int64_t row_offset, int64_t dim_size, const output_t* grad_output, const output_t* output, input_t* grad_input)
{
    using accscalar_t = acc_type<output_t, true>;
    using BlockReduce = cub::BlockReduce<accscalar_t, kNumThreads>;
    __shared__ typename BlockReduce::TempStorage temp_storage;
    __shared__ accscalar_t row_dot;

    const auto row = row_offset + static_cast<int64_t>(blockIdx.x);
    const auto base_offset = row * dim_size;
    accscalar_t thread_dot = 0;
    for (int64_t dim_index = threadIdx.x; dim_index < dim_size; dim_index += blockDim.x)
    {
        const auto offset = base_offset + dim_index;
        thread_dot += static_cast<accscalar_t>(grad_output[offset]) * static_cast<accscalar_t>(output[offset]);
    }

    const auto block_dot = BlockReduce(temp_storage).Sum(thread_dot);
    if (!threadIdx.x)
    {
        row_dot = block_dot;
    }
    __syncthreads();

    for (int64_t dim_index = threadIdx.x; dim_index < dim_size; dim_index += blockDim.x)
    {
        const auto offset = base_offset + dim_index;
        grad_input[offset] = static_cast<input_t>(static_cast<accscalar_t>(output[offset]) *
                                                  (static_cast<accscalar_t>(grad_output[offset]) - row_dot));
    }
}

template <typename scalar_t>
__global__ void log_softmax_forward_kernel(int64_t row_offset,
                                           int64_t row_count,
                                           ReductionLayout<2> layout,
                                           int64_t output_stride,
                                           const scalar_t* input,
                                           scalar_t* output)
{
    using acc_t = acc_type<scalar_t, true>;
    using BlockReduce = cub::BlockReduce<acc_t, kNumThreads>;
    __shared__ typename BlockReduce::TempStorage storage;
    __shared__ acc_t row_max;
    __shared__ acc_t row_log_sum;

    for (int64_t row = blockIdx.x; row < row_count; row += gridDim.x)
    {
        const auto offsets = layout.offsets.get(row_offset + row);
        acc_t maximum = -std::numeric_limits<acc_t>::infinity();
        for (int64_t col = threadIdx.x; col < layout.reduction_size; col += blockDim.x)
            maximum =
                SoftmaxMaxNan<acc_t>()(maximum, static_cast<acc_t>(input[offsets[0] + col * layout.reduction_stride]));
        const auto maximum_sum = BlockReduce(storage).Reduce(maximum, SoftmaxMaxNan<acc_t>());
        if (!threadIdx.x)
            row_max = maximum_sum;
        __syncthreads();

        acc_t sum = 0;
        for (int64_t col = threadIdx.x; col < layout.reduction_size; col += blockDim.x)
            sum += ::cuda::std::exp(static_cast<acc_t>(input[offsets[0] + col * layout.reduction_stride]) - row_max);
        const auto block_sum = BlockReduce(storage).Sum(sum);
        if (!threadIdx.x)
            row_log_sum = ::cuda::std::log(block_sum);
        __syncthreads();

        for (int64_t col = threadIdx.x; col < layout.reduction_size; col += blockDim.x)
            output[offsets[1] + col * output_stride] = static_cast<scalar_t>(
                (static_cast<acc_t>(input[offsets[0] + col * layout.reduction_stride]) - row_max) - row_log_sum);
        __syncthreads();
    }
}

template <typename output_t, typename input_t>
__global__ void log_softmax_backward_kernel(int64_t row_offset,
                                            int64_t row_count,
                                            ReductionLayout<3> layout,
                                            int64_t output_stride,
                                            int64_t grad_input_stride,
                                            const output_t* grad_output,
                                            const output_t* output,
                                            input_t* grad_input)
{
    using acc_t = acc_type<output_t, true>;
    using BlockReduce = cub::BlockReduce<acc_t, kNumThreads>;
    __shared__ typename BlockReduce::TempStorage storage;
    __shared__ acc_t row_sum;

    for (int64_t row = blockIdx.x; row < row_count; row += gridDim.x)
    {
        const auto offsets = layout.offsets.get(row_offset + row);
        acc_t sum = 0;
        for (int64_t col = threadIdx.x; col < layout.reduction_size; col += blockDim.x)
            sum += static_cast<acc_t>(grad_output[offsets[0] + col * layout.reduction_stride]);
        const auto block_sum = BlockReduce(storage).Sum(sum);
        if (!threadIdx.x)
            row_sum = block_sum;
        __syncthreads();

        for (int64_t col = threadIdx.x; col < layout.reduction_size; col += blockDim.x)
        {
            const auto grad = static_cast<acc_t>(grad_output[offsets[0] + col * layout.reduction_stride]);
            const auto value = static_cast<acc_t>(output[offsets[1] + col * output_stride]);
            grad_input[offsets[2] + col * grad_input_stride] =
                static_cast<input_t>(grad - ::cuda::std::exp(value) * row_sum);
        }
        __syncthreads();
    }
}

} // namespace

Tensor& _log_softmax_out(const Tensor& self, int64_t dim, bool half_to_float, Tensor& out)
{
    dim = at::maybe_wrap_dim(dim, self.dim());
    TORCH_CHECK(!half_to_float || self.scalar_type() == kHalf, "half_to_float requires a float16 input");
    TORCH_CHECK(
        out.scalar_type() == (half_to_float ? kFloat : self.scalar_type()), "Unexpected log_softmax output dtype");
    at::native::resize_output(out, self.sizes());
    at::assert_no_internal_overlap(out);
    at::assert_no_partial_overlap(out, self);
    if (!self.numel())
        return out;

    auto input = half_to_float ? self.to(kFloat) : self;
    auto output = out;
    if (!self.dim())
    {
        input = input.view({ 1 });
        output = output.view({ 1 });
    }
    const auto layout =
        make_reduction_layout<2>(input.sizes(), dim, std::array<IntArrayRef, 2>{ input.strides(), output.strides() });
    AT_DISPATCH_V2(input.scalar_type(), "log_softmax_out", AT_WRAP([&] {
                       parallel_for_each([&](int device, cudaStream_t stream) {
                           const auto [offset, count] = chunk(layout.num_segments, device);
                           if (count)
                           {
                               log_softmax_forward_kernel<<<static_cast<unsigned int>(std::min<int64_t>(count, 65535)),
                                                            kNumThreads, 0, stream>>>(
                                   offset, count, layout, output.stride(dim), input.const_data_ptr<scalar_t>(),
                                   output.data_ptr<scalar_t>());
                               C10_CUDA_KERNEL_LAUNCH_CHECK();
                           }
                       });
                   }),
                   AT_EXPAND(AT_FLOATING_TYPES), kHalf, kBFloat16);
    return out;
}

Tensor& _log_softmax_backward_data_out(
    const Tensor& grad_output, const Tensor& output, int64_t dim, ScalarType input_dtype, Tensor& grad_input)
{
    dim = at::maybe_wrap_dim(dim, output.dim());
    TORCH_CHECK(grad_output.sizes() == output.sizes(), "Expected grad_output and output to have the same shape");
    TORCH_CHECK(
        grad_output.scalar_type() == output.scalar_type(), "Expected grad_output and output to have the same dtype");
    TORCH_CHECK(grad_input.scalar_type() == input_dtype, "Expected grad_input to have the input dtype");
    at::native::resize_output(grad_input, output.sizes());
    at::assert_no_internal_overlap(grad_input);
    at::assert_no_partial_overlap(grad_input, grad_output);
    at::assert_no_partial_overlap(grad_input, output);
    if (!output.numel())
        return grad_input;

    auto grad = output.dim() ? grad_output : grad_output.view({ 1 });
    auto values = output.dim() ? output : output.view({ 1 });
    auto result = output.dim() ? grad_input : grad_input.view({ 1 });
    const auto layout = make_reduction_layout<3>(
        values.sizes(), dim, std::array<IntArrayRef, 3>{ grad.strides(), values.strides(), result.strides() });
    AT_DISPATCH_V2(
        output.scalar_type(), "log_softmax_backward_out", AT_WRAP([&] {
            using output_t = scalar_t;
            AT_DISPATCH_V2(
                input_dtype, "log_softmax_backward_input", AT_WRAP([&] {
                    parallel_for_each([&](int device, cudaStream_t stream) {
                        const auto [offset, count] = chunk(layout.num_segments, device);
                        if (count)
                        {
                            log_softmax_backward_kernel<<<static_cast<unsigned int>(std::min<int64_t>(count, 65535)),
                                                          kNumThreads, 0, stream>>>(
                                offset, count, layout, values.stride(dim), result.stride(dim),
                                grad.const_data_ptr<output_t>(), values.const_data_ptr<output_t>(),
                                result.data_ptr<scalar_t>());
                            C10_CUDA_KERNEL_LAUNCH_CHECK();
                        }
                    });
                }),
                AT_EXPAND(AT_FLOATING_TYPES), kHalf, kBFloat16);
        }),
        AT_EXPAND(AT_FLOATING_TYPES), kHalf, kBFloat16);
    return grad_input;
}

Tensor& _softmax_out(const at::Tensor& self, int64_t dim, bool half_to_float, at::Tensor& out)
{
    TORCH_CHECK(self.is_contiguous());
    TORCH_CHECK(out.is_contiguous());

    auto in = half_to_float ? self.to(out.scalar_type()) : self;

    if (dim < 0)
    {
        dim += in.dim();
    }
    TORCH_CHECK(dim >= 0 && dim < in.dim(), "Dimension out of range");
    if (!out.numel())
    {
        return out;
    }

    int64_t outer = 1;
    for (int64_t i = 0; i < dim; ++i)
    {
        outer *= in.size(i);
    }
    const int64_t dim_size = in.size(dim);
    int64_t inner = 1;
    for (int64_t i = dim + 1; i < in.dim(); ++i)
    {
        inner *= in.size(i);
    }

    AT_DISPATCH_V2(out.scalar_type(), "_softmax_out", AT_WRAP([&] {
                       if (inner == 1)
                       {
                           launch_softmax_forward(
                               outer, dim_size, in.const_data_ptr<scalar_t>(), out.data_ptr<scalar_t>());
                       }
                       else
                       {
                           mgpu_launch_func(inner, outer * inner, SoftmaxForwardFunctor<scalar_t>(), dim_size, inner,
                                            in.const_data_ptr<scalar_t>(), out.data_ptr<scalar_t>());
                       }
                   }),
                   AT_EXPAND(AT_FLOATING_TYPES));

    return out;
}

Tensor _safe_softmax(const Tensor& self, int64_t dim, std::optional<ScalarType> dtype)
{
    dim = at::maybe_wrap_dim(dim, self.dim());
    if (!self.is_contiguous() || (self.dim() && dim != self.dim() - 1) ||
        (self.scalar_type() != kFloat && self.scalar_type() != kDouble) ||
        (dtype.has_value() && *dtype != self.scalar_type()))
    {
        // Call the composite directly to avoid redispatch. Its DGX kernels need
        // contiguous input and a last-dimension mask reduction; keep dtype conversion
        // in the composite because it can change which values are -inf.
        if (self.dim())
        {
            auto input = self.movedim(dim, -1).contiguous();
            return at::native::_safe_softmax(input, -1, dtype).movedim(-1, dim).contiguous();
        }
        return at::native::_safe_softmax(self, dim, dtype);
    }

    auto out = at::empty(self.sizes(), self.options());
    if (!self.numel())
        return out;

    const auto dim_size = self.dim() ? self.size(dim) : int64_t(1);
    const auto rows = self.numel() / dim_size;
    AT_DISPATCH_V2(self.scalar_type(), "_safe_softmax", AT_WRAP([&] {
                       launch_softmax_forward<scalar_t, true>(
                           rows, dim_size, self.const_data_ptr<scalar_t>(), out.data_ptr<scalar_t>());
                   }),
                   AT_EXPAND(AT_FLOATING_TYPES));
    return out;
}

Tensor& _softmax_backward_data_out(
    const Tensor& grad_output, const Tensor& output, int64_t dim, ScalarType input_dtype, Tensor& grad_input)
{
    TORCH_CHECK(grad_output.sizes() == output.sizes(), "Expected grad_output and output to have the same shape");
    TORCH_CHECK(
        grad_output.scalar_type() == output.scalar_type(), "Expected grad_output and output to have the same dtype");
    TORCH_CHECK(grad_input.scalar_type() == input_dtype, "Expected grad_input to have the input dtype");
    at::native::resize_output(grad_input, output.sizes());

    const auto grad_output_contiguous = grad_output.contiguous();
    const auto output_contiguous = output.contiguous();
    auto grad_input_contiguous =
        grad_input.is_contiguous() ? grad_input : at::empty(output.sizes(), grad_input.options());

    if (dim < 0)
    {
        dim += output.dim();
    }
    TORCH_CHECK(dim >= 0 && dim < output.dim(), "Dimension out of range");
    if (!grad_input.numel())
    {
        return grad_input;
    }

    int64_t outer = 1;
    for (int64_t i = 0; i < dim; ++i)
    {
        outer *= output.size(i);
    }
    const int64_t dim_size = output.size(dim);
    int64_t inner = 1;
    for (int64_t i = dim + 1; i < output.dim(); ++i)
    {
        inner *= output.size(i);
    }

    AT_DISPATCH_V2(output.scalar_type(), "_softmax_backward_data_out", AT_WRAP([&] {
                       using output_t = scalar_t;
                       AT_DISPATCH_V2(
                           input_dtype, "_softmax_backward_data_out_input", AT_WRAP([&] {
                               using input_t = scalar_t;

                               if (inner == 1)
                               {
                                   parallel_for_each([&](int device, cudaStream_t stream) {
                                       int64_t row_offset, row_count;
                                       std::tie(row_offset, row_count) = chunk(outer, device);
                                       if (row_count)
                                       {
                                           const auto num_blocks = static_cast<unsigned int>(row_count);
                                           softmax_backward_block_kernel<<<num_blocks, kNumThreads, 0, stream>>>(
                                               row_offset, dim_size, grad_output_contiguous.const_data_ptr<output_t>(),
                                               output_contiguous.const_data_ptr<output_t>(),
                                               grad_input_contiguous.data_ptr<input_t>());
                                           C10_CUDA_KERNEL_LAUNCH_CHECK();
                                       }
                                   });
                               }
                               else
                               {
                                   mgpu_launch_func(inner, outer * inner, SoftmaxBackwardFunctor<output_t, input_t>(),
                                                    dim_size, inner, grad_output_contiguous.const_data_ptr<output_t>(),
                                                    output_contiguous.const_data_ptr<output_t>(),
                                                    grad_input_contiguous.data_ptr<input_t>());
                               }
                           }),
                           AT_EXPAND(AT_FLOATING_TYPES), kHalf, kBFloat16);
                   }),
                   AT_EXPAND(AT_FLOATING_TYPES));
    if (!grad_input.is_contiguous())
        grad_input.copy_(grad_input_contiguous);
    return grad_input;
}

TORCH_LIBRARY_IMPL(aten, PrivateUse1, m)
{
    m.impl(TORCH_SELECTIVE_NAME("aten::_log_softmax.out"), TORCH_FN(_log_softmax_out));
    m.impl(TORCH_SELECTIVE_NAME("aten::_log_softmax_backward_data.out"), TORCH_FN(_log_softmax_backward_data_out));
    m.impl(TORCH_SELECTIVE_NAME("aten::_safe_softmax"), TORCH_FN(_safe_softmax));
    m.impl(TORCH_SELECTIVE_NAME("aten::_softmax.out"), TORCH_FN(_softmax_out));
    m.impl(TORCH_SELECTIVE_NAME("aten::_softmax_backward_data.out"), TORCH_FN(_softmax_backward_data_out));
}

} // namespace dgx

} // namespace at::native
