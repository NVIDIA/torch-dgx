/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
#include "KernelLaunchUtils.cuh"
#include "StridedTensorUtils.cuh"

#include <c10/cuda/CUDAMathCompat.h>

#include <ATen/Dispatch_v2.h>
#include <ATen/OpMathType.h>
#include <ATen/TensorIterator.h>
#include <ATen/ops/clamp_min.h>

#include <torch/library.h>

namespace at::native
{

namespace dgx
{

namespace
{

template <class T>
struct ClampMinFunctor
{
    __device__ void operator()(size_t tid, const T* self, T min, T* out)
    {
        out[tid] = std::max(self[tid], min);
    }
};

template <class T>
struct ClampMaxFunctor
{
    __device__ void operator()(size_t tid, const T* self, T max, T* out)
    {
        out[tid] = std::min(self[tid], max);
    }
};

template <class T>
struct ClampFunctor
{
    __device__ void operator()(size_t tid, const T* self, T min, T max, T* out)
    {
        out[tid] = std::min(std::max(self[tid], min), max);
    }

    __device__ void operator()(size_t tid, const T* self, T* out)
    {
        out[tid] = self[tid];
    }
};

template <class T>
struct ThresholdBackwardFunctor
{
    __device__ T threshold_backward(T grad_output, T value, T threshold) const
    {
        return value > threshold ? grad_output : T(0);
    }

    __device__ void operator()(size_t tid, const T* grad_output, const T* self, T threshold, T* grad_input)
    {
        grad_input[tid] = threshold_backward(grad_output[tid], self[tid], threshold);
    }

    __device__ void operator()(
        size_t tid, const T* grad_output, const T* self, T threshold, T* grad_input, StridedOffsetCalculator<3> offsets)
    {
        const auto tensor_offsets = offsets.get(tid);
        grad_input[tensor_offsets[2]] =
            threshold_backward(grad_output[tensor_offsets[0]], self[tensor_offsets[1]], threshold);
    }
};

template <class T>
struct SiluFunctor
{
    __device__ T silu(T value) const
    {
        using opmath_t = at::opmath_type<T>;
        const auto x = static_cast<opmath_t>(value);
        return static_cast<T>(x / (opmath_t(1) + ::cuda::std::exp(-x)));
    }

    __device__ void operator()(size_t tid, const T* self, T* out)
    {
        out[tid] = silu(self[tid]);
    }

    __device__ void operator()(size_t tid, const T* self, T* out, StridedOffsetCalculator<2> offsets)
    {
        const auto tensor_offsets = offsets.get(tid);
        out[tensor_offsets[1]] = silu(self[tensor_offsets[0]]);
    }
};

template <class T>
struct GeluFunctor
{
    using opmath_t = at::opmath_type<T>;

    static constexpr opmath_t kBeta = opmath_t(0.7978845608028654);
    static constexpr opmath_t kKappa = opmath_t(0.044715);
    static constexpr opmath_t kInvSqrt2 = opmath_t(0.7071067811865476);

    bool tanh_approximation;

    __device__ T gelu(T value) const
    {
        const auto x = static_cast<opmath_t>(value);
        if (tanh_approximation)
        {
            const auto x_cube = x * x * x;
            const auto inner = kBeta * (x + kKappa * x_cube);
            return static_cast<T>(opmath_t(0.5) * x * (opmath_t(1) + c10::cuda::compat::tanh(inner)));
        }
        return static_cast<T>(x * opmath_t(0.5) * (opmath_t(1) + ::erf(x * kInvSqrt2)));
    }

    __device__ void operator()(size_t tid, const T* self, T* out)
    {
        out[tid] = gelu(self[tid]);
    }

    __device__ void operator()(size_t tid, const T* self, T* out, StridedOffsetCalculator<2> offsets)
    {
        const auto tensor_offsets = offsets.get(tid);
        out[tensor_offsets[1]] = gelu(self[tensor_offsets[0]]);
    }
};

template <class T>
struct GeluBackwardFunctor
{
    using opmath_t = at::opmath_type<T>;

    static constexpr opmath_t kBeta = opmath_t(0.7978845608028654);
    static constexpr opmath_t kKappa = opmath_t(0.044715);
    static constexpr opmath_t kInvSqrt2 = opmath_t(0.7071067811865476);
    static constexpr opmath_t kInvSqrt2Pi = opmath_t(0.3989422804014327);

    bool tanh_approximation;

    __device__ opmath_t derivative(T value) const
    {
        const auto x = static_cast<opmath_t>(value);
        if (tanh_approximation)
        {
            const auto x2 = x * x;
            const auto tanh_inner = c10::cuda::compat::tanh(kBeta * (x + kKappa * x * x2));
            return opmath_t(0.5) * (opmath_t(1) + tanh_inner) + opmath_t(0.5) * x *
                                                                    (opmath_t(1) - tanh_inner * tanh_inner) * kBeta *
                                                                    (opmath_t(1) + opmath_t(3) * kKappa * x2);
        }
        const auto cdf = opmath_t(0.5) * (opmath_t(1) + ::erf(x * kInvSqrt2));
        const auto pdf = c10::cuda::compat::exp(opmath_t(-0.5) * x * x) * kInvSqrt2Pi;
        return cdf + x * pdf;
    }

    __device__ void operator()(size_t tid, const T* grad_output, const T* self, T* grad_input)
    {
        grad_input[tid] = static_cast<T>(static_cast<opmath_t>(grad_output[tid]) * derivative(self[tid]));
    }

    __device__ void operator()(
        size_t tid, const T* grad_output, const T* self, T* grad_input, StridedOffsetCalculator<3> offsets)
    {
        const auto tensor_offsets = offsets.get(tid);
        grad_input[tensor_offsets[2]] =
            static_cast<T>(static_cast<opmath_t>(grad_output[tensor_offsets[0]]) * derivative(self[tensor_offsets[1]]));
    }
};

} // namespace

Tensor& clamp_min_out(const Tensor& self, const Scalar& min, Tensor& out)
{
    TORCH_CHECK(self.is_contiguous());
    TORCH_CHECK(out.is_contiguous());

    auto iter = TensorIterator::unary_op(out, self);
    AT_DISPATCH_V2(iter.common_dtype(), "clamp_min_out", AT_WRAP([&] {
                       using opmath_t = at::opmath_type<scalar_t>;
                       mgpu_launch_func(iter.numel(), ClampMinFunctor<opmath_t>(),
                                        static_cast<opmath_t*>(iter.data_ptr(1)), min.to<opmath_t>(),
                                        static_cast<opmath_t*>(iter.data_ptr(0)));
                   }),
                   AT_EXPAND(AT_ALL_TYPES));

    return out;
}

Tensor& clamp_max_out(const Tensor& self, const Scalar& max, Tensor& out)
{
    TORCH_CHECK(self.is_contiguous());
    TORCH_CHECK(out.is_contiguous());

    auto iter = TensorIterator::unary_op(out, self);
    AT_DISPATCH_V2(iter.common_dtype(), "clamp_max_out", AT_WRAP([&] {
                       using opmath_t = at::opmath_type<scalar_t>;
                       mgpu_launch_func(iter.numel(), ClampMaxFunctor<opmath_t>(),
                                        static_cast<opmath_t*>(iter.data_ptr(1)), max.to<opmath_t>(),
                                        static_cast<opmath_t*>(iter.data_ptr(0)));
                   }),
                   AT_EXPAND(AT_ALL_TYPES));

    return out;
}

Tensor& clamp_out(const Tensor& self, const ::std::optional<Scalar>& min, const ::std::optional<Scalar>& max, Tensor& out)
{
    TORCH_CHECK(self.is_contiguous());
    TORCH_CHECK(out.is_contiguous());

    auto iter = TensorIterator::unary_op(out, self);
    AT_DISPATCH_V2(
        iter.common_dtype(), "clamp_out", AT_WRAP([&] {
            using opmath_t = at::opmath_type<scalar_t>;
            if (min && max)
            {
                mgpu_launch_func(iter.numel(), ClampFunctor<opmath_t>(), static_cast<opmath_t*>(iter.data_ptr(1)),
                                 (*min).to<opmath_t>(), (*max).to<opmath_t>(), static_cast<opmath_t*>(iter.data_ptr(0)));
            }
            else if (min)
            {
                mgpu_launch_func(iter.numel(), ClampMinFunctor<opmath_t>(), static_cast<opmath_t*>(iter.data_ptr(1)),
                                 (*min).to<opmath_t>(), static_cast<opmath_t*>(iter.data_ptr(0)));
            }
            else if (max)
            {
                mgpu_launch_func(iter.numel(), ClampMaxFunctor<opmath_t>(), static_cast<opmath_t*>(iter.data_ptr(1)),
                                 (*max).to<opmath_t>(), static_cast<opmath_t*>(iter.data_ptr(0)));
            }
            else
            {
                mgpu_launch_func(iter.numel(), ClampFunctor<opmath_t>(), static_cast<opmath_t*>(iter.data_ptr(1)),
                                 static_cast<opmath_t*>(iter.data_ptr(0)));
            }
        }),
        AT_EXPAND(AT_ALL_TYPES));

    return out;
}

Tensor relu(const Tensor& self)
{
    TORCH_CHECK(self.scalar_type() != at::kBool, "Boolean inputs not supported for relu");
    return at::clamp_min(self, 0);
}

Tensor& relu_(Tensor& self)
{
    TORCH_CHECK(self.scalar_type() != at::kBool, "Boolean inputs not supported for relu");
    TORCH_CHECK(self.is_contiguous());

    AT_DISPATCH_V2(self.scalar_type(), "relu_", AT_WRAP([&] {
                       mgpu_launch_func(self.numel(), ClampMinFunctor<scalar_t>(), self.data_ptr<scalar_t>(),
                                        scalar_t(0), self.data_ptr<scalar_t>());
                   }),
                   AT_EXPAND(AT_ALL_TYPES));

    return self;
}

Tensor& threshold_backward_out(const Tensor& grad_output, const Tensor& self, const Scalar& threshold, Tensor& grad_input)
{
    TORCH_CHECK(grad_output.sizes() == self.sizes(), "Expected grad_output and self to have the same shape");
    TORCH_CHECK(grad_output.scalar_type() == self.scalar_type() && grad_input.scalar_type() == self.scalar_type(),
                "Expected grad_output, self, and grad_input to have the same dtype");
    auto iter = TensorIterator::binary_op(grad_input, grad_output, self);

    AT_DISPATCH_V2(
        iter.common_dtype(), "threshold_backward_out", AT_WRAP([&] {
            if (grad_output.is_contiguous() && self.is_contiguous() && grad_input.is_contiguous())
            {
                mgpu_launch_func(iter.numel(), ThresholdBackwardFunctor<scalar_t>(),
                                 static_cast<scalar_t*>(iter.data_ptr(1)), static_cast<scalar_t*>(iter.data_ptr(2)),
                                 threshold.to<scalar_t>(), static_cast<scalar_t*>(iter.data_ptr(0)));
            }
            else
            {
                const auto offsets = make_strided_offset_calculator<3>(
                    self.sizes(),
                    std::array<IntArrayRef, 3>{ grad_output.strides(), self.strides(), grad_input.strides() });
                mgpu_launch_func(iter.numel(), ThresholdBackwardFunctor<scalar_t>(),
                                 grad_output.const_data_ptr<scalar_t>(), self.const_data_ptr<scalar_t>(),
                                 threshold.to<scalar_t>(), grad_input.data_ptr<scalar_t>(), offsets);
            }
        }),
        AT_EXPAND(AT_ALL_TYPES), kHalf, kBFloat16);
    return grad_input;
}

Tensor& silu_out(const Tensor& self, Tensor& out)
{
    TORCH_CHECK(out.scalar_type() == self.scalar_type(), "Expected out and self to have the same dtype");
    auto iter = TensorIterator::unary_op(out, self);
    AT_DISPATCH_V2(iter.common_dtype(), "silu_out", AT_WRAP([&] {
                       if (self.is_contiguous() && out.is_contiguous())
                       {
                           mgpu_launch_func(iter.numel(), SiluFunctor<scalar_t>(),
                                            static_cast<scalar_t*>(iter.data_ptr(1)),
                                            static_cast<scalar_t*>(iter.data_ptr(0)));
                       }
                       else
                       {
                           const auto offsets = make_strided_offset_calculator<2>(
                               self.sizes(), std::array<IntArrayRef, 2>{ self.strides(), out.strides() });
                           mgpu_launch_func(iter.numel(), SiluFunctor<scalar_t>(), self.const_data_ptr<scalar_t>(),
                                            out.data_ptr<scalar_t>(), offsets);
                       }
                   }),
                   AT_EXPAND(AT_FLOATING_TYPES), kHalf, kBFloat16);
    return out;
}

Tensor& gelu_out(const Tensor& self, c10::string_view approximate, Tensor& out)
{
    TORCH_CHECK(approximate == "none" || approximate == "tanh", "approximate argument must be either none or tanh");
    TORCH_CHECK(out.scalar_type() == self.scalar_type(), "Expected out and self to have the same dtype");
    auto iter = TensorIterator::unary_op(out, self);
    const bool tanh_approximation = approximate == "tanh";
    AT_DISPATCH_V2(iter.common_dtype(), "gelu_out", AT_WRAP([&] {
                       const GeluFunctor<scalar_t> functor{ tanh_approximation };
                       if (self.is_contiguous() && out.is_contiguous())
                       {
                           mgpu_launch_func(iter.numel(), functor, static_cast<scalar_t*>(iter.data_ptr(1)),
                                            static_cast<scalar_t*>(iter.data_ptr(0)));
                       }
                       else
                       {
                           const auto offsets = make_strided_offset_calculator<2>(
                               self.sizes(), std::array<IntArrayRef, 2>{ self.strides(), out.strides() });
                           mgpu_launch_func(iter.numel(), functor, self.const_data_ptr<scalar_t>(),
                                            out.data_ptr<scalar_t>(), offsets);
                       }
                   }),
                   AT_EXPAND(AT_FLOATING_TYPES), kHalf, kBFloat16);
    return out;
}

Tensor& gelu_backward_out(const Tensor& grad_output, const Tensor& self, c10::string_view approximate, Tensor& grad_input)
{
    TORCH_CHECK(approximate == "none" || approximate == "tanh", "approximate argument must be either none or tanh");
    TORCH_CHECK(grad_output.sizes() == self.sizes(), "Expected grad_output and self to have the same shape");
    TORCH_CHECK(grad_output.scalar_type() == self.scalar_type() && grad_input.scalar_type() == self.scalar_type(),
                "Expected grad_output, self, and grad_input to have the same dtype");
    auto iter = TensorIterator::binary_op(grad_input, grad_output, self);
    const bool tanh_approximation = approximate == "tanh";
    AT_DISPATCH_V2(
        iter.common_dtype(), "gelu_backward_out", AT_WRAP([&] {
            const GeluBackwardFunctor<scalar_t> functor{ tanh_approximation };
            if (grad_output.is_contiguous() && self.is_contiguous() && grad_input.is_contiguous())
            {
                mgpu_launch_func(iter.numel(), functor, static_cast<scalar_t*>(iter.data_ptr(1)),
                                 static_cast<scalar_t*>(iter.data_ptr(2)), static_cast<scalar_t*>(iter.data_ptr(0)));
            }
            else
            {
                const auto offsets = make_strided_offset_calculator<3>(
                    self.sizes(),
                    std::array<IntArrayRef, 3>{ grad_output.strides(), self.strides(), grad_input.strides() });
                mgpu_launch_func(iter.numel(), functor, grad_output.const_data_ptr<scalar_t>(),
                                 self.const_data_ptr<scalar_t>(), grad_input.data_ptr<scalar_t>(), offsets);
            }
        }),
        AT_EXPAND(AT_FLOATING_TYPES), kHalf, kBFloat16);
    return grad_input;
}

TORCH_LIBRARY_IMPL(aten, PrivateUse1, m)
{
    m.impl(TORCH_SELECTIVE_NAME("aten::clamp_min.out"), TORCH_FN(clamp_min_out));
    m.impl(TORCH_SELECTIVE_NAME("aten::clamp_max.out"), TORCH_FN(clamp_max_out));
    m.impl(TORCH_SELECTIVE_NAME("aten::clamp.out"), TORCH_FN(clamp_out));
    m.impl(TORCH_SELECTIVE_NAME("aten::relu"), TORCH_FN(relu));
    m.impl(TORCH_SELECTIVE_NAME("aten::relu_"), TORCH_FN(relu_));
    m.impl(TORCH_SELECTIVE_NAME("aten::threshold_backward.grad_input"), TORCH_FN(threshold_backward_out));
    m.impl(TORCH_SELECTIVE_NAME("aten::silu.out"), TORCH_FN(silu_out));
    m.impl(TORCH_SELECTIVE_NAME("aten::gelu.out"), TORCH_FN(gelu_out));
    m.impl(TORCH_SELECTIVE_NAME("aten::gelu_backward.grad_input"), TORCH_FN(gelu_backward_out));
}

} // namespace dgx

} // namespace at::native
