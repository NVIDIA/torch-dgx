/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
#include "KernelLaunchUtils.cuh"
#include "StridedTensorUtils.cuh"

#include <c10/cuda/CUDAMathCompat.h>

#include <ATen/Dispatch_v2.h>
#include <ATen/OpMathType.h>
#include <ATen/native/Resize.h>
#include <ATen/ops/empty.h>

#include <torch/library.h>

namespace at::native
{

namespace dgx
{

namespace
{

template <typename scalar_t>
struct TanhFunctor
{
    __device__ void operator()(int64_t index, const scalar_t* input, scalar_t* output, StridedOffsetCalculator<2> offsets)
    {
        const auto offset = offsets.get(index);
        using opmath_t = at::opmath_type<scalar_t>;
        output[offset[1]] = static_cast<scalar_t>(::cuda::std::tanh(static_cast<opmath_t>(input[offset[0]])));
    }
};

template <typename scalar_t>
struct TanhBackwardFunctor
{
    __device__ void operator()(int64_t index,
                               const scalar_t* grad_output,
                               const scalar_t* output,
                               scalar_t* grad_input,
                               StridedOffsetCalculator<3> offsets)
    {
        const auto offset = offsets.get(index);
        using opmath_t = at::opmath_type<scalar_t>;
        const auto value = static_cast<opmath_t>(output[offset[1]]);
        grad_input[offset[2]] =
            static_cast<scalar_t>(static_cast<opmath_t>(grad_output[offset[0]]) * (opmath_t(1) - value * value));
    }
};

template <class T>
struct AbsFunctor
{
    __device__ void operator()(size_t tid, const T* __restrict__ self, T* __restrict__ result)
    {
        result[tid] = ::cuda::std::abs(self[tid]);
    }

    __device__ void operator()(size_t tid, const T* self, T* result, StridedOffsetCalculator<2> offsets)
    {
        const auto tensor_offsets = offsets.get(tid);
        result[tensor_offsets[1]] = ::cuda::std::abs(self[tensor_offsets[0]]);
    }
};

template <class T>
struct BitwiseNotFunctor
{
    __device__ void operator()(size_t tid, const T* __restrict__ self, T* __restrict__ result)
    {
        result[tid] = ~self[tid];
    }
};

template <>
struct BitwiseNotFunctor<bool>
{
    __device__ void operator()(size_t tid, const bool* __restrict__ self, bool* __restrict__ result)
    {
        result[tid] = !self[tid];
    }
};

template <class T>
struct CeilFunctor
{
    __device__ void operator()(size_t tid, const T* __restrict__ self, T* __restrict__ result)
    {
        result[tid] = ::cuda::std::ceil(self[tid]);
    }
};

template <class T>
struct CosFunctor
{
    __device__ void operator()(size_t tid, const T* __restrict__ self, T* __restrict__ result)
    {
        result[tid] = ::cuda::std::cos(self[tid]);
    }
};

template <class T>
struct ExpFunctor
{
    __device__ void operator()(size_t tid, const T* __restrict__ self, T* __restrict__ result)
    {
        result[tid] = ::cuda::std::exp(self[tid]);
    }
};

template <class T>
struct FloorFunctor
{
    __device__ void operator()(size_t tid, const T* __restrict__ self, T* __restrict__ result)
    {
        result[tid] = ::cuda::std::floor(self[tid]);
    }
};

template <class T>
struct IsNaNFunctor
{
    __device__ void operator()(size_t tid, const T* __restrict__ self, bool* __restrict__ result)
    {
        result[tid] = ::cuda::std::isnan(self[tid]);
    }

    __device__ void operator()(size_t tid, const T* self, bool* result, StridedOffsetCalculator<2> offsets)
    {
        const auto tensor_offsets = offsets.get(tid);
        result[tensor_offsets[1]] = ::cuda::std::isnan(self[tensor_offsets[0]]);
    }
};

template <class T>
struct LogFunctor
{
    __device__ void operator()(size_t tid, const T* __restrict__ self, T* __restrict__ result)
    {
        result[tid] = ::cuda::std::log(self[tid]);
    }
};

template <class T>
struct Log10Functor
{
    __device__ void operator()(size_t tid, const T* __restrict__ self, T* __restrict__ result)
    {
        result[tid] = ::cuda::std::log10(self[tid]);
    }
};

template <class T>
struct LogitFunctor
{
    __device__ void operator()(size_t tid, const T* __restrict__ self, T* __restrict__ result)
    {
        result[tid] = ::cuda::std::log(self[tid] / (T(1.) - self[tid]));
    }

    __device__ void operator()(size_t tid, const T* __restrict__ self, double eps, T* __restrict__ result)
    {
        const T lo = static_cast<T>(eps);
        const T hi = T(1.) - lo;
        const T z = ::cuda::std::min(::cuda::std::max(self[tid], lo), hi);
        result[tid] = ::cuda::std::log(z / (T(1.) - z));
    }
};

template <class T>
struct NegFunctor
{
    __device__ void operator()(size_t tid, const T* __restrict__ self, T* __restrict__ result)
    {
        result[tid] = -self[tid];
    }

    __device__ void operator()(size_t tid, const T* self, T* result, StridedOffsetCalculator<2> offsets)
    {
        const auto tensor_offsets = offsets.get(tid);
        result[tensor_offsets[1]] = -self[tensor_offsets[0]];
    }
};

template <class T>
struct ReciprocalFunctor
{
    __device__ void operator()(size_t tid, const T* __restrict__ self, T* __restrict__ result)
    {
        result[tid] = T(1.) / self[tid];
    }
};

template <class T>
struct RoundFunctor
{
    __device__ void operator()(size_t tid, const T* __restrict__ self, T* __restrict__ result)
    {
        // PyTorch rounds halfway cases to the nearest even integer.
        result[tid] = ::cuda::std::nearbyint(self[tid]);
    }
};

template <class T>
struct RsqrtFunctor
{
    __device__ void operator()(size_t tid, const T* __restrict__ self, T* __restrict__ result)
    {
        result[tid] = c10::cuda::compat::rsqrt(self[tid]);
    }
};

template <class T>
struct SigmoidFunctor
{
    __device__ void operator()(size_t tid, const T* __restrict__ self, T* __restrict__ result)
    {
        const T half = T(0.5);
        result[tid] = half * ::cuda::std::tanh(half * self[tid]) + half;
    }
};

template <class T>
struct SigmoidBackwardFunctor
{
    __device__ void operator()(size_t tid,
                               const T* __restrict__ grad_output,
                               const T* __restrict__ output,
                               T* __restrict__ grad_input)
    {
        grad_input[tid] = grad_output[tid] * (T(1.) - output[tid]) * output[tid];
    }
};

template <class T>
struct SignFunctor
{
    __device__ void operator()(size_t tid, const T* __restrict__ self, T* __restrict__ result)
    {
        const T x = self[tid];
        result[tid] = ::cuda::std::copysign(T(x != T(0)), x);
    }
};

template <class T>
struct SinFunctor
{
    __device__ void operator()(size_t tid, const T* __restrict__ self, T* __restrict__ result)
    {
        result[tid] = ::cuda::std::sin(self[tid]);
    }
};

template <class T>
struct SgnFunctor
{
    __device__ void operator()(size_t tid, const T* __restrict__ self, T* __restrict__ result)
    {
        auto z = self[tid];
        auto squared_magnitude = z.real() * z.real() + z.imag() * z.imag();
        result[tid] = squared_magnitude ? z * c10::cuda::compat::rsqrt(squared_magnitude) : T(0.);
    }
};

template <class T>
struct SqrtFunctor
{
    __device__ void operator()(size_t tid, const T* __restrict__ self, T* __restrict__ result)
    {
        result[tid] = ::cuda::std::sqrt(self[tid]);
    }
};

} // namespace

Tensor& abs_out(const Tensor& self, Tensor& result)
{
    at::native::resize_output(result, self.sizes());
    AT_DISPATCH_V2(self.scalar_type(), "abs_out", AT_WRAP([&] {
                       if (self.is_contiguous())
                       {
                           mgpu_launch_func(self.numel(), AbsFunctor<scalar_t>(), self.const_data_ptr<scalar_t>(),
                                            result.data_ptr<scalar_t>());
                       }
                       else
                       {
                           const auto offsets = make_strided_offset_calculator<2>(
                               self.sizes(), std::array<IntArrayRef, 2>{ self.strides(), result.strides() });
                           mgpu_launch_func(self.numel(), AbsFunctor<scalar_t>(), self.const_data_ptr<scalar_t>(),
                                            result.data_ptr<scalar_t>(), offsets);
                       }
                   }),
                   AT_EXPAND(AT_FLOATING_TYPES));

    return result;
}

Tensor& bitwise_not_out(const Tensor& self, Tensor& result)
{
    TORCH_CHECK(self.is_contiguous());

    at::native::resize_output(result, self.sizes());
    AT_DISPATCH_V2(self.scalar_type(), "bitwise_not_out", AT_WRAP([&] {
                       mgpu_launch_func(self.numel(), BitwiseNotFunctor<scalar_t>(), self.const_data_ptr<scalar_t>(),
                                        result.data_ptr<scalar_t>());
                   }),
                   AT_EXPAND(AT_INTEGRAL_TYPES), kBool);

    return result;
}

Tensor& ceil_out(const Tensor& self, Tensor& result)
{
    TORCH_CHECK(self.is_contiguous());

    at::native::resize_output(result, self.sizes());
    AT_DISPATCH_V2(self.scalar_type(), "ceil_out", AT_WRAP([&] {
                       mgpu_launch_func(self.numel(), CeilFunctor<scalar_t>(), self.const_data_ptr<scalar_t>(),
                                        result.data_ptr<scalar_t>());
                   }),
                   AT_EXPAND(AT_FLOATING_TYPES));

    return result;
}

Tensor& cos_out(const Tensor& self, Tensor& result)
{
    TORCH_CHECK(self.is_contiguous());

    at::native::resize_output(result, self.sizes());
    AT_DISPATCH_V2(self.scalar_type(), "cos_out", AT_WRAP([&] {
                       mgpu_launch_func(self.numel(), CosFunctor<scalar_t>(), self.const_data_ptr<scalar_t>(),
                                        result.data_ptr<scalar_t>());
                   }),
                   AT_EXPAND(AT_FLOATING_TYPES));

    return result;
}

Tensor& exp_out(const Tensor& self, Tensor& result)
{
    TORCH_CHECK(self.is_contiguous());

    at::native::resize_output(result, self.sizes());
    AT_DISPATCH_V2(self.scalar_type(), "exp_out", AT_WRAP([&] {
                       mgpu_launch_func(self.numel(), ExpFunctor<scalar_t>(), self.const_data_ptr<scalar_t>(),
                                        result.data_ptr<scalar_t>());
                   }),
                   AT_EXPAND(AT_FLOATING_TYPES));

    return result;
}

Tensor& floor_out(const Tensor& self, Tensor& result)
{
    TORCH_CHECK(self.is_contiguous());

    at::native::resize_output(result, self.sizes());
    AT_DISPATCH_V2(self.scalar_type(), "floor_out", AT_WRAP([&] {
                       mgpu_launch_func(self.numel(), FloorFunctor<scalar_t>(), self.const_data_ptr<scalar_t>(),
                                        result.data_ptr<scalar_t>());
                   }),
                   AT_EXPAND(AT_FLOATING_TYPES));

    return result;
}

Tensor isnan(const Tensor& self)
{
    Tensor result = at::empty(self.sizes(), self.options().dtype(kBool));

    if (self.is_contiguous())
    {
        AT_DISPATCH_V2(self.scalar_type(), "isnan", AT_WRAP([&] {
                           mgpu_launch_func(self.numel(), IsNaNFunctor<scalar_t>(), self.const_data_ptr<scalar_t>(),
                                            result.data_ptr<bool>());
                       }),
                       AT_EXPAND(AT_FLOATING_TYPES));
    }
    else
    {
        const auto offsets = make_strided_offset_calculator<2>(
            self.sizes(), std::array<IntArrayRef, 2>{ self.strides(), result.strides() });
        AT_DISPATCH_V2(self.scalar_type(), "isnan", AT_WRAP([&] {
                           mgpu_launch_func(self.numel(), IsNaNFunctor<scalar_t>(), self.const_data_ptr<scalar_t>(),
                                            result.data_ptr<bool>(), offsets);
                       }),
                       AT_EXPAND(AT_FLOATING_TYPES));
    }

    return result;
}

Tensor& log_out(const Tensor& self, Tensor& result)
{
    TORCH_CHECK(self.is_contiguous());

    at::native::resize_output(result, self.sizes());
    AT_DISPATCH_V2(self.scalar_type(), "log_out", AT_WRAP([&] {
                       mgpu_launch_func(self.numel(), LogFunctor<scalar_t>(), self.const_data_ptr<scalar_t>(),
                                        result.data_ptr<scalar_t>());
                   }),
                   AT_EXPAND(AT_FLOATING_TYPES));

    return result;
}

Tensor& log10_out(const Tensor& self, Tensor& result)
{
    TORCH_CHECK(self.is_contiguous());

    at::native::resize_output(result, self.sizes());
    AT_DISPATCH_V2(self.scalar_type(), "log10_out", AT_WRAP([&] {
                       mgpu_launch_func(self.numel(), Log10Functor<scalar_t>(), self.const_data_ptr<scalar_t>(),
                                        result.data_ptr<scalar_t>());
                   }),
                   AT_EXPAND(AT_FLOATING_TYPES));

    return result;
}

Tensor logit(const Tensor& self, std::optional<double> eps)
{
    TORCH_CHECK(self.is_contiguous());

    Tensor result = at::empty_like(self);
    AT_DISPATCH_V2(self.scalar_type(), "logit", AT_WRAP([&] {
                       if (eps)
                       {
                           mgpu_launch_func(self.numel(), LogitFunctor<scalar_t>(), self.const_data_ptr<scalar_t>(),
                                            eps.value(), result.data_ptr<scalar_t>());
                       }
                       else
                       {
                           mgpu_launch_func(self.numel(), LogitFunctor<scalar_t>(), self.const_data_ptr<scalar_t>(),
                                            result.data_ptr<scalar_t>());
                       }
                   }),
                   AT_EXPAND(AT_FLOATING_TYPES));

    return result;
}

Tensor& neg_out(const Tensor& self, Tensor& result)
{
    at::native::resize_output(result, self.sizes());
    AT_DISPATCH_V2(self.scalar_type(), "neg_out", AT_WRAP([&] {
                       if (self.is_contiguous() && result.is_contiguous())
                       {
                           mgpu_launch_func(self.numel(), NegFunctor<scalar_t>(), self.const_data_ptr<scalar_t>(),
                                            result.data_ptr<scalar_t>());
                       }
                       else
                       {
                           const auto offsets = make_strided_offset_calculator<2>(
                               self.sizes(), std::array<IntArrayRef, 2>{ self.strides(), result.strides() });
                           mgpu_launch_func(self.numel(), NegFunctor<scalar_t>(), self.const_data_ptr<scalar_t>(),
                                            result.data_ptr<scalar_t>(), offsets);
                       }
                   }),
                   AT_EXPAND(AT_ALL_TYPES));

    return result;
}

Tensor& reciprocal_out(const Tensor& self, Tensor& result)
{
    TORCH_CHECK(self.is_contiguous());

    at::native::resize_output(result, self.sizes());
    AT_DISPATCH_V2(self.scalar_type(), "reciprocal_out", AT_WRAP([&] {
                       mgpu_launch_func(self.numel(), ReciprocalFunctor<scalar_t>(), self.const_data_ptr<scalar_t>(),
                                        result.data_ptr<scalar_t>());
                   }),
                   AT_EXPAND(AT_FLOATING_TYPES));

    return result;
}

Tensor& round_out(const Tensor& self, Tensor& result)
{
    TORCH_CHECK(self.is_contiguous());

    at::native::resize_output(result, self.sizes());
    AT_DISPATCH_V2(self.scalar_type(), "round_out", AT_WRAP([&] {
                       mgpu_launch_func(self.numel(), RoundFunctor<scalar_t>(), self.const_data_ptr<scalar_t>(),
                                        result.data_ptr<scalar_t>());
                   }),
                   AT_EXPAND(AT_FLOATING_TYPES));

    return result;
}

Tensor& rsqrt_out(const Tensor& self, Tensor& result)
{
    TORCH_CHECK(self.is_contiguous());

    at::native::resize_output(result, self.sizes());
    AT_DISPATCH_V2(self.scalar_type(), "rsqrt_out", AT_WRAP([&] {
                       mgpu_launch_func(self.numel(), RsqrtFunctor<scalar_t>(), self.const_data_ptr<scalar_t>(),
                                        result.data_ptr<scalar_t>());
                   }),
                   AT_EXPAND(AT_FLOATING_TYPES));

    return result;
}

Tensor& sigmoid_out(const Tensor& self, Tensor& result)
{
    TORCH_CHECK(self.is_contiguous());

    at::native::resize_output(result, self.sizes());
    AT_DISPATCH_V2(self.scalar_type(), "sigmoid_out", AT_WRAP([&] {
                       mgpu_launch_func(self.numel(), SigmoidFunctor<scalar_t>(), self.const_data_ptr<scalar_t>(),
                                        result.data_ptr<scalar_t>());
                   }),
                   AT_EXPAND(AT_FLOATING_TYPES));

    return result;
}

Tensor& sigmoid_backward_out(const Tensor& grad_output, const Tensor& output, Tensor& grad_input)
{
    TORCH_CHECK(grad_output.is_contiguous());
    TORCH_CHECK(output.is_contiguous());

    at::native::resize_output(grad_input, grad_output.sizes());
    AT_DISPATCH_V2(grad_output.scalar_type(), "sigmoid_backward_out", AT_WRAP([&] {
                       mgpu_launch_func(grad_output.numel(), SigmoidBackwardFunctor<scalar_t>(),
                                        grad_output.const_data_ptr<scalar_t>(), output.const_data_ptr<scalar_t>(),
                                        grad_input.data_ptr<scalar_t>());
                   }),
                   AT_EXPAND(AT_FLOATING_TYPES));

    return grad_input;
}

Tensor& sin_out(const Tensor& self, Tensor& result)
{
    TORCH_CHECK(self.is_contiguous());

    at::native::resize_output(result, self.sizes());
    AT_DISPATCH_V2(self.scalar_type(), "sin_out", AT_WRAP([&] {
                       mgpu_launch_func(self.numel(), SinFunctor<scalar_t>(), self.const_data_ptr<scalar_t>(),
                                        result.data_ptr<scalar_t>());
                   }),
                   AT_EXPAND(AT_FLOATING_TYPES));

    return result;
}

Tensor& sign_out(const Tensor& self, Tensor& result)
{
    TORCH_CHECK(self.is_contiguous());

    at::native::resize_output(result, self.sizes());
    AT_DISPATCH_V2(self.scalar_type(), "sign_out", AT_WRAP([&] {
                       mgpu_launch_func(self.numel(), SignFunctor<scalar_t>(), self.const_data_ptr<scalar_t>(),
                                        result.data_ptr<scalar_t>());
                   }),
                   AT_EXPAND(AT_FLOATING_TYPES));

    return result;
}

Tensor& sgn_out(const Tensor& self, Tensor& result)
{
    TORCH_CHECK(self.is_contiguous());

    if (isComplexType(self.scalar_type()))
    {
        at::native::resize_output(result, self.sizes());
        AT_DISPATCH_V2(self.scalar_type(), "sgn_out", AT_WRAP([&] {
                           mgpu_launch_func(self.numel(), SgnFunctor<scalar_t>(), self.const_data_ptr<scalar_t>(),
                                            result.data_ptr<scalar_t>());
                       }),
                       AT_EXPAND(AT_COMPLEX_TYPES));
        return result;
    }
    else
    {
        return sign_out(self, result);
    }
}

Tensor& sqrt_out(const Tensor& self, Tensor& result)
{
    TORCH_CHECK(self.is_contiguous());

    at::native::resize_output(result, self.sizes());
    AT_DISPATCH_V2(self.scalar_type(), "sqrt_out", AT_WRAP([&] {
                       mgpu_launch_func(self.numel(), SqrtFunctor<scalar_t>(), self.const_data_ptr<scalar_t>(),
                                        result.data_ptr<scalar_t>());
                   }),
                   AT_EXPAND(AT_FLOATING_TYPES));

    return result;
}

Tensor& tanh_out(const Tensor& self, Tensor& out)
{
    auto iter = TensorIterator::unary_float_op(out, self);
    const Tensor input = iter.input(0).to(iter.common_dtype());
    const Tensor output = iter.output().scalar_type() == iter.common_dtype() ?
                              iter.output() :
                              at::empty(iter.output().sizes(), iter.output().options().dtype(iter.common_dtype()));
    const auto offsets = make_strided_offset_calculator<2>(
        output.sizes(), std::array<IntArrayRef, 2>{ input.strides(), output.strides() });
    AT_DISPATCH_V2(iter.common_dtype(), "tanh_out", AT_WRAP([&] {
                       mgpu_launch_func(iter.numel(), TanhFunctor<scalar_t>(), input.const_data_ptr<scalar_t>(),
                                        output.data_ptr<scalar_t>(), offsets);
                   }),
                   AT_EXPAND(AT_FLOATING_TYPES), kHalf, kBFloat16);
    if (!output.is_same(iter.output()))
        iter.output().copy_(output);
    iter.cast_outputs();
    return out;
}

Tensor& tanh_backward_out(const Tensor& grad_output, const Tensor& output, Tensor& grad_input)
{
    TORCH_CHECK(grad_output.sizes() == output.sizes(), "Expected grad_output and output to have the same shape");
    auto iter = TensorIterator::binary_op(grad_input, grad_output, output);
    const Tensor grad = iter.input(0).to(iter.common_dtype());
    const Tensor values = iter.input(1).to(iter.common_dtype());
    const Tensor result = iter.output().scalar_type() == iter.common_dtype() ?
                              iter.output() :
                              at::empty(iter.output().sizes(), iter.output().options().dtype(iter.common_dtype()));
    const auto offsets = make_strided_offset_calculator<3>(
        result.sizes(), std::array<IntArrayRef, 3>{ grad.strides(), values.strides(), result.strides() });
    AT_DISPATCH_V2(iter.common_dtype(), "tanh_backward_out", AT_WRAP([&] {
                       mgpu_launch_func(iter.numel(), TanhBackwardFunctor<scalar_t>(), grad.const_data_ptr<scalar_t>(),
                                        values.const_data_ptr<scalar_t>(), result.data_ptr<scalar_t>(), offsets);
                   }),
                   AT_EXPAND(AT_FLOATING_TYPES), kHalf, kBFloat16);
    if (!result.is_same(iter.output()))
        iter.output().copy_(result);
    iter.cast_outputs();
    return grad_input;
}

TORCH_LIBRARY_IMPL(aten, PrivateUse1, m)
{
    m.impl(TORCH_SELECTIVE_NAME("aten::tanh.out"), TORCH_FN(tanh_out));
    m.impl(TORCH_SELECTIVE_NAME("aten::tanh_backward.grad_input"), TORCH_FN(tanh_backward_out));
    m.impl(TORCH_SELECTIVE_NAME("aten::abs.out"), TORCH_FN(abs_out));
    m.impl(TORCH_SELECTIVE_NAME("aten::bitwise_not.out"), TORCH_FN(bitwise_not_out));
    m.impl(TORCH_SELECTIVE_NAME("aten::ceil.out"), TORCH_FN(ceil_out));
    m.impl(TORCH_SELECTIVE_NAME("aten::cos.out"), TORCH_FN(cos_out));
    m.impl(TORCH_SELECTIVE_NAME("aten::exp.out"), TORCH_FN(exp_out));
    m.impl(TORCH_SELECTIVE_NAME("aten::floor.out"), TORCH_FN(floor_out));
    m.impl(TORCH_SELECTIVE_NAME("aten::isnan"), TORCH_FN(isnan));
    m.impl(TORCH_SELECTIVE_NAME("aten::log.out"), TORCH_FN(log_out));
    m.impl(TORCH_SELECTIVE_NAME("aten::log10.out"), TORCH_FN(log10_out));
    m.impl(TORCH_SELECTIVE_NAME("aten::logit"), TORCH_FN(logit));
    m.impl(TORCH_SELECTIVE_NAME("aten::neg.out"), TORCH_FN(neg_out));
    m.impl(TORCH_SELECTIVE_NAME("aten::reciprocal.out"), TORCH_FN(reciprocal_out));
    m.impl(TORCH_SELECTIVE_NAME("aten::round.out"), TORCH_FN(round_out));
    m.impl(TORCH_SELECTIVE_NAME("aten::rsqrt.out"), TORCH_FN(rsqrt_out));
    m.impl(TORCH_SELECTIVE_NAME("aten::sgn.out"), TORCH_FN(sgn_out));
    m.impl(TORCH_SELECTIVE_NAME("aten::sigmoid.out"), TORCH_FN(sigmoid_out));
    m.impl(TORCH_SELECTIVE_NAME("aten::sigmoid_backward.grad_input"), TORCH_FN(sigmoid_backward_out));
    m.impl(TORCH_SELECTIVE_NAME("aten::sign.out"), TORCH_FN(sign_out));
    m.impl(TORCH_SELECTIVE_NAME("aten::sin.out"), TORCH_FN(sin_out));
    m.impl(TORCH_SELECTIVE_NAME("aten::sqrt.out"), TORCH_FN(sqrt_out));
}

} // namespace dgx

} // namespace at::native
