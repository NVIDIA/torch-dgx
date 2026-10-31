/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
#include "KernelLaunchUtils.cuh"
#include "StridedTensorUtils.cuh"

#include <c10/util/TypeSafeSignMath.h>

#include <ATen/Dispatch_v2.h>
#include <ATen/OpMathType.h>
#include <ATen/TensorIterator.h>
#include <ATen/ops/empty.h>

#include <torch/library.h>

#include <cub/util_type.cuh>
#include <cuda/std/functional>

#include <limits>
#include <type_traits>

namespace at::native
{

namespace dgx
{

namespace
{

using complex_float = c10::complex<float>;

__device__ complex_float maybe_conj(complex_float value, bool conjugated)
{
    return conjugated ? complex_float(value.real(), -value.imag()) : value;
}

template <typename T>
__device__ T strided_value(const T* data, int64_t offset)
{
    return data[offset];
}

template <typename T>
__device__ T strided_value(T scalar, int64_t offset)
{
    return scalar;
}

struct PolarFunctor
{
    __device__ complex_float polar(float magnitude, float angle) const
    {
        return complex_float(magnitude * ::cuda::std::cos(angle), magnitude * ::cuda::std::sin(angle));
    }

    __device__ void operator()(size_t tid, const float* magnitude, const float* angle, complex_float* out)
    {
        out[tid] = polar(magnitude[tid], angle[tid]);
    }

    __device__ void operator()(
        size_t tid, const float* magnitude, const float* angle, complex_float* out, StridedOffsetCalculator<3> offsets)
    {
        const auto tensor_offsets = offsets.get(tid);
        out[tensor_offsets[2]] = polar(magnitude[tensor_offsets[0]], angle[tensor_offsets[1]]);
    }
};

template <typename T, typename OpT>
struct BinaryScaleOpFunctor
{
    __device__ void operator()(size_t tid, T self, const T* __restrict__ other, T alpha, T* __restrict__ out)
    {
        out[tid] = op(self, alpha * other[tid]);
    }

    __device__ void operator()(
        size_t tid, cub::FutureValue<const T> self, const T* __restrict__ other, T alpha, T* __restrict__ out)
    {
        out[tid] = op(self, alpha * other[tid]);
    }

    __device__ void operator()(size_t tid, const T* __restrict__ self, T other, T alpha, T* __restrict__ out)
    {
        out[tid] = op(self[tid], alpha * other);
    }

    __device__ void operator()(
        size_t tid, const T* __restrict__ self, cub::FutureValue<const T> other, T alpha, T* __restrict__ out)
    {
        out[tid] = op(self[tid], alpha * other);
    }

    __device__ void operator()(size_t tid, T self, T other, T alpha, T* __restrict__ out)
    {
        out[tid] = op(self, alpha * other);
    }

    __device__ void operator()(size_t tid, T self, cub::FutureValue<const T> other, T alpha, T* __restrict__ out)
    {
        out[tid] = op(self, alpha * other);
    }

    __device__ void operator()(size_t tid, cub::FutureValue<const T> self, T other, T alpha, T* __restrict__ out)
    {
        out[tid] = op(self, alpha * other);
    }

    __device__ void operator()(
        size_t tid, cub::FutureValue<const T> self, cub::FutureValue<const T> other, T alpha, T* __restrict__ out)
    {
        out[tid] = op(self, alpha * other);
    }

    __device__ void operator()(
        size_t tid, const T* __restrict__ self, const T* __restrict__ other, T alpha, T* __restrict__ out)
    {
        out[tid] = op(self[tid], alpha * other[tid]);
    }

    __device__ void operator()(size_t tid,
                               const void* self_ptr,
                               const ScalarType self_type,
                               size_t self_elem_size,
                               T other,
                               T alpha,
                               void* out_ptr,
                               const ScalarType out_type,
                               size_t out_elem_size)
    {
        auto self = c10::fetch_and_cast<T>(self_type, static_cast<const char*>(self_ptr) + self_elem_size * tid);
        auto out = op(self, alpha * other);
        c10::cast_and_store<T>(out_type, static_cast<char*>(out_ptr) + out_elem_size * tid, out);
    }

    __device__ void operator()(size_t tid,
                               const void* self_ptr,
                               const ScalarType self_type,
                               size_t self_elem_size,
                               const void* other_ptr,
                               const ScalarType other_type,
                               size_t other_elem_size,
                               T alpha,
                               void* out_ptr,
                               const ScalarType out_type,
                               size_t out_elem_size)
    {
        auto self = c10::fetch_and_cast<T>(self_type, static_cast<const char*>(self_ptr) + self_elem_size * tid);
        auto other = c10::fetch_and_cast<T>(other_type, static_cast<const char*>(other_ptr) + other_elem_size * tid);
        auto out = op(self, alpha * other);
        c10::cast_and_store<T>(out_type, static_cast<char*>(out_ptr) + out_elem_size * tid, out);
    }

    template <typename Self, typename Other>
    __device__ void operator()(size_t tid, Self self, Other other, T alpha, T* out, StridedOffsetCalculator<3> offsets)
    {
        const auto tensor_offsets = offsets.get(tid);
        out[tensor_offsets[2]] =
            op(strided_value(self, tensor_offsets[0]), alpha * strided_value(other, tensor_offsets[1]));
    }

    OpT op{};
};

template <typename T>
using AddFunctor = BinaryScaleOpFunctor<T, ::cuda::std::plus<void>>;

template <typename T>
using SubFunctor = BinaryScaleOpFunctor<T, ::cuda::std::minus<void>>;

template <typename T, typename OpT>
struct BinaryOpFunctor
{
    __device__ void operator()(size_t tid, T self, const T* __restrict__ other, T* __restrict__ out)
    {
        out[tid] = op(self, other[tid]);
    }

    __device__ void operator()(size_t tid, cub::FutureValue<const T> self, const T* __restrict__ other, T* __restrict__ out)
    {
        out[tid] = op(self, other[tid]);
    }

    __device__ void operator()(size_t tid, const T* __restrict__ self, T other, T* __restrict__ out)
    {
        out[tid] = op(self[tid], other);
    }

    __device__ void operator()(size_t tid, const T* __restrict__ self, cub::FutureValue<const T> other, T* __restrict__ out)
    {
        out[tid] = op(self[tid], other);
    }

    __device__ void operator()(size_t tid, T self, T other, T* __restrict__ out)
    {
        out[tid] = op(self, other);
    }

    __device__ void operator()(size_t tid, T self, cub::FutureValue<const T> other, T* __restrict__ out)
    {
        out[tid] = op(self, other);
    }

    __device__ void operator()(size_t tid, cub::FutureValue<const T> self, T other, T* __restrict__ out)
    {
        out[tid] = op(self, other);
    }

    __device__ void operator()(size_t tid,
                               cub::FutureValue<const T> self,
                               cub::FutureValue<const T> other,
                               T* __restrict__ out)
    {
        out[tid] = op(self, other);
    }

    __device__ void operator()(size_t tid, const T* __restrict__ self, const T* __restrict__ other, T* __restrict__ out)
    {
        out[tid] = op(self[tid], other[tid]);
    }

    __device__ void operator()(size_t tid,
                               const void* self_ptr,
                               const ScalarType self_type,
                               size_t self_elem_size,
                               T other,
                               void* out_ptr,
                               const ScalarType out_type,
                               size_t out_elem_size)
    {
        auto self = c10::fetch_and_cast<T>(self_type, static_cast<const char*>(self_ptr) + self_elem_size * tid);
        auto out = op(self, other);
        c10::cast_and_store<T>(out_type, static_cast<char*>(out_ptr) + out_elem_size * tid, out);
    }

    __device__ void operator()(size_t tid,
                               const void* self_ptr,
                               const ScalarType self_type,
                               size_t self_elem_size,
                               const void* other_ptr,
                               const ScalarType other_type,
                               size_t other_elem_size,
                               void* out_ptr,
                               const ScalarType out_type,
                               size_t out_elem_size)
    {
        auto self = c10::fetch_and_cast<T>(self_type, static_cast<const char*>(self_ptr) + self_elem_size * tid);
        auto other = c10::fetch_and_cast<T>(other_type, static_cast<const char*>(other_ptr) + other_elem_size * tid);
        auto out = op(self, other);
        c10::cast_and_store<T>(out_type, static_cast<char*>(out_ptr) + out_elem_size * tid, out);
    }

    template <typename Self, typename Other>
    __device__ void operator()(size_t tid, Self self, Other other, T* out, StridedOffsetCalculator<3> offsets)
    {
        const auto tensor_offsets = offsets.get(tid);
        out[tensor_offsets[2]] = op(strided_value(self, tensor_offsets[0]), strided_value(other, tensor_offsets[1]));
    }

    OpT op{};
};

template <typename T>
struct MulOp
{
    bool self_conjugated = false;
    bool other_conjugated = false;

    __device__ T operator()(T self, T other) const
    {
        if constexpr (std::is_same_v<T, complex_float>)
        {
            return ::cuda::std::multiplies<void>()(
                maybe_conj(self, self_conjugated), maybe_conj(other, other_conjugated));
        }
        else
        {
            return ::cuda::std::multiplies<void>()(self, other);
        }
    }
};

template <typename T>
using MulFunctor = BinaryOpFunctor<T, MulOp<T>>;

template <typename T>
using DivFunctor = BinaryOpFunctor<T, ::cuda::std::divides<void>>;

template <typename T>
struct HypotOp
{
    __device__ T operator()(T self, T other) const
    {
        using opmath_t = at::opmath_type<T>;
        return static_cast<T>(::cuda::std::hypot(static_cast<opmath_t>(self), static_cast<opmath_t>(other)));
    }
};

template <typename T>
using HypotFunctor = BinaryOpFunctor<T, HypotOp<T>>;

template <typename T>
struct RemainderOp
{
    __device__ T operator()(T self, T other) const
    {
        if constexpr (std::is_integral_v<T>)
        {
            if constexpr (std::is_same_v<T, uint8_t>)
            {
                if (other == 0)
                {
                    return std::numeric_limits<uint8_t>::max();
                }
            }

            auto remainder = self % other;
            if (remainder != 0 && c10::signs_differ(remainder, other))
            {
                remainder += other;
            }
            return remainder;
        }
        else
        {
            auto remainder = ::fmod(self, other);
            if (remainder != 0 && c10::signs_differ(remainder, other))
            {
                remainder += other;
            }
            return remainder;
        }
    }
};

template <typename T>
using RemainderFunctor = BinaryOpFunctor<T, RemainderOp<T>>;

template <typename T>
using LogicalAndFunctor = BinaryOpFunctor<T, ::cuda::std::logical_and<void>>;

template <typename T>
using LogicalOrFunctor = BinaryOpFunctor<T, ::cuda::std::logical_or<void>>;

template <typename T>
struct PowFunctor
{
    __device__ void operator()(size_t tid, const T* self, T exponent, T* out)
    {
        out[tid] = ::cuda::std::pow(self[tid], exponent);
    }

    __device__ void operator()(size_t tid, const T* self, T exponent, T* out, StridedOffsetCalculator<2> offsets)
    {
        const auto tensor_offsets = offsets.get(tid);
        out[tensor_offsets[1]] = ::cuda::std::pow(self[tensor_offsets[0]], exponent);
    }
};

} // namespace

Tensor& add_out(const Tensor& self, const Tensor& other, const Scalar& alpha, Tensor& out)
{
    bool all_contiguous = self.is_contiguous() && other.is_contiguous() && out.is_contiguous();

    auto iter = TensorIterator::borrowing_binary_op(out, self, other);
    bool all_same_dtype = (iter.common_dtype() == iter.dtype(0) || iter.is_cpu_scalar(0)) &&
                          (iter.common_dtype() == iter.dtype(1) || iter.is_cpu_scalar(1)) &&
                          (iter.common_dtype() == iter.dtype(2) || iter.is_cpu_scalar(2));

    if (all_same_dtype)
    {
        AT_DISPATCH_V2(
            iter.common_dtype(), "add_out", AT_WRAP([&] {
                using opmath_t = at::opmath_type<scalar_t>;

                AddFunctor<opmath_t> func;

                if (iter.is_scalar(1) && !iter.is_scalar(2) && all_contiguous)
                {
                    if (iter.is_cpu_scalar(1))
                    {
                        mgpu_launch_func(iter.numel(), func, iter.scalar_value<opmath_t>(1),
                                         static_cast<opmath_t*>(iter.data_ptr(2)), alpha.to<opmath_t>(),
                                         static_cast<opmath_t*>(iter.data_ptr(0)));
                    }
                    else
                    {
                        mgpu_launch_func(iter.numel(), func,
                                         cub::FutureValue<const opmath_t>(static_cast<opmath_t*>(iter.data_ptr(1))),
                                         static_cast<opmath_t*>(iter.data_ptr(2)), alpha.to<opmath_t>(),
                                         static_cast<opmath_t*>(iter.data_ptr(0)));
                    }
                }
                else if (!iter.is_scalar(1) && iter.is_scalar(2) && all_contiguous)
                {
                    if (iter.is_cpu_scalar(2))
                    {
                        mgpu_launch_func(iter.numel(), func, static_cast<opmath_t*>(iter.data_ptr(1)),
                                         iter.scalar_value<opmath_t>(2), alpha.to<opmath_t>(),
                                         static_cast<opmath_t*>(iter.data_ptr(0)));
                    }
                    else
                    {
                        mgpu_launch_func(iter.numel(), func, static_cast<opmath_t*>(iter.data_ptr(1)),
                                         cub::FutureValue<const opmath_t>(static_cast<opmath_t*>(iter.data_ptr(2))),
                                         alpha.to<opmath_t>(), static_cast<opmath_t*>(iter.data_ptr(0)));
                    }
                }
                else if (iter.is_scalar(1) && iter.is_scalar(2) && all_contiguous)
                {
                    if (iter.is_cpu_scalar(1) && iter.is_cpu_scalar(2))
                    {
                        mgpu_launch_func(iter.numel(), func, iter.scalar_value<opmath_t>(1),
                                         iter.scalar_value<opmath_t>(2), alpha.to<opmath_t>(),
                                         static_cast<opmath_t*>(iter.data_ptr(0)));
                    }
                    else if (iter.is_cpu_scalar(1) && !iter.is_cpu_scalar(2))
                    {
                        mgpu_launch_func(iter.numel(), func, iter.scalar_value<opmath_t>(1),
                                         cub::FutureValue<const opmath_t>(static_cast<opmath_t*>(iter.data_ptr(2))),
                                         alpha.to<opmath_t>(), static_cast<opmath_t*>(iter.data_ptr(0)));
                    }
                    else if (!iter.is_cpu_scalar(1) && iter.is_cpu_scalar(2))
                    {
                        mgpu_launch_func(iter.numel(), func,
                                         cub::FutureValue<const opmath_t>(static_cast<opmath_t*>(iter.data_ptr(1))),
                                         iter.scalar_value<opmath_t>(2), alpha.to<opmath_t>(),
                                         static_cast<opmath_t*>(iter.data_ptr(0)));
                    }
                    else
                    {
                        mgpu_launch_func(iter.numel(), func,
                                         cub::FutureValue<const opmath_t>(static_cast<opmath_t*>(iter.data_ptr(1))),
                                         cub::FutureValue<const opmath_t>(static_cast<opmath_t*>(iter.data_ptr(2))),
                                         alpha.to<opmath_t>(), static_cast<opmath_t*>(iter.data_ptr(0)));
                    }
                }
                else if (self.sizes() == out.sizes() && other.sizes() == out.sizes() &&
                         self.strides() == out.strides() && other.strides() == out.strides())
                {
                    mgpu_launch_func(iter.numel(), func, static_cast<opmath_t*>(iter.data_ptr(1)),
                                     static_cast<opmath_t*>(iter.data_ptr(2)), alpha.to<opmath_t>(),
                                     static_cast<opmath_t*>(iter.data_ptr(0)));
                }
                else
                {
                    const auto self_expanded = iter.input(0).expand(out.sizes());
                    const auto other_expanded = iter.input(1).expand(out.sizes());
                    // Use tensor strides in logical dimension order, not iter.strides().
                    const auto offsets = make_strided_offset_calculator<3>(
                        out.sizes(),
                        std::array<IntArrayRef, 3>{ self_expanded.strides(), other_expanded.strides(), out.strides() });
                    if (iter.is_cpu_scalar(1) && iter.is_cpu_scalar(2))
                    {
                        mgpu_launch_func(iter.numel(), func, iter.scalar_value<opmath_t>(1),
                                         iter.scalar_value<opmath_t>(2), alpha.to<opmath_t>(), out.data_ptr<opmath_t>(),
                                         offsets);
                    }
                    else if (iter.is_cpu_scalar(1))
                    {
                        mgpu_launch_func(iter.numel(), func, iter.scalar_value<opmath_t>(1),
                                         other_expanded.const_data_ptr<opmath_t>(), alpha.to<opmath_t>(),
                                         out.data_ptr<opmath_t>(), offsets);
                    }
                    else if (iter.is_cpu_scalar(2))
                    {
                        mgpu_launch_func(iter.numel(), func, self_expanded.const_data_ptr<opmath_t>(),
                                         iter.scalar_value<opmath_t>(2), alpha.to<opmath_t>(), out.data_ptr<opmath_t>(),
                                         offsets);
                    }
                    else
                    {
                        mgpu_launch_func(iter.numel(), func, self_expanded.const_data_ptr<opmath_t>(),
                                         other_expanded.const_data_ptr<opmath_t>(), alpha.to<opmath_t>(),
                                         out.data_ptr<opmath_t>(), offsets);
                    }
                }
            }),
            AT_EXPAND(AT_ALL_TYPES));
    }
    else
    {
        AT_DISPATCH_V2(iter.common_dtype(), "add_out", AT_WRAP([&] {
                           using opmath_t = at::opmath_type<scalar_t>;

                           AddFunctor<opmath_t> func;

                           if (!iter.is_scalar(1) && iter.is_scalar(2))
                           {
                               if (iter.is_cpu_scalar(2))
                               {
                                   mgpu_launch_func(iter.numel(), func, iter.data_ptr(1), iter.dtype(1),
                                                    c10::elementSize(iter.dtype(1)), iter.scalar_value<opmath_t>(2),
                                                    alpha.to<opmath_t>(), iter.data_ptr(0), iter.dtype(0),
                                                    c10::elementSize(iter.dtype(0)));
                               }
                               else
                               {
                                   // A zero byte stride broadcasts the scalar, loaded using its storage dtype.
                                   mgpu_launch_func(iter.numel(), func, iter.data_ptr(1), iter.dtype(1),
                                                    c10::elementSize(iter.dtype(1)), iter.data_ptr(2), iter.dtype(2),
                                                    size_t(0), alpha.to<opmath_t>(), iter.data_ptr(0), iter.dtype(0),
                                                    c10::elementSize(iter.dtype(0)));
                               }
                           }
                           else if (self.sizes() == out.sizes() && other.sizes() == out.sizes() &&
                                    self.strides() == out.strides() && other.strides() == out.strides())
                           {
                               mgpu_launch_func(iter.numel(), func, iter.data_ptr(1), iter.dtype(1),
                                                c10::elementSize(iter.dtype(1)), iter.data_ptr(2), iter.dtype(2),
                                                c10::elementSize(iter.dtype(2)), alpha.to<opmath_t>(), iter.data_ptr(0),
                                                iter.dtype(0), c10::elementSize(iter.dtype(0)));
                           }
                           else
                           {
                               auto self_expanded = iter.input(0).to(iter.common_dtype()).expand(out.sizes());
                               auto other_expanded = iter.input(1).to(iter.common_dtype()).expand(out.sizes());
                               // Preserve the original conversion buffers and logical traversal order.
                               const auto offsets = make_strided_offset_calculator<3>(
                                   out.sizes(), std::array<IntArrayRef, 3>{ self_expanded.strides(),
                                                                            other_expanded.strides(), out.strides() });
                               mgpu_launch_func(iter.numel(), func, self_expanded.const_data_ptr<opmath_t>(),
                                                other_expanded.const_data_ptr<opmath_t>(), alpha.to<opmath_t>(),
                                                out.data_ptr<opmath_t>(), offsets);
                           }
                       }),
                       AT_EXPAND(AT_ALL_TYPES));
    }

    return out;
}

Tensor& sub_out(const Tensor& self, const Tensor& other, const Scalar& alpha, Tensor& out)
{
    return add_out(self, other, -alpha, out);
}

template <template <typename T> typename BinaryOpFunctor>
Tensor& binary_func_out(
    const Tensor& self, const Tensor& other, Tensor& out, bool self_conjugated = false, bool other_conjugated = false)
{
    bool all_contiguous = self.is_contiguous() && other.is_contiguous() && out.is_contiguous();

    TensorIterator iter;
    // True division promotes integer inputs before computing, including when
    // reached through a foreach slow path. Casting an integer quotient to the
    // floating output afterward loses the fractional part.
    if constexpr (std::is_same_v<BinaryOpFunctor<float>, DivFunctor<float>>)
        iter.build_borrowing_binary_float_op(out, self, other);
    else
        iter.build_borrowing_binary_op(out, self, other);
    constexpr bool supports_complex = std::is_same_v<BinaryOpFunctor<complex_float>, MulFunctor<complex_float>>;
    TORCH_CHECK_NOT_IMPLEMENTED(iter.common_dtype() != kComplexFloat || supports_complex,
                                "binary_func_out currently supports ComplexFloat for multiplication only");
    bool all_same_dtype = (iter.common_dtype() == iter.dtype(0) || iter.is_cpu_scalar(0)) &&
                          (iter.common_dtype() == iter.dtype(1) || iter.is_cpu_scalar(1)) &&
                          (iter.common_dtype() == iter.dtype(2) || iter.is_cpu_scalar(2));

    if (all_same_dtype)
    {
        AT_DISPATCH_V2(
            iter.common_dtype(), "binary_func_out", AT_WRAP([&] {
                if constexpr (supports_complex || !std::is_same_v<scalar_t, complex_float>)
                {
                    using opmath_t = at::opmath_type<scalar_t>;

                    BinaryOpFunctor<opmath_t> func;
                    if constexpr (std::is_same_v<BinaryOpFunctor<opmath_t>, MulFunctor<opmath_t>>)
                    {
                        func.op.self_conjugated = self_conjugated;
                        func.op.other_conjugated = other_conjugated;
                    }

                    if (iter.is_scalar(1) && !iter.is_scalar(2) && all_contiguous)
                    {
                        if (iter.is_cpu_scalar(1))
                        {
                            mgpu_launch_func(iter.numel(), func, iter.scalar_value<opmath_t>(1),
                                             static_cast<opmath_t*>(iter.data_ptr(2)),
                                             static_cast<opmath_t*>(iter.data_ptr(0)));
                        }
                        else
                        {
                            mgpu_launch_func(iter.numel(), func,
                                             cub::FutureValue<const opmath_t>(static_cast<opmath_t*>(iter.data_ptr(1))),
                                             static_cast<opmath_t*>(iter.data_ptr(2)),
                                             static_cast<opmath_t*>(iter.data_ptr(0)));
                        }
                    }
                    else if (!iter.is_scalar(1) && iter.is_scalar(2) && all_contiguous)
                    {
                        if (iter.is_cpu_scalar(2))
                        {
                            mgpu_launch_func(iter.numel(), func, static_cast<opmath_t*>(iter.data_ptr(1)),
                                             iter.scalar_value<opmath_t>(2), static_cast<opmath_t*>(iter.data_ptr(0)));
                        }
                        else
                        {
                            mgpu_launch_func(iter.numel(), func, static_cast<opmath_t*>(iter.data_ptr(1)),
                                             cub::FutureValue<const opmath_t>(static_cast<opmath_t*>(iter.data_ptr(2))),
                                             static_cast<opmath_t*>(iter.data_ptr(0)));
                        }
                    }
                    else if (iter.is_scalar(1) && iter.is_scalar(2) && all_contiguous)
                    {
                        if (iter.is_cpu_scalar(1) && iter.is_cpu_scalar(2))
                        {
                            mgpu_launch_func(iter.numel(), func, iter.scalar_value<opmath_t>(1),
                                             iter.scalar_value<opmath_t>(2), static_cast<opmath_t*>(iter.data_ptr(0)));
                        }
                        else if (iter.is_cpu_scalar(1) && !iter.is_cpu_scalar(2))
                        {
                            mgpu_launch_func(iter.numel(), func, iter.scalar_value<opmath_t>(1),
                                             cub::FutureValue<const opmath_t>(static_cast<opmath_t*>(iter.data_ptr(2))),
                                             static_cast<opmath_t*>(iter.data_ptr(0)));
                        }
                        else if (!iter.is_cpu_scalar(1) && iter.is_cpu_scalar(2))
                        {
                            mgpu_launch_func(iter.numel(), func,
                                             cub::FutureValue<const opmath_t>(static_cast<opmath_t*>(iter.data_ptr(1))),
                                             iter.scalar_value<opmath_t>(2), static_cast<opmath_t*>(iter.data_ptr(0)));
                        }
                        else
                        {
                            mgpu_launch_func(iter.numel(), func,
                                             cub::FutureValue<const opmath_t>(static_cast<opmath_t*>(iter.data_ptr(1))),
                                             cub::FutureValue<const opmath_t>(static_cast<opmath_t*>(iter.data_ptr(2))),
                                             static_cast<opmath_t*>(iter.data_ptr(0)));
                        }
                    }
                    else if (self.sizes() == out.sizes() && other.sizes() == out.sizes() &&
                             self.strides() == out.strides() && other.strides() == out.strides())
                    {
                        mgpu_launch_func(iter.numel(), func, static_cast<opmath_t*>(iter.data_ptr(1)),
                                         static_cast<opmath_t*>(iter.data_ptr(2)),
                                         static_cast<opmath_t*>(iter.data_ptr(0)));
                    }
                    else
                    {
                        const auto self_expanded = iter.input(0).expand(out.sizes());
                        const auto other_expanded = iter.input(1).expand(out.sizes());
                        // Use tensor strides in logical dimension order, not iter.strides().
                        const auto offsets = make_strided_offset_calculator<3>(
                            out.sizes(), std::array<IntArrayRef, 3>{ self_expanded.strides(), other_expanded.strides(),
                                                                     out.strides() });
                        if (iter.is_cpu_scalar(1) && iter.is_cpu_scalar(2))
                        {
                            mgpu_launch_func(iter.numel(), func, iter.scalar_value<opmath_t>(1),
                                             iter.scalar_value<opmath_t>(2), out.data_ptr<opmath_t>(), offsets);
                        }
                        else if (iter.is_cpu_scalar(1))
                        {
                            mgpu_launch_func(iter.numel(), func, iter.scalar_value<opmath_t>(1),
                                             other_expanded.const_data_ptr<opmath_t>(), out.data_ptr<opmath_t>(),
                                             offsets);
                        }
                        else if (iter.is_cpu_scalar(2))
                        {
                            mgpu_launch_func(iter.numel(), func, self_expanded.const_data_ptr<opmath_t>(),
                                             iter.scalar_value<opmath_t>(2), out.data_ptr<opmath_t>(), offsets);
                        }
                        else
                        {
                            mgpu_launch_func(iter.numel(), func, self_expanded.const_data_ptr<opmath_t>(),
                                             other_expanded.const_data_ptr<opmath_t>(), out.data_ptr<opmath_t>(),
                                             offsets);
                        }
                    }
                }
            }),
            AT_EXPAND(AT_ALL_TYPES), kBool, kComplexFloat);
    }
    else
    {
        AT_DISPATCH_V2(
            iter.common_dtype(), "binary_func_out", AT_WRAP([&] {
                if constexpr (supports_complex || !std::is_same_v<scalar_t, complex_float>)
                {
                    using opmath_t = at::opmath_type<scalar_t>;

                    BinaryOpFunctor<opmath_t> func;
                    if constexpr (std::is_same_v<BinaryOpFunctor<opmath_t>, MulFunctor<opmath_t>>)
                    {
                        func.op.self_conjugated = self_conjugated;
                        func.op.other_conjugated = other_conjugated;
                    }

                    if (!iter.is_scalar(1) && iter.is_scalar(2))
                    {
                        if (iter.is_cpu_scalar(2))
                        {
                            mgpu_launch_func(iter.numel(), func, iter.data_ptr(1), iter.dtype(1),
                                             c10::elementSize(iter.dtype(1)), iter.scalar_value<opmath_t>(2),
                                             iter.data_ptr(0), iter.dtype(0), c10::elementSize(iter.dtype(0)));
                        }
                        else
                        {
                            // A zero byte stride broadcasts the scalar, loaded using its storage dtype.
                            mgpu_launch_func(iter.numel(), func, iter.data_ptr(1), iter.dtype(1),
                                             c10::elementSize(iter.dtype(1)), iter.data_ptr(2), iter.dtype(2), size_t(0),
                                             iter.data_ptr(0), iter.dtype(0), c10::elementSize(iter.dtype(0)));
                        }
                    }
                    else if (self.sizes() == out.sizes() && other.sizes() == out.sizes() &&
                             self.strides() == out.strides() && other.strides() == out.strides())
                    {
                        mgpu_launch_func(iter.numel(), func, iter.data_ptr(1), iter.dtype(1),
                                         c10::elementSize(iter.dtype(1)), iter.data_ptr(2), iter.dtype(2),
                                         c10::elementSize(iter.dtype(2)), iter.data_ptr(0), iter.dtype(0),
                                         c10::elementSize(iter.dtype(0)));
                    }
                    else
                    {
                        auto self_expanded = iter.input(0).to(iter.common_dtype()).expand(out.sizes());
                        auto other_expanded = iter.input(1).to(iter.common_dtype()).expand(out.sizes());
                        // Preserve the original conversion buffers and logical traversal order.
                        const auto offsets = make_strided_offset_calculator<3>(
                            out.sizes(), std::array<IntArrayRef, 3>{ self_expanded.strides(), other_expanded.strides(),
                                                                     out.strides() });
                        mgpu_launch_func(iter.numel(), func, self_expanded.const_data_ptr<opmath_t>(),
                                         other_expanded.const_data_ptr<opmath_t>(), out.data_ptr<opmath_t>(), offsets);
                    }
                }
            }),
            AT_EXPAND(AT_ALL_TYPES), kBool, kComplexFloat);
    }

    return out;
}

Tensor& mul_out(const Tensor& self, const Tensor& other, Tensor& out)
{
    TORCH_CHECK(!out.is_conj(), "DGX multiplication does not support a conjugated output tensor");
    return binary_func_out<MulFunctor>(self, other, out, self.is_conj(), other.is_conj());
}

Tensor& polar_out(const Tensor& magnitude, const Tensor& angle, Tensor& out)
{
    TORCH_CHECK(magnitude.scalar_type() == kFloat && angle.scalar_type() == kFloat,
                "DGX polar currently supports Float inputs only");
    TORCH_CHECK(out.scalar_type() == kComplexFloat, "Expected polar output to have dtype ComplexFloat");
    TORCH_CHECK(!out.is_conj(), "DGX polar does not support a conjugated output tensor");

    auto iter = TensorIterator::binary_op(out, magnitude, angle);
    const auto magnitude_expanded = magnitude.expand(out.sizes());
    const auto angle_expanded = angle.expand(out.sizes());

    if (magnitude_expanded.is_contiguous() && angle_expanded.is_contiguous() && out.is_contiguous())
    {
        mgpu_launch_func(iter.numel(), PolarFunctor(), static_cast<float*>(iter.data_ptr(1)),
                         static_cast<float*>(iter.data_ptr(2)), static_cast<complex_float*>(iter.data_ptr(0)));
    }
    else
    {
        const auto offsets = make_strided_offset_calculator<3>(
            out.sizes(),
            std::array<IntArrayRef, 3>{ magnitude_expanded.strides(), angle_expanded.strides(), out.strides() });
        mgpu_launch_func(iter.numel(), PolarFunctor(), magnitude_expanded.const_data_ptr<float>(),
                         angle_expanded.const_data_ptr<float>(), out.data_ptr<complex_float>(), offsets);
    }
    return out;
}

Tensor& div_out(const Tensor& self, const Tensor& other, Tensor& out)
{
    return binary_func_out<DivFunctor>(self, other, out);
}

Tensor& hypot_out(const Tensor& self, const Tensor& other, Tensor& out)
{
    auto iter = TensorIterator::binary_op(out, self, other);
    const Tensor lhs = iter.input(0).to(iter.common_dtype()).expand(out.sizes());
    const Tensor rhs = iter.input(1).to(iter.common_dtype()).expand(out.sizes());
    const Tensor result = iter.output().scalar_type() == iter.common_dtype() ?
                              iter.output() :
                              at::empty(out.sizes(), out.options().dtype(iter.common_dtype()));
    const auto offsets = make_strided_offset_calculator<3>(
        result.sizes(), std::array<IntArrayRef, 3>{ lhs.strides(), rhs.strides(), result.strides() });

    AT_DISPATCH_V2(iter.common_dtype(), "hypot_out", AT_WRAP([&] {
                       if (iter.is_cpu_scalar(1) && iter.is_cpu_scalar(2))
                       {
                           mgpu_launch_func(iter.numel(), HypotFunctor<scalar_t>(), iter.scalar_value<scalar_t>(1),
                                            iter.scalar_value<scalar_t>(2), result.data_ptr<scalar_t>(), offsets);
                       }
                       else if (iter.is_cpu_scalar(1))
                       {
                           mgpu_launch_func(iter.numel(), HypotFunctor<scalar_t>(), iter.scalar_value<scalar_t>(1),
                                            rhs.const_data_ptr<scalar_t>(), result.data_ptr<scalar_t>(), offsets);
                       }
                       else if (iter.is_cpu_scalar(2))
                       {
                           mgpu_launch_func(iter.numel(), HypotFunctor<scalar_t>(), lhs.const_data_ptr<scalar_t>(),
                                            iter.scalar_value<scalar_t>(2), result.data_ptr<scalar_t>(), offsets);
                       }
                       else
                       {
                           mgpu_launch_func(iter.numel(), HypotFunctor<scalar_t>(), lhs.const_data_ptr<scalar_t>(),
                                            rhs.const_data_ptr<scalar_t>(), result.data_ptr<scalar_t>(), offsets);
                       }
                   }),
                   AT_EXPAND(AT_FLOATING_TYPES), kHalf, kBFloat16);

    if (!result.is_same(iter.output()))
        iter.output().copy_(result);
    iter.cast_outputs();
    return out;
}

Tensor& remainder_Tensor_out(const Tensor& self, const Tensor& other, Tensor& out)
{
    TORCH_CHECK_NOT_IMPLEMENTED(
        self.scalar_type() != kBool || other.scalar_type() != kBool, "\"remainder_dgx\" not implemented for 'Bool'");
    return binary_func_out<RemainderFunctor>(self, other, out);
}

Tensor& logical_and_out(const Tensor& self, const Tensor& other, Tensor& out)
{
    return binary_func_out<LogicalAndFunctor>(self, other, out);
}

Tensor& logical_or_out(const Tensor& self, const Tensor& other, Tensor& out)
{
    return binary_func_out<LogicalOrFunctor>(self, other, out);
}

Tensor& pow_Tensor_Scalar_out(const Tensor& self, const Scalar& exponent, Tensor& out)
{
    TORCH_CHECK(self.sizes() == out.sizes());

    AT_DISPATCH_V2(self.scalar_type(), "pow_Tensor_Scalar_out", AT_WRAP([&] {
                       if (self.is_contiguous() && out.is_contiguous())
                       {
                           mgpu_launch_func(self.numel(), PowFunctor<scalar_t>(), self.const_data_ptr<scalar_t>(),
                                            exponent.to<scalar_t>(), out.data_ptr<scalar_t>());
                       }
                       else
                       {
                           const auto offsets = make_strided_offset_calculator<2>(
                               self.sizes(), std::array<IntArrayRef, 2>{ self.strides(), out.strides() });
                           mgpu_launch_func(self.numel(), PowFunctor<scalar_t>(), self.const_data_ptr<scalar_t>(),
                                            exponent.to<scalar_t>(), out.data_ptr<scalar_t>(), offsets);
                       }
                   }),
                   AT_EXPAND(AT_ALL_TYPES));

    return out;
}

TORCH_LIBRARY_IMPL(aten, PrivateUse1, m)
{
    m.impl(TORCH_SELECTIVE_NAME("aten::add.out"), TORCH_FN(add_out));
    m.impl(TORCH_SELECTIVE_NAME("aten::sub.out"), TORCH_FN(sub_out));
    m.impl(TORCH_SELECTIVE_NAME("aten::mul.out"), TORCH_FN(mul_out));
    m.impl(TORCH_SELECTIVE_NAME("aten::polar.out"), TORCH_FN(polar_out));
    m.impl(TORCH_SELECTIVE_NAME("aten::div.out"), TORCH_FN(div_out));
    m.impl(TORCH_SELECTIVE_NAME("aten::hypot.out"), TORCH_FN(hypot_out));
    m.impl(TORCH_SELECTIVE_NAME("aten::remainder.Tensor_out"), TORCH_FN(remainder_Tensor_out));
    m.impl(TORCH_SELECTIVE_NAME("aten::logical_and.out"), TORCH_FN(logical_and_out));
    m.impl(TORCH_SELECTIVE_NAME("aten::logical_or.out"), TORCH_FN(logical_or_out));
    m.impl(TORCH_SELECTIVE_NAME("aten::pow.Tensor_Scalar_out"), TORCH_FN(pow_Tensor_Scalar_out));
}

} // namespace dgx

} // namespace at::native
