/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
#include "KernelLaunchUtils.cuh"
#include "StridedTensorUtils.cuh"

#include <ATen/Dispatch_v2.h>
#include <ATen/ExpandUtils.h>
#include <ATen/OpMathType.h>
#include <ATen/TensorIterator.h>
#include <ATen/native/Resize.h>
#include <ATen/ops/empty.h>
#include <ATen/ops/result_type.h>

#include <torch/library.h>

#include <cuda/std/functional>

#include <limits>
#include <optional>

namespace at::native
{

namespace dgx
{

namespace
{

template <typename BinaryFunc, typename scalar_t, typename out_t>
struct BinaryOpFunctor
{
    __device__ void operator()(size_t tid, const scalar_t* self, const scalar_t* other, out_t* out)
    {
        out[tid] = func(self[tid], other[tid]);
    }

    __device__ void operator()(size_t tid, const scalar_t* self, scalar_t other, out_t* out)
    {
        out[tid] = func(self[tid], other);
    }

    BinaryFunc func;
};

template <typename BinaryFunc, typename scalar_t, typename out_t>
struct DynamicBinaryOpStridedFunctor
{
    __device__ void operator()(
        size_t index, const scalar_t* self, const scalar_t* other, out_t* out, StridedOffsetCalculator<3> offsets)
    {
        const auto tensor_offsets = offsets.get(index);
        out[tensor_offsets[2]] = func(self[tensor_offsets[0]], other[tensor_offsets[1]]);
    }

    __device__ void operator()(
        size_t index, const scalar_t* self, scalar_t other, out_t* out, StridedOffsetCalculator<2> offsets)
    {
        const auto tensor_offsets = offsets.get(index);
        out[tensor_offsets[1]] = func(self[tensor_offsets[0]], other);
    }

    BinaryFunc func;
};

template <typename T>
struct WhereOpFunctor
{
    __device__ void operator()(size_t tid, const bool* condition, const T* self, const T* other, T* out)
    {
        out[tid] = condition[tid] ? self[tid] : other[tid];
    }
};

template <typename T>
struct StridedWhereOpFunctor
{
    __device__ void operator()(
        size_t index, const bool* condition, const T* self, const T* other, T* out, StridedOffsetCalculator<4> offsets)
    {
        const auto tensor_offsets = offsets.get(index);
        out[tensor_offsets[3]] = condition[tensor_offsets[0]] ? self[tensor_offsets[1]] : other[tensor_offsets[2]];
    }
};

template <typename T>
struct StridedConditionWhereOpFunctor
{
    __device__ void operator()(
        size_t index, const bool* condition, const T* self, const T* other, T* out, StridedOffsetCalculator<2> offsets)
    {
        const auto tensor_offsets = offsets.get(index);
        out[tensor_offsets[1]] = condition[tensor_offsets[0]] ? self[index] : other[index];
    }
};

template <typename T>
struct SingleBroadcastWhereOpFunctor
{
    __device__ void operator()(size_t index,
                               const bool* condition,
                               const T* self,
                               const T* other,
                               T* out,
                               int64_t inner_size,
                               int64_t broadcast_block_size)
    {
        const int64_t condition_index = (index / broadcast_block_size) * inner_size + index % inner_size;
        out[index] = condition[condition_index] ? self[index] : other[index];
    }
};

template <typename T>
struct RepeatedConditionWhereOpFunctor
{
    __device__ void operator()(
        size_t index, const bool* condition, const T* self, const T* other, T* out, int64_t condition_size)
    {
        out[index] = condition[index % condition_size] ? self[index] : other[index];
    }
};

template <typename T>
struct InnermostBroadcastWhereOpFunctor
{
    __device__ void operator()(
        size_t index, const bool* condition, const T* self, const T* other, T* out, int64_t broadcast_size)
    {
        out[index] = condition[index / broadcast_size] ? self[index] : other[index];
    }
};

std::optional<int64_t> single_broadcast_dim(const Tensor& tensor, IntArrayRef output_shape)
{
    if (!tensor.is_contiguous() || tensor.dim() != output_shape.size())
        return std::nullopt;

    std::optional<int64_t> broadcast_dim;
    for (const auto dim : c10::irange(tensor.dim()))
    {
        if (tensor.size(dim) == output_shape[dim])
            continue;
        if (tensor.size(dim) != 1 || broadcast_dim)
            return std::nullopt;
        broadcast_dim = dim;
    }
    return broadcast_dim;
}

} // namespace

template <typename BinaryFunc, typename scalar_t, typename out_t>
Tensor& func_Scalar_out(const Tensor& self, const Scalar& other, Tensor& out)
{
    if (self.is_contiguous() && out.is_contiguous())
    {
        using OpFunc = BinaryOpFunctor<BinaryFunc, scalar_t, out_t>;
        mgpu_launch_func(self.numel(), OpFunc(), self.data_ptr<scalar_t>(), other.to<scalar_t>(), out.data_ptr<out_t>());
    }
    else
    {
        const auto offsets =
            make_strided_offset_calculator<2>(self.sizes(), std::array<IntArrayRef, 2>{ self.strides(), out.strides() });
        using OpFunc = DynamicBinaryOpStridedFunctor<BinaryFunc, scalar_t, out_t>;
        mgpu_launch_func(self.numel(), OpFunc(), self.const_data_ptr<scalar_t>(), other.to<scalar_t>(),
                         out.data_ptr<out_t>(), offsets);
    }
    return out;
}

template <typename BinaryFunc, typename scalar_t, typename out_t>
Tensor& func_Tensor_out(const Tensor& self, const Tensor& other, Tensor& out)
{
    // TODO: Switch to common_dtype()
    auto self_ = self.expand(out.sizes());
    auto other_ = other.to(self.scalar_type()).expand(out.sizes());
    if (self_.is_contiguous() && other_.is_contiguous() && out.is_contiguous())
    {
        using OpFunc = BinaryOpFunctor<BinaryFunc, scalar_t, out_t>;
        mgpu_launch_func(out.numel(), OpFunc(), self_.const_data_ptr<scalar_t>(), other_.const_data_ptr<scalar_t>(),
                         out.data_ptr<out_t>());
    }
    else
    {
        const auto offsets = make_strided_offset_calculator<3>(
            out.sizes(), std::array<IntArrayRef, 3>{ self_.strides(), other_.strides(), out.strides() });

        using OpFunc = DynamicBinaryOpStridedFunctor<BinaryFunc, scalar_t, out_t>;
        mgpu_launch_func(out.numel(), OpFunc(), self_.const_data_ptr<scalar_t>(), other_.const_data_ptr<scalar_t>(),
                         out.data_ptr<out_t>(), offsets);
    }
    return out;
}

Tensor& ne_Scalar_out(const Tensor& self, const Scalar& other, Tensor& out)
{
    AT_DISPATCH_V2(self.scalar_type(), "ne_Scalar_out", AT_WRAP([&] {
                       func_Scalar_out<::cuda::std::not_equal_to<scalar_t>, scalar_t, bool>(self, other, out);
                   }),
                   AT_EXPAND(AT_ALL_TYPES), kBool);
    return out;
}

Tensor& ne_Tensor_out(const Tensor& self, const Tensor& other, Tensor& out)
{
    AT_DISPATCH_V2(self.scalar_type(), "ne_Tensor_out", AT_WRAP([&] {
                       func_Tensor_out<::cuda::std::not_equal_to<scalar_t>, scalar_t, bool>(self, other, out);
                   }),
                   AT_EXPAND(AT_ALL_TYPES), kBool);
    return out;
}

Tensor& eq_Scalar_out(const Tensor& self, const Scalar& other, Tensor& out)
{
    AT_DISPATCH_V2(self.scalar_type(), "eq_Scalar_out",
                   AT_WRAP([&] { func_Scalar_out<::cuda::std::equal_to<scalar_t>, scalar_t, bool>(self, other, out); }),
                   AT_EXPAND(AT_ALL_TYPES), kBool);
    return out;
}

Tensor& eq_Tensor_out(const Tensor& self, const Tensor& other, Tensor& out)
{
    AT_DISPATCH_V2(self.scalar_type(), "eq_Tensor_out",
                   AT_WRAP([&] { func_Tensor_out<::cuda::std::equal_to<scalar_t>, scalar_t, bool>(self, other, out); }),
                   AT_EXPAND(AT_ALL_TYPES), kBool);
    return out;
}

Tensor& gt_Scalar_out(const Tensor& self, const Scalar& other, Tensor& out)
{
    AT_DISPATCH_V2(self.scalar_type(), "gt_Scalar_out",
                   AT_WRAP([&] { func_Scalar_out<::cuda::std::greater<scalar_t>, scalar_t, bool>(self, other, out); }),
                   AT_EXPAND(AT_ALL_TYPES));
    return out;
}

Tensor& gt_Tensor_out(const Tensor& self, const Tensor& other, Tensor& out)
{
    AT_DISPATCH_V2(self.scalar_type(), "gt_Tensor_out",
                   AT_WRAP([&] { func_Tensor_out<::cuda::std::greater<scalar_t>, scalar_t, bool>(self, other, out); }),
                   AT_EXPAND(AT_ALL_TYPES));
    return out;
}

Tensor& ge_Scalar_out(const Tensor& self, const Scalar& other, Tensor& out)
{
    AT_DISPATCH_V2(self.scalar_type(), "ge_Scalar_out", AT_WRAP([&] {
                       func_Scalar_out<::cuda::std::greater_equal<scalar_t>, scalar_t, bool>(self, other, out);
                   }),
                   AT_EXPAND(AT_ALL_TYPES));
    return out;
}

Tensor& ge_Tensor_out(const Tensor& self, const Tensor& other, Tensor& out)
{
    AT_DISPATCH_V2(self.scalar_type(), "ge_Tensor_out", AT_WRAP([&] {
                       func_Tensor_out<::cuda::std::greater_equal<scalar_t>, scalar_t, bool>(self, other, out);
                   }),
                   AT_EXPAND(AT_ALL_TYPES));
    return out;
}

Tensor& lt_Scalar_out(const Tensor& self, const Scalar& other, Tensor& out)
{
    AT_DISPATCH_V2(self.scalar_type(), "lt_Scalar_out",
                   AT_WRAP([&] { func_Scalar_out<::cuda::std::less<scalar_t>, scalar_t, bool>(self, other, out); }),
                   AT_EXPAND(AT_ALL_TYPES));
    return out;
}

Tensor& lt_Tensor_out(const Tensor& self, const Tensor& other, Tensor& out)
{
    AT_DISPATCH_V2(self.scalar_type(), "lt_Tensor_out",
                   AT_WRAP([&] { func_Tensor_out<::cuda::std::less<scalar_t>, scalar_t, bool>(self, other, out); }),
                   AT_EXPAND(AT_ALL_TYPES));
    return out;
}

Tensor& le_Scalar_out(const Tensor& self, const Scalar& other, Tensor& out)
{
    AT_DISPATCH_V2(self.scalar_type(), "le_Scalar_out",
                   AT_WRAP([&] { func_Scalar_out<::cuda::std::less_equal<scalar_t>, scalar_t, bool>(self, other, out); }),
                   AT_EXPAND(AT_ALL_TYPES));
    return out;
}

Tensor& le_Tensor_out(const Tensor& self, const Tensor& other, Tensor& out)
{
    AT_DISPATCH_V2(self.scalar_type(), "le_Tensor_out",
                   AT_WRAP([&] { func_Tensor_out<::cuda::std::less_equal<scalar_t>, scalar_t, bool>(self, other, out); }),
                   AT_EXPAND(AT_ALL_TYPES));
    return out;
}

Tensor& bitwise_and_out(const Tensor& self, const Tensor& other, Tensor& out)
{
    AT_DISPATCH_V2(self.scalar_type(), "bitwise_and_out", AT_WRAP([&] {
                       func_Tensor_out<::cuda::std::bit_and<scalar_t>, scalar_t, scalar_t>(self, other, out);
                   }),
                   AT_EXPAND(AT_INTEGRAL_TYPES), kBool);
    return out;
}

Tensor& bitwise_or_out(const Tensor& self, const Tensor& other, Tensor& out)
{
    AT_DISPATCH_V2(self.scalar_type(), "bitwise_or_out",
                   AT_WRAP([&] { func_Tensor_out<::cuda::std::bit_or<scalar_t>, scalar_t, scalar_t>(self, other, out); }),
                   AT_EXPAND(AT_INTEGRAL_TYPES), kBool);
    return out;
}

Tensor& where_out(const Tensor& condition, const Tensor& self, const Tensor& other, Tensor& out)
{
    TORCH_CHECK(condition.scalar_type() == kBool, "where expected condition to be a boolean tensor");

    auto output_shape = at::infer_size_dimvector(condition.sizes(), self.sizes());
    output_shape = at::infer_size_dimvector(output_shape, other.sizes());
    const auto output_type = at::result_type(self, other);

    Tensor condition_expanded = condition.expand(output_shape);
    Tensor self_expanded = self.to(output_type).expand(output_shape);
    Tensor other_expanded = other.to(output_type).expand(output_shape);
    TORCH_CHECK(out.scalar_type() == output_type, "Expected out tensor to have dtype ", output_type, ", but got ",
                out.scalar_type());
    at::native::resize_output(out, output_shape);
    const auto condition_broadcast_dim = single_broadcast_dim(condition, output_shape);

    int64_t outer_size = 1;
    int64_t inner_size = 1;
    int64_t broadcast_size = 1;
    int64_t broadcast_block_size = 1;
    if (condition_broadcast_dim)
    {
        for (int64_t dim = 0; dim < *condition_broadcast_dim; ++dim)
            outer_size *= output_shape[dim];
        for (int64_t dim = *condition_broadcast_dim + 1; dim < static_cast<int64_t>(output_shape.size()); ++dim)
            inner_size *= output_shape[dim];
        broadcast_size = output_shape[*condition_broadcast_dim];
        broadcast_block_size = inner_size * broadcast_size;
    }

    AT_DISPATCH_V2(
        out.scalar_type(), "where.self_out", AT_WRAP([&] {
            if (condition_expanded.is_contiguous() && self_expanded.is_contiguous() && other_expanded.is_contiguous() &&
                out.is_contiguous())
            {
                mgpu_launch_func(out.numel(), WhereOpFunctor<scalar_t>(), condition_expanded.const_data_ptr<bool>(),
                                 self_expanded.const_data_ptr<scalar_t>(), other_expanded.const_data_ptr<scalar_t>(),
                                 out.data_ptr<scalar_t>());
            }
            else if (condition_broadcast_dim && self_expanded.is_contiguous() && other_expanded.is_contiguous() &&
                     out.is_contiguous())
            {
                if (outer_size == 1)
                {
                    mgpu_launch_func(out.numel(), RepeatedConditionWhereOpFunctor<scalar_t>(),
                                     condition.const_data_ptr<bool>(), self_expanded.const_data_ptr<scalar_t>(),
                                     other_expanded.const_data_ptr<scalar_t>(), out.data_ptr<scalar_t>(), inner_size);
                }
                else if (inner_size == 1)
                {
                    mgpu_launch_func(out.numel(), InnermostBroadcastWhereOpFunctor<scalar_t>(),
                                     condition.const_data_ptr<bool>(), self_expanded.const_data_ptr<scalar_t>(),
                                     other_expanded.const_data_ptr<scalar_t>(), out.data_ptr<scalar_t>(), broadcast_size);
                }
                else
                {
                    mgpu_launch_func(out.numel(), SingleBroadcastWhereOpFunctor<scalar_t>(),
                                     condition.const_data_ptr<bool>(), self_expanded.const_data_ptr<scalar_t>(),
                                     other_expanded.const_data_ptr<scalar_t>(), out.data_ptr<scalar_t>(), inner_size,
                                     broadcast_block_size);
                }
            }
            else if (self_expanded.is_contiguous() && other_expanded.is_contiguous() && out.is_contiguous())
            {
                const auto offsets = make_strided_offset_calculator<2>(
                    out.sizes(), std::array<IntArrayRef, 2>{ condition_expanded.strides(), out.strides() });
                mgpu_launch_func(out.numel(), StridedConditionWhereOpFunctor<scalar_t>(),
                                 condition_expanded.const_data_ptr<bool>(), self_expanded.const_data_ptr<scalar_t>(),
                                 other_expanded.const_data_ptr<scalar_t>(), out.data_ptr<scalar_t>(), offsets);
            }
            else
            {
                const auto offsets = make_strided_offset_calculator<4>(
                    out.sizes(), std::array<IntArrayRef, 4>{ condition_expanded.strides(), self_expanded.strides(),
                                                             other_expanded.strides(), out.strides() });
                mgpu_launch_func(out.numel(), StridedWhereOpFunctor<scalar_t>(),
                                 condition_expanded.const_data_ptr<bool>(), self_expanded.const_data_ptr<scalar_t>(),
                                 other_expanded.const_data_ptr<scalar_t>(), out.data_ptr<scalar_t>(), offsets);
            }
        }),
        AT_EXPAND(AT_ALL_TYPES), kHalf, kBFloat16, kBool);
    return out;
}

Tensor where(const Tensor& condition, const Tensor& self, const Tensor& other)
{
    auto output_shape = at::infer_size_dimvector(condition.sizes(), self.sizes());
    output_shape = at::infer_size_dimvector(output_shape, other.sizes());
    auto out = at::empty(output_shape, self.options().dtype(at::result_type(self, other)));
    where_out(condition, self, other, out);
    return out;
}

Tensor& isneginf_out(const Tensor& self, Tensor& out)
{
    at::native::resize_output(out, self.sizes());
    TORCH_CHECK(out.scalar_type() == kBool, "Expected out tensor to have dtype bool, but got ", out.scalar_type());

    AT_DISPATCH_V2(self.scalar_type(), "isneginf_out", AT_WRAP([&] {
                       func_Scalar_out<::cuda::std::equal_to<scalar_t>, scalar_t, bool>(
                           self, -std::numeric_limits<double>::infinity(), out);
                   }),
                   AT_EXPAND(AT_FLOATING_TYPES), kHalf, kBFloat16);
    return out;
}

TORCH_LIBRARY_IMPL(aten, PrivateUse1, m)
{
    m.impl(TORCH_SELECTIVE_NAME("aten::ne.Scalar_out"), TORCH_FN(ne_Scalar_out));
    m.impl(TORCH_SELECTIVE_NAME("aten::ne.Tensor_out"), TORCH_FN(ne_Tensor_out));
    m.impl(TORCH_SELECTIVE_NAME("aten::eq.Scalar_out"), TORCH_FN(eq_Scalar_out));
    m.impl(TORCH_SELECTIVE_NAME("aten::eq.Tensor_out"), TORCH_FN(eq_Tensor_out));
    m.impl(TORCH_SELECTIVE_NAME("aten::gt.Scalar_out"), TORCH_FN(gt_Scalar_out));
    m.impl(TORCH_SELECTIVE_NAME("aten::gt.Tensor_out"), TORCH_FN(gt_Tensor_out));
    m.impl(TORCH_SELECTIVE_NAME("aten::ge.Scalar_out"), TORCH_FN(ge_Scalar_out));
    m.impl(TORCH_SELECTIVE_NAME("aten::ge.Tensor_out"), TORCH_FN(ge_Tensor_out));
    m.impl(TORCH_SELECTIVE_NAME("aten::lt.Scalar_out"), TORCH_FN(lt_Scalar_out));
    m.impl(TORCH_SELECTIVE_NAME("aten::lt.Tensor_out"), TORCH_FN(lt_Tensor_out));
    m.impl(TORCH_SELECTIVE_NAME("aten::le.Scalar_out"), TORCH_FN(le_Scalar_out));
    m.impl(TORCH_SELECTIVE_NAME("aten::le.Tensor_out"), TORCH_FN(le_Tensor_out));
    m.impl(TORCH_SELECTIVE_NAME("aten::bitwise_and.Tensor_out"), TORCH_FN(bitwise_and_out));
    m.impl(TORCH_SELECTIVE_NAME("aten::bitwise_or.Tensor_out"), TORCH_FN(bitwise_or_out));
    m.impl(TORCH_SELECTIVE_NAME("aten::where.self"), TORCH_FN(where));
    m.impl(TORCH_SELECTIVE_NAME("aten::where.self_out"), TORCH_FN(where_out));
    m.impl(TORCH_SELECTIVE_NAME("aten::isneginf.out"), TORCH_FN(isneginf_out));
}

} // namespace dgx

} // namespace at::native
