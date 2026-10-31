/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
#include "KernelLaunchUtils.cuh"
#include "StridedTensorUtils.cuh"
#include "TensorDiagnostics.h"

#include <ATen/Dispatch_v2.h>
#include <ATen/OpMathType.h>
#include <ATen/TensorIterator.h>

#include <torch/library.h>

namespace at::native
{

namespace dgx
{

namespace
{

template <typename scalar_t>
struct AddCMul
{
    __device__ scalar_t operator()(const scalar_t& a, const scalar_t& b, const scalar_t& c, const scalar_t& d)
    {
        return a + b * c * d;
    }
};

template <typename scalar_t>
struct AddCDiv
{
    __device__ scalar_t operator()(const scalar_t& a, const scalar_t& b, const scalar_t& c, const scalar_t& d)
    {
        return a + b * c / d;
    }
};

template <typename Func, typename scalar_t>
struct ElementwiseFunctor
{
    __device__ void operator()(
        size_t tid, const scalar_t* self, const scalar_t* tensor1, const scalar_t* tensor2, scalar_t value, scalar_t* out)
    {
        out[tid] = func(self[tid], value, tensor1[tid], tensor2[tid]);
    }

    __device__ void operator()(size_t tid,
                               const scalar_t* self,
                               const scalar_t* tensor1,
                               const scalar_t* tensor2,
                               scalar_t value,
                               scalar_t* out,
                               StridedOffsetCalculator<4> offsets)
    {
        const auto tensor_offsets = offsets.get(tid);
        out[tensor_offsets[3]] =
            func(self[tensor_offsets[0]], value, tensor1[tensor_offsets[1]], tensor2[tensor_offsets[2]]);
    }

    Func func;
};

} // namespace

Tensor& addcmul_out(const Tensor& self, const Tensor& tensor1, const Tensor& tensor2, const Scalar& value, Tensor& out)
{
    if (self.sizes() == out.sizes() && tensor1.sizes() == out.sizes() && tensor2.sizes() == out.sizes() &&
        self.is_contiguous() && tensor1.is_contiguous() && tensor2.is_contiguous() && out.is_contiguous())
    {
        TensorIterator iter;
        iter.build_ternary_op(out, self, tensor1, tensor2);
        AT_DISPATCH_V2(iter.common_dtype(), "addcmul_out", AT_WRAP([&] {
                           using opmath_t = at::opmath_type<scalar_t>;

                           mgpu_launch_func(iter.numel(), ElementwiseFunctor<AddCMul<scalar_t>, opmath_t>(),
                                            static_cast<opmath_t*>(iter.data_ptr(1)),
                                            static_cast<opmath_t*>(iter.data_ptr(2)),
                                            static_cast<opmath_t*>(iter.data_ptr(3)), value.to<opmath_t>(),
                                            static_cast<opmath_t*>(iter.data_ptr(0)));
                       }),
                       AT_EXPAND(AT_ALL_TYPES));
    }
    else
    {
        auto self_ = self.expand(out.sizes());
        auto tensor1_ = tensor1.expand(out.sizes());
        auto tensor2_ = tensor2.expand(out.sizes());

        const auto offsets = make_strided_offset_calculator<4>(
            out.sizes(),
            std::array<IntArrayRef, 4>{ self_.strides(), tensor1_.strides(), tensor2_.strides(), out.strides() });
        AT_DISPATCH_V2(out.scalar_type(), "addcmul_out", AT_WRAP([&] {
                           mgpu_launch_func(out.numel(), ElementwiseFunctor<AddCMul<scalar_t>, scalar_t>(),
                                            self_.const_data_ptr<scalar_t>(), tensor1_.const_data_ptr<scalar_t>(),
                                            tensor2_.const_data_ptr<scalar_t>(), value.to<scalar_t>(),
                                            out.data_ptr<scalar_t>(), offsets);
                       }),
                       AT_EXPAND(AT_FLOATING_TYPES));
    }
    return out;
}

Tensor& addcdiv_out(const Tensor& self, const Tensor& tensor1, const Tensor& tensor2, const Scalar& value, Tensor& out)
{
    if (self.sizes() == out.sizes() && tensor1.sizes() == out.sizes() && tensor2.sizes() == out.sizes() &&
        self.is_contiguous() && tensor1.is_contiguous() && tensor2.is_contiguous() && out.is_contiguous())
    {
        TensorIterator iter;
        iter.build_ternary_op(out, self, tensor1, tensor2);
        AT_DISPATCH_V2(iter.common_dtype(), "addcdiv_out", AT_WRAP([&] {
                           using opmath_t = at::opmath_type<scalar_t>;

                           mgpu_launch_func(iter.numel(), ElementwiseFunctor<AddCDiv<scalar_t>, opmath_t>(),
                                            static_cast<opmath_t*>(iter.data_ptr(1)),
                                            static_cast<opmath_t*>(iter.data_ptr(2)),
                                            static_cast<opmath_t*>(iter.data_ptr(3)), value.to<opmath_t>(),
                                            static_cast<opmath_t*>(iter.data_ptr(0)));
                       }),
                       AT_EXPAND(AT_ALL_TYPES));
    }
    else
    {
        auto self_ = self.expand(out.sizes());
        auto tensor1_ = tensor1.expand(out.sizes());
        auto tensor2_ = tensor2.expand(out.sizes());

        const auto offsets = make_strided_offset_calculator<4>(
            out.sizes(),
            std::array<IntArrayRef, 4>{ self_.strides(), tensor1_.strides(), tensor2_.strides(), out.strides() });
        AT_DISPATCH_V2(out.scalar_type(), "addcdiv_out", AT_WRAP([&] {
                           mgpu_launch_func(out.numel(), ElementwiseFunctor<AddCDiv<scalar_t>, scalar_t>(),
                                            self_.const_data_ptr<scalar_t>(), tensor1_.const_data_ptr<scalar_t>(),
                                            tensor2_.const_data_ptr<scalar_t>(), value.to<scalar_t>(),
                                            out.data_ptr<scalar_t>(), offsets);
                       }),
                       AT_EXPAND(AT_FLOATING_TYPES));
    }
    return out;
}

TORCH_LIBRARY_IMPL(aten, PrivateUse1, m)
{
    m.impl(TORCH_SELECTIVE_NAME("aten::addcmul.out"), TORCH_FN(addcmul_out));
    m.impl(TORCH_SELECTIVE_NAME("aten::addcdiv.out"), TORCH_FN(addcdiv_out));
}

} // namespace dgx

} // namespace at::native
