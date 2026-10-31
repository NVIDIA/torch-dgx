/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
#include "KernelLaunchUtils.cuh"

#include <ATen/Dispatch_v2.h>
#include <ATen/native/Lerp.h>

#include <torch/library.h>

namespace at::native
{

namespace dgx
{

namespace
{

template <class T>
struct LerpFunctor
{
    __device__ void operator()(size_t tid, const T* self, const T* other, T alpha, T* out)
    {
        out[tid] = lerp(self[tid], other[tid], alpha);
    }

    __device__ void operator()(size_t tid, const T* self, const T* other, const T* alpha, T* out)
    {
        out[tid] = lerp(self[tid], other[tid], alpha[tid]);
    }
};

} // namespace

Tensor& lerp_out(const Tensor& self, const Tensor& end, const Scalar& weight, Tensor& out)
{
    TORCH_CHECK(self.is_contiguous());
    TORCH_CHECK(end.is_contiguous());
    TORCH_CHECK(out.is_contiguous());

    auto iter = TensorIterator::borrowing_binary_op(out, self, end);
    AT_DISPATCH_V2(iter.common_dtype(), "lerp_out", AT_WRAP([&] {
                       using opmath_t = at::opmath_type<scalar_t>;

                       mgpu_launch_func(iter.numel(), LerpFunctor<opmath_t>(), static_cast<opmath_t*>(iter.data_ptr(1)),
                                        static_cast<opmath_t*>(iter.data_ptr(2)), weight.to<opmath_t>(),
                                        static_cast<opmath_t*>(iter.data_ptr(0)));
                   }),
                   AT_EXPAND(AT_ALL_TYPES));

    return out;
}

Tensor& lerp_out(const Tensor& self, const Tensor& end, const Tensor& weight, Tensor& out)
{
    TORCH_CHECK(self.is_contiguous());
    TORCH_CHECK(end.is_contiguous());
    TORCH_CHECK(weight.is_contiguous());
    TORCH_CHECK(out.is_contiguous());

    auto iter = TensorIteratorConfig()
                    .set_check_mem_overlap(true)
                    .allow_cpu_scalars(true)
                    .promote_inputs_to_common_dtype(true)
                    .cast_common_dtype_to_outputs(true)
                    .enforce_safe_casting_to_output(true)
                    .add_output(out)
                    .add_const_input(self)
                    .add_const_input(end)
                    .add_const_input(weight)
                    .build();

    AT_DISPATCH_V2(iter.common_dtype(), "lerp_out", AT_WRAP([&] {
                       using opmath_t = at::opmath_type<scalar_t>;

                       mgpu_launch_func(iter.numel(), LerpFunctor<opmath_t>(), static_cast<opmath_t*>(iter.data_ptr(1)),
                                        static_cast<opmath_t*>(iter.data_ptr(2)),
                                        static_cast<opmath_t*>(iter.data_ptr(3)),
                                        static_cast<opmath_t*>(iter.data_ptr(0)));
                   }),
                   AT_EXPAND(AT_ALL_TYPES));

    return out;
}

TORCH_LIBRARY_IMPL(aten, PrivateUse1, m)
{
    m.impl(TORCH_SELECTIVE_NAME("aten::lerp.Scalar_out"),
           TORCH_FN(static_cast<Tensor& (*)(const Tensor&, const Tensor&, const Scalar&, Tensor&)>(&lerp_out)));
    m.impl(TORCH_SELECTIVE_NAME("aten::lerp.Tensor_out"),
           TORCH_FN(static_cast<Tensor& (*)(const Tensor&, const Tensor&, const Tensor&, Tensor&)>(&lerp_out)));
}

} // namespace dgx

} // namespace at::native
