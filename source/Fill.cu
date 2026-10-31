/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
#include "KernelLaunchUtils.cuh"
#include "StridedTensorUtils.cuh"

#include <ATen/Dispatch_v2.h>
#include <ATen/ops/zero_native.h>

#include <torch/library.h>

namespace at::native
{

namespace dgx
{

namespace
{

template <class T>
struct FillFunctor
{
    __device__ void operator()(size_t tid, T* self, T value)
    {
        self[tid] = value;
    }
    __device__ void operator()(size_t tid, T* self, T value, StridedOffsetCalculator<1> offsets)
    {
        self[offsets.get(tid)[0]] = value;
    }
};

template <class T>
struct MaskedFillFunctor
{
    __device__ void operator()(size_t tid, T* self, const bool* mask, T value, StridedOffsetCalculator<2> offsets)
    {
        const auto tensor_offsets = offsets.get(tid);

        if (mask[tensor_offsets[1]])
        {
            self[tensor_offsets[0]] = value;
        }
    }
};

} // namespace

Tensor& fill__scalar(Tensor& self, const Scalar& value)
{
    if (self.is_contiguous())
    {
        AT_DISPATCH_V2(self.scalar_type(), "fill__scalar", AT_WRAP([&] {
                           mgpu_launch_func(
                               self.numel(), FillFunctor<scalar_t>(), self.data_ptr<scalar_t>(), value.to<scalar_t>());
                       }),
                       AT_EXPAND(AT_ALL_TYPES), kHalf, kBFloat16, kBool);
    }
    else
    {
        const auto offsets =
            make_strided_offset_calculator<1>(self.sizes(), std::array<IntArrayRef, 1>{ self.strides() });
        AT_DISPATCH_V2(self.scalar_type(), "fill__scalar", AT_WRAP([&] {
                           mgpu_launch_func(self.numel(), FillFunctor<scalar_t>(), self.data_ptr<scalar_t>(),
                                            value.to<scalar_t>(), offsets);
                       }),
                       AT_EXPAND(AT_ALL_TYPES), kHalf, kBFloat16, kBool);
    }

    return self;
}

Tensor& masked_fill__scalar(Tensor& self, const Tensor& mask, const Scalar& value)
{
    TORCH_CHECK(mask.scalar_type() == kBool, "masked_fill_.Scalar only supports boolean masks");
    TORCH_CHECK(mask.device() == self.device(), "masked_fill_.Scalar expected mask to be on ", self.device(),
                " but got ", mask.device());

    const Tensor expanded_mask = mask.expand(self.sizes());
    const auto offsets = make_strided_offset_calculator<2>(
        self.sizes(), std::array<IntArrayRef, 2>{ self.strides(), expanded_mask.strides() });

    AT_DISPATCH_V2(self.scalar_type(), "masked_fill__scalar", AT_WRAP([&] {
                       mgpu_launch_func(self.numel(), MaskedFillFunctor<scalar_t>(), self.data_ptr<scalar_t>(),
                                        expanded_mask.const_data_ptr<bool>(), value.to<scalar_t>(), offsets);
                   }),
                   AT_EXPAND(AT_ALL_TYPES), kHalf, kBool);

    return self;
}

Tensor& zero_(at::Tensor& self)
{
    return at::native::zero_(self);
}

TORCH_LIBRARY_IMPL(aten, PrivateUse1, m)
{
    m.impl(TORCH_SELECTIVE_NAME("aten::fill_.Scalar"), TORCH_FN(fill__scalar));
    m.impl(TORCH_SELECTIVE_NAME("aten::masked_fill_.Scalar"), TORCH_FN(masked_fill__scalar));
    m.impl(TORCH_SELECTIVE_NAME("aten::zero_"), TORCH_FN(zero_));
}

} // namespace dgx

} // namespace at::native
