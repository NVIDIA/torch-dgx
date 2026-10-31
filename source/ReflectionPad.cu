/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
#include "KernelLaunchUtils.cuh"

#include <ATen/Dispatch_v2.h>
#include <ATen/native/Padding.h>
#include <ATen/ops/empty.h>
#include <ATen/ops/empty_like.h>

#include <torch/library.h>
#include <torch/version.h>

namespace at::native
{

namespace dgx
{

namespace
{

template <typename scalar_t>
struct ReflectionPad2dFunctor
{
    __device__ void operator()(size_t tid,
                               const PackedTensorAccessor64<scalar_t, 3, RestrictPtrTraits> self,
                               PackedTensorAccessor64<scalar_t, 3, RestrictPtrTraits> dst,
                               int64_t pad_l,
                               int64_t pad_t)
    {
        int64_t idx_0 = tid / (dst.size(2) * dst.size(1)) % dst.size(0);
        int64_t dst_idx_1 = tid / dst.size(2) % dst.size(1);
        int64_t dst_idx_2 = tid % dst.size(2);

        int64_t self_idx_1 = dst_idx_1 - pad_t;
        int64_t self_idx_2 = dst_idx_2 - pad_l;

        if (self_idx_1 < 0)
            self_idx_1 = -self_idx_1;
        else if (self_idx_1 > (self.size(1) - 1))
            self_idx_1 = (self.size(1) - 1) - (self_idx_1 - (self.size(1) - 1));

        if (self_idx_2 < 0)
            self_idx_2 = -self_idx_2;
        else if (self_idx_2 > (self.size(2) - 1))
            self_idx_2 = (self.size(2) - 1) - (self_idx_2 - (self.size(2) - 1));

        dst[idx_0][dst_idx_1][dst_idx_2] = self[idx_0][self_idx_1][self_idx_2];
    }


    __device__ void operator()(size_t tid,
                               const PackedTensorAccessor64<scalar_t, 4, RestrictPtrTraits> self,
                               PackedTensorAccessor64<scalar_t, 4, RestrictPtrTraits> dst,
                               int64_t pad_l,
                               int64_t pad_t)
    {
        int64_t idx_0 = tid / (dst.size(3) * dst.size(2) * dst.size(1)) % dst.size(0);
        int64_t idx_1 = tid / (dst.size(3) * dst.size(2)) % dst.size(1);
        int64_t dst_idx_2 = tid / dst.size(3) % dst.size(2);
        int64_t dst_idx_3 = tid % dst.size(3);

        int64_t self_idx_2 = dst_idx_2 - pad_t;
        int64_t self_idx_3 = dst_idx_3 - pad_l;

        if (self_idx_2 < 0)
            self_idx_2 = -self_idx_2;
        else if (self_idx_2 > (self.size(2) - 1))
            self_idx_2 = (self.size(2) - 1) - (self_idx_2 - (self.size(2) - 1));

        if (self_idx_3 < 0)
            self_idx_3 = -self_idx_3;
        else if (self_idx_3 > (self.size(3) - 1))
            self_idx_3 = (self.size(3) - 1) - (self_idx_3 - (self.size(3) - 1));

        dst[idx_0][idx_1][dst_idx_2][dst_idx_3] = self[idx_0][idx_1][self_idx_2][self_idx_3];
    }
};


} // namespace

Tensor reflection_pad2d(const Tensor& self, IntArrayRef padding)
{
    int plane_dim = 0;
    int dim_h = 1;
    int dim_w = 2;
    int nbatch = 1;

#if TORCH_VERSION_MAJOR > 2 || (TORCH_VERSION_MAJOR == 2 && TORCH_VERSION_MINOR >= 15)
    at::native::padding::check_valid_input(self, padding, 2);
#else
    at::native::padding::check_valid_input<2>(self, padding);
#endif

    if (self.ndimension() == 4)
    {
        nbatch = self.size(0);
        plane_dim++;
        dim_h++;
        dim_w++;
    }

    int64_t pad_l = padding[0];
    int64_t pad_r = padding[1];
    int64_t pad_t = padding[2];
    int64_t pad_b = padding[3];

    int nplane = self.size(plane_dim);
    int self_h = self.size(dim_h);
    int self_w = self.size(dim_w);

    TORCH_CHECK(pad_l < self_w && pad_r < self_w,
                "Padding size should be less than the corresponding input dimension, but "
                "got: padding (",
                pad_l, ", ", pad_r, ") at dimension ", dim_w, " of input ", self.sizes());

    TORCH_CHECK(pad_t < self_h && pad_b < self_h,
                "Padding size should be less than the corresponding input dimension, but "
                "got: padding (",
                pad_t, ", ", pad_b, ") at dimension ", dim_h, " of input ", self.sizes());

    int output_h = self_h + pad_t + pad_b;
    int output_w = self_w + pad_l + pad_r;

    TORCH_CHECK(output_w >= 1 || output_h >= 1, "input (H: ", self_h, ", W: ", self_w,
                ") is too small.  Calculated "
                "output H: ",
                output_h, " W: ", output_w);

    if (self.ndimension() == 3)
    {
        auto output = at::empty({ nplane, output_h, output_w }, self.options());
        AT_DISPATCH_V2(self.scalar_type(), "reflection_pad2d", AT_WRAP([&] {
                           mgpu_launch_func(output.numel(), ReflectionPad2dFunctor<scalar_t>(),
                                            self.packed_accessor64<scalar_t, 3, RestrictPtrTraits>(),
                                            output.packed_accessor64<scalar_t, 3, RestrictPtrTraits>(), pad_l, pad_t);
                       }),
                       AT_EXPAND(AT_ALL_TYPES));
        return output;
    }
    else
    {
        auto output = at::empty({ nbatch, nplane, output_h, output_w }, self.options());
        AT_DISPATCH_V2(self.scalar_type(), "reflection_pad2d", AT_WRAP([&] {
                           mgpu_launch_func(output.numel(), ReflectionPad2dFunctor<scalar_t>(),
                                            self.packed_accessor64<scalar_t, 4, RestrictPtrTraits>(),
                                            output.packed_accessor64<scalar_t, 4, RestrictPtrTraits>(), pad_l, pad_t);
                       }),
                       AT_EXPAND(AT_ALL_TYPES));
        return output;
    }
}

TORCH_LIBRARY_IMPL(aten, PrivateUse1, m)
{
    m.impl(TORCH_SELECTIVE_NAME("aten::reflection_pad2d"), TORCH_FN(reflection_pad2d));
}

} // namespace dgx

} // namespace at::native
