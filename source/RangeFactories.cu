/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
#include "KernelLaunchUtils.cuh"

#include <ATen/AccumulateType.h>
#include <ATen/Dispatch_v2.h>
#include <ATen/ops/empty_like.h>

#include <torch/library.h>
#include <torch/version.h>

#if TORCH_VERSION_MAJOR > 2 || (TORCH_VERSION_MAJOR == 2 && TORCH_VERSION_MINOR >= 14)
#    include <ATen/native/RangeUtils.h>
#endif

namespace at::native
{

namespace dgx
{

namespace
{

template <typename scalar_t, typename accscalar_t>
struct RangeFunctor
{
    __device__ void operator()(size_t tid, PackedTensorAccessor64<scalar_t, 1> self, accscalar_t xstep, accscalar_t xstart)
    {
        self[tid] = static_cast<scalar_t>(xstep * static_cast<accscalar_t>(tid) + xstart);
    }
};

} // namespace

Tensor& arange_out(const Scalar& start, const Scalar& end, const Scalar& step, Tensor& result)
{
    AT_DISPATCH_V2(result.scalar_type(), "arange_out", AT_WRAP([&] {
                       using accscalar_t = at::acc_type<scalar_t, true>;
                       auto xstart = start.to<accscalar_t>();
                       auto xstep = step.to<accscalar_t>();

#if TORCH_VERSION_MAJOR > 2 || (TORCH_VERSION_MAJOR == 2 && TORCH_VERSION_MINOR >= 14)
                       int64_t size = compute_arange_size<scalar_t>(start, end, step);
#else
                       auto xend = end.to<accscalar_t>();
                       TORCH_CHECK(xstep > 0 || xstep < 0, "step must be nonzero");
                       TORCH_CHECK(std::isfinite(static_cast<double>(xstart)) && std::isfinite(static_cast<double>(xend)),
                                   "unsupported range: ", xstart, " -> ", xend);
                       TORCH_CHECK(((xstep > 0) && (xend >= xstart)) || ((xstep < 0) && (xend <= xstart)),
                                   "upper bound and larger bound inconsistent with step sign");

                       // we use double precision for (start - end) / step
                       // to compute size_d for consistency across devices.
                       // The problem with using accscalar_t is that accscalar_t might be float32 on gpu for a float32
                       // scalar_t, but double on cpu for the same, and the effective output size starts differing on
                       // CPU vs GPU because of precision issues, which we dont want. the corner-case we do want to take
                       // into account is int64_t, which has higher precision than double
                       double size_d;
                       if constexpr (std::is_same_v<scalar_t, int64_t>)
                       {
                           int64_t sgn = (xstep > 0) - (xstep < 0);
                           size_d = std::ceil((xend - xstart + xstep - sgn) / xstep);
                       }
                       else
                       {
                           size_d =
                               std::ceil(static_cast<double>(end.to<double>() - start.to<double>()) / step.to<double>());
                       }

                       TORCH_CHECK(size_d >= 0 && size_d <= static_cast<double>(std::numeric_limits<int64_t>::max()),
                                   "invalid size, possible overflow?");
                       int64_t size = static_cast<int64_t>(size_d);
#endif
                       int64_t numel = result.numel();

                       if (numel != size)
                       {
                           if (numel > 0)
                           {
                               TORCH_WARN("The number of elements in the out tensor of shape ", result.sizes(), " is ",
                                          numel, " which does not match the computed number of elements ", size,
                                          ". Note that this may occur as a result of rounding error. "
                                          "The out tensor will be resized to a tensor of shape (",
                                          size, ",).");
                           }
                           result.resize_({ size });
                       }
                       bool is_contiguous = result.is_contiguous();
                       Tensor r = !is_contiguous ? at::empty_like(result, LEGACY_CONTIGUOUS_MEMORY_FORMAT) : result;

                       mgpu_launch_func(r.numel(), RangeFunctor<scalar_t, accscalar_t>(),
                                        r.packed_accessor64<scalar_t, 1>(), xstep, xstart);

                       if (!is_contiguous)
                       {
                           result.copy_(r);
                       }
                   }),

                   AT_EXPAND(AT_ALL_TYPES), at::ScalarType::Half, at::ScalarType::BFloat16);

    return result;
}

TORCH_LIBRARY_IMPL(aten, PrivateUse1, m)
{
    m.impl(TORCH_SELECTIVE_NAME("aten::arange.start_out"), TORCH_FN(arange_out));
}

} // namespace dgx

} // namespace at::native
