/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
#include "KernelLaunchUtils.cuh"

#include <ATen/Dispatch_v2.h>
#include <ATen/native/cuda/UpSample.cuh>
#include <ATen/ops/empty.h>

#include <torch/library.h>

namespace at::native
{

namespace dgx
{

namespace
{

template <typename scalar_t>
__global__ void upsample_nearest2d_kernel(size_t numel,
                                          size_t offset,
                                          const scalar_t* idata,
                                          scalar_t* odata,
                                          size_t nc,
                                          size_t height1,
                                          size_t width1,
                                          size_t height2,
                                          size_t width2,
                                          float height_scale,
                                          float width_scale)
{
    unsigned long long tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid >= numel)
    {
        return;
    }
    tid += offset;

    auto nc_iter = tid / (width2 * height2) % nc;
    auto h2 = tid / width2 % height2;
    auto w2 = tid % width2;

    const size_t h1 = height1 == height2 ? h2 : nearest_neighbor_compute_source_index(height_scale, h2, height1);
    const size_t w1 = width1 == width2 ? w2 : nearest_neighbor_compute_source_index(width_scale, w2, width1);

    size_t src_index = (nc_iter * height1 + h1) * width1 + w1;
    size_t dst_index = (nc_iter * height2 + h2) * width2 + w2;

    odata[dst_index] = idata[src_index];
}

} // namespace

Tensor& upsample_nearest2d_out(const Tensor& self,
                               IntArrayRef output_size,
                               ::std::optional<double> scales_h,
                               ::std::optional<double> scales_w,
                               Tensor& out)
{
    TORCH_CHECK(self.is_contiguous());
    TORCH_CHECK(out.is_contiguous());

    auto output_height = output_size[0];
    auto output_width = output_size[1];

    auto num_batches = self.size(0);
    auto num_channels = self.size(1);
    auto input_height = self.size(2);
    auto input_width = self.size(3);

    const float height_scale = compute_scales_value<float>(scales_h, input_height, output_height);
    const float width_scale = compute_scales_value<float>(scales_w, input_width, output_width);

    AT_DISPATCH_V2(self.scalar_type(), "upsample_nearest2d_out", AT_WRAP([&] {
                       for (const auto device_id : c10::irange(c10::cuda::device_count()))
                       {
                           C10_CUDA_CHECK(cudaSetDevice(device_id));
                           auto stream = c10::cuda::getCurrentCUDAStream(device_id);

                           int64_t device_offset, device_numel;
                           std::tie(device_offset, device_numel) =
                               aligned_chunk(output_height * output_width,
                                             num_batches * num_channels * output_height * output_width, device_id);

                           unsigned int num_blocks = ::cuda::ceil_div(device_numel, kNumThreads);
                           upsample_nearest2d_kernel<<<num_blocks, kNumThreads, 0, stream>>>(
                               device_numel, device_offset, self.const_data_ptr<scalar_t>(), out.data_ptr<scalar_t>(),
                               num_batches * num_channels, input_height, input_width, output_height, output_width,
                               height_scale, width_scale);
                           C10_CUDA_KERNEL_LAUNCH_CHECK();
                       }
                   }),
                   AT_EXPAND(AT_FLOATING_TYPES));

    merge_streams();

    return out;
}

TORCH_LIBRARY_IMPL(aten, PrivateUse1, m)
{
    m.impl(TORCH_SELECTIVE_NAME("aten::upsample_nearest2d.out"), TORCH_FN(upsample_nearest2d_out));
}

} // namespace dgx

} // namespace at::native
