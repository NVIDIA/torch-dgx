/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
#include "ScalarReductionUtils.h"
#include "StridedTensorUtils.cuh"

#include <c10/cuda/CUDAFunctions.h>

#include <ATen/Dispatch_v2.h>
#include <ATen/ops/empty.h>

#include <torch/library.h>

#include <cub/device/device_reduce.cuh>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/transform_iterator.h>

namespace at::native
{

namespace dgx
{

namespace
{

template <typename InputIterator, typename T>
void min_async(InputIterator in, T* device_outs, T* out, int64_t numel)
{
    auto reduce = [&](void* temp_storage, size_t& temp_storage_bytes, c10::DeviceIndex device, int64_t offset,
                      int64_t count, cudaStream_t stream) {
        return cub::DeviceReduce::Min(temp_storage, temp_storage_bytes, in + offset, device_outs + device, count, stream);
    };
    auto merge = [&](void* temp_storage, size_t& temp_storage_bytes, c10::DeviceIndex num_devices, cudaStream_t stream) {
        return cub::DeviceReduce::Min(temp_storage, temp_storage_bytes, device_outs, out, num_devices, stream);
    };
    scalar_reduce(numel, reduce, merge);
}

template <typename InputIterator, typename T>
void max_async(InputIterator in, T* device_outs, T* out, int64_t numel)
{
    auto reduce = [&](void* temp_storage, size_t& temp_storage_bytes, c10::DeviceIndex device, int64_t offset,
                      int64_t count, cudaStream_t stream) {
        return cub::DeviceReduce::Max(temp_storage, temp_storage_bytes, in + offset, device_outs + device, count, stream);
    };
    auto merge = [&](void* temp_storage, size_t& temp_storage_bytes, c10::DeviceIndex num_devices, cudaStream_t stream) {
        return cub::DeviceReduce::Max(temp_storage, temp_storage_bytes, device_outs, out, num_devices, stream);
    };
    scalar_reduce(numel, reduce, merge);
}

} // namespace

Tensor min(const Tensor& self)
{
    int current_device = cudaInvalidDeviceId;
    C10_CUDA_CHECK(cudaGetDevice(&current_device));

    Tensor device_results = at::empty({ c10::cuda::device_count() }, self.options());
    Tensor result = at::empty({}, self.options());
    AT_DISPATCH_V2(self.scalar_type(), "min", AT_WRAP([&] {
                       const auto input = self.const_data_ptr<scalar_t>();
                       auto device_outputs = device_results.data_ptr<scalar_t>();
                       auto output = result.data_ptr<scalar_t>();
                       if (self.is_contiguous())
                       {
                           min_async(input, device_outputs, output, self.numel());
                       }
                       else
                       {
                           const auto offsets = make_strided_offset_calculator<1>(
                               self.sizes(), std::array<IntArrayRef, 1>{ self.strides() });
                           auto logical_indices = thrust::make_counting_iterator<int64_t>(0);
                           auto strided_input = thrust::make_transform_iterator(
                               logical_indices, StridedInputOp<scalar_t>{ input, offsets });
                           min_async(strided_input, device_outputs, output, self.numel());
                       }
                   }),
                   AT_EXPAND(AT_ALL_TYPES));

    device_results.reset();
    C10_CUDA_CHECK(cudaSetDevice(current_device));
    return result;
}

Tensor max(const Tensor& self)
{
    int current_device = cudaInvalidDeviceId;
    C10_CUDA_CHECK(cudaGetDevice(&current_device));

    Tensor device_results = at::empty({ c10::cuda::device_count() }, self.options());
    Tensor result = at::empty({}, self.options());
    AT_DISPATCH_V2(self.scalar_type(), "max", AT_WRAP([&] {
                       const auto input = self.const_data_ptr<scalar_t>();
                       auto device_outputs = device_results.data_ptr<scalar_t>();
                       auto output = result.data_ptr<scalar_t>();
                       if (self.is_contiguous())
                       {
                           max_async(input, device_outputs, output, self.numel());
                       }
                       else
                       {
                           const auto offsets = make_strided_offset_calculator<1>(
                               self.sizes(), std::array<IntArrayRef, 1>{ self.strides() });
                           auto logical_indices = thrust::make_counting_iterator<int64_t>(0);
                           auto strided_input = thrust::make_transform_iterator(
                               logical_indices, StridedInputOp<scalar_t>{ input, offsets });
                           max_async(strided_input, device_outputs, output, self.numel());
                       }
                   }),
                   AT_EXPAND(AT_ALL_TYPES));

    device_results.reset();
    C10_CUDA_CHECK(cudaSetDevice(current_device));
    return result;
}

TORCH_LIBRARY_IMPL(aten, PrivateUse1, m)
{
    m.impl(TORCH_SELECTIVE_NAME("aten::min"), TORCH_FN(min));
    m.impl(TORCH_SELECTIVE_NAME("aten::max"), TORCH_FN(max));
}

} // namespace dgx

} // namespace at::native
