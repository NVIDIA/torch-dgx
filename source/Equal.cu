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

namespace at::native
{

namespace dgx
{

namespace
{

template <typename T>
struct ContiguousEqualityOp
{
    const T* lhs;
    const T* rhs;

    __device__ bool operator()(int64_t index) const
    {
        return lhs[index] == rhs[index];
    }
};

template <typename T>
struct StridedEqualityOp
{
    const T* lhs;
    const T* rhs;
    StridedOffsetCalculator<2> offsets;

    __device__ bool operator()(int64_t index) const
    {
        const auto tensor_offsets = offsets.get(index);
        return lhs[tensor_offsets[0]] == rhs[tensor_offsets[1]];
    }
};

template <typename TransformOp>
void equal_async(TransformOp transform_op, bool* device_outs, bool* out, int64_t numel)
{
    const ::cuda::std::logical_and<bool> reduction_op;
    constexpr bool init_value = true;

    auto reduce = [&](void* temp_storage, size_t& temp_storage_bytes, c10::DeviceIndex device, int64_t offset,
                      int64_t count, cudaStream_t stream) {
        auto device_in = thrust::make_counting_iterator<int64_t>(offset);
        return cub::DeviceReduce::TransformReduce(temp_storage, temp_storage_bytes, device_in, device_outs + device,
                                                  count, reduction_op, transform_op, init_value, stream);
    };
    auto merge = [&](void* temp_storage, size_t& temp_storage_bytes, c10::DeviceIndex num_devices, cudaStream_t stream) {
        return cub::DeviceReduce::Reduce(temp_storage, temp_storage_bytes, device_outs, out, num_devices, reduction_op,
                                         init_value, stream);
    };
    // equal() waits for the final result through item(), so no fan-out is needed.
    scalar_reduce(numel, reduce, merge, /*broadcast_result=*/false);
}

} // namespace

bool equal(const Tensor& self, const Tensor& other)
{
    if (self.sizes() != other.sizes())
        return false;

    int current_device = cudaInvalidDeviceId;
    C10_CUDA_CHECK(cudaGetDevice(&current_device));

    Tensor device_outs = at::empty({ c10::cuda::device_count() }, self.options().dtype(kBool));
    Tensor out = at::empty({}, self.options().dtype(kBool));
    AT_DISPATCH_V2(self.scalar_type(), "equal", AT_WRAP([&] {
                       auto device_output = device_outs.data_ptr<bool>();
                       auto output = out.data_ptr<bool>();
                       if (self.is_contiguous() && other.is_contiguous())
                       {
                           ContiguousEqualityOp<scalar_t> transform_op{ self.const_data_ptr<scalar_t>(),
                                                                        other.const_data_ptr<scalar_t>() };
                           equal_async(transform_op, device_output, output, self.numel());
                       }
                       else
                       {
                           const auto offsets = make_strided_offset_calculator<2>(
                               self.sizes(), std::array<IntArrayRef, 2>{ self.strides(), other.strides() });
                           StridedEqualityOp<scalar_t> transform_op{ self.const_data_ptr<scalar_t>(),
                                                                     other.const_data_ptr<scalar_t>(), offsets };
                           equal_async(transform_op, device_output, output, self.numel());
                       }
                   }),
                   AT_EXPAND(AT_ALL_TYPES), kBool, AT_EXPAND(AT_BAREBONES_UNSIGNED_TYPES));

    const bool result = out.item<bool>();
    // DGX allocation cleanup can change the current device.
    out.reset();
    device_outs.reset();
    C10_CUDA_CHECK(cudaSetDevice(current_device));
    return result;
}

TORCH_LIBRARY_IMPL(aten, PrivateUse1, m)
{
    m.impl(TORCH_SELECTIVE_NAME("aten::equal"), TORCH_FN(equal));
}

} // namespace dgx

} // namespace at::native
