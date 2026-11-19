/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
#include "ScalarReductionUtils.h"
#include "StridedTensorUtils.cuh"

#include <c10/cuda/CUDAFunctions.h>

#include <ATen/Dispatch_v2.h>
#include <ATen/OpMathType.h>
#include <ATen/ops/empty.h>

#include <torch/library.h>

#include <cub/device/device_reduce.cuh>
#include <cuda/std/functional>
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
        return cub::DeviceReduce::Min(temp_storage, temp_storage_bytes, static_cast<const T*>(device_outs), out,
                                        static_cast<int64_t>(num_devices), stream);
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
        return cub::DeviceReduce::Max(temp_storage, temp_storage_bytes, static_cast<const T*>(device_outs), out,
                                        static_cast<int64_t>(num_devices), stream);
    };
    scalar_reduce(numel, reduce, merge);
}

template <typename scalar_t, typename accscalar_t>
struct ProdInputOp
{
    const scalar_t* input;

    __device__ accscalar_t operator()(int64_t index) const
    {
        return static_cast<accscalar_t>(input[index]);
    }
};

template <typename TransformOp, typename accscalar_t, typename scalar_t>
void prod_async(TransformOp transform_op, accscalar_t* device_outs, scalar_t* out, int64_t numel)
{
    const ::cuda::std::multiplies<accscalar_t> reduction_op;
    const accscalar_t init_value(1);

    auto reduce = [&](void* temp_storage, size_t& temp_storage_bytes, c10::DeviceIndex device, int64_t offset,
                      int64_t count, cudaStream_t stream) {
        auto device_in = thrust::make_counting_iterator<int64_t>(offset);
        return cub::DeviceReduce::TransformReduce(temp_storage, temp_storage_bytes, device_in, device_outs + device,
                                                  count, reduction_op, transform_op, init_value, stream);
    };
    auto merge = [&](void* temp_storage, size_t& temp_storage_bytes, c10::DeviceIndex num_devices, cudaStream_t stream) {
        return cub::DeviceReduce::Reduce(
            temp_storage, temp_storage_bytes, device_outs, out, num_devices, reduction_op, init_value, stream);
    };
    scalar_reduce(numel, reduce, merge);
}

} // namespace

Tensor min(const Tensor& self)
{
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

    return result;
}

Tensor max(const Tensor& self)
{
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

    return result;
}

Tensor prod(const Tensor& self, std::optional<ScalarType> dtype)
{
    const auto result_dtype = dtype.value_or(c10::isIntegralType(self.scalar_type(), true) ? kLong : self.scalar_type());
    Tensor result = at::empty({}, self.options().dtype(result_dtype));

    AT_DISPATCH_V2(
        result_dtype, "prod", AT_WRAP([&] {
            const Tensor input = self.to(result_dtype);
            using accscalar_t = at::opmath_type<scalar_t>;
            const auto acc_dtype = c10::CppTypeToScalarType<accscalar_t>::value;
            Tensor device_results = at::empty({ c10::cuda::device_count() }, input.options().dtype(acc_dtype));
            auto device_outputs = device_results.data_ptr<accscalar_t>();
            auto output = result.data_ptr<scalar_t>();
            if (input.is_contiguous())
            {
                prod_async(ProdInputOp<scalar_t, accscalar_t>{ input.const_data_ptr<scalar_t>() }, device_outputs,
                           output, input.numel());
            }
            else
            {
                const auto offsets =
                    make_strided_offset_calculator<1>(input.sizes(), std::array<IntArrayRef, 1>{ input.strides() });
                prod_async(StridedInputOp<scalar_t, accscalar_t>{ input.const_data_ptr<scalar_t>(), offsets },
                           device_outputs, output, input.numel());
            }
        }),
        AT_EXPAND(AT_ALL_TYPES), kBool, kHalf, kBFloat16, AT_EXPAND(AT_COMPLEX_TYPES));

    return result;
}

TORCH_LIBRARY_IMPL(aten, PrivateUse1, m)
{
    m.impl(TORCH_SELECTIVE_NAME("aten::min"), TORCH_FN(min));
    m.impl(TORCH_SELECTIVE_NAME("aten::max"), TORCH_FN(max));
    m.impl(TORCH_SELECTIVE_NAME("aten::prod"), TORCH_FN(prod));
}

} // namespace dgx

} // namespace at::native
