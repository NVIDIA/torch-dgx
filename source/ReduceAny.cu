/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
#include "KernelLaunchUtils.cuh"
#include "RowReduction.cuh"
#include "ScalarReductionUtils.h"
#include "StridedTensorUtils.cuh"

#include <c10/cuda/CUDAFunctions.h>

#include <ATen/Dispatch_v2.h>
#include <ATen/ops/empty.h>

#include <torch/library.h>

#include <cub/device/device_reduce.cuh>
#include <cuda/std/bit>
#include <thrust/iterator/counting_iterator.h>

namespace at::native
{

namespace dgx
{

namespace
{

template <typename T>
struct ContiguousTruthOp
{
    const T* input;

    __device__ bool operator()(int64_t index) const
    {
        return static_cast<bool>(input[index]);
    }
};

template <typename TransformOp>
void any_async(TransformOp transform_op, bool* device_outs, bool* out, int64_t numel)
{
    const ::cuda::std::logical_or<bool> reduction_op;
    constexpr bool init_value = false;

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
    scalar_reduce(numel, reduce, merge);
}

template <typename scalar_t>
struct RowTruthTransform
{
    __device__ bool operator()(scalar_t value, int64_t) const
    {
        return static_cast<bool>(value);
    }
};

struct RowTruthOutput
{
    // Bool and Byte share the same 0/1 byte representation. Accessing bool
    // storage through unsigned char preserves aliasing rules and lets both
    // output dtypes use the same reduction kernel instantiations.
    static_assert(sizeof(bool) == sizeof(unsigned char));
    static_assert(::cuda::std::bit_cast<unsigned char>(false) == 0);
    static_assert(::cuda::std::bit_cast<unsigned char>(true) == 1);

    unsigned char* output;

    template <typename Offsets>
    __device__ void operator()(const Offsets& offsets, bool aggregate) const
    {
        output[offsets[1]] = static_cast<unsigned char>(aggregate);
    }
};

} // namespace

Tensor& any_all_out(const Tensor& self, Tensor& out)
{
    // kInt8 requires special casing that we don't support yet
    TORCH_CHECK(self.scalar_type() != kChar);

    int current_device = cudaInvalidDeviceId;
    C10_CUDA_CHECK(cudaGetDevice(&current_device));

    Tensor device_outs = at::empty({ c10::cuda::device_count() }, out.options());
    AT_DISPATCH_V2(self.scalar_type(), "any.all_out", AT_WRAP([&] {
                       auto device_outputs = device_outs.data_ptr<bool>();
                       auto output = out.data_ptr<bool>();
                       if (self.is_contiguous())
                       {
                           ContiguousTruthOp<scalar_t> transform_op{ self.const_data_ptr<scalar_t>() };
                           any_async(transform_op, device_outputs, output, self.numel());
                       }
                       else
                       {
                           const auto offsets = make_strided_offset_calculator<1>(
                               self.sizes(), std::array<IntArrayRef, 1>{ self.strides() });
                           StridedInputOp<scalar_t, bool> transform_op{ self.const_data_ptr<scalar_t>(), offsets };
                           any_async(transform_op, device_outputs, output, self.numel());
                       }
                   }),
                   kByte, kShort, kInt, kLong, kFloat, kDouble, kBool);

    device_outs.reset();
    C10_CUDA_CHECK(cudaSetDevice(current_device));
    return out;
}

Tensor& any_out(const Tensor& self, int64_t dim, bool keepdim, Tensor& out)
{
    TORCH_CHECK(self.dim() > 0, "Unsupported configuration in any_out");
    TORCH_CHECK(dim == -1 || dim == (self.dim() - 1), "Unsupported configuration in any_out");
    TORCH_CHECK(out.dim() == self.dim() - (keepdim ? 0 : 1), "Unsupported configuration in any_out");
    TORCH_CHECK(
        out.scalar_type() == kBool || out.scalar_type() == kByte, "any.out only supports bool or uint8 output tensors");

    const auto layout = make_reduction_layout<2>(
        self.sizes(), self.dim() - 1, std::array<IntArrayRef, 2>{ self.strides(), out.strides() });

    auto output = static_cast<unsigned char*>(out.data_ptr());
    AT_DISPATCH_V2(self.scalar_type(), "any_out", AT_WRAP([&] {
                       row_reduce<scalar_t, bool>(self.const_data_ptr<scalar_t>(), layout,
                                                  RowTruthTransform<scalar_t>(), ::cuda::std::logical_or<bool>(),
                                                  false, RowTruthOutput{ output });
                   }),
                   AT_EXPAND(AT_ALL_TYPES), kBool);

    return out;
}

TORCH_LIBRARY_IMPL(aten, PrivateUse1, m)
{
    m.impl(TORCH_SELECTIVE_NAME("aten::any.all_out"), TORCH_FN(any_all_out));
    m.impl(TORCH_SELECTIVE_NAME("aten::any.out"), TORCH_FN(any_out));
}

} // namespace dgx

} // namespace at::native
