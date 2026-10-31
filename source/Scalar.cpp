/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
#include <c10/cuda/CUDAStream.h>

#include <ATen/Dispatch_v2.h>
#include <ATen/EmptyTensor.h>

#include <torch/library.h>

namespace at::native
{

namespace dgx
{

#define AT_SD_BASE_TYPES                                                                                               \
    AT_EXPAND(AT_ALL_TYPES), AT_EXPAND(AT_COMPLEX_TYPES), kComplexHalf, kHalf, kBool, kBFloat16,                       \
        AT_EXPAND(AT_BAREBONES_UNSIGNED_TYPES)
#if !defined(C10_MOBILE)
#    define AT_SD_TYPES AT_EXPAND(AT_SD_BASE_TYPES), AT_EXPAND(AT_FLOAT8_TYPES)
#else
#    define AT_SD_TYPES AT_EXPAND(AT_SD_BASE_TYPES)
#endif

Scalar _local_scalar_dense_dgx(const Tensor& self)
{
    Scalar r;
    TORCH_CHECK(self.numel() > 0, "_local_scalar_dense: Empty tensor not supported");

    // DGX producers already order their results on GPU 0's current stream.
    // Only the scalar copy needs a host wait; another peer-stream fan-in is redundant.
    int current_device = cudaInvalidDeviceId;
    C10_CUDA_CHECK(cudaGetDevice(&current_device));
    constexpr int merge_device_id = 0;
    C10_CUDA_CHECK(cudaSetDevice(merge_device_id));
    auto merge_stream = c10::cuda::getCurrentCUDAStream(merge_device_id);

    AT_DISPATCH_V2(self.scalar_type(), "_local_scalar_dense_dgx", AT_WRAP([&] {
                       // Create pinned memory for the scalar value to avoid implicit
                       // locking/sync in cuda library due to pageable memory
                       auto value = at::detail::empty_cpu({ 1 }, /* size */
                                                          c10::CppTypeToScalarType<scalar_t>(), /* dtype */
                                                          std::nullopt, /* layout */
                                                          std::nullopt, /* device */
                                                          true, /* pin_memory */
                                                          std::nullopt /* memory format */
                       );
                       C10_CUDA_CHECK(cudaMemcpyAsync(value.mutable_data_ptr<scalar_t>(), self.const_data_ptr<scalar_t>(),
                                                      sizeof(scalar_t), cudaMemcpyDeviceToHost, merge_stream));
                       C10_CUDA_CHECK(cudaStreamSynchronize(merge_stream));
                       r = Scalar(*value.const_data_ptr<scalar_t>());
                   }),
                   AT_EXPAND(AT_ALL_TYPES_AND_COMPLEX), kComplexHalf, kHalf, kBool, kBFloat16,
                   AT_EXPAND(AT_FLOAT8_TYPES), AT_EXPAND(AT_BAREBONES_UNSIGNED_TYPES));

    C10_CUDA_CHECK(cudaSetDevice(current_device));
    return r;
}

TORCH_LIBRARY_IMPL(aten, PrivateUse1, m)
{
    m.impl(TORCH_SELECTIVE_NAME("aten::_local_scalar_dense"), TORCH_FN(_local_scalar_dense_dgx));
}

} // namespace dgx

} // namespace at::native
