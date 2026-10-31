/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
#include "KernelLaunchUtils.cuh"

#include <c10/cuda/CUDAStream.h>

#include <ATen/Dispatch_v2.h>
#include <ATen/ops/empty.h>

#include <torch/library.h>

#include <cub/cub.cuh>
#include <thrust/iterator/transform_iterator.h>

namespace at::native
{

namespace dgx
{

namespace
{

template <typename T>
struct NonzeroOp
{
    __device__ int64_t operator()(const T& x) const
    {
        return x != (T)0;
    }
};

template <typename T>
void count_nonzero_async(const T* in, int64_t** out, int64_t* offsets, int64_t* counts)
{
    const ::cuda::std::plus<int64_t> reduction_op;
    NonzeroOp<T> transform_op;
    for (const auto device_id : c10::irange(c10::cuda::device_count()))
    {
        C10_CUDA_CHECK(cudaSetDevice(device_id));
        auto stream = c10::cuda::getCurrentCUDAStream(device_id);

        const auto device_count = counts[device_id];
        const T* device_in = in + offsets[device_id];
        int64_t* device_out = out[device_id];
        size_t scratch_size = 0;
        void* scratch_data = nullptr;

        C10_CUDA_CHECK(cub::DeviceReduce::TransformReduce(
            scratch_data, scratch_size, device_in, device_out, device_count, reduction_op, transform_op, 0, stream));
        C10_CUDA_CHECK(cudaMallocAsync(&scratch_data, scratch_size, stream));
        C10_CUDA_CHECK(cub::DeviceReduce::TransformReduce(
            scratch_data, scratch_size, device_in, device_out, device_count, reduction_op, transform_op, 0, stream));
        C10_CUDA_CHECK(cudaFreeAsync(scratch_data, stream));
    }
}

constexpr int64_t kMaxNonzeroDims = 25;

struct LinearToCoordinatesFunctor
{
    int64_t ndim;
    int64_t sizes[kMaxNonzeroDims];

    __device__ void operator()(size_t tid, const int64_t* flagged, int64_t* out) const
    {
        int64_t linear = flagged[tid];

        for (int64_t dim = ndim; dim-- > 0;)
        {
            out[tid * ndim + dim] = linear % sizes[dim];
            linear /= sizes[dim];
        }
    }
};


} // namespace

Tensor nonzero(const Tensor& self)
{
    TORCH_CHECK(self.is_contiguous());
    TORCH_CHECK(self.dim() <= kMaxNonzeroDims, "nonzero does not support tensors with more than ", kMaxNonzeroDims,
                " dimensions");

    int device = cudaInvalidDeviceId;
    cudaGetDevice(&device);

    std::vector<int64_t> device_numels(c10::cuda::device_count());
    std::vector<int64_t> device_offsets(c10::cuda::device_count());
    for (const auto device_id : c10::irange(c10::cuda::device_count()))
    {
        std::tie(device_offsets[device_id], device_numels[device_id]) = chunk(self.numel(), device_id);
    }

    std::vector<int64_t*> device_nonzero_counts(c10::cuda::device_count());
    for (const auto device_id : c10::irange(c10::cuda::device_count()))
    {
        C10_CUDA_CHECK(cudaSetDevice(device_id));
        auto stream = c10::cuda::getCurrentCUDAStream(device_id);
        C10_CUDA_CHECK(cudaMallocAsync(&device_nonzero_counts[device_id], sizeof(int64_t), stream));
    }
    AT_DISPATCH_V2(self.scalar_type(), "count_nonzero", AT_WRAP([&] {
                       const auto input = self.const_data_ptr<scalar_t>();
                       count_nonzero_async(
                           input, device_nonzero_counts.data(), device_offsets.data(), device_numels.data());
                   }),
                   AT_EXPAND(AT_ALL_TYPES), kBool);

    // Dispatch all D2H copies concurrently. The copies are stream-ordered after the
    // CUB reduces in count_nonzero_async, so no explicit CPU sync is needed here.
    int64_t* host_nonzero_counts;
    C10_CUDA_CHECK(cudaHostAlloc(&host_nonzero_counts, c10::cuda::device_count() * sizeof(int64_t), cudaHostAllocDefault));
    for (const auto device_id : c10::irange(c10::cuda::device_count()))
    {
        C10_CUDA_CHECK(cudaSetDevice(device_id));
        auto stream = c10::cuda::getCurrentCUDAStream(device_id);
        C10_CUDA_CHECK(cudaMemcpyAsync(&host_nonzero_counts[device_id], device_nonzero_counts[device_id],
                                       sizeof(int64_t), cudaMemcpyDeviceToHost, stream));
    }

    int64_t nonzero_counts = 0;
    std::vector<int64_t> device_nonzero_offsets(c10::cuda::device_count());
    for (const auto device_id : c10::irange(c10::cuda::device_count()))
    {
        C10_CUDA_CHECK(cudaSetDevice(device_id));
        auto stream = c10::cuda::getCurrentCUDAStream(device_id);
        C10_CUDA_CHECK(cudaStreamSynchronize(stream));
        device_nonzero_offsets[device_id] = nonzero_counts;
        nonzero_counts += host_nonzero_counts[device_id];
    }
    C10_CUDA_CHECK(cudaFreeHost(host_nonzero_counts));

    auto flagged = at::empty(nonzero_counts, self.options().dtype(kLong));
    AT_DISPATCH_V2(self.scalar_type(), "nonzero_indices", AT_WRAP([&] {
                       for (const auto device_id : c10::irange(c10::cuda::device_count()))
                       {
                           C10_CUDA_CHECK(cudaSetDevice(device_id));
                           auto stream = c10::cuda::getCurrentCUDAStream(device_id);

                           // Required for the host to pass the correct value of counts[device_id]
                           auto device_count = device_numels[device_id];
                           const scalar_t* device_in = self.const_data_ptr<scalar_t>() + device_offsets[device_id];
                           int64_t* device_out = flagged.data_ptr<int64_t>() + device_nonzero_offsets[device_id];
                           size_t scratch_size = 0;
                           void* scratch_data = nullptr;

                           C10_CUDA_CHECK(cub::DeviceSelect::Flagged(
                               scratch_data, scratch_size, thrust::make_counting_iterator(device_offsets[device_id]),
                               thrust::make_transform_iterator(device_in, NonzeroOp<scalar_t>()), device_out,
                               device_nonzero_counts[device_id], device_count, stream));
                           C10_CUDA_CHECK(cudaMallocAsync(&scratch_data, scratch_size, stream));
                           C10_CUDA_CHECK(cub::DeviceSelect::Flagged(
                               scratch_data, scratch_size, thrust::make_counting_iterator(device_offsets[device_id]),
                               thrust::make_transform_iterator(device_in, NonzeroOp<scalar_t>()), device_out,
                               device_nonzero_counts[device_id], device_count, stream));
                           C10_CUDA_CHECK(cudaFreeAsync(scratch_data, stream));
                       }
                   }),
                   AT_EXPAND(AT_ALL_TYPES), kBool);


    merge_streams();

    for (const auto device_id : c10::irange(c10::cuda::device_count()))
    {
        C10_CUDA_CHECK(cudaSetDevice(device_id));
        auto stream = c10::cuda::getCurrentCUDAStream(device_id);
        C10_CUDA_CHECK(cudaFreeAsync(device_nonzero_counts[device_id], stream));
    }

    cudaSetDevice(device);

    auto out = at::empty({ nonzero_counts, self.dim() }, self.options().dtype(kLong));
    LinearToCoordinatesFunctor linear_to_coordinates{ self.dim(), {} };
    for (const auto dim : c10::irange(self.dim()))
    {
        linear_to_coordinates.sizes[dim] = self.size(dim);
    }
    mgpu_launch_func(nonzero_counts, linear_to_coordinates, flagged.const_data_ptr<int64_t>(), out.data_ptr<int64_t>());

    return out;
}

TORCH_LIBRARY_IMPL(aten, PrivateUse1, m)
{
    m.impl(TORCH_SELECTIVE_NAME("aten::nonzero"), TORCH_FN(nonzero));
}

} // namespace dgx

} // namespace at::native
