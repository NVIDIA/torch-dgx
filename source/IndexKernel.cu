/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
#include "KernelLaunchUtils.cuh"
#include "StridedTensorUtils.cuh"

#include <ATen/Dispatch_v2.h>
#include <ATen/ExpandUtils.h>
#include <ATen/MemoryOverlap.h>
#include <ATen/core/Formatting.h>
#include <ATen/native/CPUFallback.h>
#include <ATen/ops/arange.h>
#include <ATen/ops/cumsum.h>
#include <ATen/ops/empty.h>
#include <ATen/ops/empty_like.h>
#include <ATen/ops/masked_select.h>
#include <ATen/ops/nonzero.h>

#include <torch/library.h>

#include <cub/cub.cuh>
#include <cuda/atomic>

namespace at::native
{

namespace dgx
{

namespace
{

template <typename scalar_t, typename index_t>
__global__ void index_out_kernel(const scalar_t* __restrict__ input,
                                 size_t indices_offset,
                                 size_t indices_count,
                                 PackedTensorAccessor64<const index_t, 1, RestrictPtrTraits> indices,
                                 size_t output_offset,
                                 PackedTensorAccessor64<scalar_t, 1, RestrictPtrTraits> output)
{
    for (unsigned int i = threadIdx.x + blockIdx.x * blockDim.x; i < indices_count; i += blockDim.x * gridDim.x)
    {
        index_t index = indices[i + indices_offset];
        output[i + indices_offset + output_offset] = input[index];
    }
}

template <typename scalar_t, typename index_t>
__global__ void index_out_kernel(StridedInputOp<scalar_t> input,
                                 size_t indices_offset,
                                 size_t indices_count,
                                 PackedTensorAccessor64<const index_t, 1, RestrictPtrTraits> indices,
                                 size_t output_offset,
                                 PackedTensorAccessor64<scalar_t, 1, RestrictPtrTraits> output)
{
    for (size_t i = threadIdx.x + blockIdx.x * blockDim.x; i < indices_count; i += blockDim.x * gridDim.x)
    {
        const index_t linear_index = indices[i + indices_offset];
        output[i + indices_offset + output_offset] = input(linear_index);
    }
}

template <typename scalar_t, typename index_t>
__global__ void index_out_kernel(PackedTensorAccessor64<const scalar_t, 2, RestrictPtrTraits> input,
                                 size_t indices_offset,
                                 size_t indices_count,
                                 PackedTensorAccessor64<const index_t, 1, RestrictPtrTraits> indices,
                                 size_t output_offset,
                                 PackedTensorAccessor64<scalar_t, 2, RestrictPtrTraits> output)
{
    const size_t numel = indices_count * output.size(1);
    for (size_t tid = threadIdx.x + blockIdx.x * blockDim.x; tid < numel; tid += blockDim.x * gridDim.x)
    {
        const size_t i = tid / output.size(1);
        const size_t j = tid % output.size(1);
        const index_t index = indices[i + indices_offset];
        output[i + indices_offset + output_offset][j] = input[index][j];
    }
}

template <typename scalar_t, typename index_t>
__global__ void index_out_kernel(PackedTensorAccessor64<const scalar_t, 3, RestrictPtrTraits> input,
                                 size_t indices_offset,
                                 size_t indices_count,
                                 PackedTensorAccessor64<const index_t, 1, RestrictPtrTraits> indices,
                                 size_t output_offset,
                                 PackedTensorAccessor64<scalar_t, 3, RestrictPtrTraits> output)
{
    const size_t numel = indices_count * output.size(1) * output.size(2);
    for (size_t tid = threadIdx.x + blockIdx.x * blockDim.x; tid < numel; tid += blockDim.x * gridDim.x)
    {
        const size_t i = tid / (output.size(1) * output.size(2));
        const size_t j = (tid / output.size(2)) % output.size(1);
        const size_t k = tid % output.size(2);
        const index_t index = indices[i + indices_offset];
        output[i + indices_offset + output_offset][j][k] = input[index][j][k];
    }
}

template <typename scalar_t, typename index_t>
__global__ void index_out_kernel(PackedTensorAccessor64<const scalar_t, 3, RestrictPtrTraits> input,
                                 size_t indices_offset,
                                 size_t indices_count,
                                 PackedTensorAccessor64<const index_t, 1, RestrictPtrTraits> indices,
                                 size_t output_offset,
                                 PackedTensorAccessor64<scalar_t, 2, RestrictPtrTraits> output)
{
    const size_t numel = indices_count * output.size(1);
    for (size_t tid = threadIdx.x + blockIdx.x * blockDim.x; tid < numel; tid += blockDim.x * gridDim.x)
    {
        const size_t i = tid / output.size(1);
        const size_t j = tid % output.size(1);
        const index_t index = indices[i + indices_offset];
        const auto input_i = index / input.size(1);
        const auto input_j = index % input.size(1);
        output[i + indices_offset + output_offset][j] = input[input_i][input_j][j];
    }
}

template <typename scalar_t, bool accumulate = false>
struct IndexPutFunctor
{
    __device__ void operator()(size_t tid,
                               scalar_t* __restrict__ self,
                               const bool* __restrict__ mask,
                               const scalar_t* __restrict__ other)
    {
        (*this)(tid, self, mask, *other);
    }

    __device__ void operator()(size_t tid, scalar_t* __restrict__ self, const bool* __restrict__ mask, scalar_t other)
    {
        if (mask[tid])
        {
            if constexpr (accumulate)
                self[tid] += other;
            else
                self[tid] = other;
        }
    }
};

template <typename scalar_t, typename index_t, bool accumulate = false>
struct IndexPutRowsFunctor
{
    __device__ void operator()(size_t tid,
                               scalar_t* __restrict__ self,
                               int64_t self_rows,
                               int64_t row_size,
                               const index_t* __restrict__ indices,
                               const int64_t* __restrict__ last_positions,
                               const scalar_t* __restrict__ values)
    {
        put(tid, self, self_rows, row_size, indices, last_positions, values[tid]);
    }

    __device__ void operator()(size_t tid,
                               scalar_t* __restrict__ self,
                               int64_t self_rows,
                               int64_t row_size,
                               const index_t* __restrict__ indices,
                               const int64_t* __restrict__ last_positions,
                               StridedInputOp<scalar_t> values)
    {
        put(tid, self, self_rows, row_size, indices, last_positions, values(tid));
    }

    __device__ void operator()(size_t tid,
                               scalar_t* __restrict__ self,
                               int64_t self_rows,
                               int64_t row_size,
                               const index_t* __restrict__ indices,
                               const int64_t* __restrict__ last_positions,
                               scalar_t value)
    {
        put(tid, self, self_rows, row_size, indices, last_positions, value);
    }

    __device__ void operator()(size_t tid,
                               scalar_t* __restrict__ self,
                               int64_t self_rows,
                               int64_t row_size,
                               const index_t* __restrict__ indices,
                               const int64_t* __restrict__ last_positions,
                               cub::FutureValue<const scalar_t> value)
    {
        put(tid, self, self_rows, row_size, indices, last_positions, value);
    }

private:
    template <typename value_t>
    __device__ void put(size_t tid,
                        scalar_t* __restrict__ self,
                        int64_t self_rows,
                        int64_t row_size,
                        const index_t* __restrict__ indices,
                        const int64_t* __restrict__ last_positions,
                        value_t value)
    {
        const auto index_position = tid / row_size;
        auto index = static_cast<int64_t>(indices[index_position]);
        CUDA_KERNEL_ASSERT(index >= -self_rows && index < self_rows && "index out of bounds");
        if (index < 0)
            index += self_rows;
        if (last_positions[index] != static_cast<int64_t>(index_position + 1))
            return;

        const auto column = tid % row_size;
        // Accumulation is only enabled for boolean masks, whose indices are unique.
        if constexpr (accumulate)
            self[index * row_size + column] += value;
        else
            self[index * row_size + column] = value;
    }
};

template <typename index_t>
struct IndexPutLastPositionFunctor
{
    __device__ void operator()(size_t index_position,
                               int64_t self_rows,
                               const index_t* __restrict__ indices,
                               int64_t first_row,
                               int64_t last_row,
                               int64_t* __restrict__ last_positions)
    {
        auto index = static_cast<int64_t>(indices[index_position]);
        CUDA_KERNEL_ASSERT(index >= -self_rows && index < self_rows && "index out of bounds");
        if (index < 0)
            index += self_rows;
        if (index < first_row || index >= last_row)
            return;

        // Store positions offset by one so zero can represent an unseen row.
        ::cuda::atomic_ref<int64_t, ::cuda::thread_scope_device> last_position(last_positions[index]);
        last_position.fetch_max(static_cast<int64_t>(index_position + 1), ::cuda::memory_order_relaxed);
    }
};

template <typename scalar_t, typename index_t>
struct IndexAddFunctor
{
    __device__ void operator()(size_t tid,
                               PackedTensorAccessor64<scalar_t, 1, RestrictPtrTraits> self,
                               PackedTensorAccessor64<const index_t, 1, RestrictPtrTraits> index,
                               PackedTensorAccessor64<const scalar_t, 1, RestrictPtrTraits> source,
                               scalar_t alpha)
    {
        const index_t i = index[tid];
        // Input shards from different GPUs can accumulate into the same element.
        ::cuda::atomic_ref<scalar_t, ::cuda::thread_scope_system> value(self[i]);
        value.fetch_add(alpha * source[tid], ::cuda::memory_order_relaxed);
    }
};


template <typename scalar_t, typename index_t>
struct GatherFunctor
{
    __device__ void operator()(size_t tid,
                               int64_t self_dim_size,
                               int64_t index_dim_size,
                               int64_t inner,
                               const scalar_t* self,
                               const index_t* indices,
                               scalar_t* out)
    {
        const size_t i = tid / (index_dim_size * inner);
        const size_t j = tid % inner;
        const index_t index = indices[tid];
        out[tid] = self[i * self_dim_size * inner + index * inner + j];
    }
};

template <typename scalar_t, typename index_t>
struct ScatterAddFunctor
{
    __device__ void operator()(size_t tid,
                               int64_t,
                               PackedTensorAccessor64<const index_t, 1, RestrictPtrTraits> index,
                               PackedTensorAccessor64<const scalar_t, 1, RestrictPtrTraits> source,
                               int64_t first_output,
                               int64_t last_output,
                               PackedTensorAccessor64<scalar_t, 1, RestrictPtrTraits> out)
    {
        const auto output_index = index[tid];
        CUDA_KERNEL_ASSERT(output_index >= 0 && output_index < out.size(0) && "index out of bounds");
        if (output_index >= first_output && output_index < last_output)
            atomicAdd(&out[output_index], source[tid]);
    }

    __device__ void operator()(size_t tid,
                               int64_t dim,
                               PackedTensorAccessor64<const index_t, 2, RestrictPtrTraits> index,
                               PackedTensorAccessor64<const scalar_t, 2, RestrictPtrTraits> source,
                               int64_t first_output,
                               int64_t last_output,
                               PackedTensorAccessor64<scalar_t, 2, RestrictPtrTraits> out)
    {
        const auto tid_x = tid / index.size(1);
        const auto tid_y = tid % index.size(1);
        const auto output_index = index[tid_x][tid_y];
        CUDA_KERNEL_ASSERT(output_index >= 0 && output_index < out.size(dim) && "index out of bounds");
        const auto output_offset = dim == 0 ? output_index * out.size(1) + tid_y : tid_x * out.size(1) + output_index;
        if (output_offset < first_output || output_offset >= last_output)
            return;
        atomicAdd(out.data() + output_offset, source[tid_x][tid_y]);
    }

    __device__ void operator()(size_t tid,
                               int64_t dim,
                               PackedTensorAccessor64<const index_t, 3, RestrictPtrTraits> index,
                               PackedTensorAccessor64<const scalar_t, 3, RestrictPtrTraits> source,
                               int64_t first_output,
                               int64_t last_output,
                               PackedTensorAccessor64<scalar_t, 3, RestrictPtrTraits> out)
    {
        const auto tid_x = tid / (index.size(1) * index.size(2));
        const auto tid_y = tid / index.size(2) % index.size(1);
        const auto tid_z = tid % index.size(2);
        const auto output_index = index[tid_x][tid_y][tid_z];
        CUDA_KERNEL_ASSERT(output_index >= 0 && output_index < out.size(dim) && "index out of bounds");
        int64_t output_offset;
        if (dim == 0)
            output_offset = (output_index * out.size(1) + tid_y) * out.size(2) + tid_z;
        else if (dim == 1)
            output_offset = (tid_x * out.size(1) + output_index) * out.size(2) + tid_z;
        else
            output_offset = (tid_x * out.size(1) + tid_y) * out.size(2) + output_index;
        if (output_offset < first_output || output_offset >= last_output)
            return;
        atomicAdd(out.data() + output_offset, source[tid_x][tid_y][tid_z]);
    }

    __device__ void operator()(size_t tid,
                               int64_t dim,
                               PackedTensorAccessor64<const index_t, 4, RestrictPtrTraits> index,
                               PackedTensorAccessor64<const scalar_t, 4, RestrictPtrTraits> source,
                               int64_t first_output,
                               int64_t last_output,
                               PackedTensorAccessor64<scalar_t, 4, RestrictPtrTraits> out)
    {
        const auto tid_0 = tid / (index.size(1) * index.size(2) * index.size(3));
        const auto tid_1 = tid / (index.size(2) * index.size(3)) % index.size(1);
        const auto tid_2 = tid / index.size(3) % index.size(2);
        const auto tid_3 = tid % index.size(3);
        const auto output_index = index[tid_0][tid_1][tid_2][tid_3];
        CUDA_KERNEL_ASSERT(output_index >= 0 && output_index < out.size(dim) && "index out of bounds");
        int64_t output_offset;
        if (dim == 0)
            output_offset = ((output_index * out.size(1) + tid_1) * out.size(2) + tid_2) * out.size(3) + tid_3;
        else if (dim == 1)
            output_offset = ((tid_0 * out.size(1) + output_index) * out.size(2) + tid_2) * out.size(3) + tid_3;
        else if (dim == 2)
            output_offset = ((tid_0 * out.size(1) + tid_1) * out.size(2) + output_index) * out.size(3) + tid_3;
        else
            output_offset = ((tid_0 * out.size(1) + tid_1) * out.size(2) + tid_2) * out.size(3) + output_index;
        if (output_offset < first_output || output_offset >= last_output)
            return;
        atomicAdd(out.data() + output_offset, source[tid_0][tid_1][tid_2][tid_3]);
    }
};

template <typename scalar_t, typename index_t>
struct ScatterValueFunctor
{
    __device__ void operator()(size_t tid,
                               int64_t,
                               PackedTensorAccessor64<const index_t, 1, RestrictPtrTraits> index,
                               scalar_t value,
                               int64_t first_output,
                               int64_t last_output,
                               PackedTensorAccessor64<scalar_t, 1, RestrictPtrTraits> out)
    {
        const auto output_index = index[tid];
        CUDA_KERNEL_ASSERT(output_index >= 0 && output_index < out.size(0) && "index out of bounds");
        if (output_index >= first_output && output_index < last_output)
            out[output_index] = value;
    }

    __device__ void operator()(size_t tid,
                               int64_t dim,
                               PackedTensorAccessor64<const index_t, 2, RestrictPtrTraits> index,
                               scalar_t value,
                               int64_t first_output,
                               int64_t last_output,
                               PackedTensorAccessor64<scalar_t, 2, RestrictPtrTraits> out)
    {
        const auto tid_x = tid / index.size(1);
        const auto tid_y = tid % index.size(1);
        const auto output_index = index[tid_x][tid_y];
        CUDA_KERNEL_ASSERT(output_index >= 0 && output_index < out.size(dim) && "index out of bounds");
        const auto output_offset = dim == 0 ? output_index * out.size(1) + tid_y : tid_x * out.size(1) + output_index;
        if (output_offset < first_output || output_offset >= last_output)
            return;
        out.data()[output_offset] = value;
    }

    __device__ void operator()(size_t tid,
                               int64_t dim,
                               PackedTensorAccessor64<const index_t, 3, RestrictPtrTraits> index,
                               scalar_t value,
                               int64_t first_output,
                               int64_t last_output,
                               PackedTensorAccessor64<scalar_t, 3, RestrictPtrTraits> out)
    {
        const auto tid_x = tid / (index.size(1) * index.size(2));
        const auto tid_y = tid / index.size(2) % index.size(1);
        const auto tid_z = tid % index.size(2);
        const auto output_index = index[tid_x][tid_y][tid_z];
        CUDA_KERNEL_ASSERT(output_index >= 0 && output_index < out.size(dim) && "index out of bounds");
        int64_t output_offset;
        if (dim == 0)
            output_offset = (output_index * out.size(1) + tid_y) * out.size(2) + tid_z;
        else if (dim == 1)
            output_offset = (tid_x * out.size(1) + output_index) * out.size(2) + tid_z;
        else
            output_offset = (tid_x * out.size(1) + tid_y) * out.size(2) + output_index;
        if (output_offset < first_output || output_offset >= last_output)
            return;
        out.data()[output_offset] = value;
    }

    __device__ void operator()(size_t tid,
                               int64_t dim,
                               PackedTensorAccessor64<const index_t, 4, RestrictPtrTraits> index,
                               scalar_t value,
                               int64_t first_output,
                               int64_t last_output,
                               PackedTensorAccessor64<scalar_t, 4, RestrictPtrTraits> out)
    {
        const auto tid_0 = tid / (index.size(1) * index.size(2) * index.size(3));
        const auto tid_1 = tid / (index.size(2) * index.size(3)) % index.size(1);
        const auto tid_2 = tid / index.size(3) % index.size(2);
        const auto tid_3 = tid % index.size(3);
        const auto output_index = index[tid_0][tid_1][tid_2][tid_3];
        CUDA_KERNEL_ASSERT(output_index >= 0 && output_index < out.size(dim) && "index out of bounds");
        int64_t output_offset;
        if (dim == 0)
            output_offset = ((output_index * out.size(1) + tid_1) * out.size(2) + tid_2) * out.size(3) + tid_3;
        else if (dim == 1)
            output_offset = ((tid_0 * out.size(1) + output_index) * out.size(2) + tid_2) * out.size(3) + tid_3;
        else if (dim == 2)
            output_offset = ((tid_0 * out.size(1) + tid_1) * out.size(2) + output_index) * out.size(3) + tid_3;
        else
            output_offset = ((tid_0 * out.size(1) + tid_1) * out.size(2) + tid_2) * out.size(3) + output_index;
        if (output_offset < first_output || output_offset >= last_output)
            return;
        out.data()[output_offset] = value;
    }
};

template <typename scalar_t, typename index_t>
struct IndexSelectFunctor
{
    __device__ void operator()(size_t tid,
                               PackedTensorAccessor64<const scalar_t, 1, RestrictPtrTraits> self,
                               int64_t dim,
                               PackedTensorAccessor64<const index_t, 1, RestrictPtrTraits> indices,
                               PackedTensorAccessor64<scalar_t, 1, RestrictPtrTraits> out)
    {
        out[tid] = self[indices[tid]];
    }

    __device__ void operator()(size_t tid,
                               PackedTensorAccessor64<const scalar_t, 2, RestrictPtrTraits> self,
                               int64_t dim,
                               PackedTensorAccessor64<const index_t, 1, RestrictPtrTraits> indices,
                               PackedTensorAccessor64<scalar_t, 2, RestrictPtrTraits> out)
    {
        auto index = indices[tid];
        if (dim == 0)
        {
            for (auto j = 0; j < self.size(1); ++j)
            {
                out[tid][j] = self[index][j];
            }
        }
        else if (dim == 1)
        {
            for (auto i = 0; i < self.size(0); ++i)
            {
                out[i][tid] = self[i][index];
            }
        }
    }
};

struct ValidateRepeatsFunctor
{
    __device__ void operator()(size_t repeat_index,
                               const int64_t* __restrict__ cumulative_repeats,
                               int64_t num_repeats,
                               int64_t output_size)
    {
        const int64_t begin = repeat_index == 0 ? 0 : cumulative_repeats[repeat_index - 1];
        const int64_t end = cumulative_repeats[repeat_index];
        CUDA_KERNEL_ASSERT(end >= begin && "repeats can not be negative");
        if (repeat_index + 1 == num_repeats)
            CUDA_KERNEL_ASSERT(end == output_size && "allocated size does not match required size");
    }
};

template <typename scalar_t>
struct RepeatInterleaveFunctor
{
    __device__ void operator()(size_t output_index,
                               const int64_t* __restrict__ cumulative_repeats,
                               int64_t num_repeats,
                               scalar_t* __restrict__ output)
    {
        int64_t left = 0;
        int64_t right = num_repeats;
        while (left < right)
        {
            const int64_t middle = left + (right - left) / 2;
            if (cumulative_repeats[middle] <= static_cast<int64_t>(output_index))
                left = middle + 1;
            else
                right = middle;
        }

        CUDA_KERNEL_ASSERT(left < num_repeats && "allocated size does not match required size");
        output[output_index] = static_cast<scalar_t>(left);
    }
};

} // namespace

Tensor& index_out(const Tensor& self, const c10::List<::std::optional<Tensor>>& indices, Tensor& out)
{
    TORCH_CHECK(out.is_contiguous());

    if (!self.numel() && !out.numel())
        return out;

    StridedOffsetCalculator<1> input_offsets;
    if (out.dim() == 1 && !self.is_contiguous())
    {
        // Full-rank boolean masks produce logical flat indices for inputs of any rank.
        input_offsets = make_strided_offset_calculator<1>(self.sizes(), { self.strides() });
    }

    constexpr int64_t d = 0;

    AT_DISPATCH_V2(
        self.scalar_type(), "index_out", AT_WRAP([&] {
            size_t k = 0;
            for (size_t i = 0; i < indices.size(); ++i)
            {
                if (indices[i].has_value())
                {
                    Tensor index;
                    if (indices[i].value().scalar_type() == kBool)
                    {
                        Tensor mask = indices[i].value().to(self.device());
                        Tensor flat_mask = mask.contiguous().view({ -1 });
                        Tensor range = at::arange(flat_mask.numel(), flat_mask.options().dtype(kLong));
                        index = at::masked_select(range, flat_mask);
                    }
                    else
                    {
                        index = indices[i].value().to(self.device());
                    }

                    TORCH_CHECK(index.dim() == 1, "index_out does not support multidimensional indexing");
                    TORCH_CHECK(index.device() == self.device(), "index_out does not support cross-device indexing");
                    TORCH_CHECK(index.is_contiguous(), "indices are not contiguous");

                    AT_DISPATCH_INDEX_TYPES(index.scalar_type(), "index_out", [&]() -> void {
                        auto index_accessor = index.packed_accessor64<const index_t, 1, RestrictPtrTraits>();

                        parallel_for_each([&](int device, cudaStream_t stream) {
                            int64_t device_offset, device_count;
                            // Keep selected rows on the same Gaussian owners for parameters and moments.
                            // Page-aligning the index array shifts wide output rows by much more than a page.
                            std::tie(device_offset, device_count) = chunk(index.sizes()[d], device);

                            const unsigned int device_num_blocks =
                                ::cuda::ceil_div(device_count * out.stride(0), kNumThreads);
                            if (device_num_blocks)
                            {
                                C10_CUDA_CHECK(cudaMemPrefetchAsync(
                                    index.const_data_ptr<index_t>() + index.stride(0) * device_offset,
                                    device_count * index.stride(0) * sizeof(index_t),
                                    { cudaMemLocationTypeDevice, device }, 0, stream));
                                C10_CUDA_CHECK(cudaMemPrefetchAsync(
                                    out.const_data_ptr<scalar_t>() + out.stride(0) * (device_offset + k),
                                    device_count * out.stride(0) * sizeof(scalar_t),
                                    { cudaMemLocationTypeDevice, device }, 0, stream));

                                if (out.dim() == 1)
                                {
                                    auto out_accessor = out.packed_accessor64<scalar_t, 1, RestrictPtrTraits>();
                                    if (self.is_contiguous())
                                    {
                                        index_out_kernel<<<device_num_blocks, kNumThreads, 0, stream>>>(
                                            self.const_data_ptr<scalar_t>(), device_offset, device_count,
                                            index_accessor, k, out_accessor);
                                    }
                                    else
                                    {
                                        const StridedInputOp<scalar_t> self_accessor{ self.const_data_ptr<scalar_t>(),
                                                                                      input_offsets };
                                        index_out_kernel<<<device_num_blocks, kNumThreads, 0, stream>>>(
                                            self_accessor, device_offset, device_count, index_accessor, k, out_accessor);
                                    }
                                    C10_CUDA_KERNEL_LAUNCH_CHECK();
                                }
                                else if (out.dim() == 2)
                                {
                                    if (self.dim() == 3)
                                    {
                                        auto self_accessor =
                                            self.packed_accessor64<const scalar_t, 3, RestrictPtrTraits>();
                                        auto out_accessor = out.packed_accessor64<scalar_t, 2, RestrictPtrTraits>();
                                        index_out_kernel<<<device_num_blocks, kNumThreads, 0, stream>>>(
                                            self_accessor, device_offset, device_count, index_accessor, k, out_accessor);
                                        C10_CUDA_KERNEL_LAUNCH_CHECK();
                                    }
                                    else if (self.dim() > out.dim())
                                    {
                                        TORCH_CHECK(self.is_contiguous(),
                                                    "index_out only supports non-contiguous inputs up to rank 3");
                                        std::vector<int64_t> view_shape = out.sizes().vec();
                                        view_shape[0] = -1;
                                        auto self_ = self.view(view_shape);
                                        auto self_accessor =
                                            self_.packed_accessor64<const scalar_t, 2, RestrictPtrTraits>();
                                        auto out_accessor = out.packed_accessor64<scalar_t, 2, RestrictPtrTraits>();
                                        index_out_kernel<<<device_num_blocks, kNumThreads, 0, stream>>>(
                                            self_accessor, device_offset, device_count, index_accessor, k, out_accessor);
                                        C10_CUDA_KERNEL_LAUNCH_CHECK();
                                    }
                                    else
                                    {
                                        auto self_accessor =
                                            self.packed_accessor64<const scalar_t, 2, RestrictPtrTraits>();
                                        auto out_accessor = out.packed_accessor64<scalar_t, 2, RestrictPtrTraits>();
                                        index_out_kernel<<<device_num_blocks, kNumThreads, 0, stream>>>(
                                            self_accessor, device_offset, device_count, index_accessor, k, out_accessor);
                                        C10_CUDA_KERNEL_LAUNCH_CHECK();
                                    }
                                }
                                else if (out.dim() == 3)
                                {
                                    auto self_accessor = self.packed_accessor64<const scalar_t, 3, RestrictPtrTraits>();
                                    auto out_accessor = out.packed_accessor64<scalar_t, 3, RestrictPtrTraits>();
                                    index_out_kernel<<<device_num_blocks, kNumThreads, 0, stream>>>(
                                        self_accessor, device_offset, device_count, index_accessor, k, out_accessor);
                                    C10_CUDA_KERNEL_LAUNCH_CHECK();
                                }
                                else
                                {
                                    TORCH_CHECK(false, "Unsupported dimension in index_out");
                                }
                            }
                        });
                    });
                    k += index.sizes()[d];
                }
            }
        }),
        AT_EXPAND(AT_ALL_TYPES));
    return out;
}

Tensor& index_put(
    Tensor& self, const c10::List<::std::optional<Tensor>>& indices, const Tensor& values, bool accumulate, bool unsafe)
{
    TORCH_CHECK(self.is_contiguous(), "self is not contiguous");
    TORCH_CHECK(indices.size() == 1 && indices[0].has_value(), "index_put only supports a single tensor index");

    auto index = indices[0].value().to(self.device());
    TORCH_CHECK(!accumulate || index.scalar_type() == kBool,
                "accumulate = true is only supported for boolean masks in index_put");
    TORCH_CHECK(
        !unsafe || index.scalar_type() == kBool, "unsafe = true is only supported for boolean masks in index_put");
    bool is_scalar = (values.dim() == 0);
    const bool is_cpu_scalar = is_scalar && values.device().is_cpu();

    TORCH_CHECK(index.device() == self.device(), "index_put does not support cross-device indexing");
    TORCH_CHECK(index.is_contiguous(), "indices are not contiguous");
    TORCH_CHECK(is_cpu_scalar || values.device() == self.device(), "index_put expected values to be on ", self.device(),
                " but got ", values.device());
    TORCH_CHECK(values.scalar_type() == self.scalar_type(), "index_put expected values to have dtype ",
                self.scalar_type(), " but got ", values.scalar_type());
    at::assert_no_overlap(self, index);
    at::assert_no_overlap(self, values);

    const bool use_elementwise_mask = index.sizes() == self.sizes() && is_scalar;
    int64_t indexed_dims = 1;
    if (!use_elementwise_mask && index.scalar_type() == kBool)
    {
        TORCH_CHECK(index.dim() >= 1 && index.dim() <= self.dim(),
                    "index_put boolean mask rank must be between 1 and the input rank");
        for (const auto dim : c10::irange(index.dim()))
        {
            TORCH_CHECK(index.size(dim) == self.size(dim), "index_put boolean mask shape ", index.sizes(),
                        " must match the input prefix ", self.sizes().slice(0, index.dim()));
        }
        indexed_dims = index.dim();
        index = at::nonzero(index.view({ -1 })).squeeze(1);
    }

    if (use_elementwise_mask)
    {
        TORCH_CHECK(index.scalar_type() == kBool, "index_put element-wise indexing only supports bool masks");

        AT_DISPATCH_V2(
            self.scalar_type(), "index_put", AT_WRAP([&] {
                if (is_cpu_scalar)
                {
                    if (accumulate)
                        mgpu_launch_func(self.numel(), IndexPutFunctor<scalar_t, true>(), self.data_ptr<scalar_t>(),
                                         index.const_data_ptr<bool>(), values.item<scalar_t>());
                    else
                        mgpu_launch_func(self.numel(), IndexPutFunctor<scalar_t>(), self.data_ptr<scalar_t>(),
                                         index.const_data_ptr<bool>(), values.item<scalar_t>());
                }
                else
                {
                    if (accumulate)
                        mgpu_launch_func(self.numel(), IndexPutFunctor<scalar_t, true>(), self.data_ptr<scalar_t>(),
                                         index.const_data_ptr<bool>(), values.const_data_ptr<scalar_t>());
                    else
                        mgpu_launch_func(self.numel(), IndexPutFunctor<scalar_t>(), self.data_ptr<scalar_t>(),
                                         index.const_data_ptr<bool>(), values.const_data_ptr<scalar_t>());
                }
            }),
            AT_EXPAND(AT_ALL_TYPES));
    }
    else if (self.dim() >= 1 && index.dim() == 1 && (index.scalar_type() == kInt || index.scalar_type() == kLong))
    {
        auto indexed_sizes = self.sizes().slice(indexed_dims).vec();
        indexed_sizes.insert(indexed_sizes.begin(), index.numel());
        TORCH_CHECK(is_scalar || values.sizes() == indexed_sizes, "index_put values must be a scalar or have shape ",
                    indexed_sizes, ", but got ", values.sizes());

        int64_t self_rows = 1;
        for (const auto dim : c10::irange(indexed_dims))
            self_rows *= self.size(dim);
        int64_t row_size = 1;
        for (const auto dim : c10::irange<int64_t>(indexed_dims, self.dim()))
            row_size *= self.size(dim);
        if (!index.numel() || !row_size)
            return self;

        // Balance indexed rows across devices while keeping each row on one
        // device. Page-size alignment leaves the other devices idle for most
        // small and medium scatter operations.
        const int64_t row_alignment = row_size > 0 ? row_size : 1;
        Tensor last_positions = at::empty({ self_rows }, self.options().dtype(kLong));

        AT_DISPATCH_V2(
            self.scalar_type(), "index_put", AT_WRAP([&] {
                AT_DISPATCH_INDEX_TYPES(index.scalar_type(), "index_put", [&]() {
                    constexpr int64_t position_page_elements = kPageSize / sizeof(int64_t);
                    parallel_for_each([&](int device, cudaStream_t stream) {
                        int64_t first_row, device_num_rows;
                        std::tie(first_row, device_num_rows) = aligned_chunk(position_page_elements, self_rows, device);
                        if (!device_num_rows)
                            return;

                        C10_CUDA_CHECK(cudaMemsetAsync(last_positions.data_ptr<int64_t>() + first_row, 0,
                                                       device_num_rows * sizeof(int64_t), stream));
                        const unsigned int num_blocks = ::cuda::ceil_div(index.numel(), kNumThreads);
                        offset_lambda_kernel<<<num_blocks, kNumThreads, 0, stream>>>(
                            index.numel(), 0, IndexPutLastPositionFunctor<index_t>(), self_rows,
                            index.const_data_ptr<index_t>(), first_row, first_row + device_num_rows,
                            last_positions.data_ptr<int64_t>());
                        C10_CUDA_KERNEL_LAUNCH_CHECK();
                    });

                    if (is_cpu_scalar)
                    {
                        if (accumulate)
                            mgpu_launch_func(row_alignment, index.numel() * row_size,
                                             IndexPutRowsFunctor<scalar_t, index_t, true>(), self.data_ptr<scalar_t>(),
                                             self_rows, row_size, index.const_data_ptr<index_t>(),
                                             last_positions.const_data_ptr<int64_t>(), values.item<scalar_t>());
                        else
                            mgpu_launch_func(row_alignment, index.numel() * row_size,
                                             IndexPutRowsFunctor<scalar_t, index_t>(), self.data_ptr<scalar_t>(),
                                             self_rows, row_size, index.const_data_ptr<index_t>(),
                                             last_positions.const_data_ptr<int64_t>(), values.item<scalar_t>());
                    }
                    else if (is_scalar)
                    {
                        if (accumulate)
                            mgpu_launch_func(row_alignment, index.numel() * row_size,
                                             IndexPutRowsFunctor<scalar_t, index_t, true>(), self.data_ptr<scalar_t>(),
                                             self_rows, row_size, index.const_data_ptr<index_t>(),
                                             last_positions.const_data_ptr<int64_t>(),
                                             cub::FutureValue<const scalar_t>(values.const_data_ptr<scalar_t>()));
                        else
                            mgpu_launch_func(row_alignment, index.numel() * row_size,
                                             IndexPutRowsFunctor<scalar_t, index_t>(), self.data_ptr<scalar_t>(),
                                             self_rows, row_size, index.const_data_ptr<index_t>(),
                                             last_positions.const_data_ptr<int64_t>(),
                                             cub::FutureValue<const scalar_t>(values.const_data_ptr<scalar_t>()));
                    }
                    else if (values.is_contiguous())
                    {
                        if (accumulate)
                            mgpu_launch_func(
                                row_alignment, index.numel() * row_size, IndexPutRowsFunctor<scalar_t, index_t, true>(),
                                self.data_ptr<scalar_t>(), self_rows, row_size, index.const_data_ptr<index_t>(),
                                last_positions.const_data_ptr<int64_t>(), values.const_data_ptr<scalar_t>());
                        else
                            mgpu_launch_func(
                                row_alignment, index.numel() * row_size, IndexPutRowsFunctor<scalar_t, index_t>(),
                                self.data_ptr<scalar_t>(), self_rows, row_size, index.const_data_ptr<index_t>(),
                                last_positions.const_data_ptr<int64_t>(), values.const_data_ptr<scalar_t>());
                    }
                    else
                    {
                        const auto value_offsets =
                            make_strided_offset_calculator<1>(values.sizes(), { values.strides() });
                        const StridedInputOp<scalar_t> values_accessor{ values.const_data_ptr<scalar_t>(),
                                                                       value_offsets };
                        if (accumulate)
                            mgpu_launch_func(
                                row_alignment, index.numel() * row_size, IndexPutRowsFunctor<scalar_t, index_t, true>(),
                                self.data_ptr<scalar_t>(), self_rows, row_size, index.const_data_ptr<index_t>(),
                                last_positions.const_data_ptr<int64_t>(), values_accessor);
                        else
                            mgpu_launch_func(
                                row_alignment, index.numel() * row_size, IndexPutRowsFunctor<scalar_t, index_t>(),
                                self.data_ptr<scalar_t>(), self_rows, row_size, index.const_data_ptr<index_t>(),
                                last_positions.const_data_ptr<int64_t>(), values_accessor);
                    }
                });
            }),
            AT_EXPAND(AT_ALL_TYPES));
    }
    else
    {
        TORCH_CHECK(false, "Unsupported configuration for index_put");
    }

    return self;
}

Tensor index_select(const Tensor& self, int64_t dim, const Tensor& index)
{
    return AT_DISPATCH_V2(self.scalar_type(), "index_select", AT_WRAP([&] {
                              auto out_sizes = self.sizes().vec();
                              out_sizes[dim] = index.numel();
                              Tensor out = at::empty(out_sizes, self.options());

                              if (self.dim() == 1)
                              {
                                  mgpu_launch_func(index.numel(), IndexSelectFunctor<scalar_t, int64_t>(),
                                                   self.packed_accessor64<const scalar_t, 1, RestrictPtrTraits>(), dim,
                                                   index.packed_accessor64<const int64_t, 1, RestrictPtrTraits>(),
                                                   out.packed_accessor64<scalar_t, 1, RestrictPtrTraits>());
                              }
                              else if (self.dim() == 2)
                              {
                                  mgpu_launch_func(index.numel(), IndexSelectFunctor<scalar_t, int64_t>(),
                                                   self.packed_accessor64<const scalar_t, 2, RestrictPtrTraits>(), dim,
                                                   index.packed_accessor64<const int64_t, 1, RestrictPtrTraits>(),
                                                   out.packed_accessor64<scalar_t, 2, RestrictPtrTraits>());
                              }
                              else
                              {
                                  TORCH_CHECK(false, "Unsupported configuration for index_select");
                              }
                              return out;
                          }),
                          AT_EXPAND(AT_ALL_TYPES));
}

Tensor repeat_interleave(const Tensor& repeats, std::optional<c10::SymInt> output_size)
{
    TORCH_CHECK(repeats.dim() == 1, "repeat_interleave only supports 1D repeat tensors");
    TORCH_CHECK(repeats.scalar_type() == kInt || repeats.scalar_type() == kLong,
                "repeat_interleave only supports int32 and int64 repeat tensors");

    const int64_t num_repeats = repeats.numel();
    int64_t result_size = 0;
    if (output_size.has_value())
    {
        result_size = output_size->guard_int(__FILE__, __LINE__);
        TORCH_CHECK(result_size >= 0, "repeat_interleave: output_size must be non-negative");
    }

    if (num_repeats == 0)
    {
        TORCH_CHECK(result_size == 0, "allocated size does not match required size");
        return at::empty({ 0 }, repeats.options());
    }

    const Tensor cumulative_repeats = at::cumsum(repeats.contiguous(), 0, kLong);
    if (!output_size.has_value())
    {
        result_size = cumulative_repeats.select(0, num_repeats - 1).item<int64_t>();
        TORCH_CHECK(result_size >= 0, "repeats can not be negative");
    }

    mgpu_launch_func(
        num_repeats, ValidateRepeatsFunctor(), cumulative_repeats.const_data_ptr<int64_t>(), num_repeats, result_size);

    Tensor output = at::empty({ result_size }, repeats.options());
    AT_DISPATCH_V2(repeats.scalar_type(), "repeat_interleave", AT_WRAP([&] {
                       constexpr size_t kAlignment = kPageSize / sizeof(scalar_t);
                       mgpu_launch_func(kAlignment, result_size, RepeatInterleaveFunctor<scalar_t>(),
                                        cumulative_repeats.const_data_ptr<int64_t>(), num_repeats,
                                        output.data_ptr<scalar_t>());
                   }),
                   kInt, kLong);
    return output;
}

Tensor masked_select(const Tensor& self, const Tensor& mask)
{
    TORCH_CHECK(mask.scalar_type() == kBool, "masked_select: expected BoolTensor for mask");
    TORCH_CHECK(mask.device() == self.device(), "masked_select: expected self and mask to be on the same device");

    const auto expanded_size = at::infer_size_dimvector(self.sizes(), mask.sizes());
    const Tensor input = self.expand(expanded_size).contiguous();
    const Tensor flags = mask.expand(expanded_size).contiguous();

    if (!input.numel())
        return at::empty({ 0 }, self.options());

    Tensor num_selected_out = at::empty({ c10::cuda::device_count() }, self.options().dtype(kLong));

    parallel_for_each_async([&](int device, cudaStream_t stream) {
        int64_t device_offset, device_numel;
        std::tie(device_offset, device_numel) = chunk(flags.numel(), device);

        const auto device_flags = flags.const_data_ptr<bool>() + device_offset;

        void* device_temp_storage = nullptr;
        size_t temp_storage_bytes = 0;
        C10_CUDA_CHECK(cub::DeviceReduce::Sum(device_temp_storage, temp_storage_bytes, device_flags,
                                              num_selected_out.data_ptr<int64_t>() + device, device_numel, stream));
        C10_CUDA_CHECK(cudaMallocAsync(&device_temp_storage, temp_storage_bytes, stream));
        C10_CUDA_CHECK(cub::DeviceReduce::Sum(device_temp_storage, temp_storage_bytes, device_flags,
                                              num_selected_out.data_ptr<int64_t>() + device, device_numel, stream));
        C10_CUDA_CHECK(cudaFreeAsync(device_temp_storage, stream));
    });

    int64_t out_size = 0;
    for (const auto device_id : c10::irange(c10::cuda::device_count()))
    {
        c10::cuda::getCurrentCUDAStream(device_id).synchronize();
        out_size += num_selected_out.const_data_ptr<int64_t>()[device_id];
    }
    Tensor out = at::empty({ out_size }, self.options());

    int current_device = cudaInvalidDeviceId;
    C10_CUDA_CHECK(cudaGetDevice(&current_device));

    int64_t out_offset = 0;
    for (const auto device_id : c10::irange(c10::cuda::device_count()))
    {
        C10_CUDA_CHECK(cudaSetDevice(device_id));
        auto stream = c10::cuda::getCurrentCUDAStream(device_id);

        AT_DISPATCH_V2(input.scalar_type(), "masked_select", AT_WRAP([&] {
                           const auto input_ptr = input.const_data_ptr<scalar_t>();
                           const auto flags_ptr = flags.const_data_ptr<bool>();
                           auto output = out.data_ptr<scalar_t>();

                           int64_t device_offset, device_numel;
                           std::tie(device_offset, device_numel) = chunk(input.numel(), device_id);

                           auto device_input = input_ptr + device_offset;
                           auto device_flags = flags_ptr + device_offset;
                           auto device_output = output + out_offset;

                           // This increment needs to happen before the CUB launch in order to avoid a race condition
                           out_offset += num_selected_out.const_data_ptr<int64_t>()[device_id];

                           void* device_temp_storage = nullptr;
                           size_t temp_storage_bytes = 0;
                           C10_CUDA_CHECK(cub::DeviceSelect::Flagged(
                               device_temp_storage, temp_storage_bytes, device_input, device_flags, device_output,
                               num_selected_out.data_ptr<int64_t>() + device_id, device_numel, stream));
                           C10_CUDA_CHECK(cudaMallocAsync(&device_temp_storage, temp_storage_bytes, stream));
                           C10_CUDA_CHECK(cub::DeviceSelect::Flagged(
                               device_temp_storage, temp_storage_bytes, device_input, device_flags, device_output,
                               num_selected_out.data_ptr<int64_t>() + device_id, device_numel, stream));
                           C10_CUDA_CHECK(cudaFreeAsync(device_temp_storage, stream));
                       }),
                       AT_EXPAND(AT_ALL_TYPES));
    }

    merge_streams();

    C10_CUDA_CHECK(cudaSetDevice(current_device));

    return out;
}

Tensor& index_add_out(
    const Tensor& self, int64_t dim, const Tensor& index, const Tensor& source, const Scalar& alpha, Tensor& result)
{
    // TODO: Add support for duplicate indices using atomics
    TORCH_CHECK(self.dim() == 1);
    TORCH_CHECK(dim == 0);
    TORCH_CHECK(index.dim() == 1);
    TORCH_CHECK(source.dim() == 1);
    TORCH_CHECK(result.dim() == 1);

    TORCH_CHECK(index.device() == self.device(), "index_add_out does not support cross-device indexing");
    TORCH_CHECK(index.scalar_type() == kLong, "index_add_out element-wise indexing only supports long masks");

    if (!result.is_same(self))
    {
        result.copy_(self);
    }

    AT_DISPATCH_V2(self.scalar_type(), "index_add_out", AT_WRAP([&] {
                       using index_t = int64_t;
                       mgpu_launch_func(index.numel(), IndexAddFunctor<scalar_t, index_t>(),
                                        self.packed_accessor64<scalar_t, 1, RestrictPtrTraits>(),
                                        index.packed_accessor64<const index_t, 1, RestrictPtrTraits>(),
                                        source.packed_accessor64<const scalar_t, 1, RestrictPtrTraits>(),
                                        alpha.to<scalar_t>());
                   }),
                   AT_EXPAND(AT_FLOATING_TYPES));

    return result;
}

Tensor& scatter_add_out(const Tensor& self, int64_t dim, const Tensor& index, const Tensor& source, Tensor& out)
{
    TORCH_CHECK(self.device() == index.device() && self.device() == source.device() && self.device() == out.device(),
                "Expected self, index, source, and out to be on the same device");
    TORCH_CHECK(self.sizes() == out.sizes(), "Expected self and out to have the same shape");
    TORCH_CHECK(self.scalar_type() == source.scalar_type() && self.scalar_type() == out.scalar_type(),
                "Expected self, source, and out to have the same dtype");
    TORCH_CHECK(index.scalar_type() == kLong, "scatter_add expected index to have dtype int64");
    TORCH_CHECK(index.dim() == self.dim() && source.dim() == self.dim(),
                "Expected index, source, and self to have the same number of dimensions");
    TORCH_CHECK(self.dim() >= 1 && self.dim() <= 4, "scatter_add supports tensors with 1 to 4 dimensions");
    TORCH_CHECK(out.is_contiguous(), "scatter_add requires a contiguous output tensor");

    if (dim < 0)
        dim += self.dim();
    TORCH_CHECK(dim >= 0 && dim < self.dim(), "Dimension out of range");
    for (const auto d : c10::irange(self.dim()))
    {
        TORCH_CHECK(index.size(d) <= source.size(d), "Expected index size to be no larger than source size");
        if (d != dim)
            TORCH_CHECK(index.size(d) <= self.size(d), "Expected index size to be no larger than self size");
    }

    if (!out.is_same(self))
        out.copy_(self);
    if (!index.numel())
        return out;

    AT_DISPATCH_V2(
        self.scalar_type(), "scatter_add_out", AT_WRAP([&] {
            constexpr size_t kAlignment = kPageSize / sizeof(scalar_t);
            // Give each output page one owning GPU so ordinary device-scope atomics are sufficient.
            parallel_for_each([&](int device, cudaStream_t stream) {
                int64_t first_output, device_numel;
                std::tie(first_output, device_numel) = aligned_chunk(kAlignment, out.numel(), device);
                if (!device_numel)
                    return;

                const auto num_blocks = static_cast<unsigned int>(::cuda::ceil_div(index.numel(), kNumThreads));
                if (self.dim() == 1)
                {
                    offset_lambda_kernel<<<num_blocks, kNumThreads, 0, stream>>>(
                        index.numel(), 0, ScatterAddFunctor<scalar_t, int64_t>(), dim,
                        index.packed_accessor64<const int64_t, 1, RestrictPtrTraits>(),
                        source.packed_accessor64<const scalar_t, 1, RestrictPtrTraits>(), first_output,
                        first_output + device_numel, out.packed_accessor64<scalar_t, 1, RestrictPtrTraits>());
                }
                else if (self.dim() == 2)
                {
                    offset_lambda_kernel<<<num_blocks, kNumThreads, 0, stream>>>(
                        index.numel(), 0, ScatterAddFunctor<scalar_t, int64_t>(), dim,
                        index.packed_accessor64<const int64_t, 2, RestrictPtrTraits>(),
                        source.packed_accessor64<const scalar_t, 2, RestrictPtrTraits>(), first_output,
                        first_output + device_numel, out.packed_accessor64<scalar_t, 2, RestrictPtrTraits>());
                }
                else if (self.dim() == 3)
                {
                    offset_lambda_kernel<<<num_blocks, kNumThreads, 0, stream>>>(
                        index.numel(), 0, ScatterAddFunctor<scalar_t, int64_t>(), dim,
                        index.packed_accessor64<const int64_t, 3, RestrictPtrTraits>(),
                        source.packed_accessor64<const scalar_t, 3, RestrictPtrTraits>(), first_output,
                        first_output + device_numel, out.packed_accessor64<scalar_t, 3, RestrictPtrTraits>());
                }
                else
                {
                    offset_lambda_kernel<<<num_blocks, kNumThreads, 0, stream>>>(
                        index.numel(), 0, ScatterAddFunctor<scalar_t, int64_t>(), dim,
                        index.packed_accessor64<const int64_t, 4, RestrictPtrTraits>(),
                        source.packed_accessor64<const scalar_t, 4, RestrictPtrTraits>(), first_output,
                        first_output + device_numel, out.packed_accessor64<scalar_t, 4, RestrictPtrTraits>());
                }
                C10_CUDA_KERNEL_LAUNCH_CHECK();
            });
        }),
        AT_EXPAND(AT_FLOATING_TYPES));
    return out;
}

Tensor& scatter_value_out(const Tensor& self, int64_t dim, const Tensor& index, const Scalar& value, Tensor& out)
{
    TORCH_CHECK(self.device() == index.device() && self.device() == out.device(),
                "Expected self, index, and out to be on the same device");
    TORCH_CHECK(self.sizes() == out.sizes(), "Expected self and out to have the same shape");
    TORCH_CHECK(self.scalar_type() == out.scalar_type(), "Expected self and out to have the same dtype");
    TORCH_CHECK(index.scalar_type() == kLong, "scatter expected index to have dtype int64");
    TORCH_CHECK(index.dim() == self.dim(), "Expected index and self to have the same number of dimensions");
    TORCH_CHECK(self.dim() >= 1 && self.dim() <= 4, "scatter supports tensors with 1 to 4 dimensions");
    TORCH_CHECK(out.is_contiguous(), "scatter requires a contiguous output tensor");

    if (dim < 0)
        dim += self.dim();
    TORCH_CHECK(dim >= 0 && dim < self.dim(), "Dimension out of range");
    for (const auto d : c10::irange(self.dim()))
    {
        if (d != dim)
            TORCH_CHECK(index.size(d) <= self.size(d), "Expected index size to be no larger than self size");
    }

    if (!out.is_same(self))
        out.copy_(self);
    if (!index.numel())
        return out;

    AT_DISPATCH_V2(
        self.scalar_type(), "scatter_value_out", AT_WRAP([&] {
            constexpr size_t kAlignment = kPageSize / sizeof(scalar_t);
            const auto scalar_value = value.to<scalar_t>();
            parallel_for_each([&](int device, cudaStream_t stream) {
                int64_t first_output, device_numel;
                std::tie(first_output, device_numel) = aligned_chunk(kAlignment, out.numel(), device);
                if (!device_numel)
                    return;

                const auto num_blocks = static_cast<unsigned int>(::cuda::ceil_div(index.numel(), kNumThreads));
                if (self.dim() == 1)
                {
                    offset_lambda_kernel<<<num_blocks, kNumThreads, 0, stream>>>(
                        index.numel(), 0, ScatterValueFunctor<scalar_t, int64_t>(), dim,
                        index.packed_accessor64<const int64_t, 1, RestrictPtrTraits>(), scalar_value, first_output,
                        first_output + device_numel, out.packed_accessor64<scalar_t, 1, RestrictPtrTraits>());
                }
                else if (self.dim() == 2)
                {
                    offset_lambda_kernel<<<num_blocks, kNumThreads, 0, stream>>>(
                        index.numel(), 0, ScatterValueFunctor<scalar_t, int64_t>(), dim,
                        index.packed_accessor64<const int64_t, 2, RestrictPtrTraits>(), scalar_value, first_output,
                        first_output + device_numel, out.packed_accessor64<scalar_t, 2, RestrictPtrTraits>());
                }
                else if (self.dim() == 3)
                {
                    offset_lambda_kernel<<<num_blocks, kNumThreads, 0, stream>>>(
                        index.numel(), 0, ScatterValueFunctor<scalar_t, int64_t>(), dim,
                        index.packed_accessor64<const int64_t, 3, RestrictPtrTraits>(), scalar_value, first_output,
                        first_output + device_numel, out.packed_accessor64<scalar_t, 3, RestrictPtrTraits>());
                }
                else
                {
                    offset_lambda_kernel<<<num_blocks, kNumThreads, 0, stream>>>(
                        index.numel(), 0, ScatterValueFunctor<scalar_t, int64_t>(), dim,
                        index.packed_accessor64<const int64_t, 4, RestrictPtrTraits>(), scalar_value, first_output,
                        first_output + device_numel, out.packed_accessor64<scalar_t, 4, RestrictPtrTraits>());
                }
                C10_CUDA_KERNEL_LAUNCH_CHECK();
            });
        }),
        AT_EXPAND(AT_ALL_TYPES), kHalf, kBFloat16, kBool);
    return out;
}

Tensor& gather_out(const Tensor& self, int64_t dim, const Tensor& index, bool sparse_grad, Tensor& out)
{
    TORCH_CHECK(self.is_contiguous());
    TORCH_CHECK(out.is_contiguous());
    TORCH_CHECK(!sparse_grad, "gather_out does not support sparse_grad");
    TORCH_CHECK(self.dim() == index.dim(), "self and index must have the same number of dimensions");
    TORCH_CHECK(index.scalar_type() == kLong, "gather expected index to have dtype int64");
    const auto contiguous_index = index.contiguous();

    if (dim < 0)
    {
        dim += self.dim();
    }

    const int64_t self_dim_size = self.size(dim);
    const int64_t index_dim_size = index.size(dim);
    int64_t inner = 1;
    for (int64_t i = dim + 1; i < self.dim(); ++i)
    {
        inner *= self.size(i);
    }
    const int64_t out_row_size = index_dim_size * inner;
    const int64_t self_row_size = self_dim_size * inner;

    AT_DISPATCH_V2(
        self.scalar_type(), "gather_out", AT_WRAP([&] {
            using index_t = int64_t;
            const auto self_data = self.const_data_ptr<scalar_t>();
            const auto indices_data = contiguous_index.const_data_ptr<index_t>();
            const auto out_data = out.data_ptr<scalar_t>();

            parallel_for_each([&](int device, cudaStream_t stream) {
                int64_t device_offset, device_numel;
                std::tie(device_offset, device_numel) = aligned_chunk(kPageSize / sizeof(scalar_t), out.numel(), device);
                if (device_numel == 0)
                {
                    return;
                }

                C10_CUDA_CHECK(cudaMemPrefetchAsync(out_data + device_offset, device_numel * sizeof(scalar_t),
                                                    { cudaMemLocationTypeDevice, device }, 0, stream));
                C10_CUDA_CHECK(cudaMemPrefetchAsync(indices_data + device_offset, device_numel * sizeof(index_t),
                                                    { cudaMemLocationTypeDevice, device }, 0, stream));

                if (dim > 0)
                {
                    const int64_t outer_offset = device_offset / out_row_size;
                    const int64_t outer_count = (device_offset + device_numel - 1) / out_row_size + 1 - outer_offset;
                    C10_CUDA_CHECK(cudaMemPrefetchAsync(self_data + outer_offset * self_row_size,
                                                        outer_count * self_row_size * sizeof(scalar_t),
                                                        { cudaMemLocationTypeDevice, device }, 0, stream));
                }

                const unsigned int device_num_blocks = ::cuda::ceil_div(device_numel, kNumThreads);
                offset_lambda_kernel<<<device_num_blocks, kNumThreads, 0, stream>>>(
                    device_numel, device_offset, GatherFunctor<scalar_t, index_t>(), self_dim_size, index_dim_size,
                    inner, self_data, indices_data, out_data);
                C10_CUDA_KERNEL_LAUNCH_CHECK();
            });
        }),
        AT_EXPAND(AT_ALL_TYPES));

    return out;
}

TORCH_LIBRARY_IMPL(aten, PrivateUse1, m)
{
    m.impl(TORCH_SELECTIVE_NAME("aten::index.Tensor_out"), TORCH_FN(index_out));
    m.impl(TORCH_SELECTIVE_NAME("aten::_index_put_impl_"), TORCH_FN(index_put));
    m.impl(TORCH_SELECTIVE_NAME("aten::index_select"), TORCH_FN(index_select));
    m.impl(TORCH_SELECTIVE_NAME("aten::repeat_interleave.Tensor"), TORCH_FN(repeat_interleave));
    m.impl(TORCH_SELECTIVE_NAME("aten::masked_select"), TORCH_FN(masked_select));
    m.impl(TORCH_SELECTIVE_NAME("aten::index_add.out"), TORCH_FN(index_add_out));
    m.impl(TORCH_SELECTIVE_NAME("aten::scatter_add.out"), TORCH_FN(scatter_add_out));
    m.impl(TORCH_SELECTIVE_NAME("aten::scatter.value_out"), TORCH_FN(scatter_value_out));
    m.impl(TORCH_SELECTIVE_NAME("aten::gather.out"), TORCH_FN(gather_out));
}

} // namespace dgx

} // namespace at::native
