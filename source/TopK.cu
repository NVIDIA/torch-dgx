/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
#include <cub/version.cuh>

#if CUB_VERSION >= 300200
#    include "SortUtils.cuh"

#    include <c10/core/WrapDimMinimal.h>

#    include <ATen/Dispatch_v2.h>
#    include <ATen/MemoryOverlap.h>
#    include <ATen/ops/empty.h>

#    include <torch/library.h>

#    include <cub/block/block_radix_sort.cuh>
#    include <cub/device/device_topk.cuh>
#    include <cuda/std/limits>
#    include <cuda/stream_ref>
#    include <thrust/iterator/counting_iterator.h>
#    include <thrust/iterator/transform_iterator.h>

#    include <algorithm>
#    include <cstdint>
#    include <vector>

namespace at::native
{

namespace dgx
{

namespace
{

constexpr int64_t kTopKBlockSliceSize = 4 * kNumThreads;
// Conservative bounds: keep substantial work per GPU and at most one candidate per eight input items.
constexpr int64_t kTopKMinShardSize = 32768;
constexpr int64_t kTopKCandidateReduction = 8;

template <typename scalar_t>
struct TopKKeyOp
{
    using key_t = typename SortKeyTraits<scalar_t>::key_t;

    __device__ key_t operator()(scalar_t value) const
    {
        return SortKeyTraits<scalar_t>::encode(value);
    }
};

template <typename KeyIterator, typename IndexIterator, typename key_t>
void topk_pairs(KeyIterator keys_in,
                IndexIterator indices_in,
                key_t* keys_out,
                int64_t* indices_out,
                int64_t count,
                int64_t k,
                bool largest,
                bool sorted,
                cudaStream_t stream)
{
    // DeviceTopK selects unordered pairs. Sort only the selected k pairs when requested.
    const auto requirements = ::cuda::execution::require(
        ::cuda::execution::determinism::not_guaranteed, ::cuda::execution::output_ordering::unsorted);
    const auto env = ::cuda::std::execution::env{ ::cuda::stream_ref{ stream }, requirements };
    const bool sort_selected = sorted && k > 1;
    key_t* selected_keys = keys_out;
    int64_t* selected_indices = indices_out;
    size_t scratch_size = 0;
    void* scratch_data = nullptr;
    if (sort_selected)
    {
        C10_CUDA_CHECK(cudaMallocAsync(&selected_keys, k * sizeof(key_t), stream));
        C10_CUDA_CHECK(cudaMallocAsync(&selected_indices, k * sizeof(int64_t), stream));
    }
    auto select = [&] {
        if (largest)
            return cub::DeviceTopK::MaxPairs(
                scratch_data, scratch_size, keys_in, selected_keys, indices_in, selected_indices, count, k, env);
        return cub::DeviceTopK::MinPairs(
            scratch_data, scratch_size, keys_in, selected_keys, indices_in, selected_indices, count, k, env);
    };
    C10_CUDA_CHECK(select());
    C10_CUDA_CHECK(cudaMallocAsync(&scratch_data, scratch_size, stream));
    C10_CUDA_CHECK(select());
    C10_CUDA_CHECK(cudaFreeAsync(scratch_data, stream));
    if (sort_selected)
    {
        radix_sort_pairs(selected_keys, keys_out, selected_indices, indices_out, k, largest, stream);
        C10_CUDA_CHECK(cudaFreeAsync(selected_keys, stream));
        C10_CUDA_CHECK(cudaFreeAsync(selected_indices, stream));
    }
}

template <typename scalar_t>
struct GatherTopKValues
{
    int64_t slice_size;
    int64_t k;

    __device__ void operator()(int64_t index, const scalar_t* input, const int64_t* indices, scalar_t* values) const
    {
        const int64_t slice_begin = (index / k) * slice_size;
        values[index] = input[slice_begin + indices[index]];
    }
};

template <typename scalar_t, int ItemsPerThread>
__global__ __launch_bounds__(kNumThreads) void topk_block_kernel(const scalar_t* input,
                                                                 scalar_t* values,
                                                                 int64_t* indices,
                                                                 int64_t slice_count,
                                                                 int64_t slice_size,
                                                                 int64_t k,
                                                                 bool largest)
{
    using key_t = typename SortKeyTraits<scalar_t>::key_t;
    using BlockSort = cub::BlockRadixSort<key_t, kNumThreads, ItemsPerThread, int>;
    __shared__ typename BlockSort::TempStorage temp_storage;
    const key_t padding =
        largest ? ::cuda::std::numeric_limits<key_t>::lowest() : ::cuda::std::numeric_limits<key_t>::max();

    for (int64_t slice = blockIdx.x; slice < slice_count; slice += gridDim.x)
    {
        const auto* slice_input = input + slice * slice_size;
        key_t thread_keys[ItemsPerThread];
        int thread_indices[ItemsPerThread];
#    pragma unroll
        for (int item = 0; item < ItemsPerThread; ++item)
        {
            const int index = threadIdx.x * ItemsPerThread + item;
            thread_keys[item] = index < slice_size ? SortKeyTraits<scalar_t>::encode(slice_input[index]) : padding;
            thread_indices[item] = index;
        }
        // BlockRadixSort is stable: blocked input order keeps valid extreme keys ahead of equal padding.
        if (largest)
            BlockSort(temp_storage).SortDescendingBlockedToStriped(thread_keys, thread_indices);
        else
            BlockSort(temp_storage).SortBlockedToStriped(thread_keys, thread_indices);

#    pragma unroll
        for (int item = 0; item < ItemsPerThread; ++item)
        {
            const int rank = threadIdx.x + item * kNumThreads;
            if (rank < k)
            {
                const int64_t output_offset = slice * k + rank;
                indices[output_offset] = thread_indices[item];
                values[output_offset] = slice_input[thread_indices[item]];
            }
        }
        __syncthreads();
    }
}

template <typename scalar_t>
void topk_slices(const Tensor& input, Tensor& values, Tensor& indices, int64_t k, bool largest, bool sorted)
{
    using key_t = typename SortKeyTraits<scalar_t>::key_t;
    const int64_t slice_size = input.size(-1);
    const int64_t slice_count = input.numel() / slice_size;
    parallel_for_each([&](int device, cudaStream_t stream) {
        int64_t offset, count;
        std::tie(offset, count) = partition::range(slice_count, device);
        if (!count)
            return;
        const auto* local_input = input.const_data_ptr<scalar_t>() + offset * slice_size;
        auto* local_values = values.data_ptr<scalar_t>() + offset * k;
        auto* local_indices = indices.data_ptr<int64_t>() + offset * k;
        const cudaMemLocation location{ cudaMemLocationTypeDevice, device };
        C10_CUDA_CHECK(memPrefetchAsync(local_input, count * slice_size * sizeof(scalar_t), location, 0, stream));
        C10_CUDA_CHECK(memPrefetchAsync(local_values, count * k * sizeof(scalar_t), location, 0, stream));
        C10_CUDA_CHECK(memPrefetchAsync(local_indices, count * k * sizeof(int64_t), location, 0, stream));

        if (slice_size <= kTopKBlockSliceSize)
        {
            // Batch short rows in one launch. Sorted results also satisfy sorted=False.
            const int num_blocks = static_cast<int>(std::min<int64_t>(count, 4096));
            if (slice_size <= kNumThreads)
                topk_block_kernel<scalar_t, 1><<<num_blocks, kNumThreads, 0, stream>>>(
                    local_input, local_values, local_indices, count, slice_size, k, largest);
            else
                topk_block_kernel<scalar_t, 4><<<num_blocks, kNumThreads, 0, stream>>>(
                    local_input, local_values, local_indices, count, slice_size, k, largest);
            C10_CUDA_KERNEL_LAUNCH_CHECK();
            return;
        }

        key_t* selected_keys = nullptr;
        C10_CUDA_CHECK(cudaMallocAsync(&selected_keys, k * sizeof(key_t), stream));

        // Independent slices stay on one GPU; no cross-device merge is needed.
        for (int64_t slice = 0; slice < count; ++slice)
        {
            auto keys = thrust::make_transform_iterator(local_input + slice * slice_size, TopKKeyOp<scalar_t>{});
            topk_pairs(keys, thrust::make_counting_iterator<int64_t>(0), selected_keys, local_indices + slice * k,
                       slice_size, k, largest, sorted, stream);
        }
        C10_CUDA_CHECK(cudaFreeAsync(selected_keys, stream));
        lambda_kernel<<<sort_grid_size(count * k), kNumThreads, 0, stream>>>(
            count * k, GatherTopKValues<scalar_t>{ slice_size, k }, local_input, local_indices, local_values);
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    });
}

struct DeviceTopKState
{
    int64_t slice = 0;
    int64_t input_offset = 0;
    int64_t input_count = 0;
    int64_t candidate_offset = 0;
    int64_t candidate_count = 0;
    int64_t merge_offset = 0;
    int64_t merge_count = 0;
};

template <typename scalar_t>
void topk_split_slices(
    const Tensor& input, Tensor& values, Tensor& indices, int64_t k, bool largest, bool sorted, int max_devices_per_slice)
{
    using key_t = typename SortKeyTraits<scalar_t>::key_t;
    const int64_t slice_size = input.size(-1);
    const int64_t slice_count = input.numel() / slice_size;
    const int num_devices = c10::cuda::device_count();
    const int devices_per_slice = num_devices / slice_count;
    const int extra_devices = num_devices % slice_count;
    std::vector<DeviceTopKState> device_states(num_devices);
    int next_device = 0;
    int64_t candidate_count = 0;

    // Assign disjoint GPU groups to rows, spreading any leftover devices across the first groups.
    for (const auto slice : c10::irange(slice_count))
    {
        const int group_size = std::min(max_devices_per_slice, devices_per_slice + (slice < extra_devices));
        const int merge_device = next_device;
        const int64_t merge_offset = candidate_count;
        const int64_t shard_size = slice_size / group_size;
        const int64_t remainder = slice_size % group_size;
        int64_t input_offset = 0;
        for (const auto shard : c10::irange(group_size))
        {
            auto& state = device_states[next_device++];
            state.slice = slice;
            state.input_offset = input_offset;
            state.input_count = shard_size + (shard < remainder);
            state.candidate_offset = candidate_count;
            state.candidate_count = std::min(k, state.input_count);
            input_offset += state.input_count;
            candidate_count += state.candidate_count;
        }
        device_states[merge_device].merge_offset = merge_offset;
        device_states[merge_device].merge_count = candidate_count - merge_offset;
    }
    Tensor candidate_keys = at::empty({ candidate_count }, input.options().dtype<key_t>());
    Tensor candidate_indices = at::empty({ candidate_count }, indices.options());

    // Every global top-k item is among its shard's local top-k candidates.
    parallel_for_each([&](int device, cudaStream_t stream) {
        const auto& state = device_states[device];
        if (!state.input_count)
            return;
        const auto* local_input = input.const_data_ptr<scalar_t>() + state.slice * slice_size + state.input_offset;
        auto* local_keys = candidate_keys.data_ptr<key_t>() + state.candidate_offset;
        auto* local_indices = candidate_indices.data_ptr<int64_t>() + state.candidate_offset;
        C10_CUDA_CHECK(memPrefetchAsync(
            local_input, state.input_count * sizeof(scalar_t), { cudaMemLocationTypeDevice, device }, 0, stream));
        prefetch_sort_outputs(local_keys, local_indices, state.candidate_count, device, stream);
        auto keys = thrust::make_transform_iterator(local_input, TopKKeyOp<scalar_t>{});
        topk_pairs(keys, thrust::make_counting_iterator<int64_t>(state.input_offset), local_keys, local_indices,
                   state.input_count, state.candidate_count, largest, false, stream);
    });

    // Merge each row on its group's first GPU; different rows merge concurrently.
    parallel_for_each([&](int device, cudaStream_t stream) {
        const auto& state = device_states[device];
        if (!state.merge_count)
            return;
        auto* local_keys = candidate_keys.data_ptr<key_t>() + state.merge_offset;
        auto* local_candidates = candidate_indices.data_ptr<int64_t>() + state.merge_offset;
        auto* local_values = values.data_ptr<scalar_t>() + state.slice * k;
        auto* local_indices = indices.data_ptr<int64_t>() + state.slice * k;
        const auto* local_input = input.const_data_ptr<scalar_t>() + state.slice * slice_size;
        prefetch_sort_outputs(local_keys, local_candidates, state.merge_count, device, stream);
        const cudaMemLocation location{ cudaMemLocationTypeDevice, device };
        C10_CUDA_CHECK(memPrefetchAsync(local_values, k * sizeof(scalar_t), location, 0, stream));
        C10_CUDA_CHECK(memPrefetchAsync(local_indices, k * sizeof(int64_t), location, 0, stream));
        key_t* selected_keys = nullptr;
        C10_CUDA_CHECK(cudaMallocAsync(&selected_keys, k * sizeof(key_t), stream));
        topk_pairs(
            local_keys, local_candidates, selected_keys, local_indices, state.merge_count, k, largest, sorted, stream);
        C10_CUDA_CHECK(cudaFreeAsync(selected_keys, stream));
        lambda_kernel<<<sort_grid_size(k), kNumThreads, 0, stream>>>(
            k, GatherTopKValues<scalar_t>{ slice_size, k }, local_input, local_indices, local_values);
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    });
}

} // namespace

std::tuple<Tensor&, Tensor&> topk_out(
    const Tensor& self, int64_t k, int64_t dim, bool largest, bool sorted, Tensor& values, Tensor& indices)
{
    dim = c10::maybe_wrap_dim(dim, self.dim());
    const int64_t slice_size = self.dim() == 0 ? 1 : self.size(dim);
    TORCH_CHECK(k >= 0 && k <= slice_size, "selected index k out of range");
    TORCH_CHECK(values.scalar_type() == self.scalar_type(), "DGX topk values must have the input dtype");
    TORCH_CHECK(indices.scalar_type() == kLong, "DGX topk indices must have dtype int64");
    TORCH_CHECK(values.device() == self.device() && indices.device() == self.device(),
                "DGX topk outputs must be on the input device");
    TORCH_CHECK(self.scalar_type() == kByte || self.scalar_type() == kChar || self.scalar_type() == kShort ||
                    self.scalar_type() == kInt || self.scalar_type() == kLong || self.scalar_type() == kHalf ||
                    self.scalar_type() == kBFloat16 || self.scalar_type() == kFloat || self.scalar_type() == kDouble,
                "DGX topk supports uint8, signed integers, float16, bfloat16, float32, and float64 inputs");

    auto output_sizes = self.sizes().vec();
    if (self.dim() != 0)
        output_sizes[dim] = k;

    // Snapshot aliased input before resizing out= tensors can change its shape or storage.
    auto input = self.is_alias_of(values) || self.is_alias_of(indices) ? self.clone() : self;
    values.resize_(output_sizes);
    indices.resize_(output_sizes);
    assert_no_internal_overlap(values);
    assert_no_internal_overlap(indices);
    assert_no_overlap(values, indices);
    if (input.dim() == 0)
    {
        values.copy_(input);
        indices.zero_();
        return std::forward_as_tuple(values, indices);
    }
    if (!values.numel())
        return std::forward_as_tuple(values, indices);

    input = input.movedim(dim, -1).contiguous();
    auto output_values = values.movedim(dim, -1);
    auto output_indices = indices.movedim(dim, -1);
    const bool copy_values = !output_values.is_contiguous();
    const bool copy_indices = !output_indices.is_contiguous();
    if (copy_values)
        output_values = at::empty(output_values.sizes(), values.options());
    if (copy_indices)
        output_indices = at::empty(output_indices.sizes(), indices.options());

    const int64_t slice_count = input.numel() / slice_size;
    const int num_devices = c10::cuda::device_count();
    const int max_devices_per_slice = static_cast<int>(std::min<int64_t>(
        num_devices, std::min(slice_size / kTopKMinShardSize, slice_size / k / kTopKCandidateReduction)));
    AT_DISPATCH_V2(input.scalar_type(), "topk_out", AT_WRAP([&] {
                       if (slice_count < num_devices && max_devices_per_slice > 1)
                           topk_split_slices<scalar_t>(
                               input, output_values, output_indices, k, largest, sorted, max_devices_per_slice);
                       else
                           topk_slices<scalar_t>(input, output_values, output_indices, k, largest, sorted);
                   }),
                   AT_EXPAND(AT_ALL_TYPES), kHalf, kBFloat16);
    if (copy_values)
        values.copy_(output_values.movedim(-1, dim));
    if (copy_indices)
        indices.copy_(output_indices.movedim(-1, dim));
    return std::forward_as_tuple(values, indices);
}

TORCH_LIBRARY_IMPL(aten, PrivateUse1, m)
{
    m.impl(TORCH_SELECTIVE_NAME("aten::topk.values"), TORCH_FN(topk_out));
}

} // namespace dgx

} // namespace at::native

#endif
