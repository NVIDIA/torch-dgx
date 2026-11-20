/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
#include "SortUtils.cuh"

#include <c10/core/WrapDimMinimal.h>
#include <c10/cuda/CUDAStream.h>
#include <c10/util/ScopeExit.h>

#include <ATen/Dispatch_v2.h>
#include <ATen/MemoryOverlap.h>
#include <ATen/ops/empty.h>

#include <torch/library.h>

#include <cub/device/device_radix_sort.cuh>

#include <algorithm>
#include <cstdint>
#include <type_traits>
#include <vector>

namespace at::native
{

namespace dgx
{

namespace
{

constexpr int kSamplesPerDevice = 4096;

template <typename key_t>
struct KeyIndexPair
{
    key_t key;
    int64_t index;
};

template <typename key_t>
__host__ __device__ bool key_index_less(KeyIndexPair<key_t> left, KeyIndexPair<key_t> right, bool descending)
{
    if (left.key != right.key)
        return descending ? left.key > right.key : left.key < right.key;
    return left.index < right.index;
}

__device__ uint64_t sample_hash(uint64_t value)
{
    value = (value ^ (value >> 30)) * UINT64_C(0xbf58476d1ce4e5b9);
    value = (value ^ (value >> 27)) * UINT64_C(0x94d049bb133111eb);
    return value ^ (value >> 31);
}

template <typename scalar_t>
__global__ void sample_pairs_kernel(const scalar_t* input,
                                    int64_t offset,
                                    int64_t count,
                                    KeyIndexPair<typename SortKeyTraits<scalar_t>::key_t>* samples,
                                    int sample_count)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= sample_count)
        return;
    // Stratified jitter avoids sampling just one phase of a periodic input.
    const int64_t width = count / sample_count;
    const int64_t remainder = count % sample_count;
    const int64_t begin = i * width + (i < remainder ? i : remainder);
    const int64_t index = begin + sample_hash(offset + i + 1) % (width + (i < remainder));
    samples[i] = { SortKeyTraits<scalar_t>::encode(input[index]), offset + index };
}

template <typename scalar_t>
__global__ void assign_partitions_kernel(const scalar_t* input,
                                         int64_t offset,
                                         int64_t count,
                                         const KeyIndexPair<typename SortKeyTraits<scalar_t>::key_t>* splitters,
                                         int partition_count,
                                         bool descending,
                                         int local_index_bits,
                                         uint64_t* routes,
                                         unsigned long long* bucket_counts)
{
    using key_t = typename SortKeyTraits<scalar_t>::key_t;
    extern __shared__ unsigned int local_counts[];
    for (int i = threadIdx.x; i < partition_count; i += blockDim.x)
        local_counts[i] = 0;
    __syncthreads();

    const int64_t stride = int64_t(blockDim.x) * gridDim.x;
    for (int64_t i = int64_t(blockIdx.x) * blockDim.x + threadIdx.x; i < count; i += stride)
    {
        const KeyIndexPair<key_t> item{ SortKeyTraits<scalar_t>::encode(input[i]), offset + i };
        int begin = 0, end = partition_count - 1;
        while (begin < end)
        {
            const int middle = (begin + end) / 2;
            if (key_index_less(item, splitters[middle], descending))
                end = middle;
            else
                begin = middle + 1;
        }
        routes[i] = (uint64_t(begin) << local_index_bits) | uint64_t(i);
        const unsigned int peers = __match_any_sync(__activemask(), begin);
        if ((threadIdx.x & 31) == __ffs(peers) - 1)
            atomicAdd(local_counts + begin, __popc(peers));
    }
    __syncthreads();
    for (int i = threadIdx.x; i < partition_count; i += blockDim.x)
        if (local_counts[i])
            atomicAdd(bucket_counts + i, static_cast<unsigned long long>(local_counts[i]));
}

template <typename scalar_t>
__global__ void pack_partition_pairs_kernel(const scalar_t* input,
                                            int64_t offset,
                                            int64_t count,
                                            const uint64_t* routes,
                                            uint64_t local_index_mask,
                                            typename SortKeyTraits<scalar_t>::key_t* keys,
                                            int64_t* indices)
{
    const int64_t stride = int64_t(blockDim.x) * gridDim.x;
    for (int64_t i = int64_t(blockIdx.x) * blockDim.x + threadIdx.x; i < count; i += stride)
    {
        const int64_t index = int64_t(routes[i] & local_index_mask);
        keys[i] = SortKeyTraits<scalar_t>::encode(input[index]);
        indices[i] = offset + index;
    }
}

int bit_width(uint64_t maximum)
{
    int bits = 0;
    while (maximum)
    {
        ++bits;
        maximum >>= 1;
    }
    return bits;
}

template <typename key_t>
struct DeviceSortState
{
    int64_t input_offset = 0;
    int64_t input_count = 0;
    int64_t sample_offset = 0;
    int sample_count = 0;
    key_t* send_keys = nullptr;
    int64_t* send_indices = nullptr;
    key_t* received_keys = nullptr;
    int64_t* received_indices = nullptr;
    cudaEvent_t metadata_ready = nullptr;
    cudaEvent_t buffers_ready = nullptr;
    cudaEvent_t exchange_done = nullptr;
    cudaStream_t exchange_stream = nullptr;
};

template <typename scalar_t>
void partition_sort(const scalar_t* input,
                    typename SortKeyTraits<scalar_t>::key_t* keys_out,
                    int64_t* indices_out,
                    int64_t count,
                    bool descending)
{
    using key_t = typename SortKeyTraits<scalar_t>::key_t;
    using pair_t = KeyIndexPair<key_t>;
    std::vector<DeviceSortState<key_t>> device_states(c10::cuda::device_count());
    Tensor samples, splitters, bucket_counts;
    auto release_device_resources = c10::make_scope_exit([&] {
        for (const auto device : c10::irange(c10::cuda::device_count()))
        {
            C10_CUDA_CHECK_WARN(cudaSetDevice(device));
            auto stream = c10::cuda::getCurrentCUDAStream(device);
            auto& state = device_states[device];
            if (state.send_keys)
                C10_CUDA_CHECK_WARN(cudaFreeAsync(state.send_keys, stream));
            if (state.send_indices)
                C10_CUDA_CHECK_WARN(cudaFreeAsync(state.send_indices, stream));
            if (state.received_keys)
                C10_CUDA_CHECK_WARN(cudaFreeAsync(state.received_keys, stream));
            if (state.received_indices)
                C10_CUDA_CHECK_WARN(cudaFreeAsync(state.received_indices, stream));
            if (state.metadata_ready)
                C10_CUDA_CHECK_WARN(cudaEventDestroy(state.metadata_ready));
            if (state.buffers_ready)
                C10_CUDA_CHECK_WARN(cudaEventDestroy(state.buffers_ready));
            if (state.exchange_done)
                C10_CUDA_CHECK_WARN(cudaEventDestroy(state.exchange_done));
            if (state.exchange_stream)
                C10_CUDA_CHECK_WARN(cudaStreamDestroy(state.exchange_stream));
        }
    });
    // Declared after cleanup so outstanding peer reads and metadata copies
    // finish before their storage is released on error.
    auto synchronize_on_error = c10::make_scope_exit([&] {
        for (const auto device : c10::irange(c10::cuda::device_count()))
        {
            C10_CUDA_CHECK_WARN(cudaSetDevice(device));
            auto stream = c10::cuda::getCurrentCUDAStream(device);
            C10_CUDA_CHECK_WARN(cudaStreamSynchronize(stream));
            if (device_states[device].exchange_stream)
                C10_CUDA_CHECK_WARN(cudaStreamSynchronize(device_states[device].exchange_stream));
        }
    });
    int64_t total_samples = 0;
    for (const auto device : c10::irange(c10::cuda::device_count()))
    {
        auto& state = device_states[device];
        std::tie(state.input_offset, state.input_count) = partition::range(count, device);
        state.sample_offset = total_samples;
        state.sample_count = static_cast<int>(std::min<int64_t>(state.input_count, kSamplesPerDevice));
        total_samples += state.sample_count;
    }

    const auto host_options = TensorOptions().device(kCPU).pinned_memory(true);
    // Include any padding between the key and its 64-bit index.
    samples = at::empty({ total_samples * int64_t(sizeof(pair_t)) }, host_options.dtype(kByte));
    auto* host_samples = reinterpret_cast<pair_t*>(samples.data_ptr<uint8_t>());
    for (const auto device : c10::irange(c10::cuda::device_count()))
    {
        auto& state = device_states[device];
        if (!state.input_count)
            continue;
        C10_CUDA_CHECK(cudaSetDevice(device));
        auto stream = c10::cuda::getCurrentCUDAStream(device);
        C10_CUDA_CHECK(memPrefetchAsync(input + state.input_offset, state.input_count * sizeof(scalar_t),
                                        { cudaMemLocationTypeDevice, device }, 0, stream));
        // initMemPool already configures peer access for the default device pools.
        pair_t* device_samples = nullptr;
        C10_CUDA_CHECK(cudaMallocAsync(&device_samples, state.sample_count * sizeof(pair_t), stream));
        auto release = c10::make_scope_exit([&] { C10_CUDA_CHECK_WARN(cudaFreeAsync(device_samples, stream)); });
        sample_pairs_kernel<<<::cuda::ceil_div(state.sample_count, int(kNumThreads)), kNumThreads, 0, stream>>>(
            input + state.input_offset, state.input_offset, state.input_count, device_samples, state.sample_count);
        C10_CUDA_KERNEL_LAUNCH_CHECK();
        C10_CUDA_CHECK(cudaMemcpyAsync(host_samples + state.sample_offset, device_samples,
                                       state.sample_count * sizeof(pair_t), cudaMemcpyDeviceToHost, stream));
        C10_CUDA_CHECK(cudaEventCreateWithFlags(&state.metadata_ready, cudaEventDisableTiming));
        C10_CUDA_CHECK(cudaEventRecord(state.metadata_ready, stream));
    }
    for (auto& state : device_states)
        if (state.input_count)
            C10_CUDA_CHECK(cudaEventSynchronize(state.metadata_ready));

    // Quantiles of (key, original index) also distribute a run of identical keys.
    // Counts below determine the exact variable-sized output ranges; sampling
    // affects load balance only, never the global ordering or permutation.
    std::sort(host_samples, host_samples + total_samples,
              [descending](pair_t left, pair_t right) { return key_index_less(left, right, descending); });
    splitters = at::empty({ (c10::cuda::device_count() - 1) * int64_t(sizeof(pair_t)) }, host_options.dtype(kByte));
    auto* host_splitters = reinterpret_cast<pair_t*>(splitters.data_ptr<uint8_t>());
    for (const auto device : c10::irange(1, c10::cuda::device_count()))
        host_splitters[device - 1] = host_samples[total_samples * device / c10::cuda::device_count()];

    bucket_counts = at::empty({ c10::cuda::device_count(), c10::cuda::device_count() }, host_options.dtype(kLong));
    auto* host_bucket_counts = bucket_counts.data_ptr<int64_t>();
    std::fill_n(host_bucket_counts, c10::cuda::device_count() * c10::cuda::device_count(), int64_t(0));
    const int local_index_bits = bit_width(device_states.front().input_count - 1);
    const int partition_bits = bit_width(c10::cuda::device_count() - 1);
    TORCH_CHECK(local_index_bits + partition_bits <= 64, "DGX sort input is too large to encode partition routes");
    const uint64_t local_index_mask = (uint64_t(1) << local_index_bits) - 1;

    for (const auto device : c10::irange(c10::cuda::device_count()))
    {
        auto& state = device_states[device];
        if (!state.input_count)
            continue;
        C10_CUDA_CHECK(cudaSetDevice(device));
        auto stream = c10::cuda::getCurrentCUDAStream(device);
        pair_t* device_splitters = nullptr;
        unsigned long long* device_bucket_counts = nullptr;
        uint64_t* partition_routes = nullptr;
        uint64_t* grouped_routes = nullptr;
        void* scratch_data = nullptr;
        auto release = c10::make_scope_exit([&] {
            if (device_splitters)
                C10_CUDA_CHECK_WARN(cudaFreeAsync(device_splitters, stream));
            if (device_bucket_counts)
                C10_CUDA_CHECK_WARN(cudaFreeAsync(device_bucket_counts, stream));
            if (partition_routes)
                C10_CUDA_CHECK_WARN(cudaFreeAsync(partition_routes, stream));
            if (grouped_routes)
                C10_CUDA_CHECK_WARN(cudaFreeAsync(grouped_routes, stream));
            if (scratch_data)
                C10_CUDA_CHECK_WARN(cudaFreeAsync(scratch_data, stream));
        });
        C10_CUDA_CHECK(cudaMallocAsync(&device_splitters, (c10::cuda::device_count() - 1) * sizeof(pair_t), stream));
        C10_CUDA_CHECK(
            cudaMallocAsync(&device_bucket_counts, c10::cuda::device_count() * sizeof(unsigned long long), stream));
        C10_CUDA_CHECK(cudaMallocAsync(&partition_routes, state.input_count * sizeof(uint64_t), stream));
        C10_CUDA_CHECK(cudaMallocAsync(&grouped_routes, state.input_count * sizeof(uint64_t), stream));
        C10_CUDA_CHECK(cudaMemcpyAsync(device_splitters, host_splitters,
                                       (c10::cuda::device_count() - 1) * sizeof(pair_t), cudaMemcpyHostToDevice, stream));
        C10_CUDA_CHECK(
            cudaMemsetAsync(device_bucket_counts, 0, c10::cuda::device_count() * sizeof(unsigned long long), stream));
        assign_partitions_kernel<<<sort_grid_size(state.input_count), kNumThreads,
                                   c10::cuda::device_count() * sizeof(unsigned int), stream>>>(
            input + state.input_offset, state.input_offset, state.input_count, device_splitters,
            c10::cuda::device_count(), descending, local_index_bits, partition_routes, device_bucket_counts);
        C10_CUDA_KERNEL_LAUNCH_CHECK();
        C10_CUDA_CHECK(cudaMemcpyAsync(host_bucket_counts + device * c10::cuda::device_count(), device_bucket_counts,
                                       c10::cuda::device_count() * sizeof(int64_t), cudaMemcpyDeviceToHost, stream));
        C10_CUDA_CHECK(cudaEventRecord(state.metadata_ready, stream));

        // A radix pass on just the destination bits packs each source's buckets.
        // The low bits carry the local input index, without a separate arange.
        size_t scratch_size = 0;
        C10_CUDA_CHECK(cub::DeviceRadixSort::SortKeys(scratch_data, scratch_size, partition_routes, grouped_routes,
                                                      state.input_count, local_index_bits,
                                                      local_index_bits + partition_bits, stream));
        C10_CUDA_CHECK(cudaMallocAsync(&scratch_data, scratch_size, stream));
        C10_CUDA_CHECK(cub::DeviceRadixSort::SortKeys(scratch_data, scratch_size, partition_routes, grouped_routes,
                                                      state.input_count, local_index_bits,
                                                      local_index_bits + partition_bits, stream));
        C10_CUDA_CHECK(cudaMallocAsync(&state.send_keys, state.input_count * sizeof(key_t), stream));
        C10_CUDA_CHECK(cudaMallocAsync(&state.send_indices, state.input_count * sizeof(int64_t), stream));
        pack_partition_pairs_kernel<<<sort_grid_size(state.input_count), kNumThreads, 0, stream>>>(
            input + state.input_offset, state.input_offset, state.input_count, grouped_routes, local_index_mask,
            state.send_keys, state.send_indices);
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }
    for (auto& state : device_states)
        if (state.input_count)
            C10_CUDA_CHECK(cudaEventSynchronize(state.metadata_ready));

    std::vector<int64_t> send_offsets(c10::cuda::device_count() * c10::cuda::device_count());
    std::vector<int64_t> output_offsets(c10::cuda::device_count() + 1, 0);
    for (const auto source : c10::irange(c10::cuda::device_count()))
    {
        int64_t offset = 0;
        for (const auto destination : c10::irange(c10::cuda::device_count()))
        {
            const auto index = source * c10::cuda::device_count() + destination;
            send_offsets[index] = offset;
            offset += host_bucket_counts[index];
            output_offsets[destination + 1] += host_bucket_counts[index];
        }
        TORCH_INTERNAL_ASSERT(offset == device_states[source].input_count);
    }
    for (const auto device : c10::irange(c10::cuda::device_count()))
    {
        auto& state = device_states[device];
        const int64_t received_count = output_offsets[device + 1];
        output_offsets[device + 1] += output_offsets[device];
        C10_CUDA_CHECK(cudaSetDevice(device));
        auto stream = c10::cuda::getCurrentCUDAStream(device);
        if (received_count)
        {
            C10_CUDA_CHECK(cudaMallocAsync(&state.received_keys, received_count * sizeof(key_t), stream));
            C10_CUDA_CHECK(cudaMallocAsync(&state.received_indices, received_count * sizeof(int64_t), stream));
            C10_CUDA_CHECK(cudaStreamCreateWithFlags(&state.exchange_stream, cudaStreamNonBlocking));
            C10_CUDA_CHECK(cudaEventCreateWithFlags(&state.exchange_done, cudaEventDisableTiming));
        }
        C10_CUDA_CHECK(cudaEventCreateWithFlags(&state.buffers_ready, cudaEventDisableTiming));
        C10_CUDA_CHECK(cudaEventRecord(state.buffers_ready, stream));
    }
    TORCH_INTERNAL_ASSERT(output_offsets.back() == count);

    // Place output pages while the independent exchange streams move the pairs.
    // Wait for all packing first because out= may alias the original input.
    for (const auto destination : c10::irange(c10::cuda::device_count()))
    {
        const int64_t received_count = output_offsets[destination + 1] - output_offsets[destination];
        if (!received_count)
            continue;
        C10_CUDA_CHECK(cudaSetDevice(destination));
        auto stream = c10::cuda::getCurrentCUDAStream(destination);
        for (const auto& source_state : device_states)
            C10_CUDA_CHECK(cudaStreamWaitEvent(stream, source_state.buffers_ready));
        const auto output_offset = output_offsets[destination];
        prefetch_sort_outputs(keys_out + output_offset, indices_out + output_offset, received_count, destination, stream);
    }

    for (const auto destination : c10::irange(c10::cuda::device_count()))
    {
        auto& state = device_states[destination];
        const int64_t received_count = output_offsets[destination + 1] - output_offsets[destination];
        if (!received_count)
            continue;
        C10_CUDA_CHECK(cudaSetDevice(destination));
        auto stream = state.exchange_stream;
        // These events cover packing and receive-buffer allocation. Nonblocking
        // streams avoid the cross-device barriers of default-stream peer copies.
        for (const auto& source_state : device_states)
            C10_CUDA_CHECK(cudaStreamWaitEvent(stream, source_state.buffers_ready));
        int64_t received_offset = 0;
        for (const auto source : c10::irange(c10::cuda::device_count()))
        {
            const auto index = source * c10::cuda::device_count() + destination;
            const int64_t segment_count = host_bucket_counts[index];
            if (!segment_count)
                continue;
            auto& sender = device_states[source];
            C10_CUDA_CHECK(cudaMemcpyPeerAsync(state.received_keys + received_offset, destination,
                                               sender.send_keys + send_offsets[index], source,
                                               segment_count * sizeof(key_t), stream));
            C10_CUDA_CHECK(cudaMemcpyPeerAsync(state.received_indices + received_offset, destination,
                                               sender.send_indices + send_offsets[index], source,
                                               segment_count * sizeof(int64_t), stream));
            received_offset += segment_count;
        }
        TORCH_INTERNAL_ASSERT(received_offset == received_count);
        C10_CUDA_CHECK(cudaEventRecord(state.exchange_done, stream));
    }

    // Each sort waits only for its own receives and output placement.
    for (const auto destination : c10::irange(c10::cuda::device_count()))
    {
        auto& state = device_states[destination];
        const int64_t received_count = output_offsets[destination + 1] - output_offsets[destination];
        if (!received_count)
            continue;
        C10_CUDA_CHECK(cudaSetDevice(destination));
        auto stream = c10::cuda::getCurrentCUDAStream(destination);
        C10_CUDA_CHECK(cudaStreamWaitEvent(stream, state.exchange_done));
        const auto output_offset = output_offsets[destination];
        radix_sort_pairs(state.received_keys, keys_out + output_offset, state.received_indices,
                         indices_out + output_offset, received_count, descending, stream);
    }
    merge_streams();
    synchronize_on_error.release();
}

template <typename scalar_t>
struct GatherSortedValues
{
    int64_t slice_size;

    __device__ void operator()(int64_t index, const scalar_t* input, const int64_t* indices, scalar_t* values) const
    {
        const int64_t slice_begin = (index / slice_size) * slice_size;
        values[index] = input[slice_begin + indices[index]];
    }
};

} // namespace

std::tuple<Tensor&, Tensor&> sort_out(
    const Tensor& self, std::optional<bool> stable, int64_t dim, bool descending, Tensor& values, Tensor& indices)
{
    // Both radix passes are stable, and peer segments arrive in original-index
    // order. Splitters also break equal-key ties by index across partitions.
    TORCH_CHECK(self.scalar_type() == kLong || self.scalar_type() == kFloat || self.scalar_type() == kDouble,
                "DGX sort supports int64, float32, and float64 inputs");
    TORCH_CHECK(values.scalar_type() == self.scalar_type(), "DGX sort values must have the input dtype");
    TORCH_CHECK(indices.scalar_type() == kLong, "DGX sort indices must have dtype int64");
    TORCH_CHECK(values.device() == self.device() && indices.device() == self.device(),
                "DGX sort outputs must be on the input device");
    dim = c10::maybe_wrap_dim(dim, self.dim());
    values.resize_(self.sizes());
    indices.resize_(self.sizes());
    assert_no_internal_overlap(values);
    assert_no_internal_overlap(indices);
    assert_no_overlap(values, indices);
    if (!self.numel())
        return std::forward_as_tuple(values, indices);
    if (self.dim() == 0 || self.size(dim) == 1)
    {
        values.copy_(self.is_alias_of(values) ? self.clone() : self);
        indices.zero_();
        return std::forward_as_tuple(values, indices);
    }

    // Put the sorting dimension last so every slice is a contiguous row.
    auto input = self.movedim(dim, -1).contiguous();
    auto output_values = values.movedim(dim, -1);
    auto output_indices = indices.movedim(dim, -1);
    const bool copy_values = !output_values.is_contiguous();
    const bool copy_indices = !output_indices.is_contiguous();
    if (copy_values)
        output_values = at::empty(input.sizes(), values.options());
    if (copy_indices)
        output_indices = at::empty(input.sizes(), indices.options());

    const int64_t slice_size = input.size(-1);
    const int64_t slice_count = input.numel() / slice_size;
    // Preserve input needed by later rows or by the floating-point gather.
    if ((slice_count > 1 || input.scalar_type() != kLong) &&
        (input.is_alias_of(output_values) || input.is_alias_of(output_indices)))
        input = input.clone();
    AT_DISPATCH_V2(input.scalar_type(), "sort_out", AT_WRAP([&] {
                       using key_t = typename SortKeyTraits<scalar_t>::key_t;
                       Tensor sorted_keys = output_values;
                       if constexpr (!std::is_same_v<scalar_t, int64_t>)
                           sorted_keys = at::empty(input.sizes(), input.options().dtype<key_t>());

                       // Partition each row independently across all GPUs.
                       for (int64_t slice = 0; slice < slice_count; ++slice)
                       {
                           const int64_t offset = slice * slice_size;
                           partition_sort(input.const_data_ptr<scalar_t>() + offset,
                                          sorted_keys.data_ptr<key_t>() + offset,
                                          output_indices.data_ptr<int64_t>() + offset, slice_size, descending);
                       }

                       if constexpr (!std::is_same_v<scalar_t, int64_t>)
                           mgpu_launch_func(1, input.numel(), GatherSortedValues<scalar_t>{ slice_size },
                                            input.const_data_ptr<scalar_t>(), output_indices.const_data_ptr<int64_t>(),
                                            output_values.data_ptr<scalar_t>());
                   }),
                   kLong, kFloat, kDouble);
    if (copy_values)
        values.copy_(output_values.movedim(-1, dim));
    if (copy_indices)
        indices.copy_(output_indices.movedim(-1, dim));
    return std::forward_as_tuple(values, indices);
}

TORCH_LIBRARY_IMPL(aten, PrivateUse1, m)
{
    m.impl(TORCH_SELECTIVE_NAME("aten::sort.values_stable"), TORCH_FN(sort_out));
}

} // namespace dgx

} // namespace at::native
