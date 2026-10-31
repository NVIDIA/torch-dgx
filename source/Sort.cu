/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
#include "CompatibilityUtils.h"
#include "KernelLaunchUtils.cuh"

#include <c10/cuda/CUDAStream.h>
#include <c10/util/ScopeExit.h>

#include <ATen/MemoryOverlap.h>
#include <ATen/ops/empty.h>

#include <torch/library.h>

#include <cub/cub.cuh>

#include <algorithm>
#include <cstdint>
#include <vector>

namespace at::native
{

namespace dgx
{

namespace
{

constexpr int kSamplesPerDevice = 4096;

struct KeyIndexPair
{
    int64_t key;
    int64_t index;
};

__host__ __device__ bool key_index_less(KeyIndexPair left, KeyIndexPair right)
{
    return left.key < right.key || (left.key == right.key && left.index < right.index);
}

__device__ uint64_t sample_hash(uint64_t value)
{
    value = (value ^ (value >> 30)) * UINT64_C(0xbf58476d1ce4e5b9);
    value = (value ^ (value >> 27)) * UINT64_C(0x94d049bb133111eb);
    return value ^ (value >> 31);
}

__global__ void sample_pairs_kernel(
    const int64_t* input, int64_t offset, int64_t count, KeyIndexPair* samples, int sample_count)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= sample_count)
        return;
    // Stratified jitter avoids sampling just one phase of a periodic input.
    const int64_t width = count / sample_count;
    const int64_t remainder = count % sample_count;
    const int64_t begin = i * width + (i < remainder ? i : remainder);
    const int64_t index = begin + sample_hash(offset + i + 1) % (width + (i < remainder));
    samples[i] = { input[index], offset + index };
}

__global__ void assign_partitions_kernel(const int64_t* input,
                                         int64_t offset,
                                         int64_t count,
                                         const KeyIndexPair* splitters,
                                         int partition_count,
                                         int local_index_bits,
                                         uint64_t* routes,
                                         unsigned long long* bucket_counts)
{
    extern __shared__ unsigned int local_counts[];
    for (int i = threadIdx.x; i < partition_count; i += blockDim.x)
        local_counts[i] = 0;
    __syncthreads();

    const int64_t stride = int64_t(blockDim.x) * gridDim.x;
    for (int64_t i = int64_t(blockIdx.x) * blockDim.x + threadIdx.x; i < count; i += stride)
    {
        const KeyIndexPair item{ input[i], offset + i };
        int begin = 0, end = partition_count - 1;
        while (begin < end)
        {
            const int middle = (begin + end) / 2;
            if (key_index_less(item, splitters[middle]))
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

__global__ void pack_partition_pairs_kernel(const int64_t* input,
                                            int64_t offset,
                                            int64_t count,
                                            const uint64_t* routes,
                                            uint64_t local_index_mask,
                                            int64_t* keys,
                                            int64_t* indices)
{
    const int64_t stride = int64_t(blockDim.x) * gridDim.x;
    for (int64_t i = int64_t(blockIdx.x) * blockDim.x + threadIdx.x; i < count; i += stride)
    {
        const int64_t index = int64_t(routes[i] & local_index_mask);
        keys[i] = input[index];
        indices[i] = offset + index;
    }
}

int sort_grid_size(int64_t count)
{
    return static_cast<int>(std::min<int64_t>(::cuda::ceil_div(count, int64_t(kNumThreads) * 4), 4096));
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

void radix_sort_pairs(const int64_t* keys_in,
                      int64_t* keys_out,
                      const int64_t* indices_in,
                      int64_t* indices_out,
                      int64_t count,
                      cudaStream_t stream)
{
    size_t scratch_size = 0;
    void* scratch_data = nullptr;
    C10_CUDA_CHECK(cub::DeviceRadixSort::SortPairs(
        scratch_data, scratch_size, keys_in, keys_out, indices_in, indices_out, count, 0, 64, stream));
    C10_CUDA_CHECK(cudaMallocAsync(&scratch_data, scratch_size, stream));
    auto release = c10::make_scope_exit([&] { C10_CUDA_CHECK_WARN(cudaFreeAsync(scratch_data, stream)); });
    C10_CUDA_CHECK(cub::DeviceRadixSort::SortPairs(
        scratch_data, scratch_size, keys_in, keys_out, indices_in, indices_out, count, 0, 64, stream));
}

void prefetch_sort_outputs(int64_t* keys, int64_t* indices, int64_t count, int device, cudaStream_t stream)
{
    const cudaMemLocation location{ cudaMemLocationTypeDevice, device };
    C10_CUDA_CHECK(memPrefetchAsync(keys, count * sizeof(int64_t), location, 0, stream));
    C10_CUDA_CHECK(memPrefetchAsync(indices, count * sizeof(int64_t), location, 0, stream));
}

struct DeviceSortState
{
    int64_t input_offset = 0;
    int64_t input_count = 0;
    int64_t sample_offset = 0;
    int sample_count = 0;
    int64_t* send_keys = nullptr;
    int64_t* send_indices = nullptr;
    int64_t* received_keys = nullptr;
    int64_t* received_indices = nullptr;
    cudaEvent_t metadata_ready = nullptr;
    cudaEvent_t buffers_ready = nullptr;
    cudaEvent_t exchange_done = nullptr;
    cudaStream_t exchange_stream = nullptr;
};

void partition_sort(const int64_t* input, int64_t* keys_out, int64_t* indices_out, int64_t count)
{
    std::vector<DeviceSortState> device_states(c10::cuda::device_count());
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
        std::tie(state.input_offset, state.input_count) = chunk(count, device);
        state.sample_offset = total_samples;
        state.sample_count = static_cast<int>(std::min<int64_t>(state.input_count, kSamplesPerDevice));
        total_samples += state.sample_count;
    }

    const auto host_options = TensorOptions().device(kCPU).dtype(kLong).pinned_memory(true);
    samples = at::empty({ total_samples, 2 }, host_options);
    auto* host_samples = reinterpret_cast<KeyIndexPair*>(samples.data_ptr<int64_t>());
    for (const auto device : c10::irange(c10::cuda::device_count()))
    {
        auto& state = device_states[device];
        if (!state.input_count)
            continue;
        C10_CUDA_CHECK(cudaSetDevice(device));
        auto stream = c10::cuda::getCurrentCUDAStream(device);
        C10_CUDA_CHECK(memPrefetchAsync(input + state.input_offset, state.input_count * sizeof(int64_t),
                                        { cudaMemLocationTypeDevice, device }, 0, stream));
        // initMemPool already configures peer access for the default device pools.
        KeyIndexPair* device_samples = nullptr;
        C10_CUDA_CHECK(cudaMallocAsync(&device_samples, state.sample_count * sizeof(KeyIndexPair), stream));
        auto release = c10::make_scope_exit([&] { C10_CUDA_CHECK_WARN(cudaFreeAsync(device_samples, stream)); });
        sample_pairs_kernel<<<::cuda::ceil_div(state.sample_count, int(kNumThreads)), kNumThreads, 0, stream>>>(
            input + state.input_offset, state.input_offset, state.input_count, device_samples, state.sample_count);
        C10_CUDA_KERNEL_LAUNCH_CHECK();
        C10_CUDA_CHECK(cudaMemcpyAsync(host_samples + state.sample_offset, device_samples,
                                       state.sample_count * sizeof(KeyIndexPair), cudaMemcpyDeviceToHost, stream));
        C10_CUDA_CHECK(cudaEventCreateWithFlags(&state.metadata_ready, cudaEventDisableTiming));
        C10_CUDA_CHECK(cudaEventRecord(state.metadata_ready, stream));
    }
    for (auto& state : device_states)
        if (state.input_count)
            C10_CUDA_CHECK(cudaEventSynchronize(state.metadata_ready));

    // Quantiles of (key, original index) also distribute a run of identical keys.
    // Counts below determine the exact variable-sized output ranges; sampling
    // affects load balance only, never the global ordering or permutation.
    std::sort(host_samples, host_samples + total_samples, key_index_less);
    splitters = at::empty({ c10::cuda::device_count() - 1, 2 }, host_options);
    auto* host_splitters = reinterpret_cast<KeyIndexPair*>(splitters.data_ptr<int64_t>());
    for (const auto device : c10::irange(1, c10::cuda::device_count()))
        host_splitters[device - 1] = host_samples[total_samples * device / c10::cuda::device_count()];

    bucket_counts = at::empty({ c10::cuda::device_count(), c10::cuda::device_count() }, host_options);
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
        KeyIndexPair* device_splitters = nullptr;
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
        C10_CUDA_CHECK(cudaMallocAsync(&device_splitters, (c10::cuda::device_count() - 1) * sizeof(KeyIndexPair), stream));
        C10_CUDA_CHECK(
            cudaMallocAsync(&device_bucket_counts, c10::cuda::device_count() * sizeof(unsigned long long), stream));
        C10_CUDA_CHECK(cudaMallocAsync(&partition_routes, state.input_count * sizeof(uint64_t), stream));
        C10_CUDA_CHECK(cudaMallocAsync(&grouped_routes, state.input_count * sizeof(uint64_t), stream));
        C10_CUDA_CHECK(cudaMemcpyAsync(device_splitters, host_splitters,
                                       (c10::cuda::device_count() - 1) * sizeof(KeyIndexPair), cudaMemcpyHostToDevice,
                                       stream));
        C10_CUDA_CHECK(
            cudaMemsetAsync(device_bucket_counts, 0, c10::cuda::device_count() * sizeof(unsigned long long), stream));
        assign_partitions_kernel<<<sort_grid_size(state.input_count), kNumThreads,
                                   c10::cuda::device_count() * sizeof(unsigned int), stream>>>(
            input + state.input_offset, state.input_offset, state.input_count, device_splitters,
            c10::cuda::device_count(), local_index_bits, partition_routes, device_bucket_counts);
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
        C10_CUDA_CHECK(cudaMallocAsync(&state.send_keys, state.input_count * sizeof(int64_t), stream));
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
            C10_CUDA_CHECK(cudaMallocAsync(&state.received_keys, received_count * sizeof(int64_t), stream));
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
                                               segment_count * sizeof(int64_t), stream));
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
                         indices_out + output_offset, received_count, stream);
    }
    merge_streams();
    synchronize_on_error.release();
}

} // namespace

std::tuple<Tensor&, Tensor&> sort_out(
    const Tensor& self, std::optional<bool> stable, int64_t dim, bool descending, Tensor& values, Tensor& indices)
{
    TORCH_CHECK(!stable.value_or(false), "DGX sort does not support stable=True");
    TORCH_CHECK(self.dim() == 1 && (dim == -1 || dim == 0), "DGX sort supports one-dimensional tensors");
    TORCH_CHECK(!descending, "DGX sort supports ascending order only");
    TORCH_CHECK(self.scalar_type() == kLong && values.scalar_type() == kLong && indices.scalar_type() == kLong,
                "DGX sort requires int64 input, values, and indices");
    TORCH_CHECK(values.device() == self.device() && indices.device() == self.device(),
                "DGX sort outputs must be on the input device");
    values.resize_(self.sizes());
    indices.resize_(self.sizes());
    assert_no_internal_overlap(values);
    assert_no_internal_overlap(indices);
    assert_no_overlap(values, indices);
    if (!self.numel())
        return std::forward_as_tuple(values, indices);

    int current_device = cudaInvalidDeviceId;
    C10_CUDA_CHECK(cudaGetDevice(&current_device));
    auto restore_device = c10::make_scope_exit([&] { C10_CUDA_CHECK_WARN(cudaSetDevice(current_device)); });
    auto input = self.contiguous();
    auto output_values = values.is_contiguous() ? values : at::empty(self.sizes(), values.options());
    auto output_indices = indices.is_contiguous() ? indices : at::empty(self.sizes(), indices.options());
    partition_sort(input.const_data_ptr<int64_t>(), output_values.data_ptr<int64_t>(),
                   output_indices.data_ptr<int64_t>(), self.numel());
    if (!values.is_contiguous())
        values.copy_(output_values);
    if (!indices.is_contiguous())
        indices.copy_(output_indices);
    return std::forward_as_tuple(values, indices);
}

TORCH_LIBRARY_IMPL(aten, PrivateUse1, m)
{
    m.impl(TORCH_SELECTIVE_NAME("aten::sort.values_stable"), TORCH_FN(sort_out));
}

} // namespace dgx

} // namespace at::native
