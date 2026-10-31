/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
#include "KernelLaunchUtils.cuh"
#include "RowReduction.cuh"

#include <c10/cuda/CUDAStream.h>

#include <ATen/Dispatch_v2.h>
#include <ATen/MemoryOverlap.h>
#include <ATen/OpMathType.h>
#include <ATen/native/Resize.h>
#include <ATen/ops/empty.h>

#include <torch/library.h>

#include <cub/device/device_reduce.cuh>
#include <cuda/functional>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/transform_iterator.h>

#include <limits>
#include <optional>
#include <vector>

namespace at::native
{

namespace dgx
{

namespace
{

template <typename T>
struct ArgPair
{
    int64_t index;
    T value;
};

template <typename T, typename Compare>
struct ArgReductionOp
{
    static __host__ __device__ T identity_value()
    {
        const T upper =
            std::numeric_limits<T>::has_infinity ? std::numeric_limits<T>::infinity() : std::numeric_limits<T>::max();
        const T lower = std::numeric_limits<T>::has_infinity ? -std::numeric_limits<T>::infinity() :
                                                               std::numeric_limits<T>::lowest();
        return Compare()(lower, upper) ? upper : lower;
    }

    static __host__ __device__ ArgPair<T> identity()
    {
        return { std::numeric_limits<int64_t>::max(), identity_value() };
    }

    __host__ __device__ ArgPair<T> operator()(const ArgPair<T>& lhs, const ArgPair<T>& rhs) const
    {
        const bool lhs_nan = lhs.value != lhs.value;
        const bool rhs_nan = rhs.value != rhs.value;
        if (lhs_nan != rhs_nan)
        {
            return lhs_nan ? lhs : rhs;
        }
        if (lhs_nan || lhs.value == rhs.value)
        {
            return lhs.index < rhs.index ? lhs : rhs;
        }
        return Compare()(lhs.value, rhs.value) ? lhs : rhs;
    }
};

// `Acc` widens the reduced value, which lets half and bfloat16 inputs reduce through the float
// comparisons. The conversion is exact, so it preserves both the ordering and NaNs.
template <typename T, typename Acc = T>
struct ArgTransform
{
    __device__ ArgPair<Acc> operator()(T value, int64_t index) const
    {
        return { index, static_cast<Acc>(value) };
    }
};

template <typename T>
struct ArgOutput
{
    T* values;
    int64_t* indices;

    template <typename Offsets>
    __device__ void operator()(const Offsets& offsets, const ArgPair<T>& aggregate) const
    {
        values[offsets[1]] = aggregate.value;
        indices[offsets[2]] = aggregate.index;
    }
};

// argmax/argmin discard the extremum itself, so the reduction writes a single output tensor.
template <typename T>
struct ArgIndexOutput
{
    int64_t* indices;

    template <typename Offsets>
    __device__ void operator()(const Offsets& offsets, const ArgPair<T>& aggregate) const
    {
        indices[offsets[1]] = aggregate.index;
    }
};

// Gathers one column of a 2-D tensor by walking the rows: element i maps to base[i * row_stride],
// where `base` already points at the column's first row. `row_stride` is the tensor's stride(0) in
// elements, so this respects transposed/sliced (non-contiguous) inputs.
template <typename T>
struct ColumnGather
{
    const T* base;
    int64_t row_stride;
    __host__ __device__ T operator()(int64_t i) const
    {
        return base[i * row_stride];
    }
};

// Combines the per-GPU (value, local-index) argmin partials into the final per-column result. Devices
// are visited in ascending id (i.e. ascending row ranges) and the comparison is strict, so ties
// resolve to the smallest global row (matches PyTorch).
template <typename scalar_t>
__global__ void arg_min_dim0_merge_kernel(const scalar_t* partial_values,
                                          const int64_t* partial_indices,
                                          scalar_t* out_values,
                                          int64_t* out_indices,
                                          int64_t out_values_stride,
                                          int64_t out_indices_stride,
                                          int64_t num_gpus,
                                          int64_t num_rows,
                                          int64_t num_cols)
{
    const int64_t chunk_size = ::cuda::ceil_div(num_rows, num_gpus);
    for (int64_t c = threadIdx.x + blockIdx.x * blockDim.x; c < num_cols; c += blockDim.x * gridDim.x)
    {
        scalar_t min = scalar_t{};
        int64_t min_index = -1;
        bool found = false;
        for (int64_t device_id = 0; device_id < num_gpus; ++device_id)
        {
            const int64_t chunk_offset = chunk_size * device_id;
            const int64_t offset = chunk_offset < num_rows ? chunk_offset : num_rows;
            const int64_t remaining = num_rows - offset;
            const int64_t count = chunk_size < remaining ? chunk_size : remaining;
            if (count == 0)
                continue;
            const scalar_t value = partial_values[device_id * num_cols + c];
            if (!found || value < min)
            {
                min = value;
                min_index = offset + partial_indices[device_id * num_cols + c];
                found = true;
            }
        }
        out_values[c * out_values_stride] = min;
        out_indices[c * out_indices_stride] = min_index;
    }
}

template <typename scalar_t>
__global__ void arg_max_dim0_merge_kernel(const scalar_t* partial_values,
                                          const int64_t* partial_indices,
                                          scalar_t* out_values,
                                          int64_t* out_indices,
                                          int64_t out_values_stride,
                                          int64_t out_indices_stride,
                                          int64_t num_gpus,
                                          int64_t num_rows,
                                          int64_t num_cols)
{
    const int64_t chunk_size = ::cuda::ceil_div(num_rows, num_gpus);
    for (int64_t c = threadIdx.x + blockIdx.x * blockDim.x; c < num_cols; c += blockDim.x * gridDim.x)
    {
        scalar_t max = scalar_t{};
        int64_t max_index = -1;
        bool found = false;
        for (int64_t device_id = 0; device_id < num_gpus; ++device_id)
        {
            const int64_t chunk_offset = chunk_size * device_id;
            const int64_t offset = chunk_offset < num_rows ? chunk_offset : num_rows;
            const int64_t remaining = num_rows - offset;
            const int64_t count = chunk_size < remaining ? chunk_size : remaining;
            if (count == 0)
                continue;
            const scalar_t value = partial_values[device_id * num_cols + c];
            if (!found || value > max)
            {
                max = value;
                max_index = offset + partial_indices[device_id * num_cols + c];
                found = true;
            }
        }
        out_values[c * out_values_stride] = max;
        out_indices[c * out_indices_stride] = max_index;
    }
}

// Reduces a [rows, cols] tensor along dim 0 into per-column (value, argmin) results. Each GPU runs one
// cub ArgMin per column over its contiguous row-chunk; a small merge kernel then combines the partials.
// Both the host launches and merge kernel derive chunk boundaries from num_rows, avoiding host writes
// to asynchronously allocated device memory.
template <typename scalar_t>
void arg_min_dim0_async(const scalar_t* in,
                        int64_t num_rows,
                        int64_t num_cols,
                        int64_t row_stride,
                        int64_t col_stride,
                        scalar_t* out_values,
                        int64_t* out_indices,
                        int64_t out_values_stride,
                        int64_t out_indices_stride,
                        scalar_t* partial_values,
                        int64_t* partial_indices)
{
    constexpr int merge_device_id = 0;
    std::vector<cudaEvent_t> events(c10::cuda::device_count());
    for (const auto device_id : c10::irange(c10::cuda::device_count()))
    {
        C10_CUDA_CHECK(cudaSetDevice(device_id));
        auto stream = c10::cuda::getCurrentCUDAStream(device_id);

        int64_t offset = 0;
        int64_t count = 0;
        std::tie(offset, count) = chunk(num_rows, device_id);
        if (count > 0)
        {
            scalar_t* device_values = partial_values + device_id * num_cols;
            int64_t* device_indices = partial_indices + device_id * num_cols;
            auto make_iter = [&](int64_t col) {
                ColumnGather<scalar_t> gather{ in + offset * row_stride + col * col_stride, row_stride };
                return thrust::make_transform_iterator(thrust::make_counting_iterator<int64_t>(0), gather);
            };

            size_t scratch_size = 0;
            void* scratch_data = nullptr;
            C10_CUDA_CHECK(cub::DeviceReduce::ArgMin(
                scratch_data, scratch_size, make_iter(0), device_values, device_indices, count, stream));
            C10_CUDA_CHECK(cudaMallocAsync(&scratch_data, scratch_size, stream));
            for (const auto col : c10::irange(num_cols))
            {
                C10_CUDA_CHECK(cub::DeviceReduce::ArgMin(scratch_data, scratch_size, make_iter(col),
                                                         device_values + col, device_indices + col, count, stream));
            }
            C10_CUDA_CHECK(cudaFreeAsync(scratch_data, stream));
        }
    }

    for (const auto device_id : c10::irange(1, c10::cuda::device_count()))
    {
        C10_CUDA_CHECK(cudaSetDevice(device_id));
        auto stream = c10::cuda::getCurrentCUDAStream(device_id);
        C10_CUDA_CHECK(cudaEventCreateWithFlags(&events[device_id], cudaEventDisableTiming));
        C10_CUDA_CHECK(cudaEventRecord(events[device_id], stream));
    }

    cudaEvent_t merge_event = 0;
    C10_CUDA_CHECK(cudaSetDevice(merge_device_id));
    C10_CUDA_CHECK(cudaEventCreateWithFlags(&merge_event, cudaEventDisableTiming));
    auto merge_stream = c10::cuda::getCurrentCUDAStream(merge_device_id);
    for (const auto device_id : c10::irange(1, c10::cuda::device_count()))
    {
        C10_CUDA_CHECK(cudaStreamWaitEvent(merge_stream, events[device_id]));
    }

    arg_min_dim0_merge_kernel<scalar_t><<<1, kNumThreads, 0, merge_stream>>>(
        partial_values, partial_indices, out_values, out_indices, out_values_stride, out_indices_stride,
        c10::cuda::device_count(), num_rows, num_cols);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    C10_CUDA_CHECK(cudaEventRecord(merge_event, merge_stream));

    for (const auto device_id : c10::irange(1, c10::cuda::device_count()))
    {
        C10_CUDA_CHECK(cudaSetDevice(device_id));
        auto stream = c10::cuda::getCurrentCUDAStream(device_id);
        C10_CUDA_CHECK(cudaStreamWaitEvent(stream, merge_event));
    }

    for (const auto device_id : c10::irange(1, c10::cuda::device_count()))
    {
        C10_CUDA_CHECK(cudaEventDestroy(events[device_id]));
    }
    C10_CUDA_CHECK(cudaEventDestroy(merge_event));
}

// Reduces a [rows, cols] tensor along dim 0 into per-column (value, argmax) results. Each GPU runs one
// cub ArgMax per column over its contiguous row-chunk; a small merge kernel then combines the partials.
// Both the host launches and merge kernel derive chunk boundaries from num_rows, avoiding host writes
// to asynchronously allocated device memory.
template <typename scalar_t>
void arg_max_dim0_async(const scalar_t* in,
                        int64_t num_rows,
                        int64_t num_cols,
                        int64_t row_stride,
                        int64_t col_stride,
                        scalar_t* out_values,
                        int64_t* out_indices,
                        int64_t out_values_stride,
                        int64_t out_indices_stride,
                        scalar_t* partial_values,
                        int64_t* partial_indices)
{
    constexpr int merge_device_id = 0;
    std::vector<cudaEvent_t> events(c10::cuda::device_count());
    for (const auto device_id : c10::irange(c10::cuda::device_count()))
    {
        C10_CUDA_CHECK(cudaSetDevice(device_id));
        auto stream = c10::cuda::getCurrentCUDAStream(device_id);

        int64_t offset = 0;
        int64_t count = 0;
        std::tie(offset, count) = chunk(num_rows, device_id);
        if (count > 0)
        {
            scalar_t* device_values = partial_values + device_id * num_cols;
            int64_t* device_indices = partial_indices + device_id * num_cols;
            auto make_iter = [&](int64_t col) {
                ColumnGather<scalar_t> gather{ in + offset * row_stride + col * col_stride, row_stride };
                return thrust::make_transform_iterator(thrust::make_counting_iterator<int64_t>(0), gather);
            };

            size_t scratch_size = 0;
            void* scratch_data = nullptr;
            C10_CUDA_CHECK(cub::DeviceReduce::ArgMax(
                scratch_data, scratch_size, make_iter(0), device_values, device_indices, count, stream));
            C10_CUDA_CHECK(cudaMallocAsync(&scratch_data, scratch_size, stream));
            for (const auto col : c10::irange(num_cols))
            {
                C10_CUDA_CHECK(cub::DeviceReduce::ArgMax(scratch_data, scratch_size, make_iter(col),
                                                         device_values + col, device_indices + col, count, stream));
            }
            C10_CUDA_CHECK(cudaFreeAsync(scratch_data, stream));
        }
    }

    for (const auto device_id : c10::irange(1, c10::cuda::device_count()))
    {
        C10_CUDA_CHECK(cudaSetDevice(device_id));
        auto stream = c10::cuda::getCurrentCUDAStream(device_id);
        C10_CUDA_CHECK(cudaEventCreateWithFlags(&events[device_id], cudaEventDisableTiming));
        C10_CUDA_CHECK(cudaEventRecord(events[device_id], stream));
    }

    cudaEvent_t merge_event = 0;
    C10_CUDA_CHECK(cudaSetDevice(merge_device_id));
    C10_CUDA_CHECK(cudaEventCreateWithFlags(&merge_event, cudaEventDisableTiming));
    auto merge_stream = c10::cuda::getCurrentCUDAStream(merge_device_id);
    for (const auto device_id : c10::irange(1, c10::cuda::device_count()))
    {
        C10_CUDA_CHECK(cudaStreamWaitEvent(merge_stream, events[device_id]));
    }

    arg_max_dim0_merge_kernel<scalar_t><<<1, kNumThreads, 0, merge_stream>>>(
        partial_values, partial_indices, out_values, out_indices, out_values_stride, out_indices_stride,
        c10::cuda::device_count(), num_rows, num_cols);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    C10_CUDA_CHECK(cudaEventRecord(merge_event, merge_stream));

    for (const auto device_id : c10::irange(1, c10::cuda::device_count()))
    {
        C10_CUDA_CHECK(cudaSetDevice(device_id));
        auto stream = c10::cuda::getCurrentCUDAStream(device_id);
        C10_CUDA_CHECK(cudaStreamWaitEvent(stream, merge_event));
    }

    for (const auto device_id : c10::irange(1, c10::cuda::device_count()))
    {
        C10_CUDA_CHECK(cudaEventDestroy(events[device_id]));
    }
    C10_CUDA_CHECK(cudaEventDestroy(merge_event));
}

// Column-parallel dim-0 reduction: one thread per output column, each scanning all rows. This is the
// fallback for wide inputs (many columns), where it saturates the GPU in a single launch. It reads
// through PackedTensorAccessor64, so it respects non-contiguous strides.
template <typename T>
struct MinDim0Functor
{
    __device__ void operator()(size_t tid,
                               const PackedTensorAccessor64<T, 2> self,
                               PackedTensorAccessor64<T, 1> out,
                               PackedTensorAccessor64<int64_t, 1> out_values)
    {
        T min = self[0][tid];
        int64_t min_value = 0;
        for (auto i = 1; i < self.size(0); ++i)
        {
            if (self[i][tid] < min)
            {
                min = self[i][tid];
                min_value = i;
            }
        }
        out[tid] = min;
        out_values[tid] = min_value;
    }

    __device__ void operator()(size_t tid,
                               const PackedTensorAccessor64<T, 2> self,
                               PackedTensorAccessor64<T, 2> out,
                               PackedTensorAccessor64<int64_t, 2> out_values)
    {
        T min = self[0][tid];
        int64_t min_value = 0;
        for (auto i = 1; i < self.size(0); ++i)
        {
            if (self[i][tid] < min)
            {
                min = self[i][tid];
                min_value = i;
            }
        }
        out[0][tid] = min;
        out_values[0][tid] = min_value;
    }
};

template <typename T>
struct MaxDim0Functor
{
    __device__ void operator()(size_t tid,
                               const PackedTensorAccessor64<T, 2> self,
                               PackedTensorAccessor64<T, 1> out,
                               PackedTensorAccessor64<int64_t, 1> out_values)
    {
        T max = self[0][tid];
        int64_t max_value = 0;
        for (auto i = 1; i < self.size(0); ++i)
        {
            if (self[i][tid] > max)
            {
                max = self[i][tid];
                max_value = i;
            }
        }
        out[tid] = max;
        out_values[tid] = max_value;
    }

    __device__ void operator()(size_t tid,
                               const PackedTensorAccessor64<T, 2> self,
                               PackedTensorAccessor64<T, 2> out,
                               PackedTensorAccessor64<int64_t, 2> out_values)
    {
        T max = self[0][tid];
        int64_t max_value = 0;
        for (auto i = 1; i < self.size(0); ++i)
        {
            if (self[i][tid] > max)
            {
                max = self[i][tid];
                max_value = i;
            }
        }
        out[0][tid] = max;
        out_values[0][tid] = max_value;
    }
};

// The cub argreduce path issues one reduction per column per GPU, so it only pays off when the tensor
// is narrow (few columns) and tall. Above this column count the column-parallel functor path — a single
// launch with one thread per column — is faster, so it is used as the fallback for wide inputs.
constexpr int64_t kDim0ArgReduceMaxCols = 64;

void min_dim0_out(const Tensor& self, bool keepdim, Tensor& out_values, Tensor& out_indices)
{
    const int64_t num_rows = self.size(0);
    const int64_t num_cols = self.size(1);
    TORCH_CHECK_INDEX(num_rows != 0, "min(): Expected reduction dim 0 to have non-zero size.");

    if (num_cols > kDim0ArgReduceMaxCols)
    {
        AT_DISPATCH_V2(
            self.scalar_type(), "min_dim0", AT_WRAP([&] {
                if (keepdim)
                    mgpu_launch_func(num_cols, MinDim0Functor<scalar_t>(), self.packed_accessor64<scalar_t, 2>(),
                                     out_values.packed_accessor64<scalar_t, 2>(),
                                     out_indices.packed_accessor64<int64_t, 2>());
                else
                    mgpu_launch_func(num_cols, MinDim0Functor<scalar_t>(), self.packed_accessor64<scalar_t, 2>(),
                                     out_values.packed_accessor64<scalar_t, 1>(),
                                     out_indices.packed_accessor64<int64_t, 1>());
            }),
            AT_EXPAND(AT_ALL_TYPES));
        return;
    }

    int device = cudaInvalidDeviceId;
    C10_CUDA_CHECK(cudaGetDevice(&device));

    // With keepdim the outputs are [1, num_cols], so the column stride is stride(1); otherwise [num_cols].
    const int64_t out_values_stride = keepdim ? out_values.stride(1) : out_values.stride(0);
    const int64_t out_indices_stride = keepdim ? out_indices.stride(1) : out_indices.stride(0);

    Tensor partial_values = at::empty({ c10::cuda::device_count() * num_cols }, self.options());
    Tensor partial_indices = at::empty({ c10::cuda::device_count() * num_cols }, self.options().dtype(kLong));
    AT_DISPATCH_V2(self.scalar_type(), "min_dim0", AT_WRAP([&] {
                       arg_min_dim0_async(self.const_data_ptr<scalar_t>(), num_rows, num_cols, self.stride(0),
                                          self.stride(1), out_values.data_ptr<scalar_t>(),
                                          out_indices.data_ptr<int64_t>(), out_values_stride, out_indices_stride,
                                          partial_values.data_ptr<scalar_t>(), partial_indices.data_ptr<int64_t>());
                   }),
                   AT_EXPAND(AT_ALL_TYPES));

    C10_CUDA_CHECK(cudaSetDevice(device));
}

void max_dim0_out(const Tensor& self, bool keepdim, Tensor& out_values, Tensor& out_indices)
{
    const int64_t num_rows = self.size(0);
    const int64_t num_cols = self.size(1);
    TORCH_CHECK_INDEX(num_rows != 0, "max(): Expected reduction dim 0 to have non-zero size.");

    if (num_cols > kDim0ArgReduceMaxCols)
    {
        AT_DISPATCH_V2(
            self.scalar_type(), "max_dim0", AT_WRAP([&] {
                if (keepdim)
                    mgpu_launch_func(num_cols, MaxDim0Functor<scalar_t>(), self.packed_accessor64<scalar_t, 2>(),
                                     out_values.packed_accessor64<scalar_t, 2>(),
                                     out_indices.packed_accessor64<int64_t, 2>());
                else
                    mgpu_launch_func(num_cols, MaxDim0Functor<scalar_t>(), self.packed_accessor64<scalar_t, 2>(),
                                     out_values.packed_accessor64<scalar_t, 1>(),
                                     out_indices.packed_accessor64<int64_t, 1>());
            }),
            AT_EXPAND(AT_ALL_TYPES));
        return;
    }

    int device = cudaInvalidDeviceId;
    C10_CUDA_CHECK(cudaGetDevice(&device));

    // With keepdim the outputs are [1, num_cols], so the column stride is stride(1); otherwise [num_cols].
    const int64_t out_values_stride = keepdim ? out_values.stride(1) : out_values.stride(0);
    const int64_t out_indices_stride = keepdim ? out_indices.stride(1) : out_indices.stride(0);

    Tensor partial_values = at::empty({ c10::cuda::device_count() * num_cols }, self.options());
    Tensor partial_indices = at::empty({ c10::cuda::device_count() * num_cols }, self.options().dtype(kLong));
    AT_DISPATCH_V2(self.scalar_type(), "max_dim0", AT_WRAP([&] {
                       arg_max_dim0_async(self.const_data_ptr<scalar_t>(), num_rows, num_cols, self.stride(0),
                                          self.stride(1), out_values.data_ptr<scalar_t>(),
                                          out_indices.data_ptr<int64_t>(), out_values_stride, out_indices_stride,
                                          partial_values.data_ptr<scalar_t>(), partial_indices.data_ptr<int64_t>());
                   }),
                   AT_EXPAND(AT_ALL_TYPES));

    C10_CUDA_CHECK(cudaSetDevice(device));
}

// argmax and argmin share everything but the comparison, so they share their argument handling too.
// `input` is the tensor the layout describes, which is the flattened self when no dim is given, and
// `empty` means the output has no elements and there is nothing to launch.
struct ArgReduction
{
    Tensor input;
    ReductionLayout<2> layout;
    bool empty;
};

// Reduces any dimension through the row layout, which parallelizes over the output elements. A
// trailing reduction dim is contiguous; the others read with a stride. Inputs that produce few output
// elements, such as a tall and narrow matrix reduced over dim 0 or a flattened reduction, leave the
// GPUs idle. The cub path behind max_dim0_out covers that shape, but it compares with `>` instead of
// propagating NaN, so it is not reused here.
ArgReduction prepare_arg_reduction(
    const Tensor& self, std::optional<int64_t> dim, bool keepdim, const char* name, Tensor& out)
{
    TORCH_CHECK(out.scalar_type() == kLong, name, " Expected a long tensor for the output, got ", out.scalar_type());

    // Without a dim the reduction runs over the flattened input, and keepdim then leaves one singleton
    // dimension per input dimension. This matches ATen's argmax and argmin meta functions.
    const bool flatten = !dim.has_value() || self.dim() == 0;
    const Tensor input = flatten ? self.reshape({ -1 }) : self;

    int64_t reduction_dim = flatten ? 0 : dim.value();
    if (reduction_dim < 0)
    {
        reduction_dim += input.dim();
    }
    TORCH_CHECK(reduction_dim >= 0 && reduction_dim < input.dim(), name, " Invalid dim ", dim.value_or(0));
    TORCH_CHECK_INDEX(
        input.size(reduction_dim) != 0, name, " Expected reduction dim ", reduction_dim, " to have non-zero size.");

    std::vector<int64_t> out_sizes;
    if (flatten)
    {
        out_sizes.assign(keepdim ? self.dim() : 0, 1);
    }
    else
    {
        out_sizes.assign(input.sizes().begin(), input.sizes().end());
        if (keepdim)
        {
            out_sizes[reduction_dim] = 1;
        }
        else
        {
            out_sizes.erase(out_sizes.begin() + reduction_dim);
        }
    }
    at::native::resize_output(out, out_sizes);
    at::assert_no_internal_overlap(out);
    if (out.numel() == 0)
    {
        return { input, ReductionLayout<2>{}, /*empty=*/true };
    }

    // A flattened reduction produces a single output element, so its strides never enter the layout
    // and an empty stride list keeps the output rank consistent with the flattened input.
    const auto layout =
        make_reduction_layout<2>(input.sizes(), reduction_dim,
                                 std::array<IntArrayRef, 2>{ input.strides(), flatten ? IntArrayRef{} : out.strides() });
    return { input, layout, /*empty=*/false };
}

} // namespace

std::tuple<Tensor&, Tensor&> max_out(const Tensor& self, int64_t dim, bool keepdim, Tensor& max, Tensor& max_values)
{
    TORCH_CHECK(self.dim() > 0, "max(): Expected a non-scalar input tensor");
    int64_t reduction_dim = dim;
    if (reduction_dim < 0)
    {
        reduction_dim += self.dim();
    }
    TORCH_CHECK(reduction_dim >= 0 && reduction_dim < self.dim(), "Unsupported dim for max_out");

    if (reduction_dim == self.dim() - 1)
    {
        TORCH_CHECK_INDEX(
            self.size(reduction_dim) != 0, "max(): Expected reduction dim ", reduction_dim, " to have non-zero size.");
        const auto layout =
            make_reduction_layout<3>(self.sizes(), reduction_dim,
                                     std::array<IntArrayRef, 3>{ self.strides(), max.strides(), max_values.strides() });
        AT_DISPATCH_V2(self.scalar_type(), "max_out", AT_WRAP([&] {
                           using Op = ArgReductionOp<scalar_t, ::cuda::std::greater<scalar_t>>;
                           row_reduce<scalar_t, ArgPair<scalar_t>>(
                               self.const_data_ptr<scalar_t>(), layout, ArgTransform<scalar_t>(), Op(), Op::identity(),
                               ArgOutput<scalar_t>{ max.data_ptr<scalar_t>(), max_values.data_ptr<int64_t>() });
                       }),
                       AT_EXPAND(AT_ALL_TYPES));
        return std::tie(max, max_values);
    }

    if (self.dim() == 2 && reduction_dim == 0)
    {
        max_dim0_out(self, keepdim, max, max_values);
        return std::tie(max, max_values);
    }

    TORCH_CHECK(false, "Unsupported dim for max_out");
    return std::tie(max, max_values);
}

std::tuple<Tensor&, Tensor&> min_out(const Tensor& self, int64_t dim, bool keepdim, Tensor& min, Tensor& min_values)
{
    TORCH_CHECK(self.dim() > 0, "min(): Expected a non-scalar input tensor");
    int64_t reduction_dim = dim;
    if (reduction_dim < 0)
    {
        reduction_dim += self.dim();
    }
    TORCH_CHECK(reduction_dim >= 0 && reduction_dim < self.dim(), "Unsupported dim for min_out");

    if (reduction_dim == self.dim() - 1)
    {
        TORCH_CHECK_INDEX(
            self.size(reduction_dim) != 0, "min(): Expected reduction dim ", reduction_dim, " to have non-zero size.");
        const auto layout =
            make_reduction_layout<3>(self.sizes(), reduction_dim,
                                     std::array<IntArrayRef, 3>{ self.strides(), min.strides(), min_values.strides() });
        AT_DISPATCH_V2(self.scalar_type(), "min_out", AT_WRAP([&] {
                           using Op = ArgReductionOp<scalar_t, ::cuda::std::less<scalar_t>>;
                           row_reduce<scalar_t, ArgPair<scalar_t>>(
                               self.const_data_ptr<scalar_t>(), layout, ArgTransform<scalar_t>(), Op(), Op::identity(),
                               ArgOutput<scalar_t>{ min.data_ptr<scalar_t>(), min_values.data_ptr<int64_t>() });
                       }),
                       AT_EXPAND(AT_ALL_TYPES));
        return std::tie(min, min_values);
    }

    if (self.dim() == 2 && reduction_dim == 0)
    {
        min_dim0_out(self, keepdim, min, min_values);
        return std::tie(min, min_values);
    }

    TORCH_CHECK(false, "Unsupported dim for min_out");
    return std::tie(min, min_values);
}

Tensor& argmax_out(const Tensor& self, std::optional<int64_t> dim, bool keepdim, Tensor& out)
{
    const auto reduction = prepare_arg_reduction(self, dim, keepdim, "argmax():", out);
    if (reduction.empty)
    {
        return out;
    }

    AT_DISPATCH_V2(reduction.input.scalar_type(), "argmax_out", AT_WRAP([&] {
                       using acc_t = at::opmath_type<scalar_t>;
                       using Op = ArgReductionOp<acc_t, ::cuda::std::greater<acc_t>>;
                       row_reduce<scalar_t, ArgPair<acc_t>>(reduction.input.const_data_ptr<scalar_t>(), reduction.layout,
                                                            ArgTransform<scalar_t, acc_t>(), Op(), Op::identity(),
                                                            ArgIndexOutput<acc_t>{ out.data_ptr<int64_t>() });
                   }),
                   AT_EXPAND(AT_ALL_TYPES), kHalf, kBFloat16);

    return out;
}

Tensor& argmin_out(const Tensor& self, std::optional<int64_t> dim, bool keepdim, Tensor& out)
{
    const auto reduction = prepare_arg_reduction(self, dim, keepdim, "argmin():", out);
    if (reduction.empty)
    {
        return out;
    }

    AT_DISPATCH_V2(reduction.input.scalar_type(), "argmin_out", AT_WRAP([&] {
                       using acc_t = at::opmath_type<scalar_t>;
                       using Op = ArgReductionOp<acc_t, ::cuda::std::less<acc_t>>;
                       row_reduce<scalar_t, ArgPair<acc_t>>(reduction.input.const_data_ptr<scalar_t>(), reduction.layout,
                                                            ArgTransform<scalar_t, acc_t>(), Op(), Op::identity(),
                                                            ArgIndexOutput<acc_t>{ out.data_ptr<int64_t>() });
                   }),
                   AT_EXPAND(AT_ALL_TYPES), kHalf, kBFloat16);

    return out;
}

TORCH_LIBRARY_IMPL(aten, PrivateUse1, m)
{
    m.impl(TORCH_SELECTIVE_NAME("aten::max.dim_max"), TORCH_FN(max_out));
    m.impl(TORCH_SELECTIVE_NAME("aten::min.dim_min"), TORCH_FN(min_out));
    m.impl(TORCH_SELECTIVE_NAME("aten::argmax.out"), TORCH_FN(argmax_out));
    m.impl(TORCH_SELECTIVE_NAME("aten::argmin.out"), TORCH_FN(argmin_out));
}

} // namespace dgx

} // namespace at::native
