/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
#include "KernelLaunchUtils.cuh"
#include "RowReduction.cuh"
#include "StridedTensorUtils.cuh"

#include <ATen/Dispatch_v2.h>
#include <ATen/MemoryOverlap.h>
#include <ATen/OpMathType.h>
#include <ATen/TensorIterator.h>
#include <ATen/native/Resize.h>
#include <ATen/ops/empty.h>
#include <ATen/ops/empty_like.h>
#include <ATen/ops/eye_native.h>

#include <torch/library.h>

#include <cub/cub.cuh>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/transform_iterator.h>

#include <cmath>
#include <limits>

namespace at::native
{

namespace dgx
{

namespace
{

template <class T>
struct TriangleFunctor
{
    int64_t rows;
    int64_t columns;
    int64_t diagonal;
    bool upper;

    __device__ bool keep(int64_t index) const
    {
        const int64_t row = (index / columns) % rows;
        const int64_t column = index % columns;
        // Subtract the coordinates to avoid overflowing extreme diagonal offsets.
        const int64_t offset = column - row;
        return upper ? offset >= diagonal : offset <= diagonal;
    }

    __device__ void operator()(int64_t index, const T* self, T* out) const
    {
        out[index] = keep(index) ? self[index] : T(0);
    }

    __device__ void operator()(int64_t index, const T* self, T* out, StridedOffsetCalculator<2> offsets) const
    {
        const auto tensor_offsets = offsets.get(index);
        out[tensor_offsets[1]] = keep(index) ? self[tensor_offsets[0]] : T(0);
    }
};

template <class T>
struct AbsPowerOp
{
    T ord;

    __host__ __device__ T operator()(T value) const
    {
        return ::cuda::std::pow(::cuda::std::abs(value), ord);
    }

    __device__ T operator()(T value, int64_t) const
    {
        return (*this)(value);
    }
};

template <class T>
struct AbsOp
{
    __host__ __device__ T operator()(T value) const
    {
        return ::cuda::std::abs(value);
    }

    __device__ T operator()(T value, int64_t) const
    {
        return (*this)(value);
    }
};

template <class T>
struct NonzeroOp
{
    __host__ __device__ T operator()(T value) const
    {
        return value != T(0) ? T(1) : T(0);
    }

    __device__ T operator()(T value, int64_t) const
    {
        return (*this)(value);
    }
};

template <class T>
struct MaximumOp
{
    __host__ __device__ T operator()(T lhs, T rhs) const
    {
        return ::cuda::std::isnan(lhs) || lhs > rhs ? lhs : rhs;
    }
};

template <class T>
struct MinimumOp
{
    __host__ __device__ T operator()(T lhs, T rhs) const
    {
        return ::cuda::std::isnan(lhs) || lhs < rhs ? lhs : rhs;
    }
};

template <class T>
struct FinalizeVectorNormFunctor
{
    __device__ void operator()(size_t, T* out, T ord)
    {
        *out = ::cuda::std::pow(*out, T(1) / ord);
    }
};

template <class T>
struct RowVectorNormOutput
{
    T* output;
    T ord;
    bool finalize;

    template <typename Offsets>
    __device__ void operator()(const Offsets& offsets, T aggregate) const
    {
        output[offsets[1]] = finalize ? ::cuda::std::pow(aggregate, T(1) / ord) : aggregate;
    }
};

template <class T, class InputIterator, class TransformOp, class ReductionOp>
void vector_norm_async(int64_t numel,
                       InputIterator in,
                       T* device_outs,
                       T* out,
                       TransformOp transform_op,
                       ReductionOp reduction_op,
                       T init_value,
                       bool finalize,
                       T ord)
{
    constexpr int merge_device_id = 0;
    std::vector<cudaEvent_t> events(c10::cuda::device_count());
    for (const auto device_id : c10::irange(c10::cuda::device_count()))
    {
        C10_CUDA_CHECK(cudaSetDevice(device_id));
        auto stream = c10::cuda::getCurrentCUDAStream(device_id);

        int64_t device_offset, device_numel;
        std::tie(device_offset, device_numel) = chunk(numel, device_id);
        size_t scratch_size = 0;
        void* scratch_data = nullptr;

        C10_CUDA_CHECK(cub::DeviceReduce::TransformReduce(scratch_data, scratch_size, in + device_offset,
                                                          device_outs + device_id, device_numel, reduction_op,
                                                          transform_op, init_value, stream));
        C10_CUDA_CHECK(cudaMallocAsync(&scratch_data, scratch_size, stream));
        C10_CUDA_CHECK(cub::DeviceReduce::TransformReduce(scratch_data, scratch_size, in + device_offset,
                                                          device_outs + device_id, device_numel, reduction_op,
                                                          transform_op, init_value, stream));
        C10_CUDA_CHECK(cudaFreeAsync(scratch_data, stream));
    }

    for (const auto device_id : c10::irange(1, c10::cuda::device_count()))
    {
        C10_CUDA_CHECK(cudaSetDevice(device_id));
        auto stream = c10::cuda::getCurrentCUDAStream(device_id);
        C10_CUDA_CHECK(cudaEventCreateWithFlags(&events[device_id], cudaEventDisableTiming));
        C10_CUDA_CHECK(cudaEventRecord(events[device_id], stream));
    }

    C10_CUDA_CHECK(cudaSetDevice(merge_device_id));
    auto merge_stream = c10::cuda::getCurrentCUDAStream(merge_device_id);
    cudaEvent_t merge_event = nullptr;
    C10_CUDA_CHECK(cudaEventCreateWithFlags(&merge_event, cudaEventDisableTiming));
    for (const auto device_id : c10::irange(1, c10::cuda::device_count()))
    {
        C10_CUDA_CHECK(cudaStreamWaitEvent(merge_stream, events[device_id]));
    }

    size_t scratch_size = 0;
    void* scratch_data = nullptr;
    C10_CUDA_CHECK(cub::DeviceReduce::Reduce(scratch_data, scratch_size, device_outs, out, c10::cuda::device_count(),
                                             reduction_op, init_value, merge_stream));
    C10_CUDA_CHECK(cudaMallocAsync(&scratch_data, scratch_size, merge_stream));
    C10_CUDA_CHECK(cub::DeviceReduce::Reduce(scratch_data, scratch_size, device_outs, out, c10::cuda::device_count(),
                                             reduction_op, init_value, merge_stream));
    C10_CUDA_CHECK(cudaFreeAsync(scratch_data, merge_stream));

    if (finalize)
    {
        lambda_kernel<<<1, 1, 0, merge_stream>>>(1, FinalizeVectorNormFunctor<T>(), out, ord);
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }
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

} // namespace

template <class T>
struct CrossProductFunctor
{
    __device__ void operator()(size_t tid,
                               const PackedTensorAccessor64<T, 2> self,
                               const PackedTensorAccessor64<T, 2> other,
                               PackedTensorAccessor64<T, 2> out)
    {
        out[tid][0] = self[tid][1] * other[tid][2] - self[tid][2] * other[tid][1];
        out[tid][1] = self[tid][2] * other[tid][0] - self[tid][0] * other[tid][2];
        out[tid][2] = self[tid][0] * other[tid][1] - self[tid][1] * other[tid][0];
    }
};

// Zeroes the entries outside one triangle of each matrix in the batch. `upper` selects which side is
// kept, so tril and triu differ only in the comparison against the diagonal offset.
Tensor& triangle_out(const Tensor& self, int64_t diagonal, bool upper, const char* name, Tensor& out)
{
    TORCH_CHECK(self.dim() >= 2, name, ": input tensor must have at least 2 dimensions");
    TORCH_CHECK(self.device() == out.device(), name, ".out: input and output must be on the same device");
    TORCH_CHECK(self.scalar_type() == out.scalar_type(), name, ".out: input and output must have the same dtype");
    at::native::resize_output(out, self.sizes());
    at::assert_no_internal_overlap(out);
    at::assert_no_partial_overlap(out, self);
    if (self.numel() == 0)
        return out;

    AT_DISPATCH_V2(self.scalar_type(), "triangle_out", AT_WRAP([&] {
                       const auto func = TriangleFunctor<scalar_t>{ self.size(-2), self.size(-1), diagonal, upper };
                       constexpr size_t kAlignment = kPageSize / sizeof(scalar_t);
                       if (self.is_contiguous() && out.is_contiguous())
                       {
                           mgpu_launch_func(kAlignment, self.numel(), func, self.const_data_ptr<scalar_t>(),
                                            out.data_ptr<scalar_t>());
                       }
                       else
                       {
                           const auto offsets = make_strided_offset_calculator<2>(
                               self.sizes(), std::array<IntArrayRef, 2>{ self.strides(), out.strides() });
                           mgpu_launch_func(kAlignment, self.numel(), func, self.const_data_ptr<scalar_t>(),
                                            out.data_ptr<scalar_t>(), offsets);
                       }
                   }),
                   AT_EXPAND(AT_ALL_TYPES_AND_COMPLEX), kHalf, kBFloat16, kBool);

    return out;
}

Tensor& tril_out(const Tensor& self, int64_t diagonal, Tensor& out)
{
    return triangle_out(self, diagonal, /*upper=*/false, "tril", out);
}

Tensor& triu_out(const Tensor& self, int64_t diagonal, Tensor& out)
{
    return triangle_out(self, diagonal, /*upper=*/true, "triu", out);
}

Tensor& linalg_vector_norm_out(const Tensor& self,
                               const Scalar& ord,
                               OptionalIntArrayRef dim,
                               bool keepdim,
                               std::optional<ScalarType> dtype,
                               Tensor& out)
{
    TORCH_CHECK(!dtype, "dtype is not supported in linalg_vector_norm.out");

    if (!dim)
    {
        const double norm_order = ord.toDouble();
        if (self.numel() == 0)
        {
            TORCH_CHECK(!(norm_order < 0.0) && !std::isinf(norm_order), "linalg.vector_norm cannot compute the ",
                        norm_order, " norm on an empty tensor because the operation does not have an identity");
            out.zero_();
            return out;
        }

        int device = cudaInvalidDeviceId;
        C10_CUDA_CHECK(cudaGetDevice(&device));

        Tensor device_outs = at::empty({ c10::cuda::device_count() }, out.options());
        AT_DISPATCH_V2(
            out.scalar_type(), "linalg_vector_norm_out", AT_WRAP([&] {
                auto* device_outputs = device_outs.data_ptr<scalar_t>();
                auto* output = out.data_ptr<scalar_t>();
                const scalar_t p = ord.to<scalar_t>();

                if (self.is_contiguous())
                {
                    const auto* input = self.const_data_ptr<scalar_t>();
                    if (norm_order == 0.0)
                    {
                        vector_norm_async(self.numel(), input, device_outputs, output, NonzeroOp<scalar_t>(),
                                          ::cuda::std::plus<scalar_t>(), scalar_t(0), false, p);
                    }
                    else if (std::isinf(norm_order) && norm_order > 0.0)
                    {
                        vector_norm_async(self.numel(), input, device_outputs, output, AbsOp<scalar_t>(),
                                          MaximumOp<scalar_t>(), scalar_t(0), false, p);
                    }
                    else if (std::isinf(norm_order))
                    {
                        vector_norm_async(self.numel(), input, device_outputs, output, AbsOp<scalar_t>(),
                                          MinimumOp<scalar_t>(), std::numeric_limits<scalar_t>::infinity(), false, p);
                    }
                    else
                    {
                        vector_norm_async(self.numel(), input, device_outputs, output, AbsPowerOp<scalar_t>{ p },
                                          ::cuda::std::plus<scalar_t>(), scalar_t(0), true, p);
                    }
                }
                else
                {
                    const auto offsets =
                        make_strided_offset_calculator<1>(self.sizes(), std::array<IntArrayRef, 1>{ self.strides() });
                    auto logical_indices = thrust::make_counting_iterator<int64_t>(0);
                    auto input = thrust::make_transform_iterator(
                        logical_indices, StridedInputOp<scalar_t>{ self.const_data_ptr<scalar_t>(), offsets });
                    if (norm_order == 0.0)
                    {
                        vector_norm_async(self.numel(), input, device_outputs, output, NonzeroOp<scalar_t>(),
                                          ::cuda::std::plus<scalar_t>(), scalar_t(0), false, p);
                    }
                    else if (std::isinf(norm_order) && norm_order > 0.0)
                    {
                        vector_norm_async(self.numel(), input, device_outputs, output, AbsOp<scalar_t>(),
                                          MaximumOp<scalar_t>(), scalar_t(0), false, p);
                    }
                    else if (std::isinf(norm_order))
                    {
                        vector_norm_async(self.numel(), input, device_outputs, output, AbsOp<scalar_t>(),
                                          MinimumOp<scalar_t>(), std::numeric_limits<scalar_t>::infinity(), false, p);
                    }
                    else
                    {
                        vector_norm_async(self.numel(), input, device_outputs, output, AbsPowerOp<scalar_t>{ p },
                                          ::cuda::std::plus<scalar_t>(), scalar_t(0), true, p);
                    }
                }
            }),
            AT_EXPAND(AT_FLOATING_TYPES));

        C10_CUDA_CHECK(cudaSetDevice(device));
        return out;
    }

    TORCH_CHECK(self.dim() > 0, "linalg_vector_norm requires at least one dimension");
    TORCH_CHECK((*dim).size() == 1, "Only one reduction dimension is supported in linalg_vector_norm.out");
    int64_t reduction_dim = (*dim)[0];
    if (reduction_dim < 0)
    {
        reduction_dim += self.dim();
    }
    TORCH_CHECK(reduction_dim == self.dim() - 1, "dim is not supported in linalg_vector_norm.out");

    const double norm_order = ord.toDouble();
    if (self.size(reduction_dim) == 0)
    {
        TORCH_CHECK(!(norm_order < 0.0) && !std::isinf(norm_order), "linalg.vector_norm cannot compute the ",
                    norm_order, " norm on an empty dimension because the operation does not have an identity");
    }

    const auto layout = make_reduction_layout<2>(
        self.sizes(), reduction_dim, std::array<IntArrayRef, 2>{ self.strides(), out.strides() });
    AT_DISPATCH_V2(out.scalar_type(), "linalg_vector_norm_out", AT_WRAP([&] {
                       const scalar_t p = ord.to<scalar_t>();
                       auto output_op = RowVectorNormOutput<scalar_t>{ out.data_ptr<scalar_t>(), p, false };
                       if (norm_order == 0.0)
                       {
                           row_reduce<scalar_t, scalar_t>(self.const_data_ptr<scalar_t>(), layout, NonzeroOp<scalar_t>(),
                                                          ::cuda::std::plus<scalar_t>(), scalar_t(0), output_op);
                       }
                       else if (std::isinf(norm_order) && norm_order > 0.0)
                       {
                           row_reduce<scalar_t, scalar_t>(self.const_data_ptr<scalar_t>(), layout, AbsOp<scalar_t>(),
                                                          MaximumOp<scalar_t>(), scalar_t(0), output_op);
                       }
                       else if (std::isinf(norm_order))
                       {
                           row_reduce<scalar_t, scalar_t>(self.const_data_ptr<scalar_t>(), layout, AbsOp<scalar_t>(),
                                                          MinimumOp<scalar_t>(),
                                                          std::numeric_limits<scalar_t>::infinity(), output_op);
                       }
                       else
                       {
                           output_op.finalize = true;
                           row_reduce<scalar_t, scalar_t>(self.const_data_ptr<scalar_t>(), layout,
                                                          AbsPowerOp<scalar_t>{ p }, ::cuda::std::plus<scalar_t>(),
                                                          scalar_t(0), output_op);
                       }
                   }),
                   AT_EXPAND(AT_FLOATING_TYPES));

    return out;
}

Tensor& linalg_cross_out(const Tensor& self, const Tensor& other, int64_t dim, Tensor& out)
{
    TORCH_CHECK(dim == -1, "non-default dim is not supported in linalg_cross_out")

    TORCH_CHECK(self.scalar_type() == out.scalar_type(), "mismatched type in linalg_cross_out");
    TORCH_CHECK(other.scalar_type() == out.scalar_type(), "mismatched type in linalg_cross_out");
    TORCH_CHECK(self.dim() == 2);
    TORCH_CHECK(self.sizes()[1] == 3);
    TORCH_CHECK(other.dim() == 2);
    TORCH_CHECK(other.sizes()[1] == 3);
    TORCH_CHECK(out.dim() == 2);
    TORCH_CHECK(out.sizes()[1] == 3);

    AT_DISPATCH_V2(out.scalar_type(), "linalg_cross_out", AT_WRAP([&] {
                       mgpu_launch_func(out.sizes()[0], CrossProductFunctor<scalar_t>(),
                                        self.packed_accessor64<scalar_t, 2>(), other.packed_accessor64<scalar_t, 2>(),
                                        out.packed_accessor64<scalar_t, 2>());
                   }),
                   AT_EXPAND(AT_FLOATING_TYPES));

    return out;
}

Tensor& eye_out(int64_t n, Tensor& result)
{
    return at::native::eye_out_cuda(n, n, result);
}

Tensor& eye_m_out(int64_t n, int64_t m, Tensor& result)
{
    result.resize_({ n, m });
    result.zero_();

    int64_t sz = std::min<int64_t>(n, m);
    int64_t stride = result.stride(0) + result.stride(1);

    Tensor diag = result.as_strided({ sz }, { stride });
    diag.fill_(1);
    return result;
}

TORCH_LIBRARY_IMPL(aten, PrivateUse1, m)
{
    m.impl(TORCH_SELECTIVE_NAME("aten::tril.out"), TORCH_FN(tril_out));
    m.impl(TORCH_SELECTIVE_NAME("aten::triu.out"), TORCH_FN(triu_out));
    m.impl(TORCH_SELECTIVE_NAME("aten::linalg_vector_norm.out"), TORCH_FN(linalg_vector_norm_out));
    m.impl(TORCH_SELECTIVE_NAME("aten::linalg_cross.out"), TORCH_FN(linalg_cross_out));
    m.impl(TORCH_SELECTIVE_NAME("aten::eye.out"), TORCH_FN(eye_out));
    m.impl(TORCH_SELECTIVE_NAME("aten::eye.m_out"), TORCH_FN(eye_m_out));
}

} // namespace dgx

} // namespace at::native
