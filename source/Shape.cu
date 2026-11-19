/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
#include "KernelLaunchUtils.cuh"
#include "StridedTensorUtils.cuh"

#include <ATen/Dispatch_v2.h>
#include <ATen/ops/empty.h>

#include <torch/library.h>

namespace at::native
{

namespace dgx
{

namespace
{

template <typename scalar_t>
struct FlipFunctor
{
    __device__ void operator()(int64_t index, const scalar_t* input, scalar_t* output, StridedOffsetCalculator<1> offsets) const
    {
        output[index] = input[offsets.get(index)[0]];
    }
};

template <typename scalar_t, typename out_t = scalar_t>
__global__ void cat_copy_kernel(
    int64_t count, int64_t offset, const scalar_t* tensor, out_t* result, StridedOffsetCalculator<2> offsets)
{
    const int64_t tid = static_cast<int64_t>(threadIdx.x) + static_cast<int64_t>(blockIdx.x) * blockDim.x;
    if (tid < count)
    {
        const auto tensor_offsets = offsets.get(tid + offset);
        result[tensor_offsets[1]] = static_cast<out_t>(tensor[tensor_offsets[0]]);
    }
}

} // namespace

Tensor& cat_out(const ITensorListRef& tensors, int64_t dim, Tensor& result)
{
    auto materialized = tensors.materialize();
    TORCH_CHECK(result.is_contiguous(), "Concatenation op output tensor is not contiguous");
    if (!result.numel())
        return result;
    dim = maybe_wrap_dim(dim, result.dim());

    // Preserve row ownership for [N, D] and [C, N, D] feature concatenation.
    // Splitting N keeps the same Gaussian owner for one and multiple cameras.
    // Dim-0 appends use the output's row partition. Other layouts retain the
    // existing flat input partition without imposing a conflicting output prefetch.
    const int64_t shard_dim = dim == 0 ? 0 : (result.dim() <= 3 && dim == result.dim() - 1 ? dim - 1 : -1);
    parallel_for_each([&](int device, cudaStream_t stream) {
        int64_t device_offset = 0, device_count = 0;
        if (shard_dim >= 0)
        {
            std::tie(device_offset, device_count) = partition::range(result.size(shard_dim), device);
            if (!device_count)
                return;
        }

        int64_t dim_offset = 0;
        for (int64_t j = 0; j < materialized.size(); ++j)
        {
            const Tensor& tensor = materialized[j];
            // cat also accepts a one-dimensional empty tensor alongside higher-rank inputs.
            if (tensor.numel() == 0)
                continue;

            auto sizes = tensor.sizes().vec();
            int64_t source_offset = 0, logical_offset = 0, device_numel = 0;
            int64_t output_offset = dim_offset * result.stride(dim);
            if (shard_dim >= 0)
            {
                int64_t input_offset = device_offset;
                int64_t input_count = device_count;
                if (shard_dim == dim)
                {
                    const int64_t begin = std::max(device_offset, dim_offset);
                    const int64_t end = std::min(device_offset + device_count, dim_offset + tensor.size(dim));
                    input_offset = begin - dim_offset;
                    input_count = std::max(int64_t{ 0 }, end - begin);
                }
                sizes[shard_dim] = input_count;
                device_numel = tensor.numel() / tensor.size(shard_dim) * input_count;
                source_offset = input_offset * tensor.stride(shard_dim);
                output_offset += input_offset * result.stride(shard_dim);
            }
            else
            {
                std::tie(logical_offset, device_numel) = partition::range(tensor.numel(), device);
            }
            dim_offset += tensor.size(dim);
            if (!device_numel)
                continue;

            const auto offsets = make_strided_offset_calculator<2>(
                sizes, std::array<IntArrayRef, 2>{ tensor.strides(), result.strides() });
            const unsigned int num_blocks = ::cuda::ceil_div(device_numel, kNumThreads);
            if (tensor.scalar_type() == result.scalar_type())
            {
                AT_DISPATCH_V2(result.scalar_type(), "cat_out", AT_WRAP([&] {
                                   cat_copy_kernel<<<num_blocks, kNumThreads, 0, stream>>>(
                                       device_numel, logical_offset, tensor.const_data_ptr<scalar_t>() + source_offset,
                                       result.data_ptr<scalar_t>() + output_offset, offsets);
                                   C10_CUDA_KERNEL_LAUNCH_CHECK();
                               }),
                               AT_EXPAND(AT_ALL_TYPES), kBool, kComplexHalf, kComplexFloat, kComplexDouble);
            }
            else
            {
                if (tensor.scalar_type() == kInt && result.scalar_type() == kLong)
                {
                    cat_copy_kernel<<<num_blocks, kNumThreads, 0, stream>>>(
                        device_numel, logical_offset, tensor.const_data_ptr<int32_t>() + source_offset,
                        result.data_ptr<int64_t>() + output_offset, offsets);
                }
                else
                {
                    TORCH_CHECK(false, "Unsupported dtype conversion from ", tensor.scalar_type(), " to ",
                                result.scalar_type(), " for cat_out in dgx backend");
                }
                C10_CUDA_KERNEL_LAUNCH_CHECK();
            }
        }
    });

    return result;
}

Tensor _reshape_alias(const Tensor& self, c10::SymIntArrayRef size, c10::SymIntArrayRef stride)
{
    auto sizes = C10_AS_INTARRAYREF_SLOW(size);
    auto strides = C10_AS_INTARRAYREF_SLOW(stride);

    Tensor self_;
    self_ = detail::make_tensor<TensorImpl>(c10::TensorImpl::VIEW, Storage(self.storage()), self.key_set(), self.dtype());
    auto* self_tmp_ = self_.unsafeGetTensorImpl();
    self_tmp_->set_storage_offset(self.storage_offset());
    self_tmp_->set_sizes_and_strides(sizes, strides);
    return self_;
}

Tensor unfold(const Tensor& self, int64_t d, int64_t size, int64_t step)
{
    // some special handling to deal with allow d == 0 when self.dim() == 0
    auto ndim = self.dim();
    d = maybe_wrap_dim(d, ndim, /*wrap_scalar=*/true);

    auto sizes = self.sizes().vec();
    auto strides = self.strides().vec();
    int64_t max_size = self.dim() == 0 ? 1 : sizes[d];
    TORCH_CHECK(size >= 0, "size is ", size, " but must be >= 0");
    TORCH_CHECK(size <= max_size, "maximum size for tensor at dimension ", d, " is ", max_size, " but size is ", size);
    TORCH_CHECK(step > 0, "step is ", step, " but must be > 0");
    sizes.push_back(size);
    strides.push_back(self.dim() == 0 ? 1 : strides[d]);
    // The if handles the self.dim() == 0 case
    if (d < ndim)
    {
        sizes[d] = (sizes[d] - size) / step + 1;
        strides[d] *= step;
    }
    return self.as_strided(sizes, strides);
}

Tensor flip(const Tensor& self, IntArrayRef dims)
{
    auto strides = self.strides().vec();
    std::vector<bool> flipped(std::max<int64_t>(self.dim(), 1), false);
    int64_t offset = 0;
    for (auto dim : dims)
    {
        dim = maybe_wrap_dim(dim, self.dim());
        TORCH_CHECK(!flipped[dim], "dim ", dim, " appears multiple times in the list of dims");
        flipped[dim] = true;
        if (dim < self.dim())
        {
            offset += (self.size(dim) - 1) * strides[dim];
            strides[dim] = -strides[dim];
        }
    }
    auto result = at::empty(self.sizes(), self.options());
    if (!self.numel())
    {
        return result;
    }
    const auto offsets = make_strided_offset_calculator<1>(self.sizes(), std::array<IntArrayRef, 1>{ strides });
    AT_DISPATCH_V2(self.scalar_type(), "flip", AT_WRAP([&] {
                       mgpu_launch_func(self.numel(), FlipFunctor<scalar_t>(), self.const_data_ptr<scalar_t>() + offset,
                                        result.data_ptr<scalar_t>(), offsets);
                   }),
                   AT_EXPAND(AT_ALL_TYPES), kBool, kHalf, kBFloat16, kComplexHalf, kComplexFloat, kComplexDouble);
    return result;
}

TORCH_LIBRARY_IMPL(aten, PrivateUse1, m)
{
    m.impl(TORCH_SELECTIVE_NAME("aten::cat.out"), TORCH_FN(cat_out));
    m.impl(TORCH_SELECTIVE_NAME("aten::_reshape_alias"), TORCH_FN(_reshape_alias));
    m.impl(TORCH_SELECTIVE_NAME("aten::unfold"), TORCH_FN(unfold));
    m.impl(TORCH_SELECTIVE_NAME("aten::flip"), TORCH_FN(flip));
}

} // namespace dgx

} // namespace at::native
