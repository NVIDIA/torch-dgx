/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
#include "KernelLaunchUtils.cuh"

#include <ATen/AccumulateType.h>
#include <ATen/Dispatch_v2.h>
#include <ATen/native/ReduceOpsUtils.h>
#include <ATen/ops/empty.h>

#include <torch/library.h>

#include <cub/cub.cuh>

#include <limits>
#include <tuple>

namespace at::native
{

namespace dgx
{

namespace
{

constexpr size_t kMaxTensorDimensions = DimMask{}.size();

struct ReductionConfig
{
    int64_t num_reduction_dims;
    int64_t num_output_dims;
    int64_t reduction_numel;
    int64_t sizes[kMaxTensorDimensions];
    int64_t strides[kMaxTensorDimensions];
    int64_t reduction_dims[kMaxTensorDimensions];
    int64_t output_dims[kMaxTensorDimensions];
};

template <typename T>
struct WelfordData
{
    T mean;
    T m2;
    int64_t count;
};

template <typename T>
struct WelfordCombine
{
    __device__ WelfordData<T> operator()(const WelfordData<T>& lhs, const WelfordData<T>& rhs) const
    {
        if (lhs.count == 0)
        {
            return rhs;
        }
        if (rhs.count == 0)
        {
            return lhs;
        }

        const auto count = lhs.count + rhs.count;
        const auto delta = rhs.mean - lhs.mean;
        const auto rhs_fraction = static_cast<T>(rhs.count) / static_cast<T>(count);
        return { lhs.mean + delta * rhs_fraction,
                 lhs.m2 + rhs.m2 + delta * delta * static_cast<T>(lhs.count) * rhs_fraction, count };
    }
};

__device__ int64_t input_offset_for_output(int64_t output_index, const ReductionConfig& config)
{
    int64_t input_offset = 0;
    for (int64_t i = config.num_output_dims - 1; i >= 0; --i)
    {
        const auto dim = config.output_dims[i];
        const auto coordinate = output_index % config.sizes[dim];
        output_index /= config.sizes[dim];
        input_offset += coordinate * config.strides[dim];
    }
    return input_offset;
}

__device__ int64_t input_offset_for_reduction(int64_t reduction_index, const ReductionConfig& config)
{
    int64_t input_offset = 0;
    for (int64_t i = config.num_reduction_dims - 1; i >= 0; --i)
    {
        const auto dim = config.reduction_dims[i];
        const auto coordinate = reduction_index % config.sizes[dim];
        reduction_index /= config.sizes[dim];
        input_offset += coordinate * config.strides[dim];
    }
    return input_offset;
}

template <typename scalar_t, typename accscalar_t>
__global__ void var_mean_kernel(int64_t count,
                                int64_t output_offset,
                                const scalar_t* self,
                                scalar_t* variance,
                                scalar_t* mean,
                                ReductionConfig config,
                                accscalar_t correction)
{
    const auto local_output_index = static_cast<int64_t>(blockIdx.x);
    if (local_output_index >= count)
    {
        return;
    }

    const auto output_index = output_offset + local_output_index;
    const auto base_input_offset = input_offset_for_output(output_index, config);
    WelfordData<accscalar_t> accumulator{ 0, 0, 0 };

    for (int64_t reduction_index = threadIdx.x; reduction_index < config.reduction_numel; reduction_index += blockDim.x)
    {
        const auto input_offset = base_input_offset + input_offset_for_reduction(reduction_index, config);
        const auto value = static_cast<accscalar_t>(self[input_offset]);
        const auto new_count = accumulator.count + 1;
        const auto delta = value - accumulator.mean;
        accumulator.mean += delta / static_cast<accscalar_t>(new_count);
        accumulator.m2 += delta * (value - accumulator.mean);
        accumulator.count = new_count;
    }

    using BlockReduce = cub::BlockReduce<WelfordData<accscalar_t>, kNumThreads>;
    __shared__ typename BlockReduce::TempStorage temp_storage;
    accumulator = BlockReduce(temp_storage).Reduce(accumulator, WelfordCombine<accscalar_t>{});

    if (threadIdx.x == 0)
    {
        if (accumulator.count == 0)
        {
            const auto nan = ::cuda::std::numeric_limits<accscalar_t>::quiet_NaN();
            variance[output_index] = static_cast<scalar_t>(nan);
            mean[output_index] = static_cast<scalar_t>(nan);
            return;
        }

        const auto count_as_scalar = static_cast<accscalar_t>(accumulator.count);
        const auto divisor = count_as_scalar > correction ? count_as_scalar - correction : accscalar_t{ 0 };
        variance[output_index] = static_cast<scalar_t>(accumulator.m2 / divisor);
        mean[output_index] = static_cast<scalar_t>(accumulator.mean);
    }
}

ReductionConfig make_reduction_config(const Tensor& self, const DimMask& reduction_mask)
{
    ReductionConfig config{};
    config.reduction_numel = 1;

    for (const auto dim : c10::irange(self.dim()))
    {
        config.sizes[dim] = self.size(dim);
        config.strides[dim] = self.stride(dim);
        if (reduction_mask[dim])
        {
            config.reduction_dims[config.num_reduction_dims++] = dim;
            config.reduction_numel *= self.size(dim);
        }
        else
        {
            config.output_dims[config.num_output_dims++] = dim;
        }
    }
    return config;
}

template <typename scalar_t, typename accscalar_t>
void var_mean_async(const Tensor& self, Tensor& variance, Tensor& mean, const ReductionConfig& config, accscalar_t correction)
{
    parallel_for_each([&](int device, cudaStream_t stream) {
        int64_t device_offset, device_numel;
        std::tie(device_offset, device_numel) = chunk(variance.numel(), device);
        if (device_numel == 0)
        {
            return;
        }

        var_mean_kernel<scalar_t, accscalar_t><<<device_numel, kNumThreads, 0, stream>>>(
            device_numel, device_offset, self.const_data_ptr<scalar_t>(), variance.data_ptr<scalar_t>(),
            mean.data_ptr<scalar_t>(), config, correction);
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    });
}

void warn_invalid_degrees_of_freedom(const Tensor& self, const Tensor& output, double correction)
{
    const auto reduction_factor = output.numel() == 0 ? int64_t{ 0 } : self.numel() / output.numel();
    if (static_cast<double>(reduction_factor) - correction <= 0)
    {
        TORCH_WARN(
            "var_mean(): degrees of freedom is <= 0. Correction should be strictly less than the "
            "reduction factor (input numel divided by output numel).");
    }
}

} // namespace

std::tuple<Tensor, Tensor> var_mean(const Tensor& self,
                                    OptionalIntArrayRef dim,
                                    const std::optional<Scalar>& correction,
                                    bool keepdim)
{
    TORCH_CHECK(self.is_floating_point(), "var_mean only support floating point and complex dtypes");

    const auto reduction_mask = make_dim_mask(dim, self.dim());
    const auto config = make_reduction_config(self, reduction_mask);
    const auto sizes = shape_from_dim_mask(self, reduction_mask, keepdim);
    auto variance = at::empty(sizes, self.options());
    auto mean = at::empty(sizes, self.options());
    const auto correction_value = correction.value_or(Scalar(1)).toDouble();
    warn_invalid_degrees_of_freedom(self, variance, correction_value);

    AT_DISPATCH_V2(self.scalar_type(), "var_mean", AT_WRAP([&] {
                       using accscalar_t = at::acc_type<scalar_t, true>;
                       var_mean_async<scalar_t, accscalar_t>(
                           self, variance, mean, config, static_cast<accscalar_t>(correction_value));
                   }),
                   AT_EXPAND(AT_FLOATING_TYPES));

    return { variance, mean };
}

TORCH_LIBRARY_IMPL(aten, PrivateUse1, m)
{
    m.impl(TORCH_SELECTIVE_NAME("aten::var_mean.correction"), TORCH_FN(var_mean));
}

} // namespace dgx

} // namespace at::native
