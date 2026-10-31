/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
#include "DGXGenerator.h"
#include "KernelLaunchUtils.cuh"

#include <ATen/CPUGeneratorImpl.h>
#include <ATen/Dispatch_v2.h>
#include <ATen/core/PhiloxRNGEngine.h>
#include <ATen/ops/cumsum.h>
#include <ATen/ops/empty.h>
#include <ATen/ops/max.h>
#include <ATen/ops/min.h>
#include <ATen/ops/multinomial.h>
#include <ATen/ops/sum.h>

#include <torch/library.h>

#include <cuda/atomic>

#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <utility>

namespace at::native
{

namespace dgx
{

namespace
{

namespace cg = cooperative_groups;

constexpr uint64_t kPhiloxOffsetPerCall = 4;
constexpr int64_t kMaxMultinomialCategories = 1 << 24;

template <typename scalar_t>
struct MultinomialFunctor
{
    __device__ void operator()(size_t tid,
                               const scalar_t* probabilities,
                               const scalar_t* cumulative_probabilities,
                               int64_t num_categories,
                               const scalar_t* uniform_samples,
                               int64_t* output)
    {
        const scalar_t total = cumulative_probabilities[num_categories - 1];
        const scalar_t target = uniform_samples[tid] * total;

        int64_t left = 0;
        int64_t right = num_categories;
        while (left < right)
        {
            const int64_t middle = left + (right - left) / 2;
            if (cumulative_probabilities[middle] <= target)
                left = middle + 1;
            else
                right = middle;
        }

        // Rounding in low-precision cumulative sums can put the target at or
        // just above the final CDF entry. Select the last nonzero category in
        // that case rather than a zero-probability trailing category.
        if (left == num_categories)
        {
            left = num_categories - 1;
            while (left > 0 && probabilities[left] == scalar_t(0))
                --left;
        }
        output[tid] = left;
    }
};

template <typename output_t>
__device__ void atomic_add(output_t* output, output_t value)
{
    // Each histogram location is updated by only one GPU.
    ::cuda::atomic_ref<output_t, ::cuda::thread_scope_device> count(*output);
    count.fetch_add(value, ::cuda::memory_order_relaxed);
}

template <typename input_t>
struct BincountFunctor
{
    __device__ void operator()(size_t tid, const input_t* input, int64_t first_bin, int64_t last_bin, int64_t* output)
    {
        const int64_t bin = static_cast<int64_t>(input[tid]);
        if (bin < first_bin || bin >= last_bin)
            return;

        const auto active_threads = cg::coalesced_threads();
        const auto matching_bin_threads = cg::labeled_partition(active_threads, bin);
        if (matching_bin_threads.thread_rank() == 0)
            atomic_add(output + bin, static_cast<int64_t>(matching_bin_threads.size()));
    }
};

template <typename input_t, typename weight_t>
struct WeightedBincountFunctor
{
    __device__ void operator()(
        size_t tid, const input_t* input, const weight_t* weights, int64_t first_bin, int64_t last_bin, weight_t* output)
    {
        const int64_t bin = static_cast<int64_t>(input[tid]);
        if (bin < first_bin || bin >= last_bin)
            return;

        const auto active_threads = cg::coalesced_threads();
        const auto matching_bin_threads = cg::labeled_partition(active_threads, bin);
        const weight_t aggregate = cg::reduce(matching_bin_threads, weights[tid], cg::plus<weight_t>());
        if (matching_bin_threads.thread_rank() == 0)
            atomic_add(output + bin, aggregate);
    }
};

template <typename output_t>
struct ReduceBincountFunctor
{
    __device__ void operator()(size_t bin,
                               const output_t* partial_histograms,
                               int64_t num_histograms,
                               int64_t histogram_stride,
                               output_t* output)
    {
        output_t total = 0;
        for (int64_t histogram = 0; histogram < num_histograms; ++histogram)
            total += partial_histograms[histogram * histogram_stride + bin];
        output[bin] = total;
    }
};

template <typename input_t, typename output_t>
void launch_bincount(const Tensor& input, const Tensor* weights, Tensor& output)
{
    const int64_t num_devices = c10::cuda::device_count();
    const int64_t num_inputs = input.numel();
    const int64_t num_bins = output.numel();
    constexpr int64_t page_elements = kPageSize / sizeof(output_t);

    // Private histograms avoid replicated input scans when their dense
    // initialization and reduction cost less than the extra scans required by
    // output ownership. Sparse histograms instead assign disjoint output pages
    // to devices without allocating num_devices copies of a large minlength.
    const bool use_private_histograms =
        static_cast<long double>(num_devices) * num_bins < static_cast<long double>(num_devices - 1) * num_inputs;

    if (use_private_histograms)
    {
        const int64_t histogram_stride = page_elements * ((num_bins + page_elements - 1) / page_elements);
        TORCH_CHECK(histogram_stride <= std::numeric_limits<int64_t>::max() / num_devices,
                    "bincount private histogram size overflow");

        Tensor partial_storage = at::empty({ num_devices * histogram_stride }, output.options());
        output_t* partial_histograms = partial_storage.data_ptr<output_t>();

        parallel_for_each([&](int device, cudaStream_t stream) {
            output_t* partial_histogram = partial_histograms + device * histogram_stride;
            C10_CUDA_CHECK(cudaMemsetAsync(partial_histogram, 0, histogram_stride * sizeof(output_t), stream));

            int64_t input_offset, device_num_inputs;
            std::tie(input_offset, device_num_inputs) = chunk(num_inputs, device);
            const unsigned int num_blocks = ::cuda::ceil_div(device_num_inputs, kNumThreads);
            if (!num_blocks)
                return;

            if (weights)
            {
                offset_lambda_kernel<<<num_blocks, kNumThreads, 0, stream>>>(
                    device_num_inputs, input_offset, WeightedBincountFunctor<input_t, output_t>(),
                    input.const_data_ptr<input_t>(), weights->const_data_ptr<output_t>(), 0, num_bins, partial_histogram);
            }
            else
            {
                offset_lambda_kernel<<<num_blocks, kNumThreads, 0, stream>>>(
                    device_num_inputs, input_offset, BincountFunctor<input_t>(), input.const_data_ptr<input_t>(), 0,
                    num_bins, reinterpret_cast<int64_t*>(partial_histogram));
            }
            C10_CUDA_KERNEL_LAUNCH_CHECK();
        });

        parallel_for_each([&](int device, cudaStream_t stream) {
            int64_t bin_offset, device_num_bins;
            std::tie(bin_offset, device_num_bins) = aligned_chunk(page_elements, num_bins, device);
            const unsigned int num_blocks = ::cuda::ceil_div(device_num_bins, kNumThreads);
            if (!num_blocks)
                return;

            offset_lambda_kernel<<<num_blocks, kNumThreads, 0, stream>>>(
                device_num_bins, bin_offset, ReduceBincountFunctor<output_t>(), partial_histograms, num_devices,
                histogram_stride, output.data_ptr<output_t>());
            C10_CUDA_KERNEL_LAUNCH_CHECK();
        });
    }
    else
    {
        parallel_for_each([&](int device, cudaStream_t stream) {
            int64_t first_bin, device_num_bins;
            std::tie(first_bin, device_num_bins) = aligned_chunk(page_elements, num_bins, device);
            if (!device_num_bins)
                return;

            output_t* output_data = output.data_ptr<output_t>();
            C10_CUDA_CHECK(cudaMemsetAsync(output_data + first_bin, 0, device_num_bins * sizeof(output_t), stream));

            const unsigned int num_blocks = ::cuda::ceil_div(num_inputs, kNumThreads);
            if (weights)
            {
                offset_lambda_kernel<<<num_blocks, kNumThreads, 0, stream>>>(
                    num_inputs, 0, WeightedBincountFunctor<input_t, output_t>(), input.const_data_ptr<input_t>(),
                    weights->const_data_ptr<output_t>(), first_bin, first_bin + device_num_bins, output_data);
            }
            else
            {
                offset_lambda_kernel<<<num_blocks, kNumThreads, 0, stream>>>(
                    num_inputs, 0, BincountFunctor<input_t>(), input.const_data_ptr<input_t>(), first_bin,
                    first_bin + device_num_bins, reinterpret_cast<int64_t*>(output_data));
            }
            C10_CUDA_KERNEL_LAUNCH_CHECK();
        });
    }
}

Generator cpu_generator_from_dgx(const std::optional<Generator>& generator)
{
    const auto [seed, offset] = at::dgx::philox_engine_inputs(generator, kPhiloxOffsetPerCall);
    at::Philox4_32 philox(seed, 0, offset);
    const uint64_t high = philox();
    const uint64_t low = philox();
    const uint64_t cpu_seed = (high << 32) | low;
    return at::detail::createCPUGenerator(cpu_seed);
}

Tensor multinomial_cpu(const Tensor& self, int64_t num_samples, bool replacement, const std::optional<Generator>& generator)
{
    const Tensor cpu_self = self.to(kCPU);
    const Tensor validation_probabilities = cpu_self.to(kDouble);
    const double minimum = at::min(validation_probabilities).item<double>();
    const Tensor row_totals = validation_probabilities.sum(-1);
    const double minimum_total = at::min(row_totals).item<double>();
    const double maximum_total = at::max(row_totals).item<double>();
    TORCH_CHECK(std::isfinite(minimum) && minimum >= 0 && std::isfinite(minimum_total) && std::isfinite(maximum_total),
                "probability tensor contains either `inf`, `nan` or element < 0");
    TORCH_CHECK(minimum_total > 0, "invalid multinomial distribution (sum of probabilities <= 0)");

    auto cpu_generator = cpu_generator_from_dgx(generator);
    return at::multinomial(cpu_self, num_samples, replacement, std::move(cpu_generator)).to(self.device());
}

} // namespace

Tensor multinomial(const Tensor& self, int64_t num_samples, bool replacement, std::optional<Generator> generator)
{
    TORCH_CHECK(num_samples > 0, "cannot sample n_sample <= 0 samples");
    TORCH_CHECK(self.dim() == 1 || self.dim() == 2, "prob_dist must be 1 or 2 dim");
    TORCH_CHECK(isFloatingType(self.scalar_type()),
                "multinomial only supports floating-point dtypes for input, got: ", self.scalar_type());
    const int64_t num_categories = self.size(-1);
    TORCH_CHECK(replacement || num_samples <= num_categories,
                "cannot sample n_sample > prob_dist.size(-1) samples without replacement");
    TORCH_CHECK(num_categories <= kMaxMultinomialCategories, "number of categories cannot exceed 2^24");

    if (self.dim() == 2 && self.size(0) == 0)
        return at::empty({ 0, num_samples }, self.options().dtype(kLong));

    TORCH_CHECK(num_categories > 0, "invalid multinomial distribution (sum of probabilities <= 0)");

    if (!replacement || self.dim() != 1 || (self.scalar_type() != kFloat && self.scalar_type() != kDouble))
    {
        return multinomial_cpu(self, num_samples, replacement, generator);
    }

    TORCH_CHECK(self.numel() > 0, "invalid multinomial distribution (sum of probabilities <= 0)");

    const Tensor probabilities = self.contiguous();
    const double minimum = at::min(probabilities).item<double>();
    const double total = at::sum(probabilities).item<double>();
    TORCH_CHECK(std::isfinite(minimum) && std::isfinite(total) && minimum >= 0,
                "probability tensor contains either `inf`, `nan` or element < 0");
    TORCH_CHECK(total > 0, "invalid multinomial distribution (sum of probabilities <= 0)");

    const Tensor cumulative_probabilities = at::cumsum(probabilities, 0);
    Tensor uniform_samples = at::empty({ num_samples }, probabilities.options());
    uniform_samples.uniform_(0.0, 1.0, std::move(generator));
    Tensor output = at::empty({ num_samples }, self.options().dtype(kLong));

    AT_DISPATCH_V2(self.scalar_type(), "multinomial", AT_WRAP([&] {
                       mgpu_launch_func(num_samples, MultinomialFunctor<scalar_t>(),
                                        probabilities.const_data_ptr<scalar_t>(),
                                        cumulative_probabilities.const_data_ptr<scalar_t>(), probabilities.numel(),
                                        uniform_samples.const_data_ptr<scalar_t>(), output.data_ptr<int64_t>());
                   }),
                   AT_EXPAND(AT_FLOATING_TYPES));

    return output;
}

Tensor bincount(const Tensor& self, const std::optional<Tensor>& weights, int64_t minlength)
{
    TORCH_CHECK(minlength >= 0, "minlength should be >= 0");
    TORCH_CHECK(self.dim() == 1, "bincount only supports 1-d non-negative integral inputs");
    TORCH_CHECK(isIntegralType(self.scalar_type(), false), "bincount only supports integral inputs");

    const Tensor input = self.contiguous();
    ScalarType output_type = kLong;
    std::optional<Tensor> contiguous_weights;
    if (weights.has_value())
    {
        TORCH_CHECK(weights->device() == self.device(), "bincount expected weights to be on ", self.device(),
                    " but got ", weights->device());
        TORCH_CHECK(weights->dim() == 1 && weights->numel() == self.numel(),
                    "weights should be 1-d and have the same length as input");
        output_type = weights->scalar_type() == kFloat ? kFloat : kDouble;
        contiguous_weights = weights->to(output_type).contiguous();
    }

    int64_t output_size = minlength;
    if (input.numel() > 0)
    {
        const int64_t minimum = at::min(input).item<int64_t>();
        TORCH_CHECK(minimum >= 0, "bincount only supports 1-d non-negative integral inputs");
        const int64_t maximum = at::max(input).item<int64_t>();
        TORCH_CHECK(maximum < std::numeric_limits<int64_t>::max(), "maximum value cannot be represented safely");
        output_size = std::max(output_size, maximum + 1);
    }

    Tensor output = at::empty({ output_size }, self.options().dtype(output_type));
    if (input.numel() == 0)
        return output.zero_();

    AT_DISPATCH_V2(input.scalar_type(), "bincount", AT_WRAP([&] {
                       if (!contiguous_weights.has_value())
                       {
                           launch_bincount<scalar_t, int64_t>(input, nullptr, output);
                       }
                       else if (output_type == kFloat)
                       {
                           launch_bincount<scalar_t, float>(input, &contiguous_weights.value(), output);
                       }
                       else
                       {
                           launch_bincount<scalar_t, double>(input, &contiguous_weights.value(), output);
                       }
                   }),
                   AT_EXPAND(AT_INTEGRAL_TYPES));

    return output;
}

TORCH_LIBRARY_IMPL(aten, PrivateUse1, m)
{
    m.impl(TORCH_SELECTIVE_NAME("aten::multinomial"), TORCH_FN(multinomial));
    m.impl(TORCH_SELECTIVE_NAME("aten::bincount"), TORCH_FN(bincount));
}

} // namespace dgx

} // namespace at::native
