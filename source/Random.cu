/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
#include "DGXGenerator.h"
#include "KernelLaunchUtils.cuh"
#include "StridedTensorUtils.cuh"

#include <ATen/AccumulateType.h>
#include <ATen/Dispatch_v2.h>
#include <ATen/OpMathType.h>
#include <ATen/TensorIterator.h>
#include <ATen/core/TransformationHelper.h>
#include <ATen/native/DistributionTemplates.h>
#include <ATen/ops/empty.h>
#include <ATen/ops/sort.h>

#include <torch/library.h>

#include <curand_kernel.h>

#include <limits>
#include <type_traits>

namespace at::native
{

namespace dgx
{

namespace
{

constexpr uint64_t kPhiloxOffsetPerCall = 4;

template <typename scalar_t>
struct UniformDistribution
{
    using opmath_t = at::opmath_type<scalar_t>;

    opmath_t from;
    opmath_t to;
    opmath_t range;

    __device__ scalar_t operator()(curandStatePhilox4_32_10_t* state) const
    {
        opmath_t sample;
        if constexpr (std::is_same_v<opmath_t, double>)
        {
            sample = curand_uniform_double(state);
        }
        else
        {
            sample = curand_uniform(state);
        }

        const auto value = static_cast<scalar_t>(sample * range + from);
        // cuRAND samples (0, 1], whereas uniform_ promises [from, to).
        return value == static_cast<scalar_t>(to) ? static_cast<scalar_t>(from) : value;
    }
};

template <typename scalar_t>
struct NormalDistribution
{
    using accscalar_t = at::acc_type<scalar_t, true>;

    accscalar_t mean;
    accscalar_t std;

    __device__ scalar_t operator()(curandStatePhilox4_32_10_t* state) const
    {
        accscalar_t sample;
        if constexpr (std::is_same_v<accscalar_t, double>)
        {
            sample = curand_normal_double(state);
        }
        else
        {
            sample = curand_normal(state);
        }
        return static_cast<scalar_t>(sample * std + mean);
    }
};

template <typename scalar_t>
struct RandomFromToDistribution
{
    uint64_t range;
    int64_t base;

    __device__ scalar_t operator()(curandStatePhilox4_32_10_t* state) const
    {
        // Match the CUDA backend's choice to use 64 random bits for large ranges, limiting modulo skew.
        if (range >= 1ULL << 28)
        {
            const uint4 values = curand4(state);
            const uint64_t value = (static_cast<uint64_t>(values.x) << 32) | values.y;
            return transformation::uniform_int_from_to<scalar_t>(value, range, base);
        }
        return transformation::uniform_int_from_to<scalar_t>(curand(state), range, base);
    }
};

template <typename scalar_t>
struct RandomFullRangeDistribution
{
    __device__ scalar_t operator()(curandStatePhilox4_32_10_t* state) const
    {
        const uint4 values = curand4(state);
        const uint64_t value = (static_cast<uint64_t>(values.x) << 32) | values.y;
        return transformation::uniform_int_full_range<scalar_t>(value);
    }
};

template <typename scalar_t, typename Distribution>
struct ContiguousRandomFunctor
{
    __device__ void operator()(size_t index, scalar_t* self, uint64_t seed, uint64_t offset, Distribution distribution)
    {
        curandStatePhilox4_32_10_t state;
        curand_init(seed, index, offset, &state);
        self[index] = distribution(&state);
    }
};

template <typename scalar_t, typename Distribution>
struct StridedRandomFunctor
{
    __device__ void operator()(size_t index,
                               scalar_t* self,
                               StridedOffsetCalculator<1> offsets,
                               uint64_t seed,
                               uint64_t offset,
                               Distribution distribution)
    {
        curandStatePhilox4_32_10_t state;
        curand_init(seed, index, offset, &state);
        self[offsets.get(index)[0]] = distribution(&state);
    }
};

template <typename scalar_t, typename Distribution>
void random_async(Tensor self, uint64_t seed, uint64_t offset, Distribution distribution)
{
    auto* output = self.mutable_data_ptr<scalar_t>();
    if (self.is_contiguous())
    {
        mgpu_launch_func(
            self.numel(), ContiguousRandomFunctor<scalar_t, Distribution>(), output, seed, offset, distribution);
    }
    else
    {
        const auto offsets =
            make_strided_offset_calculator<1>(self.sizes(), std::array<IntArrayRef, 1>{ self.strides() });
        mgpu_launch_func(
            self.numel(), StridedRandomFunctor<scalar_t, Distribution>(), output, offsets, seed, offset, distribution);
    }
}

__device__ uint64_t random_uint64(curandStatePhilox4_32_10_t* state)
{
    const uint4 values = curand4(state);
    return (static_cast<uint64_t>(values.x) << 32) | values.y;
}

__device__ uint64_t uniform_below(curandStatePhilox4_32_10_t* state, uint64_t bound)
{
    // Rejection sampling avoids the modulo bias that would otherwise be introduced when bound does not divide 2^64.
    const uint64_t threshold = -bound % bound;
    while (true)
    {
        const uint64_t value = random_uint64(state);
        if (value >= threshold)
            return value % bound;
    }
}

struct ShuffleDuplicateKeysFunctor
{
    __device__ void operator()(size_t index, const int64_t* keys, int64_t* indices, int64_t n, uint64_t seed, uint64_t offset)
    {
        if (index >= static_cast<size_t>(n - 1) || keys[index] != keys[index + 1] ||
            (index != 0 && keys[index] == keys[index - 1]))
            return;

        int64_t island_size = 2;
        while (index + island_size < static_cast<size_t>(n) && keys[index + island_size] == keys[index])
            ++island_size;

        curandStatePhilox4_32_10_t state;
        curand_init(seed, index, offset, &state);
        int64_t* island = indices + index;
        for (int64_t i = island_size - 1; i > 0; --i)
        {
            const int64_t random_index = static_cast<int64_t>(uniform_below(&state, static_cast<uint64_t>(i + 1)));
            if (i != random_index)
            {
                const int64_t value = island[i];
                island[i] = island[random_index];
                island[random_index] = value;
            }
        }
    }
};

void shuffle_duplicate_keys_async(const Tensor& keys, Tensor& indices, uint64_t seed, uint64_t offset)
{
    mgpu_launch_func(keys.numel(), ShuffleDuplicateKeysFunctor(), keys.const_data_ptr<int64_t>(),
                     indices.mutable_data_ptr<int64_t>(), keys.numel(), seed, offset);
}

void check_randperm_dtype(int64_t n, ScalarType dtype)
{
    switch (dtype)
    {
    case kByte:
        TORCH_CHECK(n <= static_cast<int64_t>(std::numeric_limits<uint8_t>::max()) + 1,
                    "n is too large for result tensor type: Byte");
        break;
    case kChar:
        TORCH_CHECK(n <= static_cast<int64_t>(std::numeric_limits<int8_t>::max()) + 1,
                    "n is too large for result tensor type: Char");
        break;
    case kShort:
        TORCH_CHECK(n <= static_cast<int64_t>(std::numeric_limits<int16_t>::max()) + 1,
                    "n is too large for result tensor type: Short");
        break;
    case kInt:
        TORCH_CHECK(n <= static_cast<int64_t>(std::numeric_limits<int32_t>::max()) + 1,
                    "n is too large for result tensor type: Int");
        break;
    case kLong:
        break;
    case kHalf:
        TORCH_CHECK(n <= (int64_t{ 1 } << 11) + 1, "n cannot be greater than 2049 for Half type.");
        break;
    case kFloat:
        TORCH_CHECK(n <= (int64_t{ 1 } << 24) + 1, "n cannot be greater than 2^24+1 for Float type.");
        break;
    case kDouble:
        TORCH_CHECK(n <= (int64_t{ 1 } << 53) + 1, "n cannot be greater than 2^53+1 for Double type.");
        break;
    case kBFloat16:
        TORCH_CHECK(n <= (int64_t{ 1 } << 8) + 1, "n cannot be greater than 257 for BFloat16 type.");
        break;
    default:
        TORCH_CHECK(false, "randperm is not implemented for 'dgx' with dtype ", dtype);
    }
}

template <typename RNG>
struct UniformKernel
{
    void operator()(TensorIteratorBase& iter, double from, double to, const std::optional<Generator>& generator) const
    {
        const auto [seed, offset] = at::dgx::philox_engine_inputs(generator, kPhiloxOffsetPerCall);
        Tensor self = iter.output();
        AT_DISPATCH_V2(
            self.scalar_type(), "uniform_dgx", AT_WRAP([&] {
                using opmath_t = at::opmath_type<scalar_t>;
                const auto from_value = static_cast<opmath_t>(from);
                const auto to_value = static_cast<opmath_t>(to);
                random_async<scalar_t>(
                    self, seed, offset, UniformDistribution<scalar_t>{ from_value, to_value, to_value - from_value });
            }),
            AT_EXPAND(AT_FLOATING_TYPES), kHalf, kBFloat16);
    }
};

template <typename RNG>
struct NormalKernel
{
    void operator()(const TensorBase& self, double mean, double std, const std::optional<Generator>& generator) const
    {
        auto iter = TensorIterator::borrowing_nullary_op(self);
        Tensor output = iter.output();
        AT_DISPATCH_V2(output.scalar_type(), "normal_dgx", AT_WRAP([&] {
                           const auto [seed, offset] = at::dgx::philox_engine_inputs(generator, kPhiloxOffsetPerCall);
                           using accscalar_t = at::acc_type<scalar_t, true>;
                           random_async<scalar_t>(output, seed, offset,
                                                  NormalDistribution<scalar_t>{ static_cast<accscalar_t>(mean),
                                                                                static_cast<accscalar_t>(std) });
                       }),
                       AT_EXPAND(AT_FLOATING_TYPES), kHalf, kBFloat16);
    }
};

template <typename RNG>
struct RandomFromToKernel
{
    void operator()(TensorIteratorBase& iter, uint64_t range, int64_t base, const std::optional<Generator>& generator) const
    {
        const auto [seed, offset] = at::dgx::philox_engine_inputs(generator, kPhiloxOffsetPerCall);
        Tensor self = iter.output();
        AT_DISPATCH_V2(self.scalar_type(), "random_from_to_dgx", AT_WRAP([&] {
                           random_async<scalar_t>(self, seed, offset, RandomFromToDistribution<scalar_t>{ range, base });
                       }),
                       kBool, kHalf, kBFloat16, AT_EXPAND(AT_ALL_TYPES), AT_EXPAND(AT_BAREBONES_UNSIGNED_TYPES));
    }

    void operator()(TensorIteratorBase& iter, const std::optional<Generator>& generator) const
    {
        Tensor self = iter.output();
        AT_DISPATCH_ALL_TYPES_AND(at::ScalarType::BFloat16, self.scalar_type(), "random_full_range_dgx", [&] {
            if constexpr (std::is_same_v<scalar_t, int64_t> || std::is_same_v<scalar_t, double> ||
                          std::is_same_v<scalar_t, float> || std::is_same_v<scalar_t, at::BFloat16>)
            {
                const auto [seed, offset] = at::dgx::philox_engine_inputs(generator, kPhiloxOffsetPerCall);
                random_async<scalar_t>(self, seed, offset, RandomFullRangeDistribution<scalar_t>{});
            }
            else
            {
                TORCH_CHECK(false, "random_full_range_dgx handles only int64, double, float, and bfloat16 tensors");
            }
        });
    }
};

} // namespace

Tensor& uniform_(Tensor& self, double from, double to, std::optional<Generator> generator)
{
    // A DGX tensor spans every physical GPU but presents one logical device. Each logical element owns a Philox
    // subsequence, while the generator offset separates consecutive operations. This makes the result independent of
    // how elements are partitioned among physical GPUs and lets explicit and default DGX generators share the normal
    // PyTorch state semantics.
    return at::native::templates::uniform_impl_<UniformKernel, at::dgx::DGXGeneratorImpl>(
        self, from, to, std::move(generator));
}

Tensor& normal_(Tensor& self, double mean, double std, std::optional<Generator> generator)
{
    return at::native::templates::normal_impl_<NormalKernel, at::dgx::DGXGeneratorImpl>(
        self, mean, std, std::move(generator));
}

Tensor& random_from_to_(Tensor& self, int64_t from, std::optional<int64_t> to, std::optional<Generator> generator)
{
    return at::native::templates::random_from_to_impl<RandomFromToKernel, at::dgx::DGXGeneratorImpl>(
        self, from, to, std::move(generator));
}

Tensor& randperm_out(int64_t n, std::optional<Generator> generator, Tensor& out)
{
    TORCH_CHECK(n >= 0, "n must be non-negative, got ", n);
    check_randperm_dtype(n, out.scalar_type());

    out.resize_({ n });
    if (n == 0)
        return out;

    // Sorting independent random keys gives a random ordering of the values 0..n-1. A stable tie order would bias the
    // result when keys collide, so equal-key islands are randomly shuffled after sorting.
    Tensor keys = at::empty({ n }, out.options().dtype(kLong));
    const auto [seed, offset] = at::dgx::philox_engine_inputs(generator, kPhiloxOffsetPerCall);
    random_async<int64_t>(keys, seed, offset, RandomFullRangeDistribution<int64_t>{});

    auto [sorted_keys, indices] = at::sort(keys, std::optional<bool>{ false }, -1, false);
    const auto [shuffle_seed, shuffle_offset] = at::dgx::philox_engine_inputs(generator, kPhiloxOffsetPerCall);
    shuffle_duplicate_keys_async(sorted_keys, indices, shuffle_seed, shuffle_offset);
    out.copy_(indices);
    return out;
}

TORCH_LIBRARY_IMPL(aten, PrivateUse1, m)
{
    m.impl(TORCH_SELECTIVE_NAME("aten::uniform_"), TORCH_FN(uniform_));
    m.impl(TORCH_SELECTIVE_NAME("aten::normal_"), TORCH_FN(normal_));
    m.impl(TORCH_SELECTIVE_NAME("aten::random_.from"), TORCH_FN(random_from_to_));
    m.impl(TORCH_SELECTIVE_NAME("aten::randperm.generator_out"), TORCH_FN(randperm_out));
}

} // namespace dgx

} // namespace at::native
