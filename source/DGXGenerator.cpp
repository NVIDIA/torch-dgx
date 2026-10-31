/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
#include "DGXGenerator.h"

#include <c10/util/Exception.h>

#include <ATen/Functions.h>

#include <cstring>
#include <limits>
#include <mutex>

namespace at::dgx
{

namespace
{

constexpr c10::DeviceIndex kDGXDeviceIndex = 0;
constexpr uint64_t kPhiloxBlockSize = 4;

struct SerializedDGXGeneratorState
{
    uint64_t seed;
    uint64_t offset;
};

static_assert(sizeof(SerializedDGXGeneratorState) == 2 * sizeof(uint64_t));

c10::DeviceIndex normalize_device_index(c10::DeviceIndex device_index)
{
    if (device_index < 0)
    {
        return kDGXDeviceIndex;
    }
    TORCH_CHECK(device_index == kDGXDeviceIndex, "DGX only exposes logical device 0, but got device ", device_index);
    return device_index;
}

void check_philox_offset(uint64_t offset)
{
    TORCH_CHECK(offset % kPhiloxBlockSize == 0, "DGX generator offset must be a multiple of ", kPhiloxBlockSize,
                ", but got ", offset);
}

} // namespace

c10::intrusive_ptr<DGXGeneratorState> DGXGeneratorState::clone() const
{
    return c10::make_intrusive<DGXGeneratorState>(seed_, offset_);
}

DGXGeneratorImpl::DGXGeneratorImpl(c10::DeviceIndex device_index)
    : c10::GeneratorImpl(c10::Device(device_type(), normalize_device_index(device_index)),
                         c10::DispatchKeySet(c10::DispatchKey::PrivateUse1)),
      state_(c10::make_intrusive<DGXGeneratorState>(c10::default_rng_seed_val, 0))
{
}

DGXGeneratorImpl::DGXGeneratorImpl(c10::DeviceIndex device_index, c10::intrusive_ptr<DGXGeneratorState> state)
    : c10::GeneratorImpl(c10::Device(device_type(), normalize_device_index(device_index)),
                         c10::DispatchKeySet(c10::DispatchKey::PrivateUse1)),
      state_(std::move(state))
{
}

void DGXGeneratorImpl::set_current_seed(uint64_t seed)
{
    state_->seed_ = seed;
    state_->offset_ = 0;
}

void DGXGeneratorImpl::set_offset(uint64_t offset)
{
    check_philox_offset(offset);
    state_->offset_ = offset;
}

uint64_t DGXGeneratorImpl::get_offset() const
{
    return state_->offset_;
}

uint64_t DGXGeneratorImpl::current_seed() const
{
    return state_->seed_;
}

uint64_t DGXGeneratorImpl::seed()
{
    const auto seed = c10::detail::getNonDeterministicRandom(true);
    set_current_seed(seed);
    return seed;
}

void DGXGeneratorImpl::set_state(const c10::TensorImpl& new_state)
{
    at::detail::check_rng_state(new_state);
    TORCH_CHECK(new_state.numel() == sizeof(SerializedDGXGeneratorState), "Expected a DGX generator state of ",
                sizeof(SerializedDGXGeneratorState), " bytes, but found ", new_state.numel(), " bytes");

    SerializedDGXGeneratorState serialized_state;
    std::memcpy(&serialized_state, new_state.data_dtype_initialized<uint8_t>(), sizeof(serialized_state));
    check_philox_offset(serialized_state.offset);
    state_->seed_ = serialized_state.seed;
    state_->offset_ = serialized_state.offset;
}

c10::intrusive_ptr<c10::TensorImpl> DGXGeneratorImpl::get_state() const
{
    const SerializedDGXGeneratorState serialized_state{ state_->seed_, state_->offset_ };
    auto state = at::empty({ sizeof(serialized_state) }, at::TensorOptions().dtype(at::kByte).device(at::kCPU));
    std::memcpy(state.mutable_data_ptr<uint8_t>(), &serialized_state, sizeof(serialized_state));
    return state.getIntrusivePtr();
}

std::pair<uint64_t, uint64_t> DGXGeneratorImpl::philox_engine_inputs(uint64_t increment)
{
    check_philox_offset(increment);
    TORCH_CHECK(increment <= std::numeric_limits<uint64_t>::max() - state_->offset_, "DGX generator offset overflow");

    const auto offset = state_->offset_;
    state_->offset_ += increment;
    return { state_->seed_, offset };
}

DGXGeneratorImpl* DGXGeneratorImpl::clone_impl() const
{
    return new DGXGeneratorImpl(device().index(), state_->clone());
}

Generator create_dgx_generator(c10::DeviceIndex device_index)
{
    return at::make_generator<DGXGeneratorImpl>(normalize_device_index(device_index));
}

const Generator& get_default_dgx_generator(c10::DeviceIndex device_index)
{
    normalize_device_index(device_index);
    static const Generator default_generator = [] {
        auto generator = create_dgx_generator(kDGXDeviceIndex);
        generator.seed();
        return generator;
    }();
    return default_generator;
}

std::pair<uint64_t, uint64_t> philox_engine_inputs(const std::optional<Generator>& generator, uint64_t increment)
{
    auto* dgx_generator = get_generator_or_default<DGXGeneratorImpl>(generator, get_default_dgx_generator());
    std::lock_guard<std::mutex> lock(dgx_generator->mutex_);
    return dgx_generator->philox_engine_inputs(increment);
}

} // namespace at::dgx
