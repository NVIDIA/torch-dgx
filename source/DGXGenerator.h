/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
#pragma once

#include <c10/core/GeneratorImpl.h>

#include <ATen/core/Generator.h>

#include <cstdint>
#include <optional>
#include <utility>

namespace at::dgx
{

struct DGXGeneratorState : public c10::intrusive_ptr_target
{
    DGXGeneratorState(uint64_t seed, uint64_t offset) : seed_(seed), offset_(offset)
    {
    }

    c10::intrusive_ptr<DGXGeneratorState> clone() const;

    uint64_t seed_;
    uint64_t offset_;
};

class DGXGeneratorImpl final : public c10::GeneratorImpl
{
public:
    explicit DGXGeneratorImpl(c10::DeviceIndex device_index = -1);
    DGXGeneratorImpl(c10::DeviceIndex device_index, c10::intrusive_ptr<DGXGeneratorState> state);
    ~DGXGeneratorImpl() override = default;

    void set_current_seed(uint64_t seed) override;
    void set_offset(uint64_t offset) override;
    uint64_t get_offset() const override;
    uint64_t current_seed() const override;
    uint64_t seed() override;
    void set_state(const c10::TensorImpl& new_state) override;
    c10::intrusive_ptr<c10::TensorImpl> get_state() const override;

    std::pair<uint64_t, uint64_t> philox_engine_inputs(uint64_t increment);

    static c10::DeviceType device_type()
    {
        return c10::DeviceType::PrivateUse1;
    }

private:
    DGXGeneratorImpl* clone_impl() const override;

    c10::intrusive_ptr<DGXGeneratorState> state_;
};

const Generator& get_default_dgx_generator(c10::DeviceIndex device_index = -1);
Generator create_dgx_generator(c10::DeviceIndex device_index = -1);
std::pair<uint64_t, uint64_t> philox_engine_inputs(const std::optional<Generator>& generator, uint64_t increment);

} // namespace at::dgx
