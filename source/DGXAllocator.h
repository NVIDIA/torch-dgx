/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
#pragma once

#include <c10/core/Allocator.h>

struct DGXAllocator final : c10::Allocator
{
    DGXAllocator() = default;
    c10::DataPtr allocate(size_t nbytes) override;
    static void Delete(void* ptr);
    c10::DeleterFnPtr raw_deleter() const override;
    void copy_data(void* dest, const void* src, std::size_t count) const final;
};
