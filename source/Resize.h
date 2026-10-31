/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
#pragma once

#include <c10/core/Storage.h>

#include <cstddef>

namespace at::native::dgx
{

void resize_bytes_dgx(const c10::Storage& storage, size_t size_bytes);

} // namespace at::native::dgx
