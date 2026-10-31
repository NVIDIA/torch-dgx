/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
#include "TensorUtils.h"

#include <ATen/EmptyTensor.h>
#include <ATen/ops/as_strided_native.h>
#include <ATen/ops/view_as_complex_native.h>
#include <ATen/ops/view_as_real_native.h>
#include <ATen/ops/view_native.h>

#include <torch/library.h>

namespace at::native
{

namespace dgx
{

Tensor empty(IntArrayRef size,
             std::optional<ScalarType> dtype,
             std::optional<Layout> layout,
             std::optional<Device> device,
             std::optional<bool> pin_memory,
             std::optional<MemoryFormat> memory_format)
{
    auto* allocator = c10::GetAllocator(c10::DeviceType::PrivateUse1);
    constexpr c10::DispatchKeySet private_use_dks(c10::DispatchKey::PrivateUse1);
    auto out = detail::empty_generic(size, allocator, private_use_dks, c10::dtype_or_default(dtype), memory_format);
    return out;
}

Tensor empty_strided(c10::IntArrayRef size,
                     c10::IntArrayRef stride,
                     std::optional<ScalarType> dtype_opt,
                     std::optional<Layout> layout_opt,
                     std::optional<Device> device_opt,
                     std::optional<bool> pin_memory_opt)
{
    auto dtype = c10::dtype_or_default(dtype_opt);
    auto* allocator = c10::GetAllocator(c10::DeviceType::PrivateUse1);
    constexpr c10::DispatchKeySet private_use_dks(c10::DispatchKey::PrivateUse1);
    auto out = detail::empty_strided_generic(size, stride, allocator, private_use_dks, dtype);
    return out;
}

Tensor as_strided(const Tensor& self, IntArrayRef size, IntArrayRef stride, std::optional<int64_t> storage_offset_)
{
    return at::native::as_strided_tensorimpl(self, size, stride, storage_offset_);
}

Tensor view(const Tensor& self, IntArrayRef size)
{
    return at::native::view(self, size);
}

Tensor view_as_complex(const Tensor& self)
{
    return at::native::view_as_complex(self);
}

Tensor view_as_real(const Tensor& self)
{
    return at::native::view_as_real(self);
}

TORCH_LIBRARY_IMPL(aten, PrivateUse1, m)
{
    m.impl(TORCH_SELECTIVE_NAME("aten::empty.memory_format"), TORCH_FN(empty));
    m.impl(TORCH_SELECTIVE_NAME("aten::empty_strided"), TORCH_FN(empty_strided));
    m.impl(TORCH_SELECTIVE_NAME("aten::as_strided"), TORCH_FN(as_strided));
    m.impl(TORCH_SELECTIVE_NAME("aten::view"), TORCH_FN(view));
    m.impl(TORCH_SELECTIVE_NAME("aten::view_as_complex"), TORCH_FN(view_as_complex));
    m.impl(TORCH_SELECTIVE_NAME("aten::view_as_real"), TORCH_FN(view_as_real));
}

} // namespace dgx

} // namespace at::native
