/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
#include "DGXAllocator.h"
#include "DGXDeviceGuard.h"
#include "DGXGenerator.h"
#include "Resize.h"

#include <ATen/cuda/PinnedMemoryAllocator.h>
#include <ATen/detail/PrivateUse1HooksInterface.h>
#include <ATen/native/CPUFallback.h>

#include <torch/extension.h>

#include <iostream>

struct DGXHooksArgs : at::PrivateUse1HooksArgs
{
};

struct DGXHooksInterface : public at::PrivateUse1HooksInterface
{
    DGXHooksInterface(DGXHooksArgs){};
    ~DGXHooksInterface() override = default;

    bool hasPrimaryContext(c10::DeviceIndex device_index) const override
    {
        return false;
    }

    at::Allocator* getPinnedMemoryAllocator() const override
    {
        return at::cuda::getPinnedMemoryAllocator();
    }

    const at::Generator& getDefaultGenerator(c10::DeviceIndex device_index) const override
    {
        return at::dgx::get_default_dgx_generator(device_index);
    }

    at::Generator getNewGenerator(c10::DeviceIndex device_index = -1) const override
    {
        return at::dgx::create_dgx_generator(device_index);
    }

    void resizePrivateUse1Bytes(const c10::Storage& storage, size_t newsize) const override
    {
        at::native::dgx::resize_bytes_dgx(storage, newsize);
    }
};

int register_hook()
{
    at::RegisterPrivateUse1HooksInterface(new DGXHooksInterface(DGXHooksArgs{}));
    return 0;
}
int temp_register_hook = register_hook();

void cpu_fallback(const c10::OperatorHandle& op, torch::jit::Stack* stack)
{
    TORCH_WARN("The operator '", op.schema().operator_name(), "' is not currently supported ",
               "on the dgx backend and will fall back to run on the CPU.");

    for (const auto device_id : c10::irange(c10::cuda::device_count()))
    {
        c10::cuda::getCurrentCUDAStream(device_id).synchronize();
    }

    at::native::cpu_fallback(op, stack, /*error_on_views=*/true);
}

TORCH_LIBRARY_IMPL(_, PrivateUse1, m)
{
    m.fallback(torch::CppFunction::makeFromBoxedFunction<&cpu_fallback>());
}

void autograd_fallback(const c10::OperatorHandle& op, c10::DispatchKeySet dispatch_keys, torch::jit::Stack* stack)
{
    // PyTorch has separate builds, some of which don't include autograd.
    // So we define some behavior for when autograd isn't included and
    // go through a layer of indirection (VariableHooksInterface) when it is.
    // See aten/src/ATen/core/VariableHooksInterface.h for more details.
    if (!at::impl::HasVariableHooks())
    {
        op.redispatchBoxed(dispatch_keys & c10::after_autograd_keyset, stack);
        return;
    }
    at::impl::GetVariableHooks()->basic_autograd_not_implemented_fallback(op, dispatch_keys, stack);
}

TORCH_LIBRARY_IMPL(_, AutogradPrivateUse1, m)
{
    m.fallback(torch::CppFunction::makeFromBoxedFunction<&autograd_fallback>());
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("_get_default_generator", [] { return at::dgx::get_default_dgx_generator(); });
}
