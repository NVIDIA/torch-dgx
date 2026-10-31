/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
#pragma once

#include <c10/core/impl/DeviceGuardImplInterface.h>
#include <c10/cuda/CUDAFunctions.h>

namespace at
{
namespace detail
{

struct DGXDeviceGuardImpl final : public c10::impl::DeviceGuardImplInterface
{
    DGXDeviceGuardImpl() = default;

    DeviceType type() const override
    {
        return DeviceType::PrivateUse1;
    }

    Device exchangeDevice(Device device) const override
    {
        setDevice(device);
        return Device(DeviceType::PrivateUse1, 0);
    }

    Device getDevice() const override
    {
        return Device(DeviceType::PrivateUse1, 0);
    }

    void setDevice(Device device) const override
    {
        TORCH_CHECK(device.type() == DeviceType::PrivateUse1 && device.index() == 0,
                    "DGX only exposes logical device dgx:0, but got ", device);
    }

    void uncheckedSetDevice(Device) const noexcept override
    {
    }

    Stream getStream(Device device) const override
    {
        TORCH_CHECK(device.type() == DeviceType::PrivateUse1 && (device.index() == -1 || device.index() == 0),
                    "DGX only exposes logical device dgx:0, but got ", device);
        return Stream(Stream::DEFAULT, Device(DeviceType::PrivateUse1, 0));
    }

    Stream getDefaultStream(Device device) const override
    {
        return getStream(device);
    }

    Stream getNewStream(Device, int priority = 0) const override
    {
        TORCH_CHECK(false, "dgx backend doesn't support getNewStream")
        return Stream(Stream::DEFAULT, Device(DeviceType::PrivateUse1, 0));
    }

    Stream exchangeStream(Stream stream) const override
    {
        auto default_stream = getDefaultStream(stream.device());
        TORCH_CHECK(stream == default_stream, "DGX only supports the default stream");
        return default_stream;
    }

    DeviceIndex deviceCount() const noexcept override
    {
        return c10::cuda::device_count() > 0 ? 1 : 0;
    }
};

C10_REGISTER_GUARD_IMPL(PrivateUse1, DGXDeviceGuardImpl);

} // namespace detail
} // namespace at
