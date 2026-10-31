/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
#include "Resize.h"

#include "EventUtils.h"
#include "TensorUtils.h"

#include <c10/cuda/CUDAStream.h>

#include <ATen/EmptyTensor.h>
#include <ATen/native/Resize.h>
#include <ATen/ops/set_native.h>

#include <torch/library.h>

#include <cuda_runtime_api.h>

namespace at::native
{

namespace dgx
{

namespace
{

void resize_bytes_dgx_impl(StorageImpl* storage, size_t size_bytes)
{
    TORCH_CHECK(storage->resizable(), "Trying to resize storage that is not resizable");
    auto allocator = storage->allocator();
    TORCH_CHECK(allocator != nullptr, "Trying to resize storage without an allocator");

    int current_device = cudaInvalidDeviceId;
    C10_CUDA_CHECK(cudaGetDevice(&current_device));

    c10::Device device = storage->device();

    if (size_bytes == 0)
    {
        // The allocator orders release after pending work on all current streams.
        storage->set_data_ptr_noswap(at::DataPtr(nullptr, device));
        storage->set_nbytes(0);
        C10_CUDA_CHECK(cudaSetDevice(current_device));
        return;
    }

    at::DataPtr data = allocator->allocate(size_bytes);
    if (storage->data_ptr())
    {
        // Copy disjoint byte ranges on each device's current stream, as in
        // Copy.cu. Prior DGX operations join their streams before returning,
        // and DGXAllocator orders allocation readiness on all current streams,
        // so no additional join is needed before the copies.
        const auto copy_nbytes = std::min(storage->nbytes(), size_bytes);
        for (const auto device_id : c10::irange(c10::cuda::device_count()))
        {
            const auto [device_offset, device_nbytes] = chunk(copy_nbytes, device_id);
            if (device_nbytes)
            {
                C10_CUDA_CHECK(cudaSetDevice(device_id));
                auto stream = c10::cuda::getCurrentCUDAStream(device_id);
                C10_CUDA_CHECK(cudaMemcpyAsync(static_cast<uint8_t*>(data.get()) + device_offset,
                    static_cast<const uint8_t*>(storage->data()) + device_offset,
                    device_nbytes, cudaMemcpyDefault, stream));
            }
        }

        // Future work on any device must see every preserved byte. The waits
        // are device-side; publishing the new storage needs no host wait.
        merge_streams();
    }

    // DGXAllocator orders the old allocation's release after all per-device copies.
    storage->set_data_ptr_noswap(std::move(data));
    storage->set_nbytes(size_bytes);
    C10_CUDA_CHECK(cudaSetDevice(current_device));
}

} // namespace

void resize_bytes_dgx(const c10::Storage& storage, size_t size_bytes)
{
    resize_bytes_dgx_impl(storage.unsafeGetStorageImpl(), size_bytes);
}

void maybe_resize_storage_dgx(TensorImpl* self, size_t new_size_bytes)
{
    // It does not make sense to try to resize a storage
    // to hold 0 elements, and this can break
    // if storage_offset is positive but
    // new_size is 0, so just bail in that case
    // (same comment is in Resize.h)
    if (self->numel() == 0)
    {
        return;
    }

    const Storage& storage = self->unsafe_storage();
    TORCH_CHECK(storage, "Tensor: invalid null storage");
    if (new_size_bytes > storage.nbytes())
    {
        resize_bytes_dgx_impl(storage.unsafeGetStorageImpl(), new_size_bytes);
    }
}

TensorImpl* resize_impl_dgx_(TensorImpl* self, IntArrayRef size, at::OptionalIntArrayRef stride)
{
    if (self->sizes() == size && (!stride || self->strides() == stride))
    {
        return self;
    }
    const auto itemsize = self->dtype().itemsize();
    const auto storage_offset = self->storage_offset();
    size_t storage_size = 1;
    if (stride)
    {
        self->set_sizes_and_strides(size, *stride);
        storage_size = at::detail::computeStorageNbytes(size, *stride, itemsize, storage_offset);
    }
    else
    {
        self->set_sizes_contiguous(size);
        storage_size = at::detail::computeStorageNbytesContiguous(size, itemsize, storage_offset);
    }
    maybe_resize_storage_dgx(self, storage_size);

    return self;
}

const Tensor& resize_(const Tensor& self, IntArrayRef size, std::optional<MemoryFormat> optional_memory_format)
{
    TORCH_CHECK(!optional_memory_format.has_value())

    auto* self_ = self.unsafeGetTensorImpl();
    resize_impl_dgx_(self_, size, std::nullopt);

    return self;
}

Tensor& set_(Tensor& result, Storage source)
{
    return at::native::set_(result, source);
}

Tensor& set_tensor_(Tensor& result, const Tensor& source)
{
    return at::native::set_tensor_(result, source);
}

Tensor& set_storage_(Tensor& result, Storage storage, int64_t storage_offset, IntArrayRef size, IntArrayRef stride)
{
    checkSetStorage(result, std::move(storage), storage_offset, size, stride);

    result.unsafeGetTensorImpl()->set_storage_offset(storage_offset);
    at::OptionalIntArrayRef stride_opt = stride.data() != nullptr ? at::OptionalIntArrayRef(stride) : std::nullopt;
    resize_impl_dgx_(result.unsafeGetTensorImpl(), size, stride_opt);
    return result;
}

TORCH_LIBRARY_IMPL(aten, PrivateUse1, m)
{
    m.impl(TORCH_SELECTIVE_NAME("aten::resize_"), TORCH_FN(resize_));
    m.impl(TORCH_SELECTIVE_NAME("aten::set_.source_Storage"), TORCH_FN(set_));
    m.impl(TORCH_SELECTIVE_NAME("aten::set_.source_Storage_storage_offset"), TORCH_FN(set_storage_));
    m.impl(TORCH_SELECTIVE_NAME("aten::set_.source_Tensor"), TORCH_FN(set_tensor_));
}

} // namespace dgx

} // namespace at::native
