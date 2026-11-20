/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
#include "KernelLaunchUtils.cuh"

#include <ATen/native/GridSamplerUtils.h>
#include <ATen/native/cuda/GridSampler.h>
#include <ATen/ops/empty.h>
#include <ATen/ops/from_blob.h>
#include <ATen/ops/zeros.h>

#include <torch/library.h>

namespace at::native
{

namespace dgx
{

namespace
{

void check_grid_sampler(const Tensor& input, const Tensor& grid, int64_t interpolation_mode, int64_t padding_mode)
{
    // Check ranks before the common checks index tensor dimensions.
    check_grid_sampler_2d(input, grid);
    check_grid_sampler_common(input, grid);
    TORCH_CHECK(
        input.scalar_type() == grid.scalar_type(), "grid_sampler(): expected input and grid to have the same dtype");
    TORCH_CHECK(input.scalar_type() == kFloat || input.scalar_type() == kDouble || input.scalar_type() == kHalf ||
                    input.scalar_type() == kBFloat16,
                "grid_sampler(): expected a floating point input");
    TORCH_CHECK(interpolation_mode >= 0 && interpolation_mode <= 2, "grid_sampler(): invalid interpolation mode ",
                interpolation_mode);
    TORCH_CHECK(padding_mode >= 0 && padding_mode <= 2, "grid_sampler(): invalid padding mode ", padding_mode);
}

// These non-owning views preserve strides and storage offsets, including for
// noncontiguous inputs.
Tensor make_view(const Tensor& tensor, int64_t dim, int64_t start, int64_t length, c10::Device device)
{
    if (!tensor.defined())
    {
        return Tensor();
    }

    auto sizes = tensor.sizes().vec();
    sizes[dim] = length;
    auto* data = static_cast<char*>(tensor.data_ptr()) + start * tensor.stride(dim) * tensor.element_size();
    // Managed pointers may report their allocation device rather than the GPU
    // executing this shard. Set the view's target device explicitly.
    return at::for_blob(data, sizes)
        .strides(tensor.strides())
        .options(tensor.options().device(device))
        .target_device(device)
        .make_tensor();
}

} // namespace

Tensor grid_sampler_2d(
    const Tensor& input, const Tensor& grid, int64_t interpolation_mode, int64_t padding_mode, bool align_corners)
{
    check_grid_sampler(input, grid, interpolation_mode, padding_mode);
    auto output = at::empty({ input.size(0), input.size(1), grid.size(1), grid.size(2) }, input.options());
    if (output.numel() == 0)
    {
        return output;
    }

    // Split output rows so even single-image workloads use multiple GPUs.
    parallel_for_each([&](int device, cudaStream_t stream) {
        size_t start, length;
        std::tie(start, length) = partition::range(grid.size(1), device);
        if (length == 0)
        {
            return;
        }

        const c10::Device cuda_device(c10::DeviceType::CUDA, device);
        auto cuda_input = make_view(input, 0, 0, input.size(0), cuda_device);
        auto cuda_grid = make_view(grid, 1, start, length, cuda_device);
        auto cuda_output = make_view(output, 2, start, length, cuda_device);
        launch_grid_sampler_2d_forward_kernel(
            cuda_output, cuda_input, cuda_grid, interpolation_mode, padding_mode, align_corners);
    });

    return output;
}

std::tuple<Tensor, Tensor> grid_sampler_2d_backward(const Tensor& grad_output,
                                                    const Tensor& input,
                                                    const Tensor& grid,
                                                    int64_t interpolation_mode,
                                                    int64_t padding_mode,
                                                    bool align_corners,
                                                    std::array<bool, 2> output_mask)
{
    check_grid_sampler(input, grid, interpolation_mode, padding_mode);
    TORCH_CHECK(grad_output.device() == input.device() && grad_output.scalar_type() == input.scalar_type(),
                "grid_sampler(): expected grad_output to have the same device and dtype as input");
    TORCH_CHECK(grad_output.sizes() == IntArrayRef({ input.size(0), input.size(1), grid.size(1), grid.size(2) }),
                "grid_sampler(): unexpected grad_output shape ", grad_output.sizes());

    if (input.scalar_type() == kHalf || input.scalar_type() == kBFloat16)
    {
        // The CUDA backward kernel computes coordinates and accumulates in the
        // input dtype. Use float32 to preserve sampling boundaries and gradients.
        auto [grad_input, grad_grid] =
            grid_sampler_2d_backward(grad_output.to(kFloat), input.to(kFloat), grid.to(kFloat), interpolation_mode,
                                     padding_mode, align_corners, output_mask);
        return { grad_input.defined() ? grad_input.to(input.scalar_type()) : Tensor(), grad_grid.to(grid.scalar_type()) };
    }

    auto grad_input = output_mask[0] ? at::zeros(input.sizes(), input.options()) : Tensor();
    // PyTorch's CUDA kernel always writes grad_grid, regardless of output_mask[1].
    // Zero initialization also handles empty outputs and zero input channels.
    auto grad_grid = at::zeros(grid.sizes(), grid.options());
    if (grad_output.numel() == 0)
    {
        return { grad_input, grad_grid };
    }

    // Each GPU owns complete batches: input-gradient atomics must never target
    // the same managed-memory element from different GPUs.
    parallel_for_each([&](int device, cudaStream_t stream) {
        size_t start, length;
        std::tie(start, length) = partition::range(input.size(0), device);
        if (length == 0)
        {
            return;
        }

        const c10::Device cuda_device(c10::DeviceType::CUDA, device);
        auto cuda_input = make_view(input, 0, start, length, cuda_device);
        auto cuda_grid = make_view(grid, 0, start, length, cuda_device);
        auto cuda_grad_output = make_view(grad_output, 0, start, length, cuda_device);
        auto cuda_grad_input = make_view(grad_input, 0, start, length, cuda_device);
        auto cuda_grad_grid = make_view(grad_grid, 0, start, length, cuda_device);
        launch_grid_sampler_2d_backward_kernel(cuda_grad_input, cuda_grad_grid, cuda_grad_output, cuda_input, cuda_grid,
                                               interpolation_mode, padding_mode, align_corners, output_mask);
    });

    return { grad_input, grad_grid };
}

TORCH_LIBRARY_IMPL(aten, PrivateUse1, m)
{
    m.impl(TORCH_SELECTIVE_NAME("aten::grid_sampler_2d"), TORCH_FN(grid_sampler_2d));
    m.impl(TORCH_SELECTIVE_NAME("aten::grid_sampler_2d_backward"), TORCH_FN(grid_sampler_2d_backward));
}

} // namespace dgx

} // namespace at::native
