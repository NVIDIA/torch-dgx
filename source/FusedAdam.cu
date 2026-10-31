/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
#include "KernelLaunchUtils.cuh"

#include <ATen/Dispatch_v2.h>
#include <ATen/OpMathType.h>

#include <torch/library.h>

namespace at::native
{

namespace dgx
{

namespace
{

template <typename T>
__host__ __device__ inline T lerp(T v0, T v1, T t)
{
    return fma(t, v1, fma(-t, v0, v0));
}

template <class T>
struct FusedAdamFunctor
{
    __device__ void operator()(size_t tid,
                               T* __restrict__ param,
                               const T* __restrict__ grad,
                               T* __restrict__ exp_avg,
                               T* __restrict__ exp_avg_sq,
                               T beta1,
                               T beta2,
                               T step_size,
                               T bias_correction2_sqrt,
                               T eps)
    {
        exp_avg[tid] = lerp(grad[tid], exp_avg[tid], beta1);
        exp_avg_sq[tid] = lerp(grad[tid] * grad[tid], exp_avg_sq[tid], beta2);

        T denom = (::cuda::std::sqrt(exp_avg_sq[tid]) / bias_correction2_sqrt) + eps;
        param[tid] -= step_size * exp_avg[tid] / denom;
    }
};

template <class T>
struct FusedAdamWFunctor
{
    __device__ void operator()(size_t tid,
                               T* __restrict__ param,
                               const T* __restrict__ grad,
                               T* __restrict__ exp_avg,
                               T* __restrict__ exp_avg_sq,
                               T beta1,
                               T beta2,
                               T step_size,
                               T bias_correction2_sqrt,
                               T eps,
                               T lr,
                               T weight_decay)
    {
        if (weight_decay != 0)
        {
            param[tid] -= lr * weight_decay * param[tid];
        }

        exp_avg[tid] = lerp(grad[tid], exp_avg[tid], beta1);
        exp_avg_sq[tid] = lerp(grad[tid] * grad[tid], exp_avg_sq[tid], beta2);

        T denom = (::cuda::std::sqrt(exp_avg_sq[tid]) / bias_correction2_sqrt) + eps;
        param[tid] -= step_size * exp_avg[tid] / denom;
    }
};

template <class T>
__global__ void fused_adam_step_kernel(const float* __restrict__ state_step,
                                       double lr,
                                       double beta1,
                                       double beta2,
                                       T* __restrict__ step_size,
                                       T* __restrict__ bias_correction2_sqrt)
{
    const double step = static_cast<double>(*state_step);
    const double bias_correction1 = 1.0 - pow(beta1, step);
    const double bias_correction2 = 1.0 - pow(beta2, step);
    *step_size = static_cast<T>(lr / bias_correction1);
    *bias_correction2_sqrt = static_cast<T>(::cuda::std::sqrt(bias_correction2));
}

template <typename T>
bool is_aligned(T* p)
{
    return ((uint64_t)p) % (4 * sizeof(T)) == 0;
}

template <class T>
__global__ void __launch_bounds__(256) fused_adam_kernel(int64_t numel,
                                                         T* __restrict__ param,
                                                         const T* __restrict__ grad,
                                                         T* __restrict__ exp_avg,
                                                         T* __restrict__ exp_avg_sq,
                                                         const T* __restrict__ step_size_ptr,
                                                         const T* __restrict__ bias_correction2_sqrt_ptr,
                                                         T beta1,
                                                         T beta2,
                                                         T eps)
{
    const T step_size = *step_size_ptr;
    const T bias_correction2_sqrt = *bias_correction2_sqrt_ptr;

    auto param_vec = static_cast<T*>(__builtin_assume_aligned(param, sizeof(T) * 4));
    const auto grad_vec = static_cast<const T*>(__builtin_assume_aligned(grad, sizeof(T) * 4));
    auto exp_avg_vec = static_cast<T*>(__builtin_assume_aligned(exp_avg, sizeof(T) * 4));
    auto exp_avg_sq_vec = static_cast<T*>(__builtin_assume_aligned(exp_avg_sq, sizeof(T) * 4));

    FusedAdamFunctor<T> op;

    unsigned int tid = threadIdx.x + blockIdx.x * blockDim.x;
    for (unsigned int i = tid; i < numel / 4; i += blockDim.x * gridDim.x)
    {
#pragma unroll
        for (unsigned int j = 0; j < 4; ++j)
        {
            op(i * 4 + j, param_vec, grad_vec, exp_avg_vec, exp_avg_sq_vec, beta1, beta2, step_size,
               bias_correction2_sqrt, eps);
        }
    }

    unsigned int remainder = numel % 4;
    if (tid == 0 && remainder != 0)
    {
        while (remainder)
        {
            unsigned int tid = numel - remainder--;
            op(tid, param, grad, exp_avg, exp_avg_sq, beta1, beta2, step_size, bias_correction2_sqrt, eps);
        }
    }
}

template <class T>
__global__ void __launch_bounds__(256) fused_adamw_kernel(int64_t numel,
                                                         T* __restrict__ param,
                                                         const T* __restrict__ grad,
                                                         T* __restrict__ exp_avg,
                                                         T* __restrict__ exp_avg_sq,
                                                         const T* __restrict__ step_size_ptr,
                                                         const T* __restrict__ bias_correction2_sqrt_ptr,
                                                         T beta1,
                                                         T beta2,
                                                         T eps,
                                                         T lr,
                                                         T weight_decay)
{
    const T step_size = *step_size_ptr;
    const T bias_correction2_sqrt = *bias_correction2_sqrt_ptr;

    auto param_vec = static_cast<T*>(__builtin_assume_aligned(param, sizeof(T) * 4));
    const auto grad_vec = static_cast<const T*>(__builtin_assume_aligned(grad, sizeof(T) * 4));
    auto exp_avg_vec = static_cast<T*>(__builtin_assume_aligned(exp_avg, sizeof(T) * 4));
    auto exp_avg_sq_vec = static_cast<T*>(__builtin_assume_aligned(exp_avg_sq, sizeof(T) * 4));

    FusedAdamWFunctor<T> op;

    unsigned int tid = threadIdx.x + blockIdx.x * blockDim.x;
    for (unsigned int i = tid; i < numel / 4; i += blockDim.x * gridDim.x)
    {
#pragma unroll
        for (unsigned int j = 0; j < 4; ++j)
        {
            op(i * 4 + j, param_vec, grad_vec, exp_avg_vec, exp_avg_sq_vec, beta1, beta2, step_size,
               bias_correction2_sqrt, eps, lr, weight_decay);
        }
    }

    unsigned int remainder = numel % 4;
    if (tid == 0 && remainder != 0)
    {
        while (remainder)
        {
            unsigned int tid = numel - remainder--;
            op(tid, param, grad, exp_avg, exp_avg_sq, beta1, beta2, step_size, bias_correction2_sqrt, eps, lr,
               weight_decay);
        }
    }
}

} // namespace

void _fused_adam_impl_(TensorList params,
                  TensorList grads,
                  TensorList exp_avgs,
                  TensorList exp_avg_sqs,
                  TensorList max_exp_avg_sqs,
                  TensorList state_steps,
                  double lr,
                  double beta1,
                  double beta2,
                  double weight_decay,
                  double eps,
                  bool amsgrad,
                  bool maximize,
                  const ::std::optional<Tensor>& grad_scale,
                  const ::std::optional<Tensor>& found_inf)
{
    TORCH_CHECK(!amsgrad);
    TORCH_CHECK(max_exp_avg_sqs.empty());
    TORCH_CHECK(!maximize);
    TORCH_CHECK(!grad_scale);
    TORCH_CHECK(!found_inf);
    TORCH_CHECK(!weight_decay);

    TORCH_CHECK(params.size() == grads.size());
    TORCH_CHECK(params.size() == exp_avgs.size());
    TORCH_CHECK(params.size() == exp_avg_sqs.size());
    TORCH_CHECK(params.size() == state_steps.size());

    for (size_t i = 0; i < params.size(); ++i)
    {
        TORCH_CHECK(params[i].is_contiguous());
        TORCH_CHECK(grads[i].is_contiguous());
        TORCH_CHECK(exp_avgs[i].is_contiguous());
        TORCH_CHECK(exp_avg_sqs[i].is_contiguous());
        TORCH_CHECK(params[i].sizes() == grads[i].sizes());
        TORCH_CHECK(params[i].sizes() == exp_avgs[i].sizes());
        TORCH_CHECK(params[i].sizes() == exp_avg_sqs[i].sizes());
        TORCH_CHECK(state_steps[i].scalar_type() == kFloat);
        TORCH_CHECK(state_steps[i].numel() == 1);
    }

    // std::vector<cudaEvent_t> events(c10::cuda::device_count());
    // for (const auto device_id : c10::irange(c10::cuda::device_count()))
    // {
    //     C10_CUDA_CHECK(cudaSetDevice(device_id));
    //     auto stream = c10::cuda::getCurrentCUDAStream(device_id);
    //     C10_CUDA_CHECK(cudaEventCreateWithFlags(&events[device_id], cudaEventDisableTiming));
    //     C10_CUDA_CHECK(cudaEventRecord(events[device_id], stream));
    // }

    // for (const auto device_id : c10::irange(c10::cuda::device_count()))
    // {
    //     C10_CUDA_CHECK(cudaSetDevice(device_id));
    //     auto stream = c10::cuda::getStreamFromPool(false, device_id);
    //     C10_CUDA_CHECK(cudaStreamWaitEvent(stream, events[device_id]));

    //     std::vector<void*> prefetchPointers;
    //     std::vector<size_t> prefetchSizes;
    //     const cudaMemLocation location = { cudaMemLocationTypeDevice, device_id };
    //     std::vector<cudaMemLocation> prefetchLocations = { location };
    //     std::vector<size_t> prefetchLocationIndices = { 0 };

    //     for (size_t i = 0; i < params.size(); ++i)
    //     {
    //         AT_DISPATCH_V2(params[i].scalar_type(), "_fused_adam_", AT_WRAP([&] {
    //                            using opmath_t = at::opmath_type<scalar_t>;
    //                            constexpr size_t kAlignment = kPageSize / sizeof(opmath_t);
    //                            TORCH_CHECK(kAlignment % 4 == 0);

    //                            int64_t device_offset, device_numel;
    //                            std::tie(device_offset, device_numel) =
    //                                aligned_chunk(kAlignment, params[i].numel(), device_id);

    //                            const unsigned int device_num_blocks =
    //                                ::cuda::ceil_div((device_numel + 3) / 4, kNumThreads);
    //                            if (device_num_blocks)
    //                            {
    //                                prefetchPointers.emplace_back(params[i].data_ptr<opmath_t>() + device_offset);
    //                                prefetchSizes.emplace_back(sizeof(opmath_t) * device_numel);

    //                                prefetchPointers.emplace_back(grads[i].data_ptr<opmath_t>() + device_offset);
    //                                prefetchSizes.emplace_back(sizeof(opmath_t) * device_numel);

    //                                prefetchPointers.emplace_back(exp_avgs[i].data_ptr<opmath_t>() + device_offset);
    //                                prefetchSizes.emplace_back(sizeof(opmath_t) * device_numel);

    //                                prefetchPointers.emplace_back(exp_avg_sqs[i].data_ptr<opmath_t>() +
    //                                device_offset); prefetchSizes.emplace_back(sizeof(opmath_t) * device_numel);
    //                            }
    //                        }),
    //                        AT_EXPAND(AT_FLOATING_TYPES));
    //     }

    //     if (prefetchPointers.size() > 0)
    //     {
    //         C10_CUDA_CHECK(cudaMemPrefetchBatchAsync(
    //             prefetchPointers.data(), prefetchSizes.data(), prefetchPointers.size(), prefetchLocations.data(),
    //             prefetchLocationIndices.data(), prefetchLocations.size(), 0, stream));
    //     }
    //     C10_CUDA_CHECK(cudaEventRecord(events[device_id], stream));
    // }

    // for (const auto device_id : c10::irange(c10::cuda::device_count()))
    // {
    //     C10_CUDA_CHECK(cudaSetDevice(device_id));
    //     auto stream = c10::cuda::getCurrentCUDAStream(device_id);
    //     C10_CUDA_CHECK(cudaStreamWaitEvent(stream, events[device_id]));
    //     C10_CUDA_CHECK(cudaEventDestroy(events[device_id]));
    // }

    if (params.empty())
    {
        return;
    }

    // Submit the whole tensor list on each device before switching devices.
    // Step scratch is reused in stream order and is unnecessary for empty shards.
    parallel_for_each_async([&](int device, cudaStream_t stream) {
        void* device_step_storage = nullptr;
        for (size_t i = 0; i < params.size(); ++i)
        {
            AT_DISPATCH_V2(
                params[i].scalar_type(), "_fused_adam_", AT_WRAP([&] {
                    using opmath_t = at::opmath_type<scalar_t>;
                    // Align only for vectorized accesses, keeping shards near the logical partition.
                    constexpr size_t kAlignment = 4;

                    int64_t device_offset, device_numel;
                    std::tie(device_offset, device_numel) = aligned_chunk(kAlignment, params[i].numel(), device);
                    if (!device_numel)
                        return;

                    if (!device_step_storage)
                        C10_CUDA_CHECK(cudaMallocAsync(&device_step_storage, 2 * sizeof(double), stream));
                    auto step_size = static_cast<opmath_t*>(device_step_storage);
                    auto bias_correction2_sqrt = step_size + 1;
                    fused_adam_step_kernel<<<1, 1, 0, stream>>>(
                        state_steps[i].const_data_ptr<float>(), lr, beta1, beta2, step_size, bias_correction2_sqrt);
                    C10_CUDA_KERNEL_LAUNCH_CHECK();

                    const unsigned int device_num_blocks = ::cuda::ceil_div((device_numel + 3) / 4, kNumThreads);
                    if (device_num_blocks)
                    {
                        TORCH_CHECK(is_aligned(params[i].data_ptr<opmath_t>() + device_offset));
                        TORCH_CHECK(is_aligned(grads[i].const_data_ptr<opmath_t>() + device_offset));
                        TORCH_CHECK(is_aligned(exp_avgs[i].data_ptr<opmath_t>() + device_offset));
                        TORCH_CHECK(is_aligned(exp_avg_sqs[i].data_ptr<opmath_t>() + device_offset));

                        fused_adam_kernel<<<device_num_blocks, kNumThreads, 0, stream>>>(
                            device_numel, params[i].data_ptr<opmath_t>() + device_offset,
                            grads[i].const_data_ptr<opmath_t>() + device_offset,
                            exp_avgs[i].data_ptr<opmath_t>() + device_offset,
                            exp_avg_sqs[i].data_ptr<opmath_t>() + device_offset, step_size, bias_correction2_sqrt,
                            static_cast<opmath_t>(beta1), static_cast<opmath_t>(beta2), static_cast<opmath_t>(eps));
                        C10_CUDA_KERNEL_LAUNCH_CHECK();
                    }
                }),
                AT_EXPAND(AT_FLOATING_TYPES));
        }
        if (device_step_storage)
            C10_CUDA_CHECK(cudaFreeAsync(device_step_storage, stream));
    });

    merge_streams();
}

void _fused_adamw_impl_(TensorList params,
                   TensorList grads,
                   TensorList exp_avgs,
                   TensorList exp_avg_sqs,
                   TensorList max_exp_avg_sqs,
                   TensorList state_steps,
                   double lr,
                   double beta1,
                   double beta2,
                   double weight_decay,
                   double eps,
                   bool amsgrad,
                   bool maximize,
                   const ::std::optional<Tensor>& grad_scale,
                   const ::std::optional<Tensor>& found_inf)
{
    TORCH_CHECK(!amsgrad);
    TORCH_CHECK(max_exp_avg_sqs.empty());
    TORCH_CHECK(!maximize);
    TORCH_CHECK(!grad_scale);
    TORCH_CHECK(!found_inf);

    TORCH_CHECK(params.size() == grads.size());
    TORCH_CHECK(params.size() == exp_avgs.size());
    TORCH_CHECK(params.size() == exp_avg_sqs.size());
    TORCH_CHECK(params.size() == state_steps.size());

    for (size_t i = 0; i < params.size(); ++i)
    {
        TORCH_CHECK(params[i].is_contiguous());
        TORCH_CHECK(grads[i].is_contiguous());
        TORCH_CHECK(exp_avgs[i].is_contiguous());
        TORCH_CHECK(exp_avg_sqs[i].is_contiguous());
        TORCH_CHECK(params[i].sizes() == grads[i].sizes());
        TORCH_CHECK(params[i].sizes() == exp_avgs[i].sizes());
        TORCH_CHECK(params[i].sizes() == exp_avg_sqs[i].sizes());
        TORCH_CHECK(state_steps[i].scalar_type() == kFloat);
        TORCH_CHECK(state_steps[i].numel() == 1);
    }

    if (params.empty())
    {
        return;
    }

    // Submit the whole tensor list on each device before switching devices.
    // Step scratch is reused in stream order and is unnecessary for empty shards.
    parallel_for_each_async([&](int device, cudaStream_t stream) {
        void* device_step_storage = nullptr;
        for (size_t i = 0; i < params.size(); ++i)
        {
            AT_DISPATCH_V2(
                params[i].scalar_type(), "_fused_adamw_", AT_WRAP([&] {
                    using opmath_t = at::opmath_type<scalar_t>;
                    // Align only for vectorized accesses, keeping shards near the logical partition.
                    constexpr size_t kAlignment = 4;

                    int64_t device_offset, device_numel;
                    std::tie(device_offset, device_numel) = aligned_chunk(kAlignment, params[i].numel(), device);
                    if (!device_numel)
                        return;

                    if (!device_step_storage)
                        C10_CUDA_CHECK(cudaMallocAsync(&device_step_storage, 2 * sizeof(double), stream));
                    auto step_size = static_cast<opmath_t*>(device_step_storage);
                    auto bias_correction2_sqrt = step_size + 1;
                    fused_adam_step_kernel<<<1, 1, 0, stream>>>(
                        state_steps[i].const_data_ptr<float>(), lr, beta1, beta2, step_size, bias_correction2_sqrt);
                    C10_CUDA_KERNEL_LAUNCH_CHECK();

                    const unsigned int device_num_blocks = ::cuda::ceil_div((device_numel + 3) / 4, kNumThreads);
                    if (device_num_blocks)
                    {
                        TORCH_CHECK(is_aligned(params[i].data_ptr<opmath_t>() + device_offset));
                        TORCH_CHECK(is_aligned(grads[i].const_data_ptr<opmath_t>() + device_offset));
                        TORCH_CHECK(is_aligned(exp_avgs[i].data_ptr<opmath_t>() + device_offset));
                        TORCH_CHECK(is_aligned(exp_avg_sqs[i].data_ptr<opmath_t>() + device_offset));

                        fused_adamw_kernel<<<device_num_blocks, kNumThreads, 0, stream>>>(
                            device_numel, params[i].data_ptr<opmath_t>() + device_offset,
                            grads[i].const_data_ptr<opmath_t>() + device_offset,
                            exp_avgs[i].data_ptr<opmath_t>() + device_offset,
                            exp_avg_sqs[i].data_ptr<opmath_t>() + device_offset, step_size, bias_correction2_sqrt,
                            static_cast<opmath_t>(beta1), static_cast<opmath_t>(beta2), static_cast<opmath_t>(eps),
                            static_cast<opmath_t>(lr), static_cast<opmath_t>(weight_decay));
                        C10_CUDA_KERNEL_LAUNCH_CHECK();
                    }
                }),
                AT_EXPAND(AT_FLOATING_TYPES));
        }
        if (device_step_storage)
            C10_CUDA_CHECK(cudaFreeAsync(device_step_storage, stream));
    });

    merge_streams();
}

TORCH_LIBRARY_IMPL(aten, PrivateUse1, m)
{
    m.impl(TORCH_SELECTIVE_NAME("aten::_fused_adam_"), TORCH_FN(_fused_adam_impl_));
    m.impl(TORCH_SELECTIVE_NAME("aten::_fused_adamw_"), TORCH_FN(_fused_adamw_impl_));
}

} // namespace dgx

} // namespace at::native
