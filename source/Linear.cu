/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
#include "CompatibilityUtils.h"
#include "KernelLaunchUtils.cuh"
#include "NCCLUtils.h"

#include <c10/cuda/CUDAStream.h>
#include <c10/util/ScopeExit.h>
#include <c10/util/env.h>

#include <ATen/Dispatch_v2.h>
#include <ATen/MemoryOverlap.h>
#include <ATen/OpMathType.h>
#include <ATen/cuda/CUDABlas.h>
#include <ATen/native/LinearAlgebraUtils.h>
#include <ATen/native/Resize.h>
#include <ATen/ops/empty.h>
#include <ATen/ops/linear_native.h>
#include <ATen/ops/sum.h>

#include <torch/library.h>

#include <cublas_v2.h>

namespace at::native
{

namespace dgx
{

static c10::once_flag gBLASHandlesInitFlag;
std::vector<cublasHandle_t> gBLASHandles;

static void initBLASHandles()
{
    gBLASHandles.resize(c10::cuda::device_count());
    for (const auto device_id : c10::irange(c10::cuda::device_count()))
    {
        C10_CUDA_CHECK(cudaSetDevice(device_id));
        TORCH_CUDABLAS_CHECK(cublasCreate(&gBLASHandles[device_id]));
    }
}

namespace
{

cudaEvent_t record_gemm_output_ready(cudaStream_t stream)
{
    cudaEvent_t ready_event = nullptr;
    C10_CUDA_CHECK(cudaEventCreateWithFlags(&ready_event, cudaEventDisableTiming));
    // Record after input staging, ordering the managed output allocation too.
    C10_CUDA_CHECK(cudaEventRecord(ready_event, stream));
    return ready_event;
}

cudaEvent_t prefetch_gemm_output(void* output, size_t bytes, int device, cudaEvent_t ready_event)
{
    if (!bytes)
    {
        if (ready_event)
            C10_CUDA_CHECK(cudaEventDestroy(ready_event));
        return nullptr;
    }
    const auto prefetch_stream = c10::cuda::getStreamFromPool(false, device);
    cudaEvent_t prefetch_event = nullptr;
    C10_CUDA_CHECK(cudaEventCreateWithFlags(&prefetch_event, cudaEventDisableTiming));
    C10_CUDA_CHECK(cudaStreamWaitEvent(prefetch_stream, ready_event));
    C10_CUDA_CHECK(cudaEventDestroy(ready_event));
    // An idle prefetch stream can block the host. Call after submitting useful
    // GEMM work, then allow prefetching to overlap that local computation.
    C10_CUDA_CHECK(memPrefetchAsync(output, bytes, { cudaMemLocationTypeDevice, device }, 0, prefetch_stream));
    C10_CUDA_CHECK(cudaEventRecord(prefetch_event, prefetch_stream));
    return prefetch_event;
}

void wait_gemm_output(cudaEvent_t prefetch_event, cudaStream_t stream)
{
    if (!prefetch_event)
        return;
    // Destroy after submitting the wait; CUDA releases pending events asynchronously.
    C10_CUDA_CHECK(cudaStreamWaitEvent(stream, prefetch_event));
    C10_CUDA_CHECK(cudaEventDestroy(prefetch_event));
}

bool prefer_token_sharded_weight_gradient(int64_t outputs, int64_t inputs, int64_t tokens)
{
    const double reduction_volume = static_cast<double>(outputs) * inputs;
    const double feature_volume =
        static_cast<double>(tokens) *
        (std::min(outputs, inputs) + static_cast<double>(std::max(outputs, inputs)) / c10::cuda::device_count());
    return reduction_volume < feature_volume;
}

// Compare the complete fused backward when dW is requested. Assume balanced
// token ownership for X/dY and output-feature ownership for W/dW. Volumes are
// in elements with the common (device_count - 1) traffic factor omitted, as in
// the standalone dW policy above. This is a shape heuristic, not a residency
// query; staging/writeback locality and NCCL protocol overhead can differ.
bool prefer_token_sharded_linear_backward(int64_t outputs, int64_t inputs, int64_t tokens, bool need_input, bool need_bias)
{
    const auto device_count = c10::cuda::device_count();
    const double weight_volume = static_cast<double>(outputs) * inputs;
    const double input_volume = static_cast<double>(tokens) * inputs;

    // Token shards read local X/dY and reduce dW. Feature shards always
    // replicate X (even when it is larger than dY) and redistribute dY.
    double token_volume = nccl_reduce_scatter_padded_numel(outputs * inputs);
    double feature_volume = input_volume + static_cast<double>(tokens) * outputs / device_count;
    if (need_input)
    {
        // Token shards additionally replicate W; feature shards instead
        // reduce dX into complete token rows.
        token_volume += weight_volume;
        feature_volume += nccl_reduce_scatter_padded_numel(tokens * inputs, inputs);
    }
    if (need_bias)
        token_volume += nccl_reduce_scatter_padded_numel(outputs);

    return token_volume <= feature_volume;
}

template <typename scalar_t>
struct AddmmCopySelfFunctor
{
    __device__ void operator()(
        size_t i, scalar_t* result, const scalar_t* self, int64_t columns, int64_t row_stride, int64_t column_stride)
    {
        result[i] = self[(i / columns) * row_stride + (i % columns) * column_stride];
    }
};

// Copy a rectangular block without materializing the rest of a shared operand.
// Both layouts accepted by is_gemm_layout retain their orientation in scratch.
template <typename scalar_t>
void copy_gemm_block(const Tensor& source,
                     int64_t row,
                     int64_t rows,
                     int64_t column,
                     int64_t columns,
                     scalar_t* destination,
                     cudaStream_t stream)
{
    const bool row_major = source.is_contiguous();
    const auto width = row_major ? columns : rows;
    const auto height = row_major ? rows : columns;
    const auto pitch = row_major ? source.stride(0) : source.stride(1);
    const auto* input = source.const_data_ptr<scalar_t>() + row * source.stride(0) + column * source.stride(1);
    // A singleton dimension may have any stride, including a pitch smaller
    // than width. It is still a contiguous span, not a valid 2-D copy pitch.
    if (height == 1 || pitch == width)
    {
        C10_CUDA_CHECK(cudaMemcpyAsync(destination, input, width * height * sizeof(scalar_t), cudaMemcpyDefault, stream));
        return;
    }
    C10_CUDA_CHECK(cudaMemcpy2DAsync(destination, width * sizeof(scalar_t), input, pitch * sizeof(scalar_t),
                                     width * sizeof(scalar_t), height, cudaMemcpyDefault, stream));
}

// Keep repeatedly accessed GEMM operands and its output in device-local scratch.
// Managed tensors are read/written once per shard, instead of once per GEMM tile.
template <typename scalar_t>
class GemmRowScratch
{
public:
    GemmRowScratch(const Tensor& input, Tensor& output, int64_t row, int64_t rows, cudaStream_t stream)
        : stream(stream),
          row_count(rows),
          column_count(output.size(1)),
          input_leading_dimension(input.is_contiguous() ? input.size(1) : rows),
          managed_output(output.data_ptr<scalar_t>() + row * output.stride(0))
    {
        constexpr int64_t alignment = 256 / sizeof(scalar_t);
        const auto output_offset = ::cuda::round_up(rows * input.size(1), alignment);
        C10_CUDA_CHECK(cudaMallocAsync(&scratch, (output_offset + rows * column_count) * sizeof(scalar_t), stream));
        local_output = scratch + output_offset;
    }
    ~GemmRowScratch()
    {
        C10_CUDA_CHECK_WARN(cudaFreeAsync(scratch, stream));
    }
    GemmRowScratch(const GemmRowScratch&) = delete;
    GemmRowScratch& operator=(const GemmRowScratch&) = delete;

    void copy_input(const Tensor& input, int64_t row)
    {
        copy_gemm_block(input, row, row_count, 0, input.size(1), scratch, stream);
    }
    void copy_output(int device, cudaEvent_t ready_event)
    {
        const auto prefetch_event =
            prefetch_gemm_output(managed_output, row_count * column_count * sizeof(scalar_t), device, ready_event);
        wait_gemm_output(prefetch_event, stream);
        C10_CUDA_CHECK(cudaMemcpyAsync(
            managed_output, local_output, row_count * column_count * sizeof(scalar_t), cudaMemcpyDefault, stream));
    }
    const scalar_t* input() const
    {
        return scratch;
    }
    scalar_t* output() const
    {
        return local_output;
    }
    int64_t leading_dimension() const
    {
        return input_leading_dimension;
    }

private:
    cudaStream_t stream;
    int64_t row_count, column_count, input_leading_dimension;
    scalar_t* managed_output;
    scalar_t* scratch = nullptr;
    scalar_t* local_output;
};

uint32_t blasLtAlignment(const void* pointer)
{
    const auto address = reinterpret_cast<uintptr_t>(pointer);
    uint32_t alignment = 1;
    while (alignment < 256 && address % (2 * alignment) == 0)
        alignment *= 2;
    return alignment;
}

template <typename scalar_t>
bool addmmBiasLt(cublasOperation_t transa,
                 cublasOperation_t transb,
                 int64_t m,
                 int64_t n,
                 int64_t k,
                 scalar_t alpha,
                 const scalar_t* a,
                 int64_t lda,
                 const scalar_t* b,
                 int64_t ldb,
                 const scalar_t* bias,
                 scalar_t* result,
                 cudaStream_t stream)
{
    constexpr bool is_double = std::is_same_v<scalar_t, double>;
    const auto data_type = is_double ? CUDA_R_64F : CUDA_R_32F;
    auto compute_type = is_double ? CUBLAS_COMPUTE_64F : CUBLAS_COMPUTE_32F;
    if (!is_double && !at::NoTF32Guard::should_disable_tf32() &&
        at::globalContext().float32Precision(at::Float32Backend::CUDA, at::Float32Op::MATMUL) ==
            at::Float32Precision::TF32)
        compute_type = CUBLAS_COMPUTE_32F_FAST_TF32;

    cublasLtMatmulDesc_t operation = nullptr;
    cublasLtMatrixLayout_t a_layout = nullptr, b_layout = nullptr, result_layout = nullptr;
    cublasLtMatmulPreference_t preference = nullptr;
    auto release_descriptors = c10::make_scope_exit([&] {
        if (preference)
            cublasLtMatmulPreferenceDestroy(preference);
        if (result_layout)
            cublasLtMatrixLayoutDestroy(result_layout);
        if (b_layout)
            cublasLtMatrixLayoutDestroy(b_layout);
        if (a_layout)
            cublasLtMatrixLayoutDestroy(a_layout);
        if (operation)
            cublasLtMatmulDescDestroy(operation);
    });
    TORCH_CUDABLAS_CHECK(cublasLtMatmulDescCreate(&operation, compute_type, data_type));
    const auto epilogue = CUBLASLT_EPILOGUE_BIAS;
    TORCH_CUDABLAS_CHECK(cublasLtMatmulDescSetAttribute(operation, CUBLASLT_MATMUL_DESC_TRANSA, &transa, sizeof(transa)));
    TORCH_CUDABLAS_CHECK(cublasLtMatmulDescSetAttribute(operation, CUBLASLT_MATMUL_DESC_TRANSB, &transb, sizeof(transb)));
    TORCH_CUDABLAS_CHECK(
        cublasLtMatmulDescSetAttribute(operation, CUBLASLT_MATMUL_DESC_EPILOGUE, &epilogue, sizeof(epilogue)));
    TORCH_CUDABLAS_CHECK(
        cublasLtMatmulDescSetAttribute(operation, CUBLASLT_MATMUL_DESC_BIAS_POINTER, &bias, sizeof(bias)));
    TORCH_CUDABLAS_CHECK(cublasLtMatrixLayoutCreate(
        &a_layout, data_type, transa == CUBLAS_OP_N ? m : k, transa == CUBLAS_OP_N ? k : m, lda));
    TORCH_CUDABLAS_CHECK(cublasLtMatrixLayoutCreate(
        &b_layout, data_type, transb == CUBLAS_OP_N ? k : n, transb == CUBLAS_OP_N ? n : k, ldb));
    TORCH_CUDABLAS_CHECK(cublasLtMatrixLayoutCreate(&result_layout, data_type, m, n, m));
    TORCH_CUDABLAS_CHECK(cublasLtMatmulPreferenceCreate(&preference));

    // PyTorch owns a device-local workspace for each handle/stream pair. In particular, scratch
    // must not be allocated from the DGX managed tensor pool or shared between active GPUs.
    const auto handle = at::cuda::getCurrentCUDABlasLtHandle();
    const auto workspace_size = at::cuda::getCUDABlasLtWorkspaceSize();
    auto* workspace = at::cuda::getCUDABlasLtWorkspace();
    TORCH_CUDABLAS_CHECK(cublasLtMatmulPreferenceSetAttribute(
        preference, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &workspace_size, sizeof(workspace_size)));
    // C and D share the result buffer; Lt infers bias alignment from BIAS_POINTER.
    const auto result_alignment = blasLtAlignment(result);
    const uint32_t alignments[] = { blasLtAlignment(a), blasLtAlignment(b), result_alignment, result_alignment };
    const cublasLtMatmulPreferenceAttributes_t attributes[] = { CUBLASLT_MATMUL_PREF_MIN_ALIGNMENT_A_BYTES,
                                                                CUBLASLT_MATMUL_PREF_MIN_ALIGNMENT_B_BYTES,
                                                                CUBLASLT_MATMUL_PREF_MIN_ALIGNMENT_C_BYTES,
                                                                CUBLASLT_MATMUL_PREF_MIN_ALIGNMENT_D_BYTES };
    for (int i = 0; i < 4; ++i)
        TORCH_CUDABLAS_CHECK(
            cublasLtMatmulPreferenceSetAttribute(preference, attributes[i], &alignments[i], sizeof(alignments[i])));

    cublasLtMatmulHeuristicResult_t algorithm{};
    int count = 0;
    auto status = cublasLtMatmulAlgoGetHeuristic(
        handle, operation, a_layout, b_layout, result_layout, result_layout, preference, 1, &algorithm, &count);
    if (status == CUBLAS_STATUS_NOT_SUPPORTED)
        return false;
    TORCH_CUDABLAS_CHECK(status);
    if (!count || algorithm.state == CUBLAS_STATUS_NOT_SUPPORTED)
        return false;
    TORCH_CUDABLAS_CHECK(algorithm.state);

    const scalar_t zero = 0;
    status = cublasLtMatmul(handle, operation, &alpha, a, a_layout, b, b_layout, &zero, result, result_layout, result,
                            result_layout, &algorithm.algo, workspace, workspace_size, stream);
    if (status == CUBLAS_STATUS_NOT_SUPPORTED)
        return false;
    TORCH_CUDABLAS_CHECK(status);
    return true;
}

template <typename scalar_t>
cublasStatus_t cublasGemm(cublasHandle_t handle,
                          cublasOperation_t transa,
                          cublasOperation_t transb,
                          int m,
                          int n,
                          int k,
                          const scalar_t* alpha,
                          const scalar_t* A,
                          int lda,
                          const scalar_t* B,
                          int ldb,
                          const scalar_t* beta,
                          scalar_t* C,
                          int ldc)
{
    TORCH_CHECK(false, "Unsupported datatype for GEMM");
}

template <>
cublasStatus_t cublasGemm(cublasHandle_t handle,
                          cublasOperation_t transa,
                          cublasOperation_t transb,
                          int m,
                          int n,
                          int k,
                          const float* alpha,
                          const float* A,
                          int lda,
                          const float* B,
                          int ldb,
                          const float* beta,
                          float* C,
                          int ldc)
{
    return cublasSgemm(handle, transa, transb, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
}

template <>
cublasStatus_t cublasGemm(cublasHandle_t handle,
                          cublasOperation_t transa,
                          cublasOperation_t transb,
                          int m,
                          int n,
                          int k,
                          const double* alpha,
                          const double* A,
                          int lda,
                          const double* B,
                          int ldb,
                          const double* beta,
                          double* C,
                          int ldc)
{
    return cublasDgemm(handle, transa, transb, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
}

template <typename scalar_t>
cublasStatus_t cublasGemmStridedBatched(cublasHandle_t handle,
                                        cublasOperation_t transa,
                                        cublasOperation_t transb,
                                        int m,
                                        int n,
                                        int k,
                                        const scalar_t* alpha,
                                        const scalar_t* A,
                                        int lda,
                                        long long int strideA,
                                        const scalar_t* B,
                                        int ldb,
                                        long long int strideB,
                                        const scalar_t* beta,
                                        scalar_t* C,
                                        int ldc,
                                        long long int strideC,
                                        int batchCount)
{
    TORCH_CHECK(false, "Unsupported datatype for strided batched GEMM");
}

template <>
cublasStatus_t cublasGemmStridedBatched(cublasHandle_t handle,
                                        cublasOperation_t transa,
                                        cublasOperation_t transb,
                                        int m,
                                        int n,
                                        int k,
                                        const float* alpha,
                                        const float* A,
                                        int lda,
                                        long long int strideA,
                                        const float* B,
                                        int ldb,
                                        long long int strideB,
                                        const float* beta,
                                        float* C,
                                        int ldc,
                                        long long int strideC,
                                        int batchCount)
{
    return cublasSgemmStridedBatched(
        handle, transa, transb, m, n, k, alpha, A, lda, strideA, B, ldb, strideB, beta, C, ldc, strideC, batchCount);
}

template <>
cublasStatus_t cublasGemmStridedBatched(cublasHandle_t handle,
                                        cublasOperation_t transa,
                                        cublasOperation_t transb,
                                        int m,
                                        int n,
                                        int k,
                                        const double* alpha,
                                        const double* A,
                                        int lda,
                                        long long int strideA,
                                        const double* B,
                                        int ldb,
                                        long long int strideB,
                                        const double* beta,
                                        double* C,
                                        int ldc,
                                        long long int strideC,
                                        int batchCount)
{
    return cublasDgemmStridedBatched(
        handle, transa, transb, m, n, k, alpha, A, lda, strideA, B, ldb, strideB, beta, C, ldc, strideC, batchCount);
}

bool is_gemm_layout(const Tensor& matrix)
{
    return matrix.is_contiguous() || matrix.transpose(-2, -1).is_contiguous();
}

// Shard output features and reuse the same local dY for dX, dW, and db.
// dW/db have unique owners; only dX needs a reduction, published in token rows.
template <typename scalar_t>
void linear_backward_feature_shards(
    const Tensor& x, const Tensor& grad, const Tensor& w, Tensor& grad_input, Tensor& grad_weight, Tensor& grad_bias)
{
    const auto rows = x.size(0);
    const auto features = x.size(1);
    const auto outputs = grad.size(1);
    const auto input_numel = x.numel();
    const auto padded_input = grad_input.defined() ? nccl_reduce_scatter_padded_numel(input_numel, features) : 0;
    std::vector<scalar_t*> input_partials(c10::cuda::device_count(), nullptr);
    std::vector<cudaEvent_t> input_prefetch_events(c10::cuda::device_count(), nullptr);
    auto release_partials = c10::make_scope_exit([&] {
        parallel_for_each_async([&](int device, cudaStream_t stream) {
            if (input_partials[device])
                C10_CUDA_CHECK_WARN(cudaFreeAsync(input_partials[device], stream));
        });
    });

    parallel_for_each_async([&](int device, cudaStream_t stream) {
        int64_t begin, count;
        std::tie(begin, count) = chunk(outputs, device);
        // dX is published in token rows, independently of the feature compute shard.
        const auto [input_begin, input_count] = chunk(rows, device);
        if (grad_input.defined())
        {
            TORCH_CHECK(padded_input <= std::numeric_limits<size_t>::max() / sizeof(scalar_t),
                        "linear_backward: input partial size overflow");
            C10_CUDA_CHECK(cudaMallocAsync(&input_partials[device], padded_input * sizeof(scalar_t), stream));
            if (!count)
                C10_CUDA_CHECK(cudaMemsetAsync(input_partials[device], 0, padded_input * sizeof(scalar_t), stream));
            else if (padded_input > input_numel)
                C10_CUDA_CHECK(cudaMemsetAsync(
                    input_partials[device] + input_numel, 0, (padded_input - input_numel) * sizeof(scalar_t), stream));
        }
        if (!count)
        {
            if (grad_input.defined() && input_count)
            {
                const auto ready_event = record_gemm_output_ready(stream);
                input_prefetch_events[device] =
                    prefetch_gemm_output(grad_input.data_ptr<scalar_t>() + input_begin * features,
                                         input_count * features * sizeof(scalar_t), device, ready_event);
            }
            return;
        }

        constexpr size_t alignment = 256 / sizeof(scalar_t);
        const auto grad_numel = ::cuda::round_up(rows * count, alignment);
        const auto input_offset = ::cuda::round_up(input_numel, alignment);
        const size_t weight_numel = count * features;
        // W is no longer needed after dX; reuse its workspace for X and dW,
        // then reuse it again for db after dW has been published.
        TORCH_CHECK(input_offset <= std::numeric_limits<size_t>::max() / sizeof(scalar_t) &&
                        weight_numel <= std::numeric_limits<size_t>::max() / sizeof(scalar_t) - input_offset &&
                        grad_numel <= std::numeric_limits<size_t>::max() / sizeof(scalar_t) - input_offset - weight_numel,
                    "linear_backward: feature scratch size overflow");
        scalar_t* scratch = nullptr;
        C10_CUDA_CHECK(cudaMallocAsync(&scratch, (grad_numel + input_offset + weight_numel) * sizeof(scalar_t), stream));
        auto release = c10::make_scope_exit([&] { C10_CUDA_CHECK_WARN(cudaFreeAsync(scratch, stream)); });
        auto* local_grad = scratch;
        auto* workspace = scratch + grad_numel;
        copy_gemm_block(grad, 0, rows, begin, count, local_grad, stream);
        const auto grad_ld = grad.is_contiguous() ? count : rows;
        const auto handle = at::cuda::getCurrentCUDABlasHandle();
        const at::cuda::blas::PointerModeGuard pointer_mode(handle, CUBLAS_POINTER_MODE_HOST);
        const scalar_t alpha = 1, beta = 0;

        if (grad_input.defined())
        {
            copy_gemm_block(w, begin, count, 0, features, workspace, stream);
            const auto ready_event = input_count ? record_gemm_output_ready(stream) : nullptr;
            TORCH_CUDABLAS_CHECK(cublasGemm(handle, w.is_contiguous() ? CUBLAS_OP_N : CUBLAS_OP_T,
                                            grad.is_contiguous() ? CUBLAS_OP_N : CUBLAS_OP_T, features, rows, count,
                                            &alpha, workspace, w.is_contiguous() ? features : count, local_grad,
                                            grad_ld, &beta, input_partials[device], features));
            input_prefetch_events[device] =
                prefetch_gemm_output(grad_input.data_ptr<scalar_t>() + input_begin * features,
                                     input_count * features * sizeof(scalar_t), device, ready_event);
        }

        copy_gemm_block(x, 0, rows, 0, features, workspace, stream);
        auto* local_weight = workspace + input_offset;
        TORCH_CUDABLAS_CHECK(cublasGemm(handle, x.is_contiguous() ? CUBLAS_OP_N : CUBLAS_OP_T,
                                        grad.is_contiguous() ? CUBLAS_OP_T : CUBLAS_OP_N, features, count, rows, &alpha,
                                        workspace, x.is_contiguous() ? features : rows, local_grad, grad_ld, &beta,
                                        local_weight, features));
        C10_CUDA_CHECK(cudaMemcpyAsync(grad_weight.data_ptr<scalar_t>() + begin * features, local_weight,
                                       weight_numel * sizeof(scalar_t), cudaMemcpyDefault, stream));

        if (grad_bias.defined())
        {
            const auto options = grad.options().device(c10::DeviceType::CUDA, device);
            auto local_grad_view =
                at::from_blob(local_grad, { rows, count },
                              { grad.is_contiguous() ? count : 1, grad.is_contiguous() ? 1 : rows }, options);
            auto local_bias = at::from_blob(workspace, { count }, options);
            at::sum_out(local_bias, local_grad_view, IntArrayRef{ 0 });
            C10_CUDA_CHECK(cudaMemcpyAsync(grad_bias.data_ptr<scalar_t>() + begin, workspace, count * sizeof(scalar_t),
                                           cudaMemcpyDefault, stream));
        }
    });
    if (grad_input.defined())
        nccl_reduce_scatter_partials(input_partials, grad_input.data_ptr<scalar_t>(), input_numel, features,
                                     [&](int device, cudaStream_t stream) {
                                         // dW/db and the local reduction can overlap prefetching;
                                         // only the managed dX writeback needs to wait.
                                         wait_gemm_output(input_prefetch_events[device], stream);
                                     });
    merge_streams();
}

template <typename scalar_t>
cublasStatus_t cublasMatinvBatched(cublasHandle_t handle,
                                   int n,
                                   const scalar_t* const A[],
                                   int lda,
                                   scalar_t* const Ainv[],
                                   int lda_inv,
                                   int* info,
                                   int batchSize)
{
    TORCH_CHECK(false, "Unsupported datatype for batched matrix inverse");
}

template <>
cublasStatus_t cublasMatinvBatched(
    cublasHandle_t handle, int n, const float* const A[], int lda, float* const Ainv[], int lda_inv, int* info, int batchSize)
{
    return cublasSmatinvBatched(handle, n, A, lda, Ainv, lda_inv, info, batchSize);
    TORCH_CHECK(false, "Unsupported datatype for batched matrix inverse");
}

template <>
cublasStatus_t cublasMatinvBatched(cublasHandle_t handle,
                                   int n,
                                   const double* const A[],
                                   int lda,
                                   double* const Ainv[],
                                   int lda_inv,
                                   int* info,
                                   int batchSize)
{
    return cublasDmatinvBatched(handle, n, A, lda, Ainv, lda_inv, info, batchSize);
}

} // namespace

Tensor& addmm_out(
    const Tensor& self, const Tensor& mat1, const Tensor& mat2, const Scalar& beta, const Scalar& alpha, Tensor& out)
{
    int device = cudaInvalidDeviceId;
    C10_CUDA_CHECK(cudaGetDevice(&device));
    TORCH_CHECK(mat1.dim() == 2 && mat2.dim() == 2, "addmm: mat1 and mat2 must be matrices");
    TORCH_CHECK(mat1.size(1) == mat2.size(0), "addmm: mat1 and mat2 shapes cannot be multiplied");
    TORCH_CHECK(self.scalar_type() == mat1.scalar_type() && mat1.scalar_type() == mat2.scalar_type() &&
                    out.scalar_type() == mat1.scalar_type(),
                "addmm: self, mat1, mat2 and out must have the same dtype");
    TORCH_CHECK(self.device() == mat1.device() && mat1.device() == mat2.device() && out.device() == mat1.device(),
                "addmm: all tensors must be on the same device");

    const auto rows = mat1.size(0);
    const auto columns = mat2.size(1);
    const auto reduction_size = mat1.size(1);
    if (out.is_same(self))
    {
        TORCH_CHECK(self.dim() == 2 && self.size(0) == rows && self.size(1) == columns,
                    "addmm: an in-place self must match the output shape");
    }
    // Validate broadcasting even when beta is zero, without reading self's data.
    const auto self_expanded = self.expand({ rows, columns });
    // Expanded or strided aliases can evade ATen's partial-overlap check.
    TORCH_CHECK(out.is_same(self) || !out.is_alias_of(self),
                "addmm: self and out must not alias unless they are the same tensor; clone self first");
    at::native::resize_output(out, { rows, columns });
    at::assert_no_internal_overlap(out);
    if (!out.numel())
    {
        C10_CUDA_CHECK(cudaSetDevice(device));
        return out;
    }

    const auto beta_value = beta.toDouble();
    if (!reduction_size)
    {
        if (beta_value == 0.0)
            out.zero_();
        else
        {
            out.copy_(self_expanded);
            out.mul_(beta);
        }
        C10_CUDA_CHECK(cudaSetDevice(device));
        return out;
    }

    TORCH_CHECK(rows <= std::numeric_limits<int>::max() && columns <= std::numeric_limits<int>::max() &&
                    reduction_size <= std::numeric_limits<int>::max(),
                "addmm: matrix dimensions exceed the cuBLAS integer range");

    // A DGX-only workload need not have created a CUDA tensor yet. Initialize PyTorch's CUDA
    // allocator before requesting its BLAS handles and their device-local workspaces.
    at::globalContext().lazyInitDevice(c10::DeviceType::CUDA);

    const Tensor mat1_ = is_gemm_layout(mat1) ? mat1 : mat1.contiguous();
    const Tensor mat2_ = is_gemm_layout(mat2) ? mat2 : mat2.contiguous();
    auto result = out.is_contiguous() ? out : at::empty({ rows, columns }, out.options());

    // Both cuBLAS paths compute result^T = mat2^T * mat1^T in column-major layout.
    const auto transa = mat2_.is_contiguous() ? CUBLAS_OP_N : CUBLAS_OP_T;
    const auto transb = mat1_.is_contiguous() ? CUBLAS_OP_N : CUBLAS_OP_T;
    const auto lda = transa == CUBLAS_OP_N ? columns : reduction_size;

    // Match PyTorch's addmm Lt eligibility: the bias epilogue is unscaled, and it broadcasts over
    // column-major columns, giving the required row-wise bias in result.
    static const bool disable_addmm_cuda_lt = c10::utils::get_env("DISABLE_ADDMM_CUDA_LT") == "1";
    const bool use_lt = !disable_addmm_cuda_lt && beta_value == 1.0 && self.is_contiguous() &&
                        (self.dim() == 1 || (self.dim() == 2 && self.size(0) == 1)) && self.size(-1) == columns &&
                        columns > 1 && reduction_size > 1;

    AT_DISPATCH_V2(
        self.scalar_type(), "addmm_out", AT_WRAP([&] {
            using opmath_t = at::opmath_type<scalar_t>;
            const auto alpha_ = alpha.to<opmath_t>();
            const auto beta_ = beta.to<opmath_t>();
            // Keep the bias 256-byte aligned even when the RHS has an odd element count.
            constexpr int64_t bias_alignment = 256 / sizeof(scalar_t);
            const auto bias_offset = ::cuda::round_up(mat2_.numel(), bias_alignment);

            parallel_for_each([&](int device, cudaStream_t stream) {
                int64_t row_offset, row_count;
                std::tie(row_offset, row_count) = chunk(rows, device);
                if (!row_count)
                    return;

                // Stage the shared RHS and this GPU's LHS/output shard in local memory.
                scalar_t* mat2_local = nullptr;
                const bool fuse_bias = use_lt && row_count > 1;
                const auto local_numel = fuse_bias ? bias_offset + columns : mat2_.numel();
                C10_CUDA_CHECK(cudaMallocAsync(&mat2_local, local_numel * sizeof(scalar_t), stream));
                auto release_mat2 = c10::make_scope_exit([&] { C10_CUDA_CHECK_WARN(cudaFreeAsync(mat2_local, stream)); });
                C10_CUDA_CHECK(cudaMemcpyAsync(
                    mat2_local, mat2_.const_data_ptr<scalar_t>(), mat2_.nbytes(), cudaMemcpyDefault, stream));
                GemmRowScratch<scalar_t> shard(mat1_, result, row_offset, row_count, stream);
                shard.copy_input(mat1_, row_offset);
                const auto ready_event = record_gemm_output_ready(stream);
                const auto* mat1_ptr = shard.input();
                auto* result_ptr = shard.output();
                const auto ldb = shard.leading_dimension();

                if (fuse_bias)
                {
                    auto* bias_local = mat2_local + bias_offset;
                    C10_CUDA_CHECK(cudaMemcpyAsync(bias_local, self.const_data_ptr<scalar_t>(),
                                                   columns * sizeof(scalar_t), cudaMemcpyDefault, stream));
                    if (addmmBiasLt<scalar_t>(transa, transb, columns, row_count, reduction_size, alpha_, mat2_local,
                                              lda, mat1_ptr, ldb, bias_local, result_ptr, stream))
                    {
                        shard.copy_output(device, ready_event);
                        return;
                    }
                }

                // Lt can decline a shape/alignment. Initialize only this GPU's rows before falling
                // back, including scalar/column broadcasts and arbitrary beta. beta=0 never reads self.
                if (beta_value != 0.0)
                {
                    const auto* self_ptr =
                        self_expanded.const_data_ptr<scalar_t>() + row_offset * self_expanded.stride(0);
                    const auto row_stride = self_expanded.stride(0);
                    const auto column_stride = self_expanded.stride(1);
                    const auto numel = row_count * columns;
                    const auto blocks = static_cast<unsigned int>(
                        std::min<int64_t>(::cuda::ceil_div(numel, kNumThreads), std::numeric_limits<int>::max()));
                    lambda_kernel<<<blocks, kNumThreads, 0, stream>>>(numel, AddmmCopySelfFunctor<scalar_t>(), result_ptr,
                                                                      self_ptr, columns, row_stride, column_stride);
                    C10_CUDA_KERNEL_LAUNCH_CHECK();
                }
                const auto handle = at::cuda::getCurrentCUDABlasHandle();
                const at::cuda::blas::PointerModeGuard pointer_mode(handle, CUBLAS_POINTER_MODE_HOST);
                TORCH_CUDABLAS_CHECK(cublasGemm(handle, transa, transb, columns, row_count, reduction_size, &alpha_,
                                                mat2_local, lda, mat1_ptr, ldb, &beta_, result_ptr, columns));
                shard.copy_output(device, ready_event);
            });
        }),
        AT_EXPAND(AT_FLOATING_TYPES));

    if (!out.is_same(result))
        out.copy_(result);

    C10_CUDA_CHECK(cudaSetDevice(device));
    return out;
}

Tensor& mm_out(const Tensor& self, const Tensor& mat2, Tensor& out)
{
    int device = cudaInvalidDeviceId;
    C10_CUDA_CHECK(cudaGetDevice(&device));
    c10::call_once(gBLASHandlesInitFlag, initBLASHandles);

    TORCH_CHECK(self.dim() == 2);
    TORCH_CHECK(mat2.dim() == 2);
    TORCH_CHECK(out.dim() == 2);

    TORCH_CHECK(self.sizes()[1] == mat2.sizes()[0]);
    TORCH_CHECK(out.sizes()[1] == mat2.sizes()[1]);

    TORCH_CHECK(out.is_contiguous());

    Tensor self_ = is_gemm_layout(self) ? self : self.contiguous();
    Tensor mat2_ = is_gemm_layout(mat2) ? mat2 : mat2.contiguous();

    const auto rows = self_.size(0);
    const auto columns = mat2_.size(1);
    const auto reduction_size = self_.size(1);
    // For a weight gradient, compare communication for a token split (reduce a
    // full partial output per GPU) with a feature split (replicate the smaller
    // input and exchange the nonlocal fraction of the larger input). Staging
    // makes these single-pass volumes, independent of cuBLAS tile rereads.
    // Assume balanced token ownership and compare normalized volumes, using
    // the same policy for every device count.
    const bool weight_gradient = !self_.is_contiguous() && mat2_.is_contiguous();
    const bool split_reduction = weight_gradient && prefer_token_sharded_weight_gradient(rows, columns, reduction_size);
    const bool split_columns = weight_gradient && !split_reduction && columns > rows;

    if (!out.numel())
    {
        C10_CUDA_CHECK(cudaSetDevice(device));
        return out;
    }
    if (!reduction_size)
    {
        out.zero_();
        C10_CUDA_CHECK(cudaSetDevice(device));
        return out;
    }

    if (split_reduction)
    {
        AT_DISPATCH_V2(
            self.scalar_type(), "mm_out", AT_WRAP([&] {
                using opmath_t = at::opmath_type<scalar_t>;
                const auto matrix_numel = rows * columns;
                const auto padded_numel = nccl_reduce_scatter_padded_numel(matrix_numel);
                std::vector<scalar_t*> partial_results(c10::cuda::device_count());

                parallel_for_each_async([&](int device, cudaStream_t stream) {
                    int64_t reduction_offset, reduction_count;
                    std::tie(reduction_offset, reduction_count) = chunk(reduction_size, device);
                    C10_CUDA_CHECK(cudaMallocAsync(&partial_results[device], padded_numel * sizeof(scalar_t), stream));
                    if (!reduction_count)
                    {
                        C10_CUDA_CHECK(
                            cudaMemsetAsync(partial_results[device], 0, padded_numel * sizeof(scalar_t), stream));
                        return;
                    }
                    if (padded_numel > matrix_numel)
                        C10_CUDA_CHECK(cudaMemsetAsync(partial_results[device] + matrix_numel, 0,
                                                       (padded_numel - matrix_numel) * sizeof(scalar_t), stream));

                    scalar_t* mat2_local = nullptr;
                    const auto mat2_numel = reduction_count * columns;
                    C10_CUDA_CHECK(
                        cudaMallocAsync(&mat2_local, (mat2_numel + reduction_count * rows) * sizeof(scalar_t), stream));
                    auto release_mat2 =
                        c10::make_scope_exit([&] { C10_CUDA_CHECK_WARN(cudaFreeAsync(mat2_local, stream)); });
                    C10_CUDA_CHECK(cudaMemcpyAsync(mat2_local,
                                                   mat2_.const_data_ptr<scalar_t>() + reduction_offset * columns,
                                                   mat2_numel * sizeof(scalar_t), cudaMemcpyDefault, stream));
                    auto* self_local = mat2_local + mat2_numel;
                    copy_gemm_block(self_, 0, rows, reduction_offset, reduction_count, self_local, stream);

                    const auto handle = gBLASHandles[device];
                    TORCH_CUDABLAS_CHECK(cublasSetStream(handle, stream));
                    const opmath_t alpha = 1;
                    const opmath_t beta = 0;
                    TORCH_CUDABLAS_CHECK(cublasGemm(handle, CUBLAS_OP_N, CUBLAS_OP_T, columns, rows, reduction_count,
                                                    &alpha, mat2_local, columns, self_local, rows, &beta,
                                                    partial_results[device], columns));
                });

                nccl_reduce_scatter_partials(partial_results, out.data_ptr<scalar_t>(), matrix_numel);

                parallel_for_each_async([&](int device, cudaStream_t stream) {
                    C10_CUDA_CHECK(cudaFreeAsync(partial_results[device], stream));
                });
                merge_streams();
            }),
            AT_EXPAND(AT_FLOATING_TYPES));
        C10_CUDA_CHECK(cudaSetDevice(device));
        return out;
    }

    if (split_columns)
    {
        // Wide weight gradients replicate dY, not the larger activation matrix.
        AT_DISPATCH_V2(
            self.scalar_type(), "mm_out", AT_WRAP([&] {
                parallel_for_each([&](int device, cudaStream_t stream) {
                    int64_t column_offset, column_count;
                    std::tie(column_offset, column_count) = chunk(columns, device);
                    if (!column_count)
                        return;
                    scalar_t* scratch = nullptr;
                    const auto self_numel = rows * reduction_size;
                    const auto mat2_numel = reduction_size * column_count;
                    C10_CUDA_CHECK(cudaMallocAsync(
                        &scratch, (self_numel + mat2_numel + rows * column_count) * sizeof(scalar_t), stream));
                    auto release = c10::make_scope_exit([&] { C10_CUDA_CHECK_WARN(cudaFreeAsync(scratch, stream)); });
                    auto* mat2_local = scratch + self_numel;
                    auto* output_local = mat2_local + mat2_numel;
                    copy_gemm_block(self_, 0, rows, 0, reduction_size, scratch, stream);
                    copy_gemm_block(mat2_, 0, reduction_size, column_offset, column_count, mat2_local, stream);
                    const auto handle = gBLASHandles[device];
                    TORCH_CUDABLAS_CHECK(cublasSetStream(handle, stream));
                    const scalar_t alpha = 1, beta = 0;
                    TORCH_CUDABLAS_CHECK(cublasGemm(handle, CUBLAS_OP_N, CUBLAS_OP_T, column_count, rows,
                                                    reduction_size, &alpha, mat2_local, column_count, scratch, rows,
                                                    &beta, output_local, column_count));
                    C10_CUDA_CHECK(cudaMemcpy2DAsync(out.data_ptr<scalar_t>() + column_offset, columns * sizeof(scalar_t),
                                                     output_local, column_count * sizeof(scalar_t),
                                                     column_count * sizeof(scalar_t), rows, cudaMemcpyDefault, stream));
                });
            }),
            AT_EXPAND(AT_FLOATING_TYPES));
    }
    else
    {
        AT_DISPATCH_V2(self.scalar_type(), "mm_out", AT_WRAP([&] {
                           parallel_for_each([&](int device, cudaStream_t stream) {
                               int64_t row_offset, row_count;
                               std::tie(row_offset, row_count) = chunk(rows, device);
                               if (!row_count)
                                   return;
                               scalar_t* mat2_local = nullptr;
                               C10_CUDA_CHECK(cudaMallocAsync(&mat2_local, mat2_.nbytes(), stream));
                               auto release =
                                   c10::make_scope_exit([&] { C10_CUDA_CHECK_WARN(cudaFreeAsync(mat2_local, stream)); });
                               C10_CUDA_CHECK(cudaMemcpyAsync(mat2_local, mat2_.const_data_ptr<scalar_t>(),
                                                              mat2_.nbytes(), cudaMemcpyDefault, stream));
                               GemmRowScratch<scalar_t> shard(self_, out, row_offset, row_count, stream);
                               shard.copy_input(self_, row_offset);
                               const auto ready_event = record_gemm_output_ready(stream);
                               const auto handle = gBLASHandles[device];
                               TORCH_CUDABLAS_CHECK(cublasSetStream(handle, stream));
                               const scalar_t alpha = 1, beta = 0;
                               TORCH_CUDABLAS_CHECK(cublasGemm(
                                   handle, mat2_.is_contiguous() ? CUBLAS_OP_N : CUBLAS_OP_T,
                                   self_.is_contiguous() ? CUBLAS_OP_N : CUBLAS_OP_T, columns, row_count,
                                   reduction_size, &alpha, mat2_local, mat2_.is_contiguous() ? columns : reduction_size,
                                   shard.input(), shard.leading_dimension(), &beta, shard.output(), columns));
                               shard.copy_output(device, ready_event);
                           });
                       }),
                       AT_EXPAND(AT_FLOATING_TYPES));
    }

    C10_CUDA_CHECK(cudaSetDevice(device));
    return out;
}

Tensor linear(const Tensor& input, const Tensor& weight, const std::optional<Tensor>& bias)
{
    return at::native::linear(input, weight, bias);
}

std::tuple<Tensor, Tensor, Tensor> linear_backward(const Tensor& input,
                                                   const Tensor& grad_output,
                                                   const Tensor& weight,
                                                   std::array<bool, 3> output_mask)
{
    TORCH_CHECK(input.dim() >= 1 && (weight.dim() == 1 || weight.dim() == 2),
                "linear_backward: expected input with at least one dimension and 1D or 2D weight");
    TORCH_CHECK(input.device() == weight.device() && input.device() == grad_output.device(),
                "linear_backward: all tensors must be on the same device");
    TORCH_CHECK(input.scalar_type() == weight.scalar_type() && input.scalar_type() == grad_output.scalar_type(),
                "linear_backward: all tensors must have the same dtype");
    const auto features = input.size(-1);
    const auto outputs = weight.dim() == 1 ? 1 : weight.size(0);
    TORCH_CHECK(weight.size(-1) == features, "linear_backward: input and weight features must match");
    auto output_shape = input.sizes().vec();
    output_shape.pop_back();
    int64_t rows = 1;
    for (const auto size : output_shape)
    {
        TORCH_CHECK(!size || rows <= std::numeric_limits<int64_t>::max() / size,
                    "linear_backward: flattened input size overflow");
        rows *= size;
    }
    if (weight.dim() == 2)
        output_shape.push_back(outputs);
    TORCH_CHECK(grad_output.sizes() == IntArrayRef(output_shape),
                "linear_backward: grad_output must match the linear output shape");
    TORCH_CHECK(rows <= INT_MAX && features <= INT_MAX && outputs <= INT_MAX,
                "linear_backward: matrix dimensions exceed the cuBLAS integer range");

    Tensor grad_input, grad_weight, grad_bias;
    if (!output_mask[0] && !output_mask[1] && !output_mask[2])
        return { grad_input, grad_weight, grad_bias };

    int device = cudaInvalidDeviceId;
    C10_CUDA_CHECK(cudaGetDevice(&device));
    auto restore_device = c10::make_scope_exit([&] { C10_CUDA_CHECK_WARN(cudaSetDevice(device)); });
    const auto bias_shape = weight.dim() == 1 ? std::vector<int64_t>{} : std::vector<int64_t>{ outputs };
    if (!rows || !outputs)
    {
        if (output_mask[0])
            grad_input = at::empty(input.sizes(), input.options()).zero_();
        if (output_mask[1])
            grad_weight = at::empty(weight.sizes(), weight.options()).zero_();
        if (output_mask[2])
            grad_bias = at::empty(bias_shape, weight.options()).zero_();
        return { grad_input, grad_weight, grad_bias };
    }

    // Preserve row/column-major inputs in local scratch. General strided and
    // broadcast gradients are materialized only once, shared by all consumers.
    Tensor grad = grad_output.reshape({ rows, outputs });
    if (!is_gemm_layout(grad))
        grad = grad.contiguous();
    const bool need_input = output_mask[0] && features;
    const bool need_weight = output_mask[1] && features;
    Tensor x, w;
    if (need_weight)
    {
        x = input.reshape({ rows, features });
        if (!is_gemm_layout(x))
            x = x.contiguous();
    }
    if (need_input)
    {
        w = weight.reshape({ outputs, features });
        if (!is_gemm_layout(w))
            w = w.contiguous();
    }

    if (output_mask[0])
        grad_input = at::empty(input.sizes(), input.options());
    if (output_mask[1])
        grad_weight = at::empty(weight.sizes(), weight.options());
    if (output_mask[2])
        grad_bias = at::empty(bias_shape, weight.options());
    if (!need_input && !need_weight && !output_mask[2])
        return { grad_input, grad_weight, grad_bias };

    at::globalContext().lazyInitDevice(c10::DeviceType::CUDA);
    AT_DISPATCH_V2(
        input.scalar_type(), "linear_backward", AT_WRAP([&] {
            if (need_weight && !prefer_token_sharded_linear_backward(outputs, features, rows, need_input, output_mask[2]))
            {
                linear_backward_feature_shards<scalar_t>(x, grad, w, grad_input, grad_weight, grad_bias);
                return;
            }
            std::vector<scalar_t*> weight_partials(c10::cuda::device_count(), nullptr),
                bias_partials(c10::cuda::device_count(), nullptr);
            const auto weight_numel = outputs * features;
            const auto padded_weight = need_weight ? nccl_reduce_scatter_padded_numel(weight_numel) : 0;
            const auto padded_bias = output_mask[2] ? nccl_reduce_scatter_padded_numel(outputs) : 0;
            auto release_partials = c10::make_scope_exit([&] {
                parallel_for_each_async([&](int d, cudaStream_t stream) {
                    if (weight_partials[d])
                        C10_CUDA_CHECK_WARN(cudaFreeAsync(weight_partials[d], stream));
                    if (bias_partials[d])
                        C10_CUDA_CHECK_WARN(cudaFreeAsync(bias_partials[d], stream));
                });
            });

            parallel_for_each_async([&](int d, cudaStream_t stream) {
                const auto [begin, count] = chunk(rows, d);
                if (need_weight)
                {
                    C10_CUDA_CHECK(cudaMallocAsync(&weight_partials[d], padded_weight * sizeof(scalar_t), stream));
                    if (!count)
                        C10_CUDA_CHECK(cudaMemsetAsync(weight_partials[d], 0, padded_weight * sizeof(scalar_t), stream));
                    else if (padded_weight > weight_numel)
                        C10_CUDA_CHECK(cudaMemsetAsync(weight_partials[d] + weight_numel, 0,
                                                       (padded_weight - weight_numel) * sizeof(scalar_t), stream));
                }
                if (output_mask[2])
                {
                    C10_CUDA_CHECK(cudaMallocAsync(&bias_partials[d], padded_bias * sizeof(scalar_t), stream));
                    if (!count)
                        C10_CUDA_CHECK(cudaMemsetAsync(bias_partials[d], 0, padded_bias * sizeof(scalar_t), stream));
                    else if (padded_bias > outputs)
                        C10_CUDA_CHECK(cudaMemsetAsync(
                            bias_partials[d] + outputs, 0, (padded_bias - outputs) * sizeof(scalar_t), stream));
                }
                if (!count)
                    return;

                constexpr size_t alignment = 256 / sizeof(scalar_t);
                const auto grad_numel = ::cuda::round_up(count * outputs, alignment);
                const auto weight_offset = ::cuda::round_up(weight_numel, alignment);
                // Once dX is published, its weight/output workspace can stage X for dW.
                const size_t workspace_numel = std::max(need_input ? weight_offset + count * features : size_t(0),
                                                        need_weight ? count * features : size_t(0));
                TORCH_CHECK(workspace_numel <= std::numeric_limits<size_t>::max() / sizeof(scalar_t) &&
                                grad_numel <= std::numeric_limits<size_t>::max() / sizeof(scalar_t) - workspace_numel,
                            "linear_backward: scratch size overflow");
                scalar_t* scratch = nullptr;
                C10_CUDA_CHECK(cudaMallocAsync(&scratch, (grad_numel + workspace_numel) * sizeof(scalar_t), stream));
                auto release = c10::make_scope_exit([&] { C10_CUDA_CHECK_WARN(cudaFreeAsync(scratch, stream)); });
                auto* local_grad = scratch;
                auto* workspace = scratch + grad_numel;
                copy_gemm_block(grad, begin, count, 0, outputs, local_grad, stream);
                const auto grad_ld = grad.is_contiguous() ? outputs : count;
                const auto handle = at::cuda::getCurrentCUDABlasHandle();
                const at::cuda::blas::PointerModeGuard pointer_mode(handle, CUBLAS_POINTER_MODE_HOST);
                const scalar_t alpha = 1, beta = 0;

                if (need_input)
                {
                    copy_gemm_block(w, 0, outputs, 0, features, workspace, stream);
                    const auto ready_event = record_gemm_output_ready(stream);
                    auto* local_output = workspace + weight_offset;
                    TORCH_CUDABLAS_CHECK(cublasGemm(handle, w.is_contiguous() ? CUBLAS_OP_N : CUBLAS_OP_T,
                                                    grad.is_contiguous() ? CUBLAS_OP_N : CUBLAS_OP_T, features, count,
                                                    outputs, &alpha, workspace, w.is_contiguous() ? features : outputs,
                                                    local_grad, grad_ld, &beta, local_output, features));
                    const auto prefetch_event = prefetch_gemm_output(grad_input.data_ptr<scalar_t>() + begin * features,
                                                                     count * features * sizeof(scalar_t), d, ready_event);
                    wait_gemm_output(prefetch_event, stream);
                    C10_CUDA_CHECK(cudaMemcpyAsync(grad_input.data_ptr<scalar_t>() + begin * features, local_output,
                                                   count * features * sizeof(scalar_t), cudaMemcpyDefault, stream));
                }
                if (need_weight)
                {
                    copy_gemm_block(x, begin, count, 0, features, workspace, stream);
                    TORCH_CUDABLAS_CHECK(cublasGemm(handle, x.is_contiguous() ? CUBLAS_OP_N : CUBLAS_OP_T,
                                                    grad.is_contiguous() ? CUBLAS_OP_T : CUBLAS_OP_N, features, outputs,
                                                    count, &alpha, workspace, x.is_contiguous() ? features : count,
                                                    local_grad, grad_ld, &beta, weight_partials[d], features));
                }
                if (output_mask[2])
                {
                    // Reuse CUDA's parallel sum on the staged shard. These views
                    // borrow scratch owned by this call and enqueue on its stream.
                    const auto options = grad.options().device(c10::DeviceType::CUDA, d);
                    auto local_grad_view = at::from_blob(
                        local_grad, { int64_t(count), outputs },
                        { grad.is_contiguous() ? outputs : 1, grad.is_contiguous() ? 1 : int64_t(count) }, options);
                    auto local_bias = at::from_blob(bias_partials[d], { outputs }, options);
                    at::sum_out(local_bias, local_grad_view, IntArrayRef{ 0 });
                }
            });
            if (need_weight)
                nccl_reduce_scatter_partials(weight_partials, grad_weight.data_ptr<scalar_t>(), weight_numel);
            if (output_mask[2])
                nccl_reduce_scatter_partials(bias_partials, grad_bias.data_ptr<scalar_t>(), outputs);
            merge_streams();
        }),
        AT_EXPAND(AT_FLOATING_TYPES));
    return { grad_input, grad_weight, grad_bias };
}

Tensor& bmm_out(const Tensor& self_, const Tensor& mat2, Tensor& out)
{
    c10::call_once(gBLASHandlesInitFlag, initBLASHandles);

    TORCH_CHECK(self_.dim() == 3);
    TORCH_CHECK(mat2.dim() == 3);
    TORCH_CHECK(out.dim() == 3);

    TORCH_CHECK(out.is_contiguous());

    TORCH_CHECK(self_.sizes()[0] == mat2.sizes()[0]);
    TORCH_CHECK(self_.sizes()[2] == mat2.sizes()[1]);
    TORCH_CHECK(out.sizes()[2] == mat2.sizes()[2]);

    Tensor self = is_gemm_layout(self_) ? self_ : self_.contiguous();
    Tensor mat2_ = is_gemm_layout(mat2) ? mat2 : mat2.contiguous();

    parallel_for_each([&](int device, cudaStream_t stream) {
        AT_DISPATCH_V2(self.scalar_type(), "bmm_out", AT_WRAP([&] {
                           using opmath_t = at::opmath_type<scalar_t>;

                           auto handle = gBLASHandles[device];
                           TORCH_CUDABLAS_CHECK(cublasSetStream(handle, stream));

                           opmath_t alpha = 1.f;
                           opmath_t beta = 0.f;
                           const auto self_data_ptr = static_cast<const opmath_t*>(self.const_data_ptr());
                           const auto mat2_data_ptr = static_cast<const opmath_t*>(mat2_.const_data_ptr());
                           auto out_data_ptr = static_cast<opmath_t*>(out.data_ptr());

                           int64_t device_batch_offset, device_batch_count;
                           std::tie(device_batch_offset, device_batch_count) = chunk(self.size(0), device);

                           if (device_batch_count)
                           {
                               const auto self_stripe_ptr = self_data_ptr + device_batch_offset * self.strides()[0];
                               const auto mat2_stripe_ptr = mat2_data_ptr + device_batch_offset * mat2_.strides()[0];
                               auto out_stripe_ptr = out_data_ptr + device_batch_offset * out.strides()[0];

                               TORCH_CUDABLAS_CHECK(cublasGemmStridedBatched(
                                   handle, mat2_.is_contiguous() ? CUBLAS_OP_N : CUBLAS_OP_T,
                                   self.is_contiguous() ? CUBLAS_OP_N : CUBLAS_OP_T, mat2_.sizes()[2], self.sizes()[1],
                                   self.sizes()[2], &alpha, mat2_stripe_ptr,
                                   mat2_.is_contiguous() ? mat2_.sizes()[2] : mat2_.sizes()[1], mat2_.strides()[0],
                                   self_stripe_ptr, self.is_contiguous() ? self.sizes()[2] : self.sizes()[1],
                                   self.strides()[0], &beta, out_stripe_ptr, out.sizes()[2], out.strides()[0],
                                   device_batch_count));
                           }
                       }),
                       AT_EXPAND(AT_FLOATING_TYPES));
    });

    return out;
}

namespace
{

template <typename scalar_t>
Tensor get_device_pointers(const Tensor& input)
{
    auto input_data = input.const_data_ptr<scalar_t>();
    auto input_mat_stride = at::native::matrixStride(input);

    auto batch_size = at::native::batchCount(input);

    // if batch_size==0, then start=0 and end=0
    // if input_mat_stride==0, then step=sizeof(scalar_t)
    return at::arange(
        /*start=*/reinterpret_cast<int64_t>(input_data),
        /*end=*/reinterpret_cast<int64_t>(input_data + batch_size * input_mat_stride),
        /*step=*/static_cast<int64_t>(std::max<int64_t>(input_mat_stride, 1) * sizeof(scalar_t)),
        input.options().dtype(at::kLong));
}

} // namespace

std::tuple<Tensor&, Tensor&> linalg_inv_ex_out(const Tensor& A, bool check_errors, Tensor& inverse, Tensor& info)
{
    c10::call_once(gBLASHandlesInitFlag, initBLASHandles);

    TORCH_CHECK(A.dim() >= 2);
    TORCH_CHECK(A.size(-2) == A.size(-1));
    TORCH_CHECK(A.size(-1) <= 32); // limitation of cublasSmatinvBatched
    TORCH_CHECK(!check_errors);

    const auto batch_count = at::native::batchCount(A);
    const auto matrix_size = A.size(-1);

    // cuBLAS reads and writes column-major matrices. Flatten the input batch and
    // use a row-major view of inverse so that row-major A is interpreted as A^T
    // and the cuBLAS result is exposed as A^-1.
    auto A_flat = A.contiguous().view({ batch_count, matrix_size, matrix_size });
    // Need output from cuBLAS to have the same layout as A
    inverse = inverse.transpose(-2, -1);
    auto inverse_flat = inverse.view({ batch_count, matrix_size, matrix_size });
    auto info_flat = info.view({ batch_count });

    int lda = matrix_size;
    int lda_inv = matrix_size;

    AT_DISPATCH_V2(A.scalar_type(), "linalg_inv_ex_out", AT_WRAP([&] {
                       auto A_ptr_array = get_device_pointers<scalar_t>(A_flat);
                       auto Ainv_ptr_array = get_device_pointers<scalar_t>(inverse_flat);

                       parallel_for_each([&](int device, cudaStream_t stream) {
                           auto handle = gBLASHandles[device];
                           TORCH_CUDABLAS_CHECK(cublasSetStream(handle, stream));

                           auto info_data_ptr = info_flat.data_ptr<int>();

                           int64_t device_batch_offset, device_batch_count;
                           std::tie(device_batch_offset, device_batch_count) = chunk(batch_count, device);

                           if (device_batch_count)
                           {
                               auto A_ptr_array_data =
                                   reinterpret_cast<scalar_t**>(A_ptr_array.data_ptr()) + device_batch_offset;
                               auto info_stripe_ptr = info_data_ptr + device_batch_offset;
                               auto Ainv_ptr_array_data =
                                   reinterpret_cast<scalar_t**>(Ainv_ptr_array.data_ptr()) + device_batch_offset;

                               TORCH_CUDABLAS_CHECK(cublasMatinvBatched(handle, matrix_size, A_ptr_array_data, lda,
                                                                     Ainv_ptr_array_data, lda_inv, info_stripe_ptr,
                                                                     device_batch_count));
                           }
                       });
                   }),
                   AT_EXPAND(AT_FLOATING_TYPES));

    return std::tie(inverse, info);
}

TORCH_LIBRARY_IMPL(aten, PrivateUse1, m)
{
    // Retain PyTorch's forward decomposition, but give autograd a linear node
    // so its existing derivative dispatches to the combined backward below.
    m.impl(TORCH_SELECTIVE_NAME("aten::linear"), TORCH_FN(linear));
    m.impl(TORCH_SELECTIVE_NAME("aten::linear_backward"), TORCH_FN(linear_backward));
    m.impl(TORCH_SELECTIVE_NAME("aten::addmm.out"), TORCH_FN(addmm_out));
    m.impl(TORCH_SELECTIVE_NAME("aten::mm.out"), TORCH_FN(mm_out));
    m.impl(TORCH_SELECTIVE_NAME("aten::bmm.out"), TORCH_FN(bmm_out));
    m.impl(TORCH_SELECTIVE_NAME("aten::linalg_inv_ex.inverse"), TORCH_FN(linalg_inv_ex_out));
}

} // namespace dgx

} // namespace at::native
