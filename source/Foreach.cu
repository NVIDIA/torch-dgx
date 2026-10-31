/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
#include "KernelLaunchUtils.cuh"

#include <ATen/Dispatch_v2.h>
#include <ATen/OpMathType.h>
#include <ATen/native/BinaryOps.h>
#include <ATen/native/ForeachUtils.h>
#include <ATen/native/Lerp.h>
#include <ATen/ops/empty_like.h>

#include <torch/library.h>

#include <type_traits>
#include <unordered_map>

namespace at::native::dgx
{
namespace
{

enum class ForeachOp
{
    Add,
    Sub,
    Mul,
    Div,
    Maximum,
    Lerp,
    Addcmul,
    Addcdiv,
    Sqrt,
    Reciprocal,
    Neg
};

// Pass metadata by value: no device allocation, host synchronization, or asynchronous
// copy from temporary host storage. Stay below the 4 KiB kernel argument limit.
constexpr int kForeachBatchSize = 32;
constexpr int64_t kForeachBlockSize = 4096;

template <typename scalar_t>
struct ForeachTensor
{
    const scalar_t* input;
    const scalar_t* other;
    const scalar_t* third;
    scalar_t* output;
    int64_t offset;
    int64_t size;
    int64_t block_end;
    at::opmath_type<scalar_t> value;
};

template <typename scalar_t>
struct ForeachBatch
{
    ForeachTensor<scalar_t> tensors[kForeachBatchSize];
    int count = 0;
    int64_t blocks = 0;
};
static_assert(sizeof(ForeachBatch<double>) < 4000);

template <ForeachOp Op, typename T>
__device__ T foreach_value(T a, T b, T c, T value, bool has_other)
{
    if constexpr (Op == ForeachOp::Add)
        return a + (has_other ? value * b : b);
    else if constexpr (Op == ForeachOp::Sub)
        return a - (has_other ? value * b : b);
    else if constexpr (Op == ForeachOp::Mul)
        return a * b;
    else if constexpr (Op == ForeachOp::Div)
        return a / b;
    else if constexpr (Op == ForeachOp::Maximum)
        return a != a ? a : (b != b ? b : (a > b ? a : b));
    else if constexpr (Op == ForeachOp::Lerp)
        return at::native::lerp(a, b, c);
    else if constexpr (Op == ForeachOp::Addcmul)
        return a + value * b * c;
    else if constexpr (Op == ForeachOp::Addcdiv)
        return a + value * b / c;
    else if constexpr (Op == ForeachOp::Sqrt)
        return ::cuda::std::sqrt(a);
    else if constexpr (Op == ForeachOp::Reciprocal)
        return T(1) / a;
    else
        return -a;
}

template <ForeachOp Op, typename scalar_t>
__global__ __launch_bounds__(kNumThreads) void foreach_kernel(ForeachBatch<scalar_t> batch, bool scalar_rhs)
{
    using opmath_t = at::opmath_type<scalar_t>;
    for (int64_t block = blockIdx.x; block < batch.blocks; block += gridDim.x)
    {
        int first = 0, last = batch.count;
        while (first < last)
        {
            const int mid = (first + last) / 2;
            if (block < batch.tensors[mid].block_end)
                last = mid;
            else
                first = mid + 1;
        }
        const auto& tensor = batch.tensors[first];
        const auto block_begin = first ? batch.tensors[first - 1].block_end : 0;
        const auto begin = (block - block_begin) * kForeachBlockSize;
        const auto end = ::cuda::std::min(begin + kForeachBlockSize, tensor.size);
        for (int64_t local = begin + threadIdx.x; local < end; local += blockDim.x)
        {
            const auto i = tensor.offset + local;
            const opmath_t a = tensor.input[i];
            const opmath_t b = tensor.other ? static_cast<opmath_t>(tensor.other[scalar_rhs ? 0 : i]) : tensor.value;
            const opmath_t c = tensor.third ? static_cast<opmath_t>(tensor.third[i]) : tensor.value;
            tensor.output[i] = foreach_value<Op>(a, b, c, tensor.value, tensor.other != nullptr);
        }
    }
}

bool can_batch_foreach(ArrayRef<TensorList> lists, ArrayRef<Scalar> scalars, bool inplace, const Tensor* scalar_rhs)
{
    const auto& input = lists[0];
    const auto dtype = input[0].scalar_type();
    if (input[0].device().type() != kPrivateUse1 ||
        (dtype != kFloat && dtype != kDouble && dtype != kHalf && dtype != kBFloat16) ||
        !can_use_fast_route(lists, scalars))
        return false;

    // Reject aliases between different list entries, including read-after-write
    // dependencies. Within an entry, exact elementwise aliases are safe. Sharing
    // storage conservatively takes the sequential path even for disjoint views.
    std::unordered_map<c10::StorageImpl*, std::pair<size_t, const void*>> storage_entries;
    for (const auto& list : lists)
    {
        for (const auto i : c10::irange(list.size()))
        {
            const auto& tensor = list[i];
            if (tensor.is_conj() || tensor.is_neg())
                return false;
            if (inplace && tensor.numel())
            {
                auto [entry, inserted] = storage_entries.emplace(
                    tensor.storage().unsafeGetStorageImpl(), std::make_pair(i, tensor.const_data_ptr()));
                if (!inserted && entry->second != std::make_pair(i, tensor.const_data_ptr()))
                    return false;
            }
        }
    }
    if (scalar_rhs)
    {
        if (scalar_rhs->dim() != 0 || scalar_rhs->device() != input[0].device() || scalar_rhs->scalar_type() != dtype ||
            scalar_rhs->is_conj() || scalar_rhs->is_neg())
            return false;
        if (inplace && storage_entries.count(scalar_rhs->storage().unsafeGetStorageImpl()))
            return false;
    }
    return true;
}

template <bool Inplace>
using ForeachResult = std::conditional_t<Inplace, void, std::vector<Tensor>>;

template <ForeachOp Op, bool Inplace, typename Slow>
ForeachResult<Inplace> foreach_apply(ArrayRef<TensorList> lists,
                                     ArrayRef<Scalar> scalars,
                                     Slow slow,
                                     const Tensor* scalar_rhs = nullptr)
{
    check_foreach_api_restrictions(lists[0]);
    for (const auto i : c10::irange(size_t(1), lists.size()))
        check_foreach_api_restrictions(lists[0], lists[i]);
    if (!can_batch_foreach(lists, scalars, Inplace, scalar_rhs))
        return slow();
    if constexpr (Op == ForeachOp::Add || Op == ForeachOp::Sub)
        if (lists.size() > 1 || scalar_rhs)
            alpha_check(lists[0][0].scalar_type(), scalars[0]);

    std::vector<Tensor> result;
    if constexpr (!Inplace)
    {
        result.reserve(lists[0].size());
        for (const auto& tensor : lists[0])
            result.emplace_back(at::empty_like(tensor));
    }
    const TensorList output = Inplace ? lists[0] : TensorList(result);
    AT_DISPATCH_V2(lists[0][0].scalar_type(), "dgx_foreach", AT_WRAP([&] {
                       using opmath_t = at::opmath_type<scalar_t>;
                       parallel_for_each([&](int device, cudaStream_t stream) {
                           ForeachBatch<scalar_t> batch{};
                           auto launch = [&]() {
                               if (batch.blocks)
                               {
                                   const auto blocks = static_cast<unsigned int>(std::min<int64_t>(batch.blocks, 65535));
                                   foreach_kernel<Op><<<blocks, kNumThreads, 0, stream>>>(batch, scalar_rhs != nullptr);
                                   C10_CUDA_KERNEL_LAUNCH_CHECK();
                               }
                               batch.count = 0;
                               batch.blocks = 0;
                           };
                           for (const auto i : c10::irange(lists[0].size()))
                           {
                               // Match the logical ownership used by other DGX pointwise kernels.
                               const auto [offset, size] = chunk(lists[0][i].numel(), device);
                               if (!size)
                                   continue;
                               batch.blocks += ::cuda::ceil_div(static_cast<int64_t>(size), kForeachBlockSize);
                               batch.tensors[batch.count++] = {
                                   lists[0][i].const_data_ptr<scalar_t>(),
                                   scalar_rhs ? scalar_rhs->const_data_ptr<scalar_t>() :
                                                (lists.size() > 1 ? lists[1][i].const_data_ptr<scalar_t>() : nullptr),
                                   lists.size() > 2 ? lists[2][i].const_data_ptr<scalar_t>() : nullptr,
                                   output[i].data_ptr<scalar_t>(),
                                   static_cast<int64_t>(offset),
                                   static_cast<int64_t>(size),
                                   batch.blocks,
                                   scalars.empty() ? opmath_t(1) : scalars[scalars.size() == 1 ? 0 : i].to<opmath_t>()
                               };
                               if (batch.count == kForeachBatchSize)
                                   launch();
                           }
                           launch();
                       });
                   }),
                   AT_EXPAND(AT_FLOATING_TYPES), kHalf, kBFloat16);

    if constexpr (Inplace)
    {
        // Foreach's generated autograd wrappers leave version updates to kernels.
        for (const auto& tensor : lists[0])
            tensor.unsafeGetTensorImpl()->bump_version();
    }
    else
        return result;
}

#define DGX_FOREACH_SLOW(NAME, ...)                                                                                    \
    [&]() -> ForeachResult<Inplace> {                                                                                  \
        if constexpr (Inplace)                                                                                         \
            return at::native::NAME##_(__VA_ARGS__);                                                                   \
        else                                                                                                           \
            return at::native::NAME(__VA_ARGS__);                                                                      \
    }

#define DGX_FOREACH_BINARY(NAME, OP, NATIVE)                                                                           \
    template <bool Inplace>                                                                                            \
    ForeachResult<Inplace> NAME##_scalar(TensorList input, const Scalar& scalar)                                       \
    {                                                                                                                  \
        if constexpr (ForeachOp::OP == ForeachOp::Sub)                                                                 \
            for (const auto& t : input)                                                                                \
                sub_check(t, scalar);                                                                                  \
        return foreach_apply<ForeachOp::OP, Inplace>(                                                                  \
            { input }, { scalar }, DGX_FOREACH_SLOW(foreach_tensor_##NATIVE##_scalar_kernel_slow, input, scalar));     \
    }                                                                                                                  \
    template <bool Inplace>                                                                                            \
    ForeachResult<Inplace> NAME##_scalarlist(TensorList input, ArrayRef<Scalar> scalars)                               \
    {                                                                                                                  \
        check_foreach_api_restrictions(input, scalars);                                                                \
        if constexpr (ForeachOp::OP == ForeachOp::Sub)                                                                 \
            for (const auto i : c10::irange(input.size()))                                                             \
                sub_check(input[i], scalars[i]);                                                                       \
        return foreach_apply<ForeachOp::OP, Inplace>(                                                                  \
            { input }, scalars, DGX_FOREACH_SLOW(foreach_tensor_##NATIVE##_scalarlist_kernel_slow, input, scalars));   \
    }

DGX_FOREACH_BINARY(add, Add, add)
DGX_FOREACH_BINARY(sub, Sub, sub)
DGX_FOREACH_BINARY(mul, Mul, mul)
DGX_FOREACH_BINARY(div, Div, div)
DGX_FOREACH_BINARY(maximum, Maximum, clamp_min)

#define DGX_FOREACH_LIST_ALPHA(NAME, OP)                                                                               \
    template <bool Inplace>                                                                                            \
    ForeachResult<Inplace> NAME##_list(TensorList input, TensorList other, const Scalar& alpha)                        \
    {                                                                                                                  \
        check_foreach_api_restrictions(input, other);                                                                  \
        return foreach_apply<ForeachOp::OP, Inplace>(                                                                  \
            { input, other }, { alpha },                                                                               \
            DGX_FOREACH_SLOW(foreach_tensor_##NAME##_list_kernel_slow, input, other, alpha));                          \
    }
DGX_FOREACH_LIST_ALPHA(add, Add)
DGX_FOREACH_LIST_ALPHA(sub, Sub)

#define DGX_FOREACH_LIST(NAME, OP, NATIVE)                                                                             \
    template <bool Inplace>                                                                                            \
    ForeachResult<Inplace> NAME##_list(TensorList input, TensorList other)                                             \
    {                                                                                                                  \
        return foreach_apply<ForeachOp::OP, Inplace>(                                                                  \
            { input, other }, {}, DGX_FOREACH_SLOW(foreach_tensor_##NATIVE##_list_kernel_slow, input, other));         \
    }
DGX_FOREACH_LIST(mul, Mul, mul)
DGX_FOREACH_LIST(div, Div, div)
DGX_FOREACH_LIST(maximum, Maximum, clamp_min)

#define DGX_FOREACH_UNARY(NAME, OP)                                                                                    \
    template <bool Inplace>                                                                                            \
    ForeachResult<Inplace> NAME(TensorList input)                                                                      \
    {                                                                                                                  \
        return foreach_apply<ForeachOp::OP, Inplace>(                                                                  \
            { input }, {}, DGX_FOREACH_SLOW(foreach_tensor_##NAME##_slow, input));                                     \
    }
DGX_FOREACH_UNARY(sqrt, Sqrt)
DGX_FOREACH_UNARY(reciprocal, Reciprocal)
DGX_FOREACH_UNARY(neg, Neg)

#define DGX_FOREACH_POINTWISE(NAME, OP)                                                                                \
    template <bool Inplace>                                                                                            \
    ForeachResult<Inplace> NAME##_scalar(TensorList input, TensorList a, TensorList b, const Scalar& value)            \
    {                                                                                                                  \
        return foreach_apply<ForeachOp::OP, Inplace>(                                                                  \
            { input, a, b }, { value }, DGX_FOREACH_SLOW(foreach_tensor_##NAME##_scalar_slow, input, a, b, value));    \
    }                                                                                                                  \
    template <bool Inplace>                                                                                            \
    ForeachResult<Inplace> NAME##_scalarlist(TensorList input, TensorList a, TensorList b, ArrayRef<Scalar> values)    \
    {                                                                                                                  \
        check_foreach_api_restrictions(input, values);                                                                 \
        return foreach_apply<ForeachOp::OP, Inplace>(                                                                  \
            { input, a, b }, values, DGX_FOREACH_SLOW(foreach_tensor_##NAME##_scalarlist_slow, input, a, b, values));  \
    }                                                                                                                  \
    template <bool Inplace>                                                                                            \
    ForeachResult<Inplace> NAME##_tensor(TensorList input, TensorList a, TensorList b, const Tensor& values)           \
    {                                                                                                                  \
        return NAME##_scalarlist<Inplace>(input, a, b, convert_tensor_to_scalar_list(values, input.size()));           \
    }
DGX_FOREACH_POINTWISE(addcmul, Addcmul)
DGX_FOREACH_POINTWISE(addcdiv, Addcdiv)

template <bool Inplace>
ForeachResult<Inplace> lerp_scalar(TensorList input, TensorList other, const Scalar& weight)
{
    return foreach_apply<ForeachOp::Lerp, Inplace>(
        { input, other }, { weight }, DGX_FOREACH_SLOW(foreach_tensor_lerp_list_kernel_slow, input, other, weight));
}

template <bool Inplace>
ForeachResult<Inplace> lerp_scalarlist(TensorList input, TensorList other, ArrayRef<Scalar> weights)
{
    check_foreach_api_restrictions(input, weights);
    return foreach_apply<ForeachOp::Lerp, Inplace>(
        { input, other }, weights, DGX_FOREACH_SLOW(foreach_tensor_lerp_scalarlist_kernel_slow, input, other, weights));
}

template <bool Inplace>
ForeachResult<Inplace> lerp_list(TensorList input, TensorList other, TensorList weights)
{
    return foreach_apply<ForeachOp::Lerp, Inplace>(
        { input, other, weights }, {}, DGX_FOREACH_SLOW(foreach_tensor_ternary_lerp_slow, input, other, weights));
}

template <bool Inplace>
ForeachResult<Inplace> add_tensor(TensorList input, const Tensor& scalar, const Scalar& alpha)
{
    check_foreach_api_restrictions(input);
    return foreach_apply<ForeachOp::Add, Inplace>(
        { input }, { alpha }, DGX_FOREACH_SLOW(foreach_tensor_add_tensor_kernel_slow, input, scalar, alpha), &scalar);
}

#define DGX_FOREACH_TENSOR(NAME, OP)                                                                                   \
    template <bool Inplace>                                                                                            \
    ForeachResult<Inplace> NAME##_tensor(TensorList input, const Tensor& scalar)                                       \
    {                                                                                                                  \
        if (scalar.device().is_cpu())                                                                                  \
            return NAME##_scalar<Inplace>(input, scalar.item());                                                       \
        return foreach_apply<ForeachOp::OP, Inplace>(                                                                  \
            { input }, {}, DGX_FOREACH_SLOW(foreach_tensor_##NAME##_tensor_kernel_slow, input, scalar), &scalar);      \
    }
DGX_FOREACH_TENSOR(mul, Mul)
DGX_FOREACH_TENSOR(div, Div)

#undef DGX_FOREACH_TENSOR
#undef DGX_FOREACH_POINTWISE
#undef DGX_FOREACH_UNARY
#undef DGX_FOREACH_LIST
#undef DGX_FOREACH_LIST_ALPHA
#undef DGX_FOREACH_BINARY
#undef DGX_FOREACH_SLOW

} // namespace

TORCH_LIBRARY_IMPL(aten, PrivateUse1, m)
{
#define DGX_REGISTER_FOREACH(NAME, OVERLOAD, FUNCTION)                                                                 \
    m.impl(TORCH_SELECTIVE_NAME("aten::_foreach_" #NAME OVERLOAD), TORCH_FN(FUNCTION<false>));                         \
    m.impl(TORCH_SELECTIVE_NAME("aten::_foreach_" #NAME "_" OVERLOAD), TORCH_FN(FUNCTION<true>));
#define DGX_REGISTER_BINARY(NAME)                                                                                      \
    DGX_REGISTER_FOREACH(NAME, ".Scalar", NAME##_scalar)                                                               \
    DGX_REGISTER_FOREACH(NAME, ".ScalarList", NAME##_scalarlist)                                                       \
    DGX_REGISTER_FOREACH(NAME, ".List", NAME##_list)
    DGX_REGISTER_BINARY(add)
    DGX_REGISTER_BINARY(sub)
    DGX_REGISTER_BINARY(mul)
    DGX_REGISTER_BINARY(div)
    DGX_REGISTER_BINARY(maximum)
    DGX_REGISTER_BINARY(lerp)
    DGX_REGISTER_FOREACH(add, ".Tensor", add_tensor)
    DGX_REGISTER_FOREACH(mul, ".Tensor", mul_tensor)
    DGX_REGISTER_FOREACH(div, ".Tensor", div_tensor)
    DGX_REGISTER_FOREACH(addcmul, ".Scalar", addcmul_scalar)
    DGX_REGISTER_FOREACH(addcmul, ".ScalarList", addcmul_scalarlist)
    DGX_REGISTER_FOREACH(addcmul, ".Tensor", addcmul_tensor)
    DGX_REGISTER_FOREACH(addcdiv, ".Scalar", addcdiv_scalar)
    DGX_REGISTER_FOREACH(addcdiv, ".ScalarList", addcdiv_scalarlist)
    DGX_REGISTER_FOREACH(addcdiv, ".Tensor", addcdiv_tensor)
    DGX_REGISTER_FOREACH(sqrt, "", sqrt)
    DGX_REGISTER_FOREACH(reciprocal, "", reciprocal)
    DGX_REGISTER_FOREACH(neg, "", neg)
#undef DGX_REGISTER_BINARY
#undef DGX_REGISTER_FOREACH
}

} // namespace at::native::dgx
