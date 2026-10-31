/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
#include "KernelLaunchUtils.cuh"
#include "StridedTensorUtils.cuh"

#include <c10/util/complex.h>

#include <ATen/Dispatch_v2.h>
#include <ATen/MemoryOverlap.h>
#include <ATen/core/Tensor.h>
#include <ATen/ops/empty_like.h>

#include <torch/library.h>

namespace at::native
{

namespace dgx
{

namespace
{

constexpr int kCopyTileSize = 32;
constexpr int kCopyBlockRows = 8;
constexpr int kInterleavedTilePixels = 64;
constexpr int kImageTileHeight = 16;

unsigned int capped_num_blocks(int64_t required_blocks)
{
    return static_cast<unsigned int>(std::min<int64_t>(required_blocks, std::numeric_limits<int>::max()));
}

template <typename DstT, typename Func>
void for_each_destination_chunk(const Tensor& dst, Func func)
{
    parallel_for_each([&](int device, cudaStream_t stream) {
        int64_t dst_begin, device_numel;
        std::tie(dst_begin, device_numel) = chunk(dst.numel(), device);
        if (device_numel)
            func(stream, dst_begin, dst_begin + device_numel);
    });
}

bool is_batched_column_major(const Tensor& tensor)
{
    if (tensor.dim() < 2 || !tensor.is_non_overlapping_and_dense() || !tensor.numel())
        return false;

    int64_t expected_stride = 1;
    const auto row_dim = tensor.dim() - 2;
    const auto column_dim = tensor.dim() - 1;
    if (tensor.size(row_dim) != 1 && tensor.stride(row_dim) != expected_stride)
        return false;
    expected_stride *= tensor.size(row_dim);

    if (tensor.size(column_dim) != 1 && tensor.stride(column_dim) != expected_stride)
        return false;
    expected_stride *= tensor.size(column_dim);

    for (int64_t dim = tensor.dim() - 3; dim >= 0; --dim)
    {
        if (tensor.size(dim) != 1 && tensor.stride(dim) != expected_stride)
            return false;
        expected_stride *= tensor.size(dim);
    }
    return true;
}

bool is_canonical_channels_last(const Tensor& tensor)
{
    return tensor.dim() == 4 && tensor.is_non_overlapping_and_dense() &&
           tensor.strides() == c10::get_channels_last_strides_2d(tensor.sizes());
}

template <class SrcT, class DstT>
struct CopyCastFunctor
{
    bool conjugate;

    __device__ void operator()(size_t tid, const SrcT* self, DstT* dst)
    {
        if constexpr (c10::is_complex<SrcT>::value)
        {
            const auto value = self[tid];
            dst[tid] = conjugate ? DstT(value.real(), -value.imag()) : value;
        }
        else
        {
            dst[tid] = static_cast<DstT>(self[tid]);
        }
    }
};

template <typename SrcT, typename DstT, bool destination_row_major>
__global__ __launch_bounds__(kNumThreads) void tiled_copy_kernel(const SrcT* __restrict__ self,
                                                                 DstT* __restrict__ dst,
                                                                 int64_t batch,
                                                                 int64_t rows,
                                                                 int64_t columns,
                                                                 int64_t dst_begin,
                                                                 int64_t dst_end,
                                                                 int64_t first_major_tile,
                                                                 int64_t num_major_tiles)
{
    static_assert(kCopyTileSize * kCopyBlockRows == kNumThreads);
    __shared__ DstT tile[kCopyTileSize][kCopyTileSize + 1];

    const int64_t minor_size = destination_row_major ? columns : rows;
    const int64_t num_minor_tiles = ::cuda::ceil_div(minor_size, int64_t{ kCopyTileSize });
    const int64_t num_tiles = num_major_tiles * num_minor_tiles;
    for (int64_t tile_index = blockIdx.x; tile_index < num_tiles; tile_index += gridDim.x)
    {
        const int64_t major_tile = first_major_tile + tile_index / num_minor_tiles;
        const int64_t minor_tile = tile_index % num_minor_tiles;
        const int64_t major_base = major_tile * kCopyTileSize;
        const int64_t minor_base = minor_tile * kCopyTileSize;

        if constexpr (destination_row_major)
        {
            const int64_t global_row = major_base + threadIdx.x;
            for (int offset = threadIdx.y; offset < kCopyTileSize; offset += kCopyBlockRows)
            {
                const int64_t column = minor_base + offset;
                if (global_row < batch * rows && column < columns)
                {
                    const int64_t batch_index = global_row / rows;
                    const int64_t row = global_row % rows;
                    const int64_t input_offset = batch_index * rows * columns + column * rows + row;
                    tile[threadIdx.x][offset] = static_cast<DstT>(self[input_offset]);
                }
            }
            __syncthreads();

            for (int offset = threadIdx.y; offset < kCopyTileSize; offset += kCopyBlockRows)
            {
                const int64_t output_row = major_base + offset;
                const int64_t column = minor_base + threadIdx.x;
                const int64_t output_offset = output_row * columns + column;
                if (output_row < batch * rows && column < columns && output_offset >= dst_begin && output_offset < dst_end)
                    dst[output_offset] = tile[offset][threadIdx.x];
            }
        }
        else
        {
            const int64_t global_column = major_base + threadIdx.x;
            for (int offset = threadIdx.y; offset < kCopyTileSize; offset += kCopyBlockRows)
            {
                const int64_t row = minor_base + offset;
                if (global_column < batch * columns && row < rows)
                {
                    const int64_t batch_index = global_column / columns;
                    const int64_t column = global_column % columns;
                    const int64_t input_offset = batch_index * rows * columns + row * columns + column;
                    tile[offset][threadIdx.x] = static_cast<DstT>(self[input_offset]);
                }
            }
            __syncthreads();

            for (int offset = threadIdx.y; offset < kCopyTileSize; offset += kCopyBlockRows)
            {
                const int64_t output_column = major_base + offset;
                const int64_t row = minor_base + threadIdx.x;
                const int64_t output_offset = output_column * rows + row;
                if (output_column < batch * columns && row < rows && output_offset >= dst_begin && output_offset < dst_end)
                    dst[output_offset] = tile[threadIdx.x][offset];
            }
        }
        __syncthreads();
    }
}

template <typename SrcT, typename DstT, int channels>
__global__ __launch_bounds__(channels* kInterleavedTilePixels) void interleaved_to_planar_copy_kernel(
    const SrcT* __restrict__ self,
    DstT* __restrict__ dst,
    int64_t height,
    int64_t width,
    int64_t source_batch_stride,
    int64_t source_row_stride,
    int64_t source_pixel_stride,
    int64_t first_row,
    int64_t num_rows)
{
    constexpr int64_t elements_per_tile = channels * kInterleavedTilePixels;
    __shared__ DstT tile[elements_per_tile];

    const int64_t tiles_per_row = ::cuda::ceil_div(width, int64_t{ kInterleavedTilePixels });
    const int64_t num_tiles = num_rows * tiles_per_row;
    for (int64_t tile_index = blockIdx.x; tile_index < num_tiles; tile_index += gridDim.x)
    {
        const int64_t global_row = first_row + tile_index / tiles_per_row;
        const int64_t batch = global_row / height;
        const int64_t row = global_row % height;
        const int64_t column_base = (tile_index % tiles_per_row) * kInterleavedTilePixels;

        const int input_channel = threadIdx.x % channels;
        const int input_pixel = threadIdx.x / channels;
        const int64_t input_column = column_base + input_pixel;
        if (input_column < width)
        {
            const int64_t input_offset = batch * source_batch_stride + row * source_row_stride +
                                         input_column * source_pixel_stride + input_channel;
            tile[threadIdx.x] = static_cast<DstT>(self[input_offset]);
        }
        __syncthreads();

        const int output_channel = threadIdx.x / kInterleavedTilePixels;
        const int output_pixel = threadIdx.x % kInterleavedTilePixels;
        const int64_t output_column = column_base + output_pixel;
        if (output_column < width)
        {
            const int64_t output_offset = ((batch * channels + output_channel) * height + row) * width + output_column;
            dst[output_offset] = tile[output_pixel * channels + output_channel];
        }
        __syncthreads();
    }
}

template <typename SrcT, typename DstT, int channels>
__global__ __launch_bounds__(channels* kInterleavedTilePixels) void planar_to_interleaved_copy_kernel(
    const SrcT* __restrict__ self,
    DstT* __restrict__ dst,
    int64_t height,
    int64_t width,
    int64_t source_batch_stride,
    int64_t source_channel_stride,
    int64_t source_row_stride,
    int64_t destination_batch_stride,
    int64_t destination_row_stride,
    int64_t destination_pixel_stride,
    int64_t first_row,
    int64_t num_rows)
{
    constexpr int64_t elements_per_tile = channels * kInterleavedTilePixels;
    __shared__ DstT tile[elements_per_tile];

    const int64_t tiles_per_row = ::cuda::ceil_div(width, int64_t{ kInterleavedTilePixels });
    const int64_t num_tiles = num_rows * tiles_per_row;
    for (int64_t tile_index = blockIdx.x; tile_index < num_tiles; tile_index += gridDim.x)
    {
        const int64_t global_row = first_row + tile_index / tiles_per_row;
        const int64_t batch = global_row / height;
        const int64_t row = global_row % height;
        const int64_t column_base = (tile_index % tiles_per_row) * kInterleavedTilePixels;

        const int input_channel = threadIdx.x / kInterleavedTilePixels;
        const int input_pixel = threadIdx.x % kInterleavedTilePixels;
        const int64_t input_column = column_base + input_pixel;
        if (input_column < width)
        {
            const int64_t input_offset = batch * source_batch_stride + input_channel * source_channel_stride +
                                         row * source_row_stride + input_column;
            tile[threadIdx.x] = static_cast<DstT>(self[input_offset]);
        }
        __syncthreads();

        const int output_channel = threadIdx.x % channels;
        const int output_pixel = threadIdx.x / channels;
        const int64_t output_column = column_base + output_pixel;
        if (output_column < width)
        {
            const int64_t output_offset = batch * destination_batch_stride + row * destination_row_stride +
                                          output_column * destination_pixel_stride + output_channel;
            dst[output_offset] = tile[output_channel * kInterleavedTilePixels + output_pixel];
        }
        __syncthreads();
    }
}

template <typename Func>
void for_each_image_row_chunk(int64_t batch, int64_t height, int64_t width, Func func)
{
    const int64_t tile_rows_per_image = ::cuda::ceil_div(height, int64_t{ kImageTileHeight });
    const int64_t num_tile_rows = batch * tile_rows_per_image;
    const int64_t tiles_per_row = ::cuda::ceil_div(width, int64_t{ kInterleavedTilePixels });
    parallel_for_each([&](int device, cudaStream_t stream) {
        // Match the image loss's ownership: split full-width 16-pixel tile rows across the batch,
        // with every channel of each assigned row on the same device.
        int64_t tile_row_offset, tile_row_count;
        std::tie(tile_row_offset, tile_row_count) = chunk(num_tile_rows, device);
        if (!tile_row_count)
            return;

        // Convert tile-row boundaries to unpadded pixel rows across the batch. This keeps partial
        // bottom tiles on their owner without launching blocks for padding between images.
        const auto pixel_row = [&](int64_t tile_row) {
            return (tile_row / tile_rows_per_image) * height + (tile_row % tile_rows_per_image) * kImageTileHeight;
        };
        const int64_t first_row = pixel_row(tile_row_offset);
        const int64_t row_end = pixel_row(tile_row_offset + tile_row_count);
        const int64_t num_rows = row_end - first_row;

        const int64_t required_blocks = num_rows * tiles_per_row;
        const unsigned int num_blocks = capped_num_blocks(required_blocks);

        func(stream, num_blocks, first_row, num_rows);
    });
}

template <typename SrcT, typename DstT, int channels>
void launch_interleaved_to_planar_copy(const Tensor& self, const Tensor& dst)
{
    static_assert(channels == 3 || channels == 4);
    const int64_t height = self.size(2);
    const int64_t width = self.size(3);
    for_each_image_row_chunk(
        self.size(0), height, width,
        [&](cudaStream_t stream, unsigned int num_blocks, int64_t first_row, int64_t num_rows) {
            interleaved_to_planar_copy_kernel<SrcT, DstT, channels>
                <<<num_blocks, channels * kInterleavedTilePixels, 0, stream>>>(
                    self.const_data_ptr<SrcT>(), dst.data_ptr<DstT>(), height, width, self.stride(0), self.stride(2),
                    self.stride(3), first_row, num_rows);
            C10_CUDA_KERNEL_LAUNCH_CHECK();
        });
}

template <typename SrcT, typename DstT, int channels>
void launch_planar_to_interleaved_copy(const Tensor& self, const Tensor& dst)
{
    static_assert(channels == 3 || channels == 4);
    // Both tensors have logical NHWC dimensions; the source stores separate channel planes.
    const int64_t height = self.size(1);
    const int64_t width = self.size(2);
    for_each_image_row_chunk(
        self.size(0), height, width,
        [&](cudaStream_t stream, unsigned int num_blocks, int64_t first_row, int64_t num_rows) {
            planar_to_interleaved_copy_kernel<SrcT, DstT, channels>
                <<<num_blocks, channels * kInterleavedTilePixels, 0, stream>>>(
                    self.const_data_ptr<SrcT>(), dst.data_ptr<DstT>(), height, width, self.stride(0), self.stride(3),
                    self.stride(1), dst.stride(0), dst.stride(1), dst.stride(2), first_row, num_rows);
            C10_CUDA_KERNEL_LAUNCH_CHECK();
        });
}

template <typename SrcT, typename DstT>
void launch_tiled_copy(
    const Tensor& self, const Tensor& dst, int64_t batch, int64_t rows, int64_t columns, bool source_row_major)
{
    if (!self.numel())
        return;

    const bool destination_row_major = !source_row_major;
    for_each_destination_chunk<DstT>(dst, [&](cudaStream_t stream, int64_t dst_begin, int64_t dst_end) {
        const int64_t major_size = destination_row_major ? columns : rows;
        const int64_t minor_size = destination_row_major ? columns : rows;
        const int64_t first_major = dst_begin / major_size;
        const int64_t last_major = ::cuda::ceil_div(dst_end, major_size);
        const int64_t first_major_tile = first_major / kCopyTileSize;
        const int64_t last_major_tile = ::cuda::ceil_div(last_major, int64_t{ kCopyTileSize });
        const int64_t num_major_tiles = last_major_tile - first_major_tile;
        const int64_t num_minor_tiles = ::cuda::ceil_div(minor_size, int64_t{ kCopyTileSize });
        const int64_t required_blocks = num_major_tiles * num_minor_tiles;
        const unsigned int num_blocks = capped_num_blocks(required_blocks);
        const dim3 num_threads(kCopyTileSize, kCopyBlockRows);

        if (destination_row_major)
        {
            tiled_copy_kernel<SrcT, DstT, true>
                <<<num_blocks, num_threads, 0, stream>>>(self.const_data_ptr<SrcT>(), dst.data_ptr<DstT>(), batch, rows,
                                                         columns, dst_begin, dst_end, first_major_tile, num_major_tiles);
        }
        else
        {
            tiled_copy_kernel<SrcT, DstT, false>
                <<<num_blocks, num_threads, 0, stream>>>(self.const_data_ptr<SrcT>(), dst.data_ptr<DstT>(), batch, rows,
                                                         columns, dst_begin, dst_end, first_major_tile, num_major_tiles);
        }
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    });
}

template <typename SrcT, typename DstT>
bool try_launch_optimized_copy(const Tensor& self, const Tensor& dst)
{
    if (self.dim() == 4 && self.numel() && dst.is_contiguous())
    {
        const int64_t channels = self.size(1);
        const int64_t height = self.size(2);
        const int64_t width = self.size(3);
        const int64_t batch_stride = self.stride(0);
        const int64_t row_stride = self.stride(2);
        const int64_t pixel_stride = self.stride(3);
        if ((channels == 3 || channels == 4) && self.stride(1) == 1 && pixel_stride >= channels &&
            row_stride >= width * pixel_stride && batch_stride >= height * row_stride)
        {
            if (channels == 3)
                launch_interleaved_to_planar_copy<SrcT, DstT, 3>(self, dst);
            else
                launch_interleaved_to_planar_copy<SrcT, DstT, 4>(self, dst);
            return true;
        }
    }

    if (self.dim() == 4 && self.numel())
    {
        const int64_t channels = self.size(3);
        const int64_t height = self.size(1);
        const int64_t width = self.size(2);
        // Match NCHW gradients viewed as NHWC after permute backward. Pitched destinations
        // also cover copying RGB gradients into the RGB slice of an RGBD raster output.
        if ((channels == 3 || channels == 4) && self.stride(2) == 1 && self.stride(1) >= width &&
            self.stride(3) >= height * self.stride(1) && self.stride(0) >= channels * self.stride(3) &&
            dst.stride(3) == 1 && dst.stride(2) >= channels && dst.stride(1) >= width * dst.stride(2) &&
            dst.stride(0) >= height * dst.stride(1))
        {
            if (channels == 3)
                launch_planar_to_interleaved_copy<SrcT, DstT, 3>(self, dst);
            else
                launch_planar_to_interleaved_copy<SrcT, DstT, 4>(self, dst);
            return true;
        }
    }

    if (self.dim() < 2 || self.dim() != dst.dim() || !self.is_non_overlapping_and_dense() ||
        !dst.is_non_overlapping_and_dense())
        return false;

    const bool self_contiguous = self.is_contiguous();
    const bool dst_contiguous = dst.is_contiguous();
    if (self.dim() == 4)
    {
        const bool self_channels_last = is_canonical_channels_last(self);
        const bool dst_channels_last = is_canonical_channels_last(dst);
        const int64_t spatial_size = self.size(2) * self.size(3);
        const bool to_channels_last = self_contiguous && dst_channels_last;
        const bool from_channels_last = self_channels_last && dst_contiguous;
        if (to_channels_last || from_channels_last)
        {
            launch_tiled_copy<SrcT, DstT>(self, dst, self.size(0), self.size(1), spatial_size, to_channels_last);
            return true;
        }
    }

    const bool self_column_major = is_batched_column_major(self);
    const bool dst_column_major = is_batched_column_major(dst);
    int64_t batch = 1;
    for (int64_t dim = 0; dim < self.dim() - 2; ++dim)
        batch *= self.size(dim);

    const bool to_column_major = self_contiguous && dst_column_major;
    const bool from_column_major = self_column_major && dst_contiguous;
    if (to_column_major || from_column_major)
    {
        launch_tiled_copy<SrcT, DstT>(self, dst, batch, self.size(-2), self.size(-1), to_column_major);
        return true;
    }
    return false;
}

template <typename scalar_t, typename out_t>
struct ScatterCastFunctor
{
    bool conjugate;

    __host__ __device__ out_t copy(scalar_t value) const
    {
        if constexpr (c10::is_complex<scalar_t>::value)
            return conjugate ? out_t(value.real(), -value.imag()) : value;
        else
            return static_cast<out_t>(value);
    }

    __host__ __device__ void operator()(size_t tid, const scalar_t* self, out_t* out, StridedOffsetCalculator<2> offsets)
    {
        const auto tensor_offsets = offsets.get(tid);
        out[tensor_offsets[1]] = copy(self[tensor_offsets[0]]);
    }
};

template <typename SrcT, typename DstT>
void dispatch_strided_cast_copy(const Tensor& self, const Tensor& dst, bool conjugate)
{
    const auto offsets =
        make_strided_offset_calculator<2>(self.sizes(), std::array<IntArrayRef, 2>{ self.strides(), dst.strides() });
    mgpu_launch_func(self.numel(), ScatterCastFunctor<SrcT, DstT>{ conjugate }, self.const_data_ptr<SrcT>(),
                     dst.data_ptr<DstT>(), offsets);
}

template <typename SrcT, typename DstT>
void dispatch_cast_copy_typed(const Tensor& self, const Tensor& dst, bool same_dense_layout, bool conjugate)
{
    if (same_dense_layout)
    {
        mgpu_launch_func(
            self.numel(), CopyCastFunctor<SrcT, DstT>{ conjugate }, self.const_data_ptr<SrcT>(), dst.data_ptr<DstT>());
    }
    else
    {
        if (conjugate || !try_launch_optimized_copy<SrcT, DstT>(self, dst))
            dispatch_strided_cast_copy<SrcT, DstT>(self, dst, conjugate);
    }
}

void dispatch_cast_copy(const Tensor& self, const Tensor& dst, bool same_dense_layout, bool conjugate)
{
    if (self.is_complex() && self.scalar_type() == dst.scalar_type())
    {
        AT_DISPATCH_V2(self.scalar_type(), "dispatch_cast_copy_complex", AT_WRAP([&] {
                           dispatch_cast_copy_typed<scalar_t, scalar_t>(self, dst, same_dense_layout, conjugate);
                       }),
                       AT_EXPAND(AT_COMPLEX_TYPES));
        return;
    }

    AT_DISPATCH_V2(self.scalar_type(), "dispatch_cast_copy_src", AT_WRAP([&] {
                       using src_t = scalar_t;
                       AT_DISPATCH_V2(
                           dst.scalar_type(), "dispatch_cast_copy_dst", AT_WRAP([&] {
                               using dst_t = scalar_t;
                               dispatch_cast_copy_typed<src_t, dst_t>(self, dst, same_dense_layout, conjugate);
                           }),
                           AT_EXPAND(AT_ALL_TYPES), kHalf, kBFloat16, kBool, AT_EXPAND(AT_BAREBONES_UNSIGNED_TYPES));
                   }),
                   AT_EXPAND(AT_ALL_TYPES), kHalf, kBFloat16, kBool, AT_EXPAND(AT_BAREBONES_UNSIGNED_TYPES));
}

bool has_same_dense_layout(const Tensor& self, const Tensor& dst)
{
    return self.strides() == dst.strides() && self.is_non_overlapping_and_dense() && dst.is_non_overlapping_and_dense();
}

// Copies tensors with the same dtype and dense layout as one flat range per device.
void dispatch_dense_copy(const Tensor& self, const Tensor& dst)
{
    parallel_for_each_async([&](int device, cudaStream_t stream) {
        int64_t device_offset, device_numel;
        std::tie(device_offset, device_numel) = chunk(self.numel(), device);

        AT_DISPATCH_V2(self.scalar_type(), "_copy_from", AT_WRAP([&] {
                           auto dst_ptr = dst.data_ptr<scalar_t>() + device_offset;
                           const auto src_ptr = self.const_data_ptr<scalar_t>() + device_offset;
                           C10_CUDA_CHECK(cudaMemcpyAsync(
                               dst_ptr, src_ptr, device_numel * sizeof(scalar_t), cudaMemcpyDefault, stream));
                       }),
                       AT_EXPAND(AT_ALL_TYPES_AND_COMPLEX), kHalf, kBFloat16, kBool,
                       AT_EXPAND(AT_BAREBONES_UNSIGNED_TYPES));
    });

    // In PyTorch, the copy operator is assumed to be blocking when the destination is the CPU. Thus, we need to
    // synchronize the host and devices in order to ensure that the host data is valid when this operator returns.
    if (dst.is_cpu())
    {
        for (const auto device_id : c10::irange(c10::cuda::device_count()))
        {
            c10::cuda::getCurrentCUDAStream(device_id).synchronize();
        }
    }
    else
    {
        merge_streams();
    }
}

// Host copies that need a cast, a conjugation, or a layout change must not do that work on the CPU. CPU writes
// leave managed pages on the host and CPU reads migrate them from the GPUs; either way the next GPU access pays
// the migration. Instead, transfer through flat copies, which leave pages on the GPUs, and cast on the GPUs.
// empty_like preserves dense layouts and makes other layouts dense, so its results support flat copies.
void host_to_device_cast_copy(const Tensor& self, const Tensor& dst)
{
    auto src = self.resolve_conj();
    if (!src.is_non_overlapping_and_dense())
        src = src.contiguous();
    if (src.scalar_type() == dst.scalar_type() && !dst.is_conj() && has_same_dense_layout(src, dst))
    {
        dispatch_dense_copy(src, dst);
        return;
    }

    const auto staging = at::empty_like(src, src.options().device(dst.device()));
    dispatch_dense_copy(src, staging);
    dispatch_cast_copy(staging, dst, has_same_dense_layout(staging, dst), dst.is_conj());
}

void device_to_host_cast_copy(const Tensor& self, const Tensor& dst)
{
    auto src = self;
    if (src.scalar_type() != dst.scalar_type() || src.is_conj() || !src.is_non_overlapping_and_dense())
    {
        src = at::empty_like(dst, dst.options().device(self.device()));
        dispatch_cast_copy(self, src, has_same_dense_layout(self, src), self.is_conj());
    }
    if (!dst.is_conj() && has_same_dense_layout(src, dst))
    {
        dispatch_dense_copy(src, dst);
        return;
    }

    // The CPU applies the remaining layout change and conjugation to host memory only.
    const auto staging = at::empty_like(src, src.options().device(dst.device()));
    dispatch_dense_copy(src, staging);
    dst.copy_(staging);
}

} // namespace

Tensor _copy_from_and_resize(const Tensor& self, const Tensor& dst)
{
    return dst.copy_(self, false);
}

Tensor _copy_from(const Tensor& self, const Tensor& dst, bool non_blocking)
{
    TORCH_CHECK(self.is_cpu() || self.device().type() == c10::DeviceType::PrivateUse1);
    TORCH_CHECK(dst.is_cpu() || dst.device().type() == c10::DeviceType::PrivateUse1);

    TORCH_CHECK(self.layout() == kStrided);
    TORCH_CHECK(dst.layout() == kStrided);
    TORCH_CHECK(self.sizes() == dst.sizes());

    at::assert_no_internal_overlap(dst);
    at::assert_no_partial_overlap(dst, self);

    const bool same_dense_layout = has_same_dense_layout(self, dst);
    const bool conjugate = self.is_conj() != dst.is_conj();

    if (self.scalar_type() == dst.scalar_type() && same_dense_layout && !conjugate)
    {
        dispatch_dense_copy(self, dst);
    }
    else if (self.device().is_privateuseone() && dst.device().is_privateuseone())
    {
        dispatch_cast_copy(self, dst, same_dense_layout, conjugate);
    }
    else if (self.device().is_privateuseone() && dst.device().is_cpu())
    {
        device_to_host_cast_copy(self, dst);
    }
    else if (self.device().is_cpu() && dst.device().is_privateuseone())
    {
        host_to_device_cast_copy(self, dst);
    }
    else
    {
        TORCH_CHECK(false, "Unsupported combination for _copy_from");
    }

    return dst;
}

TORCH_LIBRARY_IMPL(aten, PrivateUse1, m)
{
    m.impl(TORCH_SELECTIVE_NAME("aten::_copy_from"), TORCH_FN(_copy_from));
    m.impl(TORCH_SELECTIVE_NAME("aten::_copy_from_and_resize"), TORCH_FN(_copy_from_and_resize));
}

TORCH_LIBRARY_IMPL(aten, Conjugate, m)
{
    m.impl(TORCH_SELECTIVE_NAME("aten::_copy_from"), torch::CppFunction::makeFallthrough());
}

} // namespace dgx

} // namespace at::native
