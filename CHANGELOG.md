# Changelog

Notable user-facing changes to torch-dgx are recorded here.
The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## Unreleased

Changes since 1.0.0.

### Added

- DGX support for `floor_divide`, `atan2`, `minimum`, `nan_to_num`, and `flip`.
- Full-tensor `prod` for floating-point, complex, integer, and boolean inputs,
  including strided and empty tensors and an optional output dtype.
- `amax` over one, multiple, or all dimensions, and `cumprod`.
- Dimension-wise `median` for uint8, signed integer, FP16, FP32, and FP64 inputs,
  including strided inputs and outputs. Even-length slices return the lower
  median; NaNs propagate, and ties return the first matching input index.
- `topk` for uint8, signed integer, FP16, BF16, FP32, and FP64 inputs, with largest or
  smallest selection, optional sorting, arbitrary dimensions, and strided
  inputs and outputs. Native support requires CCCL 3.2 or newer, bundled with
  CUDA 13.2.
- 2D grid sampling forward and backward for bilinear, nearest, and bicubic
  interpolation, including padding modes, strided inputs, and FP16, BF16,
  FP32, and FP64 data.
- Average pooling forward through `avg_pool2d.out` for FP16, BF16, FP32, and FP64,
  including batched and unbatched inputs, strided layouts, padding, ceil mode,
  and divisor overrides.
- Batched 2x2 `linalg.solve` and `linalg.solve_ex` for FP32 and FP64,
  including left and right solves, strided outputs, and singularity reporting.
  Inputs require matching batch dimensions.
- `index_fill_` and tensor-source `scatter`.

### Changed

- **CUDA device selection:** DGX operations, tensor allocations, and tensor
  destruction may change the calling thread's current CUDA device. Callers must
  select the intended device before subsequent CUDA work that depends on it;
  `dgx:0` does not select a physical CUDA device. See
  [the compatibility notes](docs/compatibility.md#current-cuda-device).
- Extended `cumsum` to multidimensional tensors, negative dimensions, scalar and
  empty tensors, strided layouts, and additional floating-point and complex
  dtypes.
- Extended sorting to FP32 and FP64, arbitrary dimensions, descending order,
  and strided outputs. Stable sorting preserves original floating-point values
  and handles NaNs and signed-zero ties.
- Extended indexed reads and writes to boolean data, and integer-index
  `index_put_` to accumulation with repeated indices.
- Extended dimension-wise `max` reductions to boolean data.
- Extended `gather` to strided inputs, indices, and outputs.
- Extended `linalg.cross` to unbatched, batched, broadcast, and strided inputs
  along any valid dimension.
- Extended clamp operations to strided tensors.
- Preserved tensor layouts through addition and sigmoid forward and backward.
- Extended `mean` to arbitrary single dimensions.
- Increased GPU parallelism for sums along dimension zero of 2D and 3D tensors.
- Reduced temporary memory use for mixed-dtype strided addition, matrix
  multiplication, and linear backward.

### Fixed

- Corrected `mean` for empty inputs, strided singleton reductions, and output
  dtype conversion.
- Blocking host-to-DGX uploads now complete before returning.
- Prevented premature reuse of pinned host memory during nonblocking uploads.
  Uploads from untracked host storage synchronize before returning.
