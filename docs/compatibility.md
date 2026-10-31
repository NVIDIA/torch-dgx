# Compatibility

torch-dgx is released as source and builds against the PyTorch headers and
libraries in the build environment. Rebuild the extension after changing
PyTorch or the CUDA toolchain. The minimum versions in
[pyproject.toml](../pyproject.toml) describe installation requirements; release
notes should identify the configurations actually validated for that release.

## PyTorch implementation dependencies

The backend uses PyTorch implementation interfaces for device registration,
storage management, generators, and attention. These dependencies require review
when supporting a new PyTorch version, even when the public Python API is
unchanged.

| Area | Source | Version-sensitive assumptions |
| --- | --- | --- |
| Fused attention | [Attention.cu](../source/Attention.cu) | Uses ATen's private memory-efficient attention kernel headers, parameter layouts, and kernel specializations. Forward and backward must agree on dropout indexing. |
| Backend integration | [DGXBackend.cpp](../source/DGXBackend.cpp), [DGXDeviceGuard.h](../source/DGXDeviceGuard.h), [Python registration](../torch_dgx/__init__.py) | Implements PrivateUse1 hooks and device guards, registers the device module, and uses PyTorch's CPU and autograd fallback interfaces. Review interface changes and inherited defaults. |
| Tensor storage and views | [EmptyTensor.cpp](../source/EmptyTensor.cpp), [Resize.cpp](../source/Resize.cpp) | Uses ATen allocation, native view, and storage-sizing helpers, along with TensorImpl and StorageImpl mutation. Preserve sizes, strides, offsets, and storage ownership. |
| Random generators | [DGXGenerator.h](../source/DGXGenerator.h), [DGXGenerator.cpp](../source/DGXGenerator.cpp) | Implements c10's generator interface and supplies state used by PyTorch RNG and activation-checkpoint utilities. Preserve locking and state semantics when updating the interface. |

The attention code currently selects a maximum dropout RNG extent of
`2**31 - 1` for PyTorch 2.10–2.13 and `2**63 - 1` for 2.14 and newer.
The extent is `batch * heads * query_length * key_length`. This guard accounts
for the index widths used by the included kernels; configurations exceeding the
limit use the math decomposition. The version branches describe the current
implementation, not evidence that every version has passed release validation.

Keep version checks beside the affected code and explain the upstream assumption
they protect. Add a small compatibility function when supported versions need
different signatures or when the same adjustment appears at multiple call sites.
[CompatibilityUtils.h](../source/CompatibilityUtils.h) currently contains CUDA
API adaptations. Keep those separate from unrelated PyTorch changes.

Before replacing a native helper with a public dispatched operator, check its
dispatch path. Calling the same operator from its DGX implementation can dispatch
back into that implementation. Preserve raw CUDA calls and their error checks
when a PyTorch compatibility change does not require changing them.

When changing the supported PyTorch versions, compile against the oldest and
newest supported configurations and exercise the affected behavior. Attention
changes need forward, backward, dropout, mask, and fallback coverage; storage
changes need view, resize, and copy coverage; generator changes need RNG replay
and activation-checkpoint coverage. Record the actual versions and results in
the release validation record.

## Saved-state compatibility

### RNG representation

`torch.dgx.get_rng_state()` returns a CPU `torch.uint8` tensor of shape `(16,)`.
The default DGX generator and explicitly constructed `torch.Generator(device="dgx")`
instances use the same representation:

| Byte offsets | Type | Meaning |
| --- | --- | --- |
| 0–7 | `uint64_t` | Seed |
| 8–15 | `uint64_t` | Philox offset; must be a multiple of four |

The implementation copies a C++ structure directly into these bytes. Its byte
order is native to the host, which is little-endian on x86-64. There is no version
marker or byte-order conversion; this format does not provide cross-endian
portability. The C++ implementation checks that the structure occupies exactly
16 bytes.

`torch.dgx.set_rng_state()` requires a strided CPU byte tensor containing exactly
16 elements and accepts noncontiguous input by making a contiguous copy. The
generator checks the offset alignment before replacing its state. DGX exposes
one default generator for logical device `dgx:0`; it is independent of the CPU
and CUDA generators. Save explicitly created generators separately. A seed alone
does not record how far a generator has advanced.

Treat this representation as the legacy format when changing saved state.
Preserve its field meanings. A future format change must retain a reader for the
legacy representation or provide an explicit migration procedure; old bytes
must never silently acquire a new interpretation. If a new representation needs
a version marker, keep legacy detection unambiguous and update the Python and
C++ size checks together. Existing readers reject states with a different size.

Being able to restore state and reproducing the same sequence are separate
compatibility properties. Changes to RNG algorithms, offset consumption,
operator selection, or dependencies can affect subsequent draws even when a
state remains readable. Exact replay across torch-dgx, PyTorch, or CUDA versions
is not currently guaranteed. Record those versions and the execution
configuration when reproducibility matters.

### Tensor and model checkpoints

Use PyTorch state dictionaries for model and optimizer state, and save RNG states
explicitly when resuming a random sequence is required. For checkpoints containing
DGX tensor storages, import `torch_dgx` before restoring to `dgx:0`. Use
`torch.load(..., map_location="cpu")` to restore tensor data to CPU instead; this
avoids requiring a usable DGX device for the tensor storages. Keeping checkpoint
tensors on CPU also avoids tying their saved device locations to DGX. This device
mapping does not migrate an RNG byte format or guarantee identical computation
after restoration.

The current [RNG tests](../tests/test_random.py) cover state validation and replay
within one build. The [activation-checkpoint tests](../tests/test_checkpoint.py)
cover RNG preservation during recomputation. Neither establishes that a new
release can load checkpoint files written by an older release.

For release compatibility validation, retain a small checkpoint and a nonzero
RNG-state fixture produced by an identified released version. Check tensor values,
shapes, dtypes, CPU mapping, and restoration to DGX. For RNG state, check the
restored seed and offset, and the subsequent draws for any configurations where
exact replay is promised. Keep these fixtures when updating the implementation;
regenerating them with the new version only establishes a same-version round trip.
