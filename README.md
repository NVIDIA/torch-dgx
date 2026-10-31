# torch-dgx: A Scalable PyTorch Abstraction for Multi-GPU Deep Learning

torch-dgx exposes multiple NVIDIA GPUs as an aggregate PyTorch device, `"dgx"`,
combining their memory and compute to support workloads that exceed the capacity
of one GPU. Tensors on `"dgx"` reside in CUDA unified memory, and supported
operators distribute their work across GPUs. NVLink is optional and improves
communication bandwidth between devices.

Because distribution happens at the operator level, model code does not need
explicit cross-device communication, synchronization, or tensor sharding.
After importing `torch_dgx`, adapting an existing model can be as simple as
changing the device for its parameters and inputs from `"cuda"` or `"cpu"` to
`"dgx"`.

### Project Status

torch-dgx is under active development. Operator coverage and compatibility are
still evolving; review the [supported layers and operators](docs/pytorch_support.md)
for the workloads you plan to run. For public releases, the project uses
[GitHub Releases](https://github.com/NVIDIA/torch-dgx/releases) as the canonical
location for release notes and history.

See the [compatibility notes](docs/compatibility.md) for PyTorch implementation
dependencies, the DGX RNG state format, and checkpoint compatibility limits.

### How It Works

- **Shared tensor storage.** CUDA unified memory gives all GPUs access to tensor
  data through a single virtual address space. The runtime manages where data
  resides and migrates pages on demand. Selected operators prefetch data when
  profiling shows a performance benefit.
- **Asynchronous memory management.** A stream-ordered unified memory pool uses
  `cudaMallocFromPoolAsync` and `cudaFreeAsync` to allocate, reuse, and free tensor
  storage. CUDA events ensure that each allocation is ready before use and that
  all GPUs finish accessing it before it is freed, without synchronizing the host.
- **Distributed operators.** Operators partition work across GPUs and launch
  concurrent CUDA kernels. For example, matrix multiplication assigns each GPU a
  subset of output rows. Partitioning can vary between operators and depend on
  their inputs. Kernels access data on other GPUs through unified memory, and
  cross-device synchronization preserves dependencies between operators.
- **PyTorch integration.** Tensors on `"dgx"` automatically dispatch to the
  backend's operator implementations. Automatic differentiation uses specialized
  multi-GPU backward kernels or decompositions through PyTorch autograd.

Operator implementations build on CUDA-X libraries: CUB for sorting and
reductions, cuDNN for convolution, cuBLAS for matrix multiplication, and CUTLASS
for fused scaled dot-product attention (SDPA). SDPA configurations outside the
fused kernels' supported cases use PyTorch's math decomposition on DGX.
Fused FP32 attention supports `dropout_p` from 0 to 1, including with causal
attention or supported masks. Dropout uses the DGX RNG state and regenerates
the same mask during backward without storing the full attention matrix.
When dropout consumes RNG, `batch * heads * query_length * key_length` must
not exceed `2**31 - 1` with PyTorch 2.10–2.13, or `2**63 - 1` with PyTorch
2.14 and newer, which support 64-bit RNG indexing throughout both passes.
Larger configurations use the math decomposition.

The allocator and distributed kernels let the host queue allocations, operator
execution, and frees asynchronously. Reading GPU results on the CPU, such as
through `loss.item()`, forces synchronization. Missing operator implementations
trigger a warning and PyTorch's CPU fallback, which adds synchronization and
data-transfer overhead.

### Quick Start

Prerequisites: Python 3.10 or newer, CUDA 13, cuDNN, and a CUDA-enabled build of
PyTorch 2.10 or newer. With the build dependencies listed in
[pyproject.toml](pyproject.toml) installed, build the extension and run the tests
from the repository root:

```bash
pip install --no-build-isolation -e .
pytest -v -s tests/
```

For environment setup and dependency installation, see the
[development setup](CONTRIBUTING.md#development-setup).

Import `torch_dgx` to register the backend, then place the model and its inputs on
`"dgx"`. This example, adapted from the paper, trains a small network to fit
`y = x**2` using a standard PyTorch training loop:

```python
import torch
import torch_dgx

device = "dgx"
model = torch.nn.Sequential(
    torch.nn.Linear(1, 64),
    torch.nn.ReLU(),
    torch.nn.Linear(64, 1),
).to(device)

x_train = torch.empty(100, 1, device=device).uniform_(-3, 3)
y_train = x_train**2
loss_fn = torch.nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.1, fused=True)

for step in range(10):
    optimizer.zero_grad()
    predictions = model(x_train)
    loss = loss_fn(predictions, y_train)
    loss.backward()
    optimizer.step()
    print(f"Step {step}: loss = {loss.item()}")
```

### Supported Layers and Operators

See the [supported layers and operators](docs/pytorch_support.md).

### Support

Use [GitHub Issues](https://github.com/NVIDIA/torch-dgx/issues/new/choose) to
report bugs, request features or documentation, or ask questions about installing
and using torch-dgx. Choose the form that matches your request and search existing
issues first. Support is provided by maintainers and community contributors as
time permits; response times and fixes are not guaranteed.

Report potential security vulnerabilities privately using
[SECURITY.md](SECURITY.md). Public participation follows the
[Code of Conduct](CODE_OF_CONDUCT.md).

### Contributing

Contributions are open. We welcome bug fixes, operator implementations,
performance improvements, tests, documentation, and examples within the
project's multi-GPU PyTorch scope. Discuss substantial changes in an issue before
starting implementation. Pull requests require the applicable NVIDIA Contributor
License Agreement and maintainer review. See [CONTRIBUTING.md](CONTRIBUTING.md)
for the accepted scope, development commands, submission and review process, and
maintainer responsibilities.

### License

This project is licensed under the BSD 3-Clause License. See [LICENSE](LICENSE).

Third-party dependency licenses are provided in the `licenses/` directory:

- CCCL: Thrust is licensed under the Apache License 2.0 with noted exceptions,
  libcu++ is licensed under the Apache License 2.0 with LLVM Exceptions, and
  CUB is licensed under the BSD 3-Clause License. See
  [licenses/cccl-LICENSE.txt](licenses/cccl-LICENSE.txt).
- CUTLASS is licensed under the BSD 3-Clause License, with CuTeDSL files under
  the NVIDIA End User License Agreement (EULA). See
  [licenses/cutlass-LICENSE.txt](licenses/cutlass-LICENSE.txt).
- PyTorch is licensed under the BSD 3-Clause License. See
  [licenses/pytorch-LICENSE.txt](licenses/pytorch-LICENSE.txt).
- pybind11 is licensed under the BSD 3-Clause License. See
  [licenses/pybind11-LICENSE.txt](licenses/pybind11-LICENSE.txt).
