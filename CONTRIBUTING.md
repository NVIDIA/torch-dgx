# Contributing

Thank you for your interest in contributing to `torch-dgx`.

## Contribution Policy

Contributions are open for bug fixes, operator coverage, performance improvements,
tests, documentation, and examples for the multi-GPU PyTorch backend. Discuss
substantial changes in an issue before implementation. See the
[support policy](README.md#support) and [Code of Conduct](CODE_OF_CONDUCT.md);
report potential vulnerabilities privately through [SECURITY.md](SECURITY.md).

## Contributor License Agreement

Before a contribution can be accepted, every contributor must complete the
applicable NVIDIA Contributor License Agreement (CLA): the [Individual
Contributor License Agreement (ICLA)](./ICLA.pdf) for individual contributors
or the [Corporate Contributor License Agreement (CCLA)](./CCLA.pdf) for
corporate contributors.

Email the completed and signed ICLA or CCLA to
[torch-dgx-cla@nvidia.com](mailto:torch-dgx-cla@nvidia.com).

Approved contributors will be added to [CONTRIBUTORS.md](./CONTRIBUTORS.md).

## Development Setup

Use a Linux system meeting the [README prerequisites](README.md#quick-start),
with CUDA development files, a compatible C++ compiler, visible GPUs, and
an activated standard Python virtual environment. From the repository root:

```bash
python -m pip install "torch>=2.10.0" --index-url https://download.pytorch.org/whl/cu132
python -m pip install "setuptools>=77" "packaging>=24.2" nvidia-cutlass numpy pytest ruff
python -m pip install --no-build-isolation -e .
pytest -v -s tests/
```

PyTorch's NVIDIA cuDNN package supplies both the headers and runtime libraries.
For PyTorch installations without that package, provide system cuDNN development
files instead.

Follow the [coding conventions](AGENTS.md) and the test comparison policy below.
Update tests and documentation, validate distributed changes with one and multiple
GPUs, and include reproducible before/after measurements for performance changes.

Review the [compatibility notes](docs/compatibility.md) when changing PyTorch
interfaces, RNG state, or checkpoint behavior. They describe the implementation
dependencies to review and the saved-state compatibility requirements.

### Troubleshooting source builds

The build checks for a CUDA-enabled PyTorch installation, a CUDA toolkit with
`nvcc`, and CUTLASS headers before constructing the extension. It also checks the
cuDNN headers when they are supplied by PyTorch's NVIDIA Python package.

- **CPU-only PyTorch:** install a CUDA-enabled PyTorch build matching the intended
  toolkit in the active environment, then rebuild with `--no-build-isolation`.
- **Missing CUDA toolkit or `nvcc`:** install the CUDA development toolkit and set
  `CUDA_HOME` to its root. PyTorch's bundled runtime libraries do not supply the
  development toolchain. A container build needs a CUDA development image.
- **Missing CUTLASS:** run `python -m pip install nvidia-cutlass` in the build environment.
  If the package exists but its headers are missing, reinstall it with
  `python -m pip install --force-reinstall nvidia-cutlass`.
- **Missing cuDNN headers:** for package-provided cuDNN, reinstall the NVIDIA cuDNN
  package at the version required by PyTorch. Other PyTorch builds may
  instead use system cuDNN development headers. In that case header discovery is
  left to the compiler; configure its include paths or `CUDNN_HOME` as needed.
  A runtime-only cuDNN installation is insufficient. Remove an incomplete NVIDIA
  Python package before using the system-header fallback.
- **CUDA or compiler version mismatch:** follow PyTorch's extension-build error
  or warning and select a compatible toolkit and host compiler. torch-dgx retains
  PyTorch's existing checks for these combinations.

A visible GPU is not required to compile. When building without GPUs, set
`TORCH_CUDA_ARCH_LIST` for the GPUs on which the extension will run. For example,
to target compute capability 8.9:

```bash
TORCH_CUDA_ARCH_LIST="8.9" python -m pip install --no-build-isolation -e .
```

Choose the architecture list for the deployment hardware and toolkit; GPU
availability and runtime compatibility must be validated on the deployment
system. Rebuild after changing PyTorch or the CUDA toolchain.

## Submission and Review

Fork the repository, create a branch from `master`, and open a focused pull
request against `master`. Complete the CLA and use the PR template to describe
the change, related issues, validation commands/results, environment, and any
skipped checks. Address review comments and rerun affected checks. Maintainers
review scope, correctness, compatibility, and maintainability before approval.

## Maintainer Responsibilities and Process

[Maintainers](CONTRIBUTORS.md#maintainers) triage issues, set priorities, make
technical decisions, and merge reviewed PRs after confirming CLA approval and
passing checks. Record decisions in issues or PRs and keep the contributor list
and policies current. Support and review are provided as time permits.

For public releases, update the version in `pyproject.toml`, validate supported
configurations, review documentation, notices, and required approvals, and publish
a matching version tag and [GitHub Release](https://github.com/NVIDIA/torch-dgx/releases).
GitHub Releases are the canonical public history and should describe changes,
compatibility, and known limitations.

## Test Comparison Policy

Use CPU PyTorch to compute reference tensors. PyTorch supports only one active
accelerator backend at a time, so tests must not mix its CUDA backend
(`device="cuda"`) with the DGX backend (`device="dgx"`) in the same process.
DGX uses the CUDA runtime internally to allocate GPU memory and launch kernels;
this is distinct from using PyTorch's CUDA backend. Compute expected results
with CPU tensors.

- Use CPU comparisons for focused operator correctness tests. This keeps the
  assertion independent of DGX comparison and reduction kernels. Copy tests,
  RNG reproducibility and partition-invariance tests, and focused comparison or
  reduction tests should also retain a CPU reference.
- Use DGX comparisons for integration and smoke tests. Compute the reference
  with CPU PyTorch, move it to DGX, and run `torch.equal` or `torch.allclose` on
  DGX.
- Test DGX `equal` and `allclose` directly with known Boolean outcomes. Cover
  relevant tolerance, NaN, infinity, dtype, shape, and layout cases.

Focused operator test:

```python
assert torch.allclose(actual.cpu(), expected, rtol=rtol, atol=atol)
```

Integration test:

```python
assert torch.allclose(actual, expected.to("dgx"), rtol=rtol, atol=atol)
```
