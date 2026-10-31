# Security Policy: torch-dgx

## Reporting a Vulnerability

If you discover a potential security vulnerability in torch-dgx, **do not open a
public issue, pull request, merge request, or discussion containing its details.**
Use a private reporting channel:

- **Web (preferred):** Follow the reporting instructions at the
  [NVIDIA Vulnerability Disclosure Program](https://www.nvidia.com/en-us/security/).
- **Email:** Contact [psirt@nvidia.com](mailto:psirt@nvidia.com). Encrypt sensitive
  reports with the [NVIDIA public PGP key](https://www.nvidia.com/en-us/security/pgp-key/).
- **Repository private reporting:** Where available, use the repository's
  **Security** tab > **Report a vulnerability**. Otherwise, use the NVIDIA
  program or PSIRT email above.

Include the project name, affected version or commit, vulnerability type,
reproduction steps, proof-of-concept code if available, and potential impact.
For GPU issues, also include the Python, PyTorch, CUDA and driver versions, GPU
models and count, relevant tensor shapes/dtypes, and stream or threading setup.
Remove credentials and unrelated private data from reproductions and logs.

NVIDIA's Product Security Incident Response Team (PSIRT) acknowledges reports,
investigates and assesses severity, coordinates fixes or mitigations, and
publishes security bulletins as appropriate under its
[coordinated vulnerability disclosure policy](https://www.nvidia.com/en-us/security/psirt-policies/).
Community conduct reports follow [CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md);
software vulnerability reports should use the channels above.

## Security Architecture & Context

torch-dgx is an **open-source Python/C++/CUDA library** that presents multiple
NVIDIA GPUs as one logical PyTorch device, `dgx:0`. It runs within the calling
application's process and handles tensors, model parameters, and training state.
Its security responsibilities concern tensor integrity and memory safety within
that process and its assigned GPU resources.

**Repository Exposure Classification:** Public.
Basis: intended for public release.

**Service Exposure Classification:** External / Regulated (high confidence).
Basis: external distribution as an open-source library.
These classifications describe the project's audience and distribution context.

The Python and PyTorch interfaces, native backend, and assigned GPUs share one
workload trust domain. torch-dgx does not sandbox application code or isolate
tenants. The host OS, GPU driver, and deployment environment provide isolation;
applications provide authentication, authorization, and resource controls.

Tensor data may reside in both host and GPU memory, including through CPU
fallback for unsupported operators. Input and runtime checks help detect errors
but do not replace application input validation or deployment controls. See
[README.md](README.md) for hardware and software requirements.

## Threat Model

These scenarios describe potential risks, rather than confirmed vulnerabilities.
Their impact depends on the application and deployment.

1. **Compromised dependencies or artifacts:** The native backend and build and
   installation tooling rely on trusted packages and libraries. Compromise of
   these inputs could expose application data or alter computation.
2. **Untrusted code in the application process:** Python model code and custom
   operators run with the application's privileges and can access its tensors.
   Separating work across GPUs does not isolate mutually untrusted workloads.
3. **Sensitive data disclosure:** CPU fallback, tensor outputs, and training or
   profiling tools can expose model or training data through host memory, logs,
   captures, or crash dumps.
4. **Invalid tensor inputs:** Malformed inputs to PyTorch tensor operations may
   cause failures or incorrect results. Defects in native input handling can
   affect memory safety.
5. **Resource exhaustion:** Large tensor allocations, expensive GPU operations,
   or repeated CPU fallback can consume excessive memory, compute time, or
   transfer bandwidth.
6. **Unsupported execution patterns:** Native GPU operations depend on correct
   stream ordering and tensor lifetimes. Unsupported concurrent calls or stream
   use can compromise result integrity and memory safety.

## Critical Security Assumptions

- **Trusted code and dependencies:** Load application code, model code, custom
  operators, and native libraries only from trusted sources. Protect build and
  release inputs and verify deployed artifacts.
- **Isolated deployment:** Use trusted, compatible hardware, drivers, and host
  software. Assign all GPUs visible to torch-dgx to the same workload trust
  domain and enforce isolation between mutually untrusted workloads externally.
- **Supported execution:** Use one host thread for kernel launches, allocations,
  and deallocations, including indirect framework calls. Preserve tensor
  lifetimes and CUDA stream dependencies when integrating other GPU work.
- **Validated, bounded workloads:** Validate external tensor inputs and establish
  trust in datasets and model checkpoints. Enforce memory and execution limits
  and provide recovery for failed or stalled workloads.
- **Protected application interfaces:** Applications exposing torch-dgx through
  a service must provide authentication, authorization, and secure transport.
- **Protected data:** Control access to host and GPU memory, outputs, logs, and
  diagnostic artifacts. Applications must supply any required encryption and
  data retention or erasure controls; freeing tensors does not securely erase
  their contents.

Review this policy when interfaces, execution requirements, dependencies, or
deployment exposure change.
