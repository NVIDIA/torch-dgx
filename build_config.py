# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
from importlib.metadata import PackageNotFoundError, distribution
from pathlib import Path

from torch.utils.cpp_extension import CUDA_HOME, BuildExtension, CUDAExtension
from torch.version import cuda as torch_cuda_version

ROOT = Path(__file__).parent.resolve()


def _cudnn_include_dirs():
    package = f"nvidia-cudnn-cu{torch_cuda_version.split('.')[0]}"
    try:
        cudnn = distribution(package)
    except PackageNotFoundError:
        # PyTorch builds may use system cuDNN instead.
        return []
    include_dir = Path(cudnn.locate_file("nvidia/cudnn/include"))
    if not (include_dir / "cudnn.h").is_file():
        raise RuntimeError(
            f"torch-dgx could not find cuDNN headers in {include_dir}. "
            f"Reinstall {package} at the version required by your PyTorch build. "
            "For system cuDNN, provide its development headers and "
            "remove the incomplete Python package. See CONTRIBUTING.md."
        )
    return [str(include_dir)]


def _build_include_dirs():
    # Building needs a CUDA toolchain, but does not require a visible GPU.
    if torch_cuda_version is None:
        raise RuntimeError(
            "torch-dgx requires a CUDA-enabled PyTorch build. Install one "
            "matching your CUDA toolkit before building with --no-build-isolation. "
            "See CONTRIBUTING.md for environment setup."
        )
    if not CUDA_HOME:
        raise RuntimeError(
            "torch-dgx could not locate the CUDA development toolkit. "
            "Install it and set CUDA_HOME to its root directory. "
            "The CUDA runtime bundled with PyTorch is not sufficient to build."
        )
    cuda_home = Path(CUDA_HOME)
    if not (cuda_home / "bin" / "nvcc").is_file():
        raise RuntimeError(
            f"torch-dgx could not find nvcc at {cuda_home / 'bin' / 'nvcc'}. "
            "Install the CUDA development toolkit or correct CUDA_HOME."
        )

    try:
        cutlass = distribution("nvidia-cutlass")
    except PackageNotFoundError as error:
        raise RuntimeError(
            "torch-dgx requires CUTLASS development headers. "
            "Run `python -m pip install nvidia-cutlass` in the build environment "
            "before building with --no-build-isolation."
        ) from error
    cutlass_include = Path(cutlass.locate_file("cutlass_library/source/include"))
    if not (cutlass_include / "cutlass" / "cutlass.h").is_file():
        raise RuntimeError(
            f"torch-dgx could not find CUTLASS headers in {cutlass_include}. "
            "Run `python -m pip install --force-reinstall nvidia-cutlass` "
            "in the build environment."
        )

    return [
        # Use the headers matching PyTorch's cuDNN runtime when available.
        *_cudnn_include_dirs(),
        # CUDA 13 places CCCL outside the host compiler's CUDA include path.
        str(cuda_home / "include" / "cccl"),
        str(cutlass_include),
    ]


class EditableBuildExtension(BuildExtension):
    def finalize_options(self):
        extensions = []
        for extension in self.distribution.ext_modules:
            if getattr(extension, "_torch_dgx_cuda_extension", False):
                extensions.append(extension)
                continue

            extension = CUDAExtension(
                extension.name,
                extension.sources,
                include_dirs=_build_include_dirs(),
                extra_compile_args={"nvcc": ["-lineinfo"]},
            )
            extension._torch_dgx_cuda_extension = True
            extensions.append(extension)

        self.distribution.ext_modules = extensions
        super().finalize_options()

    def run(self):
        if getattr(self, "editable_mode", False):
            self.build_temp = str(ROOT / "build" / "temp.editable")
        super().run()
