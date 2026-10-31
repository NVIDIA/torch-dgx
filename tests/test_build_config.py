# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
import importlib.util
from importlib.metadata import PackageNotFoundError
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch


spec = importlib.util.spec_from_file_location(
    "build_config", Path(__file__).resolve().parents[1] / "build_config.py"
)
build_config = importlib.util.module_from_spec(spec)
spec.loader.exec_module(build_config)


@pytest.fixture
def build_environment(tmp_path, monkeypatch):
    roots = {name: tmp_path / name for name in ("cuda", "cutlass", "cudnn")}
    for root, header in (
        (roots["cuda"], "bin/nvcc"),
        (roots["cutlass"], "cutlass_library/source/include/cutlass/cutlass.h"),
        (roots["cudnn"], "nvidia/cudnn/include/cudnn.h"),
    ):
        path = root / header
        path.parent.mkdir(parents=True)
        path.touch()

    packages = {
        "nvidia-cutlass": SimpleNamespace(locate_file=roots["cutlass"].joinpath),
        "nvidia-cudnn-cu13": SimpleNamespace(locate_file=roots["cudnn"].joinpath),
    }

    def distribution(name):
        if name not in packages:
            raise PackageNotFoundError(name)
        return packages[name]

    monkeypatch.setattr(build_config, "CUDA_HOME", str(roots["cuda"]))
    monkeypatch.setattr(build_config, "torch_cuda_version", "13.2")
    monkeypatch.setattr(build_config, "distribution", distribution)
    return roots, packages


def test_build_rejects_cpu_only_pytorch(build_environment, monkeypatch):
    monkeypatch.setattr(build_config, "torch_cuda_version", None)
    monkeypatch.setattr(build_config, "CUDA_HOME", None)
    with pytest.raises(RuntimeError, match="CUDA-enabled PyTorch"):
        build_config._build_include_dirs()


def test_build_rejects_missing_cuda_toolkit(build_environment, monkeypatch):
    monkeypatch.setattr(build_config, "CUDA_HOME", None)
    with pytest.raises(RuntimeError, match="set CUDA_HOME"):
        build_config._build_include_dirs()


def test_build_rejects_runtime_only_cuda(build_environment):
    roots, _ = build_environment
    (roots["cuda"] / "bin/nvcc").unlink()
    with pytest.raises(RuntimeError, match="could not find nvcc") as error:
        build_config._build_include_dirs()
    assert str(roots["cuda"]) in str(error.value)


def test_build_rejects_missing_cutlass(build_environment):
    _, packages = build_environment
    del packages["nvidia-cutlass"]
    with pytest.raises(RuntimeError, match="python -m pip install nvidia-cutlass"):
        build_config._build_include_dirs()


@pytest.mark.parametrize(
    "package,header,message",
    [
        (
            "cutlass",
            "cutlass_library/source/include/cutlass/cutlass.h",
            "python -m pip install --force-reinstall nvidia-cutlass",
        ),
        (
            "cudnn",
            "nvidia/cudnn/include/cudnn.h",
            "Reinstall nvidia-cudnn-cu13",
        ),
    ],
)
def test_build_rejects_incomplete_header_package(
    build_environment, package, header, message
):
    roots, _ = build_environment
    (roots[package] / header).unlink()
    with pytest.raises(RuntimeError, match=message) as error:
        build_config._build_include_dirs()
    assert str(roots[package]) in str(error.value)


def test_build_accepts_system_cudnn(build_environment):
    roots, packages = build_environment
    del packages["nvidia-cudnn-cu13"]
    include_dirs = build_config._build_include_dirs()
    assert str(roots["cudnn"] / "nvidia/cudnn/include") not in include_dirs
    assert str(roots["cutlass"] / "cutlass_library/source/include") in include_dirs


def test_build_does_not_probe_gpus(build_environment, monkeypatch):
    roots, _ = build_environment

    def unexpected_probe(*args, **kwargs):
        pytest.fail("Build prerequisites must not probe GPU availability")

    monkeypatch.setattr(torch.cuda, "is_available", unexpected_probe)
    monkeypatch.setattr(torch.cuda, "device_count", unexpected_probe)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    monkeypatch.setenv("TORCH_CUDA_ARCH_LIST", "8.9")
    include_dirs = build_config._build_include_dirs()
    assert str(roots["cudnn"] / "nvidia/cudnn/include") in include_dirs
    assert str(roots["cutlass"] / "cutlass_library/source/include") in include_dirs
