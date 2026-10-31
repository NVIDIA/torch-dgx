# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
import torch
import torch_dgx._C

from typing import Any


def _get_device_index(device: Any) -> int:
    if isinstance(device, int):
        device_index = device
    else:
        parsed_device = torch.device(device)
        if parsed_device.type != "dgx":
            raise ValueError(f"Expected a dgx device, but got {parsed_device}")
        device_index = parsed_device.index
        if device_index is None:
            device_index = DGXModule.current_device()

    if device_index != 0:
        raise ValueError(
            f"DGX only exposes logical device 0, but got device {device_index}"
        )
    return device_index


class DGXModule:
    _initialized = False

    @staticmethod
    def is_available():
        return torch.cuda.is_available()

    @staticmethod
    def is_initialized():
        return DGXModule._initialized

    @staticmethod
    def device_count():
        return int(DGXModule.is_available())

    @staticmethod
    def current_device():
        return 0

    @staticmethod
    def is_autocast_enabled():
        return torch.is_autocast_enabled("dgx")

    @staticmethod
    def get_autocast_dtype():
        return torch.get_autocast_dtype("dgx")

    @staticmethod
    def set_autocast_enabled(enable):
        torch.set_autocast_enabled("dgx", enable)

    @staticmethod
    def set_autocast_dtype(dtype):
        torch.set_autocast_dtype("dgx", dtype)

    @staticmethod
    def get_amp_supported_dtype():
        return [torch.float16]

    @staticmethod
    def manual_seed_all(seed):
        seed = int(seed)
        for generator in DGXModule.default_generators:
            generator.manual_seed(seed)

    @staticmethod
    def manual_seed(seed):
        DGXModule.default_generators[DGXModule.current_device()].manual_seed(int(seed))

    @staticmethod
    def seed_all():
        random_seed = DGXModule.default_generators[0].seed()
        for generator in DGXModule.default_generators[1:]:
            generator.manual_seed(random_seed)

    @staticmethod
    def seed():
        DGXModule.default_generators[DGXModule.current_device()].seed()

    @staticmethod
    def initial_seed():
        return DGXModule.default_generators[DGXModule.current_device()].initial_seed()

    @staticmethod
    def _is_in_bad_fork():
        return torch.cuda._is_in_bad_fork()

    @staticmethod
    def get_rng_state(device: Any = "dgx"):
        return DGXModule.default_generators[_get_device_index(device)].get_state()

    @staticmethod
    def get_rng_state_all():
        return [DGXModule.get_rng_state(i) for i in range(DGXModule.device_count())]

    @staticmethod
    def set_rng_state(new_state: torch.Tensor, device: Any = "dgx"):
        if (
            not isinstance(new_state, torch.Tensor)
            or new_state.layout != torch.strided
            or new_state.device.type != "cpu"
            or new_state.dtype != torch.uint8
        ):
            raise TypeError("RNG state must be a torch.ByteTensor")
        # Match SerializedDGXGeneratorState before allocating the contiguous copy.
        if new_state.numel() != 16:
            raise RuntimeError(
                "Expected a DGX generator state of 16 bytes, "
                f"but found {new_state.numel()} bytes"
            )
        new_state = new_state.clone(memory_format=torch.contiguous_format)
        DGXModule.default_generators[_get_device_index(device)].set_state(new_state)

    @staticmethod
    def set_rng_state_all(new_states):
        for i, state in enumerate(new_states):
            DGXModule.set_rng_state(state, i)

    class device:
        r"""Context manager for the sole logical DGX device.

        Args:
            device (torch.device or int): device index to select. It's a no-op if
                this argument is a negative integer or ``None``.
        """

        def __init__(self, device: Any):
            if device is None or (isinstance(device, int) and device < 0):
                return
            _get_device_index(device)

        def __enter__(self):
            pass

        def __exit__(self, type: Any, value: Any, traceback: Any):
            return False


torch.utils.rename_privateuse1_backend("dgx")
DGXModule.default_generators = (torch_dgx._C._get_default_generator(),)
# Checkpoint reads this flag directly when deciding whether to preserve device
# RNG state. DGX's default generator is ready before device registration.
DGXModule._initialized = True
torch._register_device_module("dgx", DGXModule)
torch.utils.generate_methods_for_privateuse1_backend(
    for_tensor=True, for_module=True, for_storage=True, unsupported_dtype=[]
)
