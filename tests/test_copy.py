# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
import math

import pytest
import torch


def _assert_copy(source, destination, expected, backing=None):
    expected_device = destination.device
    expected_dtype = destination.dtype

    destination.copy_(source)

    assert destination.device == expected_device
    assert destination.dtype == expected_dtype
    actual = destination if backing is None else backing
    assert torch.equal(actual.cpu(), expected)


_DEVICE_COPY_DTYPES = [
    pytest.param(torch.float32, torch.float32, id="same-dtype"),
    pytest.param(torch.int32, torch.float64, id="cast"),
]


def _arange_tensor(shape, dtype):
    return torch.arange(math.prod(shape), dtype=dtype).reshape(shape)


@pytest.mark.parametrize(
    "source_device,destination_device",
    [
        pytest.param("cpu", "dgx", id="cpu-to-dgx"),
        pytest.param("dgx", "cpu", id="dgx-to-cpu"),
        pytest.param("dgx", "dgx", id="dgx-to-dgx"),
    ],
)
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16], ids=str)
def test_dense_same_dtype_copy(
    dgx_setup_teardown, source_device, destination_device, dtype
):
    expected = torch.tensor([[1.5, -2.25, 3.0], [4.5, 5.25, -6.0]], dtype=dtype)
    source = expected.to(source_device)
    destination = torch.empty(expected.shape, dtype=dtype, device=destination_device)

    assert source.is_contiguous()
    assert destination.is_contiguous()

    _assert_copy(source, destination, expected)


@pytest.mark.parametrize(
    "source_device,destination_device,source_dtype,destination_dtype",
    [
        pytest.param(
            "cpu", "dgx", torch.float16, torch.int32, id="cpu-float16-to-dgx-int32"
        ),
        pytest.param(
            "dgx",
            "cpu",
            torch.bfloat16,
            torch.bfloat16,
            id="dgx-to-cpu-bfloat16",
        ),
    ],
)
def test_host_strided_copy(
    dgx_setup_teardown,
    source_device,
    destination_device,
    source_dtype,
    destination_dtype,
):
    storage_cpu = (torch.arange(24, dtype=torch.float32).reshape(4, 6) * 0.75 - 4.5).to(
        source_dtype
    )
    source_storage = storage_cpu.to(source_device)
    source = source_storage[:, ::2]
    destination = torch.empty(
        source.shape, dtype=destination_dtype, device=destination_device
    )
    expected = storage_cpu[:, ::2].to(destination_dtype)

    assert not source.is_contiguous()
    assert destination.is_contiguous()

    _assert_copy(source, destination, expected)


def test_device_dense_cast(dgx_setup_teardown):
    expected_source = torch.arange(24, dtype=torch.int32).reshape(4, 6)
    source = expected_source.to("dgx")
    destination = torch.full(source.shape, -99, dtype=torch.float64, device="dgx")

    assert source.is_contiguous()
    assert destination.is_contiguous()

    _assert_copy(source, destination, expected_source.to(torch.float64))


def test_device_strided_cast(dgx_setup_teardown):
    source_storage_cpu = (
        torch.arange(4 * 5 * 6, dtype=torch.float32).reshape(4, 5, 6) * 0.75 - 4.5
    )
    source_cpu = source_storage_cpu[1:, 1::2, ::2].transpose(0, 2)
    source_storage = source_storage_cpu.to("dgx")
    source = source_storage[1:, 1::2, ::2].transpose(0, 2)

    destination_backing = torch.full(
        (*source.shape[:-1], 2 * source.shape[-1] + 1),
        -99,
        dtype=torch.float64,
        device="dgx",
    )
    destination = destination_backing[..., 1::2]
    expected = torch.full(destination_backing.shape, -99, dtype=torch.float64)
    expected[..., 1::2] = source_cpu.to(torch.float64)

    assert not source.is_contiguous()
    assert not destination.is_contiguous()

    _assert_copy(source, destination, expected, backing=destination_backing)


def test_device_strided_same_dtype_copy(dgx_setup_teardown):
    source_backing_cpu = torch.arange(12, dtype=torch.int64)
    source_backing = source_backing_cpu.to("dgx")
    source = source_backing[1::2]
    destination_backing = torch.full((12,), -99, dtype=torch.int64, device="dgx")
    destination = destination_backing[::2]
    expected = torch.full((12,), -99, dtype=torch.int64)
    expected[::2] = source_backing_cpu[1::2]

    assert not source.is_contiguous()
    assert not destination.is_contiguous()

    _assert_copy(source, destination, expected, backing=destination_backing)


def test_device_rank8_strided_copy(dgx_setup_teardown):
    base_cpu = torch.arange(2, dtype=torch.float32).reshape(1, 1, 1, 1, 1, 1, 1, 2)
    expected = base_cpu.expand(2, 2, 2, 2, 2, 2, 2, 2).contiguous()
    source = base_cpu.to("dgx").expand(expected.shape)

    assert source.dim() == 8
    assert not source.is_contiguous()

    actual = source.contiguous()

    assert torch.equal(actual.cpu(), expected)


@pytest.mark.parametrize("dtype", [torch.complex64, torch.complex128], ids=str)
def test_complex_copy(dgx_setup_teardown, dtype):
    real_dtype = torch.float64 if dtype == torch.complex128 else torch.float32
    real = torch.arange(24, dtype=real_dtype).reshape(4, 6) + 2**-35
    source_cpu = torch.complex(real, -0.25 * real)

    dense = source_cpu.to("dgx")
    assert torch.equal(dense.cpu(), source_cpu)

    source = dense[:, ::2]
    destination = torch.empty(source.shape, dtype=dtype, device="dgx")
    destination.copy_(source)

    assert not source.is_contiguous()
    assert torch.equal(destination.cpu(), source_cpu[:, ::2])
    assert torch.equal(source.cpu(), source_cpu[:, ::2])

    destination.copy_(source_cpu[:, 1::2])
    assert torch.equal(destination.cpu(), source_cpu[:, 1::2])


@pytest.mark.parametrize("dtype", [torch.complex64, torch.complex128], ids=str)
@pytest.mark.parametrize(
    "source_device,destination_device", [("cpu", "dgx"), ("dgx", "cpu"), ("dgx", "dgx")]
)
@pytest.mark.parametrize("strided", [False, True])
@pytest.mark.parametrize(
    "source_conjugated,destination_conjugated",
    [(True, False), (False, True), (True, True)],
)
def test_complex_conjugated_copy(
    dgx_setup_teardown,
    dtype,
    source_device,
    destination_device,
    strided,
    source_conjugated,
    destination_conjugated,
):
    source_cpu = torch.tensor([1 + 2j, -3 + 4j, 5 - 6j, -7 - 8j], dtype=dtype)
    source = source_cpu.to(source_device)
    destination_backing = torch.zeros_like(source_cpu).to(destination_device)
    expected = torch.zeros_like(source_cpu)
    destination = destination_backing
    if strided:
        source_cpu = source_cpu[::2]
        source = source[::2]
        destination = destination[1::2]
        expected_view = expected[1::2]
    else:
        expected_view = expected
    if source_conjugated:
        source_cpu = source_cpu.conj()
        source = source.conj()
    if destination_conjugated:
        destination = destination.conj()
        expected_view = expected_view.conj()
    expected_view.copy_(source_cpu)

    _assert_copy(source, destination, expected, backing=destination_backing)


@pytest.mark.parametrize("shape", [(37, 53), (3, 35, 47)], ids=["2d", "batched"])
@pytest.mark.parametrize(
    "to_contiguous", [True, False], ids=["to-contiguous", "to-transposed"]
)
@pytest.mark.parametrize("source_dtype,destination_dtype", _DEVICE_COPY_DTYPES)
def test_device_tiled_transpose_copy(
    dgx_setup_teardown, shape, to_contiguous, source_dtype, destination_dtype
):
    expected = _arange_tensor(shape, source_dtype)

    if to_contiguous:
        source = expected.transpose(-2, -1).contiguous().to("dgx").transpose(-2, -1)
        destination = torch.empty(expected.shape, dtype=destination_dtype, device="dgx")
    else:
        source = expected.to("dgx")
        destination = torch.empty(
            (*expected.shape[:-2], expected.shape[-1], expected.shape[-2]),
            dtype=destination_dtype,
            device="dgx",
        ).transpose(-2, -1)

    _assert_copy(source, destination, expected.to(destination_dtype))


@pytest.mark.parametrize(
    "shape",
    [(2, 33, 97, 83), (3, 3, 257, 263), (3, 4, 257, 263)],
    ids=["channels-last", "rgb", "rgba"],
)
@pytest.mark.parametrize(
    "to_contiguous",
    [False, True],
    ids=["nchw-to-channels-last", "channels-last-to-nchw"],
)
@pytest.mark.parametrize("source_dtype,destination_dtype", _DEVICE_COPY_DTYPES)
def test_device_channels_last_copy(
    dgx_setup_teardown,
    shape,
    to_contiguous,
    source_dtype,
    destination_dtype,
):
    expected = _arange_tensor(shape, source_dtype)

    if to_contiguous:
        source = expected.permute(0, 2, 3, 1).contiguous().to("dgx").permute(0, 3, 1, 2)
        destination = torch.empty(shape, dtype=destination_dtype, device="dgx")
    else:
        source = expected.to("dgx")
        destination = torch.empty(
            shape,
            dtype=destination_dtype,
            device="dgx",
            memory_format=torch.channels_last,
        )

    _assert_copy(source, destination, expected.to(destination_dtype))
    assert destination.is_contiguous(
        memory_format=(
            torch.contiguous_format if to_contiguous else torch.channels_last
        )
    )


def test_device_pitched_rgb_to_planar_copy(dgx_setup_teardown):
    backing = _arange_tensor((3, 263, 277, 4), torch.float32)
    expected = backing[:, 3:260, 5:274, :3].permute(0, 3, 1, 2)
    source = backing.to("dgx")[:, 3:260, 5:274, :3].permute(0, 3, 1, 2)
    destination = torch.empty(expected.shape, dtype=torch.float32, device="dgx")

    assert not source.is_contiguous(memory_format=torch.channels_last)
    assert source.stride(1) == 1
    assert source.stride(3) == 4

    destination.copy_(source)

    assert torch.equal(destination.cpu(), expected)


@pytest.mark.parametrize(
    "shape", [(1, 3, 1, 1), (1, 3, 17, 65), (3, 4, 33, 127)], ids=str
)
@pytest.mark.parametrize("source_dtype,destination_dtype", _DEVICE_COPY_DTYPES)
def test_device_planar_to_interleaved_copy(
    dgx_setup_teardown, shape, source_dtype, destination_dtype
):
    planar = _arange_tensor(shape, source_dtype)
    expected = planar.permute(0, 2, 3, 1).to(destination_dtype)
    source = planar.to("dgx").permute(0, 2, 3, 1)
    destination = torch.empty(expected.shape, dtype=destination_dtype, device="dgx")

    _assert_copy(source, destination, expected)
    assert destination.is_contiguous()


def test_device_planar_to_pitched_rgb_copy(dgx_setup_teardown):
    planar = _arange_tensor((3, 3, 39, 139), torch.float32)
    source = planar.to("dgx")[:, :, 3:36, 5:132].permute(0, 2, 3, 1)
    expected = torch.full((3, 39, 139, 4), -1.0)
    destination_backing = expected.to("dgx")
    destination = destination_backing[:, 2:35, 7:134, :3]
    expected[:, 2:35, 7:134, :3] = planar[:, :, 3:36, 5:132].permute(0, 2, 3, 1)

    _assert_copy(source, destination, expected, backing=destination_backing)


@pytest.mark.parametrize("shape", [(0, 17), (1, 37), (37, 1)], ids=str)
def test_device_tiled_transpose_copy_edge_shapes(dgx_setup_teardown, shape):
    expected = _arange_tensor(shape, torch.float32)
    source = expected.t().contiguous().to("dgx").t()
    destination = torch.empty(shape, dtype=torch.float32, device="dgx")

    destination.copy_(source)

    assert torch.equal(destination.cpu(), expected)


def test_device_tiled_transpose_copy_storage_offset(dgx_setup_teardown):
    rows, columns = 41, 67
    backing = torch.arange((columns + 1) * rows, dtype=torch.int64).reshape(
        columns + 1, rows
    )
    expected = backing[1:].t()
    source = backing.to("dgx")[1:].t()
    destination = torch.empty((rows, columns), dtype=torch.int64, device="dgx")

    destination.copy_(source)

    assert torch.equal(destination.cpu(), expected)


@pytest.mark.parametrize(
    "shape,layout",
    [
        pytest.param((2, 3, 17, 29), "dense", id="dense"),
        pytest.param((2, 3, 17, 29), "packed", id="gapped-qkv"),
        pytest.param((3, 3, 37, 1733), "dense", id="large-dense"),
        pytest.param((3, 3, 37, 1733), "packed", id="large-gapped-qkv"),
        pytest.param((0, 3, 17, 29), "padded-heads", id="empty-batch"),
        pytest.param((2, 1, 1, 37), "padded-heads", id="single-head-channel"),
        pytest.param((3, 5, 33, 1), "padded-heads", id="single-position"),
    ],
)
@pytest.mark.parametrize(
    "source_dtype,destination_dtype",
    _DEVICE_COPY_DTYPES
    + [
        pytest.param(torch.float16, torch.float16, id="float16"),
        pytest.param(torch.bfloat16, torch.float32, id="bfloat16-to-float32"),
        pytest.param(torch.complex64, torch.complex64, id="complex64"),
        pytest.param(torch.complex128, torch.complex128, id="complex128"),
    ],
)
def test_device_transposed_key_copy(
    dgx_setup_teardown, shape, layout, source_dtype, destination_dtype
):
    b, h, d, s = shape
    if layout == "padded-heads":
        storage_shape = (b, s, h, d + 3)
    elif layout == "packed":
        storage_shape = (b + 2, s + 2, 3 * h * d + 7)
    else:
        storage_shape = (b, s, h * d)
    storage_cpu = (_arange_tensor(storage_shape, torch.int64) % 31).to(source_dtype)
    storage = storage_cpu.to("dgx")

    def key_view(value):
        if layout == "padded-heads":
            return value[..., :d].permute(0, 2, 3, 1)
        if layout == "packed":
            value = value[1:-1, 1:-1, h * d : 2 * h * d]
        return value.reshape(b, s, h, d).permute(0, 2, 3, 1)

    expected = key_view(storage_cpu).to(destination_dtype)
    source = key_view(storage)
    # A contiguous destination with an offset, with guards on both sides.
    destination_cpu = torch.full((math.prod(shape) + 36,), -99, dtype=destination_dtype)
    destination = destination_cpu.to("dgx")
    destination_cpu[17:-19].view(shape).copy_(expected)
    _assert_copy(
        source, destination[17:-19].view(shape), destination_cpu, backing=destination
    )


def test_device_copy_rejects_tiled_overlap(dgx_setup_teardown):
    tensor = _arange_tensor((65, 65), torch.float32).to("dgx")

    with pytest.raises(RuntimeError, match="single memory location"):
        tensor.copy_(tensor.t())


def test_device_copy_rejects_interleaved_overlap(dgx_setup_teardown):
    channels, height, width = 3, 257, 263
    backing = torch.arange(channels * height * width, dtype=torch.float32).to("dgx")
    destination = backing.view(1, channels, height, width)
    source = backing.as_strided(
        (1, channels, height, width),
        (channels * height * width, 1, channels * width, channels),
    )

    with pytest.raises(RuntimeError, match="single memory location"):
        destination.copy_(source)


def test_device_copy_rejects_internal_destination_overlap(dgx_setup_teardown):
    source = torch.arange(17, dtype=torch.float32).to("dgx")
    destination = torch.empty(1, dtype=torch.float32, device="dgx").expand_as(source)

    with pytest.raises(RuntimeError, match="single memory location"):
        destination.copy_(source)
