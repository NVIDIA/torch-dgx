# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
import pytest
import torch

datatypes = [torch.int32, torch.int64, torch.float32, torch.float64]


@pytest.mark.parametrize(
    "dtype, complex_dtype",
    [
        (torch.float16, torch.complex32),
        (torch.float32, torch.complex64),
        (torch.float64, torch.complex128),
    ],
    ids=lambda dtype: str(dtype),
)
def test_view_as_complex(dgx_setup_teardown, dtype, complex_dtype):
    cpu_base = torch.arange(96, dtype=dtype).reshape(2, 3, 4, 2, 2)
    base = cpu_base.to("dgx")
    tensor = base[:, :, ::2, 1, :]

    result = torch.view_as_complex(tensor)

    assert result.device.type == "dgx"
    assert result.dtype == complex_dtype
    assert result.shape == tensor.shape[:-1]
    assert result.stride() == tuple(stride // 2 for stride in tensor.stride()[:-1])
    assert result.storage_offset() == tensor.storage_offset() // 2
    assert result.untyped_storage().data_ptr() == tensor.untyped_storage().data_ptr()
    assert torch.equal(torch.view_as_real(result).cpu(), cpu_base[:, :, ::2, 1, :])

    base.fill_(7)
    assert torch.equal(
        torch.view_as_real(result).cpu(),
        torch.full(tensor.shape, 7, dtype=dtype),
    )


@pytest.mark.parametrize(
    "dtype, real_dtype",
    [
        (torch.complex32, torch.float16),
        (torch.complex64, torch.float32),
        (torch.complex128, torch.float64),
    ],
    ids=lambda dtype: str(dtype),
)
def test_view_as_real(dgx_setup_teardown, dtype, real_dtype):
    if dtype == torch.complex32:
        with pytest.warns(UserWarning, match="ComplexHalf support is experimental"):
            tensor = torch.empty((3, 4), dtype=dtype, device="dgx")
    else:
        tensor = torch.empty((3, 4), dtype=dtype, device="dgx")

    result = torch.view_as_real(tensor)

    assert result.device.type == "dgx"
    assert result.dtype == real_dtype
    assert result.shape == (*tensor.shape, 2)
    assert result.stride() == tuple(2 * stride for stride in tensor.stride()) + (1,)
    assert result.storage_offset() == 2 * tensor.storage_offset()
    assert result.untyped_storage().data_ptr() == tensor.untyped_storage().data_ptr()

    result.fill_(5)
    assert torch.equal(result.cpu(), torch.full(result.shape, 5, dtype=real_dtype))

    round_trip = torch.view_as_complex(result)
    assert round_trip.dtype == dtype
    assert round_trip.shape == tensor.shape
    assert round_trip.stride() == tensor.stride()
    assert round_trip.storage_offset() == tensor.storage_offset()
    assert (
        round_trip.untyped_storage().data_ptr() == tensor.untyped_storage().data_ptr()
    )

    noncontiguous = tensor[:, ::2]
    noncontiguous_result = torch.view_as_real(noncontiguous)
    assert noncontiguous_result.shape == (*noncontiguous.shape, 2)
    assert noncontiguous_result.stride() == tuple(
        2 * stride for stride in noncontiguous.stride()
    ) + (1,)
    assert (
        noncontiguous_result.untyped_storage().data_ptr()
        == tensor.untyped_storage().data_ptr()
    )


def test_view_as_complex_validates_input_layout(dgx_setup_teardown):
    with pytest.raises(RuntimeError, match="only supported for half, float and double"):
        torch.view_as_complex(torch.empty((2, 2), dtype=torch.int32, device="dgx"))

    with pytest.raises(RuntimeError, match="one or more dimensions"):
        torch.view_as_complex(torch.empty((), device="dgx"))

    with pytest.raises(RuntimeError, match="last dimension of size 2"):
        torch.view_as_complex(torch.empty((2, 3), device="dgx"))

    with pytest.raises(RuntimeError, match="last dimension with stride 1"):
        torch.view_as_complex(torch.empty((2, 2, 4), device="dgx")[..., ::2])

    with pytest.raises(RuntimeError, match="stride divisible by 2"):
        torch.view_as_complex(torch.empty_strided((2, 2), (3, 1), device="dgx"))

    base = torch.empty(5, device="dgx")
    with pytest.raises(RuntimeError, match="storage_offset divisible by 2"):
        torch.view_as_complex(base.as_strided((2, 2), (2, 1), 1))


def test_view_as_real_requires_complex_input(dgx_setup_teardown):
    tensor = torch.empty((4, 2), device="dgx")

    with pytest.raises(RuntimeError, match="only supported for complex tensors"):
        torch.view_as_real(tensor)


def test_untyped_storage_resize_preserves_pending_dgx_data(dgx_setup_teardown):
    tensor = torch.empty(1_000_003, dtype=torch.int64, device="dgx")
    tensor.fill_(17)
    storage = tensor.untyped_storage()
    original_nbytes = storage.nbytes()

    # Resize immediately after asynchronous DGX work, without a CPU copy or
    # explicit synchronization masking stream-ordering bugs.
    storage.resize_(2 * original_nbytes)

    assert storage.nbytes() == 2 * original_nbytes
    assert torch.equal(tensor.cpu(), torch.full(tensor.shape, 17, dtype=tensor.dtype))


def test_untyped_storage_resize_to_zero(dgx_setup_teardown):
    storage = torch.empty(1, device="dgx").untyped_storage()

    storage.resize_(0)
    assert storage.nbytes() == 0


@pytest.mark.parametrize("dtype", datatypes, ids=str)
def test_arange(dgx_setup_teardown, dtype):
    tensor = torch.arange(-2, 17, 2.4, dtype=dtype, device="dgx")
    expected = torch.arange(-2, 17, 2.4, dtype=dtype, device="cpu")

    torch.testing.assert_close(tensor.to("cpu"), expected)


@pytest.mark.parametrize(
    "dtype",
    datatypes + [torch.complex32, torch.complex64, torch.complex128],
    ids=str,
)
@pytest.mark.parametrize(
    "shape_a, shape_b, dim",
    [
        ((10, 3), (5, 3), 0),
        ((5, 3), (5, 6), 1),
        ((10, 3, 1), (10, 3, 4), -1),
        ((8,), (7,), 0),
        ((5, 3, 7), (9, 3, 7), 0),
        ((3, 5, 7), (3, 9, 7), 1),
        ((1, 17, 3), (1, 17, 1), -1),
        ((3, 17, 3), (3, 17, 1), -1),
        ((5, 1, 3), (5, 1, 1), 2),
        ((2, 3, 17, 3), (2, 3, 17, 1), -1),
    ],
)
@pytest.mark.parametrize("layout", ["contiguous", "strided", "expanded", "transposed"])
def test_cat(dgx_setup_teardown, dtype, shape_a, shape_b, dim, layout):
    cpu_inputs = []
    inputs = []
    for index, shape in enumerate((shape_a, shape_b)):
        input_shape = shape
        if layout == "strided":
            input_shape = (*shape[:-1], shape[-1] * (index + 2))
        elif layout == "expanded":
            input_shape = (1, *shape[1:])
        elif layout == "transposed":
            input_shape = (*shape[1:], shape[0])

        if dtype.is_complex:
            real_dtype = {
                torch.complex32: torch.float16,
                torch.complex64: torch.float32,
                torch.complex128: torch.float64,
            }[dtype]
            components = torch.randn((*input_shape, 2), dtype=real_dtype)
            cpu_tensor = torch.view_as_complex(components)
            tensor = torch.view_as_complex(components.to("dgx"))
        else:
            if dtype.is_floating_point:
                cpu_tensor = torch.randn(input_shape, dtype=dtype)
            else:
                cpu_tensor = torch.randint(-10, 10, input_shape, dtype=dtype)
            tensor = cpu_tensor.to("dgx")

        if layout == "strided":
            cpu_tensor = cpu_tensor[..., index :: index + 2]
            tensor = tensor[..., index :: index + 2]
        elif layout == "expanded":
            cpu_tensor = cpu_tensor.expand(shape)
            tensor = tensor.expand(shape)
        elif layout == "transposed":
            cpu_tensor = cpu_tensor.movedim(-1, 0)
            tensor = tensor.movedim(-1, 0)
        cpu_inputs.append(cpu_tensor)
        inputs.append(tensor)

    expected = torch.cat(cpu_inputs, dim=dim)
    results = [torch.cat(inputs, dim=dim)]

    # Include the one-dimensional empty tensor accepted by cat for any rank.
    inputs.insert(1, torch.empty(0, dtype=dtype, device="dgx"))
    results.append(torch.concat(inputs, dim=dim))

    out = torch.empty(expected.shape, dtype=dtype, device="dgx")
    assert torch.cat(inputs, dim=dim, out=out) is out
    results.append(out)

    expected_values = torch.view_as_real(expected) if dtype.is_complex else expected
    for result in results:
        assert result.dtype == expected.dtype
        assert result.device.type == "dgx"
        result_values = torch.view_as_real(result) if dtype.is_complex else result
        torch.testing.assert_close(result_values.cpu(), expected_values, rtol=0, atol=0)


@pytest.mark.parametrize("dim", [-3, 0, 3])
def test_cat_all_empty(dgx_setup_teardown, dim):
    # PyTorch accepts arbitrary dimensions when every input is a 1-D empty tensor.
    inputs = [torch.empty(0, device="dgx"), torch.empty(0, device="dgx")]
    actual = torch.cat(inputs, dim=dim)
    expected = torch.cat([torch.empty(0), torch.empty(0)], dim=dim)
    torch.testing.assert_close(actual.cpu(), expected)


@pytest.mark.parametrize("batch_shape", [(), (1,), (3,)])
def test_cat_promoted_uneven_chunks(dgx_setup_teardown, batch_shape):
    a = torch.arange(torch.Size((*batch_shape, 5, 5)).numel(), dtype=torch.int32).reshape(*batch_shape, 5, 5)
    b = torch.arange(torch.Size((*batch_shape, 5, 3)).numel(), dtype=torch.int64).reshape(*batch_shape, 5, 3)
    actual = torch.cat([a.to("dgx"), b.to("dgx")], dim=-1)
    torch.testing.assert_close(actual.cpu(), torch.cat([a, b], dim=-1))


@pytest.mark.parametrize(
    "cameras, count",
    [(1, 0), (1, 1), (1, 17), (2, 17), (3, 17), (5, 17), (1, 262_147), (3, 262_147)],
)
def test_cat_camera_features_backward(dgx_setup_teardown, cameras, count):
    # The shared-camera and multi-camera paths concatenate RGB and depth while
    # retaining N as the distributed row dimension, including uneven/empty shards.
    cpu_inputs = [
        (torch.arange(cameras * count * width).reshape(cameras, count, width) % 31)
        .float()
        .requires_grad_()
        for width in (3, 1)
    ]
    inputs = [tensor.detach().to("dgx").requires_grad_() for tensor in cpu_inputs]
    expected = torch.cat(cpu_inputs, dim=-1)
    weights = (torch.arange(expected.numel()).reshape(expected.shape) % 13).float()
    weights_dgx = weights.to("dgx")
    actual = torch.cat(inputs, dim=-1)

    # Consume the values and exercise cat backward before any CPU read.
    (actual * actual).backward(weights_dgx)
    (expected * expected).backward(weights)

    assert actual.is_contiguous()
    torch.testing.assert_close(actual.cpu(), expected, rtol=0, atol=0)
    for tensor, cpu_tensor in zip(inputs, cpu_inputs):
        torch.testing.assert_close(tensor.grad.cpu(), cpu_tensor.grad, rtol=0, atol=0)


@pytest.mark.parametrize("lengths", [(1, 0, 2), (2, 19, 1), (19, 1, 2), (262_147, 0, 17)])
@pytest.mark.parametrize("width", [1, 3])
def test_cat_append_backward(dgx_setup_teardown, lengths, width):
    cpu_inputs = [
        (torch.arange(length * width).reshape(length, width) % 31).float().requires_grad_()
        for length in lengths
    ]
    inputs = [tensor.detach().to("dgx").requires_grad_() for tensor in cpu_inputs]
    expected = torch.cat(cpu_inputs, dim=0)
    weights = (torch.arange(expected.numel()).reshape(expected.shape) % 13).float()
    weights_dgx = weights.to("dgx")
    actual = torch.cat(inputs, dim=0)

    (actual * actual).backward(weights_dgx)
    (expected * expected).backward(weights)

    torch.testing.assert_close(actual.cpu(), expected, rtol=0, atol=0)
    for tensor, cpu_tensor in zip(inputs, cpu_inputs):
        torch.testing.assert_close(tensor.grad.cpu(), cpu_tensor.grad, rtol=0, atol=0)


@pytest.mark.parametrize("dtype", datatypes, ids=str)
def test_eye(dgx_setup_teardown, dtype):
    a = torch.eye(3, dtype=dtype, device="dgx")
    b = torch.eye(4, 2, dtype=dtype, device="dgx")

    assert torch.equal(a.to("cpu"), torch.eye(3, dtype=dtype))
    assert torch.equal(b.to("cpu"), torch.eye(4, 2, dtype=dtype))


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64], ids=str)
@pytest.mark.parametrize("shape", [(3, 3), (4, 4), (5, 3, 3), (2, 3, 4, 4)])
def test_inverse(dgx_setup_teardown, dtype, shape):
    matrix_size = shape[-1]
    matrix = torch.arange(1, matrix_size * matrix_size + 1, dtype=dtype).reshape(
        matrix_size, matrix_size
    )
    matrix = matrix @ matrix.mT + matrix_size * torch.eye(matrix_size, dtype=dtype)
    A_cpu = matrix.expand(shape).clone()
    A = A_cpu.to("dgx")
    expected = torch.linalg.inv(A_cpu)

    for inverse_op in (torch.inverse, torch.linalg.inv):
        B = inverse_op(A)
        assert B.shape == A.shape
        torch.testing.assert_close(B.to("cpu"), expected)

    B, info = torch.linalg.inv_ex(A)
    assert B.shape == A.shape
    assert info.shape == A.shape[:-2]
    torch.testing.assert_close(B.to("cpu"), expected)
    assert torch.count_nonzero(info.to("cpu")) == 0
