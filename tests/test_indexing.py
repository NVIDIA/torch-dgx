# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
from math import prod

import pytest
import torch

datatypes = [torch.float32, torch.float64]

scatter_cases = [
    pytest.param((11,), 0, id="1d-dim0"),
    pytest.param((5, 7), 1, id="2d-dim1"),
    pytest.param((3, 5, 7), 0, id="3d-dim0"),
    pytest.param((3, 5, 7), 1, id="3d-dim1"),
    pytest.param((3, 5, 7), 2, id="3d-dim2"),
    pytest.param((2, 3, 5, 7), 2, id="4d-dim2"),
]


@pytest.mark.parametrize("dtype", datatypes, ids=str)
def test_index_out_integer_indices(dgx_setup_teardown, dtype):
    tensor = torch.rand((17, 3), dtype=dtype, device="dgx")
    indices = torch.tensor([0, 2, 3, 11], dtype=torch.int64)
    subtensor = tensor[indices]
    assert torch.equal(subtensor.to("cpu"), tensor.to("cpu")[indices])

    tensor = torch.rand((13, 7, 2), dtype=dtype, device="dgx")
    indices = torch.tensor([0, 5, 3, 0, 11], dtype=torch.int64)
    subtensor = tensor[indices]
    assert torch.equal(subtensor.to("cpu"), tensor.to("cpu")[indices])


@pytest.mark.parametrize("index_dtype", [torch.int32, torch.int64], ids=str)
@pytest.mark.parametrize("count", [0, 1, 17, 262_147])
@pytest.mark.parametrize("trailing_shape", [(), (3,), (15, 3)])
def test_index_out_refinement_rows(
    dgx_setup_teardown, index_dtype, count, trailing_shape
):
    # One source map is shared by scalar parameters, means and wide SH/moment rows.
    # Reordering and duplication model both retained and newly created Gaussians.
    rows = 257
    storage_shape = (2 * rows + 1, *trailing_shape)
    storage = (torch.arange(prod(storage_shape)) % 97).float().reshape(storage_shape)
    source_cpu = storage[1::2]
    source = storage.to("dgx")[1::2]
    indices_cpu = ((torch.arange(count) * 37) % rows).to(index_dtype)
    indices = indices_cpu.to("dgx")

    actual = source[indices]
    consumed = actual + 1
    expected = source_cpu[indices_cpu]

    assert actual.is_contiguous()
    torch.testing.assert_close(actual.cpu(), expected, rtol=0, atol=0)
    torch.testing.assert_close(consumed.cpu(), expected + 1, rtol=0, atol=0)


@pytest.mark.parametrize("shape", [(17, 3), (5, 7, 3)])
@pytest.mark.parametrize("select_any", [False, True])
def test_index_out_refinement_mask(dgx_setup_teardown, shape, select_any):
    storage = (torch.arange(prod(shape)) % 31).float().reshape(shape)
    mask_shape = shape[:-1]
    mask = torch.arange(prod(mask_shape)).reshape(mask_shape) % 3 == 0
    if not select_any:
        mask.fill_(False)
    source = storage.to("dgx")
    dgx_mask = mask.to("dgx")

    actual = source[dgx_mask]
    consumed = actual + 1
    expected = storage[mask]

    torch.testing.assert_close(actual.cpu(), expected, rtol=0, atol=0)
    torch.testing.assert_close(consumed.cpu(), expected + 1, rtol=0, atol=0)


@pytest.mark.parametrize("dtype", datatypes, ids=str)
def test_boolean_index_elementwise(dgx_setup_teardown, dtype):
    expected_input = torch.arange(6, dtype=dtype).reshape(2, 3)
    mask = torch.tensor([[True, False, True], [False, True, False]])
    actual_input = expected_input.to("dgx")

    result = actual_input[mask.to("dgx")]

    assert result.device == actual_input.device
    assert result.ndim == 1
    assert torch.equal(result.cpu(), expected_input[mask])


def test_boolean_index_elementwise_strided(dgx_setup_teardown):
    storage = torch.arange(2 * 3 * 11, dtype=torch.float32).reshape(2, 3, 11)
    dgx_storage = storage.to("dgx")
    expected_input = storage[..., 1::2].transpose(0, -1)
    actual_input = dgx_storage[..., 1::2].transpose(0, -1)
    assert not actual_input.is_contiguous()
    assert actual_input.storage_offset() > 0
    mask = torch.arange(expected_input.numel()).reshape(expected_input.shape) % 3 != 0

    result = actual_input[mask.to("dgx")]
    consumed = result + 1

    assert result.device == actual_input.device
    assert result.ndim == 1
    assert torch.equal(result.cpu(), expected_input[mask])
    assert torch.equal(consumed.cpu(), expected_input[mask] + 1)


def test_boolean_index_backward(dgx_setup_teardown):
    expected_input = torch.arange(24, dtype=torch.float32).reshape(2, 3, 4)
    expected_input.requires_grad_()
    actual_input = expected_input.detach().to("dgx").requires_grad_()
    mask = expected_input > 10

    expected = expected_input[mask].sum()
    actual = actual_input[mask.to("dgx")].sum()
    expected.backward()
    actual.backward()

    torch.testing.assert_close(actual.cpu(), expected)
    torch.testing.assert_close(actual_input.grad.cpu(), expected_input.grad)


@pytest.mark.parametrize("dtype", datatypes, ids=str)
def test_boolean_index_noncontiguous_prefix_mask(dgx_setup_teardown, dtype):
    storage = torch.arange(3 * 7 * 9, dtype=dtype).reshape(3, 7, 9)
    dgx_storage = storage.to("dgx")
    tensor = dgx_storage[:, 1:6, 2:8].permute(1, 2, 0)
    expected_tensor = storage[:, 1:6, 2:8].permute(1, 2, 0)
    mask = (
        torch.arange(tensor.size(0) * tensor.size(1)).reshape(
            tensor.size(0), tensor.size(1)
        )
        % 3
        != 0
    )

    assert not tensor.is_contiguous()
    assert tensor.storage_offset() > 0
    assert torch.equal(tensor[mask.to("dgx")].to("cpu"), expected_tensor[mask])


@pytest.mark.parametrize("dtype", datatypes, ids=str)
def test_index_select(dgx_setup_teardown, dtype):
    tensor = torch.rand((9, 4), dtype=dtype, device="dgx")
    indices = torch.tensor([0, 1, 3], dtype=torch.int64, device="dgx")
    subtensor = torch.index_select(tensor, 1, indices)
    assert torch.equal(
        subtensor.to("cpu"), torch.index_select(tensor.to("cpu"), 1, indices.to("cpu"))
    )


def test_index_select_1d(dgx_setup_teardown):
    tensor = torch.tensor([7], dtype=torch.int64, device="dgx")
    indices = torch.tensor([0], dtype=torch.int64, device="dgx")

    actual = torch.index_select(tensor, 0, indices)

    assert torch.equal(actual.cpu(), torch.tensor([7], dtype=torch.int64))


def test_repeat_interleave_int32(dgx_setup_teardown):
    expected = torch.arange(5, dtype=torch.int32).repeat_interleave(3, 0)
    actual = torch.arange(5, dtype=torch.int32, device="dgx").repeat_interleave(3, 0)

    assert torch.equal(actual.cpu(), expected)


@pytest.mark.parametrize("dtype", [torch.int32, torch.int64], ids=str)
@pytest.mark.parametrize(
    "values", [[], [0, 0, 0], [2, 0, 3, 1], [1, 4, 2, 0, 3]], ids=str
)
def test_repeat_interleave_tensor_output_size(dgx_setup_teardown, dtype, values):
    repeats = torch.tensor(values, dtype=dtype)
    output_size = sum(values)

    actual = torch.ops.aten.repeat_interleave.Tensor(
        repeats.to("dgx"), output_size=output_size
    )

    assert torch.equal(
        actual.cpu(),
        torch.ops.aten.repeat_interleave.Tensor(repeats, output_size=output_size),
    )


@pytest.mark.parametrize("dtype", datatypes, ids=str)
def test_index_put_elementwise_mask(dgx_setup_teardown, dtype):
    expected = torch.arange(0, 10, 1.5, dtype=dtype)
    result = expected.to("dgx")
    mask = expected.ge(5.0)

    expected[mask] = 0.0
    result[mask.to("dgx")] = 0.0

    assert torch.equal(result.cpu(), expected)


@pytest.mark.parametrize(
    "dtype,mask_shape,selection,trailing_shape,accumulate",
    [
        pytest.param(
            torch.float32,
            (6,),
            "mixed",
            (2, 3),
            False,
            id="mixed-rank1-float32",
        ),
        pytest.param(
            torch.int32,
            (6,),
            "none",
            (2, 3),
            False,
            id="empty-rank1-int32",
        ),
        pytest.param(
            torch.float32,
            (6,),
            "all",
            (2, 3),
            False,
            id="all-rank1-float32",
        ),
        pytest.param(
            torch.float64,
            (2, 4, 5),
            "mixed",
            (3,),
            False,
            id="mixed-rank3-float64",
        ),
        pytest.param(
            torch.float32,
            (2, 3),
            "mixed",
            (5,),
            True,
            id="accumulate-rank2-float32",
        ),
    ],
)
def test_index_put_boolean_prefix_mask(
    dgx_setup_teardown, dtype, mask_shape, selection, trailing_shape, accumulate
):
    shape = (*mask_shape, *trailing_shape)
    expected = torch.arange(prod(shape), dtype=dtype).reshape(shape)
    result = expected.to("dgx")
    if selection == "mixed":
        mask = torch.arange(prod(mask_shape)).reshape(mask_shape) % 3 != 0
    else:
        mask = torch.full(mask_shape, selection == "all", dtype=torch.bool)
    num_selected = int(mask.sum().item())
    values = (
        torch.arange(num_selected * prod(trailing_shape), dtype=dtype).reshape(
            num_selected, *trailing_shape
        )
        + 100
    )

    expected.index_put_((mask,), values, accumulate=accumulate)
    result.index_put_((mask.to("dgx"),), values.to("dgx"), accumulate=accumulate)

    assert torch.equal(result.to("cpu"), expected)


def test_index_put_boolean_mask_accumulate_scalar(dgx_setup_teardown):
    expected = torch.arange(30, dtype=torch.float32).reshape(2, 3, 5)
    result = expected.to("dgx")
    mask = torch.arange(expected.numel()).reshape(expected.shape) % 3 != 0
    value = torch.tensor(2.5)

    expected.index_put_((mask,), value, accumulate=True)
    result.index_put_((mask.to("dgx"),), value, accumulate=True)

    assert torch.equal(result.cpu(), expected)


@pytest.mark.parametrize(
    "dtype,shape,indices_device",
    [
        pytest.param(torch.float32, (11,), "cpu", id="rank1-float32-cpu-indices"),
        pytest.param(torch.int32, (11, 3), "dgx", id="rank2-int32"),
        pytest.param(torch.float64, (11, 2, 4), "dgx", id="rank3-float64"),
    ],
)
def test_index_put_rows_tensor_values(dgx_setup_teardown, dtype, shape, indices_device):
    expected = torch.arange(prod(shape), dtype=dtype).reshape(shape)
    result = expected.to("dgx")
    indices = torch.tensor([8, 2, 5, 1, -1], dtype=torch.int64, device=indices_device)
    values = torch.full((len(indices), *shape[1:]), -3.5, dtype=dtype)

    expected[indices.cpu()] = values
    result[indices] = values.to("dgx")

    assert torch.equal(result.cpu(), expected)


def test_index_put_rows_python_scalar(dgx_setup_teardown):
    expected = torch.arange(11, dtype=torch.float32)
    result = expected.to("dgx")
    indices = torch.tensor([8, 2, 5, 1, -1], dtype=torch.int64, device="dgx")

    expected[indices.cpu()] = -2.5
    result[indices] = -2.5

    assert torch.equal(result.cpu(), expected)


def test_index_put_rows_cpu_scalar_tensor(dgx_setup_teardown):
    shape = (11, 3)
    expected = torch.arange(prod(shape), dtype=torch.float32).reshape(shape)
    result = expected.to("dgx")
    indices = torch.tensor([8, 2, 5, 1, -1], dtype=torch.int64, device="dgx")
    value = torch.log(torch.tensor(2.0))

    expected[indices.cpu()] = value
    result[indices] = value

    assert torch.equal(result.cpu(), expected)


def test_index_put_rows_dgx_scalar_tensor(dgx_setup_teardown):
    shape = (11, 2, 4)
    expected = torch.arange(prod(shape), dtype=torch.float32).reshape(shape)
    result = expected.to("dgx")
    indices = torch.tensor([8, 2, 5, 1, -1], dtype=torch.int64, device="dgx")
    expected_value = torch.tensor(-2.5)
    value = expected_value.to("dgx")

    expected[indices.cpu()] = expected_value
    result[indices] = value

    assert torch.equal(result.cpu(), expected)


@pytest.mark.parametrize("shape", [(16, 257), ((1 << 18) + 17, 3)])
@pytest.mark.parametrize("index_dtype", [torch.int32, torch.int64], ids=str)
def test_index_put_rows_duplicate_destinations_are_deterministic(
    dgx_setup_teardown, shape, index_dtype
):
    # The larger shape puts last-position entries on multiple GPUs.
    initial = torch.arange(prod(shape), dtype=torch.float32).reshape(shape)
    indices = torch.tensor([3, shape[0] - 1, 3, 3 - shape[0], -1], dtype=index_dtype)
    values = torch.arange(indices.numel() * shape[1], dtype=torch.float32).reshape(
        indices.numel(), shape[1]
    )
    expected = initial.clone()
    expected[3] = values[3]
    expected[-1] = values[4]

    for _ in range(20):
        result = initial.to("dgx")
        result[indices.to("dgx")] = values.to("dgx")
        assert torch.equal(result.to("cpu"), expected)


def test_index_put_rows_rejects_partial_values_overlap(dgx_setup_teardown):
    result = torch.arange(8, dtype=torch.int64, device="dgx")
    indices = torch.tensor([4, 5, 6, 7], dtype=torch.int64, device="dgx")

    with pytest.raises(RuntimeError, match="single memory location"):
        result.index_put_((indices,), result[:4])


def test_index_put_rows_rejects_scalar_values_overlap(dgx_setup_teardown):
    result = torch.arange(8, dtype=torch.int64, device="dgx")
    indices = torch.tensor([4, 5, 6, 7], dtype=torch.int64, device="dgx")

    with pytest.raises(RuntimeError, match="single memory location"):
        result.index_put_((indices,), result[0])


def test_index_put_rows_rejects_index_overlap(dgx_setup_teardown):
    result = torch.arange(8, dtype=torch.int64, device="dgx")
    values = torch.full((8,), -1, dtype=torch.int64, device="dgx")

    with pytest.raises(RuntimeError, match="single memory location"):
        result.index_put_((result,), values)


@pytest.mark.parametrize("dtype", datatypes, ids=str)
def test_index_add_(dgx_setup_teardown, dtype):
    tensor = torch.rand((47,), dtype=dtype, device="dgx")
    indices = torch.tensor([0, 5, 3, 11, 44, 43, 41, 9, 9, 42], dtype=torch.int64).to(
        "dgx"
    )
    source = torch.rand_like(indices, dtype=dtype)

    result = tensor.detach().clone()
    result.index_add_(0, indices, source, alpha=0.7)

    torch.testing.assert_close(
        result.to("cpu"),
        tensor.to("cpu").index_add_(0, indices.to("cpu"), source.to("cpu"), alpha=0.7),
    )


@pytest.mark.parametrize("dtype", datatypes, ids=str)
def test_index_add_repeated_indices(dgx_setup_teardown, dtype):
    # Each input shard updates the same bins; integer-valued floats make the
    # result exact regardless of the order of concurrent atomic additions.
    indices_cpu = torch.arange(65539, dtype=torch.int64) % 7
    source_cpu = (torch.arange(indices_cpu.numel()) % 4 + 1).to(dtype)
    indices, source = indices_cpu.to("dgx"), source_cpu.to("dgx")
    expected = torch.zeros(13, dtype=dtype)
    actual = expected.to("dgx")
    for _ in range(4):
        expected.index_add_(0, indices_cpu, source_cpu, alpha=-2)
        actual.index_add_(0, indices, source, alpha=-2)
    torch.testing.assert_close(actual.cpu(), expected, rtol=0, atol=0)


@pytest.mark.parametrize("dtype", datatypes, ids=str)
@pytest.mark.parametrize("dim", [-1, 0, 1])
def test_gather_2d(dgx_setup_teardown, dtype, dim):
    tensor = torch.randn(7, 11, dtype=dtype)
    index = torch.randint(0, tensor.size(dim), (7, 11), dtype=torch.int64)
    result = torch.gather(tensor.to("dgx"), dim, index.to("dgx"))
    assert torch.equal(result.to("cpu"), torch.gather(tensor, dim, index))


@pytest.mark.parametrize("dtype", datatypes, ids=str)
@pytest.mark.parametrize("dim", [-1, 0, 1, 2])
def test_gather_3d(dgx_setup_teardown, dtype, dim):
    tensor = torch.randn(4, 5, 6, dtype=dtype)
    index_shape = list(tensor.shape)
    index_shape[dim] = 3
    index = torch.randint(0, tensor.size(dim), index_shape, dtype=torch.int64)
    result = torch.gather(tensor.to("dgx"), dim, index.to("dgx"))
    assert torch.equal(result.to("cpu"), torch.gather(tensor, dim, index))


@pytest.mark.parametrize("dtype", datatypes, ids=str)
@pytest.mark.parametrize("grad_output_layout", ["contiguous", "zero-stride"])
def test_gather_expanded_index_forward_backward(
    dgx_setup_teardown, dtype, grad_output_layout
):
    input_cpu = torch.randn(2, 5, 24, dtype=dtype, requires_grad=True)
    index_base = torch.randint(0, input_cpu.size(1), (2, 7, 1))
    index_cpu = index_base.expand(2, 7, 24)
    input = input_cpu.detach().to("dgx").requires_grad_()
    index = index_base.to("dgx").expand(2, 7, 24)
    if grad_output_layout == "zero-stride":
        grad_output_base_cpu = torch.randn(2, 7, 1, dtype=dtype)
        grad_output_cpu = grad_output_base_cpu.expand(index.shape)
        grad_output = grad_output_base_cpu.to("dgx").expand(index.shape)
    else:
        grad_output_cpu = torch.randn(index.shape, dtype=dtype)
        grad_output = grad_output_cpu.to("dgx")

    expected = torch.gather(input_cpu, 1, index_cpu)
    result = torch.gather(input, 1, index)
    expected.backward(grad_output_cpu)
    result.backward(grad_output)

    assert 0 in index.stride()
    torch.testing.assert_close(result.cpu(), expected.detach())
    torch.testing.assert_close(input.grad.cpu(), input_cpu.grad)


@pytest.mark.parametrize("dtype", datatypes, ids=str)
@pytest.mark.parametrize("shape,dim", scatter_cases)
def test_scatter_add_out(dgx_setup_teardown, dtype, shape, dim):
    self_cpu = torch.randn(shape, dtype=dtype)
    index = torch.randint(0, self_cpu.size(dim), shape, dtype=torch.int64)
    source_cpu = torch.randn_like(self_cpu)
    out = torch.empty_like(self_cpu, device="dgx")

    returned = torch.scatter_add(
        self_cpu.to("dgx"), dim, index.to("dgx"), source_cpu.to("dgx"), out=out
    )

    assert returned is out
    torch.testing.assert_close(
        out.cpu(), torch.scatter_add(self_cpu, dim, index, source_cpu)
    )


@pytest.mark.parametrize("dtype", datatypes, ids=str)
@pytest.mark.parametrize("shape,dim", scatter_cases)
def test_scatter_value_out(dgx_setup_teardown, dtype, shape, dim):
    self_cpu = torch.randn(shape, dtype=dtype)
    index = torch.randint(0, self_cpu.size(dim), shape, dtype=torch.int64)
    out = torch.empty_like(self_cpu, device="dgx")

    returned = torch.scatter(self_cpu.to("dgx"), dim, index.to("dgx"), 0.0, out=out)

    assert returned is out
    assert torch.equal(out.cpu(), torch.scatter(self_cpu, dim, index, 0.0))


@pytest.mark.parametrize(
    "mask",
    [
        pytest.param([False, True, False, True, True, False], id="mixed"),
        pytest.param([False, False, False, False, False, False], id="none"),
        pytest.param([True, True, True, True, True, True], id="all"),
    ],
)
@pytest.mark.parametrize("dtype", datatypes, ids=str)
def test_masked_select_selection_cases(dgx_setup_teardown, dtype, mask):
    expected_input = torch.arange(len(mask), dtype=dtype)
    expected_mask = torch.tensor(mask, dtype=torch.bool)
    actual_input = expected_input.to("dgx")
    actual_mask = expected_mask.to("dgx")

    result = torch.masked_select(actual_input, actual_mask)

    assert result.device == actual_input.device
    assert result.ndim == 1
    assert torch.equal(result.cpu(), torch.masked_select(expected_input, expected_mask))


@pytest.mark.parametrize("dtype", datatypes, ids=str)
def test_masked_select_noncontiguous(dgx_setup_teardown, dtype):
    storage = torch.arange(2 * 3 * 4, dtype=dtype).reshape(2, 3, 4)
    mask_storage = torch.tensor(
        [
            [[False, True], [True, False], [False, True]],
            [[True, False], [False, True], [True, False]],
        ]
    )
    expected_input = storage[..., -1]
    expected_mask = mask_storage[..., -1]
    actual_input = storage.to("dgx")[..., -1]
    actual_mask = mask_storage.to("dgx")[..., -1]

    assert not actual_input.is_contiguous()
    assert not actual_mask.is_contiguous()
    assert actual_input.storage_offset() > 0
    assert actual_mask.storage_offset() > 0

    result = torch.masked_select(actual_input, actual_mask)

    assert result.device == actual_input.device
    assert result.ndim == 1
    assert torch.equal(result.cpu(), torch.masked_select(expected_input, expected_mask))


@pytest.mark.parametrize("dtype", datatypes, ids=str)
@pytest.mark.parametrize(
    "case", ["mask", "both"], ids=["mask-broadcast", "both-broadcast"]
)
def test_masked_select_broadcast(dgx_setup_teardown, dtype, case):
    if case == "mask":
        expected_input = torch.arange(2 * 3 * 4, dtype=dtype).reshape(2, 3, 4)
        expected_mask = torch.tensor(
            [[[True], [False], [True]], [[False], [True], [False]]]
        )
    else:
        expected_input = torch.tensor([[1.0, 2.0, 3.0]], dtype=dtype)
        expected_mask = torch.tensor([[True], [False], [True]])

    actual_input = expected_input.to("dgx")
    actual_mask = expected_mask.to("dgx")
    result = torch.masked_select(actual_input, actual_mask)

    assert result.device == actual_input.device
    assert result.ndim == 1
    assert torch.equal(result.cpu(), torch.masked_select(expected_input, expected_mask))


def test_masked_select_rejects_nonbroadcastable_prefix_mask(dgx_setup_teardown):
    tensor = torch.arange(2 * 3 * 4, dtype=torch.float32, device="dgx").reshape(2, 3, 4)
    mask = torch.ones((2, 3), dtype=torch.bool, device="dgx")

    with pytest.raises(RuntimeError):
        torch.masked_select(tensor, mask)


@pytest.mark.parametrize("dtype", datatypes, ids=str)
def test_masked_select_empty_broadcast(dgx_setup_teardown, dtype):
    tensor = torch.empty((0, 3), dtype=dtype, device="dgx")
    mask = torch.ones((1, 3), dtype=torch.bool, device="dgx")

    result = torch.masked_select(tensor, mask)

    assert result.device == tensor.device
    assert result.shape == (0,)
