# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
import pytest
import torch


def check_median(source, values, indices, dim=0, keepdim=False):
    values, indices = values.cpu(), indices.cpu()
    expected = source.median(dim=dim, keepdim=keepdim).values
    torch.testing.assert_close(values, expected, rtol=0, atol=0, equal_nan=True)
    assert indices.dtype == torch.int64
    assert indices.shape == values.shape
    if source.is_floating_point():
        bit_dtype = {2: torch.int16, 4: torch.int32, 8: torch.int64}[
            source.element_size()
        ]
        source = source.contiguous().view(bit_dtype)
        values = values.contiguous().view(bit_dtype)
    if source.ndim == 0:
        assert indices.item() == 0
        gathered = source
    else:
        assert torch.all((indices >= 0) & (indices < source.size(dim)))
        gathered = source.gather(dim, indices if keepdim else indices.unsqueeze(dim))
        if not keepdim:
            gathered = gathered.squeeze(dim)
    # Equal values may have different valid indices across CPU and GPU implementations.
    assert torch.equal(values, gathered)


@pytest.mark.parametrize(
    "dtype,views",
    [
        (torch.uint8, 7),
        (torch.int8, 33),
        (torch.int16, 8),
        (torch.int32, 257),
        (torch.int64, 32),
        (torch.float16, 5),
        (torch.float32, 33),
        (torch.float64, 17),
    ],
)
def test_median_views(dgx_setup_teardown, dtype, views):
    source = torch.randint(
        0 if dtype == torch.uint8 else -50, 50, (views, 1, 17, 19)
    ).to(dtype)
    if not source.is_floating_point():
        source[0, 0, 0, 0] = torch.iinfo(dtype).min
        source[-1, 0, 0, 0] = torch.iinfo(dtype).max
    values, indices = source.to("dgx").median(dim=0)
    check_median(source, values, indices)


@pytest.mark.parametrize(
    "shape,dim,keepdim",
    [
        ((), 0, False),
        ((), -1, True),
        ((65536,), 0, False),
        ((3, 1, 4), 1, True),
        ((0, 5), 1, False),
        ((3, 4, 35), -1, True),
    ],
)
def test_median_shapes(dgx_setup_teardown, shape, dim, keepdim):
    source = torch.randn(shape)
    values, indices = source.to("dgx").median(dim=dim, keepdim=keepdim)
    check_median(source, values, indices, dim, keepdim)


@pytest.mark.parametrize("dtype,views", [(torch.float32, 32), (torch.float64, 33)])
def test_median_special_values(dgx_setup_teardown, dtype, views):
    source = torch.randn(views, 8, dtype=dtype)
    source[0, 0] = float("nan")
    source[:, 1] = float("nan")
    source[:, 2] = 0.0
    source[::2, 2] = -0.0
    source[:, 3] = float("inf")
    source[0, 3] = -float("inf")
    source[:, 4] = -float("inf")
    source[0, 4] = float("inf")
    source[:, 5] = 3.0
    source[:, 6] = -2.0
    source[::2, 6] = 2.0
    if views > 32:
        # Reuse each block's shared storage across NaN, equal-key, and selected medians.
        source = source.repeat(1, 512 * torch.cuda.device_count() + 1)
    values, indices = source.to("dgx").median(dim=0)
    check_median(source, values, indices)


@pytest.mark.parametrize("layout", ["resize", "strided", "alias"])
def test_median_out(dgx_setup_teardown, layout):
    source = torch.randn(7, 9, 11)
    tensor = source.to("dgx").transpose(0, 1)[..., 1::2]
    source = source.transpose(0, 1)[..., 1::2]
    if layout == "resize":
        values = torch.empty(0, device="dgx")
        indices = torch.empty(0, dtype=torch.int64, device="dgx")
    elif layout == "strided":
        values = torch.empty(9, 1, 10, device="dgx")[..., ::2]
        indices = torch.empty(9, 1, 10, dtype=torch.int64, device="dgx")[..., 1::2]
    else:
        values = tensor[:, :1]
        indices = torch.empty(9, 1, 5, dtype=torch.int64, device="dgx")
    result = torch.median(tensor, dim=1, keepdim=True, out=(values, indices))
    assert result.values is values and result.indices is indices
    check_median(source, values, indices, dim=1, keepdim=True)


@pytest.mark.parametrize("shape,dim", [((0, 3), 0), ((2, 3), 2)])
def test_median_invalid_dim(dgx_setup_teardown, shape, dim):
    with pytest.raises(IndexError):
        torch.empty(shape, device="dgx").median(dim=dim)
