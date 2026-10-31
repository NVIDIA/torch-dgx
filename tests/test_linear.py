# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
import pytest
import torch
from torch.library import opcheck

datatypes = [torch.float32, torch.float64]


def _matrix_with_layout(rows, columns, layout, dtype, offset=0):
    shape = (columns, rows) if layout == "transpose" else (rows, columns)
    if layout == "slice":
        shape = (rows, 2 * columns)
    storage = torch.randn(shape[0] * shape[1] + offset, dtype=dtype, device="dgx")
    matrix = storage[offset:].view(shape)
    if layout == "transpose":
        return matrix.t()
    if layout == "slice":
        matrix = matrix[:, ::2]
        assert not matrix.is_contiguous()
        assert not matrix.t().is_contiguous()
    return matrix


@pytest.mark.parametrize("dtype", datatypes, ids=str)
@pytest.mark.parametrize(
    "shape,bias_shape,alpha",
    [
        pytest.param((17, 13, 19), (), 0.7, id="scalar-bias"),
        pytest.param((17, 13, 19), (1,), 0.7, id="singleton-bias"),
        pytest.param((17, 13, 19), (19,), 0.7, id="vector-bias"),
        pytest.param((17, 13, 19), (1, 19), 0.7, id="row-bias"),
        pytest.param((17, 13, 19), (17, 1), 0.7, id="column-bias"),
        pytest.param((17, 13, 19), (17, 19), 0.7, id="matrix-bias"),
        pytest.param((0, 3, 5), (5,), 1.0, id="empty-rows"),
        pytest.param((3, 4, 0), (0,), 1.0, id="empty-columns"),
        pytest.param((3, 0, 5), (5,), 1.0, id="empty-reduction"),
        pytest.param((1, 3, 5), (5,), 1.0, id="single-row"),
        pytest.param((3, 1, 5), (5,), 1.0, id="single-reduction"),
        pytest.param((3, 4, 1), (1,), 1.0, id="single-column"),
    ],
)
@pytest.mark.parametrize("beta", [0.0, 1.0, -0.3])
def test_addmm_forward(dgx_setup_teardown, dtype, shape, bias_shape, alpha, beta):
    m, k, n = shape
    bias = torch.randn(bias_shape, dtype=dtype)
    if beta == 0:
        bias.fill_(float("nan"))
    mat1 = torch.randn((m, k), dtype=dtype)
    mat2 = torch.randn((k, n), dtype=dtype)
    expected = torch.addmm(bias, mat1, mat2, beta=beta, alpha=alpha)

    actual = torch.addmm(
        bias.to("dgx"), mat1.to("dgx"), mat2.to("dgx"), beta=beta, alpha=alpha
    )
    torch.testing.assert_close(actual.cpu(), expected)


@pytest.mark.parametrize("dtype", datatypes, ids=str)
@pytest.mark.parametrize(
    "shape,layouts,offset,alpha,beta",
    [
        ((17, 13, 19), ("contiguous", "contiguous", "vector"), 1, -0.7, 1.0),
        ((17, 13, 19), ("transpose", "contiguous", "vector"), 1, -0.7, 1.0),
        ((17, 13, 19), ("contiguous", "transpose", "vector"), 1, -0.7, 1.0),
        ((17, 13, 19), ("transpose", "transpose", "vector"), 1, -0.7, 1.0),
        ((17, 13, 19), ("contiguous", "contiguous", "slice"), 0, 1.0, 1.0),
        ((17, 13, 19), ("contiguous", "contiguous", "transpose"), 0, 1.0, 1.0),
        ((6, 4, 3), ("slice", "contiguous", "matrix"), 0, 0.7, 0.3),
        ((6, 4, 3), ("contiguous", "slice", "matrix"), 0, 0.7, 0.3),
        ((1, 13, 19), ("transpose", "contiguous", "vector"), 1, -0.7, 1.0),
    ],
    ids=[
        "offset",
        "offset-transpose-left",
        "offset-transpose-right",
        "offset-transpose-both",
        "slice-bias",
        "transpose-bias",
        "slice-left",
        "slice-right",
        "singleton-transpose-left",
    ],
)
def test_addmm_input_layouts(
    dgx_setup_teardown, dtype, shape, layouts, offset, alpha, beta
):
    m, k, n = shape
    mat1_layout, mat2_layout, bias_layout = layouts
    # Preserve misaligned pointers and uneven row shards in the transpose cases.
    mat1 = _matrix_with_layout(m, k, mat1_layout, dtype, offset)
    mat2 = _matrix_with_layout(k, n, mat2_layout, dtype, offset)
    if bias_layout == "vector":
        bias = _matrix_with_layout(1, n, "contiguous", dtype, offset).squeeze(0)
    elif bias_layout == "slice":
        bias = _matrix_with_layout(1, n, "slice", dtype).squeeze(0)
    else:
        bias = _matrix_with_layout(
            m, n, "transpose" if bias_layout == "transpose" else "contiguous", dtype
        )
    expected = torch.addmm(bias.cpu(), mat1.cpu(), mat2.cpu(), alpha=alpha, beta=beta)
    actual = torch.addmm(bias, mat1, mat2, alpha=alpha, beta=beta)
    torch.testing.assert_close(actual.cpu(), expected)


@pytest.mark.parametrize("dtype", datatypes, ids=str)
@pytest.mark.parametrize("layout", ["resize", "transpose", "strided", "inplace"])
@pytest.mark.parametrize("beta", [0.0, 1.0, -0.3])
def test_addmm_out_layout(dgx_setup_teardown, dtype, layout, beta):
    mat1 = torch.randn((17, 13), dtype=dtype, device="dgx")
    mat2 = torch.randn((13, 19), dtype=dtype, device="dgx")
    bias = torch.randn(
        (17, 19) if layout == "inplace" else (19,), dtype=dtype, device="dgx"
    )
    if layout == "resize":
        out = torch.empty(0, dtype=dtype, device="dgx")
    elif layout == "transpose":
        out = torch.empty((19, 17), dtype=dtype, device="dgx").t()
    elif layout == "strided":
        backing = torch.full((17, 38), -123.0, dtype=dtype, device="dgx")
        out = backing[:, ::2]
    else:
        out = bias
    expected = torch.addmm(bias.cpu(), mat1.cpu(), mat2.cpu(), beta=beta, alpha=0.7)
    strides = out.stride()
    pointer = out.data_ptr()

    returned = torch.addmm(bias, mat1, mat2, beta=beta, alpha=0.7, out=out)
    assert returned is out
    if layout != "resize":
        assert out.stride() == strides
        assert out.data_ptr() == pointer
    torch.testing.assert_close(out.cpu(), expected)
    if layout == "strided":
        torch.testing.assert_close(
            backing[:, 1::2].cpu(), torch.full((17, 19), -123.0, dtype=dtype)
        )


@pytest.mark.parametrize("dtype", datatypes, ids=str)
@pytest.mark.parametrize("layout", ["expanded-row", "strided-vector"])
@pytest.mark.parametrize("beta", [0.0, 1.0])
def test_addmm_rejects_aliased_bias(dgx_setup_teardown, dtype, layout, beta):
    out = torch.randn((17, 19), dtype=dtype, device="dgx")
    bias = out[:1].expand_as(out) if layout == "expanded-row" else out.flatten()[::17]
    mat1 = torch.randn((17, 13), dtype=dtype, device="dgx")
    mat2 = torch.randn((13, 19), dtype=dtype, device="dgx")
    original = out.cpu()

    with pytest.raises(RuntimeError, match="addmm: self and out must not alias"):
        torch.addmm(bias, mat1, mat2, beta=beta, out=out)
    torch.testing.assert_close(out.cpu(), original, rtol=0, atol=0)


@pytest.mark.parametrize("dtype", datatypes, ids=str)
@pytest.mark.parametrize("grad_layout", ["contiguous", "zero-stride"])
def test_addmm_bias_backward(dgx_setup_teardown, dtype, grad_layout):
    inputs = [
        torch.randn(shape, dtype=dtype, requires_grad=True)
        for shape in [(19,), (17, 13), (13, 19)]
    ]
    dgx_inputs = [tensor.detach().to("dgx").requires_grad_() for tensor in inputs]
    grad = torch.randn(
        (17, 19) if grad_layout == "contiguous" else (1, 19), dtype=dtype
    )
    dgx_grad = grad.to("dgx").expand(17, 19)
    grad = grad.expand(17, 19)
    torch.addmm(*inputs, alpha=0.7).backward(grad)
    torch.addmm(*dgx_inputs, alpha=0.7).backward(dgx_grad)
    for reference, actual in zip(inputs, dgx_inputs):
        torch.testing.assert_close(actual.grad.cpu(), reference.grad)


@pytest.mark.parametrize("precision", ["ieee", "tf32"])
@pytest.mark.parametrize("beta", [0.0, 1.0])
def test_addmm_float32_precision(dgx_setup_teardown, precision, beta):
    previous = torch.backends.cuda.matmul.fp32_precision
    try:
        torch.backends.cuda.matmul.fp32_precision = precision
        inputs = [torch.randn(shape) for shape in [(64,), (32, 64), (64, 64)]]
        expected = torch.addmm(*inputs, beta=beta)
        actual = torch.addmm(*(tensor.to("dgx") for tensor in inputs), beta=beta)
        if precision == "tf32":
            # TF32 rounds both operands to 10 mantissa bits. Bound the error
            # relative to |A| @ |B| so cancellation near zero stays well behaved.
            scale = (inputs[1].abs() @ inputs[2].abs()).clamp_min(1)
            torch.testing.assert_close(
                actual.cpu() / scale, expected / scale, rtol=0, atol=1e-3
            )
        else:
            torch.testing.assert_close(actual.cpu(), expected, rtol=1e-4, atol=1e-4)
    finally:
        torch.backends.cuda.matmul.fp32_precision = previous


@pytest.mark.parametrize("dtype", datatypes, ids=str)
def test_addmm(dgx_setup_teardown, dtype):
    in0 = torch.rand((3), dtype=dtype, device="dgx")
    in1 = torch.rand((6, 4), dtype=dtype, device="dgx")
    in2 = torch.rand((4, 3), dtype=dtype, device="dgx")

    opcheck(torch.ops.aten.addmm.default, (in0, in1, in2))

    out = torch.addmm(
        in0,
        in1,
        in2,
        alpha=0.7,
        beta=0.3,
    )
    ref = torch.addmm(in0.to("cpu"), in1.to("cpu"), in2.to("cpu"), alpha=0.7, beta=0.3)
    torch.testing.assert_close(out, ref.to("dgx"))

    in0 = torch.rand((6, 3), dtype=dtype, device="dgx")

    opcheck(torch.ops.aten.addmm.default, (in0, in1, in2), {"alpha": 0.7, "beta": 0.3})

    out = torch.addmm(
        in0,
        in1,
        in2,
        alpha=0.7,
        beta=0.3,
    ).to("cpu")
    ref = torch.addmm(in0.to("cpu"), in1.to("cpu"), in2.to("cpu"), alpha=0.7, beta=0.3)
    torch.testing.assert_close(out, ref)

    in1 = torch.rand((6, 4), dtype=dtype, device="dgx")
    in2 = torch.rand((3, 4), dtype=dtype, device="dgx").transpose(-2, -1)

    # opcheck(torch.ops.aten.addmm.default, (in0, in1, in2), {"alpha": 0.7, "beta": 0.3})

    out = torch.addmm(
        in0,
        in1,
        in2,
        alpha=0.7,
        beta=0.3,
    ).to("cpu")
    ref = torch.addmm(in0.to("cpu"), in1.to("cpu"), in2.to("cpu"), alpha=0.7, beta=0.3)
    torch.testing.assert_close(out, ref)

    in1 = torch.rand((4, 6), dtype=dtype, device="dgx").transpose(-2, -1)
    in2 = torch.rand((4, 3), dtype=dtype, device="dgx")

    # opcheck(torch.ops.aten.addmm.default, (in0, in1, in2), {"alpha": 0.7, "beta": 0.3})

    out = torch.addmm(
        in0,
        in1,
        in2,
        alpha=0.7,
        beta=0.3,
    ).to("cpu")
    ref = torch.addmm(in0.to("cpu"), in1.to("cpu"), in2.to("cpu"), alpha=0.7, beta=0.3)
    torch.testing.assert_close(out, ref)

    in1 = torch.rand((4, 6), dtype=dtype, device="dgx").transpose(-2, -1)
    in2 = torch.rand((3, 4), dtype=dtype, device="dgx").transpose(-2, -1)

    # opcheck(torch.ops.aten.addmm.default, (in0, in1, in2), {"alpha": 0.7, "beta": 0.3})

    out = torch.addmm(
        in0,
        in1,
        in2,
        alpha=0.7,
        beta=0.3,
    ).to("cpu")
    ref = torch.addmm(in0.to("cpu"), in1.to("cpu"), in2.to("cpu"), alpha=0.7, beta=0.3)
    torch.testing.assert_close(out, ref)


@pytest.mark.parametrize("dtype", datatypes, ids=str)
@pytest.mark.parametrize(
    "shape,layouts",
    [
        pytest.param((7, 5, 4), ("contiguous", "contiguous"), id="contiguous"),
        pytest.param((7, 5, 4), ("contiguous", "transpose"), id="transpose-right"),
        pytest.param((7, 5, 4), ("transpose", "contiguous"), id="transpose-left"),
        pytest.param((7, 5, 4), ("transpose", "transpose"), id="transpose-both"),
        pytest.param((7, 5, 4), ("slice", "contiguous"), id="slice-left"),
        pytest.param((7, 5, 4), ("contiguous", "slice"), id="slice-right"),
        pytest.param((11, 29, 13), ("transpose", "contiguous"), id="reduction-split"),
        pytest.param(
            (33, 5, 17), ("transpose", "contiguous"), id="weight-gradient-rows"
        ),
        pytest.param(
            (17, 5, 33), ("transpose", "contiguous"), id="weight-gradient-columns"
        ),
        pytest.param(
            (33, 25, 17), ("transpose", "contiguous"), id="weight-gradient-tokens"
        ),
        pytest.param((1, 7, 3), ("contiguous", "transpose"), id="empty-device-shard"),
        pytest.param((1, 7, 3), ("transpose", "contiguous"), id="singleton-stride"),
    ],
)
def test_mm(dgx_setup_teardown, dtype, shape, layouts):
    m, k, n = shape
    left = _matrix_with_layout(m, k, layouts[0], dtype)
    right = _matrix_with_layout(k, n, layouts[1], dtype)
    if layouts == ("contiguous", "contiguous"):
        opcheck(torch.ops.aten.mm.default, (left, right))
    result = torch.mm(left, right)
    reference = torch.mm(left.cpu(), right.cpu())
    torch.testing.assert_close(result.cpu(), reference)


@pytest.mark.parametrize("tokens", [257, 2049])
def test_residual_mlp_shards(dgx_setup_teardown, tokens):
    import copy

    reference = torch.nn.Sequential(
        torch.nn.LayerNorm(768, elementwise_affine=False),
        torch.nn.Linear(768, 3072),
        torch.nn.GELU(approximate="tanh"),
        torch.nn.Linear(3072, 768),
    )
    actual = copy.deepcopy(reference).to("dgx")
    x = torch.randn(1, tokens, 768, requires_grad=True)
    dgx_x = x.detach().to("dgx").requires_grad_()
    grad = torch.randn_like(x)
    expected = x + reference(x)
    result = dgx_x + actual(dgx_x)
    expected.backward(grad)
    result.backward(grad.to("dgx"))
    torch.testing.assert_close(result.cpu(), expected, rtol=2e-4, atol=2e-4)
    torch.testing.assert_close(dgx_x.grad.cpu(), x.grad, rtol=2e-4, atol=2e-4)
    for dgx_parameter, parameter in zip(actual.parameters(), reference.parameters()):
        torch.testing.assert_close(
            dgx_parameter.grad.cpu(), parameter.grad, rtol=3e-4, atol=3e-4
        )


@pytest.mark.parametrize("dtype", datatypes, ids=str)
def test_matmul_non_contiguous_matrix(dgx_setup_teardown, dtype):
    in0 = torch.rand((7, 10), dtype=dtype, device="dgx")[:, ::2]
    in1 = torch.rand((5, 4), dtype=dtype, device="dgx")

    out = torch.matmul(in0, in1)
    ref = torch.matmul(in0.to("cpu"), in1.to("cpu"))
    torch.testing.assert_close(out, ref.to("dgx"))

    in0 = torch.rand((7, 5), dtype=dtype, device="dgx")
    in1 = torch.rand((5, 8), dtype=dtype, device="dgx")[:, ::2]

    out = torch.matmul(in0, in1)
    ref = torch.matmul(in0.to("cpu"), in1.to("cpu"))
    torch.testing.assert_close(out, ref.to("dgx"))


@pytest.mark.parametrize("dtype", datatypes, ids=str)
def test_bmm(dgx_setup_teardown, dtype):
    in0 = torch.rand((2, 6, 4), dtype=dtype, device="dgx")
    in1 = torch.rand((2, 4, 3), dtype=dtype, device="dgx")

    out = torch.bmm(in0, in1)
    ref = torch.bmm(in0.to("cpu"), in1.to("cpu"))
    torch.testing.assert_close(out, ref.to("dgx"))

    in0 = torch.rand((2, 6, 4), dtype=dtype, device="dgx")
    in1 = torch.rand((2, 3, 4), dtype=dtype, device="dgx").transpose(-2, -1)

    out = torch.bmm(in0, in1)
    ref = torch.bmm(in0.to("cpu"), in1.to("cpu"))
    torch.testing.assert_close(out, ref.to("dgx"))

    in0 = torch.rand((2, 4, 6), dtype=dtype, device="dgx").transpose(-2, -1)
    in1 = torch.rand((2, 4, 3), dtype=dtype, device="dgx")

    out = torch.bmm(in0, in1)
    ref = torch.bmm(in0.to("cpu"), in1.to("cpu"))
    torch.testing.assert_close(out, ref.to("dgx"))

    in0 = torch.rand((2, 4, 6), dtype=dtype, device="dgx").transpose(-2, -1)
    in1 = torch.rand((2, 3, 4), dtype=dtype, device="dgx").transpose(-2, -1)

    out = torch.bmm(in0, in1)
    ref = torch.bmm(in0.to("cpu"), in1.to("cpu"))
    torch.testing.assert_close(out, ref.to("dgx"))

    in0 = torch.rand((6, 2, 4), dtype=dtype, device="dgx").transpose(0, 1)
    in1 = torch.rand((2, 3, 4), dtype=dtype, device="dgx").transpose(-2, -1)

    out = torch.bmm(in0, in1)
    ref = torch.bmm(in0.to("cpu"), in1.to("cpu"))
    torch.testing.assert_close(out, ref.to("dgx"))

    in0 = torch.rand((2, 6, 4), dtype=dtype, device="dgx")
    in1 = torch.rand((2, 4, 6), dtype=dtype, device="dgx")[:, :, ::2]
    assert not in1.is_contiguous()
    assert not in1.transpose(-2, -1).is_contiguous()

    out = torch.bmm(in0, in1)
    ref = torch.bmm(in0.to("cpu"), in1.to("cpu"))
    torch.testing.assert_close(out, ref.to("dgx"))

    in2 = torch.rand((1, 6, 4), dtype=dtype, device="dgx")
    in3 = torch.rand((1, 4, 3), dtype=dtype, device="dgx")

    out = torch.bmm(in2, in3)
    ref = torch.bmm(in2.to("cpu"), in3.to("cpu"))
    torch.testing.assert_close(out, ref.to("dgx"))
