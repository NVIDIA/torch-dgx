# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
import pytest
import torch


DTYPES = [torch.float32, torch.float64, torch.float16, torch.bfloat16]


def inputs(dtype=torch.float32):
    # Cross the metadata batch boundary; include empty tensors, scalars, partial
    # blocks, and a tensor large enough for several blocks on each GPU.
    shapes = [(), (0,), (1,), (17,), (4099,), (2, 7)] * 6 + [(131101,)]
    return [torch.rand(shape, dtype=dtype) + 0.5 for shape in shapes]


def on_dgx(tensors):
    return [t.to("dgx") for t in tensors]


def check(actual, expected):
    assert len(actual) == len(expected)
    for a, e in zip(actual, expected):
        torch.testing.assert_close(a.cpu(), e.cpu())


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("inplace", [False, True])
@pytest.mark.parametrize("overload", ["Scalar", "ScalarList", "List"])
@pytest.mark.parametrize("name", ["add", "sub", "mul", "div", "maximum"])
def test_foreach_binary(dgx_setup_teardown, dtype, inplace, overload, name):
    reference = inputs(dtype)
    actual = on_dgx(reference)
    versions = [t._version for t in actual]
    other = (
        inputs(dtype)
        if overload == "List"
        else (
            [0.5 + i / 100 for i in range(len(reference))]
            if overload == "ScalarList"
            else 0.75
        )
    )
    dgx_other = on_dgx(other) if overload == "List" else other
    # CUDA foreach uses FP32 arithmetic for reduced-precision tensors. CPU's
    # per-tensor subtraction can round scalars/intermediates before subtracting.
    if dtype in (torch.float16, torch.bfloat16):
        reference = [t.cuda() for t in reference]
        if overload == "List":
            other = [t.cuda() for t in other]
    kwargs = {"alpha": 0.375} if overload == "List" and name in ("add", "sub") else {}
    op = getattr(torch, f"_foreach_{name}{'_' if inplace else ''}")
    expected = op(reference, other, **kwargs)
    result = op(actual, dgx_other, **kwargs)
    if inplace:
        check(actual, reference)
        assert [t._version for t in actual] == [v + 1 for v in versions]
    else:
        check(result, expected)
        check(actual, reference)
        assert [t._version for t in actual] == versions


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("inplace", [False, True])
@pytest.mark.parametrize("name", ["sqrt", "reciprocal", "neg"])
def test_foreach_unary(dgx_setup_teardown, dtype, inplace, name):
    reference = inputs(dtype)
    actual = on_dgx(reference)
    op = getattr(torch, f"_foreach_{name}{'_' if inplace else ''}")
    expected = op(reference)
    result = op(actual)
    check(actual if inplace else result, reference if inplace else expected)


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("inplace", [False, True])
@pytest.mark.parametrize("overload", ["Scalar", "ScalarList", "Tensor"])
@pytest.mark.parametrize("name", ["addcmul", "addcdiv"])
def test_foreach_pointwise(dgx_setup_teardown, dtype, inplace, overload, name):
    reference, a, b = inputs(dtype), inputs(dtype), inputs(dtype)
    actual, da, db = on_dgx(reference), on_dgx(a), on_dgx(b)
    value = [0.125 + i / 100 for i in range(len(reference))]
    if overload == "Scalar":
        value = 0.375
    elif overload == "Tensor":
        value = torch.tensor(value, dtype=torch.float64)
    op = getattr(torch, f"_foreach_{name}{'_' if inplace else ''}")
    expected = op(reference, a, b, value)
    result = op(actual, da, db, value)
    check(actual if inplace else result, reference if inplace else expected)


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("inplace", [False, True])
@pytest.mark.parametrize("overload", ["Scalar", "ScalarList", "List"])
def test_foreach_lerp(dgx_setup_teardown, dtype, inplace, overload):
    reference, end = inputs(dtype), inputs(dtype)
    actual, dend = on_dgx(reference), on_dgx(end)
    weight = (
        inputs(dtype)
        if overload == "List"
        else (
            [-0.75 + i / 10 for i in range(len(reference))]
            if overload == "ScalarList"
            else 0.375
        )
    )
    dweight = on_dgx(weight) if overload == "List" else weight
    op = torch._foreach_lerp_ if inplace else torch._foreach_lerp
    expected = op(reference, end, weight)
    result = op(actual, dend, dweight)
    check(actual if inplace else result, reference if inplace else expected)


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("inplace", [False, True])
@pytest.mark.parametrize("name", ["add", "mul", "div"])
def test_foreach_device_scalar(dgx_setup_teardown, dtype, inplace, name):
    reference = inputs(dtype)
    actual = on_dgx(reference)
    # Device computation produces the scalar without a host read before use.
    scalar = torch.tensor(0.75, dtype=dtype)
    dscalar = torch._foreach_add([scalar.to("dgx")], 0.25)[0]
    kwargs = {"alpha": 0.375} if name == "add" else {}
    op = getattr(torch, f"_foreach_{name}{'_' if inplace else ''}")
    expected = op(reference, scalar + 0.25, **kwargs)
    result = op(actual, dscalar, **kwargs)
    check(actual if inplace else result, reference if inplace else expected)


@pytest.mark.parametrize("name", ["add", "mul", "lerp", "addcmul"])
@pytest.mark.parametrize("inplace", [False, True])
def test_foreach_dense_transpose(dgx_setup_teardown, name, inplace):
    a, b, c = [[torch.randn(19, 23).t() for _ in range(3)] for _ in range(3)]
    da, db, dc = on_dgx(a), on_dgx(b), on_dgx(c)
    args = (
        (b, c, 0.375) if name == "addcmul" else ((b, 0.375) if name == "lerp" else (b,))
    )
    dargs = (
        (db, dc, 0.375)
        if name == "addcmul"
        else ((db, 0.375) if name == "lerp" else (db,))
    )
    op = getattr(torch, f"_foreach_{name}{'_' if inplace else ''}")
    expected, actual = op(a, *args), op(da, *dargs)
    expected, actual = (a, da) if inplace else (expected, actual)
    check(actual, expected)
    assert [t.stride() for t in actual] == [t.stride() for t in expected]


def test_foreach_fallbacks(dgx_setup_teardown):
    # Non-dense slices, broadcasting, mixed dtypes, and integer promotion retain
    # the semantics of the ordinary operators through the native slow path.
    base = torch.randn(11, 13)
    dbase = base.to("dgx")
    torch._foreach_add_([dbase[:, ::2]], 0.75)
    torch._foreach_add_([base[:, ::2]], 0.75)
    torch.testing.assert_close(dbase.cpu(), base)
    a = [torch.randn(3, 7), torch.randn(5, dtype=torch.float64)]
    b = [torch.randn(7), torch.randn(5, dtype=torch.float64)]
    check(torch._foreach_mul(on_dgx(a), on_dgx(b)), torch._foreach_mul(a, b))
    ints = [torch.arange(1, 8)]
    check(torch._foreach_div(on_dgx(ints), 2), torch._foreach_div(ints, 2))


def test_foreach_aliases(dgx_setup_teardown):
    a, b = torch.randn(8193), torch.randn(8193)
    da, db = a.to("dgx"), b.to("dgx")
    torch._foreach_add_([da, da], 0.5)
    torch._foreach_add_([a, a], 0.5)
    check([da], [a])
    torch._foreach_add_([da, db], [db, da])
    torch._foreach_add_([a, b], [b, a])
    check([da, db], [a, b])
    # Exact aliases within each entry are safe for the batched path.
    torch._foreach_addcmul_([da, db], [da, db], [da, db], 0.25)
    torch._foreach_addcmul_([a, b], [a, b], [a, b], 0.25)
    check([da, db], [a, b])
    # Reading a scalar from a destination must finish before the next update.
    scalar = torch.tensor(0.75)
    dscalar = scalar.to("dgx")
    torch._foreach_mul_([dscalar, db], dscalar)
    torch._foreach_mul_([scalar, b], scalar)
    check([dscalar, db], [scalar, b])


def test_foreach_invalid_arguments(dgx_setup_teardown):
    a = torch.ones(7, device="dgx")
    with pytest.raises(RuntimeError, match="same number"):
        torch._foreach_add([a, a], [0.5])
    with pytest.raises(RuntimeError, match="same number"):
        torch._foreach_add([a], [a, a])
    with pytest.raises(RuntimeError):
        torch._foreach_sub([a], True)
    with pytest.raises(RuntimeError):
        torch._foreach_mul([a], torch.ones(2, device="dgx"))
    with pytest.raises(RuntimeError):
        torch._foreach_add_([a.expand(2, 7)], 1)


def test_foreach_empty_and_nonfinite(dgx_setup_teardown):
    empty = [torch.empty(0, 7, device="dgx") for _ in range(33)]
    versions = [t._version for t in empty]
    torch._foreach_add_(empty, 1)
    assert [t._version for t in empty] == [v + 1 for v in versions]
    assert all(t.shape == (0, 7) for t in torch._foreach_sqrt(empty))
    a = [torch.tensor([float("nan"), float("inf"), -float("inf"), 0.0, -0.0])]
    b = [torch.tensor([1.0, float("nan"), -float("inf"), 0.0, 0.0])]
    for op in (torch._foreach_maximum, torch._foreach_div):
        torch.testing.assert_close(
            op(on_dgx(a), on_dgx(b))[0].cpu(), op(a, b)[0], equal_nan=True
        )


@pytest.mark.parametrize("name", ["mul", "div", "lerp", "addcmul", "addcdiv", "sqrt"])
def test_foreach_autograd(dgx_setup_teardown, name):
    cpu = [
        [(torch.rand(17, 23) + 0.5).requires_grad_() for _ in range(3)]
        for _ in range(3)
    ]
    dgx = [[t.detach().to("dgx").requires_grad_() for t in group] for group in cpu]

    def evaluate(groups):
        a, b, c = groups
        op = getattr(torch, f"_foreach_{name}")
        if name in ("addcmul", "addcdiv"):
            return op(a, b, c, 0.25)
        if name == "lerp":
            return op(a, b, 0.25)
        return op(a) if name == "sqrt" else op(a, b)

    expected, actual = evaluate(cpu), evaluate(dgx)
    grads = [torch.randn_like(t) for t in expected]
    torch.autograd.backward(expected, grads)
    torch.autograd.backward(actual, on_dgx(grads))
    check(actual, expected)
    for cgroup, dgroup in zip(cpu, dgx):
        for c, d in zip(cgroup, dgroup):
            if c.grad is not None:
                torch.testing.assert_close(d.grad.cpu(), c.grad)


def test_foreach_inplace_saved_version(dgx_setup_teardown):
    a = torch.ones(19, device="dgx", requires_grad=True)
    result = a.square()
    with torch.no_grad():
        torch._foreach_mul_([a], 2)
    with pytest.raises(RuntimeError, match="modified by an inplace operation"):
        result.sum().backward()


@pytest.mark.parametrize("optimizer", ["Adam", "AdamW", "SGD"])
def test_foreach_optimizer(dgx_setup_teardown, optimizer):
    reference = [t.requires_grad_() for t in inputs()]
    actual = [t.detach().to("dgx").requires_grad_() for t in reference]
    options = dict(lr=0.01, weight_decay=0.1, maximize=True, foreach=True)
    if optimizer == "SGD":
        options.update(momentum=0.9, nesterov=True)
    else:
        options.update(amsgrad=True)
    cls = getattr(torch.optim, optimizer)
    cpu_optimizer, dgx_optimizer = cls(reference, **options), cls(actual, **options)
    for _ in range(3):
        for c, d in zip(reference, actual):
            c.grad = torch.randn_like(c)
            d.grad = c.grad.to("dgx")
        cpu_optimizer.step()
        dgx_optimizer.step()
    check(actual, reference)
    for c, d in zip(reference, actual):
        for key, value in cpu_optimizer.state[c].items():
            if isinstance(value, torch.Tensor):
                torch.testing.assert_close(dgx_optimizer.state[d][key].cpu(), value)


def test_foreach_batches_kernel_launches(dgx_setup_teardown):
    from torch.profiler import ProfilerActivity, profile

    tensors = [torch.ones(257, device="dgx") for _ in range(65)]
    torch._foreach_add_(tensors, 1)
    for device in range(torch.cuda.device_count()):
        torch.cuda.synchronize(device)
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
        torch._foreach_add_(tensors, 1)
        for device in range(torch.cuda.device_count()):
            torch.cuda.synchronize(device)
    kernels = [
        e
        for e in prof.events()
        if e.device_type != torch.autograd.DeviceType.CPU and "foreach_kernel" in e.name
    ]
    assert len(kernels) == 3 * torch.cuda.device_count()
    for tensor in tensors:
        torch.testing.assert_close(tensor.cpu(), torch.full((257,), 3.0))
