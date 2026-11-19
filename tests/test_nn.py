# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
import torch
import pytest

datatypes = [torch.float32, torch.float64]


@pytest.mark.parametrize("dtype", datatypes, ids=str)
def test_sigmoid(dgx_setup_teardown, dtype):
    module = torch.nn.Sigmoid()
    input = torch.randn(3, 5, requires_grad=True)
    input.retain_grad()
    target = torch.randn(3, 5)
    loss = torch.nn.MSELoss()
    predictions = module(input)
    output = loss(predictions, target)
    output.backward()
    reference_gradient = input.grad.detach().clone()

    module = module.to("dgx")
    input = input.to("dgx")
    input.retain_grad()
    target = target.to("dgx")
    loss = loss.to("dgx")
    predictions = module(input)
    output = loss(predictions, target)
    output.backward()
    test_gradient = input.grad.detach().clone()

    torch.testing.assert_close(test_gradient, reference_gradient.to("dgx"))


@pytest.mark.parametrize("dtype", datatypes, ids=str)
def test_embedding(dgx_setup_teardown, dtype):
    num_embeddings = 193
    embedding_dim = 9
    embedding = torch.nn.Embedding(num_embeddings, embedding_dim, dtype=dtype)
    input = torch.tensor([(7 * i + 3) % 23 for i in range(37)])
    predictions = embedding(input)
    target = torch.arange(input.numel() * embedding_dim, dtype=dtype).reshape(
        input.numel(), embedding_dim
    )
    target /= target.numel()
    loss = torch.nn.MSELoss()
    output = loss(predictions, target)
    output.backward()
    reference_gradient = embedding.weight.grad.detach().clone()

    embedding = embedding.to("dgx")
    embedding.weight.grad.zero_()
    input = input.to("dgx")
    target = target.to("dgx")
    loss = loss.to("dgx")
    predictions = embedding(input)
    output = loss(predictions, target)
    output.backward()
    test_gradient = embedding.weight.grad.detach().clone()

    torch.testing.assert_close(test_gradient, reference_gradient.to("dgx"))


@pytest.mark.parametrize("dtype", datatypes, ids=str)
def test_conv2d(dgx_setup_teardown, dtype):
    module = torch.nn.Conv2d(3, 128, 3, padding=1)
    input = torch.randn(1, 3, 32, 32)
    reference = module(input)

    module = module.to("dgx")
    input = input.to("dgx")
    output = module(input)

    torch.testing.assert_close(output, reference.to("dgx"), rtol=1e-3, atol=1e-3)


@pytest.mark.parametrize(
    "input_shape,weight_shape,stride,padding,dilation",
    [
        ((1, 3, 17, 19), (5, 3, 3, 3), (2, 1), (1, 1), (1, 1)),
        ((1, 2, 18, 17), (4, 2, 3, 3), (2, 2), (2, 1), (2, 1)),
        ((4, 3, 11, 13), (6, 3, 3, 2), (1, 2), (1, 0), (1, 1)),
    ],
    ids=["height-shard-uneven", "height-shard-dilated", "batch-shard"],
)
def test_conv2d_sharding(
    dgx_setup_teardown, input_shape, weight_shape, stride, padding, dilation
):
    input = torch.randn(input_shape)
    weight = torch.randn(weight_shape)
    reference = torch.nn.functional.conv2d(
        input, weight, stride=stride, padding=padding, dilation=dilation
    )

    output = torch.nn.functional.conv2d(
        input.to("dgx"),
        weight.to("dgx"),
        stride=stride,
        padding=padding,
        dilation=dilation,
    )

    torch.testing.assert_close(output, reference.to("dgx"), rtol=1e-4, atol=1e-5)


def test_conv2d_all_padding_height_shard(dgx_setup_teardown):
    input = torch.randn(1, 2, 1, 5)
    weight = torch.randn(3, 2, 1, 1)
    bias = torch.randn(3)
    reference = torch.nn.functional.conv2d(input, weight, bias, padding=(12, 2))

    output = torch.nn.functional.conv2d(
        input.to("dgx"), weight.to("dgx"), bias.to("dgx"), padding=(12, 2)
    )

    torch.testing.assert_close(output, reference.to("dgx"), rtol=1e-4, atol=1e-5)


@pytest.mark.parametrize("batch", [1, 5])
@pytest.mark.parametrize(
    "dtype,memory_format",
    [
        (torch.float16, torch.contiguous_format),
        (torch.bfloat16, torch.contiguous_format),
        (torch.float16, torch.channels_last),
        (torch.bfloat16, torch.channels_last),
        (torch.float32, torch.channels_last),
        (torch.float64, torch.channels_last),
    ],
    ids=[
        "float16",
        "bfloat16",
        "float16-nhwc",
        "bfloat16-nhwc",
        "float32-nhwc",
        "float64-nhwc",
    ],
)
def test_conv2d_bias(dgx_setup_teardown, batch, dtype, memory_format):
    source = torch.randn(batch, 3, 13, 17, dtype=dtype).contiguous(
        memory_format=memory_format
    )
    weight = torch.randn(5, 3, 3, 3, dtype=dtype).contiguous(
        memory_format=memory_format
    )
    bias_storage = torch.randn(10, dtype=dtype)
    bias = bias_storage[1::2]
    reference_dtype = torch.float64 if dtype == torch.float64 else torch.float32
    reference = torch.nn.functional.conv2d(
        source.to(reference_dtype), weight.to(reference_dtype), padding=1
    )
    reference = reference.to(dtype) + bias.reshape(1, -1, 1, 1)
    x, w, b = source.to("dgx"), weight.to("dgx"), bias_storage.to("dgx")[1::2]
    memory_format = torch.contiguous_format if dtype == torch.float64 else memory_format
    tolerance = {
        torch.float16: 1e-2,
        torch.bfloat16: 5e-2,
        torch.float32: 1e-4,
        torch.float64: 1e-10,
    }[dtype]
    for _ in range(2):
        output = torch.nn.functional.conv2d(x, w, b, padding=1)
        assert output.is_contiguous(memory_format=memory_format)
        torch.testing.assert_close(
            output.cpu(), reference, rtol=tolerance, atol=tolerance
        )


@pytest.mark.parametrize("dtype", datatypes, ids=str)
def test_upsampling_nearest2d(dgx_setup_teardown, dtype):
    module = torch.nn.UpsamplingNearest2d(scale_factor=2)
    input = torch.randn(1, 256, 58, 65)
    reference = module(input)

    module = module.to("dgx")
    input = input.to("dgx")
    output = module(input)

    torch.testing.assert_close(output, reference.to("dgx"))
