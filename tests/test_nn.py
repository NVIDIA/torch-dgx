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


@pytest.mark.parametrize("dtype", datatypes, ids=str)
def test_upsampling_nearest2d(dgx_setup_teardown, dtype):
    module = torch.nn.UpsamplingNearest2d(scale_factor=2)
    input = torch.randn(1, 256, 58, 65)
    reference = module(input)

    module = module.to("dgx")
    input = input.to("dgx")
    output = module(input)

    torch.testing.assert_close(output, reference.to("dgx"))
