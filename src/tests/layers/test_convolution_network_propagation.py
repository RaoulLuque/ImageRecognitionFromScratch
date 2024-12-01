import cupy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Any as NDArray

from src.add_ons.loss_function import LossFunction
from src.config import LEARNING_RATE, DEBUGGING
from src.add_ons.activation_function import ActivationFunction
from src.layers.convolution_2d_layer import Convolution2D
from src.layers.flatten_layer import FlattenLayer
from src.layers.fully_connected_layer import FCLayer
from src.utils.read_data import read_data, to_categorical


def setup_own_convolution_layer(batch_size: int = 16, number_of_channels: int = 16, number_of_filters: int = 32, height_input: int = 28, width_input: int = 28):
    return  Convolution2D(
        D_batch_size=batch_size,
        C_number_channels=number_of_channels,
        H_height_input=height_input,
        W_width_input=width_input,
        NF_number_of_filters=number_of_filters,
        HF_height_filter=5,
        WF_width_filter=5,
        S_stride=1,
        P_padding=2,
        optimizer=None,)


def get_training_data(batch_size: int):
    # Read training and test data
    (x_train, y_train, x_test, y_test) = read_data()

    # reshape and normalize input data
    x_train = x_train.reshape(x_train.shape[0], 1, 28 * 28)
    x_train = x_train.astype('float32')
    x_train /= 255
    # Convert labels into one-hot encoding
    y_train = to_categorical(y_train)

    # Create random number between 0 and 60000 - batch_size - 1
    if not DEBUGGING:
        random_number = np.random.randint(0, 60000 - batch_size - 1)
    else:
        random_number = 1024

    x_train = x_train[random_number:random_number + batch_size]
    y_train = y_train[random_number:random_number + batch_size]

    x_train = x_train.reshape(batch_size, 1, 28, 28)

    return x_train, y_train


def print_difference_numpy(input: NDArray, other: NDArray, name: str):
    result = np.allclose(input, other)
    print(f"{name} have about the same value: {result}")
    if not result:
        print(f"Max difference: {np.max(np.abs(input - other))}")
        print(f"One: {input}")
        print(f"Other: {other}")
    assert result


def forward_convolution_network(batch_size: int = 16, number_of_channels: int = 16, number_of_filters: int = 32, height_input: int = 28, width_input: int = 28, artifical_data: bool = False):
    # Obtain training data (batch_size x number_of_channels x height_input x width_input)
    if not artifical_data:
        # MNIST Training data
        input_data, target = get_training_data(batch_size)
        input_tensor = torch.from_numpy(input_data).double()
        input_tensor.requires_grad = True
        target_tensor = torch.from_numpy(target).double()
    if artifical_data:
        # Random training data
        input_data = np.random.rand(batch_size, number_of_channels, height_input, width_input)
        target = np.eye(10)[np.random.choice(10, batch_size)]
        input_tensor = torch.from_numpy(input_data).double()
        input_tensor.requires_grad = True
        target_tensor = torch.from_numpy(target).double()

    # Start forwarding step by step
    # Convolutional layer
    conv_layer_own = setup_own_convolution_layer(batch_size, number_of_channels, number_of_filters, height_input, width_input)
    conv_layer_pytorch = torch.nn.Conv2d(in_channels=number_of_channels, out_channels=number_of_filters, kernel_size=5, stride=1, padding='same')
    conv_layer_pytorch.weight = torch.nn.Parameter(torch.from_numpy(conv_layer_own.weights))
    conv_layer_pytorch.bias = torch.nn.Parameter(torch.from_numpy(conv_layer_own.bias.reshape(number_of_filters)))

    output_own = conv_layer_own.forward_propagation(input_data, batch_size, 0)
    output_pytorch_conv = conv_layer_pytorch(input_tensor)

    print_difference_numpy(output_own, output_pytorch_conv.detach().numpy(), "Convolutional layer forward")

    # Flatten layer
    flatten_layer = FlattenLayer(D_batch_size=batch_size, C_number_channels=number_of_filters, H_height_input=height_input, W_width_input=width_input)

    output_own = flatten_layer.forward_propagation(output_own, batch_size, 0)
    output_pytorch_flatten = output_pytorch_conv.view(output_pytorch_conv.size(0), -1)

    print_difference_numpy(output_own, output_pytorch_flatten.detach().numpy(), "Flatten layer forward")

    # Fully connected layer
    fc_layer_own = FCLayer(input_size=number_of_filters * height_input * width_input, output_size=10, convolutional_network=True)
    fc_layer_pytorch = nn.Linear(number_of_filters * height_input * width_input, 10)
    fc_layer_pytorch.weight = torch.nn.Parameter(torch.from_numpy(fc_layer_own.weights.T))
    fc_layer_pytorch.bias = torch.nn.Parameter(torch.from_numpy(fc_layer_own.bias.reshape(10)))

    logits_own = fc_layer_own.forward_propagation(output_own, batch_size, 0)
    logits_pytorch_fc = fc_layer_pytorch(output_pytorch_flatten)

    print_difference_numpy(logits_own, logits_pytorch_fc.detach().numpy(), "Fully connected layer forward")

    # Softmax layer
    output_own = ActivationFunction.softmax.function(logits_own)
    output_pytorch_softmax = F.softmax(logits_pytorch_fc, dim=1)

    print_difference_numpy(output_own, output_pytorch_softmax.detach().numpy(), "Softmax layer forward")

    # Loss function
    loss_own = LossFunction.categorical_cross_entropy.function(target, output_own)
    loss_own = loss_own / batch_size
    loss_pytorch = -torch.sum(target_tensor * torch.log(output_pytorch_softmax)) / batch_size

    print_difference_numpy(np.array([loss_own]), loss_pytorch.detach().numpy(), "Loss function")

    # Backward
    # Retain gradients
    input_tensor.retain_grad()
    logits_pytorch_fc.retain_grad()
    output_pytorch_flatten.retain_grad()
    output_pytorch_conv.retain_grad()

    # Compute gradients
    loss_pytorch.backward()

    # -- Softmax layer
    grad_own = LossFunction.categorical_cross_entropy.derivative(target, output_own)
    grad_pytorch = logits_pytorch_fc.grad

    print_difference_numpy(grad_own, grad_pytorch.detach().numpy(), "Softmax layer gradients")

    # -- Fully connected layer

    grad_own = fc_layer_own.backward_propagation(grad_own, LEARNING_RATE, 1)
    grad_pytorch = output_pytorch_flatten.grad

    print_difference_numpy(grad_own, grad_pytorch.detach().numpy(), "Fully connected layer gradients")

    grad_pytorch_bias = fc_layer_pytorch.bias.grad
    grad_own_bias = fc_layer_own.bias_error

    print_difference_numpy(grad_own_bias, grad_pytorch_bias.detach().numpy(), "Fully connected layer bias gradients")

    grad_own_weights = fc_layer_own.weights_error
    grad_pytorch_weights = fc_layer_pytorch.weight.grad

    print_difference_numpy(grad_own_weights.T, grad_pytorch_weights.detach().numpy(), "Fully connected layer weights gradients")

    # -- Flatten layer
    grad_own = flatten_layer.backward_propagation(grad_own, LEARNING_RATE, 1)
    grad_pytorch = output_pytorch_conv.grad

    print_difference_numpy(grad_own, grad_pytorch.detach().numpy(), "Flatten layer gradients")

    # Convolutional layer
    grad_own = conv_layer_own.backward_propagation(grad_own, LEARNING_RATE, 1)
    grad_pytorch = input_tensor.grad

    print_difference_numpy(grad_own, grad_pytorch.detach().numpy(), "Convolutional layer gradients")

    grad_own_bias = conv_layer_own.bias_error
    grad_pytorch_bias = conv_layer_pytorch.bias.grad

    print_difference_numpy(grad_own_bias, grad_pytorch_bias.detach().numpy().reshape(number_of_filters, 1), "Convolutional layer bias gradients")

    grad_own_weights = conv_layer_own.weights_error
    grad_pytorch_weights = conv_layer_pytorch.weight.grad

    print_difference_numpy(grad_own_weights, grad_pytorch_weights.detach().numpy(), "Convolutional layer weights gradients")


def test_convolution_network_propagation():
    for number_of_filters in [8, 16, 32, 64]:
        forward_convolution_network(
            batch_size=16,
            number_of_channels=1,
            number_of_filters=number_of_filters,
            height_input=28,
            width_input=28,
            artifical_data=False
        )

        forward_convolution_network(
            batch_size=32,
            number_of_channels=1,
            number_of_filters=number_of_filters,
            height_input=28,
            width_input=28,
            artifical_data=False
        )

        forward_convolution_network(
            batch_size=16,
            number_of_channels=1,
            number_of_filters=number_of_filters,
            height_input=28,
            width_input=28,
            artifical_data=True
        )

        forward_convolution_network(
            batch_size=16,
            number_of_channels=16,
            number_of_filters=number_of_filters,
            height_input=28,
            width_input=28,
            artifical_data=True
        )

        forward_convolution_network(
            batch_size=16,
            number_of_channels=32,
            number_of_filters=number_of_filters,
            height_input=28,
            width_input=28,
            artifical_data=True
        )

        forward_convolution_network(
            batch_size=16,
            number_of_channels=64,
            number_of_filters=number_of_filters,
            height_input=28,
            width_input=28,
            artifical_data=True
        )
