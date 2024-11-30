import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from nptyping.ndarray import NDArray
from torch import Tensor

from src.add_ons.learning_rate_schedulers import LearningRateScheduler
from src.add_ons.loss_function import LossFunction
from src.config import LEARNING_RATE, DEBUGGING
from src.layers.activation_function import ActivationFunction
from src.layers.activation_layer import ActivationLayer
from src.layers.convolution_2d_layer import Convolution2D
from src.layers.flatten_layer import FlattenLayer
from src.layers.fully_connected_layer import FCLayer
from src.network import Network
from src.utils.read_data import read_data, to_categorical


class SimpleCNN(nn.Module):
    def __init__(self, number_of_channels, number_of_filters, height_images, width_images, kernel_size, stride, padding):
        super(SimpleCNN, self).__init__()
        # Convolution 2D layer
        self.conv1 = nn.Conv2d(in_channels=number_of_channels, out_channels=number_of_filters, kernel_size=5, stride=1, padding='same')
        # Fully connected layer (number of features depends on the output of the convolution)
        self.fc1 = nn.Linear(number_of_filters * height_images * width_images, 10)  # Assuming 10 classes for the output

    def forward(self, x):
        # Pass through convolutional layer
        self.conv1_input = x
        x = self.conv1.forward(x)
        # x = F.relu(x)  # Activation function
        # Flatten the output
        self.flatten_input = x
        x = x.view(x.size(0), -1)  # Flatten (batch_size, features)

        # Pass through fully connected layer
        self.fc1_input = x
        x = self.fc1.forward(x)
        # Softmax layer
        self.softmax_input = x
        x = F.softmax(x, dim=1)
        return x


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


def setup_layers_and_input(batch_size: int = 16, number_of_channels: int = 16, number_of_filters: int = 32, height_input: int = 28, width_input: int = 28):
    # Convolutional layer from own implementation
    own_implementation_layer = setup_own_convolution_layer(batch_size, number_of_channels, number_of_filters, height_input, width_input)

    # Test if forward step is the same
    input_data = torch.rand(batch_size, number_of_channels, height_input, width_input).double()

    # Convolutional layer from PyTorch with same parameters
    pytorch_layer = torch.nn.Conv2d(in_channels=number_of_channels, out_channels=number_of_filters, kernel_size=5,
                                    stride=1, padding='same', dilation=1)
    own_implementation_weights = own_implementation_layer.weights
    own_implementation_bias = own_implementation_layer.bias

    pytorch_layer.weight = torch.nn.Parameter(torch.from_numpy(own_implementation_weights))
    pytorch_layer.bias = torch.nn.Parameter(torch.from_numpy(own_implementation_bias.reshape(32)))

    return own_implementation_layer, pytorch_layer, input_data


def convolution_2d_layer_forward(batch_size: int = 16, number_of_channels: int = 1, number_of_filters: int = 32, height_input: int = 28, width_input: int = 28):
    own_implementation_layer, pytorch_layer, input_data = setup_layers_and_input(batch_size, number_of_channels, number_of_filters, height_input, width_input)

    input_data, target, _ = get_training_data(batch_size)
    input_data = torch.from_numpy(input_data).double()

    # Compute forward steps
    own_output = own_implementation_layer.forward_propagation(input_data.numpy(), 0, 0)
    pytorch_output = pytorch_layer(input_data)

    result = torch.isclose(torch.from_numpy(own_output), pytorch_output, rtol=1e-05, atol=1e-08)
    result_as_numpy = result.numpy()

    # Count occurrences of false in numpy array
    count_false = (result_as_numpy == False).sum()
    print(count_false)


def create_own_model(batch_size: int = 16, number_of_channels: int = 16, number_of_filters: int = 32, height_input: int = 28, width_input: int = 28):
    model = Network()
    model.add_layer(setup_own_convolution_layer(batch_size, number_of_channels, number_of_filters, height_input, width_input))
    model.add_layer(FlattenLayer(D_batch_size=batch_size, C_number_channels=number_of_filters, H_height_input=height_input, W_width_input=width_input))
    model.add_layer(FCLayer(input_size=number_of_filters * height_input * width_input, output_size=10, convolutional_network=True))
    model.add_layer(ActivationLayer(ActivationFunction.softmax, 10, convolutional_network=True))

    model.set_loss_function(LossFunction.categorical_cross_entropy)
    # Set (hyper)parameters
    model.set_hyperparameters(
        epochs=1,
        learning_rate=LEARNING_RATE,
        learning_rate_scheduler=LearningRateScheduler.const,
        batch_size=batch_size,
        data_augmentation=None,
        convolution_network=True,
    )
    return model


def get_training_data(batch_size: int):
    # Read training and test data
    (x_train, y_train, x_test, y_test) = read_data()

    # reshape and normalize input data
    x_train = x_train.reshape(x_train.shape[0], 1, 28 * 28)
    x_train = x_train.astype('float32')
    x_train /= 255
    # Convert labels into one-hot encoding
    y_train_not_categorical = y_train.copy()
    y_train = to_categorical(y_train)

    # Create random number between 0 and 60000 - batch_size - 1
    if not DEBUGGING:
        random_number = np.random.randint(0, 60000 - batch_size - 1)
    else:
        random_number = 1024

    x_train = x_train[random_number:random_number + batch_size]
    y_train = y_train[random_number:random_number + batch_size]
    y_train_not_categorical = y_train_not_categorical[random_number:random_number + batch_size]

    x_train = x_train.reshape(batch_size, 1, 28, 28)

    return x_train, y_train, y_train_not_categorical


def print_difference(input: Tensor, other: Tensor, name):
    result = torch.isclose(input, other, rtol=1e-05, atol=1e-08)
    result_as_numpy = result.numpy()
    count_false = (result_as_numpy == False).sum()
    print(f"Number of differences for {name} are: {count_false}")


def print_difference_numpy(input: NDArray, other: NDArray, name: str):
    result = np.allclose(input, other)
    print(f"{name} have about the same value: {result}")
    if not result:
        print(f"Max difference: {np.max(np.abs(input - other))}")
        print(f"One: {input}")
        print(f"Other: {other}")


def convolution_2d_layer_backward(batch_size: int = 16, number_of_channels: int = 16, number_of_filters: int = 32, height_input: int = 28, width_input: int = 28):
    # Define the pytorch model
    model_pytorch = SimpleCNN(number_of_channels, number_of_filters, height_input, width_input, 5, 1, 2)

    # Define own model
    model_own = create_own_model(batch_size, number_of_channels, number_of_filters, height_input, width_input)

    # Set weights to be equal
    model_pytorch.conv1.weight = torch.nn.Parameter(torch.from_numpy(model_own.layers[0].weights))
    model_pytorch.conv1.bias = torch.nn.Parameter(torch.from_numpy(model_own.layers[0].bias.reshape(number_of_filters)))

    model_pytorch.fc1.weight = torch.nn.Parameter(torch.from_numpy(model_own.layers[2].weights.T))
    model_pytorch.fc1.bias = torch.nn.Parameter(torch.from_numpy(model_own.layers[2].bias.reshape(10)))

    # Define the loss function (categorical cross-entropy)
    criterion = nn.CrossEntropyLoss()

    # Define the optimizer
    optimizer = torch.optim.SGD(model_pytorch.parameters(), lr=LEARNING_RATE)  # Stochastic Gradient Descent optimizer

    # Input tensor with batch size 16 and dimensions 16 x 1 x 28 x 28
    input_data, target, _ = get_training_data(batch_size)
    input_tensor = torch.from_numpy(input_data).double()
    target_tensor = torch.from_numpy(target).double()

    # input_tensor = torch.rand(batch_size, number_of_channels, height_input, width_input).double()
    # target_tensor = torch.randint(0, 10, (batch_size,)).long()
    # input_data = input_tensor.numpy()
    # target = target_tensor.numpy()

    # Test only conv layer
    conv1 = nn.Conv2d(in_channels=number_of_channels, out_channels=number_of_filters, kernel_size=5, stride=1,
                           padding='same')
    conv1.weight = torch.nn.Parameter(torch.from_numpy(model_own.layers[0].weights))
    conv1.bias = torch.nn.Parameter(torch.from_numpy(model_own.layers[0].bias.reshape(number_of_filters)))
    output_conv_outside_of_torch_model = conv1(input_tensor)
    output_conv_outside_of_torch_model = model_pytorch.conv1.forward(input_tensor)
    own_output_test_numpy = model_own.layers[0].forward_propagation(input_data, 0, 0)
    output_from_conv_outside_of_own_network = torch.from_numpy(own_output_test_numpy)

    # Run own model
    model_own.fit(input_data, target)

    # Forward pass
    output = model_pytorch(input_tensor)
    print_difference(model_pytorch.conv1.weight, torch.from_numpy(model_own.layers[0].weights), "Weights of convolutional layer")

    # Compute the loss
    loss = criterion(output, target_tensor)

    # Backward pass (compute gradients)
    optimizer.zero_grad()  # Clear previous gradients
    loss.backward()  # Compute gradients

    # Update weights
    optimizer.step()  # Update the parameters using the gradients

    # Start comparing
    # Forward
    # Compare input of network
    input_pytorch = model_pytorch.conv1_input
    input_own = torch.from_numpy(model_own.layers[0].input)
    print_difference(input_own, input_pytorch.float(), "Input of network")

    # Compare after conv layer
    output_after_conv_pytorch = model_pytorch.flatten_input
    output_after_conv_own = torch.from_numpy(model_own.layers[1].input)
    print(np.allclose(model_own.layers[1].input, own_output_test_numpy))
    print_difference(output_after_conv_own, output_after_conv_pytorch, "Output after convolutional layer")
    print_difference(output_from_conv_outside_of_own_network, output_conv_outside_of_torch_model, "Output after convolutional layer outside of network")
    print_difference(output_after_conv_pytorch, output_conv_outside_of_torch_model,
                     "Output after convolutional layer outside of network and not pytorch")
    print_difference(output_from_conv_outside_of_own_network, output_after_conv_own,
                     "Output after convolutional layer outside of network and not own network")


    # # Compare output
    # output_pytorch = output
    # output_own = torch.from_numpy(model_own.layers[3].forward_propagation(model_own.layers[3].input, 0, 0))
    # print_difference(output_own, output_pytorch,
    #                  "Output after softmax layer")
    #
    # # Compare loss
    # print("Pytorch Loss:", loss.item())
    # print("Own loss:", model_own.loss_function.loss)
    #
    # # Backwards
    # # Compare gradients fully connected layer
    # grad_pytorch_fc = model_pytorch.fc1.weight.grad
    # grad_own_model_fc = torch.from_numpy(model_own.layers[2].weights_error.T)
    #
    # # print(grad_pytorch_fc)
    # # print(grad_own_model_fc)
    #
    # print_difference(grad_own_model_fc, grad_pytorch_fc, "Fully connected layer gradient")
    #
    # # Compare gradients conv layer
    # grad_pytorch_cov = model_pytorch.conv1.weight.grad
    # grad_own_model_conv = torch.from_numpy(model_own.layers[0].weights_error)
    #
    # print_difference(grad_own_model_conv, grad_pytorch_cov, "Convolutional layer gradient")
    #
    # # Compare weights
    # own_convolutional_updated_weights = torch.from_numpy(model_own.layers[0].weights)
    # pytorch_convolutional_updated_weights = model_pytorch.conv1.weight
    #
    # print_difference(own_convolutional_updated_weights, pytorch_convolutional_updated_weights, "Convolutional layer weights")


def forward_convolution_network(batch_size: int = 16, number_of_channels: int = 16, number_of_filters: int = 32, height_input: int = 28, width_input: int = 28):
    # Obtain training data
    # Input tensor with dimensions batch_size x 1 x 28 x 28
    input_data, target, target_not_one_hot_encoded = get_training_data(batch_size)
    input_tensor = torch.from_numpy(input_data).double()
    input_tensor.requires_grad = True
    target_tensor = torch.from_numpy(target).double()
    target_not_one_hot_encoded_tensor = torch.from_numpy(target_not_one_hot_encoded).long()

    # Start forwarding step by step
    # Convolutional layer
    conv_layer_own = setup_own_convolution_layer(batch_size, number_of_channels, number_of_filters, height_input, width_input)
    conv_layer_pytorch = torch.nn.Conv2d(in_channels=number_of_channels, out_channels=number_of_filters, kernel_size=5, stride=1, padding='same')
    conv_layer_pytorch.weight = torch.nn.Parameter(torch.from_numpy(conv_layer_own.weights))
    conv_layer_pytorch.bias = torch.nn.Parameter(torch.from_numpy(conv_layer_own.bias.reshape(number_of_filters)))

    output_own = conv_layer_own.forward_propagation(input_data, 0, 0)
    output_pytorch_conv = conv_layer_pytorch(input_tensor)

    print_difference_numpy(output_own, output_pytorch_conv.detach().numpy(), "Convolutional layer forward")

    # Flatten layer
    flatten_layer = FlattenLayer(D_batch_size=batch_size, C_number_channels=number_of_filters, H_height_input=height_input, W_width_input=width_input)

    output_own = flatten_layer.forward_propagation(output_own, 0, 0)
    output_pytorch_flatten = output_pytorch_conv.view(output_pytorch_conv.size(0), -1)

    print_difference_numpy(output_own, output_pytorch_flatten.detach().numpy(), "Flatten layer forward")

    # Fully connected layer
    fc_layer_own = FCLayer(input_size=number_of_filters * height_input * width_input, output_size=10, convolutional_network=True)
    fc_layer_pytorch = nn.Linear(number_of_filters * height_input * width_input, 10)
    fc_layer_pytorch.weight = torch.nn.Parameter(torch.from_numpy(fc_layer_own.weights.T))
    fc_layer_pytorch.bias = torch.nn.Parameter(torch.from_numpy(fc_layer_own.bias.reshape(10)))

    output_own = fc_layer_own.forward_propagation(output_own, 0, 0)
    output_pytorch_fc = fc_layer_pytorch(output_pytorch_flatten)

    print_difference_numpy(output_own, output_pytorch_fc.detach().numpy(), "Fully connected layer forward")

    # Softmax layer (PyTorch automatically applies softmax before CategoricalCrossentropy, so we will skip pytorch
    # on this one
    output_own = ActivationFunction.softmax.function(output_own)


    # Loss function
    loss_own = LossFunction.categorical_cross_entropy.function(target, output_own)
    loss_pytorch = F.cross_entropy(output_pytorch_fc, target_not_one_hot_encoded_tensor)

    print_difference_numpy(np.array([loss_own]), loss_pytorch.detach().numpy(), "Loss function")

    # Backward
    # Retain gradients
    input_tensor.retain_grad()
    output_pytorch_fc.retain_grad()
    output_pytorch_flatten.retain_grad()
    output_pytorch_conv.retain_grad()

    # Compute gradients
    loss_pytorch.backward()

    # -- Softmax layer
    grad_own = LossFunction.categorical_cross_entropy.derivative(target, output_own)
    grad_pytorch = output_pytorch_fc.grad

    print_difference_numpy(grad_own, grad_pytorch.detach().numpy(), "Softmax layer gradients")

    # -- Fully connected layer

    grad_own = fc_layer_own.backward_propagation(grad_own, LEARNING_RATE, 1)
    grad_pytorch = output_pytorch_flatten.grad

    print_difference_numpy(grad_own, grad_pytorch.detach().numpy(), "Fully connected layer gradients")

    grad_own_bias = fc_layer_own.bias_error
    grad_pytorch_bias = fc_layer_pytorch.bias.grad

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

    print_difference_numpy(grad_own_bias, grad_pytorch_bias.detach().numpy(), "Convolutional layer bias gradients")

    grad_own_weights = conv_layer_own.weights_error
    grad_pytorch_weights = conv_layer_pytorch.weight.grad

    print_difference_numpy(grad_own_weights, grad_pytorch_weights.detach().numpy(), "Convolutional layer weights gradients")


def trying_smth():
    import torch
    import torch.nn.functional as F

    # Example input tensor (logits)
    logits = torch.tensor([[1.0, 2.0, 3.0]], requires_grad=True)

    # Example target tensor (class indices)
    target = torch.tensor([2])

    # Compute the softmax probabilities
    softmax_probs = F.softmax(logits, dim=1)

    # Compute the cross-entropy loss
    loss = F.cross_entropy(logits, target)

    # Backward pass: Compute gradients
    loss.backward()

    # Gradient of the loss with respect to the input logits
    grad_pytorch = logits.grad

    # Manually compute the gradient (softmax_probs - one_hot(target))
    one_hot_target = torch.zeros_like(logits).scatter_(1, target.unsqueeze(1), 1)
    manual_grad = softmax_probs - one_hot_target

    print("Logits:", logits)
    print("Softmax probabilities:", softmax_probs)
    print("One-hot target:", one_hot_target)
    print("Gradient of the loss with respect to the input logits (PyTorch):", grad_pytorch)
    print("Manually computed gradient:", manual_grad)

if __name__ == "__main__":
    # convolution_2d_layer_forward(
    #     batch_size=16,
    #     number_of_channels=1,
    #     number_of_filters=32,
    #     height_input=28,
    #     width_input=28,
    # )

    # convolution_2d_layer_backward(
    #     batch_size=1,
    #     number_of_channels=1,
    #     number_of_filters=1,
    #     height_input=28,
    #     width_input=28,
    # )

    # trying_smth()

    forward_convolution_network(
        batch_size=1,
        number_of_channels=1,
        number_of_filters=1,
        height_input=28,
        width_input=28,
    )

