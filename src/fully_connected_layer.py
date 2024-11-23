from typing import Any

from layer import Layer
import numpy as np
from nptyping import NDArray, Shape


class FCLayer(Layer):
    """
    Fully connected layer that inherits from Layer base class
    """
    def __init__(self, input_size: int, output_size: int):
        """
        :param input_size: number of input neurons (number of neurons in previous layer)
        :param output_size: number of output neurons (number of neurons in this layer)
        """
        super().__init__()
        self.weights: NDArray[Shape["input_size, output_size"], Any] = np.random.rand(input_size, output_size) - 0.5
        self.bias: NDArray[Shape["1, output_size"], Any] = np.random.rand(1, output_size) - 0.5

    def forward_propagation(self, input_data: NDArray) -> NDArray:
        self.input: NDArray[Shape["self.weights.shape[0]"], Any] = input_data
        self.output: NDArray[Shape["self.weights.shape[1]"], Any] = np.dot(self.input, self.weights) + self.bias
        return self.output

    # computes dE/dW, dE/dB for a given output_error=dE/dY. Returns input_error=dE/dX.
    def backward_propagation(self, output_error: NDArray, learning_rate: NDArray) -> NDArray:
        input_error: NDArray = np.dot(output_error, self.weights.T)
        weights_error: NDArray = np.dot(self.input.T, output_error)

        # Update weights and bias
        self.weights -= learning_rate * weights_error
        self.bias -= learning_rate * output_error
        return input_error
