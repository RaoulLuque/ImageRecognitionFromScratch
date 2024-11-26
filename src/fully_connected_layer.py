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
        self.number_of_neurons = output_size

    def forward_propagation(self, input_data: NDArray, size_current_batch: int, current_sample_index: int) -> NDArray:
        if current_sample_index == 0:
            # If batch has just started, initialize and reset input array to save input data for backpropagation at the end of the batch
            self.input = np.zeros((size_current_batch, 1, self.weights.shape[0]))
        self.input[current_sample_index] = input_data
        self.output: NDArray[Shape["self.number_of_neurons"], Any] = np.dot(self.input[current_sample_index], self.weights) + self.bias
        return self.output

    # computes dC/dW, dC/dB for a given output_error=dC/dZ. Returns input_error=dC/dA.
    def backward_propagation(self, output_error_matrix: NDArray, learning_rate: NDArray) -> NDArray:
        weights_error_matrix: NDArray = self.input.transpose(0, 2, 1) @ output_error_matrix
        weights_error: NDArray = np.average(weights_error_matrix, axis=0)

        # Update weights and bias
        self.weights -= learning_rate * weights_error                           # subtract dC/dW from weights
        self.bias -= learning_rate * np.average(output_error_matrix, axis=0)    # subtract dC/dB from bias

        # Compute error to propagate to previous layer (multiply dC/dZ by dZ/dA to obtain dC/dA)
        input_error: NDArray = np.dot(output_error_matrix, self.weights.T)
        return input_error

    def predict(self, input_data: NDArray) -> NDArray:
        return np.dot(input_data, self.weights) + self.bias
