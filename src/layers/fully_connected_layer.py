from typing import Any

from src.layers.layer import Layer
import numpy as np
from nptyping import NDArray, Shape

from src.add_ons.optimizers import Optimizer


class FCLayer(Layer):
    """
    Fully connected layer that inherits from Layer base class
    """
    def __init__(self, input_size: int, output_size: int, optimizer: Optimizer | None = None, convolutional_network: bool = False):
        """
        Initializes the weights and bias matrices with random values between -0.5 and 0.5. Also initialize optimizer
        parameters, if it is not None.
        :param input_size: number of input neurons (number of neurons in previous layer)
        :param output_size: number of output neurons (number of neurons in this layer)
        :param optimizer: optimizer to use for updating weights and bias or None
        """
        super().__init__()
        self.weights: NDArray[Shape["input_size, output_size"], Any] = np.random.rand(input_size, output_size) - 0.5
        self.bias: NDArray[Shape["1, output_size"], Any] = np.random.rand(1, output_size) - 0.5
        self.number_of_neurons = output_size
        self.convolutional_network = convolutional_network
        self.optimizer = optimizer
        if self.optimizer is not None:
            match self.optimizer:
                case Optimizer.Adam:
                    # Initialize the Adam optimizer parameters
                    self.first_moment_weights = np.zeros_like(self.weights)
                    self.second_moment_weights = np.zeros_like(self.weights)
                    self.first_moment_bias = np.zeros_like(self.bias)
                    self.second_moment_bias = np.zeros_like(self.bias)
                    self.beta1 = 0.9
                    self.beta2 = 0.999
                    self.beta1_t = 1
                    self.beta2_t = 1
                    self.t = 0

    def forward_propagation(self, input_data: NDArray, size_of_current_batch: int, current_sample_index: int) -> NDArray:
        if not self.convolutional_network:
            if current_sample_index == 0:
                # If batch has just started, initialize and reset input array to save input data for backpropagation at the end of the batch
                self.input = np.zeros((size_of_current_batch, 1, self.weights.shape[0]))
            self.input[current_sample_index] = input_data
            self.output: NDArray[Shape["self.number_of_neurons"], Any] = np.dot(self.input[current_sample_index], self.weights) + self.bias
            return self.output
        else:
            self.input = input_data
            self.output: NDArray = np.dot(input_data, self.weights) + self.bias
            return self.output

    # computes dC/dW, dC/dB for a given output_error=dC/dZ. Returns input_error=dC/dA.
    def backward_propagation(self, output_error_matrix: NDArray, learning_rate: float, epoch: int) -> NDArray:
        if not self.convolutional_network:
            #
            weights_error_matrix: NDArray = self.input.transpose(0, 2, 1) @ output_error_matrix
        else:
            weights_error_matrix: NDArray = self.input.reshape(self.input.shape[0], 1, -1).transpose(0, 2, 1) @ output_error_matrix.reshape(output_error_matrix.shape[0], 1, -1)
        weights_error: NDArray = np.average(weights_error_matrix, axis=0)
        bias_error: NDArray = np.average(output_error_matrix, axis=0)
        # Compute error to propagate to previous layer (multiply dC/dZ by dZ/dA to obtain dC/dA)
        input_error: NDArray = np.dot(output_error_matrix, self.weights.T)

        match self.optimizer:
            case None:
                # Update weights and bias
                self.weights -= learning_rate * weights_error  # subtract dC/dW from weights
                self.bias -= learning_rate * bias_error        # subtract dC/dB from bias

            case Optimizer.Adam:
                self.weights, self.first_moment_weights, self.second_moment_weights, self.beta1_t, self.beta2_t, self.t = self.optimizer.update_parameters(
                    self.weights,
                    weights_error,
                    self.first_moment_weights,
                    self.second_moment_weights,
                    learning_rate,
                    self.beta1,
                    self.beta2,
                    self.beta1_t,
                    self.beta2_t,
                    self.t,
                    epoch,
                )
                # Betas and t does not have to updated on second call because they would have been updated in the first call
                self.bias, self.first_moment_bias, self.second_moment_bias, _, _, _ = self.optimizer.update_parameters(
                    self.bias,
                    bias_error,
                    self.first_moment_bias,
                    self.second_moment_bias,
                    learning_rate,
                    self.beta1,
                    self.beta2,
                    self.beta1_t,
                    self.beta2_t,
                    self.t,
                    epoch,
                )
        return input_error

    def predict(self, input_data: NDArray) -> NDArray:
        return np.dot(input_data, self.weights) + self.bias
