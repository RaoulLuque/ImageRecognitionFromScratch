from layer import Layer
import numpy as np
import numpy.typing as npt


# Fully connected layer that inherits from layer base class
class FCLayer(Layer):
    # input_size = number of input neurons (number of neurons in previous layer)
    # output_size = number of output neurons (number of neurons in this layer)
    def __init__(self, input_size: int, output_size: int):
        super().__init__()
        self.weights: np.ndarray = np.random.rand(input_size, output_size) - 0.5
        self.bias: np.ndarray = np.random.rand(1, output_size) - 0.5

    # returns output for a given input
    def forward_propagation(self, input_data: np.ndarray) -> np.ndarray:
        self.input: np.ndarray = input_data
        self.output: np.ndarray = np.dot(self.input, self.weights) + self.bias
        return self.output

    # computes dE/dW, dE/dB for a given output_error=dE/dY. Returns input_error=dE/dX.
    def backward_propagation(self, output_error: np.ndarray, learning_rate: np.ndarray) -> np.ndarray:
        input_error: np.ndarray = np.dot(output_error, self.weights.T)
        weights_error: np.ndarray = np.dot(self.input.T, output_error)
        # dBias = output_error

        # update parameters
        self.weights -= learning_rate * weights_error
        self.bias -= learning_rate * output_error
        return input_error
