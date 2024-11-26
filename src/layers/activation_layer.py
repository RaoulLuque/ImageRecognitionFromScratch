import numpy as np

from src.layers.layer import Layer
from nptyping import NDArray

from src.layers.activation_function import ActivationFunction


class ActivationLayer(Layer):
    """
    Activation layer that inherits from base class layer.

    Is used as intermediate layer to apply activation functions. The respective activation function is passed as a parameter on initialization.
    """
    def __init__(self, activation_function: ActivationFunction, number_of_neurons: int):
        super().__init__()
        self.activation_function = activation_function
        self.number_of_neurons = number_of_neurons

    def forward_propagation(self, input_data: NDArray) -> NDArray:
        self.input = input_data
        self.output = self.activation_function.function(self.input)
        return self.output

    # Returns input_error=dC/dZ for a given output_error=dC/dA by multiplying dA/dZ by dC/dA.
    # learning_rate and epoch is not used because there are no "learnable" parameters.
    def backward_propagation(self, output_error: NDArray, learning_rate: NDArray, epoch: int) -> NDArray:
        # If the activation function is softmax, the derivative is not applied since it has been applied already in the
        # loss layer, which has to be categorical cross entropy.
        if self.activation_function == ActivationFunction.softmax:
            return output_error
        else:
            return np.multiply(self.activation_function.derivative(self.input), output_error)

    def predict(self, input_data: NDArray) -> NDArray:
        return self.activation_function.function(input_data)
