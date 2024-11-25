from layer import Layer
from nptyping import NDArray

from src.activation_function import ActivationFunction


class ActivationLayer(Layer):
    """
    Activation layer that inherits from base class layer.

    Is used as intermediate layer to apply activation functions. The respective activation function is passed as a parameter on initialization.
    """
    def __init__(self, activation_function: ActivationFunction):
        super().__init__()
        self.activation_function = activation_function

    def forward_propagation(self, input_data: NDArray) -> NDArray:
        self.input = input_data
        self.output = self.activation_function.function(self.input)
        return self.output

    # Returns input_error=dC/dZ for a given output_error=dC/dA.
    # learning_rate is not used because there is no "learnable" parameters.
    def backward_propagation(self, output_error: NDArray, learning_rate: NDArray) -> NDArray:
        return self.activation_function.derivative(self.input) * output_error
