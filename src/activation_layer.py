from layer import Layer
from nptyping import NDArray


class ActivationLayer(Layer):
    """
    Activation layer that inherits from base class layer.

    Is used as intermediate layer to apply activation functions. The respective activation function is passed as a parameter on initialization.
    """
    def __init__(self, activation, activation_derivative):
        super().__init__()
        self.activation = activation
        self.activation_prime = activation_derivative

    def forward_propagation(self, input_data: NDArray) -> NDArray:
        self.input = input_data
        self.output = self.activation(self.input)
        return self.output

    # Returns input_error=dE/dX for a given output_error=dE/dY.
    # learning_rate is not used because there is no "learnable" parameters.
    def backward_propagation(self, output_error: NDArray, learning_rate: NDArray) -> NDArray:
        return self.activation_prime(self.input) * output_error
