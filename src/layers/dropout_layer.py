import numpy as np

from src.layers.layer import Layer
from nptyping import NDArray


class DropoutLayer(Layer):
    """
    Dropout layer that inherits from the base layer class.

    Is used as intermediate layer to randomly set input values to zero. The dropout rate is passed as a parameter on initialization.
    """
    def __init__(self, dropout_rate: float, number_of_neurons: int, convolutional_network: bool = False):
        super().__init__()
        self.dropout_rate = dropout_rate
        self.number_of_neurons = number_of_neurons
        self.convolutional_network = convolutional_network
        self.mask = None

    def forward_propagation(self, input_data: NDArray, size_of_current_batch: int, current_sample_index: int) -> NDArray:
        # For convolutional networks the input is given as the entire batch
        if self.convolutional_network:
            self.mask = np.random.binomial(1, 1 - self.dropout_rate, input_data.shape)
            return (input_data * self.mask) / (1 - self.dropout_rate)
        # For fully connected networks the input is given sample by sample (of the batch)
        else:
            if current_sample_index == 0:
                # If batch has just started, initialize and reset mask array to masks for backpropagation at the end of the batch
                self.mask = np.zeros((size_of_current_batch, 1, self.number_of_neurons))
            # Create mask for current sample to multiply with input data to set random values to zero
            self.mask[current_sample_index] = np.random.binomial(1, 1 - self.dropout_rate, (1, self.number_of_neurons))
            # Apply mask and scale up entries not set to zero to keep the sum of the entries somewhat the same
            return (input_data * self.mask[current_sample_index]) / (1 - self.dropout_rate)

    def backward_propagation(self, output_error: NDArray, learning_rate: NDArray, epoch: int) -> NDArray:
        # Multiply output error with mask to set the same values to zero as in the forward pass
        # because these should not be adjusted
        return output_error * self.mask

    def predict(self, input_data: NDArray, batch_size: int = 1) -> NDArray:
        return input_data
