from src.add_ons.optimizers import Optimizer
from src.config import EPSILON
from src.layers.layer import Layer
import numpy as np
from nptyping import NDArray

from src.utils.convolution_utils import im2col_indices, col2im_indices


# noinspection PyPep8Naming
class BatchNormalization(Layer):
    """
    Batch normalization layer that inherits from the base layer class.

    See https://cs231n.github.io/convolutional-networks/#pool for more information on the terminology used in this class.

    Terminology (in brackets: defaults in our case):
    D    = number of samples in the batch
    C    = number of channels in the input data (= 1)
    H    = height of the input data (= 28)
    W    = width of the input data (= 28)

    The input data is of shape D x C x H x W
    """
    def __init__(
            self,
            D_batch_size: int,
            C_number_channels: int = 1,
            H_height_input: int = 28,
            W_width_input: int = 28,
            optimizer: Optimizer | None = None,
    ):
        """
        Initialize the batch_normalization layer. For parameters description see class docstring.
        """
        super().__init__()
        self.D_batch_size = D_batch_size
        self.C_number_channels = C_number_channels
        self.H_height_input = H_height_input
        self.W_width_input = W_width_input
        self.gamma = np.ones((C_number_channels, H_height_input, W_width_input))
        self.beta = np.zeros((C_number_channels, H_height_input, W_width_input))
        # Mean and variance (both versions) are of shape C x H x W
        self.mean: NDArray | None = None
        self.variance: NDArray | None = None
        self.running_mean: NDArray | None = None
        self.running_variance: NDArray | None = None
        self.input_normalized: NDArray | None = None

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
        """
        Forward propagation for the batch normalization layer. Computes the output of the layer given the input data.

        This layer does not change the shape of the data.
        :param input_data: Input data of shape D x C x H x W
        :param size_of_current_batch: (not used for this layer) Size of the current batch
        :param current_sample_index: (not used for this layer) Index of the current sample in the batch
        """
        self.input = input_data

        # Mean and variance will be of shape C x H x W
        self.mean = np.mean(input_data, axis=0)
        self.variance = np.var(input_data, axis=0)

        # Normalize input data. Does not change shape.
        self.input_normalized = (input_data - self.mean) / np.sqrt(self.variance + EPSILON)
        output = self.gamma * self.input_normalized + self.beta

        self.running_mean
        self.running_variance

        return output

    def backward_propagation(self, output_error_matrix: NDArray, learning_rate: float, epoch: int) -> NDArray:
        """
        Backward propagation for the max pooling layer. Computes the input error since no weights or biases exist that could be updated.
        """

        input_with_mean_zero = self.input - self.mean
        std_inv = 1. / np.sqrt(self.variance + 1e-8)

        input_error_normalized = output_error_matrix * self.gamma
        d_variance = np.sum(input_error_normalized * input_with_mean_zero, axis=0) * -.5 * std_inv ** 3
        d_mean = np.sum(input_error_normalized * -std_inv, axis=0) + d_variance * np.mean(-2. * input_with_mean_zero, axis=0)

        input_error = (input_error_normalized * std_inv) + (d_variance * 2 * input_with_mean_zero / self.D_batch_size) + (d_mean / self.D_batch_size)
        gamma_error = np.sum(output_error_matrix * self.input_normalized, axis=0)
        beta_error = np.sum(output_error_matrix, axis=0)

        # Cache gradients for debugging
        self.gamma_error = gamma_error
        self.beta_error = beta_error
        self.input_error = input_error

        match self.optimizer:
            case None:
                # Update weights and bias
                self.gamma -= learning_rate * gamma_error  # subtract dC/dW from weights
                self.beta -= learning_rate * beta_error  # subtract dC/dB from bias

            case Optimizer.Adam:
                self.gamma, self.first_moment_weights, self.second_moment_weights, self.beta1_t, self.beta2_t, self.t = self.optimizer.update_parameters(
                    self.gamma,
                    gamma_error,
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
                self.beta, self.first_moment_bias, self.second_moment_bias, _, _, _ = self.optimizer.update_parameters(
                    self.beta,
                    beta_error,
                    self.first_moment_bias,
                    self.second_moment_bias,
                    learning_rate,
                    self.beta1,
                    self.beta2,
                    self.beta1_t,
                    self.beta2_t,
                    self.t,
                    epoch,
                    increment_t=False,
                )

        return input_error

    def predict(self, input_data: NDArray, batch_size: int = 1) -> NDArray:
        # This is a pooling layer, so we can just forward propagate
        return self.forward_propagation(input_data, batch_size, 0)
