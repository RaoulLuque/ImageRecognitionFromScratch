import cupy as np
from nptyping import NDArray
from enum import Enum


class WeightInitialization(Enum):
    """
    Enum class for loss functions.

    The default is not recommended. It only exists for compatibility reasons.
    """
    default_fully_connected = "default"
    he_bias_zero = "he_bias_zero"
    xavier_bias_zero = "xavier_bias_zero"

    def initialize_weights(self, shape: NDArray, input_size: int, output_size: int) -> NDArray:
        """Returns the initial value of the bias."""
        match self:
            case WeightInitialization.default_fully_connected:
                return default_weight_initialization(shape, input_size, output_size)
            case WeightInitialization.he_bias_zero:
                return he_weight_initialization(shape, input_size, output_size)
            case WeightInitialization.xavier_bias_zero:
                return xavier_weight_initialization(shape, input_size, output_size)

    def initialize_bias(self, shape: NDArray, input_size: int, output_size: int) -> NDArray:
        """Returns the derivative of the loss function."""
        match self:
            case WeightInitialization.default_fully_connected:
                return default_bias_initialization(shape, input_size, output_size)
            case WeightInitialization.he_bias_zero:
                return he_bias_initialization(shape, input_size, output_size)
            case WeightInitialization.xavier_bias_zero:
                return xavier_bias_initialization(shape, input_size, output_size)


def default_weight_initialization(shape: NDArray, input_size: int, output_size: int) -> NDArray:
    """
    Returns the initial value of the weights by sampling from a uniform distribution between -0.5 and 0.5.

    :param shape: Shape of the weights.
    :param input_size: Number of input neurons.
    :param output_size: Number of output neurons.
    :return: Initial value of the weights.
    """
    return np.random.rand(*shape) - 0.5


def default_bias_initialization(shape: NDArray, input_size: int, output_size: int) -> NDArray:
    """
    Returns the initial value of the bias by sampling from a uniform distribution between -0.5 and 0.5.

    :param shape: Shape of the bias.
    :param input_size: Number of input neurons.
    :param output_size: Number of output neurons.
    :return: Initial value of the bias.
    """
    return np.random.rand(*shape) - 0.5


def he_weight_initialization(shape: NDArray, input_size: int, output_size: int) -> NDArray:
    """
    Implements He initialization. Returns the initial value of the weights by sampling from a normal distribution with mean 0 and standard deviation
    sqrt(2 / input_size).

    :param shape: Shape of the weights.
    :param input_size: Number of input neurons.
    :param output_size: Number of output neurons.
    :return: Initial value of the weights.
    """
    return np.random.randn(*shape) * np.sqrt(2 / input_size)


def he_bias_initialization(shape: NDArray, input_size: int, output_size: int) -> NDArray:
    """
    Implements He initialization. Returns the initial value of the bias by setting it to zero.

    :param shape: Shape of the bias.
    :param input_size: Number of input neurons.
    :param output_size: Number of output neurons.
    :return: Initial value of the bias.
    """
    return np.zeros(shape)


def xavier_weight_initialization(shape: NDArray, input_size: int, output_size: int) -> NDArray:
    """
    Implements Glorot/Xavier initialization. Returns the initial value of the weights by sampling from a normal distribution with mean 0 and standard deviation
    sqrt(1 / input_size).

    :param shape: Shape of the weights.
    :param input_size: Number of input neurons.
    :param output_size: Number of output neurons.
    :return: Initial value of the weights.
    """
    return np.random.randn(*shape) * np.sqrt(6 / (input_size + output_size))


def xavier_bias_initialization(shape: NDArray, input_size: int, output_size: int) -> NDArray:
    """
    Implements Glorot/Xavier initialization. Returns the initial value of the bias by setting it to zero.

    :param shape: Shape of the bias.
    :param input_size: Number of input neurons.
    :param output_size: Number of output neurons.
    :return: Initial value of the bias.
    """
    return np.zeros(shape)
