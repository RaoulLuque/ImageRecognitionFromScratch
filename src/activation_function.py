import numpy as np
from enum import Enum


class ActivationFunction(Enum):
    """Enum class for activation functions."""
    ReLu = "ReLu"
    tanh = "tanh"

    def function(self, x: np.ndarray) -> np.ndarray:
        """
        Returns the activation function.
        :param x: Input to the activation function.
        :return: Result of the activation function.
        """
        match self:
            case ActivationFunction.ReLu:
                return ReLu(x)
            case ActivationFunction.tanh:
                return tanh(x)

    def derivative(self, x: np.ndarray) -> np.ndarray:
        """
        Returns the derivative of the activation function.
        :param x: Input of the derivative of the activation function.
        :return: Result of the derivative of the activation function.
        """
        match self:
            case ActivationFunction.ReLu:
                return ReLu_derivative(x)
            case ActivationFunction.tanh:
                return tanh_derivative(x)


def ReLu(x):
    """
    Computes the rectified linear unit function. That is, 0 if x < 0, x otherwise.
    :param x: Input
    :return: Output of rectified linear unit function
    """
    return np.maximum(0, x)


def ReLu_derivative(x):
    """
    Computes the derivative of the rectified linear unit function. That is, 0 if x <= 0, 1 otherwise.
    :param x: Input
    :return: Output of the derivative of the rectified linear unit function
    """
    return np.where(x <= 0, 0, 1)


def tanh(x):
    """Computes the hyperbolic tangent function."""
    return np.tanh(x)


def tanh_derivative(x):
    """Computes the derivative of the hyperbolic tangent function."""
    return 1-np.tanh(x)**2
