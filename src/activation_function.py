import numpy as np


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


def tanh_prime(x):
    """Computes the derivative of the hyperbolic tangent function."""
    return 1-np.tanh(x)**2
