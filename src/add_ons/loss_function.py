import cupy as np
from typing import Any as NDArray

from enum import Enum


class LossFunction(Enum):
    """Enum class for loss functions."""
    mse = "mean_squared_error"
    categorical_cross_entropy = "categorical_cross_entropy"

    def function(self, y_true: NDArray, y_pred: NDArray) -> np.floating:
        """Returns the loss function."""
        match self:
            case LossFunction.mse:
                return mean_squared_error(y_true, y_pred)
            case LossFunction.categorical_cross_entropy:
                self.loss = categorical_cross_entropy(y_true, y_pred)
                return self.loss

    def derivative(self, y_true: NDArray, y_pred: NDArray) -> NDArray:
        """Returns the derivative of the loss function."""
        match self:
            case LossFunction.mse:
                return mean_squared_error_derivative(y_true, y_pred)
            case LossFunction.categorical_cross_entropy:
                self.loss_gradient = categorical_cross_entropy_derivative(y_true, y_pred)
                return self.loss_gradient


def mean_squared_error(y_true: NDArray, y_pred: NDArray) -> np.floating:
    """Computes the mean squared error loss function."""
    return np.mean(np.power(y_true-y_pred, 2))


def mean_squared_error_derivative(y_true: NDArray, y_pred: NDArray) -> NDArray:
    """Computes the derivative of the mean squared error loss function."""
    return 2*(y_pred-y_true)/y_true.size


def categorical_cross_entropy(y_true: NDArray, y_pred: NDArray) -> np.floating:
    """Computes the categorical cross entropy loss function."""
    return -np.sum(y_true * np.log(y_pred))


def categorical_cross_entropy_derivative(y_true: NDArray, y_pred: NDArray) -> NDArray:
    """
    Computes the derivative of the categorical cross entropy loss function.
    That is, the combined derivative of the softmax activation function and the categorical cross entropy loss function.
    In other words, dC/dZ = A - Y, where A is the output of the softmax activation function and Y is the true label.

    *ATTENTION*: This function is only supposed to be used in combination with softmax.
    """
    return (y_pred - y_true) / y_pred.shape[0]
