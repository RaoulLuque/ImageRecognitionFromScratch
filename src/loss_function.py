import numpy as np
from nptyping import NDArray
from enum import Enum


class LossFunction(Enum):
    """Enum class for loss functions."""
    MEAN_SQUARED_ERROR = "mean_squared_error"

    def function(self, y_true: NDArray, y_pred: NDArray) -> np.floating:
        """Returns the loss function."""
        match self:
            case LossFunction.MEAN_SQUARED_ERROR:
                return mean_squared_error(y_true, y_pred)

    def derivative(self, y_true: NDArray, y_pred: NDArray) -> NDArray:
        """Returns the derivative of the loss function."""
        match self:
            case LossFunction.MEAN_SQUARED_ERROR:
                return mean_squared_error_derivative(y_true, y_pred)


def mean_squared_error(y_true: NDArray, y_pred: NDArray) -> np.floating:
    """Computes the mean squared error loss function."""
    return np.mean(np.power(y_true-y_pred, 2))


def mean_squared_error_derivative(y_true: NDArray, y_pred: NDArray) -> NDArray:
    """Computes the derivate of the mean squared error loss function."""
    return 2*(y_pred-y_true)/y_true.size
