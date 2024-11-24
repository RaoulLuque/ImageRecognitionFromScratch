import numpy as np
from nptyping import NDArray


def mean_squared_error(y_true: NDArray, y_pred: NDArray) -> np.floating:
    """Computes the mean squared error loss function."""
    return np.mean(np.power(y_true-y_pred, 2))


def mean_squared_error_derivative(y_true: NDArray, y_pred: NDArray) -> NDArray:
    """Computes the derivate of the mean squared error loss function."""
    return 2*(y_pred-y_true)/y_true.size
