from enum import Enum

import cupy as np
from typing import Any as NDArray

from src.config import EPSILON


class Optimizer(Enum):
    """
    Enum class for optimizers.

    Adam is implemented according to the paper: https://arxiv.org/abs/1412.6980. Adam parameters are updated in each
    backpropagation step, but the t parameter is updated only after the end of each epoch6.
    """
    Adam = "Adam"

    def update_parameters(
            self,
            weights: NDArray,
            weights_error: NDArray,
            first_moment: NDArray,
            second_moment: NDArray,
            learning_rate: float,
            beta1: float,
            beta2: float,
            beta1_t: float,
            beta2_t: float,
            t: int,
            epoch: int,
    ) -> tuple[NDArray, NDArray, NDArray, float, float, int]:
        """
        Update the parameters of the optimizer.

        See https://arxiv.org/abs/1412.6980 for more information on the Adam optimizer.
        :param weights: Weights to update.
        :param weights_error: Error of the weights.
        :param first_moment: First moment.
        :param second_moment: Second moment.
        :param learning_rate: Learning rate.
        :param beta1: Decay rate of the first moment.
        :param beta2: Decay rate of the second moment.
        :param beta1_t: Decay rate of the first moment to the power of t.
        :param beta2_t: Decay rate of the second moment to the power of t.
        :param t: Exponent of decay rates used to keep track of epochs.
        :param epoch: Current epoch.
        :return: Updated weights, first moment, second moment, beta1_t, beta2_t
        """
        match self:
            case Optimizer.Adam:
                if t < epoch:
                    t += 1
                    beta1_t = beta1_t * beta1
                    beta2_t = beta2_t * beta2
                first_moment = beta1 * first_moment + (1 - beta1) * weights_error
                second_moment = beta2 * second_moment + (1 - beta2) * np.power(weights_error, 2)
                first_moment_hat = first_moment / (1 - beta1_t)
                second_moment_hat = second_moment / (1 - beta2_t)
                new_weights = weights - learning_rate * first_moment_hat / (np.sqrt(second_moment_hat) + EPSILON)
                return new_weights, first_moment, second_moment, beta1_t, beta2_t, t

