import numpy as np
from nptyping.ndarray import NDArray


class Layer:
    input: NDArray | None
    output: NDArray | None
    """
    Base class for layers in the neural network
    """
    def __init__(self):
        """
        Initialize input and output of the layer as none
        """
        self.input: NDArray | None = None  # type: ignore
        self.output: NDArray | None = None  # type: ignore
        self.number_of_neurons: int | None = None

    def forward_propagation(self, input_data: NDArray, size_of_current_batch: int, current_sample_index: int) -> NDArray:
        """
        Computest the output of a layer for a given input. Possibly has side effects on attributes of the layer,
        which is why, for predictions, the predict method should be used.
        :param input_data: Input data.
        :param size_of_current_batch: Size of the current batch (necessary for mini-batch implementation).
        :param current_sample_index: Index of the current sample in the batch (necessary for mini-batch implementation).
        :return: Output of the layer given the input.
        """
        raise NotImplementedError

    def backward_propagation(self, output_error: NDArray, learning_rate: float, epoch: int) -> NDArray:
        """
        Computes the input error for a given output error.
        :param output_error: Error of the output to be propagated back.
        :param learning_rate: Learning rate.
        :param epoch: Current epoch (used only by Adam).
        :return: Error of the input to be propagated back
        """
        raise NotImplementedError

    def predict(self, input_data: NDArray) -> NDArray:
        """
        Computes the output of a layer for a given input outside of training
        :param input_data: Input data.
        :return: Output of the layer given the input data.
        """
        raise NotImplementedError
