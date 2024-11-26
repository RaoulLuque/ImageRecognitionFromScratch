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

    def forward_propagation(self, input_data: NDArray, size_current_batch: int, current_sample_index: int) -> NDArray:
        """
        Computest the output of a layer for a given input
        :param input_data:
        :return:
        """
        raise NotImplementedError

    def backward_propagation(self, output_error: NDArray, learning_rate) -> NDArray:
        """
        Computes the input error for a given output error
        :param output_error:
        :param learning_rate:
        :return:
        """
        raise NotImplementedError

    def predict(self, input_data: NDArray) -> NDArray:
        """
        Computes the output of a layer for a given input outside of training
        :param input_data:
        :return:
        """
        raise NotImplementedError
