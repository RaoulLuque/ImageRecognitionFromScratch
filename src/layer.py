import numpy as np

# Base layer class
class Layer:
    def __init__(self):
        self.input: np.ndarray | None = None
        self.output: np.ndarray | None = None

    # computes the output Y of a layer for a given input X
    def forward_propagation(self, input_data: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    # computes dE/dX for a given dE/dY (and update parameters if any)
    def backward_propagation(self, output_error, learning_rate):
        raise NotImplementedError
