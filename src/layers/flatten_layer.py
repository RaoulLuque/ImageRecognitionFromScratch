from src.layers.layer import Layer
import cupy as np
from nptyping import NDArray

from src.utils.convolution_utils import im2col_indices, col2im_indices


# noinspection PyPep8Naming
class FlattenLayer(Layer):
    """
    Flatten layer that inherits from the base layer class.

    See https://cs231n.github.io/convolutional-networks/#pool for more information on the terminology used in this class.

    Terminology (in brackets: defaults in our case):
    D    = number of samples in the batch
    C    = number of channels in the input data (= 1)
    H    = height of the input data (= 28)
    W    = width of the input data (= 28)
    HO   = height of the output data (= (H - HF + 2P) / S + 1 = (28 - 5 + 2 * 2) / 1 + 1 = 28)
    WO   = width of the output data (= (W - WF + 2P) / S + 1 = (28 - 5 + 2 * 2) / 1 + 1 = 28)

    The input data is of shape D x C x H x W = (size_of_current_batch, 1, 28, 28)
    # The output data is of shape D x NF x HO x WO = (size_of_current_batch, number_of_filters, 28, 28)


    """
    def __init__(
            self,
            D_batch_size: int,
            C_number_channels: int = 1,
            H_height_input: int = 28,
            W_width_input: int = 28,
            PS_pool_size: int = 2,
            S_stride: int = 2,
            P_padding: int = 0,
    ):
        """
        Initialize the convolutional layer. For parameters description see class docstring.
        """
        super().__init__()
        self.input_col: NDArray | None = None
        self.max_indices: NDArray | None = None

        self.D_batch_size = D_batch_size
        self.C_number_channels = C_number_channels
        self.H_height_input = H_height_input
        self.W_width_input = W_width_input
        self.PS_pool_size = PS_pool_size
        self.S_stride = S_stride
        self.P_padding = P_padding
        # computed_HO_height_out = (H_height_input - PS_pool_size) / S_stride + 1
        # assert computed_HO_height_out.is_integer(), f"Height of the output data is not an integer: {computed_HO_height_out}"
        # self.HO_height_out = int(computed_HO_height_out)
        # computed_WO_width_out = (W_width_input - PS_pool_size) / S_stride + 1
        # assert computed_WO_width_out.is_integer(), f"Width of the output data is not an integer: {computed_WO_width_out}"
        # self.WO_width_out = int(computed_WO_width_out)

    def forward_propagation(self, input_data: NDArray, size_of_current_batch: int, current_sample_index: int) -> NDArray:
        """
        Forward propagation for the flatten layer. Computes the output of the layer given the input data.

        Basically reshapes the input_data to a 1D array.
        :param input_data: input data for the layer. Shape: (D, C, H, W)
        :param size_of_current_batch: (not used in this layer type) size of the current batch
        :param current_sample_index: (not used in this layer type) index of the current sample in the batch
        :return: output of the layer. Shape: (D, C * H * W)
        """
        # Set batch_size, if it changes or model is predicting
        self.D_batch_size = size_of_current_batch

        # Cache input for debug
        self.input = input_data.copy()

        # The result will be of shape D x (C * H * W)
        output = input_data.reshape(self.D_batch_size, self.C_number_channels * self.H_height_input * self.W_width_input)

        return output

    def backward_propagation(self, output_error_matrix: NDArray, learning_rate: float, epoch: int) -> NDArray:
        """
        Backward propagation for the flatten layer. Computes the input error since no weights or biases exist that could be updated.

        Basically reshapes the output_error_matrix (a 1D vector) to the shape of the input_data (D x C x H x W).
        :param output_error_matrix: output error for the layer. Shape: (D, C * H * W)
        :param learning_rate: (not used in this layer type) learning rate
        :param epoch: (not used in this layer type) current epoch
        :return: input error for the layer. Shape: (D, C, H, W)
        """
        input_error = output_error_matrix.reshape(self.D_batch_size, self.C_number_channels, self.H_height_input, self.W_width_input)
        return input_error

    def predict(self, input_data: NDArray, batch_size: int = 1) -> NDArray:
        # This is a flatten layer, so we can just forward propagate
        return self.forward_propagation(input_data, batch_size, 0)
