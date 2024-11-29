from src.layers.layer import Layer
import numpy as np
from nptyping import NDArray

from src.utils.convolution_utils import im2col_indices, col2im_indices


# noinspection PyPep8Naming
class MaxPoolingLayer2D(Layer):
    """
    Max pooling layer that inherits from the base layer class.

    See https://cs231n.github.io/convolutional-networks/#pool for more information on the terminology used in this class.

    Terminology (in brackets: defaults in our case):
    D    = number of samples in the batch
    C    = number of channels in the input data (= 1)
    H    = height of the input data (= 28)
    W    = width of the input data (= 28)
    PS   = size (height and width) of the pooling filter (= 2)
    S    = stride (= 1)
    P    = padding (= 2 = "same", i.e. as much that layers have same height and width - in our case 2, since stride = 1 and HF = WF = 5)
    HO   = height of the output data (= (H - HF) / S + 1 = (28 - 5) / 1 + 1 = 14)
    WO   = width of the output data (= (W - WF) / S + 1 = (28 - 5) / 1 + 1 = 14)

    The input data is of shape D x C x H x W = (size_of_current_batch, 1, 28, 28)
    The output data is of shape D x C x HO x WO = (size_of_current_batch, number_of_filters, 14, 14)


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
        computed_HO_height_out = (H_height_input - PS_pool_size) / S_stride + 1
        assert computed_HO_height_out.is_integer(), f"Height of the output data is not an integer: {computed_HO_height_out}"
        self.HO_height_out = int(computed_HO_height_out)
        computed_WO_width_out = (W_width_input - PS_pool_size) / S_stride + 1
        assert computed_WO_width_out.is_integer(), f"Width of the output data is not an integer: {computed_WO_width_out}"
        self.WO_width_out = int(computed_WO_width_out)

    def forward_propagation(self, input_data: NDArray, size_of_current_batch: int, current_sample_index: int) -> NDArray:
        """
        Forward propagation for the convolutional layer. Computes the output of the layer given the input data.

        With the default parameters of pool_size = 2, stride = 1 and padding = 0,
        the layer will basically halve the height and width.
        :param input_data: input data for the layer. Shape: (D, C, H, W) = (size_of_current_batch, 1, 28, 28)
        :param size_of_current_batch: (not used for pooling layers) size of the current batch
        :param current_sample_index: (not used for pooling layers) index of the current sample in the batch
        :return: output of the layer. Shape: (D, C, HO, WO) = (size_of_current_batch, number_of_filters, 14, 14)
        """
        # This is an PS_pool_size x PS_pool_size max pooling layer with stride = 2 and padding = 0
        # Furthermore, our input_data is of form D_batch_size x C_number_channels x H_input x W_input,

        # First reshape the input_data to be of shape (D * C) x 1 x H x W
        input_reshaped = input_data.reshape(self.D_batch_size * self.C_number_channels, 1, self.H_height_input, self.W_width_input)

        # Then apply im2col_indices to the reshaped input_data
        # The result will be of shape (PS * PS * 1) x (D * C * HO * WO) = 4 x (D * C * 14 * 14)
        self.input_col = im2col_indices(input_reshaped, self.PS_pool_size, self.PS_pool_size, padding=self.P_padding, stride=self.S_stride)

        # Max pool, i.e. take the index of the max entry at each possible patch position
        # each of the (D * C * HO * WO) patch positions being of size (PS * PS)
        self.max_indices = np.argmax(self.input_col, axis=0)

        # Get the max values at each column
        # The result will be of shape 1 x (D * C * HO * WO) = 1 x (D * C * 14 * 14)
        output = self.input_col[self.max_indices, np.arange(self.max_indices.size)]

        # Reshape the output to be of shape D x C x HO x WO = (size_of_current_batch, number_of_filters, 14, 14)
        # To do so, first reshape to HO x WO x D x C
        output = output.reshape(self.HO_height_out, self.WO_width_out, self.D_batch_size, self.C_number_channels)
        # Then transpose to D x C x HO x WO
        output = output.transpose(2, 3, 0, 1)

        return output

    def backward_propagation(self, output_error_matrix: NDArray, learning_rate: float, epoch: int) -> NDArray:
        """
        Backward propagation for the max pooling layer. Computes the input error since no weights or biases exist that could be updated.

        We upscale the output error matrix back to the input scaling, since we downscaled in the forward step.
        :param output_error_matrix: output error matrix for the layer.
        Shape: (D, C, HO, WO) = (size_of_current_batch, number_of_filters, 14, 14)
        :param learning_rate: (not used in pooling layer since no learnable parameters exist) learning rate for the optimization algorithm. Used to update the weights and bias.
        :param epoch: (not used in pooling layer since no learnable parameters exist) current epoch of the training process. Used for the Adam optimizer.
        """
        # Given the output data and therefore error is of shape D x C x HO x WO = (size_of_current_batch, 1, 14, 14)
        # We want to upscale this to the input shape D x C x H x W = (size_of_current_batch, 1, 28, 28)
        # First initial an empty matrix of shape (PS * PS * 1) x (D * C * HO * WO) = 4 x (D * C * 14 * 14)
        input_error_col = np.zeros((self.PS_pool_size * self.PS_pool_size * 1, self.D_batch_size * self.C_number_channels * self.HO_height_out * self.WO_width_out))

        # The output error matrix is of shape D x C x HO x WO = (size_of_current_batch, 1, 14, 14)
        # We want to reshape this to be of shape 1 x (D * C * HO * WO) = 1 x (D * C * 14 * 14)
        # To achieve this we first transpose to HO x WO x D x C
        output_error_matrix_reshaped = output_error_matrix.transpose(2, 3, 0, 1)
        output_error_matrix_flat = output_error_matrix_reshaped.ravel()

        # Then we set the input_error_col at the max_indices to the output_error_matrix_flat
        # Essentially we are up scaling the output error matrix back to the input scaling by
        # inserting one of the (D * C * HO * WO) gradients into of the (PS * PS * 1) = 4 positions
        input_error_col[self.max_indices, np.arange(self.max_indices.size)] = output_error_matrix_flat

        # Now we can apply col2im_indices to the input_error_col to transform the col back to an image
        # The result will first be of shape (D * C) x 1 x H x W = (size_of_current_batch * number_of_channels, 1, 28, 28)
        input_error = col2im_indices(input_error_col, (self.D_batch_size * self.C_number_channels, 1, self.H_height_input, self.W_width_input), self.PS_pool_size, self.PS_pool_size, padding=self.P_padding, stride=self.S_stride)
        # Then we reshape to D x C x H x W = (size_of_current_batch, 1, 28, 28)
        input_error = input_error.reshape(self.D_batch_size, self.C_number_channels, self.H_height_input, self.W_width_input)

        return input_error

    def predict(self, input_data: NDArray) -> NDArray:
        # This is a pooling layer, so we can just forward propagate
        return self.forward_propagation(input_data, 0, 0)
