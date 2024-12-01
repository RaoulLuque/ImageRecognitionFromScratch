from src.add_ons.weight_initialization import WeightInitialization
from src.layers.layer import Layer
import cupy as np
from nptyping import NDArray

from src.add_ons.optimizers import Optimizer
from src.utils.convolution_utils import im2col_indices, col2im_indices


# noinspection PyPep8Naming
class Convolution2D(Layer):
    """
    Convolution layer that inherits from the base layer class.

    See https://cs231n.github.io/convolutional-networks/#conv for more information on the terminology used in this class.

    Terminology (in brackets: defaults in our case):
    D    = number of samples in the batch
    C    = number of channels in the input data (= 1)
    H    = height of the input data (= 28)
    W    = width of the input data (= 28)
    NF/K = number of filters in the layer
    HF/F = height of the filter (= 5)
    WF/F = width of the filter (= 5)
    S    = stride (= 1)
    P    = padding (= 2 = "same", i.e. as much that layers have same height and width - in our case 2, since stride = 1 and HF = WF = 5)
    HO   = height of the output data (= (H - HF + 2P) / S + 1 = (28 - 5 + 2 * 2) / 1 + 1 = 28)
    WO   = width of the output data (= (W - WF + 2P) / S + 1 = (28 - 5 + 2 * 2) / 1 + 1 = 28)

    The input data is of shape D x C x H x W = (size_of_current_batch, 1, 28, 28)
    The output data is of shape D x NF x HO x WO = (size_of_current_batch, number_of_filters, 28, 28)

    The weights are of shape NF x C x HF x WF = (number_of_filters, 1, 5, 5)
    The bias is of shape NF x 1 = (number_of_filters, 1)
    """
    def __init__(
            self,
            D_batch_size: int,
            C_number_channels: int = 1,
            H_height_input: int = 28,
            W_width_input: int = 28,
            NF_number_of_filters: int = 32,
            HF_height_filter: int = 5,
            WF_width_filter: int = 5,
            S_stride: int = 1,
            P_padding: int = 2,
            optimizer: Optimizer | None = None,
            weight_initialization: WeightInitialization = WeightInitialization.he_bias_zero,
    ):
        """
        Initialize the convolutional layer. For parameters description see class docstring.
        :param D_batch_size: number of samples in the batch
        :param C_number_channels: number of channels in the input data
        :param H_height_input: height of the input data
        :param W_width_input: width of the input data
        :param NF_number_of_filters: number of filters in the layer
        :param HF_height_filter: height of the filter
        :param WF_width_filter: width of the filter
        :param S_stride: stride
        :param P_padding: padding
        :param optimizer: optimizer to use for updating weights and bias or None
        :param weight_initialization: weight initialization method. Defaults to He initialization with bias zero.
        """
        super().__init__()
        self.input_col: NDArray | None = None

        self.D_batch_size = D_batch_size
        self.C_number_channels = C_number_channels
        self.H_height_input = H_height_input
        self.W_width_input = W_width_input
        self.NF_number_of_filters = NF_number_of_filters
        self.HF_height_filter = HF_height_filter
        self.WF_width_filter = WF_width_filter
        self.S_stride = S_stride
        self.P_padding = P_padding
        computed_HO_height_out = (H_height_input - HF_height_filter + 2 * P_padding) / S_stride + 1
        assert computed_HO_height_out.is_integer(), f"Height of the output data is not an integer: {computed_HO_height_out}"
        self.HO_height_out = int(computed_HO_height_out)
        computed_WO_width_out = (W_width_input - WF_width_filter + 2 * P_padding) / S_stride + 1
        assert computed_WO_width_out.is_integer(), f"Width of the output data is not an integer: {computed_WO_width_out}"
        self.WO_width_out = int(computed_WO_width_out)
        self.weight_initialization = weight_initialization

        # Weight matrix has shape NF x C x HF x WF
        self.weights = weight_initialization.initialize_weights(np.array([NF_number_of_filters, C_number_channels, HF_height_filter, WF_width_filter]), C_number_channels * H_height_input * W_width_input, NF_number_of_filters * self.HO_height_out * self.WO_width_out)

        # Bias has shape NF x 1
        self.bias = weight_initialization.initialize_bias(np.array([NF_number_of_filters, 1]), C_number_channels * H_height_input * W_width_input, NF_number_of_filters * self.HO_height_out * self.WO_width_out)

        self.optimizer = optimizer
        if self.optimizer is not None:
            match self.optimizer:
                case Optimizer.Adam:
                    # Initialize the Adam optimizer parameters
                    self.first_moment_weights = np.zeros_like(self.weights)
                    self.second_moment_weights = np.zeros_like(self.weights)
                    self.first_moment_bias = np.zeros_like(self.bias)
                    self.second_moment_bias = np.zeros_like(self.bias)
                    self.beta1 = 0.9
                    self.beta2 = 0.999
                    self.beta1_t = 1
                    self.beta2_t = 1
                    self.t = 0

    def forward_propagation(self, input_data: NDArray, size_of_current_batch: int, current_sample_index: int) -> NDArray:
        """
        Forward propagation for the convolutional layer. Computes the output of the layer given the input data.

        :param input_data: input data for the layer. Shape: (D, C, H, W) = (size_of_current_batch, 1, 28, 28)
        :param size_of_current_batch: number of samples in the batch, used to updated batch_size if it changes during training or model is predicting
        :param current_sample_index: (not used for convolutional layers) index of the current sample in the batch
        :return: output of the layer. Shape: (D, NF, HO, WO) = (size_of_current_batch, number_of_filters, 28, 28)
        """
        # Set batch_size, if it changes or model is predicting
        self.D_batch_size = size_of_current_batch

        # This is an HF_height_filter x WF_width_filter convolution with stride = 1 and padding = 1
        # Furthermore, our input_data is of form D_batch_size x C x H_input x W_input, therefore
        # our resulting input_col will be a (C * HF * WF) x (D * H * W) matrix
        input_col = im2col_indices(input_data, self.HF_height_filter, self.WF_width_filter, padding=self.P_padding, stride=self.S_stride)

        # Given we have NF_number_of_filters filters, we have a weight matrix of shape NF x C x HF x WF
        # We reshape the weights to be of shape NF x (C * HF * WF)
        W_col = self.weights.reshape(self.NF_number_of_filters, self.C_number_channels * self.HF_height_filter * self.WF_width_filter)

        # The output of the convolution is then given by the matrix multiplication of the reshaped weights and the input_col
        # This gives us a NF x (C * HF * WF) @ (C * HF * WF) x (D * H * W) = NF x (D * HO * WO) matrix
        output = W_col @ input_col + self.bias  # Check for bias dimensions

        # Reshape the output back from NF x (D * HO * WO) to be of shape
        # D x NF x HO x WO = (size_of_current_batch, number_of_filters, height_of_output, width_of_output)
        # = (size_of_current_batch, number_of_filters, 28, 28)
        # First however, we reshape to NF x HO x WO x D and then transpose to D x NF x HO x WO (for data layout reasons)
        output = output.reshape(self.NF_number_of_filters, self.HO_height_out, self.WO_width_out, self.D_batch_size)
        output = output.transpose(3, 0, 1, 2)

        self.input = input_data
        self.input_col = input_col

        return output

    def backward_propagation(self, output_error_matrix: NDArray, learning_rate: float, epoch: int) -> NDArray:
        """
        Backward propagation for the convolutional layer. Computes the input error and updates the weights and bias.

        :param output_error_matrix: output error matrix for the layer.
        Shape: (D, NF, HO, WO) = (size_of_current_batch, number_of_filters, height_of_output, width_of_output)
        = (size_of_current_batch, number_of_filters, 28, 28)
        :param learning_rate: learning rate for the optimization algorithm. Used to update the weights and bias.
        :param epoch: current epoch of the training process. Used for the Adam optimizer.
        """
        # Compute bias error
        bias_error = np.sum(output_error_matrix, axis=(0, 2, 3))
        # Reshape the output_error_matrix to be of shape NF x 1
        bias_error = bias_error.reshape(self.NF_number_of_filters, 1)

        # Compute weights error
        # Reshape the output_error_matrix from D x NF x HO x WO first to be of
        # shape NF x HO x WO x D and then of shape NF x (HO * WO * D)
        output_error_reshaped = output_error_matrix.transpose(1, 2, 3, 0).reshape(self.NF_number_of_filters, self.HO_height_out * self.WO_width_out * self.D_batch_size)

        # NF x (HO * WO * D) @ (HO * WO * D) x (C * HF * WF) = NF x (C * H * W) = NF x 784
        weights_error = output_error_reshaped @ self.input_col.T
        # Convert into correct shape (self.weights.shape = NF x C x HF x WF)
        weights_error = weights_error.reshape(self.NF_number_of_filters, self.C_number_channels, self.HF_height_filter, self.WF_width_filter)

        # Compute input error
        # Reshape the weights from NF x C x HF x WF to be of shape NF x (C * HF * WF)
        weights_reshaped = self.weights.reshape(self.NF_number_of_filters, self.C_number_channels * self.HF_height_filter * self.WF_width_filter)

        # (NF x (C * HF * WF)).T @ (NF x (HO * WO * D)) = (C * HF * WF) x (HO * WO * D)
        input_error_col = weights_reshaped.T @ output_error_reshaped

        # Convert back to the original shape
        input_error = col2im_indices(input_error_col, self.input.shape, self.HF_height_filter, self.WF_width_filter, padding=self.P_padding, stride=self.S_stride)

        # Cache gradients for debugging
        self.weights_error = weights_error
        self.bias_error = bias_error
        self.input_error = input_error

        match self.optimizer:
            case None:
                # Update weights and bias
                self.weights -= learning_rate * weights_error  # subtract dC/dW from weights
                self.bias -= learning_rate * bias_error        # subtract dC/dB from bias

            case Optimizer.Adam:
                self.weights, self.first_moment_weights, self.second_moment_weights, self.beta1_t, self.beta2_t, self.t = self.optimizer.update_parameters(
                    self.weights,
                    weights_error,
                    self.first_moment_weights,
                    self.second_moment_weights,
                    learning_rate,
                    self.beta1,
                    self.beta2,
                    self.beta1_t,
                    self.beta2_t,
                    self.t,
                    epoch,
                )
                # Betas and t does not have to updated on second call because they would have been updated in the first call
                self.bias, self.first_moment_bias, self.second_moment_bias, _, _, _ = self.optimizer.update_parameters(
                    self.bias,
                    bias_error,
                    self.first_moment_bias,
                    self.second_moment_bias,
                    learning_rate,
                    self.beta1,
                    self.beta2,
                    self.beta1_t,
                    self.beta2_t,
                    self.t,
                    epoch,
                )
        return input_error

    def predict(self, input_data: NDArray, batch_size: int = 1) -> NDArray:
        # This is a convolutional layer, so we can just forward propagate
        return self.forward_propagation(input_data, batch_size, 0)
