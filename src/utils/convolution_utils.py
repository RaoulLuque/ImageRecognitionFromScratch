import numpy as np
from nptyping.ndarray import NDArray


def get_im2col_indices(input_shape, filter_height: int = 5, filter_width: int = 3, padding: int = 2, stride: int = 1):
    """
    Get the indices for im2col.
    For im2col, see https://cs231n.github.io/convolutional-networks/#conv.

    See docstring of Convolution2D for terminology. Defaults are also set accordingly
    :param input_shape: input data shape (D, C, H, W)
    :param filter_height: height of the filter. Defaults to 5
    :param filter_width: width of the filter. Defaults to 3
    :param padding: padding size. Defaults to 2
    :param stride: stride size. Defaults to 1
    """
    # First figure out what the size of the output should be
    _, C_number_channels, H_height, W_width = input_shape
    assert (H_height + 2 * padding - filter_height) % stride == 0
    assert (W_width + 2 * padding - filter_width) % stride == 0
    out_height = (H_height + 2 * padding - filter_height) // stride + 1
    out_width = (W_width + 2 * padding - filter_width) // stride + 1

    # Generate indices for im2col
    i0 = np.repeat(np.arange(filter_height), filter_width)
    i0 = np.tile(i0, C_number_channels)
    i1 = stride * np.repeat(np.arange(out_height), out_width)
    j0 = np.tile(np.arange(filter_width), filter_height * C_number_channels)
    j1 = stride * np.tile(np.arange(out_width), out_height)
    i = i0.reshape(-1, 1) + i1.reshape(1, -1)
    j = j0.reshape(-1, 1) + j1.reshape(1, -1)

    # Generate channel indices
    k = np.repeat(np.arange(C_number_channels), filter_height * filter_width).reshape(-1, 1)

    return k, i, j


def im2col_indices(input_data: NDArray, filter_height: int = 5, filter_width: int = 5, padding: int = 2, stride: int = 1):
    """
    An implementation of im2col based on some fancy indexing.
    For im2col, see https://cs231n.github.io/convolutional-networks/#conv.

    See docstring of Convolution2D for terminology. Defaults are also set accordingly
    :param input_data: input data of shape (D, C, H, W)
    :param filter_height: height of the filter. Defaults to 5
    :param filter_width: width of the filter. Defaults to 5
    :param padding: padding size. Defaults to 2
    :param stride: stride size. Defaults to 1
    """
    # Zero-pad the input
    p = padding
    input_padded = np.pad(input_data, ((0, 0), (0, 0), (p, p), (p, p)), mode="constant")

    k, i, j = get_im2col_indices(input_data.shape, filter_height, filter_width, padding, stride)

    cols = input_padded[:, k, i, j]
    number_of_channel = input_data.shape[1]
    cols = cols.transpose(1, 2, 0).reshape(filter_height * filter_width * number_of_channel, -1)
    return cols


def col2im_indices(cols: NDArray, input_shape, filter_height: int = 5, filter_width: int = 5, padding: int = 2, stride: int = 1):
    """
    An implementation of col2im based on fancy indexing and np.add.at
    For col2im, see https://cs231n.github.io/convolutional-networks/#conv.

    See docstring of Convolution2D for terminology. Defaults are also set accordingly
    :param cols: columns of shape (C * filter_height * filter_width, height_out * width_out * D)
    :param input_shape: input data shape (D, C, H, W)
    :param filter_height: height of the filter. Defaults to 5
    :param filter_width: width of the filter. Defaults to 5
    :param padding: padding size. Defaults to 2
    :param stride: stride size. Defaults to 1
    """
    # Unpack input shape
    D_batch_size, C_number_channels, H_height, W_width = input_shape

    # Obtain padded input
    H_height_padded, W_width_padded = H_height + 2 * padding, W_width + 2 * padding
    input_padded = np.zeros((D_batch_size, C_number_channels, H_height_padded, W_width_padded), dtype=cols.dtype)

    # Get im2col indices
    k, i, j = get_im2col_indices(input_shape, filter_height, filter_width, padding, stride)

    # Reshape columns to match input shape
    cols_reshaped = cols.reshape(C_number_channels * filter_height * filter_width, -1, D_batch_size)
    cols_reshaped = cols_reshaped.transpose(2, 0, 1)

    # Add reshaped columns to padded input
    np.add.at(input_padded, (slice(None), k, i, j), cols_reshaped)

    # Return padded input if no padding, otherwise remove padding
    if padding == 0:
        return input_padded
    return input_padded[:, :, padding:-padding, padding:-padding]
