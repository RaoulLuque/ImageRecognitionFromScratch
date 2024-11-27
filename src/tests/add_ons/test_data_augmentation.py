import random
from unittest.mock import patch

import numpy as np

from src.add_ons.data_augmentation import shift_horizontally, zoom, rotate


# NDImage shift in place cuts off some precision of the floating point numbers so we need to use
# np.allclose instead of np.array_equal


def test_shift_horizontally_no_shift():
    shift_range = random.random()
    with patch("random.random", return_value=0.5):
        test_array = np.random.rand(1, 28 * 28)
        result = shift_horizontally(test_array, shift_range)
        assert np.allclose(test_array, result)


def test_shift_horizontally_half_shift():
    with patch("random.random", return_value=1):
        test_array = np.random.rand(1, 28 * 28)
        result = shift_horizontally(test_array, 0.5)
        assert test_array.shape == result.shape
        assert not np.allclose(test_array, result)
        test_array_square = test_array.reshape(28, 28)
        result_square = result.reshape(28, 28)
        assert np.allclose(test_array_square[:, :-14], result_square[:, 14:])
        assert np.allclose(result_square[:, :14], np.zeros((28, 14)))


def test_rotate_no_rotation():
    rotation_range = random.random()
    with patch("random.random", return_value=0.5):
        test_array = np.random.rand(1, 28 * 28)
        result = rotate(test_array, rotation_range)
        assert np.allclose(test_array, result)


def test_rotate_ninety_degrees():
    with patch("random.random", return_value=1):
        height = width = 28
        test_array = np.random.rand(1, height * width)
        result = rotate(test_array, -90, (height, width))
        assert test_array.shape == result.shape
        assert not np.allclose(test_array, result)
        test_array_square = test_array.reshape(height, width)
        result_square = result.reshape(height, width)
        print(test_array_square)
        print(result_square)
        # Image gets shifted slightly to the right since there exists no center
        for i in range(height - 1):
            np.allclose(test_array_square[i], result_square[:, i+1])


def test_zoom_no_zoom():
    zoom_range = random.random()
    with patch("random.random", return_value=0.5):
        test_array = np.random.rand(1, 28 * 28)
        result = zoom(test_array, zoom_range)
        assert np.allclose(test_array, result)


def test_zoom_zoom_out_fifty_percent():
    with patch("random.random", return_value=0):
        width = height = 28
        test_array = np.random.rand(1, height * width)
        result = zoom(test_array, 0.5, (height, width))
        result_square = result.reshape(height, width)
        assert test_array.shape == result.shape
        result_square_middle_taken_out = result_square.copy()
        result_square_middle_taken_out[height//4:height//4 * 3, width//4:width//4 * 3] = 0
        assert np.array_equal(result_square_middle_taken_out, np.zeros((height, width)))
        assert not np.allclose(result_square[height//4:height//4 * 3, width//4:width//4 * 3], np.zeros((height // 2, width // 2)))


def test_zoom_zoom_in_fifty_percent():
    with patch("random.random", return_value=1):
        width = height = 28
        test_array = np.random.rand(1, height * width)
        # To avoid zero entries
        test_array += 0.001
        test_array = np.minimum(test_array, 1)
        result = zoom(test_array, 1.0, (height, width))
        result_square = result.reshape(height, width)
        assert test_array.shape == result.shape
        assert np.count_nonzero(result_square == 0) == 0
