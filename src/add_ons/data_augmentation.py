import random

import cv2
import cupy as np
from nptyping import NDArray


class DataAugmentation:
    def __init__(self, chance_of_altering_data: float = 1.0, horizontal_shift_range: float = 0.2, vertical_shift_range: float = 0.2, rotation_range: float = 25, zoom_range: float = 0.2):
        self.chance_of_altering_data = chance_of_altering_data
        self.horizontal_shift_range = horizontal_shift_range
        self.vertical_shift_range = vertical_shift_range
        self.rotation_range = rotation_range
        self.zoom_range = zoom_range

    def batch_apply(self, images: NDArray) -> NDArray:
        """
        Augments the images in a batch by shifting, rotating and zooming them.
        :param images: Images to augment
        :return: Augmented images
        """
        augmented_images = []
        for image in images:
            augmented_images.append(self.apply(image))
        return np.array(augmented_images)

    def apply(self, image: NDArray) -> NDArray:
        """
        Augments the image by shifting, rotating and zooming it.
        :param image: Image to augment
        :return: Augmented image
        """
        image = zoom(image, self.zoom_range)
        image = rotate(image, self.rotation_range)
        image = shift_horizontally(image, self.horizontal_shift_range)
        image = shift_vertically(image, self.vertical_shift_range)
        return image


def shift_horizontally(image: NDArray, shift_range: float, image_size: tuple[int, int] = (28, 28)):
    """
    Shifts the image horizontally by a random amount between -shift_range and shift_range.

    Expects the image to be of size 28x28 (no matter the shape).
    :param image: Image to shift
    :param shift_range: Maximum amount to shift the image (percentage of the image width)
    :param image_size: Size of the image. Defaults to 28 * 28
    :return: Shifted image
    """
    # Randomly choose the amount to shift the image
    random_number = random.random()
    shift: float = (random_number - 0.5) * 2 * shift_range
    height, width = image_size
    pixel_shift = int(shift * width)
    image_shape = image.shape
    image = image.reshape(image_size)
    # return ndimage.shift(image, (0, int(28 * shift))).reshape(image_shape)

    # Define translation matrix
    translation_matrix = np.float32([[1, 0, pixel_shift], [0, 1, 0]])

    # Perform the shift
    shifted = cv2.warpAffine(image, translation_matrix, (image.shape[1], image.shape[0]))
    return shifted.reshape(image_shape)


def shift_vertically(image: NDArray, shift_range: float, image_size: tuple[int, int] = (28, 28)):
    """
    Shifts the image vertically by a random amount between -shift_range and shift_range.

    Expects the image to be of size 28x28 (no matter the shape).
    :param image: Image to shift
    :param shift_range: Maximum amount to shift the image (percentage of the image width)
    :param image_size: Size of the image. Defaults to 28 * 28
    :return: Shifted image
    """
    # Randomly choose the amount to shift the image
    shift: float = (random.random() - 0.5) * 2 * shift_range
    height, width = image_size
    pixel_shift = int(shift * height)
    image_shape = image.shape
    image = image.reshape(28, 28)
    # return ndimage.shift(image, (int(28 * shift), 0)).reshape(image_shape)

    # Define translation matrix
    translation_matrix = np.float32([[1, 0, 0], [0, 1, pixel_shift]])

    # Perform the shift
    shifted = cv2.warpAffine(image, translation_matrix, (image.shape[1], image.shape[0]))
    return shifted.reshape(image_shape)


def rotate(image: NDArray, rotation_range: float, image_size: tuple[int, int] = (28, 28)):
    """
    Rotates the image by a random amount between -rotation_range and rotation_range.

    Expects the image to be of size 28x28 (no matter the shape).
    :param image: Image to rotate
    :param rotation_range: Maximum amount to rotate the image (degrees).
    :param image_size: Size of the image. Defaults to 28 * 28
    :return: Rotated image
    """
    # Randomly choose the amount to rotate the image
    rotation_angle: float = (random.random() - 0.5) * 2 * rotation_range
    # image_shape = image.shape
    # image = image.reshape(28, 28)
    # return ndimage.rotate(image, rotation, reshape=False).reshape(image_shape)
    # Define rotation matrix
    image_shape = image.shape
    image = image.reshape(image_size)
    (height, width) = image.shape[:2]
    center = (width // 2, height // 2)
    scale = 1.0
    rotation_matrix = cv2.getRotationMatrix2D(center, rotation_angle, scale)

    # Perform the actual rotation and return the image
    rotated_square_image = cv2.warpAffine(image, rotation_matrix, (width, height))
    return rotated_square_image.reshape(image_shape)


def zoom(image: NDArray, zoom_range: float, image_size: tuple[int, int] = (28, 28)):
    """
    Zooms the image by a random amount between 1-zoom_range and 1+zoom_range.

    Expects the image to be of size 28x28 (no matter the shape).
    :param image: Image to zoom
    :param zoom_range: Maximum amount to zoom the image (percentage of the image width).
    :param image_size: Size of the image. Defaults to 28 * 28
    :return: Zoomed image
    """
    # Randomly choose the amount to zoom the image
    zoom_factor: float = 1 + (random.random() - 0.5) * 2 * zoom_range
    image_shape = image.shape
    image = image.reshape(image_size)

    height, width = image.shape[:2]  # It's also the final desired shape
    new_height, new_width = int(height * zoom_factor), int(width * zoom_factor)

    # Crop only the part that will remain in the result (more efficient)
    # Centered bbox of the final desired size in resized (larger/smaller) image coordinates
    y1, x1 = max(0, new_height - height) // 2, max(0, new_width - width) // 2
    y2, x2 = y1 + height, x1 + width
    bbox = np.array([y1, x1, y2, x2])
    # Map back to original image coordinates
    bbox = (bbox / zoom_factor).astype(np.int_)
    y1, x1, y2, x2 = bbox
    cropped_image = image[y1:y2, x1:x2]

    # Handle padding when downscaling
    resize_height, resize_width = min(new_height, height), min(new_width, width)
    pad_height1, pad_width1 = (height - resize_height) // 2, (width - resize_width) // 2
    pad_height2, pad_width2 = (height - resize_height) - pad_height1, (width - resize_width) - pad_width1
    pad_spec = [(pad_height1, pad_height2), (pad_width1, pad_width2)] + [(0, 0)] * (image.ndim - 2)

    result = cv2.resize(cropped_image, (resize_width, resize_height))
    # noinspection PyTypeChecker
    result = np.pad(result, pad_spec, mode='constant')
    assert result.shape[0] == height and result.shape[1] == width
    result = np.reshape(result, image_shape)
    return result
