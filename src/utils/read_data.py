import cupy as np
from nptyping import NDArray


def read_data() -> tuple[NDArray, NDArray, NDArray, NDArray]:
    training_images = read_training_data()
    training_labels = read_training_labels()

    test_images = read_test_data()
    test_labels = read_test_labels()

    # Bring images into (#images, 28, 28) shape
    training_images = training_images.reshape(training_images.shape[0], 28, 28)
    test_images = test_images.reshape(test_images.shape[0], 28, 28)

    return training_images, training_labels, test_images, test_labels


def read_training_labels() -> NDArray:
    f = open('./data/train-labels.idx1-ubyte', 'rb')

    # Read header (8 bytes)
    f.read(8)

    # The rest of the data are the labels. Each label is one byte
    labels_training = np.array([x for x in f.read()])
    return labels_training


def read_training_data() -> NDArray:
    f = open('./data/train-images.idx3-ubyte', 'rb')

    # Read header (16 bytes)
    f.read(16)

    # The rest of the data are the images.
    # The intensity on the grey scale of each pixel is stored in one byte
    # The images are of size 28 x 28 which is why the take up 784 adjacent bytes
    images_training = []
    while True:
        image = f.read(784)
        if len(image) != 784:
            break
        else:
            images_training.append([x for x in image])

    # The rest of the data are the labels. Each label is one byte
    images_training = np.array(images_training)
    return images_training


def read_test_labels() -> NDArray:
    f = open('./data/t10k-labels.idx1-ubyte', 'rb')

    f.read(8)

    labels_testing = np.array([x for x in f.read()])
    return labels_testing


def read_test_data() -> NDArray:
    f = open('./data/t10k-images.idx3-ubyte', 'rb')

    # Read header (16 bytes)
    f.read(16)

    images_testing = []
    while True:
        image_test = f.read(784)
        if len(image_test) != 784:
            break
        else:
            images_testing.append([x for x in image_test])

    images_testing = np.array(images_testing)
    return images_testing


def to_categorical(labels: NDArray) -> NDArray:
    """
    Converts input to one-hot encoding. Does this by creating a 2D array with 1s at the index of the label and 0s elsewhere.
    Then creates a new NDArray with the corresponding row for each entry in the labels NDArray.
    :param labels: Labels input to be converted
    :return: One-hot encoding of the input
    """
    return np.eye(10)[labels]
