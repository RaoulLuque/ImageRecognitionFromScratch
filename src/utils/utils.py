import cupy as np
from nptyping.ndarray import NDArray


def shuffle_in_unison(array_one: NDArray, array_two: NDArray) -> tuple[NDArray, NDArray]:
    """
    Shuffle two numpy arrays in unison.
    :param array_one: First array to be shuffled.
    :param array_two: Second array to be shuffled.
    :return: Tuple of shuffled arrays
    """
    assert len(array_one) == len(array_two)
    p = np.random.permutation(len(array_one))
    return array_one[p], array_two[p]


def create_batches(x_train, y_train, batch_size) -> tuple[list[NDArray], list[NDArray]]:
    number_of_batches = len(x_train) // batch_size
    x_train_rest = x_train[x_train.shape[0] - x_train.shape[0] % batch_size:]
    x_train = x_train[:x_train.shape[0] - x_train.shape[0] % batch_size]
    y_train_rest = y_train[y_train.shape[0] - y_train.shape[0] % batch_size:]
    y_train = y_train[:y_train.shape[0] - y_train.shape[0] % batch_size]
    x_train_split = np.split(x_train, number_of_batches)
    y_train_split = np.split(y_train, number_of_batches)
    if not len(x_train) % batch_size == 0:
        x_train_split.append(x_train_rest)
        y_train_split.append(y_train_rest)
    return x_train_split, y_train_split
