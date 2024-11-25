import numpy as np
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
