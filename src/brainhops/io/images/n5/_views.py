"""How the data of an N5 image is decoded from the array that the dataset
stores, and encoded back.

The lazy array of a dataset presents the samples in the order of the
`dimensions` of the dataset, whose first dimension varies fastest in a
block. That order is already the brainhops order, `[x, y, z, ...]`, so
the two functions below are the identity. They invert each other, so that
`_image_to_model(_image_to_disk(data))` gives `data` back.
"""

# internals
from brainhops._core.typing import ArrayProtocol

# this format


def _image_to_model(raw: ArrayProtocol) -> ArrayProtocol:
    """Return the data of an image from the array that the dataset stores.

    The array is returned as it is, so that the lazy array of a file
    stays lazy and reads nothing. The `data` of the object returns that
    lazy array, which reads its values only when a computation needs
    them.
    """
    return raw


def _image_to_disk(data: ArrayProtocol) -> ArrayProtocol:
    """Return the array that the dataset stores from the data of an image.

    The array is the data itself.
    """
    return data
