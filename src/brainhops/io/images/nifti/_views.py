"""How the voxels of a NIfTI image are decoded from the array that the
file stores, and encoded back.

The two functions convert between `rawdata`, the array as the file stores it,
and `data`, the array that the data model exposes. They are pure and
invert each other, so that `image_to_model(image_to_disk(data))` gives
`data` back.
"""

# internals
from brainhops._core._internal_api import ArrayProtocol


def image_to_model(raw: ArrayProtocol) -> ArrayProtocol:
    """Return the data of an image from the array that the file stores.

    An image stores its data as it is, so the array is returned as it
    is. The lazy array of a file therefore stays lazy and reads nothing
    until a computation needs its values, and the `data` of the image
    returns that lazy array.
    """
    return raw


def image_to_disk(data: ArrayProtocol) -> ArrayProtocol:
    """Return the array that the file of an image stores from its data.

    An image stores its data as it is, so the array is returned unchanged.
    """
    return data
