"""How the data of a Zarr image is decoded from the array that the store
keeps, and encoded back.

A Zarr image stores its data as it is, so the pair of functions below is
the identity. The two functions are pure and invert each
other, so that `image_to_model(image_to_disk(data))` gives `data` back.
"""

# internals
from brainhops._core.typing import ArrayProtocol


def image_to_model(raw: ArrayProtocol) -> ArrayProtocol:
    """Return the data of an image from the array that the store keeps.

    The array is returned as it is, so that the lazy array of a file
    stays lazy and reads nothing. The `data` of the object returns that
    lazy array, which reads its values only when a computation needs
    them.
    """
    return raw


def image_to_disk(data: ArrayProtocol) -> ArrayProtocol:
    """Return the array that the store of an image keeps from its data.

    An image stores its data as it is, so the array is returned unchanged.
    """
    return data
