"""How the data of a NRRD image is decoded from the array that the file
stores, and encoded back.

The lazy array of a NRRD image already presents the samples in the axis
order of the image, spatial axes first, whatever the order of the file.
That order depends on the record, so it belongs to the lazy array and
not to the pair of functions below, which is therefore the identity. The
two functions are pure and invert each other, so that
`image_to_model(image_to_disk(data))` gives `data` back.
"""

# internals
from brainhops._core.typing import ArrayProtocol


def image_to_model(raw: ArrayProtocol) -> ArrayProtocol:
    """Return the data of an image from the array that the file stores.

    The array is returned as it is, so that the lazy array of a file
    stays lazy and reads nothing. The `data` of the object returns that
    lazy array, which reads its values only when a computation needs
    them.
    """
    return raw


def image_to_disk(data: ArrayProtocol) -> ArrayProtocol:
    """Return the array that the file of an image stores from its data.

    An image stores its data as it is, so the array is returned unchanged.
    """
    return data
