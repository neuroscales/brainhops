"""How the data of an MGH image is decoded from the array that the file
stores, and encoded back.

An MGH image stores its voxels as the data model holds them: three
spatial axes and an optional axis of frames, in Fortran order. The lazy
array of a file already presents them in this order, so the pair of
functions below is the identity. The two functions are pure and invert
each other, so that `image_to_model(image_to_disk(data))` gives
`data` back.
"""

# internals
from brainhops._core._internal_api import ArrayProtocol


def image_to_model(raw: ArrayProtocol) -> ArrayProtocol:
    """Return the data of an image from the array that the file stores.

    The array is returned as it is, so that the lazy array of a file
    stays lazy and reads nothing until a computation needs its values.
    The `data` of the image returns that lazy array.
    """
    return raw


def image_to_disk(data: ArrayProtocol) -> ArrayProtocol:
    """Return the array that the file of an image stores from its data.

    An image stores its data as it is, so the array is returned unchanged.
    """
    return data
