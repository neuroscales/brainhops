"""How the data of an MRtrix image is decoded from the array that the file
stores, and encoded back.

The lazy array of an MRtrix image already presents the voxels in the
axis order of the header, whatever `layout` the file uses, and it
applies the intensity `scaling` of the header when its values are read.
Both depend on the record, so they belong to the lazy array and not to
the pair of functions below, which is therefore the identity. The two
functions are pure and invert each other, so that
`_image_to_model(_image_to_disk(data))` gives `data` back.
"""

# internals
from brainhops._core.typing import ArrayProtocol


def _image_to_model(raw: ArrayProtocol) -> ArrayProtocol:
    """Return the data of an image from the array that the file stores.

    The array is returned as it is, so that the lazy array of a file
    stays lazy and reads nothing. The `data` of the object returns that
    lazy array, which reads its values only when a computation needs
    them.
    """
    return raw


def _image_to_disk(data: ArrayProtocol) -> ArrayProtocol:
    """Return the array that the file of an image stores from its data.

    An image stores its data as it is, so the array is returned unchanged.
    """
    return data
