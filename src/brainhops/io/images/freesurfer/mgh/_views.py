"""How the data of an MGH image is decoded from the array that the file
stores, and encoded back.

An MGH image stores its voxels as the data model holds them: three
spatial axes and an optional axis of frames, in Fortran order. The pair
of functions below is therefore the identity, except that a nibabel
proxy, which reads the voxels of the file, is wrapped in a lazy array of
brainhops. The two functions are pure and invert each other, so that
`_image_to_model(_image_to_disk(data))` gives `data` back.
"""

# dependencies
from nibabel.arrayproxy import ArrayProxy

# internals
from brainhops._core.typing import ArrayProtocol
from brainhops.io.base._utils_proxies import NibabelProxy


def _image_to_model(raw: ArrayProtocol) -> ArrayProtocol:
    """Return the data of an image from the array that the file stores.

    A nibabel proxy is wrapped in a lazy array (see
    [`NibabelProxy`][brainhops.io.base._utils_proxies.NibabelProxy]),
    which reads nothing until a computation needs its values. The
    `data` of the image returns that lazy array. Any other array is
    returned as it is.
    """
    if not isinstance(raw, ArrayProxy):
        return raw
    return NibabelProxy(raw)


def _image_to_disk(data: ArrayProtocol) -> ArrayProtocol:
    """Return the array that the file of an image stores from its data.

    An image stores its data as it is, so the array is returned unchanged.
    """
    return data
