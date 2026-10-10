"""How the data of an MGH image is decoded from the array that the file
stores, and encoded back.

An MGH image stores its voxels as the data model holds them: three
spatial axes and an optional axis of frames, in Fortran order. The pair
of functions below is therefore the identity, except that a nibabel
proxy, which reads the voxels lazily, is read with the array backend.
The backend loads the proxy with NumPy and keeps it lazy with Dask. The
two functions are pure and invert each other, so that
`_image_to_model(_image_to_disk(data))` gives `data` back.
"""

# dependencies
import numpy as np
from nibabel.arrayproxy import ArrayProxy

# internals
from brainhops._core.typing import ArrayProtocol
from brainhops.backends import get_array_backend


def _image_to_model(raw: ArrayProtocol) -> ArrayProtocol:
    """Return the data of an image from the array that the file stores.

    A proxy is read with the array backend. The NumPy array that it gives
    is made read-only, because it is a view that the writer does not read
    back: the writer writes `raw`, so an edit in place would be lost. Any
    other array is returned as it is, so that a Dask array stays lazy and
    an array that was set as data stays the array of `raw`.
    """
    if not isinstance(raw, ArrayProxy):
        return raw
    data = get_array_backend().asarray(raw)
    if isinstance(data, np.ndarray):
        data.flags.writeable = False
    return data


def _image_to_disk(data: ArrayProtocol) -> ArrayProtocol:
    """Return the array that the file of an image stores from its data.

    An image stores its data as it is, so the array is returned unchanged.
    """
    return data
