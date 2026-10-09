"""How the data of a NIfTI object is decoded from the array that the file
stores, and encoded back.

Each pair of functions below converts between `raw`, the array as the
file stores it, and `data`, the array that the data model exposes. The
two functions of a pair are pure and invert each other, so that
`to_model(to_disk(data))` gives `data` back.
"""

# dependencies
from nibabel.arrayproxy import ArrayProxy

# internals
from brainhops._core.typing import ArrayProtocol
from brainhops.backends import get_array_backend


def _image_to_model(raw: ArrayProtocol) -> ArrayProtocol:
    """Return the data of an image from the array that the file stores.

    An image stores its data as it is. The proxy of a file, which reads
    the voxels lazily, is read with the array backend, which loads it with
    NumPy and keeps it lazy with Dask. Any other array is returned as it
    is, so that an image built from a Dask array keeps it lazy.
    """
    if isinstance(raw, ArrayProxy):
        return get_array_backend().asarray(raw)
    return raw


def _image_to_disk(data: ArrayProtocol) -> ArrayProtocol:
    """Return the array that the file of an image stores from its data.

    An image stores its data as it is, so the array is returned unchanged.
    """
    return data
