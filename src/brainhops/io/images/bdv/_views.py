"""How the data of a BigDataViewer image is decoded from the array that
the HDF5 file stores, and encoded back.

The HDF5 file stores the voxels of a view in the order z, y, x, and the
lazy array of an image presents them in that order. The data of the
image are in the order x, y, z of ImgLib2, so the two functions below
reverse the axes. The reversal does not depend on the record, so it
belongs to this pair of pure functions rather than to the lazy array.
The functions invert each other, so that
`_image_to_model(_image_to_disk(data))` gives `data` back.
"""

# internals
from brainhops._core.typing import ArrayProtocol
from brainhops.io.base._utils_proxies import load_proxy
from brainhops.io.common.bdv._hdf5 import _BdvProxy


def _image_to_model(raw: ArrayProtocol) -> ArrayProtocol:
    """Return the data of an image from the array that the file stores.

    The axes are reversed. A lazy array is first read with the array
    backend, which loads it with NumPy and keeps it lazy with Dask. The
    NumPy array that it gives is made read-only, because it is a view
    that is never written back, so an edit in place would be lost. Any
    other array is reversed as it is, so that the data of an image whose
    data were set is a view of `rawdata`.
    """
    if isinstance(raw, _BdvProxy):
        raw = load_proxy(raw)
    return raw.T


def _image_to_disk(data: ArrayProtocol) -> ArrayProtocol:
    """Return the array that the file of an image stores from its data.

    The axes are reversed, to the order z, y, x of the file.
    """
    return data.T
