"""How the data of a Zarr image is decoded from the array that the store
keeps, and encoded back.

A Zarr image stores its data as it is, so the pair of functions below is
the identity, except that the lazy array is read with the array backend.
The NumPy backend reads it, and the Dask backend keeps it lazy, with one
block per chunk of the store. The two functions are pure and invert each
other, so that `_image_to_model(_image_to_disk(data))` gives `data` back.
"""

# dependencies
import numpy as np

# internals
from brainhops._core.dependencies import da
from brainhops._core.typing import ArrayProtocol
from brainhops.backends import get_array_backend
from brainhops.io.common.zarr._data import _ZarrProxy


def _image_to_model(raw: ArrayProtocol) -> ArrayProtocol:
    """Return the data of an image from the array that the store keeps.

    A lazy array is read with the array backend. The NumPy array that it
    gives is made read-only, because it is a view that the writer does
    not read back: the writer writes `rawdata`, so an edit in place
    would be lost. The Dask backend gets a Dask array that reads the
    chunks of the store when it is computed. Any other array is returned
    as it is, so that an array that was set as data stays the array of
    `rawdata`.
    """
    if not isinstance(raw, _ZarrProxy):
        return raw
    backend = get_array_backend()
    if da is not None and backend is da:
        return raw.to_dask()
    data = backend.asarray(raw)
    if isinstance(data, np.ndarray):
        data.flags.writeable = False
    return data


def _image_to_disk(data: ArrayProtocol) -> ArrayProtocol:
    """Return the array that the store of an image keeps from its data.

    An image stores its data as it is, so the array is returned unchanged.
    """
    return data
