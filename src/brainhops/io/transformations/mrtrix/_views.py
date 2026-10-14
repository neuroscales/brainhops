"""How the vectors of an MRtrix warp are decoded from the array that the
file stores, and encoded back.

The lazy array of an MRtrix file already presents the values in the axis
order of the header, whatever `layout` the file uses, and it applies the
intensity `scaling` of the header when its values are read. Both depend
on the record, so they belong to the lazy array and not to the pair of
functions below, which is therefore the identity, except that the lazy
array is read with the array backend. The backend loads it with NumPy
and keeps it lazy with Dask. The two functions are pure and invert each
other, so that `_warp_to_model(_warp_to_disk(data))` gives `data` back.
"""

# dependencies
import numpy as np

# internals
from brainhops._core.typing import ArrayProtocol
from brainhops.backends import get_array_backend
from brainhops.io.common.mrtrix._data import _MrtrixProxy


def _warp_to_model(raw: ArrayProtocol) -> ArrayProtocol:
    """Return the vectors of a warp from the array that the file stores.

    A lazy array is read with the array backend. The NumPy array that it
    gives is made read-only, because the writer writes `rawdata` and not this
    array, so an edit in place would be lost. Any other array is returned
    as it is, so that a Dask array stays lazy and an array that was set as
    data stays the array of `rawdata`.
    """
    if not isinstance(raw, _MrtrixProxy):
        return raw
    data = get_array_backend().asarray(raw)
    if isinstance(data, np.ndarray):
        data.flags.writeable = False
    return data


def _warp_to_disk(data: ArrayProtocol) -> ArrayProtocol:
    """Return the array that the file of a warp stores from its vectors.

    A warp stores its vectors as they are, so the array is returned
    unchanged.
    """
    return data
