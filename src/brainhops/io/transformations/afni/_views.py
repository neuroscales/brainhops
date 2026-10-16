"""How the sub-bricks of an AFNI warp are decoded from the array that the
BRIK stores, and encoded back.

The lazy array of a BRIK already has the shape of the dataset, and it
applies the scale factors of the header when its values are read. Both
depend on the record, so they belong to the lazy array and not to the
pair of functions below, which is therefore the identity, except that
the lazy array is read with the array backend. The backend loads it with
NumPy and keeps it lazy with Dask. The two functions are pure and invert
each other, so that `_warp_to_model(_warp_to_disk(data))` gives `data`
back.
"""

# internals
from brainhops._core.typing import ArrayProtocol
from brainhops.io.base._utils_proxies import load_proxy
from brainhops.io.common.afni._data import _BrikProxy


def _warp_to_model(raw: ArrayProtocol) -> ArrayProtocol:
    """Return the sub-bricks of a warp from the array that the BRIK stores.

    A lazy array is read with the array backend. The NumPy array that it
    gives is made read-only, because the writer writes `rawdata` and not this
    array, so an edit in place would be lost. Any other array is returned
    as it is, so that a Dask array stays lazy and an array that was set as
    data stays the array of `rawdata`.
    """
    if not isinstance(raw, _BrikProxy):
        return raw
    return load_proxy(raw)


def _warp_to_disk(data: ArrayProtocol) -> ArrayProtocol:
    """Return the array that the BRIK of a warp stores from its sub-bricks.

    A warp stores its sub-bricks as they are, so the array is returned
    unchanged.
    """
    return data
