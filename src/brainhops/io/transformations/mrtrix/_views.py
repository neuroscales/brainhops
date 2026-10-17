"""How the vectors of an MRtrix warp are decoded from the array that the
file stores, and encoded back.

The lazy array of an MRtrix file already presents the values in the axis
order of the header, whatever `layout` the file uses, and it applies the
intensity `scaling` of the header when its values are read. Both depend
on the record, so they belong to the lazy array and not to the pair of
functions below, which is therefore the identity. The two functions are
pure and invert each other, so that
`_warp_to_model(_warp_to_disk(data))` gives `data` back.
"""

# internals
from brainhops._core.typing import ArrayProtocol


def _warp_to_model(raw: ArrayProtocol) -> ArrayProtocol:
    """Return the vectors of a warp from the array that the file stores.

    The array is returned as it is, so that the lazy array of a file
    stays lazy and reads nothing. The `data` of the object reads it
    once, as the backend setting says.
    """
    return raw


def _warp_to_disk(data: ArrayProtocol) -> ArrayProtocol:
    """Return the array that the file of a warp stores from its vectors.

    A warp stores its vectors as they are, so the array is returned
    unchanged.
    """
    return data
