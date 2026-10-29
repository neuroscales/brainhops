"""How the sub-bricks of an AFNI warp are decoded from the array that the
BRIK stores, and encoded back.

The lazy array of a BRIK already has the shape of the dataset, and it
applies the scale factors of the header when its values are read. Both
depend on the record, so they belong to the lazy array and not to the
pair of functions below, which is therefore the identity. The two
functions are pure and invert each other, so that
`warp_to_model(warp_to_disk(data))` gives `data` back.
"""

# internals
from brainhops._core.typing import ArrayProtocol


def warp_to_model(raw: ArrayProtocol) -> ArrayProtocol:
    """Return the sub-bricks of a warp from the array that the BRIK stores.

    The array is returned as it is, so that the lazy array of a file
    stays lazy and reads nothing until a computation needs its values.
    The `data` of the object returns that lazy array.
    """
    return raw


def warp_to_disk(data: ArrayProtocol) -> ArrayProtocol:
    """Return the array that the BRIK of a warp stores from its sub-bricks.

    A warp stores its sub-bricks as they are, so the array is returned
    unchanged.
    """
    return data
