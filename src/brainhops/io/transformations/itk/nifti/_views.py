"""How the data of an ITK vector field is decoded from the array that the
NIfTI file stores, and encoded back.

The two functions convert between `rawdata`, the array as the file stores it,
and `data`, the array that the data model exposes. They are pure and
invert each other, so that `_itk_field_to_model(_itk_field_to_disk(data))`
gives `data` back.

ITK stores a vector field in five dimensions, and it pads a field with
fewer than three spatial dimensions with singleton axes: a 2-D field has
shape (X, Y, 1, 1, 2), and a 3-D field (X, Y, Z, 1, 3). The data model
holds them with shapes (X, Y, 2) and (X, Y, Z, 3).
"""

# internals
from brainhops._core.backends import array_backend
from brainhops._core.typing import ArrayProtocol


def _itk_field_to_model(raw: ArrayProtocol) -> ArrayProtocol:
    """Return the data of an ITK vector field from the array that the file
    stores.

    The singleton axes between the spatial axes and the components are
    dropped from a five-dimensional array whose last axis has two or three
    components. They are dropped by indexing them at zero, so that the
    lazy array of a file stays lazy and reads nothing. Any other array is
    returned as it is.
    """
    data = raw
    shape = tuple(int(d) for d in data.shape)
    if len(shape) != 5 or shape[4] not in (2, 3):
        return data
    ndim = shape[4]
    if any(size != 1 for size in shape[ndim:4]):
        return data
    index = (slice(None),) * ndim + (0,) * (4 - ndim) + (slice(None),)
    return data[index]


def _itk_field_to_disk(data: ArrayProtocol) -> ArrayProtocol:
    """Return the array that the file of an ITK vector field stores from
    its data.

    A field with two or three components, one per spatial axis, receives
    the singleton axes that pad it to five dimensions. Any other array is
    returned as it is.
    """
    shape = tuple(int(d) for d in data.shape)
    ndim = len(shape) - 1
    if ndim not in (2, 3) or shape[-1] != ndim:
        return data
    singletons = (1,) * (4 - ndim)
    xp = array_backend(data)
    return xp.reshape(data, (*shape[:ndim], *singletons, ndim))
