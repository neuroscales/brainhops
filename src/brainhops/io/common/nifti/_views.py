"""How the data of a NIfTI object is decoded from the array that the file
stores, and encoded back.

Each pair of functions below converts between `raw`, the array as the
file stores it, and `data`, the array that the data model exposes. The
two functions of a pair are pure and invert each other, so that
`to_model(to_disk(data))` gives `data` back. A nibabel proxy, which reads
the voxels lazily, is read with the array backend, which loads it with
NumPy and keeps it lazy with Dask.
"""

# dependencies
import numpy as np
from nibabel.arrayproxy import ArrayProxy

# internals
from brainhops._core.affines import to_homogeneous
from brainhops._core.typing import ArrayProtocol
from brainhops.backends import get_array_backend


def _read(raw: ArrayProtocol) -> ArrayProtocol:
    """Return an array from a stored array, which may be a nibabel proxy.

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


# ----------------------------------------------------------------------
#   IMAGES
# ----------------------------------------------------------------------


def _image_to_model(raw: ArrayProtocol) -> ArrayProtocol:
    """Return the data of an image from the array that the file stores.

    An image stores its data as it is, so the stored array is only read.
    """
    return _read(raw)


def _image_to_disk(data: ArrayProtocol) -> ArrayProtocol:
    """Return the array that the file of an image stores from its data.

    An image stores its data as it is, so the array is returned unchanged.
    """
    return data


# ----------------------------------------------------------------------
#   VECTOR FIELDS
# ----------------------------------------------------------------------
# The NIfTI standard stores a field of vectors as a five-dimensional array
# of shape (X, Y, Z, 1, C): the fourth axis is time, and it has length
# one, so that the components lie along the fifth axis. The data model
# holds the same field with shape (X, Y, Z, C).


def _field_to_model(raw: ArrayProtocol) -> ArrayProtocol:
    """Return the data of a vector field from the array that the file
    stores.

    The singleton time axis of a five-dimensional array is dropped, and
    any other array is returned as it is.
    """
    data = _read(raw)
    shape = tuple(int(d) for d in data.shape)
    if len(shape) == 5 and shape[3] == 1:
        data = data[:, :, :, 0, :]
    return data


def _field_to_disk(data: ArrayProtocol) -> ArrayProtocol:
    """Return the array that the file of a vector field stores from its
    data.

    A four-dimensional field receives the singleton time axis before its
    components, and any other array is returned as it is.
    """
    if len(data.shape) == 4:
        return get_array_backend(data).expand_dims(data, axis=3)
    return data


# ----------------------------------------------------------------------
#   ITK VECTOR FIELDS
# ----------------------------------------------------------------------
# ITK stores a vector field in five dimensions too, and it pads a field
# with fewer than three spatial dimensions with singleton axes: a 2-D
# field has shape (X, Y, 1, 1, 2), and a 3-D field (X, Y, Z, 1, 3). The
# data model holds them with shapes (X, Y, 2) and (X, Y, Z, 3).


def _itk_field_to_model(raw: ArrayProtocol) -> ArrayProtocol:
    """Return the data of an ITK vector field from the array that the file
    stores.

    The singleton axes between the spatial axes and the components are
    dropped from a five-dimensional array whose last axis has two or three
    components. Any other array is returned as it is.
    """
    data = _read(raw)
    shape = tuple(int(d) for d in data.shape)
    if len(shape) != 5 or shape[4] not in (2, 3):
        return data
    ndim = shape[4]
    if any(size != 1 for size in shape[ndim:4]):
        return data
    return get_array_backend(data).reshape(data, (*shape[:ndim], ndim))


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
    backend = get_array_backend(data)
    return backend.reshape(data, (*shape[:ndim], *singletons, ndim))


# ----------------------------------------------------------------------
#   AFFINES
# ----------------------------------------------------------------------
# A NIfTI affine that is set as data is held as its (4, 4) homogeneous
# matrix, the form that the header stores. The data model holds the
# compact (3, 4) matrix, without the last row.


def _affine_to_model(raw: ArrayProtocol) -> ArrayProtocol:
    """Return the compact matrix of an affine from its homogeneous matrix."""
    return raw[:-1]


def _affine_to_disk(data: ArrayProtocol) -> ArrayProtocol:
    """Return the homogeneous matrix of an affine from its compact matrix."""
    return to_homogeneous(get_array_backend(data).asarray(data))
