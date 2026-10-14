"""How the data of a NIfTI transformation is decoded from the array that
the file stores, and encoded back.

Each pair of functions below converts between `rawdata`, the array as
the file stores it, and `data`, the array that the data model exposes.
The two functions of a pair are pure and invert each other, so that
`to_model(to_disk(data))` gives `data` back. The pairs are used by the
NIfTI transformations of this package and by the other formats that
store a field as a NIfTI vector image, such as the NiftyReg, SPM and FSL
fields.
"""

# internals
from brainhops._core.affines import to_homogeneous
from brainhops._core.typing import ArrayProtocol
from brainhops.backends import get_array_backend
from brainhops.io.common.nifti._files import _read

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
