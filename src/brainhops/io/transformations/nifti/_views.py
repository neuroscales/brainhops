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

# dependencies
import typing_extensions as tx

# core
from brainhops._core._internal_api import (
    ArrayProtocol,
    array_backend,
    to_homogeneous,
)
from brainhops.errors import ParserContentError, WriterError

# ----------------------------------------------------------------------
#   VECTOR FIELDS
# ----------------------------------------------------------------------
# The NIfTI standard stores a field of vectors as a five-dimensional array
# of shape (X, Y, Z, 1, C): the fourth axis is time, and it has length
# one, so that the components lie along the fifth axis. The data model
# holds the same field with shape (X, Y, Z, C). A file whose fourth axis
# is longer than one holds a field for each volume of a time series,
# which the data model cannot represent yet (batched transformations,
# #576), so such an array is refused in both directions rather than
# read as a four-dimensional grid of vectors.


def _check_single_field(
    array: tx.Optional[ArrayProtocol], error: tx.Type[Exception]
) -> None:
    """Refuse an array that stores one vector field per volume.

    A five-dimensional array whose fourth axis is longer than one holds
    a field for each volume of a time series. The data model does not
    support such a batch of fields yet, so `error` is raised. Any other
    array, and a missing array, are accepted.

    Parameters
    ----------
    array : ArrayProtocol or None
        The array that the file stores, or that is about to be written.
    error : type[Exception]
        The class of the exception to raise, which is
        `ParserContentError` when a file is read and `WriterError` when
        a file is written.

    Raises
    ------
    Exception
        The class given by `error`, if the array holds more than one
        field.
    """
    if array is None:
        return
    shape = tuple(int(d) for d in array.shape)
    if len(shape) == 5 and shape[3] != 1:
        raise error(
            f"A NIfTI vector field is stored as a (X, Y, Z, 1, C) array, "
            f"and this array has the shape {shape}, which holds a field "
            f"for each of its {shape[3]} volumes. A field per volume is "
            f"not supported yet (batched transformations, #576)."
        )


def _field_to_model(raw: ArrayProtocol) -> ArrayProtocol:
    """Return the data of a vector field from the array that the file
    stores.

    The singleton time axis of a five-dimensional array is dropped, and
    any other array is returned as it is. A five-dimensional array whose
    time axis is longer than one holds a field per volume, which is not
    supported yet, so it is refused.

    Raises
    ------
    ParserContentError
        If the array holds a field for each of several volumes.
    """
    _check_single_field(raw, ParserContentError)
    data = raw
    shape = tuple(int(d) for d in data.shape)
    if len(shape) == 5:
        data = data[:, :, :, 0, :]
    return data


def _field_to_disk(data: ArrayProtocol) -> ArrayProtocol:
    """Return the array that the file of a vector field stores from its
    data.

    A four-dimensional field receives the singleton time axis before its
    components, and any other array is returned as it is. A
    five-dimensional array whose fourth axis is longer than one holds a
    field per volume, which is not supported yet, so it is refused
    rather than written.

    Raises
    ------
    WriterError
        If the array holds a field for each of several volumes.
    """
    if len(data.shape) == 4:
        return array_backend(data).expand_dims(data, axis=3)
    _check_single_field(data, WriterError)
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
    return to_homogeneous(array_backend(data).asarray(data))
