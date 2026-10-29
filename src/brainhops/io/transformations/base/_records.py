"""
Private helpers of the affine formats whose matrix lives in their record.

Some affine formats store a file so short that it is read in one pass,
such as an LTA file or a NiftyReg affine. The whole file is then held as
a record by the metadata of the transformation, and the matrix of the
affine lives in that record rather than in a `rawdata` array. These
formats read their record, decode their matrix and encode a new matrix
in the same way, so the helpers that do it are written once here and
called explicitly by each format.

The record of such a format stores the homogeneous matrix of a 3-D
affine as a tuple of rows of floats, and the data model of an affine
holds the same matrix without its last row.
"""

__all__ = [
    "matrix_to_disk",
    "matrix_to_model",
    "read_only",
    "record_of",
]

import numpy as np
import typing_extensions as tx

from brainhops._core.typing import ArrayProtocol


def record_of(xform: tx.Any) -> tx.Any:
    """Return the record of the metadata of a transformation, or `None`.

    Parameters
    ----------
    xform : Transformation
        A transformation format that holds its file as metadata.

    Returns
    -------
    object
        The record of the metadata, or `None` when the transformation has
        no metadata or its metadata has no record.
    """
    metadata = xform.metadata
    return None if metadata is None else metadata.raw


def read_only(matrix: tx.Any) -> np.ndarray:
    """Return a copy of a matrix decoded from a record, made read-only.

    The decoded matrix is cached, but a writer reads the record and not
    the cached matrix, so an edit made in place would be lost. The copy is
    therefore read-only by construction, whatever the flags of the record,
    and a new matrix is assigned as data instead, which stays pending until
    the transformation is consolidated.

    Parameters
    ----------
    matrix : array_like
        The matrix decoded from a record.

    Returns
    -------
    np.ndarray
        A read-only copy of the matrix, in double precision.
    """
    matrix = np.array(matrix, dtype=np.float64)
    matrix.flags.writeable = False
    return matrix


def matrix_to_model(matrix: tx.Any) -> np.ndarray:
    """Return the data of an affine from the homogeneous matrix of a record.

    The record stores the homogeneous matrix, whose last row is dropped.

    Parameters
    ----------
    matrix : array_like
        The homogeneous matrix that the record stores.

    Returns
    -------
    np.ndarray
        The matrix without its last row.
    """
    return np.asarray(matrix, dtype=np.float64)[:-1]


def matrix_to_disk(data: tx.Optional[ArrayProtocol]) -> tx.Tuple:
    """Return the homogeneous matrix that a record stores for an affine.

    The record stores the homogeneous matrix as a tuple of rows of floats.
    Data of `None`, which the data model reads as the identity, is stored
    as the identity of a 3-D affine.

    Parameters
    ----------
    data : array_like, optional
        The matrix of the affine without its last row.

    Returns
    -------
    tuple of tuple of float
        The rows of the homogeneous matrix.
    """
    if data is None:
        homogeneous = np.eye(4)
    else:
        data = np.asarray(data, dtype=np.float64)
        last = np.zeros((1, data.shape[-1]))
        last[0, -1] = 1.0
        homogeneous = np.concatenate([data, last])
    return tuple(tuple(float(v) for v in row) for row in homogeneous)
