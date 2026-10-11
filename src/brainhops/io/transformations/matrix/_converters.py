"""
Private converter of a bare matrix to its own format.

A bare matrix has no coordinate systems of its own: they are given to it
when it is read. Another transformation is therefore relabelled as a
matrix by the converter of [`brainhops.io.transformations.base`][], and
keeps its endpoints. A matrix that is converted to its own class, or to
another container of a bare matrix, is changed by the rules of any
affine, which keep its type.
"""

import typing_extensions as tx

from brainhops.datamodel._transformations.compute.convert import converter
from brainhops.datamodel._transformations.compute.converters import (
    _convert_withlog,
)

from ._xform import MatrixAffine


@converter
def _(t: MatrixAffine, cls: tx.Type[MatrixAffine], **kwargs) -> MatrixAffine:
    # Within the format, a matrix is changed by the rules of any affine (a
    # new `matrix=`, say), which keep its type.
    return _convert_withlog(t, cls, "matrix", **kwargs)
