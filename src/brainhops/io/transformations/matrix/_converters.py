"""
Private converters into the plain matrix formats.

A plain matrix file holds an affine in any coordinate systems, which the
caller states when the file is read, so a plain matrix format has no
system of its own to refuse. No exact conversion into these formats is
written yet (#312), so each family with an affine form is relabelled as
each of them by the converter
[`relabelling`][brainhops.io.transformations.base._converters.relabelling],
which this module registers. The relabelled transformation keeps the
matrix of the affine and its systems, so it is the same map. A
transformation that is converted to its own class is changed by the
rules of any affine, and one of another container is relabelled from its
decoded matrix, since the record of one container means nothing to
another.
"""

import typing_extensions as tx

from brainhops.datamodel import transformations as _xforms
from brainhops.datamodel._transformations.compute.convert import converter
from brainhops.datamodel._transformations.compute.converters import (
    _convert_withlog,
)
from brainhops.io.transformations.base._converters import relabelling

from ._xform import (
    CsvMatrixAffine,
    Mat73MatrixAffine,
    MatLegacyMatrixAffine,
    MatrixAffine,
    NpyMatrixAffine,
    NpzMatrixAffine,
    TsvMatrixAffine,
    TxtMatrixAffine,
)

_FORMATS = (
    TxtMatrixAffine,
    CsvMatrixAffine,
    TsvMatrixAffine,
    NpyMatrixAffine,
    NpzMatrixAffine,
    MatLegacyMatrixAffine,
    Mat73MatrixAffine,
)
"""The registered plain matrix formats."""

# Each family with an affine form is relabelled as each container, since
# no exact conversion into them is written yet.
for _format in _FORMATS:
    converter(_xforms.Identity, _format)(relabelling)
    converter(_xforms.Translation, _format)(relabelling)
    converter(_xforms.Scaling, _format)(relabelling)
    converter(_xforms.Permutation, _format)(relabelling)
    converter(_xforms.Linear, _format)(relabelling)
    converter(_xforms.Affine, _format)(relabelling)
    converter(_xforms.AffineExponential, _format)(relabelling)
    converter(_xforms.SubspaceTransformation, _format)(relabelling)
    converter(_xforms.Sequence, _format)(relabelling)


@converter(TxtMatrixAffine, TxtMatrixAffine)
@converter(CsvMatrixAffine, CsvMatrixAffine)
@converter(TsvMatrixAffine, TsvMatrixAffine)
@converter(NpyMatrixAffine, NpyMatrixAffine)
@converter(NpzMatrixAffine, NpzMatrixAffine)
@converter(MatLegacyMatrixAffine, MatLegacyMatrixAffine)
@converter(Mat73MatrixAffine, Mat73MatrixAffine)
def _(t: MatrixAffine, cls: tx.Type[MatrixAffine], **kwargs) -> MatrixAffine:
    # A transformation that is already in this format is changed by the
    # rules of any affine, for example when it is given a new `matrix=`,
    # and these rules keep its type and its record.
    return _convert_withlog(t, cls, "matrix", **kwargs)
