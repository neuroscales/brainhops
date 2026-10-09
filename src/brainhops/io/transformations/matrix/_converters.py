"""
Private converters into the bare matrix formats.

Each family that the converters of the data model would otherwise
rebuild as a `MatrixAffine` is refused here until an exact conversion is
written (#312), so that a transformation is never relabelled as the
format. A format converted to its own class is changed as any
transformation of its family is.
"""

import typing_extensions as tx

from brainhops.datamodel import transformations as _xforms
from brainhops.datamodel._transformations.compute.convert import converter
from brainhops.datamodel._transformations.compute.converters import (
    _convert_withlog,
)
from brainhops.io.transformations.base.conversions import no_exact_conversion

from ._xform import MatrixAffine


@converter(_xforms.Identity, MatrixAffine)
@converter(_xforms.Translation, MatrixAffine)
@converter(_xforms.Scaling, MatrixAffine)
@converter(_xforms.Permutation, MatrixAffine)
@converter(_xforms.Linear, MatrixAffine)
@converter(_xforms.AffineExponential, MatrixAffine)
@converter(_xforms.SubspaceTransformation, MatrixAffine)
@converter(_xforms.Sequence, MatrixAffine)
@converter
def _(t: _xforms.Affine, cls: tx.Type[MatrixAffine], **kwargs) -> MatrixAffine:
    # A bare matrix stores an affine with the conventions it was read
    # with, and nothing converts to it yet.
    raise no_exact_conversion(t, cls)


@converter
def _(t: MatrixAffine, cls: tx.Type[MatrixAffine], **kwargs) -> MatrixAffine:
    # Within its own format, an affine is changed by the rules of any
    # affine (a new `matrix=`, say), which keep its type. Another variant
    # of the format is not converted to yet.
    if not isinstance(t, cls):
        raise no_exact_conversion(t, cls)
    return _convert_withlog(t, cls, "matrix", **kwargs)
