"""
Private converters into the FLIRT format.

These converters make a conversion into the format, such as
`t.to(FlirtTransform)`, fail with a reason instead of building a wrong
object. The matrix of a `FlirtTransform` is derived from a raw matrix
and from the geometry of two images, and a conversion is not given
those images. Each family that the converters of the data model would
otherwise rebuild as a `FlirtTransform` is therefore refused. A
`FlirtTransform` that is converted to its own class is changed by the
rules of any affine.
"""

import typing_extensions as tx

from brainhops.datamodel import transformations as _xforms
from brainhops.datamodel._transformations.compute.convert import converter
from brainhops.datamodel._transformations.compute.converters import (
    _convert_withlog,
)
from brainhops.io.transformations.base._conversions import unrepresentable

from ._xform import FlirtTransform


@converter(_xforms.Identity, FlirtTransform)
@converter(_xforms.Translation, FlirtTransform)
@converter(_xforms.Scaling, FlirtTransform)
@converter(_xforms.Permutation, FlirtTransform)
@converter(_xforms.Linear, FlirtTransform)
@converter(_xforms.AffineExponential, FlirtTransform)
@converter(_xforms.SubspaceTransformation, FlirtTransform)
@converter(_xforms.Sequence, FlirtTransform)
@converter
def _(
    t: _xforms.Affine, cls: tx.Type[FlirtTransform], **kwargs
) -> FlirtTransform:
    # The RAS endpoints of a FLIRT transform would accept an affine
    # rebuilt with the same parameters, but the matrix of the transform
    # is not stored. It is derived from the raw matrix and the images.
    raise unrepresentable(
        t,
        cls,
        "a FLIRT matrix maps the scaled millimetres of a moving and a "
        "reference image, which a conversion is not given. Build the "
        "FlirtTransform from its raw matrix and its two images instead.",
    )


@converter
def _(
    t: FlirtTransform, cls: tx.Type[FlirtTransform], **kwargs
) -> FlirtTransform:
    # A transformation that is already in this format is changed by the
    # rules of any affine, for example when it is given a new `matrix=`,
    # and these rules keep its type.
    return _convert_withlog(t, cls, "matrix", **kwargs)
