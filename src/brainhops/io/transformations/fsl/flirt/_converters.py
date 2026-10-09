"""
Private converters into the FLIRT format.

A FLIRT matrix is derived from a raw matrix and the geometry of two
images, which a conversion is not given, so each family that the
converters of the data model would otherwise rebuild as a
`FlirtTransform` is refused here, with that reason. A `FlirtTransform`
converted to its own class is changed as any affine is.
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
    # The RAS endpoints of a FLIRT transform would accept a relabelled
    # affine, but its matrix is not stored: it is derived from the images.
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
    # Within its own format, an affine is changed by the rules of any
    # affine (a new `matrix=`, say), which keep its type.
    return _convert_withlog(t, cls, "matrix", **kwargs)
