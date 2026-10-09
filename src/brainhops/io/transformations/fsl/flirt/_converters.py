"""
Private converters into the FLIRT format.

Each family that the converters of the data model would otherwise
rebuild as a `FlirtTransform` is refused here until an exact conversion
is written (#312), so that a transformation is never relabelled as the
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
    # FLIRT stores an affine between the scaled voxels of two images, and
    # nothing converts to it yet.
    raise no_exact_conversion(t, cls)


@converter
def _(
    t: FlirtTransform, cls: tx.Type[FlirtTransform], **kwargs
) -> FlirtTransform:
    # Within its own format, an affine is changed by the rules of any
    # affine (a new `matrix=`, say), which keep its type.
    return _convert_withlog(t, cls, "matrix", **kwargs)
