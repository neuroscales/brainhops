"""
Private converters into the FNIRT format.

Each family that the converters of the data model would otherwise
rebuild as a `FnirtWarpField` is refused here until an exact conversion
is written (#312), so that a transformation is never relabelled as the
format. A format converted to its own class is changed as any
transformation of its family is.
"""

import typing_extensions as tx

from brainhops.datamodel import transformations as _xforms
from brainhops.datamodel._transformations.compute.convert import converter
from brainhops.datamodel._transformations.compute.converters import (
    smart_replace,
)
from brainhops.io.transformations.base.conversions import no_exact_conversion

from ._base import FnirtWarpField


@converter
def _(
    t: _xforms.Sequence, cls: tx.Type[FnirtWarpField], **kwargs
) -> FnirtWarpField:
    # FNIRT stores a field between the scaled voxels of two images, and
    # nothing converts to it yet.
    raise no_exact_conversion(t, cls)


@converter
def _(
    t: FnirtWarpField, cls: tx.Type[FnirtWarpField], **kwargs
) -> FnirtWarpField:
    # Within its own format, a chain is changed by the rules of any chain,
    # which keep its type.
    return smart_replace(t, cls, **kwargs)
