"""
Private converters into the FNIRT format.

The format derives its coordinate systems from what it holds, so it has
no endpoints to check, and a transformation relabelled as it would not
be what the format says it holds. Each family that the converters of the
data model would otherwise rebuild as a `FnirtWarpField` is therefore
refused here, until an exact conversion is written (#312). A format
converted to its own class is changed as any transformation of its
family is.
"""

import typing_extensions as tx

from brainhops.datamodel import transformations as _xforms
from brainhops.datamodel._transformations.compute.convert import converter
from brainhops.datamodel._transformations.compute.converters import (
    smart_replace,
)
from brainhops.io.transformations.base._conversions import no_exact_conversion

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
