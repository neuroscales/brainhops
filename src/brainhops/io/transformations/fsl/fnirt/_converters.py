"""
Private converters into the FNIRT format.

These converters make a conversion into the format, such as
`t.to(FnirtWarpField)`, fail with a reason instead of building a wrong
object. The format derives its coordinate systems from the content it
holds, so it has no endpoints to check. Without these converters, the
converters of the data model would rebuild a transformation as a
`FnirtWarpField` with the same parameters, and the result would not be
the map that the format says it holds. Each family that would be rebuilt
in this way is refused until an exact conversion is written (#312). A
`FnirtWarpField` that is converted to its own class is changed by the
rules of its family, as any other transformation of that family is.
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
    # no exact conversion into it is written yet.
    raise no_exact_conversion(t, cls)


@converter
def _(
    t: FnirtWarpField, cls: tx.Type[FnirtWarpField], **kwargs
) -> FnirtWarpField:
    # A transformation that is already in this format is changed by the
    # rules of any chain, and these rules keep its type.
    return smart_replace(t, cls, **kwargs)
