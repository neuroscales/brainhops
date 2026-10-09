"""
Private converters into the ITK NIfTI field format.

These converters make a conversion into the format, such as
`t.to(ItkNiftiField)`, fail with a reason instead of building a wrong
object. The format derives its coordinate systems from the content it
holds, so it has no endpoints to check. Without these converters, the
converters of the data model would rebuild a transformation as an
`ItkNiftiField` with the same parameters, and the result would not be
the map that the format says it holds. Each family that would be rebuilt
in this way is refused until an exact conversion is written (#312). An
`ItkNiftiField` that is converted to its own class is changed by the
rules of its family, as any other transformation of that family is.
"""

import typing_extensions as tx

from brainhops.datamodel import transformations as _xforms
from brainhops.datamodel._transformations.compute.convert import converter
from brainhops.datamodel._transformations.compute.converters import (
    smart_replace,
)
from brainhops.io.transformations.base._conversions import no_exact_conversion

from ._fields import ItkNiftiField


@converter
def _(
    t: _xforms.Sequence, cls: tx.Type[ItkNiftiField], **kwargs
) -> ItkNiftiField:
    # ITK stores a field in LPS on the grid of its file, and no exact
    # conversion into it is written yet.
    raise no_exact_conversion(t, cls)


@converter
def _(
    t: ItkNiftiField, cls: tx.Type[ItkNiftiField], **kwargs
) -> ItkNiftiField:
    # A transformation that is already in this format is changed by the
    # rules of any chain, and these rules keep its type. A conversion
    # into another variant of the format is not written yet.
    if not isinstance(t, cls):
        raise no_exact_conversion(t, cls)
    return smart_replace(t, cls, **kwargs)
