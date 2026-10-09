"""
Private converters into the M3Z format.

These converters make a conversion into the format, such as
`t.to(M3zMorph)`, fail with a reason instead of building a wrong object.
The format derives its coordinate systems from the content it holds, so
it has no endpoints to check. Without these converters, the converters
of the data model would rebuild a transformation as an `M3zMorph` with
the same parameters, and the result would not be the map that the format
says it holds. Each family that would be rebuilt in this way is refused
until an exact conversion is written (#312). An `M3zMorph` that is
converted to its own class is changed by the rules of its family, as any
other transformation of that family is.
"""

import typing_extensions as tx

from brainhops.datamodel import transformations as _xforms
from brainhops.datamodel._transformations.compute.convert import converter
from brainhops.datamodel._transformations.compute.converters import (
    smart_replace,
)
from brainhops.io.transformations.base._conversions import no_exact_conversion

from ._xform import M3zMorph


@converter
def _(t: _xforms.Sequence, cls: tx.Type[M3zMorph], **kwargs) -> M3zMorph:
    # An M3Z morph stores a field of positions on the grid of an atlas,
    # and no exact conversion into it is written yet.
    raise no_exact_conversion(t, cls)


@converter
def _(t: M3zMorph, cls: tx.Type[M3zMorph], **kwargs) -> M3zMorph:
    # A transformation that is already in this format is changed by the
    # rules of any chain, and these rules keep its type.
    return smart_replace(t, cls, **kwargs)
