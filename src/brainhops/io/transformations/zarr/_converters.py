"""
Private converters into the OME-Zarr field format.

These converters make a conversion into the format, such as
`t.to(OmeZarrField)`, fail with a reason instead of building a wrong
object. The format derives its coordinate systems from the content it
holds, so it has no endpoints to check. Without these converters, the
converters of the data model would rebuild a transformation as an
`OmeZarrField` with the same parameters, and the result would not be the
map that the format says it holds. Each family that would be rebuilt in
this way is refused until an exact conversion is written (#312). An
`OmeZarrField` that is converted to its own class is changed by the
rules of its family, as any other transformation of that family is.
"""

import typing_extensions as tx

from brainhops.datamodel import transformations as _xforms
from brainhops.datamodel._transformations.compute.convert import converter
from brainhops.datamodel._transformations.compute.converters import (
    smart_replace,
)
from brainhops.io.transformations.base._conversions import no_exact_conversion

from ._xforms import OmeZarrField


@converter
def _(
    t: _xforms.Sequence, cls: tx.Type[OmeZarrField], **kwargs
) -> OmeZarrField:
    # OME-Zarr stores a pyramid of fields on the axes of its store, and
    # no exact conversion into it is written yet.
    raise no_exact_conversion(t, cls)


@converter
def _(t: OmeZarrField, cls: tx.Type[OmeZarrField], **kwargs) -> OmeZarrField:
    # A transformation that is already in this format is changed by the
    # rules of any chain, and these rules keep its type.
    return smart_replace(t, cls, **kwargs)
