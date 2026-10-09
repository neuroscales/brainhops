"""
Private converters into the OME-Zarr field format.

The format derives its coordinate systems from what it holds, so it has
no endpoints to check, and a transformation relabelled as it would not
be what the format says it holds. Each family that the converters of the
data model would otherwise rebuild as an `OmeZarrField` is therefore
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

from ._xforms import OmeZarrField


@converter
def _(
    t: _xforms.Sequence, cls: tx.Type[OmeZarrField], **kwargs
) -> OmeZarrField:
    # OME-Zarr stores a pyramid of fields on the axes of its store, and
    # nothing converts to it yet.
    raise no_exact_conversion(t, cls)


@converter
def _(t: OmeZarrField, cls: tx.Type[OmeZarrField], **kwargs) -> OmeZarrField:
    # Within its own format, a chain is changed by the rules of any chain,
    # which keep its type.
    return smart_replace(t, cls, **kwargs)
