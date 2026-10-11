"""
Private converter of the OME-Zarr field format to itself.

OME-Zarr stores a pyramid of fields on the axes of its store, and no
exact conversion into it is written yet (#312). Another transformation
is therefore relabelled as an `OmeZarrField` by the converter of
[`brainhops.io.transformations.base`][], which keeps its parameters only
when its coordinate systems are not known or are compatible with those
of the format. An `OmeZarrField` that is converted to its own class is
changed by the rules of its family, as any other transformation of that
family is.
"""

import typing_extensions as tx

from brainhops.datamodel._transformations.compute.convert import converter
from brainhops.datamodel._transformations.compute.converters import (
    smart_replace,
)

from ._xforms import OmeZarrField


@converter
def _(t: OmeZarrField, cls: tx.Type[OmeZarrField], **kwargs) -> OmeZarrField:
    # A transformation that is already in this format is changed by the
    # rules of any chain, and these rules keep its type.
    return smart_replace(t, cls, **kwargs)
