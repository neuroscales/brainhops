"""
Private converters into the OME-Zarr field format.

OME-Zarr stores a pyramid of fields on the axes of its store, and no
exact conversion into it is written yet (#312). A chain is therefore
relabelled as an `OmeZarrField` by the converter
[`relabelling`][brainhops.io.transformations.base._converters.relabelling],
which this module registers. It keeps the parameters of the chain only
when its coordinate systems are not known or are compatible with those
of the format. An `OmeZarrField` that is converted to its own class is
changed by the rules of its family, as any other transformation of that
family is.
"""

import typing_extensions as tx

from brainhops.datamodel import transformations as _xforms
from brainhops.datamodel._transformations.compute.convert import converter
from brainhops.datamodel._transformations.compute.converters import (
    smart_replace,
)
from brainhops.io.transformations.base._converters import relabelling

from ._xforms import OmeZarrField

# OME-Zarr stores a pyramid of fields on the axes of its store, and no
# exact conversion into it is written yet, so a chain is relabelled.
converter(_xforms.Sequence, OmeZarrField)(relabelling)


@converter
def _(t: OmeZarrField, cls: tx.Type[OmeZarrField], **kwargs) -> OmeZarrField:
    # A transformation that is already in this format is changed by the
    # rules of any chain, and these rules keep its type.
    return smart_replace(t, cls, **kwargs)
