"""
Private converters into the M3Z format.

An M3Z morph stores a field of positions on the grid of an atlas, and no
exact conversion into it is written yet (#312). A chain is therefore
relabelled as an `M3zMorph` by the converter
[`relabelling`][brainhops.io.transformations.base._converters.relabelling],
which this module registers. It keeps the parameters of the chain only
when its coordinate systems are not known or are compatible with those
of the format. An `M3zMorph` that is converted to its own class is
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

from ._xform import M3zMorph

# An M3Z morph stores a field of positions on the grid of an atlas, and no
# exact conversion into it is written yet, so a chain is relabelled.
converter(_xforms.Sequence, M3zMorph)(relabelling)


@converter
def _(t: M3zMorph, cls: tx.Type[M3zMorph], **kwargs) -> M3zMorph:
    # A transformation that is already in this format is changed by the
    # rules of any chain, and these rules keep its type.
    return smart_replace(t, cls, **kwargs)
