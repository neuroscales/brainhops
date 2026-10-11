"""
Private converter of the M3Z format to itself.

An M3Z morph stores a field of positions on the grid of an atlas, and no
exact conversion into it is written yet (#312). Another transformation
is therefore relabelled as an `M3zMorph` by the converter of
[`brainhops.io.transformations.base`][], which keeps its parameters only
when its coordinate systems are not known or are compatible with those
of the format. An `M3zMorph` that is converted to its own class is
changed by the rules of its family, as any other transformation of that
family is.
"""

import typing_extensions as tx

from brainhops.datamodel._transformations.compute.convert import converter
from brainhops.datamodel._transformations.compute.converters import (
    smart_replace,
)

from ._xform import M3zMorph


@converter
def _(t: M3zMorph, cls: tx.Type[M3zMorph], **kwargs) -> M3zMorph:
    # A transformation that is already in this format is changed by the
    # rules of any chain, and these rules keep its type.
    return smart_replace(t, cls, **kwargs)
