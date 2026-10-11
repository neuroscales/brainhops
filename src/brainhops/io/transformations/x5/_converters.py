"""
Private converter of the X5 format to itself.

An X5 file holds a chain of transforms, and no exact conversion into it
is written yet (#312). Another transformation is therefore relabelled as
an `X5Transform` by the converter of
[`brainhops.io.transformations.base`][], which keeps its parameters only
when its coordinate systems are not known or are compatible with those
of the format. An `X5Transform` that is converted to its own class is
changed by the rules of any chain, which keep its type.
"""

import typing_extensions as tx

from brainhops.datamodel._transformations.compute.convert import converter
from brainhops.datamodel._transformations.compute.converters import (
    smart_replace,
)

from ._xform import X5Transform


@converter
def _(t: X5Transform, cls: tx.Type[X5Transform], **kwargs) -> X5Transform:
    # A transformation that is already in this format is changed by the
    # rules of any chain, and these rules keep its type.
    return smart_replace(t, cls, **kwargs)
