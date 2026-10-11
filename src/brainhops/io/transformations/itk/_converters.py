"""
Private converter of an ITK chain of transforms to its own format.

An ITK file holds a chain of transforms, and no exact conversion into it
is written yet (#312). Another transformation is therefore relabelled as
an `ItkTransform` by the converter of
[`brainhops.io.transformations.base`][], which keeps its parameters only
when its coordinate systems are not known or are compatible with those
of the format. An `ItkTransform` that is converted to its own class is
changed by the rules of any chain, and a chain in another ITK container
is relabelled.
"""

import typing_extensions as tx

from brainhops.datamodel._transformations.compute.convert import converter
from brainhops.datamodel._transformations.compute.converters import (
    smart_replace,
)
from brainhops.io.transformations.base._converters import relabel

from ._xform import ItkTransform


@converter
def _(t: ItkTransform, cls: tx.Type[ItkTransform], **kwargs) -> ItkTransform:
    # A transformation that is already in this format is changed by the
    # rules of any chain, and these rules keep its type. Another container
    # holds the same chain, so it is relabelled, as any other
    # transformation is.
    if not isinstance(t, cls):
        return relabel(t, cls, **kwargs)
    return smart_replace(t, cls, **kwargs)
